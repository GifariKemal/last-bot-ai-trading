"""
Event Scorer - Computes gold_bias score for a given timestamp from the SQLite database.
Used by the event backtest engine to inject fundamental bias into confluence scoring.

Output: EventContext dict with:
  gold_bias        : float -1.0 to +1.0 (bullish positive)
  bias_confidence  : float 0.0 to 1.0 (how strong/certain the bias is)
  liquidity_factor : float 0.0 to 1.0 (1.0 = normal, <1 = holiday/low liquidity)
  confluence_delta : float modifier to add/subtract from SMC confluence score
  sources          : list of what contributed to the score
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timezone
from database import get_events_in_window, get_dxy_trend, get_news_sentiment, get_liquidity_factor


# ─── Default Scorer Config (overridden by Optuna during optimization) ─────────

DEFAULT_CONFIG = {
    # Event source weights (sum doesn't have to = 1, they're independent components)
    'economic_weight':  0.45,   # Weight for NFP/CPI/FOMC events
    'dxy_weight':       0.30,   # Weight for DXY trend
    'news_weight':      0.25,   # Weight for news sentiment

    # Bias → confluence_delta conversion
    'boost_multiplier':   0.15,  # Max boost when gold_bias aligns with signal: +15% confluence
    'penalty_multiplier': 0.10,  # Max penalty when opposing: -10% confluence

    # DXY thresholds
    'dxy_strong_threshold': 0.3,   # >0.3% change = strong DXY move
    'dxy_weak_threshold':   0.1,   # <0.1% change = neutral

    # Confidence thresholds
    'min_confidence_to_boost': 0.40,   # Only apply boost if confidence >= this
    'min_confidence_to_penalize': 0.30, # Only apply penalty if confidence >= this

    # Event impact scaling
    'high_impact_scale':   1.0,
    'medium_impact_scale': 0.6,

    # Lookback windows (hours)
    'event_lookback_hours': 8,
    'event_lookahead_hours': 1,
    'news_lookback_hours': 12,
    'dxy_bars_back': 4,
}


class EventScorer:
    """Computes event-based gold bias for a given market timestamp."""

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def score(self, timestamp: datetime, signal_direction: str = None) -> dict:
        """
        Compute the event context for a given timestamp.

        Args:
            timestamp: The bar timestamp to score (UTC datetime)
            signal_direction: 'BUY' or 'SELL' or None (if None, just return raw bias)

        Returns:
            EventContext dict with gold_bias, bias_confidence, liquidity_factor,
            confluence_delta, and sources list
        """
        cfg = self.config
        sources = []
        weighted_bias = 0.0
        total_weight = 0.0

        # ── 1. Economic Events Component ──────────────────────────────────────
        econ_bias, econ_conf, econ_sources = self._score_economic_events(timestamp)
        if econ_conf > 0:
            impact_scale = cfg['high_impact_scale']
            weighted_bias += econ_bias * econ_conf * impact_scale * cfg['economic_weight']
            total_weight  += econ_conf * impact_scale * cfg['economic_weight']
            sources.extend(econ_sources)

        # ── 2. DXY Trend Component ────────────────────────────────────────────
        dxy_bias, dxy_conf, dxy_source = self._score_dxy(timestamp)
        if dxy_conf > 0:
            weighted_bias += dxy_bias * dxy_conf * cfg['dxy_weight']
            total_weight  += dxy_conf * cfg['dxy_weight']
            if dxy_source:
                sources.append(dxy_source)

        # ── 3. News Sentiment Component ───────────────────────────────────────
        news_bias, news_conf, news_source = self._score_news(timestamp)
        if news_conf > 0:
            weighted_bias += news_bias * news_conf * cfg['news_weight']
            total_weight  += news_conf * cfg['news_weight']
            if news_source:
                sources.append(news_source)

        # ── 4. Aggregate ──────────────────────────────────────────────────────
        if total_weight > 0:
            gold_bias = weighted_bias / total_weight
        else:
            gold_bias = 0.0

        # Clamp to [-1, 1]
        gold_bias = max(-1.0, min(1.0, gold_bias))

        # Bias confidence: how strong is the signal?
        bias_confidence = min(abs(gold_bias) * 1.5, 1.0)  # Scale up: 0.67 bias → 100% conf

        # Liquidity factor
        liquidity_factor = get_liquidity_factor(timestamp)

        # ── 5. Confluence Delta (signal-direction aware) ───────────────────────
        confluence_delta = self._compute_confluence_delta(
            gold_bias, bias_confidence, liquidity_factor, signal_direction
        )

        return {
            'gold_bias':        round(gold_bias, 4),
            'bias_confidence':  round(bias_confidence, 4),
            'liquidity_factor': round(liquidity_factor, 4),
            'confluence_delta': round(confluence_delta, 4),
            'sources':          sources,
            # Sub-component scores for debugging/reporting
            'econ_bias':  round(econ_bias, 4),
            'dxy_bias':   round(dxy_bias, 4),
            'news_bias':  round(news_bias, 4),
        }

    def _score_economic_events(self, timestamp: datetime) -> tuple:
        """Score from economic events (NFP, CPI, FOMC) in the lookback window."""
        cfg = self.config
        events = get_events_in_window(
            timestamp,
            lookback_hours=cfg['event_lookback_hours'],
            lookahead_hours=cfg['event_lookahead_hours']
        )

        if not events:
            return 0.0, 0.0, []

        weighted_bias = 0.0
        total_weight  = 0.0
        sources = []

        for ev in events:
            bias = ev.get('gold_bias', 0.0) or 0.0
            if bias == 0.0:
                continue

            # Time decay: events lose impact as they age
            try:
                ev_dt = datetime.fromisoformat(ev['event_date'])
                if ev_dt.tzinfo is None:
                    ev_dt = ev_dt.replace(tzinfo=timezone.utc)
                hours_ago = max(0, (timestamp - ev_dt).total_seconds() / 3600)
                # Events fresher than 2 hours: full weight; decays over 8 hours
                time_decay = max(0.2, 1.0 - (hours_ago / 10))
            except Exception:
                time_decay = 0.5

            impact = ev.get('impact', 'HIGH')
            impact_scale = cfg['high_impact_scale'] if impact == 'HIGH' else cfg['medium_impact_scale']

            w = impact_scale * time_decay
            weighted_bias += bias * w
            total_weight  += abs(bias) * w  # Confidence proportional to bias magnitude

            sources.append(
                f"{ev['event_name']} {ev.get('beat_miss', '')} "
                f"(bias={bias:+.2f}, decay={time_decay:.2f})"
            )

        if total_weight == 0:
            return 0.0, 0.0, sources

        avg_bias = weighted_bias / max(total_weight, 0.01)
        confidence = min(total_weight / len(events), 1.0)  # Normalize
        return avg_bias, confidence, sources

    def _score_dxy(self, timestamp: datetime) -> tuple:
        """Score from DXY momentum. Stronger USD = bearish gold."""
        cfg = self.config
        dxy_change = get_dxy_trend(timestamp, bars_back=cfg['dxy_bars_back'])

        if dxy_change == 0.0:
            return 0.0, 0.0, None

        # DXY rising (positive) → bearish gold (negative bias)
        # DXY falling (negative) → bullish gold (positive bias)
        abs_change = abs(dxy_change)

        if abs_change >= cfg['dxy_strong_threshold']:
            confidence = 1.0
        elif abs_change >= cfg['dxy_weak_threshold']:
            # Linear scale between weak and strong threshold
            confidence = (abs_change - cfg['dxy_weak_threshold']) / (
                cfg['dxy_strong_threshold'] - cfg['dxy_weak_threshold']
            )
        else:
            return 0.0, 0.0, None

        # Invert: DXY up = bearish gold
        bias = -1.0 if dxy_change > 0 else 1.0
        # Scale bias by change magnitude (max 0.9 bias)
        scaled_bias = bias * min(abs_change / 0.5, 0.9)

        source = f"DXY {dxy_change:+.3f}% ({cfg['dxy_bars_back']}×H1)"
        return scaled_bias, confidence, source

    def _score_news(self, timestamp: datetime) -> tuple:
        """Score from news sentiment in lookback window."""
        cfg = self.config
        sentiment = get_news_sentiment(timestamp, lookback_hours=cfg['news_lookback_hours'])

        if sentiment == 0.0:
            return 0.0, 0.0, None

        # Sentiment already in [-1, 1] range from gold_bias stored in DB
        # Confidence based on absolute sentiment strength
        confidence = min(abs(sentiment) * 2.0, 1.0)  # 0.5 → full confidence

        source = f"News sentiment={sentiment:+.3f} ({cfg['news_lookback_hours']}h)"
        return sentiment, confidence, source

    def _compute_confluence_delta(
        self,
        gold_bias: float,
        bias_confidence: float,
        liquidity_factor: float,
        signal_direction: str
    ) -> float:
        """
        Convert gold_bias into a confluence score delta.

        Rules:
        - If signal aligns with gold_bias AND confidence >= threshold → boost confluence
        - If signal opposes gold_bias AND confidence >= threshold → penalize confluence
        - Low liquidity days reduce the delta (less reliable fundamental impact)
        - If signal_direction is None, return raw signed delta (positive = bullish boost)
        """
        cfg = self.config

        if abs(gold_bias) < 0.05:
            return 0.0  # Too neutral to affect anything

        # Raw delta: proportional to bias strength
        raw_delta = abs(gold_bias) * liquidity_factor

        if signal_direction is None:
            # Return signed: positive = bullish context, negative = bearish context
            return gold_bias * liquidity_factor * cfg['boost_multiplier']

        # Determine alignment
        signal_is_bullish = signal_direction.upper() == 'BUY'
        bias_is_bullish   = gold_bias > 0

        aligned = (signal_is_bullish == bias_is_bullish)

        if aligned:
            if bias_confidence >= cfg['min_confidence_to_boost']:
                return raw_delta * cfg['boost_multiplier']
            else:
                return 0.0  # Not confident enough to boost
        else:
            if bias_confidence >= cfg['min_confidence_to_penalize']:
                return -raw_delta * cfg['penalty_multiplier']
            else:
                return 0.0  # Not confident enough to penalize


# ─── Convenience function ─────────────────────────────────────────────────────

_default_scorer = None


def get_event_context(timestamp: datetime, signal_direction: str = None, config: dict = None) -> dict:
    """
    Get the event context for a timestamp. Caches the scorer instance.

    Args:
        timestamp: UTC datetime of the bar
        signal_direction: 'BUY', 'SELL', or None
        config: Optional scorer config dict (for Optuna optimization)

    Returns:
        EventContext dict
    """
    global _default_scorer
    if config is not None:
        scorer = EventScorer(config)
    else:
        if _default_scorer is None:
            _default_scorer = EventScorer()
        scorer = _default_scorer

    return scorer.score(timestamp, signal_direction)


# ─── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Test scorer at various key dates
    test_dates = [
        (datetime(2026, 1, 10, 13, 30, tzinfo=timezone.utc), 'SELL', "NFP beat day (Jan 10)"),
        (datetime(2026, 2,  7, 13, 30, tzinfo=timezone.utc), 'BUY',  "NFP miss day (Feb 7)"),
        (datetime(2026, 2, 12, 13, 30, tzinfo=timezone.utc), 'BUY',  "CPI hot day (Feb 12)"),
        (datetime(2026, 2, 17, 10,  0, tzinfo=timezone.utc), 'SELL', "Chinese New Year (today)"),
        (datetime(2025, 12, 25, 12,  0, tzinfo=timezone.utc), 'BUY',  "Christmas Day"),
    ]

    scorer = EventScorer()
    print("\n" + "=" * 70)
    print("EVENT SCORER TEST")
    print("=" * 70)

    for ts, direction, label in test_dates:
        ctx = scorer.score(ts, direction)
        print(f"\n{label} | {ts.strftime('%Y-%m-%d %H:%M UTC')} | Signal: {direction}")
        print(f"  gold_bias={ctx['gold_bias']:+.3f}  "
              f"confidence={ctx['bias_confidence']:.2f}  "
              f"liquidity={ctx['liquidity_factor']:.2f}  "
              f"confluence_delta={ctx['confluence_delta']:+.4f}")
        if ctx['sources']:
            for s in ctx['sources']:
                print(f"  → {s}")
        else:
            print("  → No events in window (neutral)")
