"""
Market Regime Detector
Classifies current market state into one of 8 regimes using multi-factor scoring.
Used by adaptive scorer, structure SL/TP, and optimizer V3.
"""

from typing import Dict, Optional
import polars as pl

from ..core.constants import MarketRegime, TrendDirection, VolatilityLevel
from ..bot_logger import get_logger


class RegimeDetector:
    """Detect market regime from price data with pre-calculated indicators."""

    def __init__(self, config: Optional[Dict] = None):
        self.logger = get_logger()
        self.config = config or {}
        self._lookback = self.config.get("lookback_bars", 100)

    def detect(self, df: pl.DataFrame) -> Dict:
        """
        Detect current market regime.

        Args:
            df: DataFrame with pre-calculated indicators (atr_14, ema_20, ema_50,
                bb_bandwidth, rsi_14, market_structure_trend, etc.)

        Returns:
            Dict with regime, confidence, and component scores.
        """
        if df is None or len(df) < self._lookback:
            return self._default_result()

        recent = df.tail(self._lookback)
        latest = df.tail(1).to_dicts()[0]

        # Component scores
        atr_pct = self._atr_percentile(recent)
        ema_spread = self._ema_spread(latest)
        bb_width_pct = self._bb_width_percentile(recent)
        trend_score = self._trend_score(recent, latest)
        has_choch = self._recent_choch(recent)
        vol_expansion = self._volatility_expansion(recent)

        components = {
            "atr_percentile": atr_pct,
            "ema_spread_pct": ema_spread,
            "bb_width_percentile": bb_width_pct,
            "trend_score": trend_score,       # -1 to +1 (bear to bull)
            "has_recent_choch": has_choch,
            "volatility_expanding": vol_expansion,
        }

        regime = self._classify(components)
        confidence = self._confidence(components, regime)

        return {
            "regime": regime,
            "confidence": confidence,
            "components": components,
        }

    def _atr_percentile(self, df: pl.DataFrame) -> float:
        """ATR percentile within lookback window."""
        if "atr_14" not in df.columns:
            return 50.0
        atr_vals = df["atr_14"].drop_nulls().to_list()
        if len(atr_vals) < 10:
            return 50.0
        current = atr_vals[-1]
        below = sum(1 for v in atr_vals if v <= current)
        return (below / len(atr_vals)) * 100

    def _ema_spread(self, latest: Dict) -> float:
        """EMA20-EMA50 spread as % of price."""
        ema20 = latest.get("ema_20")
        ema50 = latest.get("ema_50")
        close = latest.get("close")
        if not all((ema20, ema50, close)) or close == 0:
            return 0.0
        return ((ema20 - ema50) / close) * 100

    def _bb_width_percentile(self, df: pl.DataFrame) -> float:
        """Bollinger Bandwidth percentile within lookback."""
        if "bb_bandwidth" not in df.columns:
            return 50.0
        bw_vals = df["bb_bandwidth"].drop_nulls().to_list()
        if len(bw_vals) < 10:
            return 50.0
        current = bw_vals[-1]
        below = sum(1 for v in bw_vals if v <= current)
        return (below / len(bw_vals)) * 100

    def _trend_score(self, df: pl.DataFrame, latest: Dict) -> float:
        """
        Trend score from -1 (strong bearish) to +1 (strong bullish).
        Uses EMA alignment + price position relative to EMAs.
        """
        ema20 = latest.get("ema_20")
        ema50 = latest.get("ema_50")
        ema100 = latest.get("ema_100")
        close = latest.get("close")

        if not all((ema20, ema50, close)):
            return 0.0

        score = 0.0
        # EMA ordering
        if ema20 > ema50:
            score += 0.4
        else:
            score -= 0.4

        if ema100 and ema50 > ema100:
            score += 0.2
        elif ema100:
            score -= 0.2

        # Price relative to EMA20
        if close > ema20:
            score += 0.2
        else:
            score -= 0.2

        # EMA slope (last 10 bars)
        if "ema_20" in df.columns and len(df) >= 10:
            ema_vals = df["ema_20"].tail(10).drop_nulls().to_list()
            if len(ema_vals) >= 2:
                slope = (ema_vals[-1] - ema_vals[0]) / max(ema_vals[0], 1)
                slope_contribution = max(-0.2, min(0.2, slope * 100))
                score += slope_contribution

        return max(-1.0, min(1.0, score))

    def _recent_choch(self, df: pl.DataFrame) -> bool:
        """Check for CHoCH in last 10 bars."""
        if "choch_bullish" in df.columns:
            recent = df.tail(10)
            bull_choch = recent["choch_bullish"].drop_nulls().to_list()
            bear_choch = recent.get_column("choch_bearish").drop_nulls().to_list() if "choch_bearish" in recent.columns else []
            return any(bull_choch) or any(bear_choch)
        return False

    def _volatility_expansion(self, df: pl.DataFrame) -> bool:
        """Check if volatility is expanding (recent ATR > 20% above older)."""
        if "atr_14" not in df.columns or len(df) < 20:
            return False
        recent = df["atr_14"].tail(10).drop_nulls().to_list()
        older = df["atr_14"].tail(30).head(20).drop_nulls().to_list()
        if not recent or not older:
            return False
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        return recent_avg > older_avg * 1.2 if older_avg > 0 else False

    def _classify(self, c: Dict) -> MarketRegime:
        """Classify regime from component scores."""
        atr_pct = c["atr_percentile"]
        ema_spread = c["ema_spread_pct"]
        trend = c["trend_score"]
        has_choch = c["has_recent_choch"]
        vol_exp = c["volatility_expanding"]
        bb_pct = c["bb_width_percentile"]

        # REVERSAL: CHoCH present + trend changing direction
        if has_choch and abs(trend) < 0.4:
            return MarketRegime.REVERSAL

        # VOLATILE_BREAKOUT: high ATR + expanding vol + high BB width
        if atr_pct > 80 and vol_exp and bb_pct > 75:
            return MarketRegime.VOLATILE_BREAKOUT

        # STRONG_TREND: high ATR percentile + strong EMA spread + strong trend score
        if atr_pct > 50 and abs(ema_spread) > 0.15 and abs(trend) > 0.5:
            if trend > 0:
                return MarketRegime.STRONG_TREND_UP
            else:
                return MarketRegime.STRONG_TREND_DOWN

        # WEAK_TREND: moderate trend indicators
        if abs(trend) > 0.2 and abs(ema_spread) > 0.05:
            if trend > 0:
                return MarketRegime.WEAK_TREND_UP
            else:
                return MarketRegime.WEAK_TREND_DOWN

        # RANGE: low trend score + low ATR
        if abs(trend) <= 0.2:
            if atr_pct < 30 and bb_pct < 30:
                return MarketRegime.RANGE_TIGHT
            else:
                return MarketRegime.RANGE_WIDE

        # Default fallback
        if trend > 0:
            return MarketRegime.WEAK_TREND_UP
        elif trend < 0:
            return MarketRegime.WEAK_TREND_DOWN
        return MarketRegime.RANGE_WIDE

    def _confidence(self, c: Dict, regime: MarketRegime) -> float:
        """Calculate confidence in regime classification (0-1)."""
        # Stronger signals = higher confidence
        trend_strength = abs(c["trend_score"])
        ema_strength = min(1.0, abs(c["ema_spread_pct"]) / 0.3)

        if regime.is_trending:
            return min(1.0, (trend_strength + ema_strength) / 2)
        elif regime in (MarketRegime.RANGE_TIGHT, MarketRegime.RANGE_WIDE):
            # Low trend = high confidence in range
            return min(1.0, 1.0 - trend_strength)
        elif regime == MarketRegime.VOLATILE_BREAKOUT:
            return min(1.0, c["atr_percentile"] / 100)
        elif regime == MarketRegime.REVERSAL:
            return 0.6 if c["has_recent_choch"] else 0.4

        return 0.5

    def _default_result(self) -> Dict:
        return {
            "regime": MarketRegime.RANGE_WIDE,
            "confidence": 0.3,
            "components": {},
        }
