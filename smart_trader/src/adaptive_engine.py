"""
Adaptive Engine — regime-driven parameters, performance-based tuning, algo pre-score.
Conservative adaptation: ±25% max shift, 20-50 trade window, hard bounds on all params.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger


# ── Baseline Regime Parameters ────────────────────────────────────────────────

_DEFAULT_BASELINES = {
    "trending": {
        "sl_atr_mult": 2.0, "tp_atr_mult": 4.0, "min_confidence": 0.68,
        "be_trigger_mult": 0.7, "lock_trigger_mult": 1.5, "trail_keep_pct": 0.50,
        "stale_tighten_min": 90, "scratch_exit_min": 180,
    },
    "ranging": {
        "sl_atr_mult": 1.5, "tp_atr_mult": 3.0, "min_confidence": 0.75,
        "be_trigger_mult": 0.5, "lock_trigger_mult": 1.2, "trail_keep_pct": 0.55,
        "stale_tighten_min": 60, "scratch_exit_min": 120,
    },
    "breakout": {
        "sl_atr_mult": 2.5, "tp_atr_mult": 5.0, "min_confidence": 0.72,
        "be_trigger_mult": 0.8, "lock_trigger_mult": 1.5, "trail_keep_pct": 0.45,
        "stale_tighten_min": 120, "scratch_exit_min": 240,
    },
    "reversal": {
        "sl_atr_mult": 1.5, "tp_atr_mult": 3.5, "min_confidence": 0.78,
        "be_trigger_mult": 0.5, "lock_trigger_mult": 1.0, "trail_keep_pct": 0.55,
        "stale_tighten_min": 60, "scratch_exit_min": 120,
    },
}

# ── Hard Bounds (cannot be exceeded regardless of adaptation) ─────────────────

_DEFAULT_BOUNDS = {
    "sl_atr_mult":       (1.0, 3.5),
    "tp_atr_mult":       (2.0, 7.0),
    "min_confidence":    (0.60, 0.90),
    "be_trigger_mult":   (0.3, 1.0),
    "lock_trigger_mult": (0.8, 2.5),
    "trail_keep_pct":    (0.30, 0.70),
    "stale_tighten_min": (30, 180),
    "scratch_exit_min":  (60, 360),
}


class AdaptiveEngine:
    """Regime param lookup, performance tuning, algo pre-score, persistence."""

    def __init__(
        self,
        config_baselines: dict = None,
        bounds: dict = None,
        state_path: str = "data/adaptive_state.json",
        min_trades: int = 20,
        max_shift_pct: float = 0.25,
    ):
        self.baselines = config_baselines or _DEFAULT_BASELINES
        self.bounds = bounds or _DEFAULT_BOUNDS
        self.state_path = Path(state_path)
        self.min_trades = min_trades
        self.max_shift = max_shift_pct

        # Per-regime adjustment multipliers (1.0 = baseline)
        self.adjustments: dict[str, dict[str, float]] = {}
        # Per-session/zone weights
        self.session_weights: dict[str, float] = {}
        self.zone_weights: dict[str, float] = {}
        self.last_update: str = ""

        self._load()

    # ── Parameter Access ──────────────────────────────────────────────────────

    def get_params(self, regime_cat: str) -> dict:
        """Get effective params = baseline × adjustment, clamped to bounds."""
        base = self.baselines.get(regime_cat, self.baselines.get("ranging", {}))
        adj = self.adjustments.get(regime_cat, {})

        result = {}
        for key, base_val in base.items():
            mult = adj.get(key, 1.0)
            effective = base_val * mult
            # Clamp to bounds
            lo, hi = self.bounds.get(key, (0, 999))
            result[key] = max(lo, min(hi, effective))
        return result

    def get_entry_params(self, regime_cat: str) -> dict:
        """Get entry-relevant params: sl_atr_mult, tp_atr_mult, min_confidence."""
        p = self.get_params(regime_cat)
        return {
            "sl_atr_mult": p.get("sl_atr_mult", 2.0),
            "tp_atr_mult": p.get("tp_atr_mult", 4.0),
            "min_confidence": p.get("min_confidence", 0.70),
        }

    def get_exit_params(self, regime_cat: str) -> dict:
        """Get exit-relevant params for executor."""
        p = self.get_params(regime_cat)
        return {
            "be_trigger_mult": p.get("be_trigger_mult", 0.7),
            "lock_trigger_mult": p.get("lock_trigger_mult", 1.5),
            "trail_keep_pct": p.get("trail_keep_pct", 0.50),
            "stale_tighten_min": p.get("stale_tighten_min", 90),
            "scratch_exit_min": p.get("scratch_exit_min", 180),
        }

    def get_session_weight(self, session: str) -> float:
        return self.session_weights.get(session, 1.0)

    def get_zone_weight(self, zone_type: str) -> float:
        return self.zone_weights.get(zone_type, 1.0)

    # ── Algo Pre-Score ────────────────────────────────────────────────────────

    def algo_pre_score(
        self,
        signal_count: int,
        regime_cat: str,
        session: str,
        direction: str,
        ema_trend: str,
        rsi: float,
        pd_zone: str,
        has_choch: bool,
    ) -> tuple[float, bool]:
        """
        Compute algorithmic pre-score before calling Claude.
        Returns (score, should_call_claude).
        Threshold: score < 0.35 → skip Claude call.
        """
        score = 0.0

        # Signal count: 0-0.30 (3 signals = 0.15, 5+ = 0.30)
        score += min(signal_count * 0.05, 0.30)

        # Session weight: 0-0.15
        sw = self.get_session_weight(session)
        score += min(sw * 0.12, 0.15)

        # EMA alignment: 0-0.15
        if direction == "LONG" and ema_trend == "BULLISH":
            score += 0.15
        elif direction == "SHORT" and ema_trend == "BEARISH":
            score += 0.15
        elif has_choch:
            # Counter-trend with CHoCH gets partial credit
            score += 0.08
        # else: counter-trend without CHoCH = 0

        # RSI sweet spot: 0-0.10
        if direction == "LONG" and 30 <= rsi <= 60:
            score += 0.10
        elif direction == "SHORT" and 40 <= rsi <= 70:
            score += 0.10
        elif 25 <= rsi <= 75:
            score += 0.05

        # P/D alignment: 0-0.10
        if direction == "LONG" and pd_zone == "DISCOUNT":
            score += 0.10
        elif direction == "SHORT" and pd_zone == "PREMIUM":
            score += 0.10
        elif pd_zone == "EQUILIBRIUM":
            score += 0.05

        should_call = score >= 0.35
        return round(score, 3), should_call

    # ── Counter-Trend Override ────────────────────────────────────────────────

    def should_allow_counter_trend(self, has_choch: bool, regime_cat: str) -> bool:
        """Allow counter-trend entry if CHoCH detected AND regime is reversal/ranging."""
        return has_choch and regime_cat in ("reversal", "ranging")

    # ── Performance-Based Adaptation ──────────────────────────────────────────

    def update_from_performance(self, tracker) -> None:
        """
        Adapt parameters based on trade history. Called after each trade close.
        Requires ≥min_trades closed trades before any adaptation.
        """
        metrics = tracker.get_rolling_metrics(window=50)
        if metrics["total_trades"] < self.min_trades:
            return

        # Per-regime adaptation
        regime_metrics = tracker.get_metrics_by_key("regime", window=50)
        for regime_label, rm in regime_metrics.items():
            if rm["count"] < 5:
                continue
            # Map regime label to category
            cat = self._label_to_category(regime_label)
            if not cat:
                continue
            adj = self.adjustments.setdefault(cat, {})
            self._adapt_regime(adj, rm)

        # Per-session adaptation
        session_metrics = tracker.get_metrics_by_key("session", window=50)
        for sess, sm in session_metrics.items():
            if sm["count"] < 3:
                continue
            pf = sm["profit_factor"]
            if pf > 1.5:
                self.session_weights[sess] = min(1.25, self.session_weights.get(sess, 1.0) + 0.05)
            elif pf < 0.7:
                self.session_weights[sess] = max(0.75, self.session_weights.get(sess, 1.0) - 0.05)

        # Per-zone-type adaptation
        zone_metrics = tracker.get_metrics_by_key("zone_type", window=50)
        for zt, zm in zone_metrics.items():
            if zm["count"] < 3:
                continue
            pf = zm["profit_factor"]
            if pf > 1.5:
                self.zone_weights[zt] = min(1.25, self.zone_weights.get(zt, 1.0) + 0.05)
            elif pf < 0.7:
                self.zone_weights[zt] = max(0.75, self.zone_weights.get(zt, 1.0) - 0.05)

        self.last_update = datetime.now(timezone.utc).isoformat()
        self._save()

        logger.info(
            f"Adaptive update | trades={metrics['total_trades']} | "
            f"WR={metrics['win_rate']:.0%} | PF={metrics['profit_factor']:.2f} | "
            f"adjustments={len(self.adjustments)} regimes | "
            f"session_weights={self.session_weights} | zone_weights={self.zone_weights}"
        )

    def _adapt_regime(self, adj: dict, rm: dict) -> None:
        """Apply conservative adaptation to regime adjustments."""
        wr = rm["win_rate"]
        pf = rm["profit_factor"]
        avg_pnl = rm["avg_pnl_pts"]

        # Win rate → min_confidence
        if wr > 0.60:
            adj["min_confidence"] = max(1.0 - self.max_shift, adj.get("min_confidence", 1.0) - 0.03)
        elif wr < 0.40:
            adj["min_confidence"] = min(1.0 + self.max_shift, adj.get("min_confidence", 1.0) + 0.03)

        # Profit factor → SL
        if pf > 2.0:
            adj["sl_atr_mult"] = max(1.0 - self.max_shift, adj.get("sl_atr_mult", 1.0) * 0.97)
        elif pf < 0.8:
            adj["sl_atr_mult"] = min(1.0 + self.max_shift, adj.get("sl_atr_mult", 1.0) * 1.03)

        # Avg PnL → TP
        if avg_pnl > 5:
            adj["tp_atr_mult"] = min(1.0 + self.max_shift, adj.get("tp_atr_mult", 1.0) * 1.03)
        elif avg_pnl < -5:
            adj["tp_atr_mult"] = max(1.0 - self.max_shift, adj.get("tp_atr_mult", 1.0) * 0.97)

    def _label_to_category(self, regime_label: str) -> str:
        """Map regime short_label or name to category."""
        if not regime_label:
            return ""
        label = regime_label.upper()
        if "STR_" in label or "STRONG" in label:
            return "trending"
        if "WK_" in label or "WEAK" in label:
            return "trending"
        if "RNG" in label or "RANGE" in label:
            return "ranging"
        if "VOL" in label or "BREAKOUT" in label:
            return "breakout"
        if "REV" in label or "REVERSAL" in label:
            return "reversal"
        return ""

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.adjustments = data.get("adjustments", {})
                self.session_weights = data.get("session_weights", {})
                self.zone_weights = data.get("zone_weights", {})
                self.last_update = data.get("last_update", "")
                logger.info(
                    f"AdaptiveEngine: loaded state | "
                    f"{len(self.adjustments)} regime adjustments | "
                    f"last_update={self.last_update}"
                )
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"AdaptiveEngine: failed to load {self.state_path}: {e}")

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump({
                    "adjustments": self.adjustments,
                    "session_weights": self.session_weights,
                    "zone_weights": self.zone_weights,
                    "last_update": self.last_update,
                }, f, indent=2)
        except IOError as e:
            logger.warning(f"AdaptiveEngine: failed to save: {e}")
