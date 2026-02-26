"""
Adaptive Confluence Scorer
Regime-conditional scoring that replaces fixed-weight ConfluenceScorer.
Applies signal quality multipliers and recency weighting.
"""

import json
from typing import Dict, Optional
from pathlib import Path

import polars as pl

from ..core.constants import TrendDirection, MarketRegime
from ..bot_logger import get_logger


# Default regime weight profiles
DEFAULT_REGIME_WEIGHTS = {
    "trending": {
        "smc_weight": 0.45,
        "tech_weight": 0.20,
        "market_weight": 0.25,
        "mtf_weight": 0.10,
        "min_confluence": 0.50,
        "atr_sl_mult": 3.0,
    },
    "ranging": {
        "smc_weight": 0.50,
        "tech_weight": 0.25,
        "market_weight": 0.15,
        "mtf_weight": 0.10,
        "min_confluence": 0.60,
        "atr_sl_mult": 2.5,
    },
    "breakout": {
        "smc_weight": 0.35,
        "tech_weight": 0.25,
        "market_weight": 0.30,
        "mtf_weight": 0.10,
        "min_confluence": 0.55,
        "atr_sl_mult": 3.5,
    },
    "reversal": {
        "smc_weight": 0.50,
        "tech_weight": 0.20,
        "market_weight": 0.15,
        "mtf_weight": 0.15,
        "min_confluence": 0.65,
        "atr_sl_mult": 2.5,
    },
    "volatile": {
        "smc_weight": 0.35,
        "tech_weight": 0.25,
        "market_weight": 0.30,
        "mtf_weight": 0.10,
        "min_confluence": 0.60,
        "atr_sl_mult": 4.0,
    },
}

# Raw SMC floor: if SMC fill < 30% of max, cap final score below any regime threshold
MIN_SMC_FILL = 0.30
MAX_SCORE_ON_WEAK_SMC = 0.45

# Opposing CHoCH penalty: when the other direction has a CHoCH (reversal signal)
OPPOSING_CHOCH_PENALTY = 0.15

# ── Phase 2: BOS Quality Filter (V5) ──────────────────────────────────────────
# BOS without any confirmation (no CHoCH/FVG/OB/LiqSweep) → higher false-breakout risk.
# These are read from confluence_weights config so Optuna can tune them.
#
# bos_solo_penalty      : extra score deduction for naked BOS  (tunable 0.0–0.20)
# bos_ranging_fill_floor: stricter fill floor for BOS in ranging/volatile regimes
#                         (tunable 0.30–0.55; floor of MIN_SMC_FILL applies)
# Regimes where BOS is noisiest (structure breaks = market noise, not intent):
BOS_SOLO_REGIMES = frozenset([
    MarketRegime.RANGE_TIGHT,
    MarketRegime.RANGE_WIDE,
    MarketRegime.VOLATILE_BREAKOUT,
])

# Default signal quality multipliers (overridden by decomposition results)
DEFAULT_SIGNAL_QUALITY = {
    "fvg": 1.0,
    "order_block": 1.0,
    "liquidity_sweep": 1.0,
    "structure_break": 1.0,
    "choch": 1.2,
    "bos": 0.8,
}

# Map decomposition signal names to scorer key names
DECOMP_KEY_MAP = {
    "ob": "order_block",
    "liq_sweep": "liquidity_sweep",
}


class AdaptiveConfluenceScorer:
    """Regime-adaptive confluence scorer."""

    def __init__(self, config: Dict, regime_weights: Optional[Dict] = None):
        self.logger = get_logger()
        self.config = config
        self.base_weights = config.get("confluence_weights", {})

        # Load regime-specific weight profiles
        self.regime_weights = regime_weights or config.get(
            "regime_weights", DEFAULT_REGIME_WEIGHTS
        )

        # Load signal quality tiers (from decomposition or defaults)
        self.signal_quality = self._load_signal_quality()

        # Compute max possible raw scores for proper normalization
        self._smc_base_max = (
            self.base_weights.get("fvg", 0.20) +
            self.base_weights.get("order_block", 0.25) +
            self.base_weights.get("liquidity_sweep", 0.20) +
            self.base_weights.get("structure_break", 0.30)
        )  # = 0.95 by default
        self._tech_base_max = (
            self.base_weights.get("ema_alignment", 0.10) +
            self.base_weights.get("rsi_confirmation", 0.08) +
            self.base_weights.get("macd_confirmation", 0.07)
        )  # = 0.25 by default

        # LTF analyzer (lazy init)
        self._ltf_analyzer = None

        # Phase 2: BOS quality filter params (read from confluence_weights config)
        self._bos_solo_penalty       = self.base_weights.get("bos_solo_penalty", 0.0)
        self._bos_ranging_fill_floor = self.base_weights.get("bos_ranging_fill_floor", MIN_SMC_FILL)

    def _load_signal_quality(self) -> Dict:
        """Load signal quality multipliers from decomposition results or defaults."""
        tier_path = Path("data/signal_analysis/decomposition_results.json")
        if tier_path.exists():
            try:
                with open(tier_path) as f:
                    data = json.load(f)
                tiers = data.get("tiers", {})
                quality = {}
                for tier_signals in tiers.values():
                    for entry in tier_signals:
                        if not entry.get("is_combo"):
                            quality[entry["signal"]] = entry["multiplier"]
                if quality:
                    # Normalize keys: decomposition uses "ob"/"liq_sweep",
                    # scorer expects "order_block"/"liquidity_sweep"
                    normalized = {}
                    for k, v in quality.items():
                        normalized[DECOMP_KEY_MAP.get(k, k)] = v
                    self.logger.info(f"Loaded signal quality from decomposition: {normalized}")
                    return normalized
            except Exception as e:
                self.logger.warning(f"Could not load signal tiers: {e}")

        return DEFAULT_SIGNAL_QUALITY.copy()

    def get_regime_params(self, regime: MarketRegime) -> Dict:
        """Get weight parameters for a specific regime."""
        category = regime.category
        return self.regime_weights.get(category, DEFAULT_REGIME_WEIGHTS["trending"])

    def calculate_score(
        self,
        direction: TrendDirection,
        smc_signals: Dict,
        technical_indicators: Dict,
        market_analysis: Dict,
        mtf_analysis: Dict,
        regime: MarketRegime = MarketRegime.RANGE_WIDE,
        ltf_data: Dict = None,
        opposing_smc: Dict = None,
    ) -> Dict:
        """
        Calculate regime-adaptive confluence score.

        Args:
            direction: Trade direction
            smc_signals: SMC signal data
            technical_indicators: Technical indicator values
            market_analysis: Market condition analysis
            mtf_analysis: Multi-timeframe analysis
            regime: Current market regime
            ltf_data: M5 data for LTF confirmation

        Returns:
            Score dictionary with breakdown
        """
        try:
            params = self.get_regime_params(regime)
            breakdown = {}

            # 1. SMC factors (normalize to 0-1 fill ratio, then apply regime weight)
            smc_raw = self._score_smc_factors(direction, smc_signals)
            smc_fill = smc_raw["total"] / self._smc_base_max if self._smc_base_max > 0 else 0
            smc_score = smc_fill * params["smc_weight"]
            breakdown["smc"] = smc_raw

            # 2. Technical factors (normalize to 0-1 fill ratio, then apply regime weight)
            tech_raw = self._score_technical_factors(direction, technical_indicators)
            tech_fill = tech_raw["total"] / self._tech_base_max if self._tech_base_max > 0 else 0
            tech_score = tech_fill * params["tech_weight"]
            breakdown["technical"] = tech_raw

            # 3. Bonus factors (MTF, LTF - no more ICT P/D zone)
            bonus_raw = self._score_bonus_factors(
                mtf_analysis, direction=direction, ltf_data=ltf_data,
                mtf_weight_scale=params["mtf_weight"] / 0.10,
            )
            bonus_score = bonus_raw["total"]
            breakdown["bonus"] = bonus_raw

            # 4. Market condition adjustments (including counter-trend penalty, opposing CHoCH)
            adjustments = self._apply_adjustments(market_analysis, regime, direction, opposing_smc)
            breakdown["adjustments"] = adjustments

            raw_score = smc_score + tech_score + bonus_score + adjustments["total"]

            # ── Phase 2: BOS Quality Filter ─────────────────────────────────
            # Detect "BOS solo": BOS present but NO other SMC confirmation.
            # CHoCH/FVG/OB/LiqSweep all add conviction; naked BOS = noise-prone.
            is_bos_solo = (
                smc_signals.get("structure", {}).get("bos", False)
                and not smc_signals.get("structure", {}).get("choch", False)
                and not smc_signals.get("fvg", {}).get("in_zone", False)
                and not smc_signals.get("order_block", {}).get("at_zone", False)
                and not smc_signals.get("liquidity", {}).get("swept", False)
            )

            # 1. BOS solo penalty: subtract from raw score
            bos_penalized = False
            if is_bos_solo and self._bos_solo_penalty > 0:
                raw_score -= self._bos_solo_penalty
                bos_penalized = True

            final_score = max(0.0, min(1.0, raw_score))

            # 2. Regime-adaptive fill floor: BOS in ranging/volatile needs more confirmation
            effective_min_fill = MIN_SMC_FILL
            if is_bos_solo and regime in BOS_SOLO_REGIMES:
                effective_min_fill = max(MIN_SMC_FILL, self._bos_ranging_fill_floor)

            # Fix 1 (enhanced): Raw SMC floor — weak SMC can't be rescued by tech/MTF alone
            smc_capped = False
            if smc_fill < effective_min_fill:
                if final_score > MAX_SCORE_ON_WEAK_SMC:
                    smc_capped = True
                    final_score = MAX_SCORE_ON_WEAK_SMC
            # ─────────────────────────────────────────────────────────────────

            min_conf = params["min_confluence"]

            self.logger.debug(
                f"Score [{direction.value}]: smc_fill={smc_fill:.0%}*{params['smc_weight']:.2f}={smc_score:.3f} | "
                f"tech={tech_score:.3f} | bonus={bonus_score:.3f} | adj={adjustments['total']:.3f} | "
                f"final={final_score:.3f} (min={min_conf:.3f})"
                f"{' [SMC_CAPPED]' if smc_capped else ''}"
                f"{f' [BOS_SOLO pen={self._bos_solo_penalty:.2f} floor={effective_min_fill:.0%}]' if is_bos_solo else ''}"
            )

            return {
                "score": final_score,
                "raw_score": raw_score,
                "breakdown": breakdown,
                "passing": final_score >= min_conf,
                "regime": regime.value,
                "min_confluence": min_conf,
                "smc_capped": smc_capped,
            }

        except Exception as e:
            self.logger.error(f"Error in adaptive confluence scoring: {e}")
            return {
                "score": 0.0,
                "raw_score": 0.0,
                "breakdown": {},
                "passing": False,
                "regime": regime.value,
                "min_confluence": 0.60,
            }

    def _score_smc_factors(self, direction: TrendDirection, smc_signals: Dict) -> Dict:
        """Score SMC factors with signal quality multipliers."""
        score = 0.0
        details = {}

        # FVG
        if smc_signals.get("fvg", {}).get("in_zone"):
            base_w = self.base_weights.get("fvg", 0.20)
            quality = self.signal_quality.get("fvg", 1.0)
            val = base_w * quality
            score += val
            details["fvg"] = round(val, 4)

        # Order Block
        if smc_signals.get("order_block", {}).get("at_zone"):
            base_w = self.base_weights.get("order_block", 0.25)
            quality = self.signal_quality.get("order_block", 1.0)
            val = base_w * quality
            score += val
            details["order_block"] = round(val, 4)

        # Liquidity Sweep
        if smc_signals.get("liquidity", {}).get("swept"):
            base_w = self.base_weights.get("liquidity_sweep", 0.20)
            quality = self.signal_quality.get("liquidity_sweep", 1.0)
            val = base_w * quality
            score += val
            details["liquidity_sweep"] = round(val, 4)

        # Structure Break (CHoCH vs BOS with different quality)
        if smc_signals.get("structure", {}).get("choch"):
            base_w = self.base_weights.get("structure_break", 0.30)
            quality = self.signal_quality.get("choch", 1.2)
            val = base_w * quality
            score += val
            details["choch"] = round(val, 4)
        elif smc_signals.get("structure", {}).get("bos"):
            base_w = self.base_weights.get("structure_break", 0.30) * 0.7
            quality = self.signal_quality.get("bos", 0.8)
            val = base_w * quality
            score += val
            details["bos"] = round(val, 4)

        return {"total": score, "details": details}

    def _score_technical_factors(
        self, direction: TrendDirection, indicators: Dict
    ) -> Dict:
        """Score technical indicator factors."""
        score = 0.0
        details = {}

        # EMA alignment
        ema_data = indicators.get("ema", {})
        if ema_data:
            ema_20 = ema_data.get(20)
            ema_50 = ema_data.get(50)
            if ema_20 and ema_50:
                if direction == TrendDirection.BULLISH and ema_20 > ema_50:
                    val = self.base_weights.get("ema_alignment", 0.10)
                    score += val
                    details["ema_alignment"] = val
                elif direction == TrendDirection.BEARISH and ema_20 < ema_50:
                    val = self.base_weights.get("ema_alignment", 0.10)
                    score += val
                    details["ema_alignment"] = val

        # RSI confirmation
        rsi = indicators.get("rsi")
        if rsi:
            if direction == TrendDirection.BULLISH and 40 < rsi < 70:
                val = self.base_weights.get("rsi_confirmation", 0.08)
                score += val
                details["rsi_confirmation"] = val
            elif direction == TrendDirection.BEARISH and 30 < rsi < 60:
                val = self.base_weights.get("rsi_confirmation", 0.08)
                score += val
                details["rsi_confirmation"] = val

        # MACD confirmation
        macd = indicators.get("macd", {})
        histogram = macd.get("histogram")
        if histogram:
            if direction == TrendDirection.BULLISH and histogram > 0:
                val = self.base_weights.get("macd_confirmation", 0.07)
                score += val
                details["macd_confirmation"] = val
            elif direction == TrendDirection.BEARISH and histogram < 0:
                val = self.base_weights.get("macd_confirmation", 0.07)
                score += val
                details["macd_confirmation"] = val

        return {"total": score, "details": details}

    def _score_bonus_factors(
        self, mtf_analysis: Dict, direction=None, ltf_data=None,
        mtf_weight_scale: float = 1.0,
    ) -> Dict:
        """Score bonus factors (MTF + LTF only, no ICT P/D zone)."""
        score = 0.0
        details = {}

        # MTF alignment — only benefit the direction MTF supports
        if mtf_analysis.get("is_aligned"):
            mtf_trend = mtf_analysis.get("dominant_trend")
            # Check if MTF trend matches trade direction
            direction_matches = True  # default: apply if no direction info
            if direction is not None and mtf_trend is not None:
                # dominant_trend can be TrendDirection enum or string
                trend_val = mtf_trend.value if hasattr(mtf_trend, "value") else str(mtf_trend).upper()
                dir_val = direction.value if hasattr(direction, "value") else str(direction).upper()
                direction_matches = trend_val == dir_val

            if direction_matches:
                mtf_weight = self.base_weights.get("mtf_alignment_bonus", 0.15)
                scaled = mtf_weight * mtf_weight_scale
                score += scaled
                details["mtf_alignment"] = round(scaled, 4)
            else:
                details["mtf_alignment"] = 0.0
                details["mtf_skipped"] = "direction_mismatch"

        # LTF (M5) confirmation
        if ltf_data is not None and direction is not None:
            if self._ltf_analyzer is None:
                from .ltf_confirmation import LTFConfirmation
                self._ltf_analyzer = LTFConfirmation(self.config)

            ltf_result = self._ltf_analyzer.calculate_confirmation(
                direction,
                ltf_data["m5_df"],
                ltf_data["current_m15_time"],
            )
            ltf_bonus_weight = self.base_weights.get("ltf_confirmation_bonus", 0.10)
            ltf_contribution = ltf_result["score"] * ltf_bonus_weight
            score += ltf_contribution
            details["ltf_confirmation"] = round(ltf_contribution, 4)
            details["ltf_details"] = ltf_result.get("details", {})

        return {"total": score, "details": details}

    def _apply_adjustments(
        self, market_analysis: Dict, regime: MarketRegime,
        direction: TrendDirection = None,
        opposing_smc: Dict = None,
    ) -> Dict:
        """Apply regime-aware adjustments including counter-trend and opposing CHoCH penalty."""
        adjustments = 0.0
        details = {}

        # Volatility adjustment
        volatility = market_analysis.get("volatility")
        if volatility and isinstance(volatility, dict):
            vol_adj = volatility.get("adjustments", {}).get("confluence_adjustment", 0)
            if vol_adj != 0:
                adjustments -= vol_adj
                details["volatility"] = -vol_adj

        # Regime-specific adjustment for volatile breakout: require higher score
        if regime == MarketRegime.VOLATILE_BREAKOUT:
            adjustments -= 0.05
            details["regime_volatile"] = -0.05

        # Counter-trend penalty: trading against a trending regime
        # e.g., BUY in WEAK_TREND_DOWN or SELL in STRONG_TREND_UP
        if direction and regime.is_trending:
            is_counter_trend = (
                (regime.is_bearish and direction == TrendDirection.BULLISH) or
                (regime.is_bullish and direction == TrendDirection.BEARISH)
            )
            if is_counter_trend:
                adjustments -= 0.10
                details["counter_trend"] = -0.10

        # Opposing CHoCH penalty: if the other direction has a CHoCH (reversal signal),
        # penalize this direction — CHoCH signals potential reversal against us
        if opposing_smc and opposing_smc.get("structure", {}).get("choch"):
            adjustments -= OPPOSING_CHOCH_PENALTY
            details["opposing_choch"] = -OPPOSING_CHOCH_PENALTY

        return {"total": adjustments, "details": details}

    def get_score_summary(self, score_data: Dict) -> str:
        """Get human-readable score summary."""
        score = score_data["score"]
        passing = "PASS" if score_data["passing"] else "FAIL"
        regime = score_data.get("regime", "UNKNOWN")

        breakdown = score_data.get("breakdown", {})
        smc_total = breakdown.get("smc", {}).get("total", 0)
        tech_total = breakdown.get("technical", {}).get("total", 0)
        bonus_total = breakdown.get("bonus", {}).get("total", 0)

        return (
            f"Confluence: {score:.2f} ({passing}) [{regime}] | "
            f"SMC: {smc_total:.2f} | Tech: {tech_total:.2f} | Bonus: {bonus_total:.2f}"
        )
