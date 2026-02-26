"""
Entry Quality Engine — Dynamic Entry Gate
==========================================
Classifies SMC entry signals into quality tiers AFTER the existing scoring pipeline.

DESIGN PRINCIPLE: Additive layer on top of AdaptiveConfluenceScorer.
  - Does NOT replace or duplicate scoring logic.
  - Wraps the output of calculate_score() and adds tier classification.
  - Used IDENTICALLY by BacktestEngine and TradingBot.

TIER DEFINITIONS:
  TIER_A (HIGH):   Institutional-grade setup
    - Score >= TIER_A_SCORE_WITH_ZONE AND zone fill confirmed (FVG in_zone / OB at_zone)
    - OR Score >= TIER_A_SCORE_NO_ZONE (very high confluence without zone — rare)
    - ICT model: CHoCH/BOS + zone retracement + (optionally) liquidity sweep
    - ~20-30% of all passing signals

  TIER_B (MEDIUM): Structural setup
    - Passes regime min_confluence (passing=True) but not TIER_A
    - Valid structure signal without zone precision
    - ~40-50% of all passing signals

  TIER_C (LOW):    Marginal / rejected
    - Below regime min_confluence (passing=False)
    - OR SMC-capped (smc_fill < MIN_SMC_FILL floor)
    - These are TRACKED but not entered
    - Analysis confirms whether skipping was correct

SYNC GUARANTEE:
  - Pure classification function — no state, no side effects
  - BacktestEngine and TradingBot call classify() with identical inputs
  - No look-ahead bias: operates on scores computed from bar[i] close data
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class EntryTier(Enum):
    TIER_A = "HIGH"
    TIER_B = "MEDIUM"
    TIER_C = "LOW"

    @property
    def label(self) -> str:
        return {"HIGH": "[A:HIGH]", "MEDIUM": "[B:MED]", "LOW": "[C:LOW]"}[self.value]

    @property
    def emoji(self) -> str:
        return {"HIGH": "A", "MEDIUM": "B", "LOW": "C"}[self.value]


@dataclass
class EntryQuality:
    tier: EntryTier
    score: float
    passing: bool
    has_zone: bool              # FVG in_zone OR OB at_zone
    displacement_strength: float  # Max recent bar range / ATR (momentum proxy)
    smc_capped: bool            # True if SMC fill was below floor
    reasons: List[str] = field(default_factory=list)

    @property
    def should_enter(self) -> bool:
        """Enter on TIER_A and TIER_B only. Compatible with existing bot logic."""
        return self.tier in (EntryTier.TIER_A, EntryTier.TIER_B)

    @property
    def sl_buffer_multiplier(self) -> float:
        """
        TIER_A: zone provides precise invalidation level → standard SL buffer.
        TIER_B: no zone precision → slightly wider SL to absorb noise.
        TIER_C: not traded.
        """
        return {
            EntryTier.TIER_A: 1.00,
            EntryTier.TIER_B: 1.20,  # 20% wider SL for zone-less entries
            EntryTier.TIER_C: 0.00,
        }[self.tier]

    def to_dict(self) -> Dict:
        return {
            "tier": self.tier.value,
            "tier_label": self.tier.label,
            "score": round(self.score, 4),
            "passing": self.passing,
            "has_zone": self.has_zone,
            "displacement_strength": round(self.displacement_strength, 3),
            "smc_capped": self.smc_capped,
            "reasons": self.reasons,
            "sl_buffer_mult": self.sl_buffer_multiplier,
        }


class EntryQualityEngine:
    """
    Classifies entry signals into quality tiers.

    Usage (both backtest and live are identical):
        engine = EntryQualityEngine()
        quality = engine.classify(
            score_result=adaptive_scorer.calculate_score(...),
            smc_signals=bullish_smc,
            technical={"atr": 5.2, "recent_bar_ranges": [4.1, 6.3, 5.0]},
        )
        if quality.should_enter:
            open_position(sl_buffer_mult=quality.sl_buffer_multiplier)
    """

    # TIER_A thresholds (calibrated from 6-month backtest — zone fill PF=89 vs 2.69 no-zone)
    # Lower threshold captures more zone-fill setups without over-classifying.
    # TIER_A_SCORE_WITH_ZONE=0.65 → ~4% of trades (balance quality vs quantity)
    TIER_A_SCORE_WITH_ZONE: float = 0.65   # Score + zone fill = institutional
    TIER_A_SCORE_NO_ZONE: float = 0.85     # Exceptional confluence without zone (rare)

    # Displacement strength thresholds (relative to ATR)
    DISP_STRONG_THRESHOLD: float = 1.5    # > 1.5x ATR = strong displacement
    DISP_WEAK_THRESHOLD: float = 0.50     # < 0.5x ATR = weak/no displacement

    def classify(
        self,
        score_result: Dict,
        smc_signals: Dict,
        technical: Dict,
    ) -> EntryQuality:
        """
        Classify entry signal quality tier.

        Args:
            score_result: Output from AdaptiveConfluenceScorer.calculate_score().
                          Required keys: score, passing, smc_capped, min_confluence.
            smc_signals:  Raw SMC signal dict for this direction
                          (bullish_smc or bearish_smc from get_bullish/bearish_signals).
            technical:    Dict with:
                          - "atr": current ATR value
                          - "recent_bar_ranges": list of bar (high-low) values for last N bars
                            (used for displacement strength proxy)

        Returns:
            EntryQuality with tier, score, reasons, and operational properties.
        """
        score = score_result.get("score", 0.0)
        passing = score_result.get("passing", False)
        smc_capped = score_result.get("smc_capped", False)
        reasons: List[str] = []

        # ── Zone Detection ──────────────────────────────────────────────────────
        # ICT principle: price retracing into FVG/OB = optimal trade entry zone.
        # Tier A requires this for precise SL placement and high-probability fill.
        has_fvg = smc_signals.get("fvg", {}).get("in_zone", False)
        has_ob  = smc_signals.get("order_block", {}).get("at_zone", False)
        has_zone = has_fvg or has_ob

        if has_fvg:
            reasons.append("zone:FVG")
        elif has_ob:
            reasons.append("zone:OB")

        # ── Displacement Strength ───────────────────────────────────────────────
        # PTJ principle: momentum of displacement candle = conviction of structure break.
        # Use max of recent 3 bars (captures the displacement, not just current bar).
        atr = max(technical.get("atr", 1.0) or 1.0, 0.001)
        recent_ranges = technical.get("recent_bar_ranges", [])
        if recent_ranges:
            max_recent_range = max(recent_ranges[-3:]) if len(recent_ranges) >= 3 else max(recent_ranges)
        else:
            max_recent_range = 0.0
        displacement_strength = max_recent_range / atr

        if displacement_strength >= self.DISP_STRONG_THRESHOLD:
            reasons.append(f"disp:strong({displacement_strength:.2f}x)")
        elif displacement_strength < self.DISP_WEAK_THRESHOLD and displacement_strength > 0:
            reasons.append(f"disp:weak({displacement_strength:.2f}x)")

        # ── Tier Classification ─────────────────────────────────────────────────

        # TIER_C: Below regime minimum or SMC-capped
        if not passing:
            tier = EntryTier.TIER_C
            min_conf = score_result.get("min_confluence", "?")
            reasons.append(f"below_min_conf({min_conf:.3f})")

        elif smc_capped:
            # SMC fill too weak — scoring system already capped it below threshold.
            # Even if "passing" is somehow True, downgrade to C.
            tier = EntryTier.TIER_C
            reasons.append("smc_floor_capped")

        # TIER_A: High score + zone fill (ICT institutional model)
        elif score >= self.TIER_A_SCORE_WITH_ZONE and has_zone:
            tier = EntryTier.TIER_A
            reasons.append(f"A:score{score:.3f}>={self.TIER_A_SCORE_WITH_ZONE}+zone")

        # TIER_A: Exceptional score without zone (rare — very strong confluence)
        elif score >= self.TIER_A_SCORE_NO_ZONE:
            tier = EntryTier.TIER_A
            reasons.append(f"A:score{score:.3f}>={self.TIER_A_SCORE_NO_ZONE}(no_zone)")

        # TIER_B: Passes regime threshold but not TIER_A
        else:
            tier = EntryTier.TIER_B
            reasons.append(f"B:passing(score={score:.3f})")

        return EntryQuality(
            tier=tier,
            score=score,
            passing=passing,
            has_zone=has_zone,
            displacement_strength=round(displacement_strength, 3),
            smc_capped=smc_capped,
            reasons=reasons,
        )

    def classify_both_directions(
        self,
        bull_score: Dict,
        bear_score: Dict,
        bull_smc: Dict,
        bear_smc: Dict,
        technical: Dict,
    ) -> Dict:
        """
        Classify both directions at once. Returns dict with 'bullish' and 'bearish' quality.
        Convenience wrapper for backtest loop where both directions are computed per bar.
        """
        return {
            "bullish": self.classify(bull_score, bull_smc, technical),
            "bearish": self.classify(bear_score, bear_smc, technical),
        }
