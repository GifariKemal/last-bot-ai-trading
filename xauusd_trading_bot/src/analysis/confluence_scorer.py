"""
Confluence Scorer (V2 — DEPRECATED)
Calculates confluence score by combining multiple analysis factors.
Determines trade quality and entry validity.

DEPRECATED: Replaced by AdaptiveConfluenceScorer in V3.
Kept for backward compatibility with live trading_bot.py until V3 migration.
"""

from typing import Dict
import polars as pl

from ..core.constants import TrendDirection, FVGType, OrderBlockType
from ..bot_logger import get_logger


class ConfluenceScorer:
    """Calculate confluence scores for trade signals."""

    def __init__(self, config: Dict):
        """
        Initialize confluence scorer.

        Args:
            config: Configuration with weights
        """
        self.logger = get_logger()
        self.config = config
        self.weights = config.get("confluence_weights", {})
        self._ltf_analyzer = None  # Lazy init on first use

    def calculate_score(
        self,
        direction: TrendDirection,
        smc_signals: Dict,
        technical_indicators: Dict,
        market_analysis: Dict,
        mtf_analysis: Dict,
        ltf_data: Dict = None,
    ) -> Dict:
        """
        Calculate comprehensive confluence score.

        Args:
            direction: Trade direction
            smc_signals: SMC signal data
            technical_indicators: Technical indicator values
            market_analysis: Market condition analysis
            mtf_analysis: Multi-timeframe analysis

        Returns:
            Score dictionary with breakdown
        """
        try:
            score = 0.0
            breakdown = {}

            # 1. SMC Factors
            smc_score = self._score_smc_factors(direction, smc_signals)
            score += smc_score["total"]
            breakdown["smc"] = smc_score

            # 2. Technical Factors
            tech_score = self._score_technical_factors(direction, technical_indicators)
            score += tech_score["total"]
            breakdown["technical"] = tech_score

            # 3. Bonus Factors
            bonus_score = self._score_bonus_factors(
                mtf_analysis, market_analysis,
                direction=direction, ltf_data=ltf_data,
                smc_signals=smc_signals,
            )
            score += bonus_score["total"]
            breakdown["bonus"] = bonus_score

            # 4. Adjustments
            adjustments = self._apply_adjustments(score, market_analysis)
            final_score = score + adjustments["total"]
            breakdown["adjustments"] = adjustments

            return {
                "score": max(0.0, min(1.0, final_score)),  # Clamp to 0-1
                "raw_score": score,
                "breakdown": breakdown,
                "passing": final_score >= self.config.get("strategy", {}).get("entry", {}).get("min_confluence_score", 0.65),
            }

        except Exception as e:
            self.logger.error(f"Error calculating confluence score: {e}")
            return {
                "score": 0.0,
                "raw_score": 0.0,
                "breakdown": {},
                "passing": False,
            }

    def _score_smc_factors(self, direction: TrendDirection, smc_signals: Dict) -> Dict:
        """Score SMC factors."""
        score = 0.0
        details = {}

        # FVG
        if direction == TrendDirection.BULLISH and smc_signals.get("fvg", {}).get("in_zone"):
            fvg_weight = self.weights.get("fvg", 0.20)
            score += fvg_weight
            details["fvg"] = fvg_weight
        elif direction == TrendDirection.BEARISH and smc_signals.get("fvg", {}).get("in_zone"):
            fvg_weight = self.weights.get("fvg", 0.20)
            score += fvg_weight
            details["fvg"] = fvg_weight

        # Order Block
        if smc_signals.get("order_block", {}).get("at_zone"):
            ob_weight = self.weights.get("order_block", 0.25)
            score += ob_weight
            details["order_block"] = ob_weight

        # Liquidity Sweep
        if smc_signals.get("liquidity", {}).get("swept"):
            liq_weight = self.weights.get("liquidity_sweep", 0.20)
            score += liq_weight
            details["liquidity_sweep"] = liq_weight

        # Structure Break
        if smc_signals.get("structure", {}).get("choch"):
            choch_weight = self.weights.get("structure_break", 0.30)
            score += choch_weight
            details["choch"] = choch_weight
        elif smc_signals.get("structure", {}).get("bos"):
            bos_weight = self.weights.get("structure_break", 0.30) * 0.7  # BOS worth less than CHoCH
            score += bos_weight
            details["bos"] = bos_weight

        return {
            "total": score,
            "details": details,
        }

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
                    ema_weight = self.weights.get("ema_alignment", 0.10)
                    score += ema_weight
                    details["ema_alignment"] = ema_weight
                elif direction == TrendDirection.BEARISH and ema_20 < ema_50:
                    ema_weight = self.weights.get("ema_alignment", 0.10)
                    score += ema_weight
                    details["ema_alignment"] = ema_weight

        # RSI confirmation
        rsi = indicators.get("rsi")
        if rsi:
            if direction == TrendDirection.BULLISH and 40 < rsi < 70:
                # Bullish but not overbought
                rsi_weight = self.weights.get("rsi_confirmation", 0.08)
                score += rsi_weight
                details["rsi_confirmation"] = rsi_weight
            elif direction == TrendDirection.BEARISH and 30 < rsi < 60:
                # Bearish but not oversold
                rsi_weight = self.weights.get("rsi_confirmation", 0.08)
                score += rsi_weight
                details["rsi_confirmation"] = rsi_weight

        # MACD confirmation
        macd = indicators.get("macd", {})
        if macd:
            histogram = macd.get("histogram")
            if histogram:
                if direction == TrendDirection.BULLISH and histogram > 0:
                    macd_weight = self.weights.get("macd_confirmation", 0.07)
                    score += macd_weight
                    details["macd_confirmation"] = macd_weight
                elif direction == TrendDirection.BEARISH and histogram < 0:
                    macd_weight = self.weights.get("macd_confirmation", 0.07)
                    score += macd_weight
                    details["macd_confirmation"] = macd_weight

        return {
            "total": score,
            "details": details,
        }

    def _score_bonus_factors(
        self, mtf_analysis: Dict, market_analysis: Dict,
        direction=None, ltf_data: Dict = None, smc_signals: Dict = None,
    ) -> Dict:
        """Score bonus factors."""
        score = 0.0
        details = {}

        # MTF alignment
        if mtf_analysis.get("is_aligned"):
            mtf_weight = self.weights.get("mtf_alignment_bonus", 0.15)
            score += mtf_weight
            details["mtf_alignment"] = mtf_weight

        # ICT Premium/Discount Zone (Semi-Hard Gate)
        # Right zone  → bonus  (deeper = bigger bonus, max pd_zone_bonus)
        # Wrong zone  → penalty (fixed -pd_zone_penalty, regardless of depth)
        # No swing data → neutral (0 contribution)
        if smc_signals is not None and direction is not None:
            current_price = smc_signals.get("current_price")
            structure = smc_signals.get("structure", {})
            swing_high = structure.get("recent_swing_high")
            swing_low = structure.get("recent_swing_low")

            if current_price and swing_high and swing_low and swing_high > swing_low:
                dealing_range = swing_high - swing_low
                midpoint = swing_low + (dealing_range * 0.5)
                pd_bonus = self.weights.get("pd_zone_bonus", 0.08)
                pd_penalty = self.weights.get("pd_zone_penalty", 0.08)

                if direction == TrendDirection.BEARISH and current_price > midpoint:
                    # Premium zone → SELL favorable: bonus proportional to depth
                    depth = min(1.0, (current_price - midpoint) / (dealing_range * 0.5))
                    contribution = round(pd_bonus * depth, 4)
                    score += contribution
                    details["pd_zone"] = contribution
                    details["pd_zone_type"] = "premium"
                elif direction == TrendDirection.BULLISH and current_price < midpoint:
                    # Discount zone → BUY favorable: bonus proportional to depth
                    depth = min(1.0, (midpoint - current_price) / (dealing_range * 0.5))
                    contribution = round(pd_bonus * depth, 4)
                    score += contribution
                    details["pd_zone"] = contribution
                    details["pd_zone_type"] = "discount"
                elif direction == TrendDirection.BEARISH and current_price <= midpoint:
                    # Discount zone → SELL unfavorable: penalty
                    score -= pd_penalty
                    details["pd_zone"] = -pd_penalty
                    details["pd_zone_type"] = "discount_penalty"
                elif direction == TrendDirection.BULLISH and current_price >= midpoint:
                    # Premium zone → BUY unfavorable: penalty
                    score -= pd_penalty
                    details["pd_zone"] = -pd_penalty
                    details["pd_zone_type"] = "premium_penalty"

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
            ltf_bonus_weight = self.weights.get("ltf_confirmation_bonus", 0.10)
            ltf_contribution = ltf_result["score"] * ltf_bonus_weight
            score += ltf_contribution
            details["ltf_confirmation"] = round(ltf_contribution, 4)
            details["ltf_details"] = ltf_result.get("details", {})

        return {
            "total": score,
            "details": details,
        }

    def _apply_adjustments(self, score: float, market_analysis: Dict) -> Dict:
        """Apply adjustments based on market conditions."""
        adjustments = 0.0
        details = {}

        # Volatility adjustment
        volatility = market_analysis.get("volatility")
        if volatility:
            vol_adj = volatility.get("adjustments", {}).get("confluence_adjustment", 0)
            if vol_adj != 0:
                adjustments -= vol_adj  # Subtract because high vol requires higher score
                details["volatility"] = -vol_adj

        # Market condition adjustment
        condition = market_analysis.get("condition")
        if condition:
            market_state_config = self.config.get("market_conditions", {}).get("market_state", {})

            # Get adjustment based on condition
            condition_name = condition.value.lower().replace("_", "")
            if "trending" in condition_name:
                conf_adj = market_state_config.get("trending", {}).get("confluence_adjustment", 0)
            elif "ranging" in condition_name:
                conf_adj = market_state_config.get("ranging", {}).get("confluence_adjustment", 0)
            elif "volatile" in condition_name:
                conf_adj = market_state_config.get("volatile", {}).get("confluence_adjustment", 0)
            else:
                conf_adj = 0

            if conf_adj != 0:
                adjustments -= conf_adj
                details["market_condition"] = -conf_adj

        return {
            "total": adjustments,
            "details": details,
        }

    def get_score_summary(self, score_data: Dict) -> str:
        """
        Get human-readable score summary.

        Args:
            score_data: Score calculation results

        Returns:
            Summary string
        """
        score = score_data["score"]
        passing = "PASS" if score_data["passing"] else "FAIL"

        breakdown = score_data.get("breakdown", {})
        smc_total = breakdown.get("smc", {}).get("total", 0)
        tech_total = breakdown.get("technical", {}).get("total", 0)
        bonus_total = breakdown.get("bonus", {}).get("total", 0)

        summary = (
            f"Confluence: {score:.2f} ({passing}) | "
            f"SMC: {smc_total:.2f} | "
            f"Tech: {tech_total:.2f} | "
            f"Bonus: {bonus_total:.2f}"
        )

        return summary
