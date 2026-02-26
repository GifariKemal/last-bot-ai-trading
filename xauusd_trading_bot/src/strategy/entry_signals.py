"""
Entry Signal Generator
Generates entry signals based on SMC strategy and confluence analysis.
"""

from typing import Dict, Optional
from datetime import datetime
import polars as pl

from ..core.constants import SignalType, TrendDirection, MarketRegime
from ..bot_logger import get_logger


class EntrySignalGenerator:
    """Generate entry signals based on strategy rules."""

    # Minimum number of active SMC signals required for entry
    MIN_SMC_SIGNALS = 2

    # RSI bounce protection: don't enter if RSI was at extreme within N bars
    RSI_BOUNCE_LOOKBACK = 5  # bars
    RSI_EXTREME_OVERSOLD = 25  # Don't SELL if RSI was below this recently
    RSI_EXTREME_OVERBOUGHT = 75  # Don't BUY if RSI was above this recently

    # RSI hard block: filters genuinely overextended RSI readings (Check #5)
    # XAUUSD strong trend: RSI typically 65-80. Above 85 = overextended, risk of sharp reversal.
    # Bounce protection (Check #6) handles RSI coming DOWN from 75+; hard block handles RSI STILL RISING.
    RSI_HARD_OVERBOUGHT = 85  # Hard block LONG above this (overextended bull)
    RSI_HARD_OVERSOLD = 15   # Hard block SHORT below this (overextended bear)

    def __init__(self, config: Dict):
        """
        Initialize entry signal generator.

        Args:
            config: Strategy configuration
        """
        self.logger = get_logger()
        self.config = config
        self.entry_config = config.get("strategy", {}).get("entry", {})
        self.min_confluence = self.entry_config.get("min_confluence_score", 0.65)
        self._recent_rsi_values = []  # Track recent RSI for bounce detection

    def update_rsi_history(self, rsi_value: float) -> None:
        """Track RSI values for bounce detection."""
        self._recent_rsi_values.append(rsi_value)
        if len(self._recent_rsi_values) > self.RSI_BOUNCE_LOOKBACK:
            self._recent_rsi_values = self._recent_rsi_values[-self.RSI_BOUNCE_LOOKBACK:]

    def _count_active_smc_signals(self, smc_signals: Dict) -> int:
        """Count how many SMC signals are active."""
        count = 0
        if smc_signals.get("fvg", {}).get("in_zone", False):
            count += 1
        if smc_signals.get("order_block", {}).get("at_zone", False):
            count += 1
        if smc_signals.get("liquidity", {}).get("swept", False):
            count += 1
        if smc_signals.get("structure", {}).get("choch", False):
            count += 1
        if smc_signals.get("structure", {}).get("bos", False):
            count += 1
        return count

    def _is_rsi_bouncing_from_oversold(self) -> bool:
        """Check if RSI recently hit extreme oversold and is now bouncing up."""
        if len(self._recent_rsi_values) < 2:
            return False
        had_extreme = any(v < self.RSI_EXTREME_OVERSOLD for v in self._recent_rsi_values[:-1])
        current_rising = self._recent_rsi_values[-1] > self.RSI_EXTREME_OVERSOLD
        return had_extreme and current_rising

    def _is_rsi_bouncing_from_overbought(self) -> bool:
        """Check if RSI recently hit extreme overbought and is now bouncing down."""
        if len(self._recent_rsi_values) < 2:
            return False
        had_extreme = any(v > self.RSI_EXTREME_OVERBOUGHT for v in self._recent_rsi_values[:-1])
        current_falling = self._recent_rsi_values[-1] < self.RSI_EXTREME_OVERBOUGHT
        return had_extreme and current_falling

    def _get_min_smc_for_regime(self, regime: MarketRegime) -> int:
        """Get minimum SMC signals required based on regime.
        Config-driven: reads settings.yaml regime_weights.*.min_smc_signals."""
        regime_weights = self.config.get("regime_weights", {})
        category = regime.category  # "trending", "ranging", "breakout", "reversal"
        min_smc = regime_weights.get(category, {}).get("min_smc_signals", self.MIN_SMC_SIGNALS)
        return int(min_smc)

    def generate_signal(
        self,
        current_price: float,
        confluence_scores: Dict,
        smc_signals: Dict,
        market_analysis: Dict,
        mtf_analysis: Dict,
        technical_indicators: Dict,
        signal_time: Optional[datetime] = None,
        open_position_count: int = 0,
        regime: MarketRegime = MarketRegime.RANGE_WIDE,
    ) -> Dict:
        """
        Generate entry signal if conditions are met.

        Args:
            current_price: Current market price
            confluence_scores: Bullish and bearish confluence scores
            smc_signals: SMC signal data
            market_analysis: Market condition analysis
            mtf_analysis: Multi-timeframe analysis
            technical_indicators: Technical indicator values

        Returns:
            Signal dictionary
        """
        try:
            # Use provided time or current time
            self._current_signal_time = signal_time or datetime.utcnow()

            # Min SMC signals: regime-dynamic, increased for pyramid positions
            base_min_smc = self._get_min_smc_for_regime(regime)
            self._min_smc_for_entry = max(base_min_smc, self.MIN_SMC_SIGNALS) if open_position_count >= 1 else base_min_smc

            # Check bullish conditions
            bullish_signal = self._check_bullish_entry(
                current_price,
                confluence_scores["bullish"],
                smc_signals["bullish"],
                market_analysis,
                mtf_analysis,
                technical_indicators,
            )

            # Check bearish conditions
            bearish_signal = self._check_bearish_entry(
                current_price,
                confluence_scores["bearish"],
                smc_signals["bearish"],
                market_analysis,
                mtf_analysis,
                technical_indicators,
            )

            # Determine which signal to take (if any)
            signal = self._select_best_signal(bullish_signal, bearish_signal)

            return signal

        except Exception as e:
            self.logger.error(f"Error generating entry signal: {e}")
            return self._no_signal()

    def _check_bullish_entry(
        self,
        current_price: float,
        confluence_score: Dict,
        smc_signals: Dict,
        market_analysis: Dict,
        mtf_analysis: Dict,
        technical_indicators: Dict,
    ) -> Dict:
        """Check bullish entry conditions."""

        # Base requirements
        if not confluence_score.get("passing", False):
            return self._no_signal(reason="Bullish confluence too low")

        score = confluence_score["score"]

        # Build entry criteria checks
        # NOTE: confluence_met = True because adaptive scorer's "passing" flag
        # (checked above) is the single source of truth for the threshold.
        # self.min_confluence (from trading_rules.yaml) was a legacy double-gate
        # that overrode the regime-adaptive thresholds set by the V3 optimizer.
        checks = {
            "confluence_met": True,  # Already verified by confluence_score["passing"] above
            "fvg_or_ob": False,
            "structure_support": False,
            "mtf_aligned": True,  # Default true if not required
            "market_favorable": True,
        }

        reasons = []

        # 1. Check FVG or Order Block requirement
        if self.entry_config.get("require_fvg_or_ob", True):
            in_fvg = smc_signals.get("fvg", {}).get("in_zone", False)
            at_ob = smc_signals.get("order_block", {}).get("at_zone", False)

            checks["fvg_or_ob"] = in_fvg or at_ob

            if not checks["fvg_or_ob"]:
                reasons.append("Not at FVG or Order Block")
        else:
            checks["fvg_or_ob"] = True

        # 2. Check structure support (BOS or CHoCH)
        if self.entry_config.get("require_structure_support", True):
            has_choch = smc_signals.get("structure", {}).get("choch", False)
            has_bos = smc_signals.get("structure", {}).get("bos", False)

            checks["structure_support"] = has_choch or has_bos

            if not checks["structure_support"]:
                reasons.append("No structure support (no BOS/CHoCH)")
        else:
            checks["structure_support"] = True

        # 3. Check MTF alignment
        if self.entry_config.get("require_mtf_alignment", True):
            checks["mtf_aligned"] = mtf_analysis.get("is_aligned", False)
            mtf_direction = mtf_analysis.get("dominant_trend")

            if not checks["mtf_aligned"] or mtf_direction != TrendDirection.BULLISH:
                reasons.append("MTF not aligned bullish")
                checks["mtf_aligned"] = False
        else:
            checks["mtf_aligned"] = True

        # 4. Check market favorability
        checks["market_favorable"] = market_analysis.get("is_favorable", False)
        if not checks["market_favorable"]:
            reasons.append("Market conditions unfavorable")

        # 5. Check technical confirmations (extreme overbought only — not normal trending RSI)
        # RSI 70-85 is normal in trending XAUUSD bull moves; only block at extreme >85
        rsi = technical_indicators.get("rsi")
        if rsi and rsi > self.RSI_HARD_OVERBOUGHT:
            reasons.append(f"RSI extreme overbought ({rsi:.1f} > {self.RSI_HARD_OVERBOUGHT})")
            checks["technical_ok"] = False
        else:
            checks["technical_ok"] = True

        # 6. RSI bounce protection - don't BUY into overbought bounce
        if self._is_rsi_bouncing_from_overbought():
            reasons.append("RSI bouncing from extreme overbought (bounce in progress)")
            checks["rsi_bounce_ok"] = False
        else:
            checks["rsi_bounce_ok"] = True

        # 7. SMC signal count gate (position-aware: 1 SMC for first entry, 2 for pyramid)
        smc_count = self._count_active_smc_signals(smc_signals)
        min_smc = getattr(self, "_min_smc_for_entry", self.MIN_SMC_SIGNALS)
        pos_label = "2/3" if min_smc > 1 else "1"
        if smc_count < min_smc:
            reasons.append(f"Only {smc_count} SMC signal(s), need {min_smc}+ (position #{pos_label})")
            checks["min_smc_signals"] = False
        else:
            checks["min_smc_signals"] = True

        # 8. H1 Bias filter — REMOVED in V3 (replaced by regime-adaptive scoring)
        checks["h1_bias_ok"] = True

        # All checks must pass
        all_pass = all(checks.values())

        if all_pass:
            return {
                "type": SignalType.ENTRY_LONG,
                "direction": "BUY",
                "price": current_price,
                "confidence": score,
                "timestamp": self._current_signal_time,
                "valid": True,
                "checks": checks,
                "smc_context": self._extract_smc_context(smc_signals),
                "reasons": ["All entry conditions met"],
            }
        else:
            return self._no_signal(
                reason=f"Bullish checks failed: {', '.join(reasons)}"
            )

    def _check_bearish_entry(
        self,
        current_price: float,
        confluence_score: Dict,
        smc_signals: Dict,
        market_analysis: Dict,
        mtf_analysis: Dict,
        technical_indicators: Dict,
    ) -> Dict:
        """Check bearish entry conditions."""

        # Base requirements
        if not confluence_score.get("passing", False):
            return self._no_signal(reason="Bearish confluence too low")

        score = confluence_score["score"]

        # Build entry criteria checks
        # NOTE: confluence_met = True because adaptive scorer's "passing" flag
        # (checked above) is the single source of truth for the threshold.
        checks = {
            "confluence_met": True,  # Already verified by confluence_score["passing"] above
            "fvg_or_ob": False,
            "structure_support": False,
            "mtf_aligned": True,
            "market_favorable": True,
        }

        reasons = []

        # 1. Check FVG or Order Block requirement
        if self.entry_config.get("require_fvg_or_ob", True):
            in_fvg = smc_signals.get("fvg", {}).get("in_zone", False)
            at_ob = smc_signals.get("order_block", {}).get("at_zone", False)

            checks["fvg_or_ob"] = in_fvg or at_ob

            if not checks["fvg_or_ob"]:
                reasons.append("Not at FVG or Order Block")
        else:
            checks["fvg_or_ob"] = True

        # 2. Check structure support
        if self.entry_config.get("require_structure_support", True):
            has_choch = smc_signals.get("structure", {}).get("choch", False)
            has_bos = smc_signals.get("structure", {}).get("bos", False)

            checks["structure_support"] = has_choch or has_bos

            if not checks["structure_support"]:
                reasons.append("No structure support (no BOS/CHoCH)")
        else:
            checks["structure_support"] = True

        # 3. Check MTF alignment
        if self.entry_config.get("require_mtf_alignment", True):
            checks["mtf_aligned"] = mtf_analysis.get("is_aligned", False)
            mtf_direction = mtf_analysis.get("dominant_trend")

            if not checks["mtf_aligned"] or mtf_direction != TrendDirection.BEARISH:
                reasons.append("MTF not aligned bearish")
                checks["mtf_aligned"] = False
        else:
            checks["mtf_aligned"] = True

        # 4. Check market favorability
        checks["market_favorable"] = market_analysis.get("is_favorable", False)
        if not checks["market_favorable"]:
            reasons.append("Market conditions unfavorable")

        # 5. Check technical confirmations (extreme oversold only — not normal trending RSI)
        # RSI 15-30 is normal in trending XAUUSD bear moves; only block at extreme <15
        rsi = technical_indicators.get("rsi")
        if rsi and rsi < self.RSI_HARD_OVERSOLD:
            reasons.append(f"RSI extreme oversold ({rsi:.1f} < {self.RSI_HARD_OVERSOLD})")
            checks["technical_ok"] = False
        else:
            checks["technical_ok"] = True

        # 6. RSI bounce protection - don't SELL into oversold bounce
        if self._is_rsi_bouncing_from_oversold():
            reasons.append("RSI bouncing from extreme oversold (bounce in progress)")
            checks["rsi_bounce_ok"] = False
        else:
            checks["rsi_bounce_ok"] = True

        # 7. SMC signal count gate (position-aware: 1 SMC for first entry, 2 for pyramid)
        smc_count = self._count_active_smc_signals(smc_signals)
        min_smc = getattr(self, "_min_smc_for_entry", self.MIN_SMC_SIGNALS)
        pos_label = "2/3" if min_smc > 1 else "1"
        if smc_count < min_smc:
            reasons.append(f"Only {smc_count} SMC signal(s), need {min_smc}+ (position #{pos_label})")
            checks["min_smc_signals"] = False
        else:
            checks["min_smc_signals"] = True

        # 8. H1 Bias filter — REMOVED in V3 (replaced by regime-adaptive scoring)
        checks["h1_bias_ok"] = True

        # All checks must pass
        all_pass = all(checks.values())

        if all_pass:
            return {
                "type": SignalType.ENTRY_SHORT,
                "direction": "SELL",
                "price": current_price,
                "confidence": score,
                "timestamp": self._current_signal_time,
                "valid": True,
                "checks": checks,
                "smc_context": self._extract_smc_context(smc_signals),
                "reasons": ["All entry conditions met"],
            }
        else:
            return self._no_signal(
                reason=f"Bearish checks failed: {', '.join(reasons)}"
            )

    def _select_best_signal(
        self, bullish_signal: Dict, bearish_signal: Dict
    ) -> Dict:
        """
        Select the best signal if both are valid.

        Args:
            bullish_signal: Bullish signal data
            bearish_signal: Bearish signal data

        Returns:
            Best signal
        """
        bullish_valid = bullish_signal.get("valid", False)
        bearish_valid = bearish_signal.get("valid", False)

        # Only one valid
        if bullish_valid and not bearish_valid:
            return bullish_signal
        elif bearish_valid and not bullish_valid:
            return bearish_signal

        # Both valid - choose higher confidence
        elif bullish_valid and bearish_valid:
            bullish_conf = bullish_signal.get("confidence", 0)
            bearish_conf = bearish_signal.get("confidence", 0)

            if bullish_conf > bearish_conf:
                return bullish_signal
            else:
                return bearish_signal

        # Neither valid
        else:
            bull_reason = bullish_signal.get("reasons", ["?"])[0]
            bear_reason = bearish_signal.get("reasons", ["?"])[0]
            return self._no_signal(
                reason=f"BULL: {bull_reason} | BEAR: {bear_reason}"
            )

    def _extract_smc_context(self, smc_signals: Dict) -> Dict:
        """Extract key SMC context for the signal."""
        return {
            "in_fvg": smc_signals.get("fvg", {}).get("in_zone", False),
            "at_order_block": smc_signals.get("order_block", {}).get("at_zone", False),
            "liquidity_swept": smc_signals.get("liquidity", {}).get("swept", False),
            "has_choch": smc_signals.get("structure", {}).get("choch", False),
            "has_bos": smc_signals.get("structure", {}).get("bos", False),
        }

    def _no_signal(self, reason: str = "No entry conditions met") -> Dict:
        """Return no signal structure."""
        return {
            "type": SignalType.NEUTRAL,
            "direction": None,
            "price": None,
            "confidence": 0.0,
            "timestamp": getattr(self, '_current_signal_time', datetime.utcnow()),
            "valid": False,
            "checks": {},
            "smc_context": {},
            "reasons": [reason],
        }

    def get_signal_summary(self, signal: Dict) -> str:
        """
        Get human-readable signal summary.

        Args:
            signal: Signal data

        Returns:
            Summary string
        """
        if not signal.get("valid", False):
            return f"NO SIGNAL: {signal.get('reasons', ['Unknown'])[0]}"

        direction = signal.get("direction", "NONE")
        confidence = signal.get("confidence", 0)
        price = signal.get("price", 0)

        smc = signal.get("smc_context", {})
        smc_factors = []
        if smc.get("in_fvg"):
            smc_factors.append("FVG")
        if smc.get("at_order_block"):
            smc_factors.append("OB")
        if smc.get("has_choch"):
            smc_factors.append("CHoCH")
        elif smc.get("has_bos"):
            smc_factors.append("BOS")

        factors_str = "+".join(smc_factors) if smc_factors else "None"

        return (
            f"SIGNAL: {direction} @ {price:.2f} | "
            f"Confidence: {confidence:.2f} | "
            f"SMC: {factors_str}"
        )
