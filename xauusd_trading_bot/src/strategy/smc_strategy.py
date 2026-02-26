"""
Smart Money Concepts (SMC) Strategy
Main strategy orchestrator that combines all components to generate trading decisions.
"""

from typing import Dict, Optional, List
import polars as pl

from .entry_signals import EntrySignalGenerator
from .exit_signals import ExitSignalGenerator
from .signal_validator import SignalValidator
from ..core.constants import SignalType, TrendDirection, MarketRegime
from ..bot_logger import get_logger


class SMCStrategy:
    """Main SMC trading strategy."""

    def __init__(self, config: Dict):
        """
        Initialize SMC strategy.

        Args:
            config: Complete strategy configuration
        """
        self.logger = get_logger()
        self.config = config

        # Initialize components
        self.entry_generator = EntrySignalGenerator(config)
        self.exit_generator = ExitSignalGenerator(config)
        self.validator = SignalValidator(config)

        self.logger.info("SMC Strategy initialized")

    def analyze_and_signal(
        self,
        current_price: float,
        smc_analysis: Dict,
        technical_indicators: Dict,
        market_analysis: Dict,
        mtf_analysis: Dict,
        confluence_scores: Dict,
        current_positions: List[Dict],
        account_info: Dict,
        market_data: Dict,
        regime: MarketRegime = MarketRegime.RANGE_WIDE,
    ) -> Dict:
        """
        Complete analysis and signal generation.

        Args:
            current_price: Current market price
            smc_analysis: SMC indicator analysis
            technical_indicators: Technical indicator values
            market_analysis: Market condition analysis
            mtf_analysis: Multi-timeframe analysis
            confluence_scores: Calculated confluence scores
            current_positions: List of open positions
            account_info: Account information
            market_data: Current market data (spread, etc.)

        Returns:
            Complete strategy decision dictionary
        """
        try:
            # 1. Check exit conditions for existing positions
            exit_signals = []
            for position in current_positions:
                exit_signal = self.exit_generator.check_exit_conditions(
                    position,
                    current_price,
                    smc_analysis,
                    market_analysis,
                    technical_indicators,
                    regime=regime,
                )
                if exit_signal.get("should_exit", False):
                    exit_signals.append({
                        "position": position,
                        "signal": exit_signal,
                    })

            # 2. Generate entry signal (position limits handled dynamically by caller)
            entry_signal = self.entry_generator.generate_signal(
                current_price,
                confluence_scores,
                smc_analysis,
                market_analysis,
                mtf_analysis,
                technical_indicators,
                signal_time=market_data.get("time"),
                open_position_count=len(current_positions),
                regime=regime,
            )

            # 3. Validate entry signal
            if entry_signal.get("valid", False):
                validation = self.validator.validate_entry_signal(
                    entry_signal,
                    current_positions,
                    account_info,
                    market_data,
                )

                if validation.get("passed", False):
                    # Bug #54: do NOT add to history here — micro_account check
                    # in trading_bot.py runs AFTER this and may still reject the trade.
                    # Cooldown must only activate on EXECUTED trades, not validated-but-
                    # rejected ones.  trading_bot.py calls add_signal_to_history() after
                    # the order actually executes.
                    entry_signal["validation"] = validation
                else:
                    # Mark as invalid
                    entry_signal["valid"] = False
                    entry_signal["validation"] = validation
                    entry_signal["rejection_reason"] = validation.get("message")
                    entry_signal["reasons"] = [validation.get("message", "Validation failed")]

            # 4. Prepare complete decision
            decision = {
                "timestamp": market_data.get("time"),
                "current_price": current_price,
                "entry_signal": entry_signal,
                "exit_signals": exit_signals,
                "has_entry": entry_signal and entry_signal.get("valid", False),
                "has_exits": len(exit_signals) > 0,
                "market_analysis": {
                    "condition": market_analysis.get("condition"),
                    "trend": market_analysis.get("trend"),
                    "volatility": market_analysis.get("volatility"),
                    "favorable": market_analysis.get("is_favorable"),
                },
                "confluence": {
                    "bullish": confluence_scores.get("bullish", {}).get("score", 0),
                    "bearish": confluence_scores.get("bearish", {}).get("score", 0),
                },
                "mtf": {
                    "aligned": mtf_analysis.get("is_aligned", False),
                    "dominant": mtf_analysis.get("dominant_trend"),
                },
            }

            return decision

        except Exception as e:
            self.logger.error(f"Error in strategy analysis: {e}")
            return {
                "timestamp": None,
                "current_price": current_price,
                "entry_signal": None,
                "exit_signals": [],
                "has_entry": False,
                "has_exits": False,
                "error": str(e),
            }

    def get_decision_summary(self, decision: Dict) -> str:
        """
        Get human-readable decision summary.

        Args:
            decision: Strategy decision

        Returns:
            Summary string
        """
        lines = []

        # Entry signal
        if decision.get("has_entry"):
            entry = decision["entry_signal"]
            lines.append(self.entry_generator.get_signal_summary(entry))
        else:
            entry = decision.get("entry_signal")
            if entry:
                reasons = entry.get("reasons", ["No conditions met"])
                lines.append(f"NO ENTRY: {reasons[0]}")
            else:
                lines.append("NO ENTRY: Position limit reached or no signal")

        # Exit signals
        if decision.get("has_exits"):
            for exit_item in decision["exit_signals"]:
                exit_sig = exit_item["signal"]
                lines.append(self.exit_generator.get_exit_summary(exit_sig))

        # Market condition
        market = decision.get("market_analysis", {})
        lines.append(
            f"Market: {market.get('condition')} | "
            f"Trend: {market.get('trend')} | "
            f"Favorable: {market.get('favorable')}"
        )

        # Confluence
        conf = decision.get("confluence", {})
        lines.append(
            f"Confluence - Bull: {conf.get('bullish', 0):.2f}, "
            f"Bear: {conf.get('bearish', 0):.2f}"
        )

        return "\n".join(lines)

    def get_strategy_state(self) -> Dict:
        """
        Get current strategy state.

        Returns:
            State dictionary
        """
        return {
            "recent_signals": len(self.validator.recent_signals),
            "config": {
                "min_confluence": self.entry_generator.min_confluence,
                "max_positions": self.config.get("trading", {}).get("max_open_positions", 3),
            },
        }
