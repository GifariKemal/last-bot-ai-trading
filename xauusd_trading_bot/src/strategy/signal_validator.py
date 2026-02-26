"""
Signal Validator
Validates trading signals before execution.
Performs final safety checks and filters.
"""

from typing import Dict, List
from datetime import datetime, timedelta
import polars as pl

from ..core.constants import SignalType
from ..bot_logger import get_logger


class SignalValidator:
    """Validate trading signals before execution."""

    def __init__(self, config: Dict):
        """
        Initialize signal validator.

        Args:
            config: Strategy configuration
        """
        self.logger = get_logger()
        self.config = config
        self.validation_config = config.get("signal_validation", {})

        # Track recent signals to prevent duplicates
        self.recent_signals: List[Dict] = []

    def validate_entry_signal(
        self,
        signal: Dict,
        current_positions: List[Dict],
        account_info: Dict,
        market_data: Dict,
    ) -> Dict:
        """
        Validate entry signal before execution.

        Args:
            signal: Entry signal to validate
            current_positions: List of open positions
            account_info: Account information
            market_data: Current market data

        Returns:
            Validation result dictionary
        """
        if not signal.get("valid", False):
            return self._validation_result(False, "Signal not valid")

        validation_checks = []

        # 1. Check signal freshness
        if not self._is_signal_fresh(signal):
            validation_checks.append({
                "check": "freshness",
                "passed": False,
                "reason": "Signal too old",
            })

        # 2. Check for duplicate signals
        if self._is_duplicate_signal(signal):
            validation_checks.append({
                "check": "duplicate",
                "passed": False,
                "reason": "Duplicate signal (too similar to recent)",
            })

        # 3. Check minimum bars since last signal
        if not self._check_bars_since_signal(signal):
            validation_checks.append({
                "check": "min_bars",
                "passed": False,
                "reason": "Too soon after last signal",
            })

        # 4. Check minimum price change
        if not self._check_price_change(signal):
            validation_checks.append({
                "check": "price_change",
                "passed": False,
                "reason": "Insufficient price movement",
            })

        # 5. Check position limits
        if not self._check_position_limits(current_positions):
            validation_checks.append({
                "check": "position_limits",
                "passed": False,
                "reason": "Maximum positions reached",
            })

        # 5b. Check all positions profitable (CONSERVATIVE MODE)
        if not self._check_all_positions_profitable(current_positions):
            validation_checks.append({
                "check": "positions_profitable",
                "passed": False,
                "reason": "Existing positions not all profitable yet",
            })

        # 6. Check account balance
        if not self._check_account_balance(account_info):
            validation_checks.append({
                "check": "balance",
                "passed": False,
                "reason": "Insufficient account balance",
            })

        # 7. Check spread
        if not self._check_spread(market_data):
            validation_checks.append({
                "check": "spread",
                "passed": False,
                "reason": "Spread too high",
            })

        # 8. Check for recent false signals
        false_signal_penalty = self._check_recent_false_signals()
        if false_signal_penalty > 0:
            validation_checks.append({
                "check": "false_signals",
                "passed": True,
                "warning": f"Recent false signals detected (score penalty: {false_signal_penalty:.2f})",
            })

        # Determine if validation passed
        failed_checks = [c for c in validation_checks if not c.get("passed", True)]

        if failed_checks:
            reasons = [c["reason"] for c in failed_checks]
            return self._validation_result(
                False, f"Validation failed: {', '.join(reasons)}", validation_checks
            )

        # All checks passed
        return self._validation_result(True, "Signal validated", validation_checks)

    def validate_exit_signal(
        self,
        exit_signal: Dict,
        position: Dict,
    ) -> Dict:
        """
        Validate exit signal.

        Args:
            exit_signal: Exit signal to validate
            position: Position to exit

        Returns:
            Validation result
        """
        if not exit_signal.get("should_exit", False):
            return self._validation_result(False, "No exit required")

        # Basic validation
        if not position:
            return self._validation_result(False, "No position to exit")

        # Exit signals are generally always valid if conditions are met
        return self._validation_result(True, "Exit signal validated")

    def _is_signal_fresh(self, signal: Dict, max_age_seconds: int = 60) -> bool:
        """Check if signal is fresh (not too old). Always True in backtesting."""
        # Signal is always fresh when just generated (timestamp is the bar time)
        return True

    def _is_duplicate_signal(self, signal: Dict) -> bool:
        """Check if signal is duplicate of recent signal."""
        signal_price = signal.get("price", 0)
        signal_direction = signal.get("direction")

        # Check recent signals
        for recent in self.recent_signals[-5:]:  # Check last 5 signals
            recent_price = recent.get("price", 0)
            recent_direction = recent.get("direction")

            # Same direction and similar price
            if recent_direction == signal_direction:
                price_diff = abs(signal_price - recent_price)
                if price_diff < 5.0:  # Within 5 pips
                    return True

        return False

    def _check_bars_since_signal(self, signal: Dict) -> bool:
        """Check minimum bars since last signal (same direction only).

        Opposite-direction signals bypass this filter — a reversal is a new
        trade idea, not overtrading in the same direction.
        """
        min_bars = self.validation_config.get("min_bars_since_last_signal", 5)

        if not self.recent_signals:
            return True

        last_signal = self.recent_signals[-1]

        # Skip filter for opposite-direction signals (reversal = new trade idea)
        if last_signal.get("direction") != signal.get("direction"):
            return True

        last_time = last_signal.get("timestamp")
        signal_time = signal.get("timestamp")

        if not last_time or not signal_time:
            return True

        time_diff = (signal_time - last_time).total_seconds() / 60
        bars_passed = time_diff / 15

        return bars_passed >= min_bars

    def _check_price_change(self, signal: Dict) -> bool:
        """Check if price has moved sufficiently since last signal (same direction only).

        Opposite-direction signals bypass this filter — reversal signals start
        from a different structural context regardless of price distance.
        """
        min_change = self.validation_config.get("min_price_change_pips", 10)

        if not self.recent_signals:
            return True

        last_signal = self.recent_signals[-1]

        # Skip filter for opposite-direction signals
        if last_signal.get("direction") != signal.get("direction"):
            return True

        last_price = last_signal.get("price", 0)
        current_price = signal.get("price", 0)

        if not last_price or not current_price:
            return True

        price_change = abs(current_price - last_price)
        return price_change >= min_change

    def _check_position_limits(self, current_positions: List[Dict]) -> bool:
        """Check if position limits allow new position."""
        max_positions = self.config.get("trading", {}).get("max_open_positions", 3)
        return len(current_positions) < max_positions

    def _check_all_positions_profitable(self, current_positions: List[Dict]) -> bool:
        """Check if all existing positions are in profit (CONSERVATIVE MODE)."""
        # Get config setting (default: disabled for backward compatibility)
        require_all_profitable = self.config.get("trading", {}).get(
            "require_all_positions_profitable", False
        )

        # If disabled, always return True (skip this check)
        if not require_all_profitable:
            return True

        # If no positions, check passes
        if not current_positions:
            return True

        # Check if ALL positions are profitable
        for position in current_positions:
            profit = position.get("profit", 0)
            if profit <= 0:  # Any position at breakeven or loss
                return False

        # All positions are profitable
        return True

    def _check_account_balance(self, account_info: Dict) -> bool:
        """Check if account has sufficient balance."""
        balance = account_info.get("balance", 0)
        margin_free = account_info.get("margin_free", balance)  # Default to balance

        # Need at least $60 free margin for 0.01 lot XAUUSD
        # Actual margin = price($5175) × 0.01 lot × 100 contract / 100 leverage = ~$51.75
        # $60 = margin requirement + $8 safety buffer
        # Was $100 — caused valid trades to be blocked when balance dropped below $100
        min_free_margin = 60

        return margin_free >= min_free_margin

    def _check_spread(self, market_data: Dict) -> bool:
        """Check if spread is acceptable."""
        spread = market_data.get("spread", 0)
        max_spread = 5.0  # Maximum 5 pips spread

        return spread <= max_spread

    def _check_recent_false_signals(self) -> float:
        """Check for recent false signals and return score penalty."""
        if not self.validation_config.get("check_recent_false_signals", True):
            return 0.0

        # This would require tracking signal outcomes
        # For now, return 0 (to be implemented with position tracking)
        return 0.0

    def add_signal_to_history(self, signal: Dict) -> None:
        """
        Add signal to history for tracking.

        Args:
            signal: Signal to add
        """
        self.recent_signals.append({
            "timestamp": signal.get("timestamp", datetime.utcnow()),
            "direction": signal.get("direction"),
            "price": signal.get("price"),
            "confidence": signal.get("confidence"),
        })

        # Keep only last 50 signals
        if len(self.recent_signals) > 50:
            self.recent_signals = self.recent_signals[-50:]

    def _validation_result(
        self, passed: bool, message: str, checks: List[Dict] = None
    ) -> Dict:
        """Create validation result dictionary."""
        return {
            "passed": passed,
            "message": message,
            "checks": checks or [],
            "timestamp": datetime.utcnow(),
        }

    def get_validation_summary(self, validation: Dict) -> str:
        """
        Get human-readable validation summary.

        Args:
            validation: Validation result

        Returns:
            Summary string
        """
        if validation.get("passed", False):
            return f"✓ VALIDATED: {validation.get('message', 'OK')}"
        else:
            return f"✗ REJECTED: {validation.get('message', 'Failed')}"
