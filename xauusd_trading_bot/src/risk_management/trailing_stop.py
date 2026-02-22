"""
Trailing Stop Manager
Manages trailing stop logic for open positions.
"""

from typing import Dict, Optional
from datetime import datetime
from ..bot_logger import get_logger


class TrailingStopManager:
    """Manage trailing stops for positions."""

    def __init__(self, config: Dict):
        """
        Initialize trailing stop manager.

        Args:
            config: Risk configuration
        """
        self.logger = get_logger()
        self.config = config

        self.trailing_config = config.get("trailing_stop", {})
        self.enabled = self.trailing_config.get("enabled", True)
        self.activation_percent = self.trailing_config.get("activation_percent", 10.0)
        self.trail_distance_percent = self.trailing_config.get("trail_distance_percent", 5.0)
        self.use_atr_trail = self.trailing_config.get("use_atr_trail", False)
        self.atr_trail_multiplier = self.trailing_config.get("atr_trail_multiplier", 1.5)

        # Breakeven config
        self.breakeven_config = config.get("breakeven", {})
        self.move_to_be_percent = self.breakeven_config.get("move_to_be_percent", 15.0)
        self.be_buffer_pips = self.breakeven_config.get("be_buffer_pips", 2.0)

        # Track peak profit for each position
        self.peak_profits = {}  # position_id -> peak_profit_pips

    def check_trailing_update(
        self,
        position: Dict,
        current_price: float,
        atr: float = None,
    ) -> Optional[Dict]:
        """
        Check if trailing stop should be updated.

        Args:
            position: Position data
            current_price: Current market price
            atr: Optional ATR value for ATR-based trailing

        Returns:
            Dictionary with new SL if update needed, None otherwise
        """
        if not self.enabled:
            return None

        try:
            position_id = position.get("ticket", 0)
            direction = position.get("type", "").upper()
            entry_price = position.get("open_price", 0)
            current_sl = position.get("sl", 0)

            if not entry_price or not current_sl:
                return None

            # Calculate current profit
            if direction == "BUY":
                profit_pips = current_price - entry_price
                unrealized_pnl = profit_pips * position.get("volume", 0.01) * 100
            else:  # SELL
                profit_pips = entry_price - current_price
                unrealized_pnl = profit_pips * position.get("volume", 0.01) * 100

            profit_percent = (abs(profit_pips) / entry_price) * 100

            # Update peak profit
            if position_id not in self.peak_profits:
                self.peak_profits[position_id] = profit_pips
            else:
                self.peak_profits[position_id] = max(
                    self.peak_profits[position_id], profit_pips
                )

            peak_profit = self.peak_profits[position_id]

            # Check if should move to breakeven first
            if profit_percent >= self.move_to_be_percent:
                be_level = self._calculate_breakeven_level(position)
                if self._should_move_to_breakeven(current_sl, be_level, direction):
                    return {
                        "new_sl": be_level,
                        "reason": "move_to_breakeven",
                        "profit_percent": profit_percent,
                        "peak_profit": peak_profit,
                    }

            # Check if trailing should activate
            if profit_percent < self.activation_percent:
                return None

            # Calculate new trailing SL
            if self.use_atr_trail and atr:
                new_sl = self._calculate_atr_trailing_sl(
                    current_price, direction, atr
                )
            else:
                new_sl = self._calculate_percent_trailing_sl(
                    current_price, direction, peak_profit
                )

            # Check if should update
            if self._should_update_sl(current_sl, new_sl, direction):
                return {
                    "new_sl": round(new_sl, 2),
                    "reason": "trailing_stop",
                    "profit_percent": profit_percent,
                    "peak_profit": peak_profit,
                    "current_profit": profit_pips,
                }

            return None

        except Exception as e:
            self.logger.error(f"Error checking trailing stop: {e}")
            return None

    def _calculate_percent_trailing_sl(
        self,
        current_price: float,
        direction: str,
        peak_profit: float,
    ) -> float:
        """Calculate trailing SL based on percentage from peak."""

        trail_distance = current_price * (self.trail_distance_percent / 100)

        if direction == "BUY":
            # Trail below current price
            new_sl = current_price - trail_distance
        else:  # SELL
            # Trail above current price
            new_sl = current_price + trail_distance

        return new_sl

    def _calculate_atr_trailing_sl(
        self,
        current_price: float,
        direction: str,
        atr: float,
    ) -> float:
        """Calculate trailing SL based on ATR."""

        trail_distance = atr * self.atr_trail_multiplier

        if direction == "BUY":
            new_sl = current_price - trail_distance
        else:  # SELL
            new_sl = current_price + trail_distance

        return new_sl

    def _calculate_breakeven_level(self, position: Dict) -> float:
        """Calculate breakeven level with buffer."""

        entry_price = position.get("open_price", 0)
        direction = position.get("type", "").upper()

        if direction == "BUY":
            return entry_price + self.be_buffer_pips
        else:  # SELL
            return entry_price - self.be_buffer_pips

    def _should_move_to_breakeven(
        self,
        current_sl: float,
        be_level: float,
        direction: str,
    ) -> bool:
        """Check if should move SL to breakeven."""

        if direction == "BUY":
            # Only move up to BE
            return be_level > current_sl
        else:  # SELL
            # Only move down to BE
            return be_level < current_sl

    def _should_update_sl(
        self,
        current_sl: float,
        new_sl: float,
        direction: str,
    ) -> bool:
        """Check if SL should be updated."""

        if direction == "BUY":
            # Only move SL up, never down
            return new_sl > current_sl
        else:  # SELL
            # Only move SL down, never up
            return new_sl < current_sl

    def on_position_closed(self, position_id: int) -> None:
        """
        Clean up tracking data when position closes.

        Args:
            position_id: Position ticket ID
        """
        if position_id in self.peak_profits:
            del self.peak_profits[position_id]
            self.logger.debug(f"Cleaned up tracking for position {position_id}")

    def get_position_stats(self, position_id: int) -> Optional[Dict]:
        """
        Get statistics for a position.

        Args:
            position_id: Position ticket ID

        Returns:
            Stats dictionary or None
        """
        if position_id not in self.peak_profits:
            return None

        return {
            "peak_profit_pips": self.peak_profits[position_id],
            "tracking_since": "position_open",  # Could add timestamp tracking
        }

    def reset_all_tracking(self) -> None:
        """Reset all position tracking (e.g., on bot restart)."""
        self.peak_profits.clear()
        self.logger.info("Reset all trailing stop tracking")

    def get_trailing_status(self, position: Dict, current_price: float) -> Dict:
        """
        Get current trailing stop status for a position.

        Args:
            position: Position data
            current_price: Current market price

        Returns:
            Status dictionary
        """
        try:
            position_id = position.get("ticket", 0)
            direction = position.get("type", "").upper()
            entry_price = position.get("open_price", 0)

            # Calculate profit
            if direction == "BUY":
                profit_pips = current_price - entry_price
            else:
                profit_pips = entry_price - current_price

            profit_percent = (abs(profit_pips) / entry_price) * 100

            # Get peak profit
            peak_profit = self.peak_profits.get(position_id, profit_pips)

            # Determine status
            if profit_percent < self.activation_percent:
                status = "inactive"
                reason = f"Profit ({profit_percent:.1f}%) below activation ({self.activation_percent}%)"
            else:
                status = "active"
                reason = "Trailing stop active"

            return {
                "status": status,
                "reason": reason,
                "current_profit_pips": profit_pips,
                "current_profit_percent": profit_percent,
                "peak_profit_pips": peak_profit,
                "activation_threshold": self.activation_percent,
                "trail_distance_percent": self.trail_distance_percent,
            }

        except Exception as e:
            self.logger.error(f"Error getting trailing status: {e}")
            return {
                "status": "error",
                "reason": str(e),
            }
