"""
Micro Account Manager
Safety constraints for $50-$100 accounts.
Max $2 risk per trade, spread impact checks, position limits, loss protection.
"""

from typing import Dict, Optional
from ..bot_logger import get_logger


class MicroAccountManager:
    """Risk management for micro ($50-$100) trading accounts."""

    def __init__(self, config: Optional[Dict] = None):
        self.logger = get_logger()
        config = config or {}

        micro_cfg = config.get("micro_account", {})
        self.enabled = micro_cfg.get("enabled", True)
        self.max_balance_threshold = micro_cfg.get("max_balance_threshold", 500.0)
        self.max_risk_dollars = micro_cfg.get("max_risk_dollars", 2.0)
        self.max_risk_pct = micro_cfg.get("max_risk_percent", 2.0)
        self.max_positions = micro_cfg.get("max_positions", 2)
        self.max_spread_pct_of_sl = micro_cfg.get("max_spread_pct_of_sl", 10.0)
        self.lot_size = micro_cfg.get("fixed_lot", 0.01)

        # Consecutive loss protection
        loss_cfg = micro_cfg.get("loss_protection", {})
        self.reduce_after_losses = loss_cfg.get("reduce_after_losses", 2)
        self.risk_reduction_pct = loss_cfg.get("risk_reduction_percent", 25.0)
        self.pause_after_losses = loss_cfg.get("pause_after_losses", 3)

        # State tracking
        self.consecutive_losses = 0
        self.is_paused = False

    def is_micro_account(self, balance: float) -> bool:
        """Check if account qualifies as micro."""
        return balance <= self.max_balance_threshold

    def validate_trade(
        self,
        balance: float,
        sl_distance: float,
        spread: float,
        open_positions: int,
        consecutive_losses: int = 0,
    ) -> Dict:
        """
        Validate whether a trade is safe for micro account.

        Args:
            balance: Current account balance
            sl_distance: Stop loss distance in price units
            spread: Current spread in price units
            open_positions: Number of currently open positions
            consecutive_losses: Current consecutive loss count

        Returns:
            Dict with approved/rejected status and reasons
        """
        if not self.enabled or not self.is_micro_account(balance):
            return {"approved": True, "reason": "Not micro account", "adjustments": {}}

        reasons = []
        adjustments = {}

        # 1. Position limit check
        if open_positions >= self.max_positions:
            reasons.append(f"Max {self.max_positions} positions for micro account")
            return {"approved": False, "reasons": reasons, "adjustments": {}}

        # 2. Risk per trade check
        # At 0.01 lot, 1 pip = $0.01 for XAUUSD (100 oz/lot * 0.01 lot = 1 oz)
        # Actually: 0.01 lot = 0.01 * 100 = $1.00 per $1 move
        risk_dollars = sl_distance * self.lot_size * 100  # $1/pip at 0.01 lot
        risk_pct = (risk_dollars / balance * 100) if balance > 0 else 100

        if risk_dollars > self.max_risk_dollars:
            reasons.append(
                f"Risk ${risk_dollars:.2f} exceeds max ${self.max_risk_dollars:.2f} "
                f"(SL={sl_distance:.1f} pips)"
            )

        if risk_pct > self.max_risk_pct:
            reasons.append(
                f"Risk {risk_pct:.1f}% exceeds max {self.max_risk_pct:.1f}%"
            )

        # 3. Spread impact check
        if sl_distance > 0:
            spread_pct = (spread / sl_distance) * 100
            if spread_pct > self.max_spread_pct_of_sl:
                reasons.append(
                    f"Spread {spread_pct:.1f}% of SL exceeds max {self.max_spread_pct_of_sl}%"
                )

        # 4. Consecutive loss protection
        self.consecutive_losses = consecutive_losses
        if consecutive_losses >= self.pause_after_losses:
            reasons.append(
                f"{consecutive_losses} consecutive losses — trading paused"
            )
            self.is_paused = True

        if reasons:
            return {"approved": False, "reasons": reasons, "adjustments": {}}

        # 5. Risk reduction after losses
        if consecutive_losses >= self.reduce_after_losses:
            reduction = self.risk_reduction_pct / 100
            adjustments["risk_reduction"] = reduction
            adjustments["note"] = (
                f"Risk reduced {self.risk_reduction_pct}% after "
                f"{consecutive_losses} losses"
            )

        return {
            "approved": True,
            "reasons": [],
            "adjustments": adjustments,
            "risk_dollars": round(risk_dollars, 2),
            "risk_percent": round(risk_pct, 2),
        }

    def get_max_sl_distance(self, balance: float) -> float:
        """
        Calculate maximum allowable SL distance for micro account.
        At 0.01 lot: max_risk / (lot * 100) = max SL distance

        Args:
            balance: Current balance

        Returns:
            Maximum SL distance in price units
        """
        max_risk = min(
            self.max_risk_dollars,
            balance * self.max_risk_pct / 100,
        )
        # 0.01 lot * 100 oz/lot = 1 oz; risk = SL_distance * 1
        max_sl = max_risk / (self.lot_size * 100)
        return max_sl

    def calculate_recovery_plan(
        self, balance: float, target_balance: float, win_rate: float = 0.55
    ) -> Dict:
        """
        Calculate how many trades needed to recover to target balance.

        Args:
            balance: Current balance
            target_balance: Target balance to recover to
            win_rate: Expected win rate (0-1)

        Returns:
            Recovery plan dictionary
        """
        if balance >= target_balance:
            return {"needed": False, "trades_to_recover": 0}

        deficit = target_balance - balance
        # Average profit per trade = (WR * avg_win) + ((1-WR) * avg_loss)
        # Assume avg_win = $1.50, avg_loss = -$1.00 at 0.01 lot, 15 pip SL, 22.5 pip TP
        avg_win = 1.50  # Approximate
        avg_loss = -1.00
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        if expectancy <= 0:
            return {
                "needed": True,
                "trades_to_recover": float("inf"),
                "note": "Negative expectancy — strategy needs improvement",
            }

        trades_needed = int(deficit / expectancy) + 1

        return {
            "needed": True,
            "deficit": round(deficit, 2),
            "expectancy_per_trade": round(expectancy, 2),
            "trades_to_recover": trades_needed,
            "win_rate_assumed": win_rate,
        }

    def reset_loss_counter(self):
        """Reset consecutive loss tracking (call on win or pause expiry)."""
        self.consecutive_losses = 0
        self.is_paused = False
