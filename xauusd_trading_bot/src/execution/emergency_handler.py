"""
Emergency Handler
Handles emergency situations and provides emergency stop functionality.
"""

from typing import Dict, List
from datetime import datetime
from ..bot_logger import get_logger


class EmergencyHandler:
    """Handle emergency situations."""

    def __init__(self, mt5_connector, order_executor):
        """
        Initialize emergency handler.

        Args:
            mt5_connector: MT5Connector instance
            order_executor: OrderExecutor instance
        """
        self.logger = get_logger()
        self.mt5 = mt5_connector
        self.executor = order_executor

        # Emergency state
        self.emergency_active = False
        self.emergency_reason = None
        self.emergency_timestamp = None

    def emergency_stop(self, reason: str = "Emergency stop triggered") -> Dict:
        """
        Execute emergency stop - close all positions immediately.

        Args:
            reason: Reason for emergency stop

        Returns:
            Emergency stop result
        """
        try:
            self.logger.critical(f"EMERGENCY STOP: {reason}")

            # Set emergency state
            self.emergency_active = True
            self.emergency_reason = reason
            self.emergency_timestamp = datetime.now()

            # Close all positions
            result = self.executor.close_all_positions(reason=f"EMERGENCY: {reason}")

            if result.get("success"):
                self.logger.info(
                    f"Emergency stop completed: {result.get('closed_count')} positions closed"
                )
            else:
                self.logger.error(
                    f"Emergency stop failed: {result.get('error')}"
                )

            return {
                "success": result.get("success", False),
                "reason": reason,
                "timestamp": self.emergency_timestamp,
                "positions_closed": result.get("closed_count", 0),
                "details": result,
            }

        except Exception as e:
            self.logger.critical(f"Error during emergency stop: {e}")
            return {
                "success": False,
                "error": str(e),
                "reason": reason,
            }

    def check_emergency_conditions(
        self,
        account_info: Dict,
        positions: List[Dict]
    ) -> Dict:
        """
        Check if emergency stop conditions are met.

        Args:
            account_info: Account information
            positions: Open positions

        Returns:
            Emergency check result
        """
        triggers = []

        # 1. Check account balance drop
        balance = account_info.get("balance", 0)
        equity = account_info.get("equity", 0)

        # If equity drops more than 20% below balance
        if balance > 0:
            equity_drop_percent = ((balance - equity) / balance) * 100
            if equity_drop_percent > 20:
                triggers.append({
                    "trigger": "equity_drop",
                    "severity": "critical",
                    "message": f"Equity dropped {equity_drop_percent:.1f}% below balance",
                    "values": {"balance": balance, "equity": equity},
                })

        # 2. Check margin level
        margin_level = account_info.get("margin_level", 0)
        if margin_level > 0 and margin_level < 150:
            triggers.append({
                "trigger": "low_margin",
                "severity": "critical" if margin_level < 100 else "warning",
                "message": f"Margin level critically low: {margin_level:.1f}%",
                "values": {"margin_level": margin_level},
            })

        # 3. Check total unrealized loss — percentage-based, not absolute dollar
        # Absolute $500 is meaningless: too strict for $100 account, too loose for $100k.
        # Use max_drawdown_percent from risk config (default 15%).
        _ap = self.executor.config.get("risk", {}).get("account_protection", {})
        max_loss_pct = _ap.get("max_drawdown_percent", 15.0)
        total_loss = sum(p.get("profit", 0) for p in positions if p.get("profit", 0) < 0)
        if balance > 0:
            loss_pct = (abs(total_loss) / balance) * 100
            if loss_pct > max_loss_pct:
                triggers.append({
                    "trigger": "large_loss",
                    "severity": "critical",
                    "message": f"Unrealized loss {loss_pct:.1f}% of balance (limit {max_loss_pct:.0f}%)",
                    "values": {"total_loss": total_loss, "loss_pct": loss_pct},
                })

        # 4. Check for stuck positions (future implementation)
        # Could check for positions open too long with large losses

        # Determine if emergency action needed
        critical_triggers = [t for t in triggers if t.get("severity") == "critical"]

        if critical_triggers:
            return {
                "emergency_needed": True,
                "triggers": triggers,
                "critical_count": len(critical_triggers),
                "recommended_action": "emergency_stop",
            }

        return {
            "emergency_needed": False,
            "triggers": triggers,
            "status": "normal",
        }

    def reset_emergency_state(self) -> None:
        """Reset emergency state (after manual intervention)."""
        if self.emergency_active:
            self.logger.info("Resetting emergency state")
            self.emergency_active = False
            self.emergency_reason = None
            self.emergency_timestamp = None

    def is_emergency_active(self) -> bool:
        """
        Check if emergency state is active.

        Returns:
            True if emergency active
        """
        return self.emergency_active

    def get_emergency_status(self) -> Dict:
        """
        Get current emergency status.

        Returns:
            Status dictionary
        """
        return {
            "emergency_active": self.emergency_active,
            "reason": self.emergency_reason,
            "timestamp": self.emergency_timestamp,
        }

    def safe_shutdown(self, reason: str = "Safe shutdown") -> Dict:
        """
        Perform safe shutdown - close positions gracefully.

        Args:
            reason: Reason for shutdown

        Returns:
            Shutdown result
        """
        try:
            self.logger.info(f"Safe shutdown initiated: {reason}")

            # Get all positions
            positions = self.mt5.get_positions()

            if not positions:
                self.logger.info("No positions to close during shutdown")
                return {
                    "success": True,
                    "message": "No positions to close",
                }

            # Close positions one by one with checks
            results = []
            for position in positions:
                ticket = position.get("ticket")

                # Log position details
                pos_type = position.get("type", "UNKNOWN")
                profit = position.get("profit", 0)
                self.logger.info(
                    f"Closing position #{ticket} ({pos_type}, P&L: ${profit:.2f})"
                )

                # Close position
                result = self.executor.execute_exit(ticket, reason)
                results.append(result)

            # Summary
            successful = [r for r in results if r.get("success")]
            failed = [r for r in results if not r.get("success")]

            self.logger.info(
                f"Shutdown complete: {len(successful)}/{len(positions)} closed"
            )

            return {
                "success": len(failed) == 0,
                "total_positions": len(positions),
                "closed_count": len(successful),
                "failed_count": len(failed),
                "results": results,
            }

        except Exception as e:
            self.logger.error(f"Error during safe shutdown: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def handle_connection_loss(self) -> Dict:
        """
        Handle MT5 connection loss.

        Returns:
            Handling result
        """
        try:
            self.logger.error("MT5 connection lost!")

            # Try to reconnect
            self.logger.info("Attempting to reconnect...")
            reconnect_result = self.mt5.connect()

            if reconnect_result:
                self.logger.info("Reconnected successfully")
                return {
                    "success": True,
                    "action": "reconnected",
                }

            # Reconnection failed
            self.logger.error("Reconnection failed")

            # If positions exist and can't reconnect, this is critical
            # User should be alerted
            return {
                "success": False,
                "action": "reconnection_failed",
                "recommendation": "Manual intervention required",
            }

        except Exception as e:
            self.logger.error(f"Error handling connection loss: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def validate_account_safety(self, account_info: Dict) -> Dict:
        """
        Validate that account is in safe state.

        Args:
            account_info: Account information

        Returns:
            Validation result
        """
        issues = []

        balance = account_info.get("balance", 0)
        equity = account_info.get("equity", 0)
        margin_level = account_info.get("margin_level", 0)
        margin_free = account_info.get("margin_free", 0)

        # Check balance
        if balance < 100:
            issues.append({
                "issue": "low_balance",
                "severity": "warning",
                "message": f"Balance very low: ${balance:.2f}",
            })

        # Check equity vs balance
        if balance > 0:
            equity_ratio = (equity / balance) * 100
            if equity_ratio < 80:
                issues.append({
                    "issue": "equity_drop",
                    "severity": "warning",
                    "message": f"Equity at {equity_ratio:.1f}% of balance",
                })

        # Check margin level
        if margin_level > 0:
            if margin_level < 200:
                issues.append({
                    "issue": "low_margin_level",
                    "severity": "critical" if margin_level < 150 else "warning",
                    "message": f"Margin level: {margin_level:.1f}%",
                })

        # Check free margin
        if margin_free < 50:
            issues.append({
                "issue": "low_free_margin",
                "severity": "warning",
                "message": f"Free margin: ${margin_free:.2f}",
            })

        # Determine safety status
        critical = [i for i in issues if i.get("severity") == "critical"]

        if critical:
            return {
                "safe": False,
                "status": "critical",
                "issues": issues,
                "recommendation": "Stop trading or close positions",
            }

        if issues:
            return {
                "safe": True,
                "status": "warning",
                "issues": issues,
                "recommendation": "Monitor closely",
            }

        return {
            "safe": True,
            "status": "ok",
            "issues": [],
        }

    def get_emergency_summary(self) -> str:
        """
        Get human-readable emergency summary.

        Returns:
            Summary string
        """
        if not self.emergency_active:
            return "Emergency Handler: Normal (no emergency active)"

        lines = [
            "EMERGENCY ACTIVE:",
            f"  Reason: {self.emergency_reason}",
            f"  Since: {self.emergency_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "  Status: Trading halted",
            "  Action: Manual reset required",
        ]

        return "\n".join(lines)
