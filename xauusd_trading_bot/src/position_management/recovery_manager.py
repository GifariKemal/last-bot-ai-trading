"""
Recovery Manager
Manages recovery strategies for losing positions.
NO martingale, NO aggressive averaging - only intelligent recovery.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from ..bot_logger import get_logger


class RecoveryManager:
    """Manage recovery strategies for losing positions."""

    def __init__(self, config: Dict, position_tracker):
        """
        Initialize recovery manager.

        Args:
            config: Configuration dictionary
            position_tracker: PositionTracker instance
        """
        self.logger = get_logger()
        self.config = config
        self.tracker = position_tracker

        # Recovery settings
        self.recovery_enabled = True
        self.min_loss_for_recovery = config.get("risk", {}).get("micro_account", {}).get("max_risk_dollars", 2.0)
        self.max_recovery_time_hours = 48  # Give up after 48 hours

    def analyze_position_recovery(self, ticket: int) -> Optional[Dict]:
        """
        Analyze recovery options for a losing position.

        Args:
            ticket: Position ticket ID

        Returns:
            Recovery analysis or None
        """
        position = self.tracker.get_position(ticket)
        if not position:
            return None

        profit = position.get("profit", 0)

        # Only analyze losing positions
        if profit >= 0:
            return {
                "needs_recovery": False,
                "reason": "Position in profit",
            }

        # Check if loss is significant enough
        if abs(profit) < self.min_loss_for_recovery:
            return {
                "needs_recovery": False,
                "reason": f"Loss too small for recovery (${abs(profit):.2f})",
            }

        stats = self.tracker.get_position_stats(ticket)
        if not stats:
            return None

        position_type = position.get("type", "").upper()
        entry_price = position.get("open_price", 0)
        current_price = position.get("current_price", entry_price)
        sl = position.get("sl", 0)
        tp = position.get("tp", 0)

        # Analyze market movement
        recovery_options = []

        # Option 1: Move to breakeven if price improves
        if self._is_price_improving(position_type, entry_price, current_price):
            improvement = self._calculate_improvement(
                position_type, entry_price, current_price
            )
            if improvement > 5:  # At least 5 pips improvement
                recovery_options.append({
                    "type": "move_to_breakeven",
                    "priority": 1,
                    "description": "Price improving, move SL to breakeven",
                    "improvement_pips": improvement,
                    "new_sl": entry_price,
                })

        # Option 2: Exit on opposite structure break
        # (Would need SMC analysis, placeholder for now)
        recovery_options.append({
            "type": "structure_exit",
            "priority": 2,
            "description": "Exit if opposite structure breaks",
            "note": "Requires SMC analysis",
        })

        # Option 3: Time-based exit
        time_in_position = position.get("time_in_position", 0)
        max_time = self.max_recovery_time_hours * 3600

        if time_in_position > max_time:
            recovery_options.append({
                "type": "time_exit",
                "priority": 1,
                "description": f"Exit after {self.max_recovery_time_hours}h with no recovery",
                "time_in_position_hours": time_in_position / 3600,
            })

        # Option 4: Scale out (partial close)
        if abs(profit) > 100:  # Large loss
            recovery_options.append({
                "type": "partial_close",
                "priority": 3,
                "description": "Consider closing 50% to reduce exposure",
                "close_percent": 50,
            })

        # Option 5: Wait for retracement
        # Check if position is deeply in loss but has potential
        pips_to_entry = abs(stats["pips_from_entry"])
        if pips_to_entry > 50:  # More than 50 pips against
            recovery_options.append({
                "type": "wait_retracement",
                "priority": 4,
                "description": "Deep loss, wait for retracement to reduce loss",
                "current_loss_pips": pips_to_entry,
            })

        # Sort by priority
        recovery_options.sort(key=lambda x: x.get("priority", 99))

        return {
            "needs_recovery": True,
            "ticket": ticket,
            "current_loss": profit,
            "recovery_options": recovery_options,
            "recommended_action": recovery_options[0] if recovery_options else None,
            "position_stats": stats,
        }

    def _is_price_improving(
        self,
        position_type: str,
        entry_price: float,
        current_price: float
    ) -> bool:
        """Check if price is moving favorably."""
        if position_type == "BUY":
            return current_price > entry_price
        else:  # SELL
            return current_price < entry_price

    def _calculate_improvement(
        self,
        position_type: str,
        entry_price: float,
        current_price: float
    ) -> float:
        """Calculate price improvement in pips."""
        if position_type == "BUY":
            return max(0, current_price - entry_price)
        else:  # SELL
            return max(0, entry_price - current_price)

    def get_recovery_recommendations(self) -> List[Dict]:
        """
        Get recovery recommendations for all losing positions.

        Returns:
            List of recovery recommendations
        """
        losing_positions = self.tracker.get_losing_positions()
        recommendations = []

        for position in losing_positions:
            ticket = position.get("ticket")
            analysis = self.analyze_position_recovery(ticket)

            if analysis and analysis.get("needs_recovery"):
                recommendations.append(analysis)

        # Sort by loss size (largest losses first)
        recommendations.sort(
            key=lambda x: abs(x.get("current_loss", 0)),
            reverse=True
        )

        return recommendations

    def execute_recovery_action(
        self,
        ticket: int,
        action_type: str,
        parameters: Dict = None
    ) -> Dict:
        """
        Execute a recovery action.

        Args:
            ticket: Position ticket ID
            action_type: Type of action to take
            parameters: Action parameters

        Returns:
            Action result
        """
        position = self.tracker.get_position(ticket)
        if not position:
            return {
                "success": False,
                "reason": "Position not found",
            }

        parameters = parameters or {}

        if action_type == "move_to_breakeven":
            return self._action_move_to_breakeven(ticket, parameters)

        elif action_type == "time_exit":
            return self._action_time_exit(ticket, parameters)

        elif action_type == "partial_close":
            return self._action_partial_close(ticket, parameters)

        elif action_type == "structure_exit":
            return self._action_structure_exit(ticket, parameters)

        elif action_type == "wait_retracement":
            return self._action_wait_retracement(ticket, parameters)

        else:
            return {
                "success": False,
                "reason": f"Unknown action type: {action_type}",
            }

    def _action_move_to_breakeven(self, ticket: int, params: Dict) -> Dict:
        """Move stop loss to breakeven."""
        position = self.tracker.get_position(ticket)
        entry_price = position.get("open_price", 0)
        position_type = position.get("type", "").upper()

        # Add small buffer
        buffer = params.get("buffer_pips", 2.0)

        if position_type == "BUY":
            new_sl = entry_price + buffer
        else:
            new_sl = entry_price - buffer

        self.logger.info(
            f"Recovery: Move #{ticket} SL to breakeven @ {new_sl:.2f}"
        )

        return {
            "success": True,
            "action": "move_to_breakeven",
            "ticket": ticket,
            "new_sl": new_sl,
            "requires_execution": True,  # Needs to be sent to MT5
        }

    def _action_time_exit(self, ticket: int, params: Dict) -> Dict:
        """Exit position due to time limit."""
        self.logger.info(
            f"Recovery: Time exit for #{ticket} (exceeded max recovery time)"
        )

        return {
            "success": True,
            "action": "time_exit",
            "ticket": ticket,
            "reason": "Exceeded maximum recovery time",
            "requires_execution": True,
        }

    def _action_partial_close(self, ticket: int, params: Dict) -> Dict:
        """Close partial position to reduce exposure."""
        close_percent = params.get("close_percent", 50)

        self.logger.info(
            f"Recovery: Partial close {close_percent}% of #{ticket}"
        )

        return {
            "success": True,
            "action": "partial_close",
            "ticket": ticket,
            "close_percent": close_percent,
            "requires_execution": True,
            "note": "Not yet implemented in execution layer",
        }

    def _action_structure_exit(self, ticket: int, params: Dict) -> Dict:
        """Exit on structure break."""
        self.logger.info(
            f"Recovery: Structure exit suggested for #{ticket}"
        )

        return {
            "success": True,
            "action": "structure_exit",
            "ticket": ticket,
            "reason": "Opposite structure break detected",
            "requires_execution": True,
        }

    def _action_wait_retracement(self, ticket: int, params: Dict) -> Dict:
        """Wait for price retracement."""
        self.logger.info(
            f"Recovery: Waiting for retracement on #{ticket}"
        )

        return {
            "success": True,
            "action": "wait_retracement",
            "ticket": ticket,
            "reason": "Monitoring for favorable retracement",
            "requires_execution": False,  # Just monitoring
        }

    def get_recovery_summary(self) -> str:
        """
        Get human-readable recovery summary.

        Returns:
            Summary string
        """
        losing_positions = self.tracker.get_losing_positions()

        if not losing_positions:
            return "No positions need recovery"

        recommendations = self.get_recovery_recommendations()

        lines = [
            f"Positions Needing Recovery: {len(recommendations)}",
        ]

        for rec in recommendations[:3]:  # Show top 3
            ticket = rec.get("ticket")
            loss = rec.get("current_loss", 0)
            action = rec.get("recommended_action", {})

            if action:
                lines.append(
                    f"  #{ticket}: Loss ${abs(loss):.2f} -> "
                    f"{action.get('description', 'N/A')}"
                )

        return "\n".join(lines)

    def should_prevent_new_trades(self) -> Dict:
        """
        Check if new trades should be prevented due to recovery needs.

        Returns:
            Prevention recommendation
        """
        losing_positions = self.tracker.get_losing_positions()

        # Prevent new trades if multiple large losses
        large_losses = [
            p for p in losing_positions
            if abs(p.get("profit", 0)) > 100
        ]

        if len(large_losses) >= 2:
            total_loss = sum(p.get("profit", 0) for p in large_losses)
            return {
                "prevent": True,
                "reason": f"{len(large_losses)} large losing positions (${total_loss:.2f})",
                "recommendation": "Focus on recovering existing positions",
            }

        return {
            "prevent": False,
            "reason": "Recovery situation acceptable",
        }

    def get_position_recovery_status(self, ticket: int) -> str:
        """
        Get recovery status for a position.

        Args:
            ticket: Position ticket ID

        Returns:
            Status string
        """
        analysis = self.analyze_position_recovery(ticket)

        if not analysis:
            return "Position not found"

        if not analysis.get("needs_recovery"):
            return analysis.get("reason", "No recovery needed")

        action = analysis.get("recommended_action")
        if action:
            return f"Recovery: {action.get('description', 'Unknown action')}"

        return "Under recovery analysis"
