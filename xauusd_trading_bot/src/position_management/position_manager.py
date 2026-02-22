"""
Position Manager
Manages position limits, correlation, and intelligent position decisions.
"""

from typing import Dict, List, Optional
from datetime import datetime
from ..bot_logger import get_logger


class PositionManager:
    """Manage positions with intelligent decision making."""

    def __init__(self, config: Dict, position_tracker):
        """
        Initialize position manager.

        Args:
            config: Configuration dictionary
            position_tracker: PositionTracker instance
        """
        self.logger = get_logger()
        self.config = config
        self.tracker = position_tracker

        # Limits from config
        limits_config = config.get("position_limits", {})
        self.max_open_positions = limits_config.get("max_open_positions", 3)
        self.max_positions_per_direction = limits_config.get(
            "max_positions_per_direction", 2
        )
        self.min_position_distance_pips = limits_config.get(
            "min_position_distance", 50.0
        )
        self.max_total_lots = limits_config.get("max_total_lots", 0.05)

        # Correlation settings
        correlation_config = config.get("correlation", {})
        self.max_correlated_positions = correlation_config.get(
            "max_correlated_positions", 2
        )
        self.correlation_check_enabled = correlation_config.get(
            "correlation_check_enabled", True
        )

    def can_open_position(
        self,
        direction: str,
        entry_price: float,
        lot_size: float,
        reason: str = None
    ) -> Dict:
        """
        Check if a new position can be opened.

        Args:
            direction: "BUY" or "SELL"
            entry_price: Proposed entry price
            lot_size: Proposed lot size
            reason: Optional reason for logging

        Returns:
            Dictionary with allowed status and reason
        """
        checks = []

        # 1. Check max open positions
        current_count = self.tracker.get_position_count()
        if current_count >= self.max_open_positions:
            checks.append({
                "check": "max_positions",
                "passed": False,
                "reason": f"Already at max positions ({self.max_open_positions})",
            })
        else:
            checks.append({
                "check": "max_positions",
                "passed": True,
                "current": current_count,
                "limit": self.max_open_positions,
            })

        # 2. Check max positions per direction
        same_direction = self.tracker.get_positions_by_type(direction)
        if len(same_direction) >= self.max_positions_per_direction:
            checks.append({
                "check": "max_per_direction",
                "passed": False,
                "reason": f"Already at max {direction} positions ({self.max_positions_per_direction})",
            })
        else:
            checks.append({
                "check": "max_per_direction",
                "passed": True,
                "current": len(same_direction),
                "limit": self.max_positions_per_direction,
            })

        # 3. Check total exposure
        exposure = self.tracker.get_total_exposure()
        total_lots = exposure["total_lots"]
        if total_lots + lot_size > self.max_total_lots:
            checks.append({
                "check": "total_exposure",
                "passed": False,
                "reason": f"Would exceed max exposure ({self.max_total_lots} lots)",
            })
        else:
            checks.append({
                "check": "total_exposure",
                "passed": True,
                "current_lots": total_lots,
                "new_lots": lot_size,
                "total_after": total_lots + lot_size,
                "limit": self.max_total_lots,
            })

        # 4. Check position spacing (avoid too close positions)
        spacing_check = self._check_position_spacing(
            direction, entry_price, same_direction
        )
        checks.append(spacing_check)

        # 5. Check correlation (if enabled)
        if self.correlation_check_enabled:
            correlation_check = self._check_correlation(direction)
            checks.append(correlation_check)

        # Determine if allowed
        failed_checks = [c for c in checks if not c.get("passed", True)]

        if failed_checks:
            return {
                "allowed": False,
                "reason": "; ".join([c["reason"] for c in failed_checks if "reason" in c]),
                "checks": checks,
                "failed_checks": failed_checks,
            }

        return {
            "allowed": True,
            "reason": "All position checks passed",
            "checks": checks,
        }

    def _check_position_spacing(
        self,
        direction: str,
        entry_price: float,
        same_direction_positions: List[Dict]
    ) -> Dict:
        """
        Check if position is far enough from existing positions.

        Args:
            direction: Position direction
            entry_price: Proposed entry price
            same_direction_positions: Existing positions in same direction

        Returns:
            Check result
        """
        if not same_direction_positions:
            return {
                "check": "position_spacing",
                "passed": True,
                "reason": "No existing positions in same direction",
            }

        # Check distance to each existing position
        for position in same_direction_positions:
            existing_entry = position.get("open_price", 0)
            distance = abs(entry_price - existing_entry)

            if distance < self.min_position_distance_pips:
                return {
                    "check": "position_spacing",
                    "passed": False,
                    "reason": f"Too close to existing position (min {self.min_position_distance_pips} pips)",
                    "distance": distance,
                    "existing_ticket": position.get("ticket"),
                }

        return {
            "check": "position_spacing",
            "passed": True,
            "min_distance": self.min_position_distance_pips,
        }

    def _check_correlation(self, direction: str) -> Dict:
        """
        Check position correlation rules.

        For XAUUSD, all positions are same instrument, so just check
        if too many same-direction positions exist.

        Args:
            direction: Position direction

        Returns:
            Check result
        """
        same_direction = self.tracker.get_positions_by_type(direction)

        if len(same_direction) >= self.max_correlated_positions:
            return {
                "check": "correlation",
                "passed": False,
                "reason": f"Too many correlated {direction} positions ({self.max_correlated_positions})",
            }

        return {
            "check": "correlation",
            "passed": True,
            "correlated_positions": len(same_direction),
            "limit": self.max_correlated_positions,
        }

    def suggest_position_to_close(self, reason: str = "need_space") -> Optional[Dict]:
        """
        Suggest which position to close (if at limit and need to open new).

        Args:
            reason: Reason for closure suggestion

        Returns:
            Position to close, or None
        """
        positions = self.tracker.get_all_positions()
        if not positions:
            return None

        # Strategy: Close the worst-performing position
        losing_positions = self.tracker.get_losing_positions()

        if losing_positions:
            # Close the one with largest loss
            worst = min(losing_positions, key=lambda p: p.get("profit", 0))
            return {
                "ticket": worst.get("ticket"),
                "reason": f"Worst performing position ({reason})",
                "current_profit": worst.get("profit", 0),
            }

        # If no losing positions, close oldest breakeven position
        breakeven_positions = [
            p for p in positions
            if abs(p.get("profit", 0)) < 5.0  # Within $5 of breakeven
        ]

        if breakeven_positions:
            oldest = min(
                breakeven_positions,
                key=lambda p: p.get("tracked_since", datetime.now())
            )
            return {
                "ticket": oldest.get("ticket"),
                "reason": f"Oldest breakeven position ({reason})",
                "current_profit": oldest.get("profit", 0),
            }

        # Last resort: close oldest position
        oldest = self.tracker.get_oldest_position()
        if oldest:
            return {
                "ticket": oldest.get("ticket"),
                "reason": f"Oldest position ({reason})",
                "current_profit": oldest.get("profit", 0),
            }

        return None

    def get_position_priority_list(self) -> List[Dict]:
        """
        Get positions ordered by priority (for monitoring/action).

        Returns:
            List of positions with priority scores
        """
        positions = self.tracker.get_all_positions()
        if not positions:
            return []

        prioritized = []

        for position in positions:
            # Calculate priority score (higher = needs more attention)
            score = 0
            reasons = []

            profit = position.get("profit", 0)
            time_in_position = position.get("time_in_position", 0)

            # Losing positions get higher priority
            if profit < 0:
                score += abs(profit) / 10  # More loss = higher priority
                reasons.append("losing")

            # Positions open too long
            if time_in_position > 24 * 3600:  # > 24 hours
                score += 20
                reasons.append("stale")

            # Positions with large drawdown from peak
            max_drawdown = position.get("max_drawdown", 0)
            if max_drawdown > 20:  # > 20 pips drawdown
                score += max_drawdown
                reasons.append("large_drawdown")

            # Positions close to SL
            stats = self.tracker.get_position_stats(position.get("ticket"))
            if stats and stats.get("pips_to_sl"):
                if stats["pips_to_sl"] < 5:  # Within 5 pips of SL
                    score += 30
                    reasons.append("near_sl")

            prioritized.append({
                "ticket": position.get("ticket"),
                "priority_score": score,
                "reasons": reasons,
                "position": position,
            })

        # Sort by priority (highest first)
        prioritized.sort(key=lambda x: x["priority_score"], reverse=True)

        return prioritized

    def should_reduce_exposure(self) -> Dict:
        """
        Check if exposure should be reduced.

        Returns:
            Dictionary with recommendation
        """
        exposure = self.tracker.get_total_exposure()
        losing = self.tracker.get_losing_positions()

        # Check if too many losing positions
        if len(losing) >= 2:
            total_loss = sum(p.get("profit", 0) for p in losing)
            return {
                "should_reduce": True,
                "reason": f"{len(losing)} losing positions (total: ${total_loss:.2f})",
                "suggested_action": "close_worst_loser",
                "positions_to_close": 1,
            }

        # Check if approaching max exposure
        if exposure["total_lots"] >= self.max_total_lots * 0.9:
            return {
                "should_reduce": True,
                "reason": f"Near max exposure ({exposure['total_lots']:.2f}/{self.max_total_lots})",
                "suggested_action": "no_new_positions",
            }

        return {
            "should_reduce": False,
            "reason": "Exposure within acceptable limits",
        }

    def get_management_summary(self) -> str:
        """
        Get human-readable management summary.

        Returns:
            Summary string
        """
        exposure = self.tracker.get_total_exposure()

        lines = [
            "Position Management Status:",
            f"  Open: {exposure['total_positions']}/{self.max_open_positions}",
            f"  Exposure: {exposure['total_lots']:.2f}/{self.max_total_lots} lots",
            f"  Net Position: {exposure['net_direction']} ({exposure['net_exposure']:.2f} lots)",
            f"  Total P&L: ${exposure['total_profit']:.2f}",
        ]

        # Check if should reduce
        reduce_check = self.should_reduce_exposure()
        if reduce_check["should_reduce"]:
            lines.append(f"  WARNING: {reduce_check['reason']}")

        # Priority positions
        priority = self.get_position_priority_list()
        if priority and priority[0]["priority_score"] > 0:
            top = priority[0]
            lines.append(
                f"  High Priority: #{top['ticket']} (score: {top['priority_score']:.0f}, "
                f"reasons: {', '.join(top['reasons'])})"
            )

        return "\n".join(lines)

    def validate_new_signal(self, signal: Dict) -> Dict:
        """
        Validate if a new signal should be executed based on position management.

        Args:
            signal: Trading signal

        Returns:
            Validation result
        """
        direction = signal.get("direction")
        price = signal.get("price", 0)
        lot_size = signal.get("lot_size", 0.01)

        # Check if can open
        can_open = self.can_open_position(direction, price, lot_size)

        if not can_open["allowed"]:
            return {
                "validated": False,
                "reason": can_open["reason"],
                "checks": can_open.get("checks"),
            }

        # Check if should reduce exposure instead
        reduce_check = self.should_reduce_exposure()
        if reduce_check["should_reduce"]:
            return {
                "validated": False,
                "reason": f"Should reduce exposure: {reduce_check['reason']}",
                "suggestion": reduce_check.get("suggested_action"),
            }

        return {
            "validated": True,
            "reason": "Position management checks passed",
        }
