"""
Session Manager
Manages trading sessions and applies session-based adjustments.
"""

from typing import Dict, Optional
from datetime import datetime, time
import pytz
from .session_detector import SessionDetector
from .dst_utils import get_session_times_utc
from ..bot_logger import get_logger


class SessionManager:
    """Manage trading sessions and session-based strategy adjustments."""

    def __init__(self, config: Dict):
        """
        Initialize session manager.

        Args:
            config: Session configuration
        """
        self.logger = get_logger()
        self.config = config
        self.detector = SessionDetector(config)

        # Current session cache
        self._current_session = None
        self._last_check_time = None

    def update(self, current_time: datetime = None) -> None:
        """
        Update session information.

        Args:
            current_time: Optional datetime (uses UTC now if not provided)
        """
        if current_time is None:
            current_time = datetime.now(pytz.UTC)

        # Update current session
        self._current_session = self.detector.get_current_session(current_time)
        self._last_check_time = current_time

        # Log session changes
        if self._current_session:
            self.logger.debug(
                f"Current session: {self._current_session['name']} "
                f"(weight: {self._current_session['weight']:.2f})"
            )

    def is_trading_allowed(self, current_time: datetime = None) -> Dict:
        """
        Check if trading is allowed.

        Args:
            current_time: Optional datetime

        Returns:
            Dictionary with allowed status and details
        """
        return self.detector.is_trading_allowed(current_time)

    def get_current_session(self, force_update: bool = False) -> Optional[Dict]:
        """
        Get current session.

        Args:
            force_update: Force refresh of session data

        Returns:
            Current session info or None
        """
        if force_update or self._current_session is None:
            self.update()

        return self._current_session

    def apply_session_adjustments(
        self,
        base_params: Dict,
        current_time: datetime = None
    ) -> Dict:
        """
        Apply session-based adjustments to strategy parameters.

        Args:
            base_params: Base strategy parameters
            current_time: Optional datetime

        Returns:
            Adjusted parameters
        """
        session = self.detector.get_current_session(current_time)
        if not session:
            return base_params.copy()

        adjusted = base_params.copy()

        # Get session-specific adjustments
        session_adjustments = self.detector.get_session_adjustments(current_time)

        # Apply confluence adjustment
        confluence_adj = session.get("confluence_adjustment", 0.0)
        if "min_confluence_score" in adjusted:
            adjusted["min_confluence_score"] = max(
                0.50,  # Minimum allowed
                adjusted["min_confluence_score"] + confluence_adj
            )

        # Apply position size adjustment
        if session_adjustments.get("reduce_position_size"):
            size_multiplier = session_adjustments.get("size_multiplier", 0.7)
            if "position_size" in adjusted:
                adjusted["position_size"] *= size_multiplier

        # Apply trade frequency adjustment
        if session_adjustments.get("allow_more_trades"):
            if "max_trades_per_hour" in adjusted:
                adjusted["max_trades_per_hour"] = int(
                    adjusted["max_trades_per_hour"] * 1.5
                )

        # Apply aggressiveness adjustment
        if session_adjustments.get("avoid_aggressive_entries"):
            adjusted["aggressive_entries_allowed"] = False
        elif session_adjustments.get("aggressive_entries_allowed"):
            adjusted["aggressive_entries_allowed"] = True

        # Apply session weight
        weight = session.get("weight", 1.0)
        adjusted["session_weight"] = weight

        return adjusted

    def get_adjusted_confluence_threshold(
        self,
        base_threshold: float,
        current_time: datetime = None
    ) -> float:
        """
        Get adjusted confluence threshold for current session.

        Args:
            base_threshold: Base confluence threshold
            current_time: Optional datetime

        Returns:
            Adjusted threshold
        """
        adjustment = self.detector.get_confluence_adjustment(current_time)
        adjusted = base_threshold + adjustment

        # Ensure reasonable bounds
        return max(0.50, min(0.85, adjusted))

    def get_position_size_multiplier(self, current_time: datetime = None) -> float:
        """
        Get position size multiplier for current session.

        Args:
            current_time: Optional datetime

        Returns:
            Size multiplier (0.5 - 1.0)
        """
        adjustments = self.detector.get_session_adjustments(current_time)

        if adjustments.get("reduce_position_size"):
            return adjustments.get("size_multiplier", 0.7)

        if adjustments.get("increase_position_size"):
            return 1.0  # Keep at 1.0 for conservative approach

        return 1.0

    def should_close_positions_early(self, current_time: datetime = None) -> Dict:
        """
        Check if positions should be closed early (e.g., Friday close).

        Args:
            current_time: Optional datetime

        Returns:
            Dictionary with close recommendation
        """
        if current_time is None:
            current_time = datetime.now(pytz.UTC)

        # Check Friday early close
        if self.detector.friday_close_early and current_time.weekday() == 4:
            friday_close = self.detector.friday_close_time
            current_time_only = current_time.time()

            # Within 30 minutes of close time
            close_minutes = friday_close.hour * 60 + friday_close.minute
            current_minutes = current_time_only.hour * 60 + current_time_only.minute

            minutes_until_close = close_minutes - current_minutes

            if 0 <= minutes_until_close <= 30:
                return {
                    "should_close": True,
                    "reason": "Approaching Friday close time",
                    "minutes_until_close": minutes_until_close,
                }

        return {
            "should_close": False,
            "reason": "No early close needed",
        }

    def get_ny_close_utc(self, dt: datetime = None) -> str:
        """Return DST-aware NY close as 'HH:MM' string."""
        ny_close = self.detector._ny_close_time
        if ny_close is None:
            times = get_session_times_utc(dt)
            ny_close = times["new_york_end"]
        return ny_close.strftime("%H:%M")

    def get_pre_close_hour(self, dt: datetime = None) -> int:
        """Return DST-aware pre-close profit lock hour (int)."""
        pre_close = self.detector._pre_close_hour
        if pre_close is None:
            times = get_session_times_utc(dt)
            pre_close = times["pre_close_hour"]
        return pre_close

    def should_exit_ny_close(self, current_time: datetime = None) -> Dict:
        """
        Check if positions should be evaluated for NY Close exit.

        Uses DST-aware NY close time from session detector.

        Args:
            current_time: Optional datetime (uses UTC now if not provided)

        Returns:
            Dictionary with exit recommendation
        """
        if current_time is None:
            current_time = datetime.now(pytz.UTC)

        ny_close_config = self.config.get("ny_close_exit", {})
        if not ny_close_config.get("enabled", False):
            return {"should_evaluate": False, "reason": "NY close exit disabled"}

        # Use DST-aware NY close time
        ny_close = self.detector._ny_close_time
        if ny_close is None:
            ny_close = time(22, 0)
        ny_close_hour = ny_close.hour
        ny_close_minute = ny_close.minute
        ny_close_str = ny_close.strftime("%H:%M")

        buffer_minutes = ny_close_config.get("buffer_minutes", 30)

        current_time_only = current_time.time()
        current_minutes = current_time_only.hour * 60 + current_time_only.minute
        close_minutes = ny_close_hour * 60 + ny_close_minute

        minutes_until_close = close_minutes - current_minutes
        if minutes_until_close < 0:
            minutes_until_close += 24 * 60

        if 0 <= minutes_until_close <= buffer_minutes:
            return {
                "should_evaluate": True,
                "reason": f"{minutes_until_close}min until NY Close",
                "minutes_until_close": minutes_until_close,
                "ny_close_utc": ny_close_str,
            }

        return {
            "should_evaluate": False,
            "reason": f"Not in NY close window ({minutes_until_close}min away)",
            "minutes_until_close": minutes_until_close,
        }

    def get_session_statistics(self) -> Dict:
        """
        Get statistics about session usage.

        Returns:
            Statistics dictionary
        """
        current = self.get_current_session()

        stats = {
            "current_session": current.get("name") if current else None,
            "current_weight": current.get("weight") if current else 0.0,
            "is_preferred": self.detector.is_preferred_session(),
            "trading_allowed": self.is_trading_allowed().get("allowed"),
        }

        return stats

    def get_optimal_entry_times(self) -> Dict:
        """
        Get information about optimal entry times.

        Returns:
            Dictionary with timing recommendations
        """
        preferences = self.config.get("preferences", {})
        preferred_sessions = preferences.get("preferred_sessions", [])

        optimal_times = {}

        for session_key in preferred_sessions:
            session_info = self.detector.get_session_info(session_key)
            if session_info:
                optimal_times[session_key] = {
                    "name": session_info["name"],
                    "start_utc": str(session_info["start"]),
                    "end_utc": str(session_info["end"]),
                    "weight": session_info["weight"],
                    "description": session_info["description"],
                }

        return optimal_times

    def validate_trade_timing(
        self,
        signal: Dict,
        current_time: datetime = None
    ) -> Dict:
        """
        Validate if trade timing is appropriate.

        Args:
            signal: Trading signal
            current_time: Optional datetime

        Returns:
            Validation result
        """
        # Check if trading allowed
        trading_check = self.is_trading_allowed(current_time)
        if not trading_check["allowed"]:
            return {
                "valid": False,
                "reason": trading_check["reason"],
            }

        # Check if in preferred session
        is_preferred = self.detector.is_preferred_session(current_time)
        session = self.detector.get_current_session(current_time)

        if not is_preferred:
            # Lower confidence requirement for non-preferred sessions
            confidence = signal.get("confidence", 0)
            min_confidence = 0.75  # Higher requirement

            if confidence < min_confidence:
                return {
                    "valid": False,
                    "reason": f"Non-preferred session requires higher confidence (>= {min_confidence})",
                    "session": session.get("name") if session else "Unknown",
                }

        # Check if approaching session end
        if session and current_time:
            end_time = session["end"]
            current_time_only = current_time.time()

            # Calculate minutes until session end
            end_minutes = end_time.hour * 60 + end_time.minute
            current_minutes = current_time_only.hour * 60 + current_time_only.minute

            minutes_until_end = end_minutes - current_minutes
            if minutes_until_end < 0:
                minutes_until_end += 24 * 60

            # Don't enter new trades within 15 minutes of session end
            if minutes_until_end < 15:
                return {
                    "valid": False,
                    "reason": f"Too close to session end ({minutes_until_end} min remaining)",
                }

        return {
            "valid": True,
            "reason": "Trade timing validated",
            "session": session.get("name") if session else "Unknown",
            "is_preferred_session": is_preferred,
        }

    def get_manager_summary(self, current_time: datetime = None) -> str:
        """
        Get human-readable manager summary.

        Args:
            current_time: Optional datetime

        Returns:
            Summary string
        """
        if current_time is None:
            current_time = datetime.now(pytz.UTC)

        trading_check = self.is_trading_allowed(current_time)
        session = self.detector.get_current_session(current_time)

        lines = [
            "Session Management:",
            f"  Time (UTC): {current_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Trading Allowed: {trading_check['allowed']}",
        ]

        if session:
            lines.append(f"  Current Session: {session['name']}")
            lines.append(f"  Session Weight: {session['weight']:.2f}")
            lines.append(f"  Confluence Adj: {session['confluence_adjustment']:+.2f}")
            lines.append(f"  Preferred: {self.detector.is_preferred_session(current_time)}")

            # Show adjustments
            adjustments = self.detector.get_session_adjustments(current_time)
            if adjustments:
                adj_items = []
                if adjustments.get("reduce_position_size"):
                    adj_items.append(f"Size: {adjustments.get('size_multiplier', 0.7):.0%}")
                if adjustments.get("avoid_aggressive_entries"):
                    adj_items.append("Avoid Aggressive")
                if adj_items:
                    lines.append(f"  Adjustments: {', '.join(adj_items)}")
        else:
            lines.append("  No active session")
            next_session = self.detector.get_next_session(current_time)
            if next_session:
                minutes = next_session.get("minutes_until", 0)
                hours = minutes // 60
                mins = minutes % 60
                lines.append(f"  Next: {next_session['name']} in {hours}h {mins}m")

        # Early close check
        early_close = self.should_close_positions_early(current_time)
        if early_close["should_close"]:
            lines.append(f"  WARNING: {early_close['reason']}")

        return "\n".join(lines)
