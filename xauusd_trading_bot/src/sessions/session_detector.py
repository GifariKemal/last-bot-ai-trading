"""
Session Detector
Detects current trading session based on UTC time.
"""

from typing import Dict, List, Optional
from datetime import datetime, time
import pytz
from ..bot_logger import get_logger


class SessionDetector:
    """Detect current trading session."""

    def __init__(self, config: Dict):
        """
        Initialize session detector.

        Args:
            config: Session configuration
        """
        self.logger = get_logger()
        self.config = config
        self.sessions_config = config.get("sessions", {})

        # Parse session times
        self.sessions = {}
        for session_key, session_data in self.sessions_config.items():
            self.sessions[session_key] = {
                "name": session_data.get("name"),
                "start": self._parse_time(session_data.get("start_utc")),
                "end": self._parse_time(session_data.get("end_utc")),
                "weight": session_data.get("weight", 1.0),
                "confluence_adjustment": session_data.get("min_confluence_adjustment", 0.0),
                "description": session_data.get("description", ""),
            }

        # Restrictions
        restrictions = config.get("restrictions", {})
        self.trading_days = restrictions.get("trading_days", [0, 1, 2, 3, 4])
        self.avoid_weekends = restrictions.get("avoid_weekends", True)
        self.friday_close_early = restrictions.get("friday_close_early", True)
        self.friday_close_time = self._parse_time(
            restrictions.get("friday_close_time_utc", "21:00")
        )

        # Parse blackout hours (e.g. ["21:00-22:00"] for daily maintenance)
        self.blackout_hours = []
        for period_str in restrictions.get("blackout_hours", []):
            try:
                start_s, end_s = period_str.split("-", 1)
                self.blackout_hours.append(
                    (self._parse_time(start_s.strip()), self._parse_time(end_s.strip()))
                )
            except Exception:
                pass

    def _parse_time(self, time_str: str) -> time:
        """
        Parse time string to time object.

        Args:
            time_str: Time string in HH:MM format

        Returns:
            time object
        """
        try:
            hour, minute = map(int, time_str.split(":"))
            return time(hour, minute)
        except Exception as e:
            self.logger.error(f"Error parsing time '{time_str}': {e}")
            return time(0, 0)

    def _is_closed_period(self, current_time: datetime) -> bool:
        """Return True if market is closed (weekend/maintenance/friday close)."""
        day_of_week = current_time.weekday()
        current_t = current_time.time()

        if day_of_week == 5:  # Saturday — all day
            return True
        if day_of_week == 6 and current_t < time(23, 0):  # Sunday before 23:00 UTC
            return True
        if self.friday_close_early and day_of_week == 4 and current_t >= self.friday_close_time:
            return True
        for (start_b, end_b) in self.blackout_hours:
            if start_b <= current_t < end_b:
                return True
        return False

    def get_current_session(self, current_time: datetime = None) -> Optional[Dict]:
        """
        Get current trading session.

        Returns None during weekend, maintenance, or Friday close — even if the
        clock time would normally match a session window.

        Args:
            current_time: Optional datetime (uses UTC now if not provided)

        Returns:
            Session info dictionary or None
        """
        if current_time is None:
            current_time = datetime.now(pytz.UTC)
        elif current_time.tzinfo is None:
            # Assume UTC if no timezone
            current_time = pytz.UTC.localize(current_time)

        # No session during market-closed periods
        if self._is_closed_period(current_time):
            return None

        current_time_only = current_time.time()

        # Find all active sessions
        active_sessions = []

        for session_key, session_data in self.sessions.items():
            start = session_data["start"]
            end = session_data["end"]

            # Handle sessions that cross midnight
            if start <= end:
                # Normal session (e.g., 08:00 - 16:00)
                if start <= current_time_only < end:
                    active_sessions.append({
                        "key": session_key,
                        **session_data,
                    })
            else:
                # Session crosses midnight (e.g., 22:00 - 02:00)
                if current_time_only >= start or current_time_only < end:
                    active_sessions.append({
                        "key": session_key,
                        **session_data,
                    })

        # Return highest weighted session if multiple active
        if active_sessions:
            return max(active_sessions, key=lambda s: s["weight"])

        return None

    def get_all_active_sessions(self, current_time: datetime = None) -> List[Dict]:
        """
        Get all currently active sessions.

        Returns empty list during weekend/maintenance/Friday close.

        Args:
            current_time: Optional datetime

        Returns:
            List of active sessions
        """
        if current_time is None:
            current_time = datetime.now(pytz.UTC)
        elif current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)

        if self._is_closed_period(current_time):
            return []

        current_time_only = current_time.time()
        active_sessions = []

        for session_key, session_data in self.sessions.items():
            start = session_data["start"]
            end = session_data["end"]

            if start <= end:
                if start <= current_time_only < end:
                    active_sessions.append({
                        "key": session_key,
                        **session_data,
                    })
            else:
                if current_time_only >= start or current_time_only < end:
                    active_sessions.append({
                        "key": session_key,
                        **session_data,
                    })

        return active_sessions

    def is_trading_allowed(self, current_time: datetime = None) -> Dict:
        """
        Check if trading is allowed at current time.

        Exness XAUUSDm actual market hours (verified from M15 candle gaps):
          - Daily maintenance: 22:00–23:00 UTC (last candle 21:45, resumes 23:00)
          - Weekly close:  Friday ~22:00 UTC  (last candle = Fri 21:45)
          - Weekly reopen: Sunday 23:00 UTC
          - Saturday + most of Sunday: fully closed

        Args:
            current_time: Optional datetime

        Returns:
            Dictionary with allowed/reason/status/opens_in_minutes
        """
        if current_time is None:
            current_time = datetime.now(pytz.UTC)
        elif current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)

        day_of_week = current_time.weekday()   # 0=Mon … 5=Sat, 6=Sun
        current_t = current_time.time()
        current_minutes = current_t.hour * 60 + current_t.minute

        # ── Saturday: fully closed all day ───────────────────────────
        if day_of_week == 5:
            # Sat remaining + until Sun 23:00
            sat_remaining = (24 * 60) - current_minutes
            opens_in = sat_remaining + (23 * 60)   # + Sunday until 23:00
            h, m = divmod(opens_in, 60)
            return {
                "allowed": False,
                "reason": f"Weekend — market opens Sunday 23:00 UTC (in {h}h {m:02d}m)",
                "status": "WEEKEND",
                "opens_in_minutes": opens_in,
            }

        # ── Sunday: closed until 23:00 UTC ───────────────────────────
        if day_of_week == 6:
            if current_minutes < 23 * 60:  # Before 23:00 UTC
                opens_in = (23 * 60) - current_minutes
                h, m = divmod(opens_in, 60)
                return {
                    "allowed": False,
                    "reason": f"Weekend — market opens Sunday 23:00 UTC (in {h}h {m:02d}m)",
                    "status": "WEEKEND",
                    "opens_in_minutes": opens_in,
                }
            # After 23:00 Sunday = market is open (fall through to session check)

        # ── Friday early close (check BEFORE maintenance) ────────────
        # Stops new entries before weekend close
        if self.friday_close_early and day_of_week == 4:
            if current_t >= self.friday_close_time:
                # Remaining Friday + Saturday + until Sun 23:00
                fri_remaining = (24 * 60) - current_minutes
                opens_in = fri_remaining + (24 * 60) + (23 * 60)
                h, m = divmod(opens_in, 60)
                return {
                    "allowed": False,
                    "reason": (
                        f"Friday close {self.friday_close_time.strftime('%H:%M')} UTC "
                        f"— market reopens Sunday 23:00 UTC (in {h}h {m:02d}m)"
                    ),
                    "status": "WEEKEND",
                    "opens_in_minutes": opens_in,
                }

        # ── Daily maintenance break 00:00–01:00 UTC ──────────────────
        for (start_b, end_b) in self.blackout_hours:
            if start_b <= current_t < end_b:
                end_min = end_b.hour * 60 + end_b.minute
                opens_in = end_min - current_minutes
                return {
                    "allowed": False,
                    "reason": (
                        f"Server maintenance "
                        f"{start_b.strftime('%H:%M')}–{end_b.strftime('%H:%M')} UTC "
                        f"(resumes in {opens_in}m)"
                    ),
                    "status": "MAINTENANCE",
                    "opens_in_minutes": opens_in,
                }

        # ── Active session check ──────────────────────────────────────
        current_session = self.get_current_session(current_time)
        if current_session:
            return {
                "allowed": True,
                "reason": f"In {current_session['name']}",
                "status": "OPEN",
                "session": current_session,
            }

        # ── Off-session (still a weekday, still market hours) ─────────
        return {
            "allowed": True,
            "reason": "Off-session (low weight)",
            "status": "OPEN",
            "session": {
                "key": "off_session",
                "name": "Off-Session",
                "weight": 0.5,
                "confluence_adjustment": 0.10,
                "description": "Outside main sessions",
            },
        }

    def get_session_info(self, session_key: str) -> Optional[Dict]:
        """
        Get information about a specific session.

        Args:
            session_key: Session identifier

        Returns:
            Session info or None
        """
        return self.sessions.get(session_key)

    def get_next_session(self, current_time: datetime = None) -> Optional[Dict]:
        """
        Get the next upcoming session.

        Handles weekend: during Saturday or Sunday before 23:00 UTC,
        returns a synthetic "Market Open" entry pointing to Sunday 23:00 UTC.

        Args:
            current_time: Optional datetime

        Returns:
            Next session info or None
        """
        if current_time is None:
            current_time = datetime.now(pytz.UTC)
        elif current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)

        day_of_week = current_time.weekday()
        current_t = current_time.time()
        current_minutes = current_t.hour * 60 + current_t.minute

        # During weekend closed period → next event is Sunday 23:00 UTC market open
        if day_of_week == 5:  # Saturday
            sat_remaining = (24 * 60) - current_minutes
            opens_in = sat_remaining + (23 * 60)   # rest of Sat + Sun until 23:00
            return {
                "key": "market_open",
                "name": "Market Open",
                "minutes_until": opens_in,
                "weight": 1.0,
                "confluence_adjustment": 0.0,
                "description": "Exness XAUUSDm reopens Sunday 23:00 UTC",
            }
        if day_of_week == 6 and current_minutes < 23 * 60:  # Sunday before 23:00
            opens_in = (23 * 60) - current_minutes
            return {
                "key": "market_open",
                "name": "Market Open",
                "minutes_until": opens_in,
                "weight": 1.0,
                "confluence_adjustment": 0.0,
                "description": "Exness XAUUSDm reopens Sunday 23:00 UTC",
            }

        # Friday close → market won't reopen until Sunday 23:00
        if self.friday_close_early and day_of_week == 4 and current_t >= self.friday_close_time:
            fri_remaining = (24 * 60) - current_minutes
            opens_in = fri_remaining + (24 * 60) + (23 * 60)  # Fri + Sat + Sun until 23:00
            return {
                "key": "market_open",
                "name": "Market Open",
                "minutes_until": opens_in,
                "weight": 1.0,
                "confluence_adjustment": 0.0,
                "description": "Exness XAUUSDm reopens Sunday 23:00 UTC",
            }

        # During maintenance → next available is end of maintenance window
        for (start_b, end_b) in self.blackout_hours:
            if start_b <= current_t < end_b:
                end_min = end_b.hour * 60 + end_b.minute
                opens_in = end_min - current_minutes
                return {
                    "key": "maintenance_end",
                    "name": "Trading Resumes",
                    "minutes_until": opens_in,
                    "weight": 1.0,
                    "confluence_adjustment": 0.0,
                    "description": f"Maintenance ends {end_b.strftime('%H:%M')} UTC",
                }

        # Normal case: find next session start (intraday)
        upcoming_sessions = []
        for session_key, session_data in self.sessions.items():
            start = session_data["start"]
            session_minutes = start.hour * 60 + start.minute
            minutes_until = session_minutes - current_minutes
            if minutes_until < 0:
                minutes_until += 24 * 60
            upcoming_sessions.append({
                "key": session_key,
                "minutes_until": minutes_until,
                **session_data,
            })

        if upcoming_sessions:
            return min(upcoming_sessions, key=lambda s: s["minutes_until"])

        return None

    def get_market_status(self, current_time: datetime = None) -> Dict:
        """
        Get current market status for display/notification purposes.

        Returns:
            Dict with keys: is_open, status ("OPEN"/"WEEKEND"/"MAINTENANCE"),
                            reason (str), opens_in_minutes (int or None)
        """
        check = self.is_trading_allowed(current_time)
        return {
            "is_open": check.get("allowed", False),
            "status": check.get("status", "OPEN" if check.get("allowed") else "CLOSED"),
            "reason": check.get("reason", ""),
            "opens_in_minutes": check.get("opens_in_minutes"),
        }

    def get_session_weight(self, current_time: datetime = None) -> float:
        """
        Get session weight for current time.

        Args:
            current_time: Optional datetime

        Returns:
            Session weight (0.0 - 1.2+)
        """
        session = self.get_current_session(current_time)
        if session:
            return session.get("weight", 1.0)
        return 0.0

    def get_confluence_adjustment(self, current_time: datetime = None) -> float:
        """
        Get confluence score adjustment for current session.

        Args:
            current_time: Optional datetime

        Returns:
            Adjustment value (can be negative to lower threshold)
        """
        session = self.get_current_session(current_time)
        if session:
            return session.get("confluence_adjustment", 0.0)
        return 0.0

    def get_session_summary(self, current_time: datetime = None) -> str:
        """
        Get human-readable session summary.

        Args:
            current_time: Optional datetime

        Returns:
            Summary string
        """
        if current_time is None:
            current_time = datetime.now(pytz.UTC)

        current_session = self.get_current_session(current_time)

        if not current_session:
            next_session = self.get_next_session(current_time)
            if next_session:
                minutes = next_session.get("minutes_until", 0)
                hours = minutes // 60
                mins = minutes % 60
                return f"No active session. Next: {next_session['name']} in {hours}h {mins}m"
            return "No active session"

        lines = [
            f"Current Session: {current_session['name']}",
            f"  Weight: {current_session['weight']:.2f}",
            f"  Confluence Adj: {current_session['confluence_adjustment']:+.2f}",
        ]

        # Check if overlap
        all_active = self.get_all_active_sessions(current_time)
        if len(all_active) > 1:
            lines.append(f"  Active Sessions: {', '.join([s['name'] for s in all_active])}")

        return "\n".join(lines)

    def is_preferred_session(self, current_time: datetime = None) -> bool:
        """
        Check if current session is preferred.

        Args:
            current_time: Optional datetime

        Returns:
            True if preferred session
        """
        preferences = self.config.get("preferences", {})
        preferred = preferences.get("preferred_sessions", [])

        session = self.get_current_session(current_time)
        if session:
            return session.get("key") in preferred

        return False

    def get_session_adjustments(self, current_time: datetime = None) -> Dict:
        """
        Get strategy adjustments for current session.

        Args:
            current_time: Optional datetime

        Returns:
            Adjustments dictionary
        """
        session = self.get_current_session(current_time)
        if not session:
            return {}

        preferences = self.config.get("preferences", {})
        session_adjustments = preferences.get("session_adjustments", {})

        session_key = session.get("key")
        return session_adjustments.get(session_key, {})
