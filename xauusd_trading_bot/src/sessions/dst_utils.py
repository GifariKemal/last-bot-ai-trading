"""
DST-aware session time utilities.

Computes correct UTC session boundaries by checking London/NY DST status
via pytz. All functions are pure — no state, no side effects.

Session times (UTC):
| Session     | Winter (no DST) | Summer (DST)  |
|-------------|-----------------|---------------|
| Asian       | 00:00-09:00     | 00:00-09:00   |  Japan has no DST
| London      | 08:00-17:00     | 07:00-16:00   |  Europe/London
| New York    | 13:00-22:00     | 12:00-21:00   |  America/New_York
| Overlap     | intersection(L, NY)                   |
| Maintenance | NY_end to NY_end+1h                   |
"""

from datetime import datetime, time
from typing import Dict

import pytz

_TZ_LONDON = pytz.timezone("Europe/London")
_TZ_NEW_YORK = pytz.timezone("America/New_York")


def is_dst(tz_name: str, dt: datetime) -> bool:
    """Check if a timezone is currently in DST at the given UTC datetime."""
    tz = pytz.timezone(tz_name)
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    local = dt.astimezone(tz)
    return bool(local.dst())


def get_dst_status(dt: datetime = None) -> Dict[str, bool]:
    """Return DST status for London and New York at given UTC datetime."""
    if dt is None:
        dt = datetime.now(pytz.UTC)
    elif dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    return {
        "london_dst": is_dst("Europe/London", dt),
        "new_york_dst": is_dst("America/New_York", dt),
    }


def get_session_times_utc(dt: datetime = None) -> Dict:
    """
    Compute all session/maintenance/close times for a given UTC datetime.

    Returns dict with keys:
        asian_start, asian_end,
        london_start, london_end,
        new_york_start, new_york_end,
        overlap_start, overlap_end,
        maintenance_start, maintenance_end,
        friday_close_time, pre_close_hour
    """
    status = get_dst_status(dt)
    london_dst = status["london_dst"]
    ny_dst = status["new_york_dst"]

    # Asian: Japan has no DST — always 00:00-09:00 UTC
    asian_start = time(0, 0)
    asian_end = time(9, 0)

    # London: winter 08:00-17:00, summer 07:00-16:00
    if london_dst:
        london_start = time(7, 0)
        london_end = time(16, 0)
    else:
        london_start = time(8, 0)
        london_end = time(17, 0)

    # New York: winter 13:00-22:00, summer 12:00-21:00
    if ny_dst:
        ny_start = time(12, 0)
        ny_end = time(21, 0)
    else:
        ny_start = time(13, 0)
        ny_end = time(22, 0)

    # Overlap = intersection of London and NY
    overlap_start = max(london_start, ny_start)
    overlap_end = min(london_end, ny_end)

    # Maintenance = NY close to NY close + 1h
    maintenance_start = ny_end
    maint_end_hour = (ny_end.hour + 1) % 24
    maintenance_end = time(maint_end_hour, 0)

    # Friday close = maintenance_start - 30min
    fri_close_minute = ny_end.minute - 30
    fri_close_hour = ny_end.hour
    if fri_close_minute < 0:
        fri_close_minute += 60
        fri_close_hour -= 1
    friday_close_time = time(fri_close_hour, fri_close_minute)

    # Pre-close profit lock hour = maintenance_start - 1h
    pre_close_hour = ny_end.hour - 1

    return {
        "asian_start": asian_start,
        "asian_end": asian_end,
        "london_start": london_start,
        "london_end": london_end,
        "new_york_start": ny_start,
        "new_york_end": ny_end,
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "maintenance_start": maintenance_start,
        "maintenance_end": maintenance_end,
        "friday_close_time": friday_close_time,
        "pre_close_hour": pre_close_hour,
        "dst_status": status,
    }


def format_dst_summary(dt: datetime = None) -> str:
    """Human-readable DST summary for logging."""
    times = get_session_times_utc(dt)
    status = times["dst_status"]
    london_label = "SUMMER (BST)" if status["london_dst"] else "WINTER (GMT)"
    ny_label = "SUMMER (EDT)" if status["new_york_dst"] else "WINTER (EST)"

    return (
        f"DST: London={london_label}, NY={ny_label} | "
        f"Sessions: Asian {times['asian_start'].strftime('%H:%M')}-{times['asian_end'].strftime('%H:%M')}, "
        f"London {times['london_start'].strftime('%H:%M')}-{times['london_end'].strftime('%H:%M')}, "
        f"NY {times['new_york_start'].strftime('%H:%M')}-{times['new_york_end'].strftime('%H:%M')}, "
        f"Overlap {times['overlap_start'].strftime('%H:%M')}-{times['overlap_end'].strftime('%H:%M')} | "
        f"Maintenance {times['maintenance_start'].strftime('%H:%M')}-{times['maintenance_end'].strftime('%H:%M')}"
    )
