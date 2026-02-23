"""
Tests: dst_utils — DST-aware session time computation.

Verifies correct session boundaries for winter, summer, and transition periods.
"""

from datetime import datetime, time
import pytz
import pytest

from src.sessions.dst_utils import (
    is_dst,
    get_dst_status,
    get_session_times_utc,
    format_dst_summary,
)


def _utc(year, month, day, hour=12):
    return datetime(year, month, day, hour, 0, 0, tzinfo=pytz.UTC)


# ── DST status detection ─────────────────────────────────────────────────────

def test_winter_dst_status():
    """January: both London and NY are in winter (no DST)."""
    dt = _utc(2026, 1, 15)
    status = get_dst_status(dt)
    assert status["london_dst"] is False
    assert status["new_york_dst"] is False


def test_summer_dst_status():
    """July: both London and NY are in summer (DST active)."""
    dt = _utc(2026, 7, 15)
    status = get_dst_status(dt)
    assert status["london_dst"] is True
    assert status["new_york_dst"] is True


def test_spring_transition_ny_first():
    """Mid-March: NY springs forward first (2nd Sun March), London still winter.
    2026: US DST starts March 8, UK BST starts March 29."""
    dt = _utc(2026, 3, 15)  # After US DST, before UK BST
    status = get_dst_status(dt)
    assert status["new_york_dst"] is True
    assert status["london_dst"] is False


def test_fall_transition_ny_last():
    """Late October: London falls back first, NY still in DST.
    2026: UK BST ends Oct 25, US DST ends Nov 1."""
    dt = _utc(2026, 10, 28)  # After UK fall-back, before US
    status = get_dst_status(dt)
    assert status["london_dst"] is False
    assert status["new_york_dst"] is True


# ── Winter session times ──────────────────────────────────────────────────────

def test_winter_session_times():
    """Winter: Asian 00-09, London 08-17, NY 13-22, Overlap 13-17."""
    dt = _utc(2026, 1, 15)
    times = get_session_times_utc(dt)

    assert times["asian_start"] == time(0, 0)
    assert times["asian_end"] == time(9, 0)
    assert times["london_start"] == time(8, 0)
    assert times["london_end"] == time(17, 0)
    assert times["new_york_start"] == time(13, 0)
    assert times["new_york_end"] == time(22, 0)
    assert times["overlap_start"] == time(13, 0)
    assert times["overlap_end"] == time(17, 0)


def test_winter_maintenance():
    """Winter maintenance: 22:00-23:00 UTC."""
    dt = _utc(2026, 1, 15)
    times = get_session_times_utc(dt)
    assert times["maintenance_start"] == time(22, 0)
    assert times["maintenance_end"] == time(23, 0)


def test_winter_friday_close():
    """Winter Friday close: 21:30, pre-close hour: 21."""
    dt = _utc(2026, 1, 15)
    times = get_session_times_utc(dt)
    assert times["friday_close_time"] == time(21, 30)
    assert times["pre_close_hour"] == 21


# ── Summer session times ─────────────────────────────────────────────────────

def test_summer_session_times():
    """Summer: Asian 00-09, London 07-16, NY 12-21, Overlap 12-16."""
    dt = _utc(2026, 7, 15)
    times = get_session_times_utc(dt)

    assert times["asian_start"] == time(0, 0)
    assert times["asian_end"] == time(9, 0)
    assert times["london_start"] == time(7, 0)
    assert times["london_end"] == time(16, 0)
    assert times["new_york_start"] == time(12, 0)
    assert times["new_york_end"] == time(21, 0)
    assert times["overlap_start"] == time(12, 0)
    assert times["overlap_end"] == time(16, 0)


def test_summer_maintenance():
    """Summer maintenance: 21:00-22:00 UTC."""
    dt = _utc(2026, 7, 15)
    times = get_session_times_utc(dt)
    assert times["maintenance_start"] == time(21, 0)
    assert times["maintenance_end"] == time(22, 0)


def test_summer_friday_close():
    """Summer Friday close: 20:30, pre-close hour: 20."""
    dt = _utc(2026, 7, 15)
    times = get_session_times_utc(dt)
    assert times["friday_close_time"] == time(20, 30)
    assert times["pre_close_hour"] == 20


# ── Transition period: overlap narrows ────────────────────────────────────────

def test_spring_transition_overlap():
    """Spring transition (NY DST, London winter): overlap is 13:00-16:00.
    London 08-17, NY 12-21 -> overlap max(08,12)-min(17,21) = 12-17.
    Wait — NY is DST (12-21), London is winter (08-17).
    Overlap = max(08:00, 12:00) - min(17:00, 21:00) = 12:00-17:00.
    """
    dt = _utc(2026, 3, 15)
    times = get_session_times_utc(dt)
    assert times["overlap_start"] == time(12, 0)
    assert times["overlap_end"] == time(17, 0)


def test_fall_transition_overlap():
    """Fall transition (London winter, NY DST): NY 12-21, London 08-17.
    Overlap = max(08,12)-min(17,21) = 12:00-17:00."""
    dt = _utc(2026, 10, 28)
    times = get_session_times_utc(dt)
    assert times["overlap_start"] == time(12, 0)
    assert times["overlap_end"] == time(17, 0)


# ── format_dst_summary ───────────────────────────────────────────────────────

def test_format_winter_summary():
    """Winter summary contains WINTER labels."""
    dt = _utc(2026, 1, 15)
    summary = format_dst_summary(dt)
    assert "WINTER" in summary
    assert "London" in summary
    assert "NY" in summary


def test_format_summer_summary():
    """Summer summary contains SUMMER labels."""
    dt = _utc(2026, 7, 15)
    summary = format_dst_summary(dt)
    assert "SUMMER" in summary


# ── is_dst helper ─────────────────────────────────────────────────────────────

def test_is_dst_london_winter():
    assert is_dst("Europe/London", _utc(2026, 1, 15)) is False


def test_is_dst_london_summer():
    assert is_dst("Europe/London", _utc(2026, 7, 15)) is True


def test_is_dst_tokyo_always_false():
    """Japan has no DST."""
    assert is_dst("Asia/Tokyo", _utc(2026, 1, 15)) is False
    assert is_dst("Asia/Tokyo", _utc(2026, 7, 15)) is False
