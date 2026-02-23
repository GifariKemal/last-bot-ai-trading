"""
Tests: SessionDetector — market hours accuracy (DST-aware).

Base date 2026-02-23 is winter (no DST), so expected session times are:
  Asian 00:00-09:00, London 08:00-17:00, NY 13:00-22:00, Overlap 13:00-17:00
  Maintenance 22:00-23:00, Friday close 21:30
"""

import pytest
from conftest import make_utc
from src.sessions.session_detector import SessionDetector


@pytest.fixture
def detector(session_cfg):
    return SessionDetector(session_cfg)


# ── Weekend: market fully closed ─────────────────────────────────────────────

def test_saturday_noon_closed(detector):
    """Saturday all day must be WEEKEND, no trading."""
    result = detector.is_trading_allowed(make_utc(5, 12, 0))
    assert result["allowed"] is False
    assert result["status"] == "WEEKEND"


def test_saturday_midnight_closed(detector):
    """Saturday 00:00 UTC still closed."""
    result = detector.is_trading_allowed(make_utc(5, 0, 0))
    assert result["allowed"] is False


def test_sunday_morning_closed(detector):
    """Sunday 10:00 UTC fully closed."""
    result = detector.is_trading_allowed(make_utc(6, 10, 0))
    assert result["allowed"] is False
    assert result["status"] == "WEEKEND"


def test_sunday_evening_closed(detector):
    """Sunday 21:00 UTC still closed (old assumption: was OPEN at this time)."""
    result = detector.is_trading_allowed(make_utc(6, 21, 0))
    assert result["allowed"] is False
    assert result["status"] == "WEEKEND"


def test_get_current_session_returns_none_saturday(detector):
    """get_current_session must return None on Saturday (not match time-of-day sessions)."""
    session = detector.get_current_session(make_utc(5, 14, 0))  # Sat 14:00 would be London hours
    assert session is None


# ── Daily maintenance 22:00–23:00 UTC (winter) ───────────────────────────────

def test_maintenance_monday_22_30(detector):
    """Monday 22:30 UTC is inside maintenance window."""
    result = detector.is_trading_allowed(make_utc(0, 22, 30))
    assert result["allowed"] is False
    assert result["status"] == "MAINTENANCE"


def test_maintenance_wednesday_22_00(detector):
    """Wednesday 22:00 UTC is start of maintenance."""
    result = detector.is_trading_allowed(make_utc(2, 22, 0))
    assert result["allowed"] is False
    assert result["status"] == "MAINTENANCE"


def test_maintenance_ends_at_23_00(detector):
    """Monday 23:00 UTC: maintenance is OVER, trading resumes."""
    result = detector.is_trading_allowed(make_utc(0, 23, 0))
    assert result["allowed"] is True


def test_pre_maintenance_still_open(detector):
    """Wednesday 21:59 UTC: still open (maintenance starts at 22:00)."""
    result = detector.is_trading_allowed(make_utc(2, 21, 59))
    assert result["allowed"] is True


# ── Friday early close ────────────────────────────────────────────────────────

def test_friday_20_still_open(detector):
    """Friday 20:00 UTC: before 21:30 close, still open."""
    result = detector.is_trading_allowed(make_utc(4, 20, 0))
    assert result["allowed"] is True


def test_friday_21_30_closed(detector):
    """Friday 21:30 UTC: exactly at close time -> WEEKEND."""
    result = detector.is_trading_allowed(make_utc(4, 21, 30))
    assert result["allowed"] is False
    assert result["status"] == "WEEKEND"


def test_friday_21_45_closed(detector):
    """Friday 21:45 UTC: after close, last candle already happened -> WEEKEND."""
    result = detector.is_trading_allowed(make_utc(4, 21, 45))
    assert result["allowed"] is False
    assert result["status"] == "WEEKEND"


# ── Weekday trading ───────────────────────────────────────────────────────────

def test_monday_asian_open(detector):
    """Monday 02:00 UTC is Asian session (inside 00:00-09:00)."""
    result = detector.is_trading_allowed(make_utc(0, 2, 0))
    assert result["allowed"] is True


def test_tuesday_london_open(detector):
    """Tuesday 09:00 UTC is London session."""
    result = detector.is_trading_allowed(make_utc(1, 9, 0))
    assert result["allowed"] is True


def test_thursday_ny_open(detector):
    """Thursday 15:00 UTC is NY session."""
    result = detector.is_trading_allowed(make_utc(3, 15, 0))
    assert result["allowed"] is True


# ── Session weight: overlap wins during 13:00–17:00 (winter) ─────────────────

def test_overlap_highest_weight_14h(detector):
    """At 14:00 UTC, overlap (1.18) beats London (1.16) and NY (1.16)."""
    session = detector.get_current_session(make_utc(0, 14, 0))
    assert session is not None
    assert session["key"] == "overlap"
    assert session["weight"] == 1.18


def test_london_only_at_10h(detector):
    """At 10:00 UTC only London is active (no overlap, no NY)."""
    session = detector.get_current_session(make_utc(0, 10, 0))
    assert session is not None
    assert session["key"] == "london"


def test_ny_only_at_20h(detector):
    """At 20:00 UTC only NY is active (London ended at 17:00 winter)."""
    session = detector.get_current_session(make_utc(0, 20, 0))
    assert session is not None
    assert session["key"] == "new_york"


# ── get_all_active_sessions ───────────────────────────────────────────────────

def test_multiple_sessions_during_overlap(detector):
    """During overlap (13:00–17:00 winter), both London and NY are in list."""
    sessions = detector.get_all_active_sessions(make_utc(0, 14, 0))
    keys = [s["key"] for s in sessions]
    assert "london" in keys
    assert "new_york" in keys
    assert "overlap" in keys


def test_no_sessions_during_maintenance(detector):
    """During maintenance, get_all_active_sessions returns empty list."""
    sessions = detector.get_all_active_sessions(make_utc(0, 22, 30))
    assert sessions == []


def test_no_sessions_on_weekend(detector):
    """On weekend, get_all_active_sessions returns empty list."""
    sessions = detector.get_all_active_sessions(make_utc(5, 14, 0))
    assert sessions == []


# ── DST integration ───────────────────────────────────────────────────────────

def test_dst_status_cached(detector):
    """Detector caches DST status and has ny_close_time set."""
    assert detector._last_dst_status is not None
    assert detector._ny_close_time is not None
    assert detector._pre_close_hour is not None
