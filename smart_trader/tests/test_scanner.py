"""
Tests for smart_trader/src/scanner.py

Covers:
- find_nearby_zones
- direction_for_zone
- is_spike_window
- current_session
- check_risk_filters
"""
import sys
import os
import pytest
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scanner import (
    find_nearby_zones,
    direction_for_zone,
    is_spike_window,
    current_session,
    check_risk_filters,
)


# ── find_nearby_zones ─────────────────────────────────────────────────────────

class TestFindNearbyZones:

    def test_price_inside_low_high_zone_distance_zero(self, sample_zones):
        # BULL_OB: low=2895, high=2900; price=2897 is inside → distance=0
        result = find_nearby_zones(2897.0, sample_zones, proximity_pts=5.0)
        bull_ob = next((z for z in result if z["type"] == "BULL_OB"), None)
        assert bull_ob is not None
        assert bull_ob["distance_pts"] == 0

    def test_price_just_outside_within_proximity_returned_with_correct_distance(self, sample_zones):
        # BULL_OB: low=2895, high=2900; price=2902 is 2 pts above high
        # proximity=5 → should be returned with distance=2
        result = find_nearby_zones(2902.0, sample_zones, proximity_pts=5.0)
        bull_ob = next((z for z in result if z["type"] == "BULL_OB"), None)
        assert bull_ob is not None
        assert bull_ob["distance_pts"] == 2.0

    def test_price_just_below_low_within_proximity_correct_distance(self, sample_zones):
        # BULL_OB: low=2895, high=2900; price=2893 is 2 pts below low
        # proximity=5 → distance=2
        result = find_nearby_zones(2893.0, sample_zones, proximity_pts=5.0)
        bull_ob = next((z for z in result if z["type"] == "BULL_OB"), None)
        assert bull_ob is not None
        assert bull_ob["distance_pts"] == 2.0

    def test_bos_level_only_within_proximity_returned(self, sample_zones):
        # BOS_BULL: level=2920; price=2918 → distance=2, within proximity=5
        result = find_nearby_zones(2918.0, sample_zones, proximity_pts=5.0)
        bos_bull = next((z for z in result if z["type"] == "BOS_BULL"), None)
        assert bos_bull is not None
        assert bos_bull["distance_pts"] == 2.0

    def test_price_too_far_returns_empty(self, sample_zones):
        # price=3100, all zones are near 2870-2955 → nothing within 5 pts
        result = find_nearby_zones(3100.0, sample_zones, proximity_pts=5.0)
        assert result == []

    def test_results_sorted_by_distance_closest_first(self, sample_zones):
        # BOS_BULL at 2920 (dist=0), BULL_FVG low=2910/high=2915 (dist=5),
        # price=2920 is exactly at BOS_BULL level
        result = find_nearby_zones(2920.0, sample_zones, proximity_pts=10.0)
        distances = [z["distance_pts"] for z in result]
        assert distances == sorted(distances)

    def test_multiple_zones_sorted_ascending(self):
        zones = [
            {"type": "BULL_OB", "low": 2900.0, "high": 2905.0, "level": None},
            {"type": "BEAR_OB", "low": 2920.0, "high": 2925.0, "level": None},
            {"type": "BOS_BULL", "low": None, "high": None, "level": 2910.0},
        ]
        # price=2912: dist from BOS_BULL(2910)=2, dist from BULL_OB high(2905)=7,
        # dist from BEAR_OB low(2920)=8
        result = find_nearby_zones(2912.0, zones, proximity_pts=10.0)
        assert len(result) == 3
        assert result[0]["type"] == "BOS_BULL"
        assert result[0]["distance_pts"] == 2.0

    def test_proximity_boundary_exact_not_included_when_beyond(self):
        zones = [{"type": "BULL_OB", "low": 2900.0, "high": 2905.0, "level": None}]
        # price=2910, high=2905, dist=5, proximity=4 → NOT included
        result = find_nearby_zones(2910.0, zones, proximity_pts=4.0)
        assert result == []

    def test_proximity_boundary_exact_included_at_edge(self):
        zones = [{"type": "BULL_OB", "low": 2900.0, "high": 2905.0, "level": None}]
        # price=2910, high=2905, dist=5, proximity=5 → included (<=)
        result = find_nearby_zones(2910.0, zones, proximity_pts=5.0)
        assert len(result) == 1
        assert result[0]["distance_pts"] == 5.0

    def test_zone_without_level_and_without_low_high_skipped(self):
        zones = [{"type": "UNKNOWN", "low": None, "high": None, "level": None}]
        result = find_nearby_zones(2900.0, zones, proximity_pts=50.0)
        assert result == []

    def test_original_zone_keys_preserved(self, sample_zones):
        result = find_nearby_zones(2897.0, sample_zones, proximity_pts=5.0)
        bull_ob = next((z for z in result if z["type"] == "BULL_OB"), None)
        assert bull_ob is not None
        assert "detected_at" in bull_ob
        assert "distance_pts" in bull_ob
        assert bull_ob["low"] == 2895.0
        assert bull_ob["high"] == 2900.0


# ── direction_for_zone ────────────────────────────────────────────────────────

class TestDirectionForZone:

    def test_bull_ob_returns_long(self):
        assert direction_for_zone({"type": "BULL_OB"}) == "LONG"

    def test_bull_fvg_returns_long(self):
        assert direction_for_zone({"type": "BULL_FVG"}) == "LONG"

    def test_bull_breaker_returns_long(self):
        assert direction_for_zone({"type": "BULL_BREAKER"}) == "LONG"

    def test_bos_bull_returns_long(self):
        assert direction_for_zone({"type": "BOS_BULL"}) == "LONG"

    def test_choch_bull_returns_long(self):
        assert direction_for_zone({"type": "CHOCH_BULL"}) == "LONG"

    def test_bear_ob_returns_short(self):
        assert direction_for_zone({"type": "BEAR_OB"}) == "SHORT"

    def test_bear_fvg_returns_short(self):
        assert direction_for_zone({"type": "BEAR_FVG"}) == "SHORT"

    def test_bear_breaker_returns_short(self):
        assert direction_for_zone({"type": "BEAR_BREAKER"}) == "SHORT"

    def test_bos_bear_returns_short(self):
        assert direction_for_zone({"type": "BOS_BEAR"}) == "SHORT"

    def test_choch_bear_returns_short(self):
        assert direction_for_zone({"type": "CHOCH_BEAR"}) == "SHORT"

    def test_unknown_type_returns_none(self):
        assert direction_for_zone({"type": "UNKNOWN_ZONE"}) is None

    def test_empty_type_returns_none(self):
        assert direction_for_zone({"type": ""}) is None

    def test_missing_type_key_returns_none(self):
        assert direction_for_zone({}) is None


# ── is_spike_window ───────────────────────────────────────────────────────────

class TestIsSpikeWindow:

    def _utc(self, hour, minute):
        return datetime(2026, 2, 22, hour, minute, 0, tzinfo=timezone.utc)

    def test_london_spike_midpoint_returns_true(self):
        # 07:50 UTC → inside London spike window 07:45–08:00
        assert is_spike_window(self._utc(7, 50)) is True

    def test_ny_spike_midpoint_returns_true(self):
        # 12:50 UTC → inside NY spike window 12:45–13:00
        assert is_spike_window(self._utc(12, 50)) is True

    def test_outside_any_window_returns_false(self):
        # 10:00 UTC → not in any spike window
        assert is_spike_window(self._utc(10, 0)) is False

    def test_london_spike_at_boundary_start_inclusive(self):
        # 07:45 exactly → start of London window, inclusive
        assert is_spike_window(self._utc(7, 45)) is True

    def test_london_spike_at_boundary_end_exclusive(self):
        # 08:00 exactly → end of London window, exclusive (not included)
        assert is_spike_window(self._utc(8, 0)) is False

    def test_ny_spike_at_boundary_start_inclusive(self):
        # 12:45 exactly → start of NY window, inclusive
        assert is_spike_window(self._utc(12, 45)) is True

    def test_ny_spike_at_boundary_end_exclusive(self):
        # 13:00 exactly → end of NY window, exclusive
        assert is_spike_window(self._utc(13, 0)) is False

    def test_just_before_london_window_returns_false(self):
        # 07:44 UTC → just before London spike, not included
        assert is_spike_window(self._utc(7, 44)) is False

    def test_just_after_london_window_returns_false(self):
        # 08:01 UTC → just after London spike ended
        assert is_spike_window(self._utc(8, 1)) is False

    def test_midnight_returns_false(self):
        assert is_spike_window(self._utc(0, 0)) is False


# ── current_session ───────────────────────────────────────────────────────────

class TestCurrentSession:

    def _utc(self, hour, minute=0):
        return datetime(2026, 2, 22, hour, minute, 0, tzinfo=timezone.utc)

    def test_13_00_utc_returns_overlap(self):
        result = current_session(self._utc(13, 0))
        assert result["name"] == "OVERLAP"

    def test_12_00_utc_returns_overlap_boundary_start(self):
        # OVERLAP starts at 12:00 UTC
        result = current_session(self._utc(12, 0))
        assert result["name"] == "OVERLAP"

    def test_09_00_utc_returns_london(self):
        result = current_session(self._utc(9, 0))
        assert result["name"] == "LONDON"

    def test_16_30_utc_returns_new_york(self):
        result = current_session(self._utc(16, 30))
        assert result["name"] == "NEW_YORK"

    def test_03_00_utc_returns_asian(self):
        result = current_session(self._utc(3, 0))
        assert result["name"] == "ASIAN"

    def test_20_00_utc_returns_off_hours(self):
        result = current_session(self._utc(20, 0))
        assert result["name"] == "OFF_HOURS"

    def test_17_00_utc_returns_off_hours_boundary(self):
        # OFF_HOURS starts at 17:00 UTC
        result = current_session(self._utc(17, 0))
        assert result["name"] == "OFF_HOURS"

    def test_session_has_priority_key(self):
        result = current_session(self._utc(13, 0))
        assert "priority" in result
        assert result["priority"] == "HIGHEST"

    def test_london_priority_is_high(self):
        result = current_session(self._utc(9, 0))
        assert result["priority"] == "HIGH"

    def test_asian_priority_is_low(self):
        result = current_session(self._utc(3, 0))
        assert result["priority"] == "LOW"

    def test_0000_utc_returns_asian(self):
        # 00:00 UTC → ASIAN (0:00-7:00)
        result = current_session(self._utc(0, 0))
        assert result["name"] == "ASIAN"

    def test_16_00_utc_returns_new_york_boundary_start(self):
        # NEW_YORK starts at 16:00 UTC
        result = current_session(self._utc(16, 0))
        assert result["name"] == "NEW_YORK"


# ── check_risk_filters ────────────────────────────────────────────────────────

class _FakeRaw:
    def __init__(self, magic):
        self.magic = magic


def _make_position(direction, magic=202602, profit=1.0):
    return {"type": direction, "volume": 0.01, "profit": profit,
            "_raw": _FakeRaw(magic)}


class TestCheckRiskFilters:

    def test_empty_positions_good_account_passes(self, mock_account, mock_positions_empty):
        ok, reason = check_risk_filters(
            mock_account, mock_positions_empty,
            direction="LONG",
            max_positions=3,
            max_per_direction=2,
            max_drawdown_pct=10.0,
            free_margin_pct=20.0,
        )
        assert ok is True
        assert reason == ""

    def test_positions_at_max_fails(self, mock_account, mock_positions_at_limit):
        # mock_positions_at_limit has 1 position, set max_positions=1
        ok, reason = check_risk_filters(
            mock_account, mock_positions_at_limit,
            direction="LONG",
            max_positions=1,
            max_per_direction=2,
            max_drawdown_pct=10.0,
            free_margin_pct=20.0,
        )
        assert ok is False
        assert "Max positions" in reason

    def test_same_direction_at_max_per_direction_fails(self, mock_account):
        positions = [
            _make_position("LONG"),
            _make_position("LONG"),
        ]
        ok, reason = check_risk_filters(
            mock_account, positions,
            direction="LONG",
            max_positions=5,
            max_per_direction=2,
            max_drawdown_pct=10.0,
            free_margin_pct=20.0,
        )
        assert ok is False
        assert "LONG" in reason

    def test_drawdown_exceeded_fails(self):
        account = {"balance": 100.0, "equity": 80.0, "margin_free": 95.0}
        # drawdown = (100 - 80) / 100 * 100 = 20%
        ok, reason = check_risk_filters(
            account, [],
            direction="LONG",
            max_positions=3,
            max_per_direction=2,
            max_drawdown_pct=10.0,
            free_margin_pct=5.0,
        )
        assert ok is False
        assert "Drawdown" in reason

    def test_free_margin_too_low_fails(self):
        account = {"balance": 100.0, "equity": 99.0, "margin_free": 10.0}
        # margin_ratio = 10/100*100 = 10% < 20%
        ok, reason = check_risk_filters(
            account, [],
            direction="LONG",
            max_positions=3,
            max_per_direction=2,
            max_drawdown_pct=50.0,
            free_margin_pct=20.0,
        )
        assert ok is False
        assert "margin" in reason.lower()

    def test_foreign_magic_positions_not_counted(self, mock_account):
        # 3 positions with wrong magic — should be ignored, so max_positions=1 should still pass
        positions = [
            _make_position("LONG", magic=999999),
            _make_position("LONG", magic=999999),
            _make_position("SHORT", magic=999999),
        ]
        ok, reason = check_risk_filters(
            mock_account, positions,
            direction="LONG",
            max_positions=1,
            max_per_direction=1,
            max_drawdown_pct=10.0,
            free_margin_pct=20.0,
        )
        assert ok is True

    def test_mix_of_our_and_foreign_only_ours_counted(self, mock_account):
        # 2 foreign + 1 ours LONG; max_per_direction=1 → fails on direction check
        positions = [
            _make_position("LONG", magic=999999),
            _make_position("LONG", magic=999999),
            _make_position("LONG", magic=202602),
        ]
        ok, reason = check_risk_filters(
            mock_account, positions,
            direction="LONG",
            max_positions=5,
            max_per_direction=1,
            max_drawdown_pct=10.0,
            free_margin_pct=20.0,
        )
        assert ok is False
        assert "LONG" in reason

    def test_opposite_direction_not_counted_for_max_per_direction(self, mock_account):
        # Have 1 LONG (our bot), try to enter SHORT → max_per_direction=1 for SHORT
        positions = [_make_position("LONG", magic=202602)]
        ok, reason = check_risk_filters(
            mock_account, positions,
            direction="SHORT",
            max_positions=3,
            max_per_direction=1,
            max_drawdown_pct=10.0,
            free_margin_pct=20.0,
        )
        assert ok is True

    def test_zero_balance_skips_drawdown_check(self):
        account = {"balance": 0.0, "equity": 0.0, "margin_free": 0.0}
        # balance=0 → drawdown check skipped → should not fail on drawdown/margin
        ok, reason = check_risk_filters(
            account, [],
            direction="LONG",
            max_positions=3,
            max_per_direction=2,
            max_drawdown_pct=5.0,
            free_margin_pct=20.0,
        )
        # Should pass (balance=0 skips both checks)
        assert ok is True

    def test_position_without_raw_key_not_counted(self, mock_account):
        # Position without "_raw" key should not be counted as ours
        positions = [{"type": "LONG", "volume": 0.01, "profit": 1.0}]
        ok, reason = check_risk_filters(
            mock_account, positions,
            direction="LONG",
            max_positions=1,
            max_per_direction=1,
            max_drawdown_pct=10.0,
            free_margin_pct=20.0,
        )
        # No "_raw" → `p.get("_raw")` is falsy → not counted → passes
        assert ok is True

    def test_one_long_fixture_does_not_block_short_entry(
        self, mock_account, mock_positions_one_long
    ):
        ok, reason = check_risk_filters(
            mock_account, mock_positions_one_long,
            direction="SHORT",
            max_positions=3,
            max_per_direction=2,
            max_drawdown_pct=10.0,
            free_margin_pct=20.0,
        )
        assert ok is True

    def test_returns_tuple_of_bool_and_string(self, mock_account, mock_positions_empty):
        result = check_risk_filters(
            mock_account, mock_positions_empty,
            direction="LONG",
            max_positions=3,
            max_per_direction=2,
            max_drawdown_pct=10.0,
            free_margin_pct=20.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
