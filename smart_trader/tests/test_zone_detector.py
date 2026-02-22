"""
Tests for smart_trader/src/zone_detector.py

Covers: detect_fvg, detect_ob, detect_bos, detect_all_zones, merge_zones
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from zone_detector import detect_fvg, detect_ob, detect_bos, detect_all_zones, merge_zones


# ── Helpers ───────────────────────────────────────────────────────────────────

REQUIRED_ZONE_KEYS = {"type", "detected_at"}


def _assert_zone_structure(zones: list):
    """All returned zone dicts must have at minimum: type, detected_at."""
    for z in zones:
        for key in REQUIRED_ZONE_KEYS:
            assert key in z, f"Zone missing required key '{key}': {z}"


def _make_df_with_time(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame that has 'time' as a regular column (not index)."""
    df = pd.DataFrame(rows)
    return df


def _make_flat_ohlcv(n: int, base: float = 2900.0) -> pd.DataFrame:
    """Minimal flat candles with time column, no index."""
    now = datetime(2026, 2, 10, 8, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        rows.append({
            "time": now + timedelta(hours=i),
            "open": base,
            "high": base + 2.0,
            "low": base - 2.0,
            "close": base,
            "tick_volume": 200,
        })
    return pd.DataFrame(rows)


def _make_trending_ohlcv(n: int, step: float = 5.0, base: float = 2900.0,
                         impulse_every: int = 5, impulse_size: float = 20.0) -> pd.DataFrame:
    """
    Trending OHLCV with periodic large impulse candles to trigger OB detection.
    impulse_every: every N candles, insert a strong bullish impulse candle.
    The candle BEFORE each impulse is a small bearish retrace (the OB candidate).
    """
    now = datetime(2026, 2, 10, 8, 0, tzinfo=timezone.utc)
    rows = []
    price = base
    for i in range(n):
        t = now + timedelta(hours=i)
        if i > 0 and i % impulse_every == 0:
            # strong bullish impulse
            o = price
            c = price + impulse_size
            hi = c + 1.0
            lo = o - 1.0
        elif i > 0 and (i + 1) % impulse_every == 0:
            # bearish retrace candle right before the impulse (becomes the OB)
            o = price + 2.0   # open slightly above
            c = price         # close at current price → bearish (close < open)
            hi = o + 1.0
            lo = c - 1.0
        else:
            o = price
            c = price + step
            hi = c + 2.0
            lo = o - 2.0
        rows.append({
            "time": t,
            "open": round(o, 2),
            "high": round(hi, 2),
            "low": round(lo, 2),
            "close": round(c, 2),
            "tick_volume": 300,
        })
        price = c
    return pd.DataFrame(rows)


def _make_bos_ohlcv() -> pd.DataFrame:
    """
    50-bar dataset designed to produce at least one BOS.
    First 20 bars: oscillate around 2900.
    Next 10 bars: break above the swing high with closes > 2915.
    Remaining bars: continue above.
    """
    now = datetime(2026, 2, 10, 8, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(20):
        base = 2900.0
        amp = 5.0 * (1 if i % 2 == 0 else -1)
        o = base + amp
        c = base - amp
        rows.append({
            "time": now + timedelta(hours=i),
            "open": round(o, 2),
            "high": round(max(o, c) + 3.0, 2),
            "low": round(min(o, c) - 3.0, 2),
            "close": round(c, 2),
            "tick_volume": 200,
        })
    # Break above — swing high was ~2908; close well above
    for i in range(20, 50):
        p = 2920.0 + (i - 20) * 0.5
        rows.append({
            "time": now + timedelta(hours=i),
            "open": round(p, 2),
            "high": round(p + 3.0, 2),
            "low": round(p - 3.0, 2),
            "close": round(p + 1.0, 2),
            "tick_volume": 250,
        })
    return pd.DataFrame(rows)


# ── detect_fvg ───────────────────────────────────────────────────────────────

class TestDetectFvg:

    def test_returns_list(self, df_with_fvg):
        result = detect_fvg(df_with_fvg.reset_index())
        assert isinstance(result, list)

    def test_detects_at_least_one_fvg(self, df_with_fvg):
        result = detect_fvg(df_with_fvg.reset_index())
        assert len(result) >= 1, "Expected at least 1 FVG in df_with_fvg fixture"

    def test_fvg_zone_keys(self, df_with_fvg):
        result = detect_fvg(df_with_fvg.reset_index())
        _assert_zone_structure(result)

    def test_fvg_type_values(self, df_with_fvg):
        result = detect_fvg(df_with_fvg.reset_index())
        for z in result:
            assert z["type"] in ("BULL_FVG", "BEAR_FVG"), f"Unexpected FVG type: {z['type']}"

    def test_fvg_has_low_and_high(self, df_with_fvg):
        result = detect_fvg(df_with_fvg.reset_index())
        for z in result:
            assert "low" in z
            assert "high" in z
            # For FVGs the gap must be positive
            if z["low"] is not None and z["high"] is not None:
                assert z["high"] >= z["low"]

    def test_empty_df_returns_empty(self):
        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close"])
        result = detect_fvg(empty)
        assert result == []

    def test_short_df_less_than_3_returns_empty(self):
        df = _make_flat_ohlcv(2)
        result = detect_fvg(df)
        assert result == []

    def test_short_df_less_than_5_returns_empty_or_minimal(self):
        df = _make_flat_ohlcv(4)
        result = detect_fvg(df)
        # With 4 flat bars, no FVG gap should exceed min_gap_pts
        assert isinstance(result, list)

    def test_max_10_zones_returned(self, df_m15_neutral):
        result = detect_fvg(df_m15_neutral.reset_index())
        assert len(result) <= 10

    def test_no_fvg_in_flat_data(self):
        """Flat candles with tiny spreads should produce no FVGs (gap < 2pts)."""
        df = _make_flat_ohlcv(30)
        result = detect_fvg(df)
        assert result == []

    def test_bull_fvg_low_high_ordering(self, df_with_fvg):
        result = detect_fvg(df_with_fvg.reset_index())
        bull_fvgs = [z for z in result if z["type"] == "BULL_FVG"]
        for z in bull_fvgs:
            # Bull FVG: low = prev candle high, high = current candle low
            assert z["high"] > z["low"]

    def test_detected_at_is_string(self, df_with_fvg):
        result = detect_fvg(df_with_fvg.reset_index())
        for z in result:
            assert isinstance(z["detected_at"], str)

    def test_works_without_time_column(self):
        """df without 'time' column — should not raise."""
        df = pd.DataFrame({
            "open":  [2900, 2901, 2902, 2900, 2920],
            "high":  [2905, 2906, 2907, 2905, 2925],
            "low":   [2895, 2896, 2897, 2895, 2915],
            "close": [2902, 2903, 2904, 2902, 2922],
        })
        result = detect_fvg(df)
        assert isinstance(result, list)

    def test_source_is_detector(self, df_with_fvg):
        result = detect_fvg(df_with_fvg.reset_index())
        for z in result:
            assert z.get("source") == "detector"


# ── detect_ob ────────────────────────────────────────────────────────────────

class TestDetectOb:

    def test_returns_list(self, df_m15_bullish):
        result = detect_ob(df_m15_bullish.reset_index())
        assert isinstance(result, list)

    def test_ob_zone_required_keys(self, df_m15_bullish):
        result = detect_ob(df_m15_bullish.reset_index())
        _assert_zone_structure(result)

    def test_ob_type_values(self, df_m15_bullish):
        result = detect_ob(df_m15_bullish.reset_index())
        for z in result:
            assert z["type"] in ("BULL_OB", "BEAR_OB"), f"Unexpected OB type: {z['type']}"

    def test_ob_has_low_and_high_keys(self, df_m15_bullish):
        result = detect_ob(df_m15_bullish.reset_index())
        for z in result:
            assert "low" in z
            assert "high" in z

    def test_ob_level_key_present(self, df_m15_bullish):
        result = detect_ob(df_m15_bullish.reset_index())
        for z in result:
            assert "level" in z

    def test_detects_ob_in_trending_data(self):
        df = _make_trending_ohlcv(50, step=2.0, impulse_every=5, impulse_size=20.0)
        result = detect_ob(df)
        assert len(result) >= 1, "Expected at least 1 OB in strong trending data with impulses"

    def test_empty_df_returns_empty(self):
        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close"])
        result = detect_ob(empty)
        assert result == []

    def test_short_df_less_than_10_returns_empty(self):
        df = _make_flat_ohlcv(9)
        result = detect_ob(df)
        assert result == []

    def test_short_df_exactly_4_returns_empty(self):
        df = _make_flat_ohlcv(4)
        result = detect_ob(df)
        assert result == []

    def test_max_8_zones_returned(self):
        df = _make_trending_ohlcv(100, step=1.0, impulse_every=4, impulse_size=15.0)
        result = detect_ob(df)
        assert len(result) <= 8

    def test_source_is_detector(self):
        df = _make_trending_ohlcv(50, step=2.0, impulse_every=5, impulse_size=20.0)
        result = detect_ob(df)
        for z in result:
            assert z.get("source") == "detector"

    def test_bull_ob_high_gte_low(self):
        df = _make_trending_ohlcv(50, step=2.0, impulse_every=5, impulse_size=20.0)
        result = detect_ob(df)
        for z in result:
            if z["low"] is not None and z["high"] is not None:
                assert z["high"] >= z["low"]

    def test_detected_at_is_string(self):
        df = _make_trending_ohlcv(50, step=2.0, impulse_every=5, impulse_size=20.0)
        result = detect_ob(df)
        for z in result:
            assert isinstance(z["detected_at"], str)


# ── detect_bos ───────────────────────────────────────────────────────────────

class TestDetectBos:

    def test_returns_list(self, df_m15_bullish):
        result = detect_bos(df_m15_bullish.reset_index())
        assert isinstance(result, list)

    def test_bos_zone_required_keys(self, df_m15_bullish):
        result = detect_bos(df_m15_bullish.reset_index())
        _assert_zone_structure(result)

    def test_bos_has_level_key(self, df_m15_bullish):
        result = detect_bos(df_m15_bullish.reset_index())
        for z in result:
            assert "level" in z, f"BOS zone missing 'level' key: {z}"

    def test_bos_level_is_numeric(self, df_m15_bullish):
        result = detect_bos(df_m15_bullish.reset_index())
        for z in result:
            if z["level"] is not None:
                assert isinstance(z["level"], (int, float))

    def test_bos_type_values(self, df_m15_bullish):
        result = detect_bos(df_m15_bullish.reset_index())
        for z in result:
            assert z["type"] in ("BOS_BULL", "BOS_BEAR"), f"Unexpected BOS type: {z['type']}"

    def test_detects_bos_in_breakout_data(self):
        df = _make_bos_ohlcv()
        result = detect_bos(df)
        assert len(result) >= 1, "Expected at least 1 BOS in breakout data"

    def test_bos_bull_detected(self):
        df = _make_bos_ohlcv()
        result = detect_bos(df)
        bull_bos = [z for z in result if z["type"] == "BOS_BULL"]
        assert len(bull_bos) >= 1, "Expected at least 1 BOS_BULL in breakout data"

    def test_empty_df_returns_empty(self):
        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close"])
        result = detect_bos(empty)
        assert result == []

    def test_short_df_less_than_lookback_x3_returns_empty(self):
        # lookback=5, so need < 15 bars
        df = _make_flat_ohlcv(14)
        result = detect_bos(df)
        assert result == []

    def test_short_df_4_bars_returns_empty(self):
        df = _make_flat_ohlcv(4)
        result = detect_bos(df)
        assert result == []

    def test_max_8_zones_returned(self):
        df = _make_bos_ohlcv()
        result = detect_bos(df)
        assert len(result) <= 8

    def test_source_is_detector(self):
        df = _make_bos_ohlcv()
        result = detect_bos(df)
        for z in result:
            assert z.get("source") == "detector"

    def test_detected_at_is_string(self):
        df = _make_bos_ohlcv()
        result = detect_bos(df)
        for z in result:
            assert isinstance(z["detected_at"], str)

    def test_custom_lookback(self):
        df = _make_bos_ohlcv()
        result_5 = detect_bos(df, lookback=5)
        result_3 = detect_bos(df, lookback=3)
        # Both should be lists; different lookbacks may yield different counts
        assert isinstance(result_5, list)
        assert isinstance(result_3, list)


# ── detect_all_zones ─────────────────────────────────────────────────────────

class TestDetectAllZones:

    def test_returns_list(self, df_m15_neutral):
        result = detect_all_zones(df_m15_neutral.reset_index())
        assert isinstance(result, list)

    def test_all_zones_have_required_keys(self, df_m15_bullish):
        result = detect_all_zones(df_m15_bullish.reset_index())
        _assert_zone_structure(result)

    def test_combines_fvg_ob_bos(self):
        df = _make_bos_ohlcv()
        result = detect_all_zones(df)
        types = {z["type"] for z in result}
        # Should include at least one BOS type from breakout data
        bos_types = {t for t in types if "BOS" in t}
        assert len(bos_types) >= 1

    def test_uses_impulse_trending_data_for_ob(self):
        df = _make_trending_ohlcv(50, step=2.0, impulse_every=5, impulse_size=20.0)
        result = detect_all_zones(df)
        ob_zones = [z for z in result if "OB" in z["type"]]
        assert len(ob_zones) >= 1

    def test_uses_fvg_fixture(self, df_with_fvg):
        result = detect_all_zones(df_with_fvg.reset_index())
        fvg_zones = [z for z in result if "FVG" in z["type"]]
        assert len(fvg_zones) >= 1

    def test_empty_df_returns_empty(self):
        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close"])
        result = detect_all_zones(empty)
        assert result == []

    def test_short_df_less_than_5_returns_empty_or_minimal(self):
        df = _make_flat_ohlcv(4)
        result = detect_all_zones(df)
        # With 4 bars, FVG needs 3 (could fire), OB needs 10 (won't fire), BOS needs 15 (won't fire)
        # But flat data has no gap > 2pts, so effectively empty
        assert isinstance(result, list)

    def test_zone_count_equals_sum_of_individual_detectors(self):
        df = _make_bos_ohlcv()
        fvg = detect_fvg(df)
        ob = detect_ob(df)
        bos = detect_bos(df)
        all_z = detect_all_zones(df)
        assert len(all_z) == len(fvg) + len(ob) + len(bos)

    def test_custom_params_passed_through(self):
        df = _make_trending_ohlcv(50, step=2.0, impulse_every=5, impulse_size=20.0)
        result_default = detect_all_zones(df)
        result_custom = detect_all_zones(df, min_fvg_gap=0.5, ob_impulse_mult=1.0, bos_lookback=3)
        # Both should return lists; custom should be >= default (looser thresholds)
        assert isinstance(result_default, list)
        assert isinstance(result_custom, list)

    def test_all_zone_types_are_known(self, df_m15_bullish):
        known_types = {"BULL_FVG", "BEAR_FVG", "BULL_OB", "BEAR_OB", "BOS_BULL", "BOS_BEAR"}
        result = detect_all_zones(df_m15_bullish.reset_index())
        for z in result:
            assert z["type"] in known_types, f"Unknown zone type: {z['type']}"


# ── merge_zones ──────────────────────────────────────────────────────────────

class TestMergeZones:

    def _make_zone(self, zone_type: str, low: float, detected_at: str = "2026-02-10T08:00:00") -> dict:
        return {
            "type": zone_type,
            "low": low,
            "high": low + 5.0,
            "level": None,
            "detected_at": detected_at,
            "source": "detector",
        }

    def _make_bos_zone(self, zone_type: str, level: float, detected_at: str = "2026-02-10T08:00:00") -> dict:
        return {
            "type": zone_type,
            "low": None,
            "high": None,
            "level": level,
            "detected_at": detected_at,
            "source": "detector",
        }

    def test_returns_list(self):
        result = merge_zones([], [])
        assert isinstance(result, list)

    def test_empty_both_returns_empty(self):
        result = merge_zones([], [])
        assert result == []

    def test_only_cached_returns_cached(self):
        cached = [self._make_zone("BULL_OB", 2900.0)]
        result = merge_zones([], cached)
        assert len(result) == 1
        assert result[0]["type"] == "BULL_OB"

    def test_only_detected_returns_detected(self):
        detected = [self._make_zone("BEAR_OB", 2950.0)]
        result = merge_zones(detected, [])
        assert len(result) == 1
        assert result[0]["type"] == "BEAR_OB"

    def test_same_zone_in_both_appears_once(self):
        """Identical zone type at same price in both detected and cached → only one in output."""
        zone = self._make_zone("BULL_OB", 2900.0)
        # Small price difference within tolerance (default 2.0)
        zone_close = self._make_zone("BULL_OB", 2900.5)
        result = merge_zones([zone_close], [zone])
        bull_ob_zones = [z for z in result if z["type"] == "BULL_OB"]
        assert len(bull_ob_zones) == 1, (
            f"Expected 1 BULL_OB (dedup), got {len(bull_ob_zones)}"
        )

    def test_different_zones_both_kept(self):
        """Different zone types at same price → both kept."""
        detected = [self._make_zone("BEAR_OB", 2900.0)]
        cached = [self._make_zone("BULL_OB", 2900.0)]
        result = merge_zones(detected, cached)
        assert len(result) == 2

    def test_same_type_far_apart_both_kept(self):
        """Same type but price far apart (> tolerance) → both kept."""
        detected = [self._make_zone("BULL_OB", 2800.0)]
        cached = [self._make_zone("BULL_OB", 2900.0)]
        result = merge_zones(detected, cached)
        bull_obs = [z for z in result if z["type"] == "BULL_OB"]
        assert len(bull_obs) == 2

    def test_cached_takes_priority_on_dup(self):
        """When a zone is deduplicated, the cached version is kept (not detected)."""
        cached_zone = self._make_zone("BULL_OB", 2900.0)
        cached_zone["source"] = "cache"
        detected_zone = self._make_zone("BULL_OB", 2900.5)
        detected_zone["source"] = "detector"
        result = merge_zones([detected_zone], [cached_zone])
        bull_obs = [z for z in result if z["type"] == "BULL_OB"]
        assert len(bull_obs) == 1
        assert bull_obs[0]["source"] == "cache"

    def test_multiple_detected_multiple_cached(self):
        """3 detected + 3 cached with 1 overlap → 5 zones total."""
        cached = [
            self._make_zone("BULL_OB", 2900.0),
            self._make_zone("BEAR_OB", 2950.0),
            self._make_bos_zone("BOS_BULL", 2920.0),
        ]
        detected = [
            self._make_zone("BULL_OB", 2900.3),   # dup of cached[0]
            self._make_zone("BULL_FVG", 2880.0),   # unique
            self._make_bos_zone("BOS_BEAR", 2870.0),  # unique
        ]
        result = merge_zones(detected, cached)
        assert len(result) == 5

    def test_bos_zone_dedup_by_level(self):
        """BOS zones use 'level' for dedup reference (low is None)."""
        cached_bos = self._make_bos_zone("BOS_BULL", 2920.0)
        detected_bos = self._make_bos_zone("BOS_BULL", 2920.5)  # within tolerance
        result = merge_zones([detected_bos], [cached_bos])
        bos_zones = [z for z in result if z["type"] == "BOS_BULL"]
        assert len(bos_zones) == 1

    def test_custom_tolerance_tight(self):
        """With tight tolerance (0.1), slightly different zones both kept."""
        detected = [self._make_zone("BULL_OB", 2900.5)]
        cached = [self._make_zone("BULL_OB", 2900.0)]
        result = merge_zones(detected, cached, dedup_tolerance=0.1)
        bull_obs = [z for z in result if z["type"] == "BULL_OB"]
        assert len(bull_obs) == 2

    def test_custom_tolerance_wide(self):
        """With wide tolerance, zones 3 pts apart are deduped."""
        detected = [self._make_zone("BULL_OB", 2903.0)]
        cached = [self._make_zone("BULL_OB", 2900.0)]
        result = merge_zones(detected, cached, dedup_tolerance=5.0)
        bull_obs = [z for z in result if z["type"] == "BULL_OB"]
        assert len(bull_obs) == 1

    def test_empty_detected_returns_all_cached(self, sample_zones):
        result = merge_zones([], sample_zones)
        assert len(result) == len(sample_zones)

    def test_output_zones_have_required_keys(self, sample_zones):
        new_zone = {
            "type": "BULL_FVG",
            "low": 2800.0,
            "high": 2810.0,
            "level": None,
            "detected_at": "2026-02-10T10:00:00",
            "source": "detector",
        }
        result = merge_zones([new_zone], sample_zones)
        _assert_zone_structure(result)

    def test_preserves_zone_data_intact(self):
        """Merged zones retain all original fields."""
        cached_zone = {
            "type": "BULL_OB",
            "low": 2895.0,
            "high": 2900.0,
            "level": None,
            "detected_at": "2026-02-10T08:00:00",
            "source": "cache",
            "extra_field": "preserved",
        }
        result = merge_zones([], [cached_zone])
        assert result[0].get("extra_field") == "preserved"

    def test_does_not_mutate_inputs(self):
        """merge_zones should not modify the original detected or cached lists."""
        detected = [self._make_zone("BULL_FVG", 2880.0)]
        cached = [self._make_zone("BULL_OB", 2900.0)]
        detected_copy = list(detected)
        cached_copy = list(cached)
        merge_zones(detected, cached)
        assert detected == detected_copy
        assert cached == cached_copy

    def test_all_detected_new_appended(self):
        """When no overlaps, all detected zones appear in output alongside cached."""
        detected = [
            self._make_zone("BULL_FVG", 2850.0),
            self._make_zone("BEAR_FVG", 2960.0),
        ]
        cached = [self._make_zone("BULL_OB", 2900.0)]
        result = merge_zones(detected, cached)
        assert len(result) == 3
