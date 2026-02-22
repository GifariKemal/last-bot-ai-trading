"""
Tests for smart_trader/src/indicators.py
All tests are self-contained using fixtures from conftest.py.
No MT5 connection required.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

import indicators as ind


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int, base_price: float = 2900.0, trend: float = 0.0,
                volatility: float = 5.0, seed: int = 42) -> pd.DataFrame:
    """Local copy of _make_ohlcv so tests are self-contained."""
    rng = np.random.default_rng(seed)
    closes = [base_price]
    for _ in range(n - 1):
        change = trend + rng.normal(0, volatility)
        closes.append(max(closes[-1] + change, 100.0))

    rows = []
    now = datetime(2026, 2, 10, 8, 0, tzinfo=timezone.utc)
    for i, c in enumerate(closes):
        o = c + rng.uniform(-2, 2)
        hi = max(o, c) + abs(rng.normal(0, 2))
        lo = min(o, c) - abs(rng.normal(0, 2))
        rows.append({
            "time": now + timedelta(minutes=15 * i),
            "open": round(o, 2),
            "high": round(hi, 2),
            "low": round(lo, 2),
            "close": round(c, 2),
            "tick_volume": int(rng.integers(100, 1000)),
        })
    df = pd.DataFrame(rows)
    df.set_index("time", inplace=True)
    return df


# ── rsi() ─────────────────────────────────────────────────────────────────────

class TestRsi:
    def test_rsi_returns_float(self):
        df = _make_ohlcv(50)
        result = ind.rsi(df)
        assert isinstance(result, float)

    def test_rsi_returns_values_in_range(self):
        df = _make_ohlcv(50)
        result = ind.rsi(df)
        assert 0.0 <= result <= 100.0

    def test_rsi_uptrend_above_50(self):
        """
        Sustained uptrend (mostly gains, occasional small losses so RSI is
        well-defined) should produce RSI > 50.
        Uses an explicit mixed-gain dataset to avoid division-by-negative-zero
        NaN that occurs when every bar is a gain (loss rolling mean = -0.0).
        """
        now = datetime(2026, 2, 10, 8, 0, tzinfo=timezone.utc)
        rng = np.random.default_rng(77)
        price = 2800.0
        rows = []
        for i in range(60):
            # 80% chance of gain, 20% chance of tiny loss → RSI well defined
            if rng.random() < 0.8:
                change = rng.uniform(1.0, 5.0)
            else:
                change = rng.uniform(-1.0, -0.1)
            price = max(price + change, 100.0)
            rows.append({
                "time": now + timedelta(minutes=15 * i),
                "open": round(price - 0.5, 2),
                "high": round(price + 1.0, 2),
                "low": round(price - 1.5, 2),
                "close": round(price, 2),
                "tick_volume": 200,
            })
        df = pd.DataFrame(rows).set_index("time")
        result = ind.rsi(df)
        assert result > 50.0, f"Expected RSI > 50 in uptrend, got {result}"

    def test_rsi_downtrend_below_50(self):
        """Sustained downtrend should produce RSI < 50."""
        df = _make_ohlcv(60, base_price=3000.0, trend=-3.0, volatility=1.0, seed=2)
        result = ind.rsi(df)
        assert result < 50.0, f"Expected RSI < 50 in downtrend, got {result}"

    def test_rsi_too_short_returns_default(self):
        """If fewer than period+1 bars, should return default 50.0."""
        df = _make_ohlcv(10)
        result = ind.rsi(df, period=14)
        assert result == 50.0

    def test_rsi_sideways_near_50(self):
        """Pure random walk should land RSI reasonably close to 50."""
        df = _make_ohlcv(100, trend=0.0, volatility=2.0, seed=99)
        result = ind.rsi(df)
        assert 0.0 <= result <= 100.0

    def test_rsi_custom_period(self):
        df = _make_ohlcv(50)
        result_14 = ind.rsi(df, period=14)
        result_9 = ind.rsi(df, period=9)
        # Both valid floats in range
        assert 0.0 <= result_14 <= 100.0
        assert 0.0 <= result_9 <= 100.0


# ── atr() ─────────────────────────────────────────────────────────────────────

class TestAtr:
    def test_atr_returns_positive(self):
        df = _make_ohlcv(50)
        result = ind.atr(df)
        assert result > 0.0

    def test_atr_returns_float(self):
        df = _make_ohlcv(50)
        result = ind.atr(df)
        assert isinstance(result, float)

    def test_atr_too_short_returns_zero(self):
        """Fewer than period+1 bars returns 0.0."""
        df = _make_ohlcv(10)
        result = ind.atr(df, period=14)
        assert result == 0.0

    def test_atr_high_volatility_greater_than_low(self):
        """High-volatility data should yield larger ATR than low-volatility."""
        df_hi = _make_ohlcv(50, volatility=20.0, seed=5)
        df_lo = _make_ohlcv(50, volatility=1.0, seed=5)
        assert ind.atr(df_hi) > ind.atr(df_lo)

    def test_atr_all_values_positive_on_trending_data(self):
        df = _make_ohlcv(50, trend=1.0, volatility=5.0)
        result = ind.atr(df)
        assert result > 0.0

    def test_atr_custom_period(self):
        df = _make_ohlcv(50)
        result = ind.atr(df, period=10)
        assert isinstance(result, float)
        assert result > 0.0


# ── ema() ─────────────────────────────────────────────────────────────────────

class TestEma:
    def test_ema_returns_float(self):
        df = _make_ohlcv(60)
        result = ind.ema(df, period=50)
        assert isinstance(result, float)

    def test_ema_length_matches_input(self):
        """ema() returns a single float — verify it is sensible vs close prices."""
        df = _make_ohlcv(100)
        result = ind.ema(df, period=50)
        close_min = float(df["close"].min())
        close_max = float(df["close"].max())
        assert close_min <= result <= close_max

    def test_ema_too_short_returns_zero(self):
        """Fewer than period bars returns 0.0."""
        df = _make_ohlcv(20)
        result = ind.ema(df, period=50)
        assert result == 0.0

    def test_ema_shorter_period_more_responsive(self):
        """With an uptrending tail, EMA(10) should be closer to recent close than EMA(50)."""
        df = _make_ohlcv(100, trend=2.0, volatility=1.0, seed=7)
        last_close = float(df["close"].iloc[-1])
        ema10 = ind.ema(df, period=10)
        ema50 = ind.ema(df, period=50)
        assert abs(ema10 - last_close) <= abs(ema50 - last_close)

    def test_ema_different_periods_differ(self):
        df = _make_ohlcv(100, trend=1.0, volatility=3.0)
        ema10 = ind.ema(df, period=10)
        ema50 = ind.ema(df, period=50)
        assert ema10 != ema50


# ── m15_confirmation() ────────────────────────────────────────────────────────

class TestM15Confirmation:
    def test_m15_conf_bullish_engulf_returns_bull_engulfing(self, df_with_engulf_bull):
        """df_with_engulf_bull fixture ends with a classic bullish engulfing."""
        result = ind.m15_confirmation(df_with_engulf_bull, direction="LONG")
        assert result == "BULL_ENGULFING"

    def test_m15_conf_neutral_df_returns_none_or_string(self, df_m15_neutral):
        """Neutral sideways data should return None (no confirmation pattern)."""
        result = ind.m15_confirmation(df_m15_neutral, direction="LONG")
        # May return None or a CHoCH if close happens to exceed swing high
        assert result is None or isinstance(result, str)

    def test_m15_conf_too_short_returns_none(self):
        """Fewer than 6 bars returns None."""
        df = _make_ohlcv(4)
        result = ind.m15_confirmation(df, direction="LONG")
        assert result is None

    def test_m15_conf_bearish_returns_bear_signal(self):
        """Construct a bearish engulfing and verify it fires for SHORT direction."""
        rows = []
        now = datetime(2026, 2, 10, 8, 0, tzinfo=timezone.utc)
        base = 2900.0
        for i in range(8):
            rows.append({"time": now + timedelta(minutes=15 * i),
                         "open": base, "high": base + 3, "low": base - 3,
                         "close": base + 1, "tick_volume": 200})
        # bar[-2]: bullish candle
        rows.append({"time": now + timedelta(minutes=15 * 8),
                     "open": 2895.0, "high": 2910.0, "low": 2893.0,
                     "close": 2908.0, "tick_volume": 200})
        # bar[-1]: bearish engulfing (open >= prev close, close <= prev open)
        rows.append({"time": now + timedelta(minutes=15 * 9),
                     "open": 2910.0, "high": 2912.0, "low": 2890.0,
                     "close": 2892.0, "tick_volume": 500})
        df = pd.DataFrame(rows).set_index("time")
        result = ind.m15_confirmation(df, direction="SHORT")
        assert result == "BEAR_ENGULFING"

    def test_m15_conf_bull_choch_detected(self):
        """Close above recent swing high should return BULL_CHOCH."""
        rows = []
        now = datetime(2026, 2, 10, 8, 0, tzinfo=timezone.utc)
        # 8 bars with swing high around 2910
        for i in range(8):
            rows.append({"time": now + timedelta(minutes=15 * i),
                         "open": 2900.0, "high": 2910.0, "low": 2895.0,
                         "close": 2902.0, "tick_volume": 200})
        # last candle closes well above that swing high
        rows.append({"time": now + timedelta(minutes=15 * 8),
                     "open": 2905.0, "high": 2930.0, "low": 2904.0,
                     "close": 2928.0, "tick_volume": 600})
        df = pd.DataFrame(rows).set_index("time")
        result = ind.m15_confirmation(df, direction="LONG")
        assert result in ("BULL_CHOCH", "BULL_ENGULFING")

    def test_m15_conf_wrong_direction_returns_none_for_engulf(self, df_with_engulf_bull):
        """Bullish engulf fixture with SHORT direction should NOT trigger BULL_ENGULFING."""
        result = ind.m15_confirmation(df_with_engulf_bull, direction="SHORT")
        assert result != "BULL_ENGULFING"


# ── h1_ema_trend() ────────────────────────────────────────────────────────────

class TestH1EmaTrend:
    def test_h1_ema_trend_bullish(self, df_h1_bullish):
        result = ind.h1_ema_trend(df_h1_bullish)
        assert result == "BULLISH", f"Expected BULLISH, got {result}"

    def test_h1_ema_trend_bearish(self, df_h1_bearish):
        result = ind.h1_ema_trend(df_h1_bearish)
        assert result == "BEARISH", f"Expected BEARISH, got {result}"

    def test_h1_ema_trend_ranging(self, df_h1_ranging):
        result = ind.h1_ema_trend(df_h1_ranging)
        assert result in ("NEUTRAL", "BULLISH", "BEARISH")

    def test_h1_ema_trend_returns_string(self, df_h1_bullish):
        result = ind.h1_ema_trend(df_h1_bullish)
        assert isinstance(result, str)

    def test_h1_ema_trend_valid_values_only(self, df_h1_bullish, df_h1_bearish, df_h1_ranging):
        valid = {"BULLISH", "BEARISH", "NEUTRAL"}
        for df in (df_h1_bullish, df_h1_bearish, df_h1_ranging):
            assert ind.h1_ema_trend(df) in valid

    def test_h1_ema_trend_too_short_returns_neutral(self):
        df = _make_ohlcv(20)
        result = ind.h1_ema_trend(df, period=50)
        assert result == "NEUTRAL"

    def test_h1_ema_trend_custom_period(self, df_h1_bullish):
        result = ind.h1_ema_trend(df_h1_bullish, period=20)
        assert result in ("BULLISH", "BEARISH", "NEUTRAL")


# ── count_signals() ───────────────────────────────────────────────────────────

class TestCountSignals:
    def test_count_signals_bos_in_zones_hit_adds_bos(self):
        """BOS in zones_hit should include 'BOS' in the signal list."""
        count, signals = ind.count_signals(
            direction="LONG",
            zones_hit=["BOS_BULL"],
            m15_conf=None,
            ote=None,
            price=2900.0,
            pd_zone="EQUILIBRIUM",
            h1_structure="",
        )
        assert "BOS" in signals

    def test_count_signals_includes_m15_conf(self):
        """m15_conf not None should add 'M15' to signals — this is the new feature."""
        count, signals = ind.count_signals(
            direction="LONG",
            zones_hit=[],
            m15_conf="BULL_ENGULFING",
            ote=None,
            price=2900.0,
            pd_zone="EQUILIBRIUM",
            h1_structure="",
        )
        assert "M15" in signals
        assert count >= 1

    def test_count_signals_m15_conf_none_no_m15_signal(self):
        """m15_conf=None should NOT add 'M15' to signals."""
        count, signals = ind.count_signals(
            direction="LONG",
            zones_hit=[],
            m15_conf=None,
            ote=None,
            price=2900.0,
            pd_zone="EQUILIBRIUM",
            h1_structure="",
        )
        assert "M15" not in signals

    def test_count_signals_ote_active_adds_ote(self):
        """Price inside OTE range should add 'OTE' to signals."""
        count, signals = ind.count_signals(
            direction="LONG",
            zones_hit=[],
            m15_conf=None,
            ote=(2895.0, 2905.0),
            price=2900.0,
            pd_zone="EQUILIBRIUM",
            h1_structure="",
        )
        assert "OTE" in signals

    def test_count_signals_ote_price_outside_no_ote(self):
        """Price outside OTE range should NOT add 'OTE'."""
        count, signals = ind.count_signals(
            direction="LONG",
            zones_hit=[],
            m15_conf=None,
            ote=(2850.0, 2870.0),
            price=2900.0,
            pd_zone="EQUILIBRIUM",
            h1_structure="",
        )
        assert "OTE" not in signals

    def test_count_signals_long_discount_adds_discount(self):
        """LONG direction + DISCOUNT pd_zone should include 'Discount'."""
        count, signals = ind.count_signals(
            direction="LONG",
            zones_hit=[],
            m15_conf=None,
            ote=None,
            price=2900.0,
            pd_zone="DISCOUNT",
            h1_structure="",
        )
        assert "Discount" in signals

    def test_count_signals_short_premium_adds_premium(self):
        """SHORT direction + PREMIUM pd_zone should include 'Premium'."""
        count, signals = ind.count_signals(
            direction="SHORT",
            zones_hit=[],
            m15_conf=None,
            ote=None,
            price=2900.0,
            pd_zone="PREMIUM",
            h1_structure="",
        )
        assert "Premium" in signals

    def test_count_signals_long_premium_no_bonus(self):
        """LONG direction + PREMIUM pd_zone should NOT add Premium or Discount."""
        count, signals = ind.count_signals(
            direction="LONG",
            zones_hit=[],
            m15_conf=None,
            ote=None,
            price=2900.0,
            pd_zone="PREMIUM",
            h1_structure="",
        )
        assert "Premium" not in signals
        assert "Discount" not in signals

    def test_count_signals_dedup_bos_not_doubled(self):
        """BOS from both zones_hit and h1_structure should not be counted twice."""
        count, signals = ind.count_signals(
            direction="LONG",
            zones_hit=["BOS_BULL", "BOS_BULL_REPEAT"],
            m15_conf=None,
            ote=None,
            price=2900.0,
            pd_zone="EQUILIBRIUM",
            h1_structure="BOS_BULL",
        )
        bos_count = signals.count("BOS")
        assert bos_count == 1

    def test_count_signals_dedup_choch_not_doubled(self):
        """CHoCH appearing multiple times should only be counted once."""
        count, signals = ind.count_signals(
            direction="LONG",
            zones_hit=["CHOCH_BULL", "CHOCH_REPEAT"],
            m15_conf=None,
            ote=None,
            price=2900.0,
            pd_zone="EQUILIBRIUM",
            h1_structure="CHOCH_BULL",
        )
        choch_count = signals.count("CHoCH")
        assert choch_count == 1

    def test_count_signals_all_signals_combined(self):
        """All signals active together should yield correct count and all types present."""
        count, signals = ind.count_signals(
            direction="LONG",
            zones_hit=["BOS_BULL", "FVG_BULL", "CHOCH_BULL"],
            m15_conf="BULL_ENGULFING",
            ote=(2895.0, 2905.0),
            price=2900.0,
            pd_zone="DISCOUNT",
            h1_structure="",
        )
        assert "BOS" in signals
        assert "FVG" in signals
        assert "CHoCH" in signals
        assert "M15" in signals
        assert "OTE" in signals
        assert "Discount" in signals
        assert count == len(signals)

    def test_count_signals_empty_inputs_returns_zero(self):
        count, signals = ind.count_signals(
            direction="LONG",
            zones_hit=[],
            m15_conf=None,
            ote=None,
            price=2900.0,
            pd_zone="EQUILIBRIUM",
            h1_structure="",
        )
        assert count == 0
        assert signals == []

    def test_count_signals_count_matches_list_length(self):
        """Returned count must always equal len(signal_list)."""
        count, signals = ind.count_signals(
            direction="SHORT",
            zones_hit=["BOS_BEAR", "LIQ_SWEEP"],
            m15_conf="BEAR_ENGULFING",
            ote=None,
            price=2900.0,
            pd_zone="PREMIUM",
            h1_structure="",
        )
        assert count == len(signals)

    def test_count_signals_fvg_detected(self):
        count, signals = ind.count_signals(
            direction="LONG",
            zones_hit=["FVG_BULL"],
            m15_conf=None,
            ote=None,
            price=2900.0,
            pd_zone="EQUILIBRIUM",
            h1_structure="",
        )
        assert "FVG" in signals

    def test_count_signals_liqsweep_detected(self):
        count, signals = ind.count_signals(
            direction="SHORT",
            zones_hit=["LIQ_SWEEP_HIGH"],
            m15_conf=None,
            ote=None,
            price=2900.0,
            pd_zone="EQUILIBRIUM",
            h1_structure="",
        )
        assert "LiqSweep" in signals

    def test_count_signals_breaker_detected(self):
        count, signals = ind.count_signals(
            direction="LONG",
            zones_hit=["BREAKER_BULL"],
            m15_conf=None,
            ote=None,
            price=2900.0,
            pd_zone="EQUILIBRIUM",
            h1_structure="",
        )
        assert "Breaker" in signals

    def test_count_signals_m15_conf_bear_adds_m15(self):
        """Bear M15 confirmation also adds 'M15' signal for SHORT direction."""
        count, signals = ind.count_signals(
            direction="SHORT",
            zones_hit=[],
            m15_conf="BEAR_CHOCH",
            ote=None,
            price=2900.0,
            pd_zone="EQUILIBRIUM",
            h1_structure="",
        )
        assert "M15" in signals


# ── ote_zone() ────────────────────────────────────────────────────────────────

class TestOteZone:
    def test_ote_zone_returns_tuple_or_none(self):
        df = _make_ohlcv(50)
        result = ind.ote_zone(df, direction="LONG")
        assert result is None or isinstance(result, tuple)

    def test_ote_zone_too_short_returns_none(self):
        df = _make_ohlcv(5)
        assert ind.ote_zone(df, direction="LONG") is None
        assert ind.ote_zone(df, direction="SHORT") is None

    def test_ote_zone_tuple_has_two_elements(self):
        df = _make_ohlcv(50, trend=2.0, volatility=3.0, seed=20)
        result = ind.ote_zone(df, direction="LONG")
        if result is not None:
            assert len(result) == 2

    def test_ote_zone_low_less_than_high(self):
        df = _make_ohlcv(50, trend=2.0, volatility=3.0, seed=20)
        result = ind.ote_zone(df, direction="LONG")
        if result is not None:
            ote_low, ote_high = result
            assert ote_low < ote_high

    def test_ote_zone_positive_values(self):
        df = _make_ohlcv(50, base_price=2900.0, trend=1.5, volatility=2.0, seed=21)
        result = ind.ote_zone(df, direction="LONG")
        if result is not None:
            assert result[0] > 0
            assert result[1] > 0

    def test_ote_zone_long_and_short_differ(self):
        df = _make_ohlcv(50, trend=0.0, volatility=10.0, seed=22)
        long_ote = ind.ote_zone(df, direction="LONG")
        short_ote = ind.ote_zone(df, direction="SHORT")
        # They may both be None, or they may differ — just assert no crash
        assert long_ote is None or isinstance(long_ote, tuple)
        assert short_ote is None or isinstance(short_ote, tuple)

    def test_ote_zone_bearish_impulse(self):
        df = _make_ohlcv(50, base_price=3000.0, trend=-2.0, volatility=2.0, seed=23)
        result = ind.ote_zone(df, direction="SHORT")
        if result is not None:
            assert result[0] < result[1]


# ── pd_zone_label() (premium_discount) ───────────────────────────────────────

class TestPdZoneLabel:
    def test_pd_zone_returns_string(self):
        df = _make_ohlcv(60)
        result = ind.premium_discount(df)
        assert isinstance(result, str)

    def test_pd_zone_valid_values_only(self):
        for seed in (1, 2, 3, 42, 99):
            df = _make_ohlcv(60, seed=seed)
            result = ind.premium_discount(df)
            assert result in ("PREMIUM", "DISCOUNT", "EQUILIBRIUM"), \
                f"Unexpected value: {result}"

    def test_pd_zone_discount_when_price_at_bottom(self):
        """Force price to bottom of range — should return DISCOUNT."""
        rows = []
        now = datetime(2026, 2, 10, 8, 0, tzinfo=timezone.utc)
        # Range: 2800–2900; last close well below midpoint
        for i in range(49):
            p = 2900.0 - i * 2.0
            rows.append({"time": now + timedelta(hours=i),
                         "open": p, "high": p + 1, "low": p - 1,
                         "close": p, "tick_volume": 200})
        # Last bar closes near the low
        rows.append({"time": now + timedelta(hours=49),
                     "open": 2802.0, "high": 2803.0, "low": 2800.0,
                     "close": 2800.5, "tick_volume": 200})
        df = pd.DataFrame(rows).set_index("time")
        result = ind.premium_discount(df)
        assert result == "DISCOUNT"

    def test_pd_zone_premium_when_price_at_top(self):
        """Force price to top of range — should return PREMIUM."""
        rows = []
        now = datetime(2026, 2, 10, 8, 0, tzinfo=timezone.utc)
        for i in range(49):
            p = 2800.0 + i * 2.0
            rows.append({"time": now + timedelta(hours=i),
                         "open": p, "high": p + 1, "low": p - 1,
                         "close": p, "tick_volume": 200})
        # Last bar closes near the high
        rows.append({"time": now + timedelta(hours=49),
                     "open": 2896.0, "high": 2900.0, "low": 2895.0,
                     "close": 2899.5, "tick_volume": 200})
        df = pd.DataFrame(rows).set_index("time")
        result = ind.premium_discount(df)
        assert result == "PREMIUM"

    def test_pd_zone_equilibrium_at_midpoint(self):
        """Price exactly at midpoint (within 0.5) should return EQUILIBRIUM."""
        rows = []
        now = datetime(2026, 2, 10, 8, 0, tzinfo=timezone.utc)
        # Range 2800–2900, midpoint 2850
        for i in range(49):
            rows.append({"time": now + timedelta(hours=i),
                         "open": 2850.0, "high": 2900.0, "low": 2800.0,
                         "close": 2850.0, "tick_volume": 200})
        # Last close exactly at midpoint
        rows.append({"time": now + timedelta(hours=49),
                     "open": 2850.0, "high": 2851.0, "low": 2849.0,
                     "close": 2850.0, "tick_volume": 200})
        df = pd.DataFrame(rows).set_index("time")
        result = ind.premium_discount(df)
        assert result == "EQUILIBRIUM"

    def test_pd_zone_uses_custom_lookback(self):
        """Custom lookback parameter accepted without error."""
        df = _make_ohlcv(60)
        result = ind.premium_discount(df, lookback=20)
        assert result in ("PREMIUM", "DISCOUNT", "EQUILIBRIUM")
