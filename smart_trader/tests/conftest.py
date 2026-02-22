"""
Shared pytest fixtures for smart_trader test suite.
All fixtures use synthetic but realistic XAUUSD-like data.
No MT5 connection required.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── OHLCV Generators ─────────────────────────────────────────────────────────

def _make_ohlcv(n: int, base_price: float = 2900.0, trend: float = 0.0,
                volatility: float = 5.0, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with optional trend."""
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


@pytest.fixture
def df_m15_neutral():
    """100 bars M15 OHLCV — sideways market."""
    return _make_ohlcv(100, base_price=2900.0, trend=0.0, volatility=3.0)


@pytest.fixture
def df_m15_bullish():
    """100 bars M15 OHLCV — uptrend."""
    return _make_ohlcv(100, base_price=2880.0, trend=0.5, volatility=2.0)


@pytest.fixture
def df_m15_bearish():
    """100 bars M15 OHLCV — downtrend."""
    return _make_ohlcv(100, base_price=2950.0, trend=-0.5, volatility=2.0)


@pytest.fixture
def df_h1_bullish():
    """100 bars H1 OHLCV — strong uptrend for EMA trend detection."""
    return _make_ohlcv(100, base_price=2800.0, trend=2.0, volatility=4.0, seed=10)


@pytest.fixture
def df_h1_bearish():
    """100 bars H1 OHLCV — strong downtrend."""
    return _make_ohlcv(100, base_price=3000.0, trend=-2.0, volatility=4.0, seed=11)


@pytest.fixture
def df_h1_ranging():
    """100 bars H1 OHLCV — ranging."""
    return _make_ohlcv(100, base_price=2900.0, trend=0.0, volatility=3.0, seed=12)


@pytest.fixture
def df_with_fvg():
    """
    15 bars with a clear bullish FVG:
    bar[-3].high < bar[-1].low  → gap between them (FVG).
    """
    rows = []
    base = 2900.0
    now = datetime(2026, 2, 10, 8, 0, tzinfo=timezone.utc)
    prices = [(base + i * 2) for i in range(15)]
    for i, p in enumerate(prices):
        o = p - 1
        c = p + 1
        hi = p + 2
        lo = p - 2
        rows.append({"time": now + timedelta(minutes=15*i),
                     "open": o, "high": hi, "low": lo, "close": c,
                     "tick_volume": 200})
    df = pd.DataFrame(rows).set_index("time")

    # Force FVG: bar[-3].high < bar[-1].low
    df.iloc[-3, df.columns.get_loc("high")] = 2920.0
    df.iloc[-2, df.columns.get_loc("open")] = 2921.0
    df.iloc[-2, df.columns.get_loc("close")] = 2930.0
    df.iloc[-2, df.columns.get_loc("high")] = 2935.0
    df.iloc[-2, df.columns.get_loc("low")] = 2920.0
    df.iloc[-1, df.columns.get_loc("low")] = 2928.0
    df.iloc[-1, df.columns.get_loc("open")] = 2929.0
    df.iloc[-1, df.columns.get_loc("close")] = 2932.0
    df.iloc[-1, df.columns.get_loc("high")] = 2934.0
    return df


@pytest.fixture
def df_with_engulf_bull():
    """
    10 bars ending with a bullish engulfing on M15.
    bar[-1] fully engulfs bar[-2] (bearish).
    """
    rows = []
    now = datetime(2026, 2, 10, 8, 0, tzinfo=timezone.utc)
    base = 2900.0
    for i in range(8):
        rows.append({"time": now + timedelta(minutes=15*i),
                     "open": base, "high": base+3, "low": base-3,
                     "close": base+1, "tick_volume": 200})
    # bar[-2]: bearish candle
    rows.append({"time": now + timedelta(minutes=15*8),
                 "open": 2905.0, "high": 2907.0, "low": 2898.0,
                 "close": 2899.0, "tick_volume": 200})
    # bar[-1]: bullish engulfing (open < prev close, close > prev open)
    rows.append({"time": now + timedelta(minutes=15*9),
                 "open": 2897.0, "high": 2910.0, "low": 2896.0,
                 "close": 2908.0, "tick_volume": 500})
    return pd.DataFrame(rows).set_index("time")


# ── Zone Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_zones():
    """Mixed bull/bear zones for scanner tests."""
    return [
        {"type": "BULL_OB",      "low": 2895.0, "high": 2900.0, "level": None,   "detected_at": "2026-02-10T08:00:00"},
        {"type": "BULL_FVG",     "low": 2910.0, "high": 2915.0, "level": None,   "detected_at": "2026-02-10T08:15:00"},
        {"type": "BOS_BULL",     "low": None,   "high": None,   "level": 2920.0, "detected_at": "2026-02-10T09:00:00"},
        {"type": "BEAR_OB",      "low": 2950.0, "high": 2955.0, "level": None,   "detected_at": "2026-02-10T07:00:00"},
        {"type": "BOS_BEAR",     "low": None,   "high": None,   "level": 2880.0, "detected_at": "2026-02-10T06:00:00"},
        {"type": "BULL_BREAKER", "low": 2870.0, "high": 2875.0, "level": None,   "detected_at": "2026-02-10T05:00:00"},
    ]


@pytest.fixture
def mock_account():
    """Sample MT5 account dict."""
    return {
        "balance": 100.0,
        "equity": 98.0,
        "margin_free": 95.0,
        "profit": -2.0,
    }


@pytest.fixture
def mock_positions_empty():
    return []


class _FakeRaw:
    def __init__(self, magic): self.magic = magic

@pytest.fixture
def mock_positions_one_long():
    """One open LONG position from our bot (magic=202602)."""
    return [{"type": "LONG", "volume": 0.01, "profit": 1.5,
             "_raw": _FakeRaw(202602)}]


@pytest.fixture
def mock_positions_at_limit():
    """Max positions reached (1 LONG, max_positions=1)."""
    return [{"type": "LONG", "volume": 0.01, "profit": 1.5,
             "_raw": _FakeRaw(202602)}]
