"""
Technical indicators — all calculated in Python, no Claude needed.
RSI, ATR, Premium/Discount, OTE, M15 confirmation.
"""
import numpy as np
import pandas as pd
from typing import Optional


def atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range."""
    if len(df) < period + 1:
        return 0.0
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def rsi(df: pd.DataFrame, period: int = 14) -> float:
    """Relative Strength Index."""
    if len(df) < period + 1:
        return 50.0
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def premium_discount(df_h1: pd.DataFrame, lookback: int = 50) -> str:
    """
    Determines if price is in Premium (above 50%) or Discount (below 50%)
    of the recent H1 swing range.
    """
    recent     = df_h1.tail(min(lookback, len(df_h1)))
    swing_high = recent["high"].max()
    swing_low  = recent["low"].min()
    midpoint   = (swing_high + swing_low) / 2
    current    = float(df_h1["close"].iloc[-1])

    if current > midpoint + 0.5:
        return "PREMIUM"
    elif current < midpoint - 0.5:
        return "DISCOUNT"
    return "EQUILIBRIUM"


def ote_zone(df_h1: pd.DataFrame, direction: str) -> Optional[tuple[float, float]]:
    """
    Finds the OTE (Optimal Trade Entry) zone = 61.8–79% retracement
    of the most recent impulse swing that created a BOS.
    Returns (ote_low, ote_high) or None.
    """
    n = len(df_h1)
    if n < 10:
        return None

    highs  = df_h1["high"].values
    lows   = df_h1["low"].values
    window = min(30, n)

    if direction == "LONG":
        # Look for last swing low → swing high (bullish impulse)
        sl_idx = int(np.argmin(lows[-window:])) + (n - window)
        sh_idx = int(np.argmax(highs[sl_idx:])) + sl_idx
        if sh_idx <= sl_idx:
            return None
        rng      = highs[sh_idx] - lows[sl_idx]
        ote_low  = highs[sh_idx] - rng * 0.79
        ote_high = highs[sh_idx] - rng * 0.618
    else:
        # Look for last swing high → swing low (bearish impulse)
        sh_idx = int(np.argmax(highs[-window:])) + (n - window)
        sl_idx = int(np.argmin(lows[sh_idx:])) + sh_idx
        if sl_idx <= sh_idx:
            return None
        rng      = highs[sh_idx] - lows[sl_idx]
        ote_low  = lows[sl_idx] + rng * 0.618
        ote_high = lows[sl_idx] + rng * 0.79

    if ote_low >= ote_high or ote_low <= 0:
        return None
    return (round(ote_low, 2), round(ote_high, 2))


def m15_confirmation(df_m15: pd.DataFrame, direction: str) -> Optional[str]:
    """
    Detects bullish/bearish CHoCH or engulfing on M15.
    Returns signal name or None.
    """
    if len(df_m15) < 6:
        return None

    last = df_m15.iloc[-1]
    prev = df_m15.iloc[-2]
    body_last = abs(last["close"] - last["open"])
    body_prev = abs(prev["close"] - prev["open"])

    if direction == "LONG":
        # Bullish engulfing
        if (last["close"] > last["open"]
                and last["open"] <= prev["close"]
                and last["close"] >= prev["open"]
                and body_last > body_prev * 0.8):
            return "BULL_ENGULFING"
        # Bullish CHoCH: close above recent swing high
        swing_high = df_m15["high"].iloc[-6:-1].max()
        if last["close"] > swing_high:
            return "BULL_CHOCH"
    else:
        # Bearish engulfing
        if (last["close"] < last["open"]
                and last["open"] >= prev["close"]
                and last["close"] <= prev["open"]
                and body_last > body_prev * 0.8):
            return "BEAR_ENGULFING"
        # Bearish CHoCH: close below recent swing low
        swing_low = df_m15["low"].iloc[-6:-1].min()
        if last["close"] < swing_low:
            return "BEAR_CHOCH"

    return None


def ema(df: pd.DataFrame, period: int = 50) -> float:
    """Exponential Moving Average — last value."""
    if len(df) < period:
        return 0.0
    return float(df["close"].ewm(span=period, adjust=False).mean().iloc[-1])


def h1_ema_trend(df_h1: pd.DataFrame, period: int = 50) -> str:
    """
    H1 trend direction from EMA slope + price position.
    Returns BULLISH, BEARISH, or NEUTRAL.
    """
    if len(df_h1) < period + 5:
        return "NEUTRAL"

    ema_series = df_h1["close"].ewm(span=period, adjust=False).mean()
    ema_now = float(ema_series.iloc[-1])
    ema_5ago = float(ema_series.iloc[-6])
    price = float(df_h1["close"].iloc[-1])

    slope = ema_now - ema_5ago  # positive = uptrend
    above_ema = price > ema_now

    if slope > 1.0 and above_ema:
        return "BULLISH"
    if slope < -1.0 and not above_ema:
        return "BEARISH"
    return "NEUTRAL"


def h4_bias(df_h4: pd.DataFrame, lookback: int = 20) -> str:
    """
    H4 bias from HH/HL or LH/LL sequence.
    Returns BULLISH, BEARISH, or RANGING.
    """
    if len(df_h4) < 6:
        return "RANGING"

    recent = df_h4.tail(min(lookback, len(df_h4)))
    highs  = recent["high"].values
    lows   = recent["low"].values

    # Find local swings (simplified)
    hh = highs[-1] > highs[-3] and highs[-3] > highs[-5]
    hl = lows[-1]  > lows[-3]  and lows[-3]  > lows[-5]
    lh = highs[-1] < highs[-3] and highs[-3] < highs[-5]
    ll = lows[-1]  < lows[-3]  and lows[-3]  < lows[-5]

    if hh and hl:
        return "BULLISH"
    if lh and ll:
        return "BEARISH"
    return "RANGING"


def daily_range_consumed(df_h1: pd.DataFrame, threshold: float = 1.20) -> bool:
    """
    Check if today's price range already exceeds threshold × average daily range.
    Post-impulse filter: skip entry when daily range is exhausted.
    """
    if len(df_h1) < 48:  # need at least 2 days of H1 data
        return False

    # Approximate today's range from last ~24 H1 candles
    today = df_h1.tail(24)
    today_range = float(today["high"].max() - today["low"].min())

    # Average daily range from prior bars (approx: rolling 24-bar windows)
    prior = df_h1.iloc[:-24]
    if len(prior) < 24:
        return False
    # Simple avg: total range / number of days
    total_bars = len(prior)
    n_days = max(1, total_bars // 24)
    daily_ranges = []
    for d in range(n_days):
        start = d * 24
        end = min(start + 24, total_bars)
        chunk = prior.iloc[start:end]
        if len(chunk) >= 12:
            daily_ranges.append(float(chunk["high"].max() - chunk["low"].min()))
    if not daily_ranges:
        return False

    avg_daily = sum(daily_ranges) / len(daily_ranges)
    return today_range > avg_daily * threshold


def count_signals(
    direction: str,
    zones_hit: list[str],
    m15_conf: Optional[str],
    ote: Optional[tuple],
    price: float,
    pd_zone: str,
    h1_structure: str,
) -> tuple[int, list[str]]:
    """
    Count SMC confluence signals.
    Returns (count, signal_list).
    """
    signals = []

    # Check ALL zone types for structure + zone signals
    all_types = zones_hit + [h1_structure]
    for z in all_types:
        zu = z.upper()
        if "BOS" in zu and "BOS" not in signals:
            signals.append("BOS")
        if "CHOCH" in zu and "CHoCH" not in signals:
            signals.append("CHoCH")
        if "OB" in zu and "OB" not in signals and "BOS" not in zu:
            signals.append("OB")
        if "FVG" in zu and "FVG" not in signals:
            signals.append("FVG")
        if "BREAKER" in zu and "Breaker" not in signals:
            signals.append("Breaker")
        if "LIQ" in zu and "LiqSweep" not in signals:
            signals.append("LiqSweep")

    # M15 confirmation bonus
    if m15_conf:
        signals.append("M15")

    # OTE bonus
    if ote and ote[0] <= price <= ote[1]:
        signals.append("OTE")

    # Premium/Discount bonus
    if direction == "LONG" and pd_zone == "DISCOUNT":
        signals.append("Discount")
    elif direction == "SHORT" and pd_zone == "PREMIUM":
        signals.append("Premium")

    # Remove duplicates, keep order
    seen = set()
    unique = []
    for s in signals:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return len(unique), unique
