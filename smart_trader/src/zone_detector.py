"""
Zone Detector — detect SMC zones (FVG, OB, BOS) from H1 candle data.
Self-contained: no dependency on claude_trader cache.
Refreshes every scan cycle with latest H1 candles.
"""
import pandas as pd
from typing import Optional
from loguru import logger


# ── FVG Detection ────────────────────────────────────────────────────────────

def detect_fvg(df: pd.DataFrame, min_gap_pts: float = 2.0) -> list[dict]:
    """
    Detect Fair Value Gaps from 3-candle patterns on H1.
    Bull FVG: candle[i-2].high < candle[i].low  (gap up)
    Bear FVG: candle[i-2].low  > candle[i].high (gap down)
    Returns list of zone dicts with type, low, high.
    """
    zones = []
    if len(df) < 3:
        return zones

    highs = df["high"].values
    lows = df["low"].values
    times = df["time"].values if "time" in df.columns else [None] * len(df)

    for i in range(2, len(df)):
        # Bull FVG: gap between candle i-2 high and candle i low
        if lows[i] > highs[i - 2] + min_gap_pts:
            zones.append({
                "type": "BULL_FVG",
                "low": float(highs[i - 2]),
                "high": float(lows[i]),
                "level": None,
                "detected_at": str(times[i]) if times[i] is not None else "",
                "source": "detector",
            })

        # Bear FVG: gap between candle i low and candle i-2 high
        if highs[i] < lows[i - 2] - min_gap_pts:
            zones.append({
                "type": "BEAR_FVG",
                "low": float(highs[i]),
                "high": float(lows[i - 2]),
                "level": None,
                "detected_at": str(times[i]) if times[i] is not None else "",
                "source": "detector",
            })

    # Only keep recent FVGs (last 10)
    return zones[-10:]


# ── Order Block Detection ────────────────────────────────────────────────────

def detect_ob(df: pd.DataFrame, impulse_mult: float = 1.5) -> list[dict]:
    """
    Detect Order Blocks: last opposing candle before a strong impulse move.
    Bull OB: last bearish candle before bullish impulse (close[i] - open[i] > ATR * mult)
    Bear OB: last bullish candle before bearish impulse
    """
    zones = []
    if len(df) < 10:
        return zones

    opens = df["open"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    times = df["time"].values if "time" in df.columns else [None] * len(df)

    # Simple ATR approximation (average range of last 14 candles)
    ranges = highs - lows
    avg_range = ranges[-14:].mean() if len(ranges) >= 14 else ranges.mean()
    impulse_threshold = avg_range * impulse_mult

    for i in range(1, len(df)):
        body = closes[i] - opens[i]

        # Bullish impulse: find last bearish candle before it
        if body > impulse_threshold:
            for j in range(i - 1, max(i - 5, -1), -1):
                if closes[j] < opens[j]:  # bearish candle
                    zones.append({
                        "type": "BULL_OB",
                        "low": float(lows[j]),
                        "high": float(highs[j]),
                        "level": None,
                        "detected_at": str(times[j]) if times[j] is not None else "",
                        "source": "detector",
                    })
                    break

        # Bearish impulse: find last bullish candle before it
        if body < -impulse_threshold:
            for j in range(i - 1, max(i - 5, -1), -1):
                if closes[j] > opens[j]:  # bullish candle
                    zones.append({
                        "type": "BEAR_OB",
                        "low": float(lows[j]),
                        "high": float(highs[j]),
                        "level": None,
                        "detected_at": str(times[j]) if times[j] is not None else "",
                        "source": "detector",
                    })
                    break

    return zones[-8:]


# ── BOS / CHoCH Detection ───────────────────────────────────────────────────

def _swing_points(highs, lows, lookback: int = 5) -> tuple[list, list]:
    """Find swing highs and swing lows."""
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(highs) - lookback):
        # Swing high: highest in window
        if highs[i] == max(highs[i - lookback:i + lookback + 1]):
            swing_highs.append((i, float(highs[i])))
        # Swing low: lowest in window
        if lows[i] == min(lows[i - lookback:i + lookback + 1]):
            swing_lows.append((i, float(lows[i])))

    return swing_highs, swing_lows


def detect_bos(df: pd.DataFrame, lookback: int = 5) -> list[dict]:
    """
    Detect Break of Structure (BOS) levels.
    Bull BOS: price breaks above previous swing high
    Bear BOS: price breaks below previous swing low
    """
    zones = []
    if len(df) < lookback * 3:
        return zones

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    times = df["time"].values if "time" in df.columns else [None] * len(df)

    swing_highs, swing_lows = _swing_points(highs, lows, lookback)

    # Check if recent candles broke swing levels
    for idx, level in swing_highs:
        # Look for break above this swing high
        for i in range(idx + 1, min(idx + lookback * 3, len(df))):
            if closes[i] > level:
                zones.append({
                    "type": "BOS_BULL",
                    "low": None,
                    "high": None,
                    "level": level,
                    "detected_at": str(times[i]) if times[i] is not None else "",
                    "source": "detector",
                })
                break

    for idx, level in swing_lows:
        for i in range(idx + 1, min(idx + lookback * 3, len(df))):
            if closes[i] < level:
                zones.append({
                    "type": "BOS_BEAR",
                    "low": None,
                    "high": None,
                    "level": level,
                    "detected_at": str(times[i]) if times[i] is not None else "",
                    "source": "detector",
                })
                break

    return zones[-8:]


# ── Main Detection Entry Point ───────────────────────────────────────────────

def detect_all_zones(
    df_h1: pd.DataFrame,
    min_fvg_gap: float = 2.0,
    ob_impulse_mult: float = 1.5,
    bos_lookback: int = 5,
) -> list[dict]:
    """
    Run all zone detectors on H1 data.
    Returns merged list of all detected zones.
    """
    zones = []
    zones.extend(detect_fvg(df_h1, min_fvg_gap))
    zones.extend(detect_ob(df_h1, ob_impulse_mult))
    zones.extend(detect_bos(df_h1, bos_lookback))

    logger.debug(
        f"Zone detector: {len(zones)} zones "
        f"(FVG={sum(1 for z in zones if 'FVG' in z['type'])}, "
        f"OB={sum(1 for z in zones if 'OB' in z['type'])}, "
        f"BOS={sum(1 for z in zones if 'BOS' in z['type'])})"
    )
    return zones


def merge_zones(
    detected: list[dict],
    cached: list[dict],
    dedup_tolerance: float = 2.0,
) -> list[dict]:
    """
    Merge detected zones with claude_trader cached zones.
    Dedup by proximity (if two zones of same type are within tolerance, keep one).
    """
    all_zones = list(cached)  # cache takes priority

    for dz in detected:
        dz_type = dz.get("type", "")
        dz_ref = dz.get("low") or dz.get("level") or 0

        # Check if similar zone already in cached
        is_dup = False
        for cz in cached:
            if cz.get("type") != dz_type:
                continue
            cz_ref = cz.get("low") or cz.get("level") or 0
            if abs(dz_ref - cz_ref) < dedup_tolerance:
                is_dup = True
                break

        if not is_dup:
            all_zones.append(dz)

    return all_zones
