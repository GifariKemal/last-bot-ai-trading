"""
Market context builder for LLM prompt construction.

Extracts the latest indicator snapshot and last 20 OHLCV bars from the
feature-enriched DataFrame and returns a clean dict ready for the Kimi prompt.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

# Candle pattern columns and their readable labels
PATTERN_LABELS: dict[str, str] = {
    "cdl_hammer":        "Hammer (bullish reversal)",
    "cdl_engulfing":     "Engulfing (momentum shift)",
    "cdl_shooting_star": "Shooting Star (bearish rejection)",
}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def build_context(df: pd.DataFrame, recent_bars: int = 20) -> dict:
    """
    Build market context dict from feature-enriched DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of FeatureEngineer.build_features() — must have all indicator columns.
    recent_bars : int
        Number of recent M15 bars to include in the OHLCV table.

    Returns
    -------
    dict with all values needed for KimiPredictor._build_prompt().
    """
    clean = df.dropna(subset=["rsi_14", "adx_14", "atr_14"])
    if len(clean) < recent_bars:
        raise ValueError(
            f"Insufficient clean rows ({len(clean)}) for context "
            f"(need at least {recent_bars})."
        )

    latest    = clean.iloc[-1]
    bar_slice = clean.iloc[-recent_bars:]

    # ── Recent bars table ──────────────────────────────────────────────────
    bars_list = []
    for ts, row in bar_slice.iterrows():
        bars_list.append({
            "time":   ts.strftime("%Y-%m-%d %H:%M"),
            "open":   float(row["open"]),
            "high":   float(row["high"]),
            "low":    float(row["low"]),
            "close":  float(row["close"]),
            "volume": float(row.get("tick_volume", 0)),
        })

    # ── Active session labels ──────────────────────────────────────────────
    active_sessions: list[str] = []
    if latest.get("is_overlap", 0):
        active_sessions.append("London+NY Overlap (high liquidity)")
    else:
        if latest.get("is_london",   0): active_sessions.append("London")
        if latest.get("is_new_york", 0): active_sessions.append("New York")
    if latest.get("is_asian", 0):        active_sessions.append("Asian")

    # ── Active candle patterns ─────────────────────────────────────────────
    patterns_found: list[str] = []
    for col, label in PATTERN_LABELS.items():
        val = latest.get(col, 0)
        if val == 1:
            patterns_found.append(f"BULLISH {label}")
        elif val == -1:
            patterns_found.append(f"BEARISH {label}")
    patterns_str = " | ".join(patterns_found) if patterns_found else ""

    # ── Day of week ────────────────────────────────────────────────────────
    dow_idx = int(latest.get("day_of_week", latest.name.dayofweek))
    day_name = DAY_NAMES[dow_idx] if dow_idx < len(DAY_NAMES) else str(dow_idx)

    ctx = {
        # timing
        "time":            latest.name.strftime("%Y-%m-%d %H:%M UTC"),
        "day_of_week":     day_name,
        "active_sessions": active_sessions,
        # recent bars
        "recent_bars":     bars_list,
        # momentum
        "rsi":             float(latest["rsi_14"]),
        "macd":            float(latest["macd"]),
        "macd_signal":     float(latest["macd_signal"]),
        "macd_hist":       float(latest["macd_hist"]),
        "stoch_k":         float(latest["stoch_k"]),
        "stoch_d":         float(latest["stoch_d"]),
        "cci":             float(latest["cci_20"]),
        # trend
        "ema50_dist":      float(latest["close_vs_ema50"]),
        "ema200_dist":     float(latest["close_vs_ema200"]),
        # regime
        "adx":             float(latest["adx_14"]),
        "di_diff":         float(latest["di_diff"]),
        "atr":             float(latest["atr_14"]),
        "atr_regime":      float(latest["atr_regime"]),
        # volatility
        "bb_pct":          float(latest["bb_pct"]),
        "bb_width":        float(latest["bb_width"]),
        # HTF
        "h1_rsi":          float(latest.get("h1_rsi",      50.0)),
        "h1_adx":          float(latest.get("h1_adx",      20.0)),
        "h1_bb_pos":       float(latest.get("h1_bb_pos",    0.5)),
        "h1_ema_bias":     int(latest.get("h1_ema50_bias",   0)),
        "h4_rsi":          float(latest.get("h4_rsi",      50.0)),
        "h4_adx":          float(latest.get("h4_adx",      20.0)),
        "h4_bb_pos":       float(latest.get("h4_bb_pos",    0.5)),
        "h4_ema_bias":     int(latest.get("h4_ema50_bias",   0)),
        # session H/L proximity
        "dist_high_8h":    float(latest.get("dist_to_high_8h",  0.0)),
        "dist_low_8h":     float(latest.get("dist_to_low_8h",   0.0)),
        "dist_high_24h":   float(latest.get("dist_to_high_24h", 0.0)),
        "dist_low_24h":    float(latest.get("dist_to_low_24h",  0.0)),
        # patterns
        "patterns":        patterns_str,
    }

    logger.debug(
        f"Market context built | time: {ctx['time']} | "
        f"RSI: {ctx['rsi']:.1f} | ADX: {ctx['adx']:.1f} | "
        f"session: {', '.join(active_sessions) or 'off-session'} | "
        f"patterns: {patterns_str or 'none'}"
    )
    return ctx
