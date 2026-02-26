"""
Smart Trader -- Entry Chart Visualizer (HD v3)

Generates a high-definition TradingView-style chart for each trade entry:
  - M15 candlestick chart (last 60 bars)
  - SMC zones: OB/FVG as shaded bands, BOS/CHoCH/LiqSweep as horizontal levels
  - Entry line, SL line, TP line with labels (right side, anti-collision)
  - Zone labels on LEFT side (inside chart) to avoid overlap
  - EMA(50) H1 overlay
  - RSI(14) subplot with overbought/oversold zones
  - Compact trade info panel (right side)
  - 2-row bottom info bar
  - Dark professional theme

Output: 3200x2000px PNG (16x10 @ 200dpi)
"""
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from loguru import logger

# ── Style ────────────────────────────────────────────────────────────────────

BG_COLOR     = "#131722"
BG_PANEL     = "#1c2030"
CANDLE_UP    = "#26a69a"
CANDLE_DOWN  = "#ef5350"
GRID_COLOR   = "#1e222d"
TEXT_COLOR   = "#d1d4dc"
TEXT_DIM     = "#787b86"
AXIS_COLOR   = "#363a45"
EMA_COLOR    = "#FFD700"
RSI_COLOR    = "#BB86FC"
SL_COLOR     = "#ef5350"
TP_COLOR     = "#26a69a"

# SMC zone rendering config: (fill_color, edge_color, alpha, style)
_ZONE_STYLE = {
    "BULL_OB":       ("#2196F3", "#42A5F5", 0.22, "band"),
    "BEAR_OB":       ("#FF9800", "#FFB74D", 0.22, "band"),
    "BULL_FVG":      ("#4CAF50", "#66BB6A", 0.16, "band"),
    "BEAR_FVG":      ("#F44336", "#EF5350", 0.16, "band"),
    "BOS_BULL":      ("#00BCD4", "#00BCD4", 0.80, "level"),
    "BOS_BEAR":      ("#E91E63", "#E91E63", 0.80, "level"),
    "BULL_CHOCH":    ("#9C27B0", "#AB47BC", 0.70, "level"),
    "BEAR_CHOCH":    ("#9C27B0", "#AB47BC", 0.70, "level"),
    "BULL_LIQSWEEP": ("#FFEB3B", "#FDD835", 0.18, "band"),
    "BEAR_LIQSWEEP": ("#FFEB3B", "#FDD835", 0.18, "band"),
    "BULL_BREAKER":  ("#795548", "#8D6E63", 0.18, "band"),
    "BEAR_BREAKER":  ("#795548", "#8D6E63", 0.18, "band"),
}

_ZONE_SHORT = {
    "BULL_OB": "OB", "BEAR_OB": "OB",
    "BULL_FVG": "FVG", "BEAR_FVG": "FVG",
    "BOS_BULL": "BOS", "BOS_BEAR": "BOS",
    "BULL_CHOCH": "CHoCH", "BEAR_CHOCH": "CHoCH",
    "BULL_LIQSWEEP": "LiqSweep", "BEAR_LIQSWEEP": "LiqSweep",
    "BULL_BREAKER": "Breaker", "BEAR_BREAKER": "Breaker",
}

CHART_DIR = Path(__file__).resolve().parent.parent / "logs" / "charts"


# ── Public API ───────────────────────────────────────────────────────────────

def generate_entry_chart(
    df_m15: pd.DataFrame,
    df_h1: pd.DataFrame,
    entry: dict,
    nearby_zones: list,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Generate HD entry chart. Returns path to PNG or None on error."""
    try:
        return _render(df_m15, df_h1, entry, nearby_zones, output_path)
    except Exception as e:
        logger.warning(f"Chart generation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


# ── Anti-collision for labels ────────────────────────────────────────────────

def _spread_labels(positions: list[float], min_gap: float) -> list[float]:
    """
    Given a list of y-positions, spread them apart so no two are
    closer than min_gap. Returns adjusted positions in same order.
    """
    if not positions:
        return []
    indexed = sorted(enumerate(positions), key=lambda x: x[1])
    adjusted = [0.0] * len(positions)
    adjusted[indexed[0][0]] = indexed[0][1]
    prev_y = indexed[0][1]
    for i in range(1, len(indexed)):
        orig_idx, y = indexed[i]
        if y - prev_y < min_gap:
            y = prev_y + min_gap
        adjusted[orig_idx] = y
        prev_y = y
    return adjusted


# ── Renderer ─────────────────────────────────────────────────────────────────

def _render(df_m15, df_h1, entry, nearby_zones, output_path) -> str:
    # ── Data prep (last 60 M15 bars = 15 hours context) ─────────────────────
    df = df_m15.tail(60).copy().reset_index(drop=True)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volume": "Volume", "tick_volume": "Volume",
    })
    for c in ("Open", "High", "Low", "Close"):
        if c not in df.columns:
            return None
    if "Volume" not in df.columns:
        df["Volume"] = 0

    n_bars = len(df)

    # ── Entry data ──────────────────────────────────────────────────────────
    direction  = entry.get("direction", "LONG")
    price      = entry.get("price", df["Close"].iloc[-1])
    sl         = entry.get("sl", 0)
    tp         = entry.get("tp", 0)
    confidence = entry.get("confidence", 0)
    reason     = entry.get("reason", "")
    signals    = entry.get("signals", [])
    if isinstance(signals, str):
        signals = [s.strip() for s in signals.split(",")]
    zone_type  = entry.get("zone_type", "")
    regime     = entry.get("regime", "")
    session    = entry.get("session", "")
    rsi_val    = entry.get("rsi", 50)
    ema_trend  = entry.get("ema_trend", "NEUTRAL")
    pd_zone    = entry.get("pd_zone", "EQUILIBRIUM")
    atr_val    = entry.get("atr", 20)
    pre_score  = entry.get("pre_score", 0)
    ticket     = entry.get("ticket", 0)
    lot        = entry.get("lot", 0.01)

    is_long = direction == "LONG"
    entry_color = CANDLE_UP if is_long else CANDLE_DOWN
    entry_label = "BUY" if is_long else "SELL"
    dir_icon = "LONG" if is_long else "SHORT"

    # Computed values
    sl_dist = abs(price - sl) if sl else 0
    tp_dist = abs(tp - price) if tp else 0
    rr = tp_dist / sl_dist if sl_dist > 0 else 0

    # Tags
    ema_tag = "Aligned" if (is_long and ema_trend == "BULLISH") or (not is_long and ema_trend == "BEARISH") else "Counter" if ema_trend != "NEUTRAL" else "Neutral"
    pd_tag = "Aligned" if (is_long and pd_zone == "DISCOUNT") or (not is_long and pd_zone == "PREMIUM") else "Opposing" if pd_zone != "EQUILIBRIUM" else "Neutral"

    # ── EMA(50) from H1 ────────────────────────────────────────────────────
    ema_val = None
    if df_h1 is not None and len(df_h1) >= 55:
        col = "close" if "close" in df_h1.columns else "Close"
        if col in df_h1.columns:
            ema_val = float(pd.Series(df_h1[col].values).ewm(span=50, adjust=False).mean().iloc[-1])

    # ── RSI on M15 ──────────────────────────────────────────────────────────
    rsi_arr = _calc_rsi(df["Close"], 14)

    # ── Figure layout ────────────────────────────────────────────────────────
    # Chart takes left 78%, right panel takes 22%
    fig = plt.figure(figsize=(16, 10), facecolor=BG_COLOR)
    gs = fig.add_gridspec(
        3, 1, height_ratios=[7, 2, 1], hspace=0.05,
        left=0.06, right=0.77, top=0.92, bottom=0.04,
    )
    ax = fig.add_subplot(gs[0])
    ax_rsi = fig.add_subplot(gs[1], sharex=ax)
    ax_info = fig.add_subplot(gs[2])
    ax_info.axis("off")

    for a in (ax, ax_rsi):
        a.set_facecolor(BG_COLOR)
        a.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in a.spines.values():
            spine.set_color(AXIS_COLOR)
        a.grid(True, color=GRID_COLOR, alpha=0.3, linewidth=0.5)

    # ── Candlesticks ────────────────────────────────────────────────────────
    for i, (_, row) in enumerate(df.iterrows()):
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        col = CANDLE_UP if c >= o else CANDLE_DOWN
        body_lo, body_hi = min(o, c), max(o, c)
        bh = max(body_hi - body_lo, 0.05)
        ax.bar(i, bh, bottom=body_lo, width=0.55, color=col, edgecolor=col, linewidth=0.4)
        ax.plot([i, i], [l, h], color=col, linewidth=0.6)

    # ── X-axis time labels ──────────────────────────────────────────────────
    step = max(1, n_bars // 10)
    ticks = list(range(0, n_bars, step))
    labels = []
    for p in ticks:
        t = df.index[p]
        wib = t + timedelta(hours=7) if hasattr(t, "strftime") else t
        labels.append(wib.strftime("%d/%m\n%H:%M") if hasattr(wib, "strftime") else str(p))
    ax_rsi.set_xticks(ticks)
    ax_rsi.set_xticklabels(labels, fontsize=7, color=TEXT_DIM)
    ax_rsi.set_xlabel("WIB", fontsize=7, color=TEXT_DIM)
    plt.setp(ax.get_xticklabels(), visible=False)

    # ── Price range for label positioning ──────────────────────────────────
    y_lo = df["Low"].min()
    y_hi = df["High"].max()
    all_y = [y_lo, y_hi, price]
    if sl:
        all_y.append(sl)
    if tp:
        all_y.append(tp)
    y_range = max(all_y) - min(all_y)
    pad = y_range * 0.10
    ax_y_min = min(all_y) - pad
    ax_y_max = max(all_y) + pad

    # Minimum gap for anti-collision (in price units)
    label_min_gap = y_range * 0.04

    # ── EMA(50) line (label added to zone anti-collision below) ─────────────
    if ema_val is not None:
        ax.axhline(ema_val, color=EMA_COLOR, linewidth=1.0, linestyle="--", alpha=0.5,
                   zorder=3)

    # ── SMC Zones (draw bands/levels + LEFT-SIDE labels) ──────────────────
    # Collect ALL left-side labels (zones + EMA) for anti-collision
    zone_label_positions = []  # (y_pos, label_text, color)
    drawn_keys = set()

    # Add EMA label to the left-side anti-collision system
    if ema_val is not None:
        zone_label_positions.append((ema_val, f" EMA50 {ema_val:.0f} ", EMA_COLOR))
        drawn_keys.add("EMA50")

    for z in nearby_zones:
        zt = z.get("type", "")
        style_info = _ZONE_STYLE.get(zt, ("#888", "#888", 0.15, "band"))
        fill_c, edge_c, alpha, style = style_info
        label = _ZONE_SHORT.get(zt, zt)
        z_hi = z.get("high", 0)
        z_lo = z.get("low", 0)
        z_lv = z.get("level", 0)

        if style == "band":
            if z_hi and z_lo:
                bot, height = z_lo, z_hi - z_lo
            elif z_lv:
                band = atr_val * 0.12
                bot, height = z_lv - band, band * 2
            else:
                continue
            if bot > ax_y_max or bot + height < ax_y_min:
                continue
            rect = mpatches.Rectangle(
                (-1, bot), n_bars + 2, height,
                facecolor=fill_c, alpha=alpha,
                edgecolor=edge_c, linewidth=0.7, linestyle="-",
            )
            ax.add_patch(rect)
            mid_y = bot + height / 2
        else:
            level = z_hi or z_lv or z_lo
            if not level or level > ax_y_max or level < ax_y_min:
                continue
            ax.axhline(level, color=edge_c, linewidth=1.0, linestyle="-.", alpha=alpha)
            mid_y = level

        # Collect label for left-side placement (deduplicate by type+price)
        bull_bear = "B" if "BULL" in zt else "S"
        lkey = f"{label}_{mid_y:.0f}"
        if lkey not in drawn_keys:
            zone_label_positions.append((mid_y, f" {label} ({bull_bear}) {mid_y:.0f} ", edge_c))
            drawn_keys.add(lkey)

    # Anti-collision for zone labels on LEFT side (including EMA)
    if zone_label_positions:
        raw_ys = [item[0] for item in zone_label_positions]
        spread_ys = _spread_labels(raw_ys, label_min_gap)
        for i, (_, text, color) in enumerate(zone_label_positions):
            ax.text(0.5, spread_ys[i], text,
                    fontsize=6.5, color=color, fontweight="bold",
                    va="center", ha="left", zorder=8,
                    bbox=dict(facecolor=BG_COLOR, edgecolor=color,
                              alpha=0.90, pad=1.5, boxstyle="round,pad=0.2"))

    # ── Entry / SL / TP lines ───────────────────────────────────────────────
    # Entry line + label (only Entry gets on-chart label — SL/TP in right panel)
    ax.axhline(price, color=entry_color, linewidth=2.0, alpha=0.9, zorder=5)
    ax.text(n_bars + 0.3, price, f" {entry_label} {price:.2f} ",
            fontsize=8, color="white", fontweight="bold", va="center", zorder=9,
            bbox=dict(facecolor=entry_color, edgecolor="none",
                      alpha=0.90, pad=2, boxstyle="round,pad=0.3"))

    # SL line + zone shading (no text label — shown in right panel)
    if sl > 0:
        ax.axhline(sl, color=SL_COLOR, linewidth=1.5, linestyle="--", alpha=0.85, zorder=4)
        if is_long:
            ax.axhspan(sl, price, alpha=0.04, color=SL_COLOR)
        else:
            ax.axhspan(price, sl, alpha=0.04, color=SL_COLOR)

    # TP line + zone shading (no text label — shown in right panel)
    if tp > 0:
        ax.axhline(tp, color=TP_COLOR, linewidth=1.5, linestyle="--", alpha=0.85, zorder=4)
        if is_long:
            ax.axhspan(price, tp, alpha=0.04, color=TP_COLOR)
        else:
            ax.axhspan(tp, price, alpha=0.04, color=TP_COLOR)

    # Entry arrow
    marker = "^" if is_long else "v"
    ax.scatter([n_bars - 1], [price], marker=marker, s=250,
               color=entry_color, edgecolors="white", linewidth=1.2, zorder=10)

    # ── Y-axis ──────────────────────────────────────────────────────────────
    ax.set_ylim(ax_y_min, ax_y_max)
    ax.set_xlim(-1, n_bars + 4)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    # ── RSI ──────────────────────────────────────────────────────────────────
    rx = list(range(len(rsi_arr)))
    ax_rsi.plot(rx, rsi_arr, color=RSI_COLOR, linewidth=1.3)
    ax_rsi.axhline(70, color=SL_COLOR, linewidth=0.6, linestyle="--", alpha=0.4)
    ax_rsi.axhline(30, color=TP_COLOR, linewidth=0.6, linestyle="--", alpha=0.4)
    ax_rsi.axhline(50, color=AXIS_COLOR, linewidth=0.5, alpha=0.3)
    ax_rsi.fill_between(rx, 70, 90, alpha=0.06, color=SL_COLOR)
    ax_rsi.fill_between(rx, 10, 30, alpha=0.06, color=TP_COLOR)
    ax_rsi.set_ylim(10, 90)
    ax_rsi.set_ylabel("RSI(14)", fontsize=8, color=TEXT_DIM)
    ax_rsi.yaxis.set_label_position("right")
    ax_rsi.yaxis.tick_right()

    if rsi_arr:
        last_rsi = rsi_arr[-1]
        rc = SL_COLOR if last_rsi > 70 else TP_COLOR if last_rsi < 30 else RSI_COLOR
        ax_rsi.scatter([len(rsi_arr) - 1], [last_rsi], color=rc, s=50,
                       zorder=5, edgecolors="white", linewidth=0.5)
        ax_rsi.text(len(rsi_arr) + 0.3, last_rsi, f"{last_rsi:.1f}",
                    fontsize=7, color=rc, va="center", fontweight="bold")

    # ── Bottom info bar (signals only — everything else is in right panel)
    sig_str = " + ".join(signals) if signals else "none"
    t1 = sum(1 for s in signals if s in ("BOS", "OB", "LiqSweep"))
    t2 = sum(1 for s in signals if s in ("FVG", "CHoCH", "Breaker", "M15", "OTE"))

    bottom_text = f"Signals: {sig_str}   [Tier-1: {t1}  |  Tier-2: {t2}]   |   Pre-score: {pre_score:.2f}"
    ax_info.text(0.5, 0.5, bottom_text,
                 transform=ax_info.transAxes, fontsize=7.5, color=TEXT_COLOR,
                 fontfamily="monospace", va="center", ha="center",
                 fontweight="bold",
                 bbox=dict(facecolor=BG_PANEL, edgecolor=AXIS_COLOR,
                           alpha=0.9, pad=4, boxstyle="round,pad=0.3"))

    # ── Title + reason ──────────────────────────────────────────────────────
    now_wib = (datetime.now(timezone.utc) + timedelta(hours=7)).strftime("%d %b %Y  %H:%M WIB")
    title = f"XAUUSD M15  |  Smart Trader {entry_label}  |  #{ticket}  |  {now_wib}"
    fig.suptitle(title, fontsize=12, color=TEXT_COLOR, fontweight="bold", y=0.98)

    if reason:
        fig.text(0.42, 0.952, f'"{reason}"', fontsize=8, color=TEXT_DIM,
                 ha="center", fontstyle="italic")

    # ── Right panel — compact trade card ─────────────────────────────────
    _draw_right_panel(
        fig, entry_color, dir_icon, price, sl, tp, sl_dist, tp_dist, rr,
        confidence, regime, session, ema_trend, ema_tag, pd_zone, pd_tag,
        rsi_val, atr_val, lot, signals,
    )

    # ── Save HD ──────────────────────────────────────────────────────────────
    if output_path is None:
        CHART_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(CHART_DIR / f"entry_{ticket}_{ts}.png")
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=200, facecolor=BG_COLOR, edgecolor="none",
                bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    logger.info(f"Entry chart saved: {output_path}")
    return output_path


# ── Right panel drawer ───────────────────────────────────────────────────────

def _draw_right_panel(
    fig, entry_color, dir_icon, price, sl, tp, sl_dist, tp_dist, rr,
    confidence, regime, session, ema_trend, ema_tag, pd_zone, pd_tag,
    rsi_val, atr_val, lot, signals,
):
    """Draw structured trade info card on the right side of the figure."""
    x = 0.80   # Start x position (figure coords)
    y = 0.91   # Start y position (figure coords)
    line_h = 0.022  # Line height
    sec_gap = 0.008  # Gap between sections

    def _line(text, color, size=7, bold=False):
        nonlocal y
        fig.text(x, y, text, fontsize=size, color=color,
                 fontfamily="monospace",
                 fontweight="bold" if bold else "normal")
        y -= line_h

    def _separator():
        nonlocal y
        fig.text(x, y, "- - - - - - - - - - - -", fontsize=4, color=AXIS_COLOR,
                 fontfamily="monospace")
        y -= sec_gap

    # TRADE section
    _line("TRADE", entry_color, 9, bold=True)
    _line(f"  {dir_icon} @ {price:.2f}", entry_color, 8, bold=True)
    _separator()

    # RISK section
    _line("RISK", SL_COLOR, 7.5, bold=True)
    _line(f"  SL: {sl:.2f} ({sl_dist:.0f}pt)", SL_COLOR, 7)
    _separator()

    # TARGET section
    _line("TARGET", TP_COLOR, 7.5, bold=True)
    _line(f"  TP: {tp:.2f} ({tp_dist:.0f}pt)", TP_COLOR, 7)
    _line(f"  R:R  {rr:.1f}", TP_COLOR, 7, bold=True)
    _separator()

    # CONFIDENCE section
    conf_bar = _conf_visual(confidence)
    _line("CONFIDENCE", "#FFD700", 7.5, bold=True)
    _line(f"  {conf_bar}", "#FFD700", 7)
    _separator()

    # CONTEXT section
    _line("CONTEXT", TEXT_DIM, 7.5, bold=True)
    _line(f"  Regime:  {regime}", TEXT_DIM, 6.5)
    _line(f"  Session: {session}", TEXT_DIM, 6.5)

    # Shorten labels to prevent overflow
    ema_short = {"BULLISH": "BULL", "BEARISH": "BEAR", "NEUTRAL": "NEUT"}.get(ema_trend, ema_trend)
    pd_short = {"PREMIUM": "PREM", "DISCOUNT": "DISC", "EQUILIBRIUM": "EQ"}.get(pd_zone, pd_zone)
    ema_color = TP_COLOR if ema_tag == "Aligned" else SL_COLOR if ema_tag == "Counter" else TEXT_DIM
    _line(f"  EMA: {ema_short} ({ema_tag})", ema_color, 6.5)

    pd_color = TP_COLOR if pd_tag == "Aligned" else SL_COLOR if pd_tag == "Opposing" else TEXT_DIM
    _line(f"  P/D: {pd_short} ({pd_tag})", pd_color, 6.5)

    _line(f"  RSI: {rsi_val:.1f}", RSI_COLOR, 6.5)
    _line(f"  ATR: {atr_val:.1f}pt", TEXT_DIM, 6.5)
    _line(f"  Lot: {lot}", TEXT_DIM, 6.5)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _conf_visual(conf: float) -> str:
    filled = round(conf * 10)
    return "|" * filled + "." * (10 - filled) + f" {conf:.2f}"


def _calc_rsi(close: pd.Series, period: int = 14) -> list:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_g = gain.rolling(period, min_periods=period).mean()
    avg_l = loss.rolling(period, min_periods=period).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).tolist()


def generate_entry_chart_bytes(
    df_m15: pd.DataFrame, df_h1: pd.DataFrame,
    entry: dict, nearby_zones: list,
) -> Optional[bytes]:
    """Generate chart and return as bytes."""
    path = generate_entry_chart(df_m15, df_h1, entry, nearby_zones)
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None
