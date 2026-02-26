"""
Smart Trader — main orchestrator loop.

Architecture:
  Python  ->  fetch MT5 data, calculate indicators, scan zones, pre-filter
  Claude  ->  validate setup (compact ~600c prompt, no MCP, ~20-30s response)
  Python  ->  execute order, manage positions (BE / lock / trail)

Run:
  cd smart_trader && python main.py
  python smart_trader/main.py --config smart_trader/config.yaml
"""
import os
import sys
import time
import signal
import argparse

# Remove Claude Code env vars at startup so subprocess calls work cleanly
os.environ.pop("CLAUDECODE", None)
os.environ.pop("CLAUDE_CODE_SESSION", None)

# Add src/ to path so modules resolve from there
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from datetime import datetime, timezone, timedelta
from typing import Optional
from pathlib import Path
from loguru import logger
import yaml
import MetaTrader5 as mt5

import mt5_client as mt5c
import indicators as ind
import scanner   as scan
import executor  as exe
import claude_validator as validator
import zone_detector as zdet
import console_format as cfmt
import logger_config
import telegram_notifier as tg
import llm_comparator as llm_cmp
import regime_detector as rdet
import trade_tracker as tt
import adaptive_engine as ae
import chart_visualizer as cv


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(cfg: dict) -> None:
    """Delegate to logger_config for 5-sink structured logging."""
    logger_config.setup_logging(cfg)


# ── Trade journal ─────────────────────────────────────────────────────────────

_trade_log: list[dict] = []

# Fixed column order — superset of ENTRY + EXIT fields (prevents CSV misalignment)
_CSV_FIELDS = [
    "timestamp", "event", "direction", "confidence", "reason",
    "price", "sl", "tp", "lot", "zone_type", "signals", "rr",
    "ticket", "entry_price", "close_price", "pnl_pts", "pnl_usd",
    "close_type", "age_min", "last_sl", "last_tp",
]

def log_trade(event: str, data: dict) -> None:
    import csv
    trades_csv = _cfg.get("paths", {}).get("trades_csv", "logs/trades.csv")
    Path(trades_csv).parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event":     event,
        **data,
    }
    _trade_log.append(row)
    write_header = not Path(trades_csv).exists() or Path(trades_csv).stat().st_size == 0
    with open(trades_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


# ── Already-attempted guard ───────────────────────────────────────────────────

_attempted_zones: set[str] = set()  # zone_key -> block re-entry same cycle
_last_scan_report: Optional[datetime] = None  # last heartbeat Telegram report
_last_market_closed_log: Optional[datetime] = None  # throttle market-closed log
_session_trades: int = 0  # entries placed this session

# ── Position tracking (detect broker-side closes: SL/TP hit) ────────────────
# {ticket: {direction, entry, sl, tp, time_open, volume}}
_tracked_positions: dict[int, dict] = {}
_last_close_info: Optional[dict] = None    # {time, direction, zone_type, pnl_pts, entry_price}
_last_entry_zone_type: Optional[str] = None  # zone type of most recent entry
_IC_MARKETS_OFFSET = timedelta(hours=2)      # IC Markets server time is UTC+2

# ── Adaptive system modules (initialized in main()) ─────────────────────────
_regime_det: Optional[rdet.RegimeDetector] = None
_tracker: Optional[tt.TradeTracker] = None
_adaptive: Optional[ae.AdaptiveEngine] = None


def _zone_key(zone: dict, direction: str) -> str:
    """Stable key using price level — detected_at changes each cycle and would unblock zones."""
    level = zone.get("level") or zone.get("high") or zone.get("low") or 0
    return f"{direction}_{zone.get('type')}_{level:.1f}"


def _warmup_cooldown_from_history(magic: int) -> None:
    """
    On startup, reconstruct _last_close_info from MT5 deal history so the
    adaptive cooldown survives bot restarts.  Looks for most recent closing deal
    with our magic number within the last 24 hours.
    """
    global _last_close_info
    try:
        now = datetime.now(timezone.utc)
        from_time = now - timedelta(days=1)
        deals = mt5.history_deals_get(from_time, now)
        if not deals:
            logger.info("Cooldown warmup | no deals in last 24h")
            return

        # Log all our bot's deals for debugging
        our_deals = [d for d in deals if d.magic == magic]
        logger.info(
            f"Cooldown warmup | {len(deals)} total deals, "
            f"{len(our_deals)} with magic={magic}"
        )
        for i, d in enumerate(our_deals):
            logger.info(
                f"  our_deal[{i}] ticket={d.ticket} pos_id={d.position_id} "
                f"entry={d.entry} type={d.type} price={d.price} "
                f"profit={d.profit} volume={d.volume} "
                f"time={datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat()}"
            )

        # Walk backward to find most recent OUT deal with our magic
        for d in reversed(deals):
            if d.entry == 1 and d.magic == magic:  # DEAL_ENTRY_OUT + our bot
                close_price = d.price
                # OUT deal: type=1(SELL)=close LONG pos, type=0(BUY)=close SHORT pos
                direction = "LONG" if d.type == 1 else "SHORT"

                # Find matching IN deal for entry price
                entry_price = close_price  # fallback
                for d2 in deals:
                    if d2.position_id == d.position_id and d2.entry == 0:
                        entry_price = d2.price
                        break

                if direction == "LONG":
                    pnl_pts = close_price - entry_price
                else:
                    pnl_pts = entry_price - close_price

                close_time = datetime.fromtimestamp(d.time, tz=timezone.utc)
                _last_close_info = {
                    "time": close_time,
                    "direction": direction,
                    "zone_type": None,   # not recoverable from deal history
                    "pnl_pts": pnl_pts,
                    "entry_price": entry_price,
                }
                elapsed = (now - close_time).total_seconds() / 60
                logger.info(
                    f"Cooldown warmup | SELECTED deal ticket={d.ticket} "
                    f"pos_id={d.position_id} | close {elapsed:.0f}min ago | "
                    f"{direction} | entry={entry_price} close={close_price} "
                    f"{pnl_pts:+.1f}pt | ${d.profit:+.2f}"
                )
                return
        logger.info("Cooldown warmup | no OUT deals found for our magic")
    except Exception as e:
        logger.info(f"Cooldown warmup error: {e}")


# ── H1-candle re-entry cooldown ──────────────────────────────────────────────

def _next_h1_boundary(dt: datetime) -> datetime:
    """Return the start of the next H1 candle after dt."""
    return dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)


def _check_reentry_cooldown(
    direction: str, zone_type: str, atr: float,
    session_name: str, base_min_conf: float,
) -> tuple[bool, float]:
    """
    H1-candle re-entry cooldown: wait until the next H1 candle opens.
    New candle = fresh structure data, so cooldown resets naturally.

    Returns (allowed, effective_min_confidence).
    - Same zone + same direction -> BLOCKED until next H1
    - Same direction, different zone -> elevated confidence until next H1
    - Different direction -> allowed (fresh reversal signal)
    """
    if _last_close_info is None:
        return True, base_min_conf

    now = datetime.now(timezone.utc)
    next_h1 = _next_h1_boundary(_last_close_info["time"])

    if now >= next_h1:
        return True, base_min_conf  # new H1 candle, cooldown expired

    remaining = (next_h1 - now).total_seconds() / 60
    same_dir = direction == _last_close_info["direction"]
    same_zone = zone_type == _last_close_info.get("zone_type")

    if same_dir and same_zone:
        logger.info(
            f"  Skip {direction} — H1 cooldown {remaining:.0f}min left | "
            f"same zone {zone_type} (wait for next H1 candle)"
        )
        return False, 0

    if same_dir:
        elevated = max(base_min_conf, 0.80)
        logger.info(
            f"  H1 cooldown ({remaining:.0f}min left) — "
            f"different zone, min_conf raised {base_min_conf:.2f}->{elevated:.2f}"
        )
        return True, elevated

    # Different direction: fresh reversal signal, normal threshold
    return True, base_min_conf


def _track_and_detect_closes(symbol: str, magic: int) -> None:
    """
    Compare currently open positions against tracked set.
    If a tracked position disappears, it was closed externally (SL/TP hit by broker).
    Log to trade journal and send Telegram notification.
    """
    global _tracked_positions, _last_close_info

    now = datetime.now(timezone.utc)
    positions = mt5c.get_positions(symbol)
    our_pos = [p for p in positions if p.get("_raw") and p["_raw"].magic == magic]
    current_tickets = {p["ticket"] for p in our_pos}

    # ── Update tracking for new/existing positions ────────────────────────────
    for p in our_pos:
        t = p["ticket"]
        if t in _tracked_positions:
            # Update mutable fields only (SL/TP change from BE/trail)
            _tracked_positions[t]["sl"] = p["sl"]
            _tracked_positions[t]["tp"] = p["tp"]
            _tracked_positions[t]["volume"] = p["volume"]
            # Track peak profit (highest P/L ever reached)
            cur = p.get("price_current", p["price_open"])
            pnl = (cur - p["price_open"]) if p["type"] == "LONG" \
                else (p["price_open"] - cur)
            prev_peak = _tracked_positions[t].get("peak_profit", 0)
            if pnl > prev_peak:
                _tracked_positions[t]["peak_profit"] = round(pnl, 2)
        else:
            # New position — use MT5 open time adjusted for IC Markets UTC+2 offset.
            _tracked_positions[t] = {
                "direction":  p["type"],
                "entry":      p["price_open"],
                "sl":         p["sl"],
                "entry_sl":   p["sl"],   # original SL — never updated, used for stable sl_dist
                "tp":         p["tp"],
                "time_open":  p["time_open"] - _IC_MARKETS_OFFSET,
                "volume":     p["volume"],
                "peak_profit": 0.0,
            }

    # ── Detect disappeared positions ──────────────────────────────────────────
    disappeared = set(_tracked_positions.keys()) - current_tickets
    # Exclude tickets the bot itself closed (scratch/claude) to prevent false "CLOSED BY BROKER"
    bot_closed = exe.get_bot_closed_tickets()
    disappeared -= bot_closed
    # Record close info for bot-closed tickets (cooldown tracking, no notification)
    for t in bot_closed:
        info = _tracked_positions.pop(t, None)
        if info:
            cp, pnl_usd = mt5c.get_deal_close_info(t, magic=magic)
            entry, d = info["entry"], info["direction"]
            pnl_pts = ((cp - entry) if d == "LONG" else (entry - cp)) if cp else 0
            _last_close_info = {
                "time": now,
                "direction": d,
                "zone_type": _last_entry_zone_type,
                "pnl_pts": pnl_pts,
                "entry_price": entry,
            }
            # Record in adaptive tracker
            if _tracker:
                age = (now - info["time_open"]).total_seconds() / 60 if info.get("time_open") else 0
                _tracker.record_exit(t, {
                    "exit_price": cp, "pnl_pts": pnl_pts,
                    "pnl_usd": pnl_usd or 0, "close_type": "bot_closed",
                    "duration_min": age,
                })
                if _adaptive:
                    _adaptive.update_from_performance(_tracker)
    exe.clear_bot_closed_tickets()

    for ticket in disappeared:
        info = _tracked_positions.pop(ticket)
        direction = info["direction"]
        entry     = info["entry"]
        last_sl   = info["sl"]
        last_tp   = info["tp"]
        volume    = info["volume"]

        # Estimate close price and type based on last known SL/TP
        # Check deal history from MT5 for exact close price
        close_price, pnl_usd = mt5c.get_deal_close_info(ticket, magic=magic)

        if close_price is None:
            # Fallback: estimate from last SL (most likely scenario)
            close_price = last_sl
            pnl_usd = 0.0

        pnl_pts = (close_price - entry) if direction == "LONG" else (entry - close_price)

        # Determine close type
        sl_dist = abs(close_price - last_sl)
        tp_dist = abs(close_price - last_tp)
        if tp_dist < 1.0:
            close_type = "tp_hit"
        elif sl_dist < 1.0:
            close_type = "sl_hit"
        else:
            close_type = "unknown"

        # Age
        age_min = 0.0
        if info["time_open"]:
            age_min = (now - info["time_open"]).total_seconds() / 60

        if age_min < 60:
            dur_str = f"{age_min:.0f}min"
        else:
            h = int(age_min // 60)
            m = int(age_min % 60)
            dur_str = f"{h}h{m:02d}m"

        # ── Log to trade journal ──────────────────────────────────────────────
        logger.bind(kind="JOURNAL").info(
            f"EXIT | ticket={ticket} | {direction} | pnl_pts={pnl_pts:+.1f} | "
            f"pnl_usd=${pnl_usd:+.2f} | age={dur_str} | "
            f"entry={entry:.2f} | close={close_price:.2f} | "
            f"last_sl={last_sl:.2f} | last_tp={last_tp:.2f} | "
            f"reason={close_type}"
        )
        logger.bind(kind="TRADE").info(
            f"CLOSED BY BROKER | ticket={ticket} | {direction} | "
            f"{close_price:.2f} | P/L={pnl_pts:+.1f}pt (${pnl_usd:+.2f}) | "
            f"{close_type} | {dur_str}"
        )
        logger.info(
            f"  Position closed externally | ticket={ticket} | {direction} | "
            f"{entry:.2f} -> {close_price:.2f} | {pnl_pts:+.1f}pt (${pnl_usd:+.2f}) | "
            f"{close_type} | {dur_str}"
        )

        # ── Telegram notification ─────────────────────────────────────────────
        _notif = tg.get()
        if _notif:
            _notif.send_position_closed_external(
                ticket=ticket,
                direction=direction,
                entry_price=entry,
                last_sl=last_sl,
                last_tp=last_tp,
                close_price=close_price,
                pnl_pts=pnl_pts,
                pnl_usd=pnl_usd,
                age_min=age_min,
                close_type=close_type,
            )

        # ── CSV log ───────────────────────────────────────────────────────────
        log_trade("EXIT", {
            "direction":   direction,
            "ticket":      ticket,
            "entry_price": entry,
            "close_price": close_price,
            "pnl_pts":     round(pnl_pts, 1),
            "pnl_usd":     round(pnl_usd, 2),
            "close_type":  close_type,
            "age_min":     round(age_min, 0),
            "last_sl":     last_sl,
            "last_tp":     last_tp,
        })

        # ── Record in adaptive tracker ─────────────────────────────────────────
        if _tracker:
            _tracker.record_exit(ticket, {
                "exit_price": close_price, "pnl_pts": pnl_pts,
                "pnl_usd": pnl_usd, "close_type": close_type,
                "duration_min": age_min,
            })
            if _adaptive:
                _adaptive.update_from_performance(_tracker)

        # ── Record for adaptive re-entry cooldown ─────────────────────────────
        _last_close_info = {
            "time": now,
            "direction": direction,
            "zone_type": _last_entry_zone_type,
            "pnl_pts": pnl_pts,
            "entry_price": entry,
        }


# ── One scan cycle ────────────────────────────────────────────────────────────

def run_scan_cycle(cfg: dict) -> None:
    symbol     = cfg["mt5"]["symbol"]
    t_cfg      = cfg["trading"]
    c_cfg      = cfg["claude"]
    db_path    = cfg["paths"]["zone_cache_db"]
    proximity  = t_cfg["zone_proximity_pts"]
    min_conf   = t_cfg["min_confidence"]
    min_signals = 1                          # bare minimum: at least 1 SMC signal to report
    max_pos    = t_cfg["max_positions"]
    max_dir    = t_cfg["max_per_direction"]
    max_dd     = t_cfg["max_drawdown_pct"]
    free_m     = t_cfg["free_margin_pct"]
    sl_mult    = t_cfg["sl_atr_mult"]
    tp_mult    = t_cfg["tp_atr_mult"]

    now_utc = datetime.now(timezone.utc)
    wib_str = now_utc.strftime("%H:%M") + " UTC"

    # ── Market closed guard ───────────────────────────────────────────────────
    global _last_market_closed_log
    if not mt5c.is_market_open(symbol):
        if _last_market_closed_log is None or \
                (now_utc - _last_market_closed_log).total_seconds() >= 3600:
            logger.info(f"[{wib_str}] Market closed - bot standing by")
            _last_market_closed_log = now_utc
        return

    _last_market_closed_log = None  # reset when market reopens

    # ── Spike window guard ────────────────────────────────────────────────────
    if scan.is_spike_window(now_utc):
        logger.info(f"[{wib_str}] Spike window — skipping")
        return

    # ── Fetch data ────────────────────────────────────────────────────────────
    price_info = mt5c.get_price(symbol)
    if not price_info:
        logger.warning("No price data")
        return

    price = price_info["mid"]
    account = mt5c.get_account()
    positions = mt5c.get_positions(symbol)

    # ── Detect positions closed by broker (SL/TP hit) ──────────────────────
    magic = cfg.get("mt5", {}).get("magic", 202602)
    _track_and_detect_closes(symbol, magic)

    df_h4  = mt5c.get_candles(symbol, mt5.TIMEFRAME_H4,  25)
    df_h1  = mt5c.get_candles(symbol, mt5.TIMEFRAME_H1,  55)
    df_m15 = mt5c.get_candles(symbol, mt5.TIMEFRAME_M15, 12)

    if df_h1.empty or df_m15.empty:
        logger.warning("Candle data unavailable")
        return

    # ── Indicators ────────────────────────────────────────────────────────────
    atr_val   = ind.atr(df_h1, 14)
    rsi_val   = ind.rsi(df_h1, 14)
    pd_zone   = ind.premium_discount(df_h1)
    h4_bias   = "DISABLED"  # H4 bias disabled — not used in any gate/filter
    ema_trend = ind.h1_ema_trend(df_h1)
    session   = scan.current_session(now_utc)

    # ── Regime detection ──────────────────────────────────────────────────────
    regime_result = _regime_det.detect(df_h1) if _regime_det else rdet.RegimeDetector._default_result()
    regime = regime_result["regime"]
    regime_cat = regime.category
    regime_label = regime_result["short_label"]
    has_choch = regime_result["components"].get("has_choch", False)

    # ── Manage existing positions — moved to main loop (1-min interval) ───────

    # ── Common status line (reused across log messages) ────────────────────────
    spread_val = price_info.get("spread", 0)
    our_pos = [p for p in positions if p.get("_raw") and p["_raw"].magic == cfg.get("mt5", {}).get("magic", 202602)]
    status_line = (
        f"Price={price:.2f} | Spread={spread_val:.1f} | RSI={rsi_val:.0f} | "
        f"ATR={atr_val:.1f} | EMA={ema_trend} | P/D={pd_zone} | "
        f"H4={h4_bias} | {session['name']} | Regime={regime_label} | Pos={len(our_pos)}"
    )

    # ── Session guard: no new entries during OFF_HOURS ─────────────────────────
    if session["name"] == "OFF_HOURS":
        logger.info(f"[{wib_str}] OFF_HOURS — {status_line}")
        return

    # ── Post-impulse filter (daily range exhausted) ──────────────────────────
    if ind.daily_range_consumed(df_h1, 1.20):
        logger.info(f"[{wib_str}] {status_line} | Daily range exhausted — skip entry")
        return

    # ── Zone scan (dual source: detector + cache) ──────────────────────────────
    detected_zones = zdet.detect_all_zones(df_h1)
    cached_zones = scan.get_active_zones(db_path)
    zones = zdet.merge_zones(detected_zones, cached_zones)

    if not zones:
        logger.info(f"[{wib_str}] {status_line} | Zones=0 — no structure detected")
        return

    nearby = scan.find_nearby_zones(price, zones, proximity)
    if not nearby:
        nz = scan.nearest_zone(price, zones)
        nz_info = (
            f"nearest={nz['type']}@{nz.get('high') or nz.get('level', 0):.0f}"
            f"({nz['distance_pts']:.0f}pt away)"
        ) if nz else "no zones"
        logger.info(
            f"[{wib_str}] {status_line} | "
            f"Zones={len(zones)} | {nz_info} — waiting for zone proximity (<{proximity}pt)"
        )
        return

    logger.info(
        f"[{wib_str}] {status_line} | "
        f"Zones={len(zones)} | {len(nearby)} nearby — evaluating setup"
    )

    # ── Market log — full scan snapshot ───────────────────────────────────────
    logger.bind(kind="MARKET").debug(
        f"Scan | price={price:.2f} | spread={price_info.get('spread', 0):.1f}pt | "
        f"RSI={rsi_val:.1f} | ATR={atr_val:.1f} | EMA={ema_trend} | P/D={pd_zone} | "
        f"H4={h4_bias} | session={session['name']} | "
        f"zones_total={len(zones)} | nearby={len(nearby)} | "
        f"balance=${account.get('balance', 0):.2f} | equity=${account.get('equity', 0):.2f} | "
        f"open_pos={len([p for p in positions if p.get('_raw') and p['_raw'].magic == 202602])}"
    )

    # ── Collect nearby zones per direction ────────────────────────────────────
    long_zones = []
    short_zones = []
    for z in nearby:
        d = scan.direction_for_zone(z)
        if d is None:
            logger.debug(f"  Zone {z.get('type')} — no direction mapping, skipping")
        elif d == "LONG":
            long_zones.append(z)
        else:
            short_zones.append(z)

    # ── Evaluate best zone per direction (combined signals) ──────────────────
    for direction, dir_zones in [("LONG", long_zones), ("SHORT", short_zones)]:
        if not dir_zones:
            continue

        # Pick the closest zone as primary entry point
        primary = dir_zones[0]  # already sorted by distance
        zkey = _zone_key(primary, direction)
        if zkey in _attempted_zones:
            continue

        # Use nearby zone types for signal counting — prevents inflating count with distant zones
        nearby_zone_types = [z["type"] for z in dir_zones]

        # ── Cross-zone signal aggregation: enrich with wider-range zones ────
        # Zones within 1x ATR but beyond proximity may confirm structure
        # (e.g., BOS_BEAR@5155 near price + BEAR_FVG@5192 already broken through)
        wide_range = max(atr_val, proximity * 2)
        wide_zones = scan.find_nearby_zones(price, zones, wide_range)
        for wz in wide_zones:
            wd = scan.direction_for_zone(wz)
            if wd != direction:
                continue
            wz_type = wz.get("type", "")
            if wz_type not in nearby_zone_types:
                nearby_zone_types.append(wz_type)
                logger.debug(
                    f"  Cross-zone: added {wz_type}@"
                    f"{wz.get('high') or wz.get('level', 0):.0f} "
                    f"({wz.get('distance_pts', 0):.1f}pt) to {direction} signals"
                )

        # M15 confirmation
        m15_conf = ind.m15_confirmation(df_m15, direction)

        # OTE
        ote = ind.ote_zone(df_h1, direction)

        # Signal count from nearby zones of same direction only
        signal_count, signals = ind.count_signals(
            direction, nearby_zone_types, m15_conf, ote, price, pd_zone
        )

        # Market log — direction evaluation detail (SL/TP/RR not yet computed here)
        logger.bind(kind="MARKET").debug(
            f"  Eval {direction} | zone={primary['type']} | "
            f"dist={primary.get('distance_pts', 0):.1f}pt | "
            f"signals={signal_count}[{', '.join(signals)}] | "
            f"m15={m15_conf or 'none'} | OTE={'yes' if ote else 'no'}"
        )

        zone = primary

        # ── GATE 1: Minimum signal count (bare minimum — at least 1 SMC signal) ─
        tier3_present = any(s in ("Premium", "Discount") for s in signals)
        sig_label = (f"{signal_count}+T3" if tier3_present else str(signal_count))
        if signal_count < min_signals:
            logger.info(
                f"  Skip {direction} — {sig_label}/{min_signals} signals "
                f"[{', '.join(signals)}] (need >={min_signals} Tier-1/2)"
            )
            continue

        # ── GATE 2: Re-entry cooldown (mechanical safety) ──────────────────
        zone_has_choch = any("CHOCH" in s.upper() for s in signals)
        choch_detected = has_choch or zone_has_choch

        _ep = _adaptive.get_entry_params(regime_cat) if _adaptive else {}
        base_min_conf = _ep.get("min_confidence", min_conf) if _adaptive else min_conf
        reentry_ok, effective_min_conf = _check_reentry_cooldown(
            direction, zone.get("type", ""), atr_val, session["name"], base_min_conf,
        )
        if not reentry_ok:
            continue

        # ── GATE 3: Risk filters (mechanical safety) ───────────────────────
        risk_ok, risk_msg = scan.check_risk_filters(
            account, positions, direction, max_pos, max_dir, max_dd, free_m
        )
        if not risk_ok:
            logger.info(f"  Skip {direction} — {risk_msg}")
            continue

        # ── Data capture: algo pre-score (INFO only, not a gate) ───────────
        if _adaptive:
            pre_score, _ = _adaptive.algo_pre_score(
                signal_count, regime_cat, session["name"], direction,
                ema_trend, rsi_val, pd_zone, choch_detected, signals,
            )
        else:
            pre_score = 0.50

        # ── Data capture: signal quality analysis for Claude ───────────────
        has_structure = any(s in ("BOS", "CHoCH") for s in signals)
        has_ob = "OB" in signals
        has_fvg = "FVG" in signals
        has_m15 = "M15" in signals
        tier1_count = sum(1 for s in signals if s in ("BOS", "OB", "LiqSweep"))
        tier2_count = sum(1 for s in signals if s in ("FVG", "CHoCH", "Breaker", "M15", "OTE"))

        # EMA alignment assessment (data for Claude, not a gate)
        ema_aligned = (direction == "LONG" and ema_trend == "BULLISH") or \
                      (direction == "SHORT" and ema_trend == "BEARISH")
        ema_counter = (direction == "LONG" and ema_trend == "BEARISH") or \
                      (direction == "SHORT" and ema_trend == "BULLISH")

        # P/D zone assessment (data for Claude, not a gate)
        pd_aligned = (direction == "LONG" and pd_zone == "DISCOUNT") or \
                     (direction == "SHORT" and pd_zone == "PREMIUM")
        pd_opposing = (direction == "LONG" and pd_zone == "PREMIUM") or \
                      (direction == "SHORT" and pd_zone == "DISCOUNT")

        # All nearby zone details for Claude
        nearby_zone_details = [
            f"{z['type']}@{z.get('high') or z.get('level', 0):.0f}({z.get('distance_pts', 0):.1f}pt)"
            for z in dir_zones
        ]

        # ── Build SL/TP (adaptive regime params) ──────────────────────────────
        if atr_val <= 0:
            atr_val = 15.0  # fallback

        entry_params = _adaptive.get_entry_params(regime_cat) if _adaptive else {}
        sl_mult_eff = entry_params.get("sl_atr_mult", sl_mult)
        tp_mult_eff = entry_params.get("tp_atr_mult", tp_mult)
        sl_dist = atr_val * sl_mult_eff
        tp_dist = atr_val * tp_mult_eff

        if direction == "LONG":
            sl = round(price - sl_dist, 2)
            tp = round(price + tp_dist, 2)
        else:
            sl = round(price + sl_dist, 2)
            tp = round(price - tp_dist, 2)

        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        logger.bind(kind="MARKET").debug(
            f"  Setup {direction} | SL={sl:.2f} | TP={tp:.2f} | RR={rr:.1f} | "
            f"sl_dist={sl_dist:.1f}pt | tp_dist={tp_dist:.1f}pt"
        )

        # ── Recent context from claude_trader cache ───────────────────────────
        try:
            sys.path.insert(0, str(Path(db_path).parent.parent))
            import market_cache
            context = _build_context(market_cache.get_recent_snapshots(4))
        except Exception:
            context = ""

        # ── Setup dict for Claude (enriched — Claude is the primary decision maker)
        setup = {
            "price":        price,
            "spread":       price_info.get("spread", 0),
            "atr":          atr_val,
            "session":      session["name"],
            "h4_bias":      h4_bias,
            "h1_structure": primary.get("type", ""),
            "ema_trend":    ema_trend,
            "rsi":          rsi_val,
            "pd_zone":      pd_zone,
            "regime":             regime_label,
            "regime_category":    regime_cat,
            "regime_confidence":  regime_result["confidence"],
            "has_choch":          choch_detected,
            "pre_score":          pre_score,
            "direction":    direction,
            "zone_type":    zone.get("type"),
            "zone_level":   zone.get("level") or zone.get("high", price),
            "distance_pts": zone.get("distance_pts", 0),
            "m15_conf":     m15_conf,
            "ote":          ote,
            "signal_count": signal_count,
            "signals":      signals,
            "sl":           sl,
            "tp":           tp,
            "rr":           rr,
            "lot":          exe.calc_lot(
                account.get("balance", 100),
                t_cfg.get("risk_percent", 1.0),
                sl_dist,
                t_cfg.get("min_lot", 0.01),
            ),
            "context":      context,
            # ── Enriched data (Claude decides, not hard gates) ──────────
            "has_structure":     has_structure,
            "has_ob":            has_ob,
            "has_fvg":           has_fvg,
            "has_m15":           has_m15,
            "tier1_count":       tier1_count,
            "tier2_count":       tier2_count,
            "ema_aligned":       ema_aligned,
            "ema_counter":       ema_counter,
            "pd_aligned":        pd_aligned,
            "pd_opposing":       pd_opposing,
            "nearby_zones":      nearby_zone_details,
        }

        # Log enriched data for observability
        _ema_tag = "aligned" if ema_aligned else ("counter" if ema_counter else "neutral")
        _pd_tag = "aligned" if pd_aligned else ("opposing" if pd_opposing else "neutral")
        logger.info(
            f"  Validating {direction} | {zone['type']} @ {zone.get('level') or zone.get('high', price):.0f} | "
            f"signals={signal_count} [{', '.join(signals)}] | T1={tier1_count} T2={tier2_count} | "
            f"struct={'Y' if has_structure else 'N'} | EMA={_ema_tag} | P/D={_pd_tag} | "
            f"pre_score={pre_score:.2f} | "
            f"dist={zone['distance_pts']:.1f}pt | SL={sl:.0f} | TP={tp:.0f} | RR={rr:.1f}"
        )

        # ── Call Claude ───────────────────────────────────────────────────────
        _attempted_zones.add(zkey)
        response, claude_metrics = validator.validate(setup, c_cfg)

        if response is None:
            logger.warning("Claude validation failed — skipping")
            continue

        decision   = response["decision"]
        confidence = response["confidence"]
        reason     = response["reason"]
        _cl_lat_s  = claude_metrics["latency_ms"] / 1000
        _cl_tokens = claude_metrics["est_input_tokens"] + claude_metrics["est_output_tokens"]

        if decision == "NO_TRADE":
            logger.info(
                f"  Claude >> NO_TRADE (conf={confidence:.2f}) | {reason} "
                f"| {_cl_lat_s:.1f}s | ~{_cl_tokens} tokens"
            )
            logger.bind(kind="MARKET").debug(
                f"  CLAUDE NO_TRADE | conf={confidence:.2f} | {reason}"
            )
            continue

        if confidence < effective_min_conf:
            logger.info(
                f"  Claude >> {decision} REJECTED | conf={confidence:.2f} < min {effective_min_conf:.2f} | {reason} "
                f"| {_cl_lat_s:.1f}s | ~{_cl_tokens} tokens"
            )
            logger.bind(kind="MARKET").debug(
                f"  CLAUDE REJECTED | conf={confidence:.2f} < min={effective_min_conf:.2f} | {reason}"
            )
            continue

        logger.info(
            f"  Claude >> {decision} APPROVED | conf={confidence:.2f} | {reason} "
            f"| {_cl_lat_s:.1f}s | ~{_cl_tokens} tokens"
        )
        logger.bind(kind="MARKET").debug(
            f"  CLAUDE APPROVED | {decision} | conf={confidence:.2f} | {reason}"
        )

        # ── LLM Comparison (background, non-blocking) ────────────────────────
        llm_cmp.compare_entry_async(setup, response, c_cfg)

        # ── Execute ───────────────────────────────────────────────────────────
        # ALWAYS use bot-calculated SL/TP (Claude's SL override caused lot sizing issues)
        exec_sl = sl
        exec_tp = tp

        result = exe.place_trade(symbol, decision, exec_sl, exec_tp, account.get("balance", 100), t_cfg)
        if result and result.get("success"):
            global _last_entry_zone_type
            _last_entry_zone_type = zone.get("type")
            ticket = result.get("ticket", 0)
            log_trade("ENTRY", {
                "direction":  decision,
                "confidence": confidence,
                "reason":     reason,
                "price":      result["price"],
                "sl":         exec_sl,
                "tp":         exec_tp,
                "lot":        setup["lot"],
                "zone_type":  zone.get("type"),
                "signals":    ", ".join(signals),
                "rr":         rr,
            })
            # ── Record in adaptive trade tracker ──────────────────────────
            if _tracker:
                _tracker.record_entry(ticket, {
                    "direction": decision,
                    "entry_price": result["price"],
                    "sl": exec_sl,
                    "tp": exec_tp,
                    "lot": setup["lot"],
                    "zone_type": zone.get("type"),
                    "signals": signals,
                    "signal_count": signal_count,
                    "confidence": confidence,
                    "claude_reason": reason,
                    "session": session["name"],
                    "regime": regime_label,
                    "ema_trend": ema_trend,
                    "rsi": rsi_val,
                    "atr": atr_val,
                    "pd_zone": pd_zone,
                })
            logger.info(
                f"  ORDER FILLED | ticket={ticket} | {decision} @ {result['price']:.2f} | "
                f"SL={exec_sl:.2f} | TP={exec_tp:.2f} | lot={setup['lot']:.2f} | "
                f"RR={rr:.1f} | conf={confidence:.2f} | {session['name']} | "
                f"regime={regime_label} | pre_score={pre_score:.2f}"
            )
            # ── Journal log — structured ENTRY record ─────────────────────
            logger.bind(kind="JOURNAL").info(
                f"ENTRY | ticket={ticket} | {decision} | "
                f"price={result['price']:.2f} | sl={exec_sl:.2f} | tp={exec_tp:.2f} | "
                f"lot={setup['lot']:.2f} | rr={rr:.1f} | "
                f"zone={zone.get('type')} | signals=[{', '.join(signals)}] | "
                f"T1={tier1_count} T2={tier2_count} | struct={'Y' if has_structure else 'N'} | "
                f"EMA={_ema_tag} | P/D={_pd_tag} | regime={regime_label} | "
                f"pre_score={pre_score:.2f} | "
                f"conf={confidence:.2f} | session={session['name']} | reason={reason}"
            )

            # ── Telegram: entry notification ──────────────────────────────────
            _tg = tg.get()
            if _tg and result and result.get("success"):
                _tg.send_entry(
                    direction=decision,
                    price=result["price"],
                    sl=exec_sl,
                    tp=exec_tp,
                    lot=setup["lot"],
                    ticket=ticket,
                    zone_type=zone.get("type", ""),
                    zone_dist=zone.get("distance_pts", 0),
                    signals=signals,
                    signal_count=signal_count,
                    confidence=confidence,
                    claude_reason=reason,
                    session=session["name"],
                    ema_trend=ema_trend,
                    rsi=rsi_val,
                    atr=atr_val,
                    claude_latency_ms=claude_metrics["latency_ms"],
                    claude_tokens=_cl_tokens,
                    regime=regime_label,
                    pd_zone=pd_zone,
                    pre_score=pre_score,
                    tier1_count=tier1_count,
                    tier2_count=tier2_count,
                    has_structure=has_structure,
                    ema_aligned=ema_aligned,
                    pd_aligned=pd_aligned,
                )
                global _session_trades
                _session_trades += 1

            # ── Chart visualization — HD entry chart to Telegram ──────────────
            try:
                # Fetch extended M15 data for chart (60 bars = 15h context)
                df_m15_chart = mt5c.get_candles(symbol, mt5.TIMEFRAME_M15, 65)
                chart_entry = {
                    "ticket":     ticket,
                    "direction":  decision,
                    "price":      result["price"],
                    "sl":         exec_sl,
                    "tp":         exec_tp,
                    "confidence": confidence,
                    "reason":     reason,
                    "signals":    signals,
                    "zone_type":  zone.get("type", ""),
                    "regime":     regime_label,
                    "session":    session["name"],
                    "rsi":        rsi_val,
                    "ema_trend":  ema_trend,
                    "pd_zone":    pd_zone,
                    "atr":        atr_val,
                    "pre_score":  pre_score,
                    "lot":        setup["lot"],
                }
                chart_path = cv.generate_entry_chart(
                    df_m15_chart, df_h1, chart_entry, nearby,
                )
                if chart_path and _tg:
                    caption = (
                        f"<b>ENTRY CHART</b> | XAUUSD {decision}\n"
                        f"Conf: {confidence:.2f} | {regime_label} | {session['name']}\n"
                        f"Signals: {' + '.join(signals)}"
                    )
                    _tg.send_chart(chart_path, caption)
                    logger.info(f"  Entry chart sent to Telegram: {chart_path}")
            except Exception as e:
                logger.warning(f"  Chart generation skipped: {e}")

        # Only one trade per scan cycle
        break


def _valid_sl(sl: float, price: float, direction: str) -> bool:
    if sl <= 0:
        return False
    if direction == "LONG":
        return sl < price
    return sl > price


def _valid_tp(tp: float, price: float, direction: str) -> bool:
    if tp <= 0:
        return False
    if direction == "LONG":
        return tp > price
    return tp < price


def _send_heartbeat_report(cfg: dict) -> None:
    """Fetch fresh snapshot and send hourly analysis report via Telegram."""
    _tg = tg.get()
    if not _tg:
        return
    try:
        symbol    = cfg["mt5"]["symbol"]
        db_path   = cfg["paths"]["zone_cache_db"]
        proximity = cfg["trading"]["zone_proximity_pts"]
        magic     = cfg.get("mt5", {}).get("magic", 202602)

        # Market closed — send simple status instead of full scan report
        if not mt5c.is_market_open(symbol):
            _tg.send_error("Market Closed", "Bot standing by — akan aktif kembali saat market buka")
            return

        price_info = mt5c.get_price(symbol)
        if not price_info:
            return
        price     = price_info["mid"]
        account   = mt5c.get_account()
        positions = mt5c.get_positions(symbol)

        df_h4  = mt5c.get_candles(symbol, mt5.TIMEFRAME_H4, 25)
        df_h1  = mt5c.get_candles(symbol, mt5.TIMEFRAME_H1, 55)
        now_utc = datetime.now(timezone.utc)

        atr_val   = ind.atr(df_h1, 14)            if not df_h1.empty else 0.0
        rsi_val   = ind.rsi(df_h1, 14)            if not df_h1.empty else 50.0
        pd_zone   = ind.premium_discount(df_h1)   if not df_h1.empty else "EQUILIBRIUM"
        ema_trend = ind.h1_ema_trend(df_h1)       if not df_h1.empty else "NEUTRAL"
        h4_bias   = "DISABLED"
        session   = scan.current_session(now_utc)

        detected_zones = zdet.detect_all_zones(df_h1) if not df_h1.empty else []
        cached_zones   = scan.get_active_zones(db_path)
        zones  = zdet.merge_zones(detected_zones, cached_zones)
        nearby = scan.find_nearby_zones(price, zones, proximity) if zones else []

        # Find nearest zone regardless of proximity
        nz = scan.nearest_zone(price, zones) if zones else None

        our_pos = [p for p in positions if p.get("_raw") and p["_raw"].magic == magic]

        # Build position details for report
        pos_details = []
        for p in our_pos:
            pos_details.append({
                "ticket":    p["ticket"],
                "direction": p["type"],
                "entry":     p["price_open"],
                "current":   p["price_current"],
                "pnl":       p["profit"],
                "sl":        p["sl"],
                "tp":        p["tp"],
            })

        _tg.send_hourly_report(
            price=price,
            spread=price_info.get("spread", 0),
            rsi=rsi_val,
            atr=atr_val,
            ema_trend=ema_trend,
            pd_zone=pd_zone,
            h4_bias=h4_bias,
            session=session["name"],
            zones_total=len(zones),
            nearby_zones=nearby,
            nearest_zone=nz,
            proximity=proximity,
            balance=account.get("balance", 0),
            equity=account.get("equity", 0),
            open_positions=pos_details,
            session_trades=_session_trades,
        )
    except Exception as e:
        logger.warning(f"Heartbeat report error: {e}")


def _build_context(snapshots: list[dict]) -> str:
    if not snapshots:
        return ""
    lines = []
    for s in snapshots:
        ts  = (s.get("timestamp") or "")[:16].replace("T", " ")
        act = s.get("action") or "?"
        px  = s.get("price") or 0
        h4  = s.get("h4_bias") or "?"
        lines.append(f"{ts} | {act} | H4:{h4} | ${px:.0f}")
    return "\n".join(lines)


# ── Graceful shutdown ─────────────────────────────────────────────────────────

_running = True

def _handle_signal(sig, frame):
    global _running
    logger.info("Shutdown signal received — stopping after current cycle")
    _running = False

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

_cfg: dict = {}


# ── Manage Positions (1-min interval) ────────────────────────────────────────

def _run_manage_positions(cfg: dict) -> None:
    """Run automated exit management (BE, trail, scratch, stale tighten)."""
    symbol = cfg["mt5"]["symbol"]
    df_h1 = mt5c.get_candles(symbol, mt5.TIMEFRAME_H1, 55)
    regime_result = _regime_det.detect(df_h1) if _regime_det and not df_h1.empty else rdet.RegimeDetector._default_result()
    regime_cat = regime_result["regime"].category
    exit_params = _adaptive.get_exit_params(regime_cat) if _adaptive else None
    exe.manage_positions(symbol, cfg.get("mt5", {}), exit_params=exit_params, tracked=_tracked_positions)


# ── Claude Exit Review (15-min interval) ────────────────────────────────────

def _run_exit_review(cfg: dict) -> None:
    """Run Claude exit review with full market context for all open bot positions."""
    symbol  = cfg["mt5"]["symbol"]
    db_path = cfg["paths"]["zone_cache_db"]
    proximity = cfg.get("trading", {}).get("zone_proximity_pts", 7.0)

    positions = mt5c.get_positions(symbol)
    magic = cfg.get("mt5", {}).get("magic", 202602)
    our_pos = [p for p in positions if p.get("_raw") and p["_raw"].magic == magic]
    has_positions = len(our_pos) > 0

    if not has_positions:
        # No positions — skip Claude review but heartbeat still fires below
        logger.debug("── Exit Review — no open positions ──")
    else:
        logger.info(f"── Exit Review ({len(our_pos)} position(s)) ──")

    now_utc = datetime.now(timezone.utc)

    # ── Claude exit review only when positions are open ──────────────────────
    if has_positions:
        df_h1  = mt5c.get_candles(symbol, mt5.TIMEFRAME_H1, 55)
        df_m15 = mt5c.get_candles(symbol, mt5.TIMEFRAME_M15, 20)
        price_info = mt5c.get_price(symbol)

        price     = price_info["mid"] if price_info else 0
        spread    = price_info.get("spread", 0) if price_info else 0
        atr_val   = ind.atr(df_h1, 14) if not df_h1.empty else 15
        rsi_val   = ind.rsi(df_h1, 14) if not df_h1.empty else 50
        pd_zone   = ind.premium_discount(df_h1) if not df_h1.empty else "EQUILIBRIUM"
        ema_trend = ind.h1_ema_trend(df_h1) if not df_h1.empty else "NEUTRAL"
        session   = scan.current_session(now_utc)

        # Zone detection + proximity
        detected_zones = zdet.detect_all_zones(df_h1) if not df_h1.empty else []
        cached_zones = scan.get_active_zones(db_path)
        all_zones = zdet.merge_zones(detected_zones, cached_zones)
        nearby = scan.find_nearby_zones(price, all_zones, proximity * 2) if all_zones else []
        nearby_str = ", ".join(z["type"] for z in nearby[:6]) if nearby else "none"

        # Regime
        _exit_regime_label = "UNKNOWN"
        if _regime_det and not df_h1.empty:
            _exit_regime_result = _regime_det.detect(df_h1)
            _exit_regime_label = _exit_regime_result["short_label"]

        # M15 candle structure (last 3 candles summary)
        m15_summary = ""
        if not df_m15.empty and len(df_m15) >= 3:
            last3 = df_m15.tail(3)
            parts = []
            for _, row in last3.iterrows():
                body = row["close"] - row["open"]
                tag = "BULL" if body > 0 else "BEAR"
                parts.append(f"{tag}({abs(body):.1f})")
            m15_summary = " → ".join(parts)

        # H1 candle structure (last candle)
        h1_summary = ""
        if not df_h1.empty:
            last_h1 = df_h1.iloc[-1]
            h1_body = last_h1["close"] - last_h1["open"]
            h1_range = last_h1["high"] - last_h1["low"]
            h1_tag = "BULL" if h1_body > 0 else "BEAR"
            h1_summary = f"{h1_tag} body={abs(h1_body):.1f} range={h1_range:.1f}"

        # Daily range consumed
        daily_consumed = ind.daily_range_consumed(df_h1, 1.20) if not df_h1.empty else False

        # RSI momentum direction + EMA distance
        rsi_dir = ind.rsi_direction(df_h1) if not df_h1.empty else "FLAT"
        ema_dist = ind.ema_distance(df_h1) if not df_h1.empty else 0.0

        # Opposing zones for each open position direction
        opposing_str = ""
        for p in our_pos:
            opp_dir = "SHORT" if p["type"] == "LONG" else "LONG"
            opp_zones = [z for z in nearby if scan.direction_for_zone(z) == opp_dir]
            if opp_zones:
                opp_types = [z["type"] for z in opp_zones[:3]]
                opposing_str = ", ".join(opp_types)

        market_data = {
            "atr":             atr_val,
            "rsi":             rsi_val,
            "rsi_direction":   rsi_dir,
            "pd_zone":         pd_zone,
            "ema_trend":       ema_trend,
            "ema_distance":    ema_dist,
            "session":         session["name"],
            "nearby_signals":  nearby_str,
            "opposing_zones":  opposing_str or "none",
            "regime":          _exit_regime_label,
            "spread":          spread,
            "m15_structure":   m15_summary,
            "h1_structure":    h1_summary,
            "daily_exhausted": daily_consumed,
        }

        exe.review_positions_with_claude(
            symbol=symbol,
            cfg=cfg.get("mt5", {}),
            claude_cfg=cfg["claude"],
            market_data=market_data,
            tracked=_tracked_positions,
        )

    # ── Telegram heartbeat ALWAYS fires (every 15 min) ───────────────────────
    global _last_scan_report
    _last_scan_report = now_utc
    try:
        _send_heartbeat_report(cfg)
    except Exception as e:
        logger.warning(f"Heartbeat error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _cfg

    parser = argparse.ArgumentParser(description="Smart Trader — split architecture bot")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    _cfg = load_config(args.config)
    setup_logging(_cfg)

    mt5_cfg = _cfg.get("mt5", {})
    account_cfg = _cfg.get("account", {})

    logger.info("=" * 60)
    logger.info("Smart Trader starting")
    logger.info(
        f"Symbol: {mt5_cfg.get('symbol')} | Account: {account_cfg.get('login')} | "
        f"Proximity: {_cfg.get('trading', {}).get('zone_proximity_pts', 5.0)}pt | "
        f"Min Signals: 3 | Min Conf: {_cfg.get('trading', {}).get('min_confidence', 0.70)} | "
        f"Max Pos: {_cfg.get('trading', {}).get('max_positions', 1)} | "
        f"SL: {_cfg.get('trading', {}).get('sl_atr_mult', 2.0)}x ATR | "
        f"TP: {_cfg.get('trading', {}).get('tp_atr_mult', 4.0)}x ATR | "
        f"Adaptive: {_cfg.get('adaptive', {}).get('enabled', False)}"
    )
    logger.info("=" * 60)

    # Connect to MT5
    if not mt5c.connect(
        terminal_path=mt5_cfg.get("terminal_path", ""),
        login=account_cfg.get("login", 0),
        password=account_cfg.get("password", ""),
        server=account_cfg.get("server", ""),
    ):
        logger.error("MT5 connection failed — exiting")
        sys.exit(1)

    # ── Warmup: restore cooldown state from deal history ──────────────────────
    _warmup_cooldown_from_history(magic=mt5_cfg.get("magic", 202602))

    # ── Initialize adaptive system modules ──────────────────────────────────
    global _regime_det, _tracker, _adaptive
    adaptive_cfg = _cfg.get("adaptive", {})
    _regime_det = rdet.RegimeDetector(lookback=50)
    _tracker = tt.TradeTracker(state_path="data/trade_history.json")
    _adaptive = ae.AdaptiveEngine(
        config_baselines=adaptive_cfg.get("regime_params"),
        bounds={k: tuple(v) for k, v in adaptive_cfg.get("bounds", {}).items()} if adaptive_cfg.get("bounds") else None,
        state_path="data/adaptive_state.json",
        min_trades=adaptive_cfg.get("min_trades_for_adaptation", 20),
        max_shift_pct=adaptive_cfg.get("max_shift_pct", 0.25),
    )
    logger.info(
        f"Adaptive system initialized | regime_detector | trade_tracker | adaptive_engine"
    )

    # ── Initialize Telegram notifier ──────────────────────────────────────────
    tg_cfg = _cfg.get("telegram", {})
    tg.init(
        token=str(tg_cfg.get("token", "")),
        chat_id=str(tg_cfg.get("chat_id", "")),
        enabled=bool(tg_cfg.get("enabled", False)),
    )
    _tg = tg.get()
    if _tg:
        acct       = mt5c.get_account()
        t_cfg_main = _cfg.get("trading", {})
        c_cfg_main = _cfg.get("claude", {})
        _tg.send_bot_started(
            balance=acct.get("balance", 0),
            equity=acct.get("equity", 0),
            login=account_cfg.get("login", 0),
            server=account_cfg.get("server", ""),
            symbol=mt5_cfg.get("symbol", "XAUUSD"),
            leverage=acct.get("leverage", 100),
            model=c_cfg_main.get("model", "claude-opus-4-6"),
            sl_mult=t_cfg_main.get("sl_atr_mult", 2.0),
            tp_mult=t_cfg_main.get("tp_atr_mult", 3.0),
            max_pos=t_cfg_main.get("max_positions", 1),
            min_signals=1,
            min_conf=t_cfg_main.get("min_confidence", 0.70),
        )

    # ── Initialize LLM Comparator ────────────────────────────────────────────
    llm_cmp_cfg = _cfg.get("llm_comparison", {})
    llm_cmp.configure(llm_cmp_cfg)

    interval = _cfg.get("scanner", {}).get("interval_sec", 30)
    cycle = 0

    manage_interval = 2         # every 2 cycles × 30s = 1 min
    exit_review_interval = 30   # every 30 cycles × 30s = 15 min

    _reconnect_failures = 0
    try:
        while _running:
            cycle += 1
            logger.debug(f"── Cycle {cycle} ──")

            # ── MT5 health check — reconnect if connection lost ────────────
            if not mt5c.is_connected():
                if mt5c.reconnect():
                    _reconnect_failures = 0
                    logger.info("MT5 connection restored — resuming")
                else:
                    _reconnect_failures += 1
                    backoff = min(30 * _reconnect_failures, 300)
                    logger.warning(
                        f"MT5 reconnect failed (attempt {_reconnect_failures}) — "
                        f"retry in {backoff}s"
                    )
                    time.sleep(backoff)
                    continue

            # ── Manage positions — automated BE/trail/scratch/stale (1-min)
            if cycle % manage_interval == 0:
                try:
                    _run_manage_positions(_cfg)
                except Exception as e:
                    logger.exception(f"Manage positions error: {e}")

            try:
                run_scan_cycle(_cfg)
            except Exception as e:
                logger.exception(f"Cycle {cycle} error: {e}")

            # ── Claude exit review + Telegram heartbeat (15-min) ──────────
            if cycle % exit_review_interval == 0:
                try:
                    _run_exit_review(_cfg)
                except Exception as e:
                    logger.exception(f"Exit review error: {e}")

            # Hourly cleanup: re-allow zone evaluation + prune stale ticket sets
            if cycle % (3600 // interval) == 0:
                _attempted_zones.clear()
                exe.prune_closed_tickets(set(_tracked_positions.keys()))

            time.sleep(interval)
    finally:
        _tg = tg.get()
        if _tg:
            _tg.send_bot_stopped(
                reason="User stop / SIGINT",
                session_trades=_session_trades,
            )
        mt5c.disconnect()
        logger.info("Smart Trader stopped.")


if __name__ == "__main__":
    main()
