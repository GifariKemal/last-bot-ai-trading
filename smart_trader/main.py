"""
Smart Trader — main orchestrator loop.

Architecture:
  Python  →  fetch MT5 data, calculate indicators, scan zones, pre-filter
  Claude  →  validate setup (compact ~600c prompt, no MCP, ~20-30s response)
  Python  →  execute order, manage positions (BE / lock / trail)

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

from datetime import datetime, timezone
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
    write_header = not Path(trades_csv).exists()
    with open(trades_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


# ── Already-attempted guard ───────────────────────────────────────────────────

_attempted_zones: set[str] = set()  # zone_key → block re-entry same cycle
_last_scan_report: Optional[datetime] = None  # last heartbeat Telegram report
_last_market_closed_log: Optional[datetime] = None  # throttle market-closed log
_session_trades: int = 0  # entries placed this session


def _zone_key(zone: dict, direction: str) -> str:
    return f"{direction}_{zone.get('type')}_{zone.get('detected_at', '')}"


# ── One scan cycle ────────────────────────────────────────────────────────────

def run_scan_cycle(cfg: dict) -> None:
    symbol     = cfg["mt5"]["symbol"]
    t_cfg      = cfg["trading"]
    c_cfg      = cfg["claude"]
    db_path    = cfg["paths"]["zone_cache_db"]
    proximity  = t_cfg["zone_proximity_pts"]
    min_conf   = t_cfg["min_confidence"]
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
    h4_bias   = ind.h4_bias(df_h4) if not df_h4.empty else scan.get_last_h4_bias(db_path)
    ema_trend = ind.h1_ema_trend(df_h1)
    session   = scan.current_session(now_utc)

    # ── Manage existing positions ─────────────────────────────────────────────
    exe.manage_positions(symbol, cfg.get("mt5", {}))

    # ── Common status line (reused across log messages) ────────────────────────
    spread_val = price_info.get("spread", 0)
    our_pos = [p for p in positions if p.get("_raw") and p["_raw"].magic == cfg.get("mt5", {}).get("magic", 202602)]
    status_line = (
        f"Price={price:.2f} | Spread={spread_val:.1f} | RSI={rsi_val:.0f} | "
        f"ATR={atr_val:.1f} | EMA={ema_trend} | P/D={pd_zone} | "
        f"H4={h4_bias} | {session['name']} | Pos={len(our_pos)}"
    )

    # ── Session guard: no new entries during OFF_HOURS ─────────────────────────
    if session["name"] == "OFF_HOURS":
        logger.info(f"[{wib_str}] OFF_HOURS — {status_line}")
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

    # Also collect ALL zones (not just nearby) for structure signal counting
    all_long_types = [z["type"] for z in zones if scan.direction_for_zone(z) == "LONG"]
    all_short_types = [z["type"] for z in zones if scan.direction_for_zone(z) == "SHORT"]

    # ── Evaluate best zone per direction (combined signals) ──────────────────
    for direction, dir_zones in [("LONG", long_zones), ("SHORT", short_zones)]:
        if not dir_zones:
            continue

        # Pick the closest zone as primary entry point
        primary = dir_zones[0]  # already sorted by distance
        zkey = _zone_key(primary, direction)
        if zkey in _attempted_zones:
            continue

        # Use ALL zone types for signal counting (BOS/CHoCH don't need proximity)
        all_zone_types = all_long_types if direction == "LONG" else all_short_types
        h1_structure = primary.get("type", "")

        # M15 confirmation
        m15_conf = ind.m15_confirmation(df_m15, direction)

        # OTE
        ote = ind.ote_zone(df_h1, direction)

        # Signal count from ALL nearby zones of same direction
        signal_count, signals = ind.count_signals(
            direction, all_zone_types, m15_conf, ote, price, pd_zone, h1_structure
        )

        # Market log — direction evaluation detail (SL/TP/RR not yet computed here)
        logger.bind(kind="MARKET").debug(
            f"  Eval {direction} | zone={primary['type']} | "
            f"dist={primary.get('distance_pts', 0):.1f}pt | "
            f"signals={signal_count}[{', '.join(signals)}] | "
            f"m15={m15_conf or 'none'} | OTE={'yes' if ote else 'no'}"
        )

        zone = primary

        # RSI extreme gate
        if direction == "LONG" and rsi_val > 85:
            logger.info(f"  Skip LONG — RSI {rsi_val:.0f} > 85 (overbought)")
            logger.bind(kind="MARKET").debug(f"  SKIP LONG | reason=RSI_overbought({rsi_val:.0f})")
            continue
        if direction == "SHORT" and rsi_val < 15:
            logger.info(f"  Skip SHORT — RSI {rsi_val:.0f} < 15 (oversold)")
            logger.bind(kind="MARKET").debug(f"  SKIP SHORT | reason=RSI_oversold({rsi_val:.0f})")
            continue

        # ── Trend filter: don't trade against H1 EMA trend ──────────────
        if direction == "SHORT" and ema_trend == "BULLISH":
            logger.info(f"  Skip SHORT — H1 EMA trend is BULLISH (counter-trend)")
            logger.bind(kind="MARKET").debug(f"  SKIP SHORT | reason=EMA_counter_trend(BULLISH)")
            continue
        if direction == "LONG" and ema_trend == "BEARISH":
            logger.info(f"  Skip LONG — H1 EMA trend is BEARISH (counter-trend)")
            logger.bind(kind="MARKET").debug(f"  SKIP LONG | reason=EMA_counter_trend(BEARISH)")
            continue

        # Risk filters
        risk_ok, risk_msg = scan.check_risk_filters(
            account, positions, direction, max_pos, max_dir, max_dd, free_m
        )
        if not risk_ok:
            logger.info(f"  Skip {direction} — {risk_msg}")
            continue

        # ── Build SL/TP ───────────────────────────────────────────────────────
        if atr_val <= 0:
            atr_val = 15.0  # fallback

        sl_dist = atr_val * sl_mult
        tp_dist = atr_val * tp_mult

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

        # ── Setup dict for Claude ─────────────────────────────────────────────
        setup = {
            "price":        price,
            "spread":       price_info.get("spread", 0),
            "atr":          atr_val,
            "session":      session["name"],
            "h4_bias":      h4_bias,
            "h1_structure": h1_structure,
            "ema_trend":    ema_trend,
            "rsi":          rsi_val,
            "pd_zone":      pd_zone,
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
        }

        logger.info(
            f"  Validating {direction} | {zone['type']} @ {zone.get('level') or zone.get('high', price):.0f} | "
            f"signals={signal_count} [{', '.join(signals)}] | "
            f"dist={zone['distance_pts']:.1f}pt | SL={sl:.0f} | TP={tp:.0f} | RR={rr:.1f} | "
            f"lot={setup['lot']:.2f}"
        )

        # ── Call Claude ───────────────────────────────────────────────────────
        _attempted_zones.add(zkey)
        response = validator.validate(setup, c_cfg)

        if response is None:
            logger.warning("Claude validation failed — skipping")
            continue

        decision   = response["decision"]
        confidence = response["confidence"]
        reason     = response["reason"]

        if decision == "NO_TRADE":
            logger.info(
                f"  Claude >> NO_TRADE (conf={confidence:.2f}) | {reason}"
            )
            logger.bind(kind="MARKET").debug(
                f"  CLAUDE NO_TRADE | conf={confidence:.2f} | {reason}"
            )
            continue

        if confidence < min_conf:
            logger.info(
                f"  Claude >> {decision} REJECTED | conf={confidence:.2f} < min {min_conf} | {reason}"
            )
            logger.bind(kind="MARKET").debug(
                f"  CLAUDE REJECTED | conf={confidence:.2f} < min={min_conf} | {reason}"
            )
            continue

        logger.info(
            f"  Claude >> {decision} APPROVED | conf={confidence:.2f} | {reason}"
        )
        logger.bind(kind="MARKET").debug(
            f"  CLAUDE APPROVED | {decision} | conf={confidence:.2f} | {reason}"
        )

        # ── Execute ───────────────────────────────────────────────────────────
        # ALWAYS use bot-calculated SL/TP (Claude's SL override caused lot sizing issues)
        exec_sl = sl
        exec_tp = tp

        result = exe.place_trade(symbol, decision, exec_sl, exec_tp, account.get("balance", 100), t_cfg)
        if result and result.get("success"):
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
            logger.info(
                f"  ORDER FILLED | ticket={ticket} | {decision} @ {result['price']:.2f} | "
                f"SL={exec_sl:.2f} | TP={exec_tp:.2f} | lot={setup['lot']:.2f} | "
                f"RR={rr:.1f} | conf={confidence:.2f} | {session['name']}"
            )
            # ── Journal log — structured ENTRY record ─────────────────────
            logger.bind(kind="JOURNAL").info(
                f"ENTRY | ticket={ticket} | {decision} | "
                f"price={result['price']:.2f} | sl={exec_sl:.2f} | tp={exec_tp:.2f} | "
                f"lot={setup['lot']:.2f} | rr={rr:.1f} | "
                f"zone={zone.get('type')} | signals=[{', '.join(signals)}] | "
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
                )
                global _session_trades
                _session_trades += 1

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
        h4_bias   = ind.h4_bias(df_h4)            if not df_h4.empty else scan.get_last_h4_bias(db_path)
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


# ── Claude Exit Review ───────────────────────────────────────────────────────

def _run_exit_review(cfg: dict) -> None:
    """Run Claude exit review for all open bot positions."""
    symbol  = cfg["mt5"]["symbol"]
    db_path = cfg["paths"]["zone_cache_db"]

    positions = mt5c.get_positions(symbol)
    # Only review our positions
    our_pos = [p for p in positions if p.get("_raw") and p["_raw"].magic == 202602]
    if not our_pos:
        return

    logger.info(f"── Exit Review ({len(our_pos)} position(s)) ──")

    # Gather current market data
    df_h1 = mt5c.get_candles(symbol, mt5.TIMEFRAME_H1, 55)
    now_utc = datetime.now(timezone.utc)

    atr_val   = ind.atr(df_h1, 14) if not df_h1.empty else 15
    rsi_val   = ind.rsi(df_h1, 14) if not df_h1.empty else 50
    pd_zone   = ind.premium_discount(df_h1) if not df_h1.empty else "EQUILIBRIUM"
    ema_trend = ind.h1_ema_trend(df_h1) if not df_h1.empty else "NEUTRAL"
    session   = scan.current_session(now_utc)

    # Get nearby signal summary
    detected_zones = zdet.detect_all_zones(df_h1) if not df_h1.empty else []
    cached_zones = scan.get_active_zones(db_path)
    all_zones = zdet.merge_zones(detected_zones, cached_zones)
    zone_types = [z["type"] for z in all_zones[:8]]
    nearby_str = ", ".join(zone_types) if zone_types else "none"

    market_data = {
        "atr":             atr_val,
        "rsi":             rsi_val,
        "pd_zone":         pd_zone,
        "ema_trend":       ema_trend,
        "session":         session["name"],
        "nearby_signals":  nearby_str,
    }

    exe.review_positions_with_claude(
        symbol=symbol,
        cfg=cfg.get("mt5", {}),
        claude_cfg=cfg["claude"],
        market_data=market_data,
    )


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
        f"SL: {_cfg.get('trading', {}).get('sl_atr_mult', 3.0)}x ATR | "
        f"TP: {_cfg.get('trading', {}).get('tp_atr_mult', 5.0)}x ATR"
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
            sl_mult=t_cfg_main.get("sl_atr_mult", 3.0),
            tp_mult=t_cfg_main.get("tp_atr_mult", 5.0),
            max_pos=t_cfg_main.get("max_positions", 1),
            min_signals=3,
            min_conf=t_cfg_main.get("min_confidence", 0.70),
        )

    interval = _cfg.get("scanner", {}).get("interval_sec", 30)
    tg_interval_min = tg_cfg.get("scan_report_interval_min", 30)
    cycle = 0

    exit_review_interval = 10  # every 10 cycles = ~5 min

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

            try:
                run_scan_cycle(_cfg)
            except Exception as e:
                logger.exception(f"Cycle {cycle} error: {e}")

            # ── Claude smart exit review (profit-focused) ─────────────────
            if cycle % exit_review_interval == 0:
                try:
                    _run_exit_review(_cfg)
                except Exception as e:
                    logger.exception(f"Exit review error: {e}")

            # ── Telegram heartbeat scan report ────────────────────────────
            global _last_scan_report
            now_loop = datetime.now(timezone.utc)
            if (_last_scan_report is None or
                    (now_loop - _last_scan_report).total_seconds() >= tg_interval_min * 60):
                _last_scan_report = now_loop
                try:
                    _send_heartbeat_report(_cfg)
                except Exception as e:
                    logger.warning(f"Heartbeat error: {e}")

            # Clear attempted zones set each hour to allow re-evaluation
            if cycle % (3600 // interval) == 0:
                _attempted_zones.clear()

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
