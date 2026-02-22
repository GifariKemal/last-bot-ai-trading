#!/usr/bin/env python3
"""
Claude Autonomous Trader
Runs Claude Opus 4.6 as an autonomous XAUUSD trading agent.

Cycle timing : H1 candle boundaries (XX:00 UTC / WIB)
Intelligence : last N cycle summaries + cached structure levels
Broker       : Finex Demo (61045904)
Logs         : logs/trader.log   — full structured log (loguru, auto-rotate)
               logs/trades.csv   — trade journal (entry / SL moves / closes)
               data/market_cache.db — SQLite: H1 structure + cycle history
Timestamps   : all dual — WIB (GMT+7) primary / UTC secondary
"""

import subprocess
import time
import os
import sys
import csv
import json
import re
import shutil
import signal
import atexit
import textwrap
from datetime import datetime, timezone, timedelta
from pathlib import Path

from loguru import logger
import market_cache

# ══════════════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════════════
MODEL              = "claude-opus-4-6"
SCRIPT_DIR         = Path(__file__).parent
TRADING_RULES_FILE = SCRIPT_DIR / "trading_rules.md"
LOGS_DIR           = SCRIPT_DIR / "logs"
LOG_FILE           = LOGS_DIR / "trader.log"
TRADES_CSV         = LOGS_DIR / "trades.csv"
INTELLIGENCE_DIR   = SCRIPT_DIR / "intelligence"
PID_FILE           = SCRIPT_DIR / "trader.pid"
CLAUDE_TIMEOUT     = 300
CLAUDE_CMD         = r"C:\Users\Administrator\AppData\Roaming\npm\claude.cmd"
MEMORY_CYCLES      = 7

WIB = timezone(timedelta(hours=7))   # GMT+7

LOGS_DIR.mkdir(exist_ok=True)
INTELLIGENCE_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Timestamp helpers — dual WIB / UTC
# ══════════════════════════════════════════════════════════════════════════════

def ts_wib() -> str:
    """HH:MM:SS WIB"""
    return datetime.now(WIB).strftime("%H:%M:%S")

def ts_utc() -> str:
    """HH:MM:SS UTC"""
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def ts_dual() -> str:
    """HH:MM WIB / HH:MM UTC  — for cycle headers"""
    utc = datetime.now(timezone.utc)
    wib = utc.astimezone(WIB)
    return f"{wib.strftime('%H:%M')} WIB / {utc.strftime('%H:%M')} UTC"

def ts_full() -> str:
    """YYYY-MM-DD HH:MM:SS WIB / HH:MM:SS UTC  — for startup/summary"""
    utc = datetime.now(timezone.utc)
    wib = utc.astimezone(WIB)
    return (f"{wib.strftime('%Y-%m-%d %H:%M:%S')} WIB"
            f"  /  {utc.strftime('%H:%M:%S')} UTC")

def ts_csv() -> str:
    """ISO for CSV files (UTC)"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# ══════════════════════════════════════════════════════════════════════════════
#  Session detection
# ══════════════════════════════════════════════════════════════════════════════

_SESSIONS = [
    # (start_h, start_m, end_h, end_m, name, priority, icon)
    # Spike windows first — highest priority / avoid
    (7,  45,  8,   0,  "LONDON SPIKE",  "AVOID",   "⚡"),
    (12, 45, 13,   0,  "NY SPIKE",      "AVOID",   "⚡"),
    # Main sessions — overlap must precede NY
    (12,  0, 16,   0,  "OVERLAP",       "HIGHEST", "🔥"),
    ( 7,  0, 12,   0,  "LONDON",        "HIGH",    "🇬🇧"),
    (16,  0, 17,   0,  "NEW YORK",      "HIGH",    "🇺🇸"),
    ( 0,  0,  7,   0,  "ASIAN",         "LOW",     "🌏"),
    (17,  0, 24,   0,  "OFF-HOURS",     "MEDIUM",  "🌙"),
]

def detect_session(dt_utc: datetime = None) -> dict:
    """
    Returns session info for the given UTC datetime (defaults to now).
    Result keys: name, priority, icon, utc_range, wib_range
    """
    if dt_utc is None:
        dt_utc = datetime.now(timezone.utc)

    h, m = dt_utc.hour, dt_utc.minute
    hm   = h * 60 + m   # minutes since midnight UTC

    for sh, sm, eh, em, name, priority, icon in _SESSIONS:
        s_min = sh * 60 + sm
        e_min = (eh * 60 + em) if eh < 24 else 24 * 60
        if s_min <= hm < e_min:
            # Build human-readable time ranges
            def _wib_h(utc_h, utc_m=0):
                return f"{(utc_h + 7) % 24:02d}:{utc_m:02d}"

            utc_range = f"{sh:02d}:{sm:02d}–{eh % 24:02d}:{em:02d} UTC"
            wib_range = f"{_wib_h(sh, sm)}–{_wib_h(eh % 24, em)} WIB"
            return {
                "name":      name,
                "priority":  priority,
                "icon":      icon,
                "utc_range": utc_range,
                "wib_range": wib_range,
            }

    # Fallback (shouldn't happen)
    return {"name": "OFF-HOURS", "priority": "MEDIUM", "icon": "🌙",
            "utc_range": "17:00–00:00 UTC", "wib_range": "00:00–07:00 WIB"}


def format_session_header() -> str:
    """One-line session context to prepend to each Claude prompt."""
    now_utc = datetime.now(timezone.utc)
    now_wib = now_utc.astimezone(WIB)
    sess    = detect_session(now_utc)
    return (
        f"[CURRENT TIME: {now_wib.strftime('%H:%M')} WIB / "
        f"{now_utc.strftime('%H:%M')} UTC  |  "
        f"SESSION: {sess['icon']} {sess['name']}  |  "
        f"Priority: {sess['priority']}  |  "
        f"Range: {sess['utc_range']} / {sess['wib_range']}]"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Loguru — console (coloured + word-wrap) + file (plain, rotating)
#  Timestamps: WIB primary (machine local = WIB), UTC secondary in messages
# ══════════════════════════════════════════════════════════════════════════════
logger.remove()

_CONSOLE_PREFIX_LEN = 28   # "HH:MM:SS WIB/HH:MM UTC ⚡ LEVEL  " approx

def _wrap_for_console(msg: str) -> str:
    cols  = shutil.get_terminal_size((120, 40)).columns
    max_w = max(40, cols - _CONSOLE_PREFIX_LEN)
    lines = []
    for line in msg.splitlines():
        if len(line) <= max_w:
            lines.append(line)
        else:
            wrapped = textwrap.fill(
                line, width=cols,
                initial_indent="",
                subsequent_indent=" " * _CONSOLE_PREFIX_LEN,
                break_long_words=False,
                break_on_hyphens=False,
            )
            lines.append(wrapped)
    return "\n".join(lines)


class _ConsoleSink:
    def __init__(self):
        self._stream = open(sys.stdout.fileno(), mode="w",
                            encoding="utf-8", buffering=1, closefd=False)

    def write(self, message):
        text      = message.record["message"]
        wrapped   = _wrap_for_console(text)
        formatted = str(message)
        out = formatted.replace(text, wrapped, 1) if text and text in formatted else formatted
        self._stream.write(out)
        self._stream.flush()


# Console: WIB time shown (machine local = WIB), coloured
logger.add(
    _ConsoleSink(),
    level="DEBUG",
    colorize=True,
    format=(
        "<dim>{time:HH:mm:ss} WIB</dim> "
        "<level>{level.icon} {level: <7}</level> "
        "<white>{message}</white>"
    ),
    enqueue=False,
)

# File: dual timestamp WIB + UTC, plain text, rotating
logger.add(
    str(LOG_FILE),
    level="DEBUG",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} WIB | {level: <8} | {message}",
    rotation="10 MB",
    retention=5,
    enqueue=True,
)

# Custom levels
logger.level("CYCLE",   no=25, color="<bold><cyan>",    icon="🔄")
logger.level("TRADE",   no=35, color="<bold><green>",   icon="💰")
logger.level("SIGNAL",  no=26, color="<bold><yellow>",  icon="📊")
logger.level("CACHE",   no=24, color="<bold><blue>",    icon="🗄️")
logger.level("SESSION", no=27, color="<bold><magenta>", icon="🕐")

log = logger


# ══════════════════════════════════════════════════════════════════════════════
#  Terminal-aware block logger
# ══════════════════════════════════════════════════════════════════════════════
def log_block(text: str, top: str = "┌─ Claude output", pad: str = "│  "):
    cols  = shutil.get_terminal_size((120, 40)).columns
    max_w = max(40, cols - len(pad) - _CONSOLE_PREFIX_LEN)
    bar_w = min(54, cols - _CONSOLE_PREFIX_LEN - 4)

    log.info(top + " " + "─" * max(0, bar_w - len(top)))
    for raw in text.strip().splitlines():
        line = raw.rstrip()
        if not line:
            log.info(pad.rstrip())
            continue
        if len(line) <= max_w:
            log.info(f"{pad}{line}")
        else:
            chunks = textwrap.wrap(
                line, width=max_w,
                subsequent_indent=pad,
                break_long_words=False,
                break_on_hyphens=False,
            )
            for i, chunk in enumerate(chunks):
                log.info(f"{pad}{chunk}" if i == 0 else chunk)
    log.info("└" + "─" * bar_w)


# ══════════════════════════════════════════════════════════════════════════════
#  Trades CSV
# ══════════════════════════════════════════════════════════════════════════════
_CSV_COLS = [
    "timestamp_utc", "timestamp_wib", "event_type", "direction",
    "entry_price", "sl", "tp", "lot", "rr",
    "close_price", "pnl_usd",
    "session", "h4_bias", "atr", "notes",
]

def _csv_init():
    if not TRADES_CSV.exists():
        with open(TRADES_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_CSV_COLS).writeheader()
        log.info(f"Trade journal created: {TRADES_CSV.name}")

def write_trade_event(event_type: str, data: dict):
    row = {c: "" for c in _CSV_COLS}
    now_utc = datetime.now(timezone.utc)
    now_wib = now_utc.astimezone(WIB)
    row["timestamp_utc"] = now_utc.strftime("%Y-%m-%d %H:%M:%S")
    row["timestamp_wib"] = now_wib.strftime("%Y-%m-%d %H:%M:%S")
    row["event_type"]    = event_type
    row.update({k: v for k, v in data.items() if k in _CSV_COLS})
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_CSV_COLS).writerow(row)

    _icons = {"ENTRY":"🟢","PARTIAL_CLOSE":"🟡","FULL_CLOSE":"🔴",
              "SL_BE":"🔵","SL_LOCK":"🔵","SL_TRAIL":"🔵"}
    log.log("TRADE",
        f"{_icons.get(event_type,'•')} {event_type} "
        f"| {data.get('direction','')} "
        f"| Entry={data.get('entry_price','')} SL={data.get('sl','')} "
        f"TP={data.get('tp','')} Lot={data.get('lot','')} "
        f"| {data.get('notes','')}"
    )

def _parse_trade_events(output: str, meta: dict):
    base = {k: meta.get(k, "") for k in ("session", "h4_bias", "atr")}
    act_m = re.search(
        r'(?:Action|ACTION)\s*:\s*(ENTERED\s+LONG|ENTERED\s+SHORT)',
        output, re.IGNORECASE)
    if act_m:
        direction = "LONG" if "LONG" in act_m.group(1).upper() else "SHORT"
        em = re.search(r'\bEntry\s*[=:]\s*([\d]{4}\.[\d]+)', output, re.IGNORECASE)
        sm = re.search(r'\bSL\s*[=:]\s*([\d]{4}\.[\d]+)',    output, re.IGNORECASE)
        tm = re.search(r'\bTP\s*[=:]\s*([\d]{4}\.[\d]+)',    output, re.IGNORECASE)
        lm = re.search(r'\bLot\s*[=:]\s*([\d.]+)',           output, re.IGNORECASE)
        rm = re.search(r'\bRR\s*[=:]\s*([\d.]+)',            output, re.IGNORECASE)
        write_trade_event("ENTRY", {**base, "direction": direction,
            "entry_price": em.group(1) if em else "",
            "sl": sm.group(1) if sm else "", "tp": tm.group(1) if tm else "",
            "lot": lm.group(1) if lm else "", "rr": rm.group(1) if rm else ""})
    for m in re.finditer(r'SL\s+moved?\s+to\s+([\d.]+).*?(?:breakeven|BE)',
                         output, re.IGNORECASE):
        write_trade_event("SL_BE",    {**base, "notes": f"SL→{m.group(1)} BE"})
    for m in re.finditer(r'SL\s+moved?\s+to\s+([\d.]+).*?(?:profit.?lock|lock)',
                         output, re.IGNORECASE):
        write_trade_event("SL_LOCK",  {**base, "notes": f"SL→{m.group(1)} Lock"})
    for m in re.finditer(r'SL\s+(?:moved?|trail(?:ed)?)\s+to\s+([\d.]+).*?trail',
                         output, re.IGNORECASE):
        write_trade_event("SL_TRAIL", {**base, "notes": f"SL→{m.group(1)} Trail"})
    for m in re.finditer(
            r'(?:closed?|close)\s+50%[^\n]*?(?:at|@)\s*([\d.]+)'
            r'(?:[^\n]*?PnL[:\s]*\$?([-\d.]+))?', output, re.IGNORECASE):
        write_trade_event("PARTIAL_CLOSE", {**base,
            "close_price": m.group(1),
            "pnl_usd": m.group(2) if m.lastindex and m.lastindex >= 2 else "",
            "notes": "Partial 50%"})
    for m in re.finditer(
            r'TP\s+hit\s+at\s+([\d.]+)(?:[^\n]*?PnL[:\s]*\$?([-\d.]+))?',
            output, re.IGNORECASE):
        write_trade_event("FULL_CLOSE", {**base,
            "close_price": m.group(1),
            "pnl_usd": m.group(2) if m.lastindex and m.lastindex >= 2 else "",
            "notes": "TP hit"})
    for m in re.finditer(
            r'SL\s+hit\s+at\s+([\d.]+)(?:[^\n]*?PnL[:\s]*\$?([-\d.]+))?',
            output, re.IGNORECASE):
        write_trade_event("FULL_CLOSE", {**base,
            "close_price": m.group(1),
            "pnl_usd": m.group(2) if m.lastindex and m.lastindex >= 2 else "",
            "notes": "SL hit"})


# ══════════════════════════════════════════════════════════════════════════════
#  Duplicate prevention
# ══════════════════════════════════════════════════════════════════════════════
def check_existing_instance():
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            r = subprocess.run(["tasklist", "/FI", f"PID eq {old_pid}", "/NH"],
                               capture_output=True, text=True)
            if str(old_pid) in r.stdout:
                log.error(f"Another instance running (PID {old_pid}) — exiting")
                sys.exit(1)
        except Exception:
            pass
    PID_FILE.write_text(str(os.getpid()))

def _cleanup():
    try:
        if PID_FILE.exists() and PID_FILE.read_text().strip() == str(os.getpid()):
            PID_FILE.unlink()
    except Exception:
        pass

atexit.register(_cleanup)


# ══════════════════════════════════════════════════════════════════════════════
#  Graceful shutdown
# ══════════════════════════════════════════════════════════════════════════════
running = True

def _handle_signal(sig, frame):
    global running
    log.warning("━" * 55)
    log.warning(f"SHUTDOWN signal at {ts_dual()}")
    log.warning("━" * 55)
    running = False

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ══════════════════════════════════════════════════════════════════════════════
#  H1 timing
# ══════════════════════════════════════════════════════════════════════════════
def seconds_until_next_h1() -> float:
    now = datetime.now(timezone.utc)
    return 3600 - (now.minute * 60 + now.second + now.microsecond / 1e6) + 5

def next_boundary_str() -> str:
    now_utc = datetime.now(timezone.utc)
    next_h  = (now_utc.hour + 1) % 24
    utc_str = f"{next_h:02d}:00 UTC"
    wib_h   = (next_h + 7) % 24
    wib_str = f"{wib_h:02d}:00 WIB"
    return f"{wib_str} / {utc_str}"


# ══════════════════════════════════════════════════════════════════════════════
#  Intelligence memory (JSON files)
# ══════════════════════════════════════════════════════════════════════════════
def get_total_cycles() -> int:
    return len(list(INTELLIGENCE_DIR.glob("*_cycle*.json")))


def save_cycle_memory(session_cycle: int, output: str, success: bool) -> dict:
    now_utc = datetime.now(timezone.utc)
    now_wib = now_utc.astimezone(WIB)
    total   = get_total_cycles() + 1
    fname   = now_utc.strftime("%Y%m%d_%H%M") + f"_cycle{session_cycle}.json"

    rep_m  = re.search(r'={20,}.*?={20,}', output, re.DOTALL)
    report = rep_m.group(0) if rep_m else output[-3000:]

    act_m  = re.search(r'(?:Action|ACTION)\s*:\s*(ENTERED\s+LONG|ENTERED\s+SHORT|NO\s+TRADE)',
                       output, re.IGNORECASE)
    action = act_m.group(1).strip().upper() if act_m else "NO TRADE"

    px_m   = re.search(r'XAUUSD\s*[@|]\s*([\d]{4}\.[\d]+)', output) or \
             re.search(r'@\s*([\d]{4}\.[\d]+)', output)
    h4_m   = re.search(r'H4[^|\n]{0,15}(BULLISH|BEARISH|RANGING|NEUTRAL)',
                       output, re.IGNORECASE)
    h1_m   = re.search(r'H1[^|\n]{0,5}[:\s]+([^\n|]{5,60})', output, re.IGNORECASE)
    sess_m = re.search(r'Session:\s*([^\n|]{3,40})', output, re.IGNORECASE)
    atr_m  = re.search(r'ATR[^\d$]*\$?([\d]{1,3}\.[\d]+)', output, re.IGNORECASE)
    rsn_m  = re.search(r'(?:Reason|REASON)\s*:\s*(.+?)(?=\n[A-Z\[]|\Z)',
                       output, re.DOTALL | re.IGNORECASE)
    lng_m  = re.search(r'LONG\s+SETUP\s*:\s*(.+?)(?:\n|$)',  output, re.IGNORECASE)
    sht_m  = re.search(r'SHORT\s+SETUP\s*:\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
    ent_m  = re.search(r'\bEntry\s*[=:]\s*([\d]{4}\.[\d]+)', output, re.IGNORECASE)
    sl_m   = re.search(r'\bSL\s*[=:]\s*([\d]{4}\.[\d]+)',    output, re.IGNORECASE)
    tp_m   = re.search(r'\bTP\s*[=:]\s*([\d]{4}\.[\d]+)',    output, re.IGNORECASE)
    lot_m  = re.search(r'\bLot\s*[=:]\s*([\d.]+)',           output, re.IGNORECASE)

    price = float(px_m.group(1))  if px_m  else None
    atr   = float(atr_m.group(1)) if atr_m else None

    data = {
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_wib": now_wib.strftime("%Y-%m-%d %H:%M:%S"),
        "session_cycle": session_cycle,
        "total_cycle":   total,
        "success":       success,
        "action":        action,
        "price":         price,
        "h4_bias":       h4_m.group(1).upper()   if h4_m   else "?",
        "h1_trend":      h1_m.group(1).strip()   if h1_m   else "?",
        "session":       sess_m.group(1).strip() if sess_m else "?",
        "atr":           atr,
        "reason":        rsn_m.group(1).strip().replace('\n', ' ')[:200] if rsn_m else "",
        "long_setup":    lng_m.group(1).strip()  if lng_m  else "?",
        "short_setup":   sht_m.group(1).strip()  if sht_m  else "?",
        "trade":         None,
        "report":        report[:3000],
    }
    if ent_m:
        data["trade"] = {
            "entry": float(ent_m.group(1)) if ent_m else None,
            "sl":    float(sl_m.group(1))  if sl_m  else None,
            "tp":    float(tp_m.group(1))  if tp_m  else None,
            "lot":   float(lot_m.group(1)) if lot_m else None,
        }

    (INTELLIGENCE_DIR / fname).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {"session": data["session"], "h4_bias": data["h4_bias"], "atr": atr}
    _parse_trade_events(output, meta)
    market_cache.store_snapshot(data)
    applied = market_cache.apply_structure_update(total, output, price=price)

    log.log("SIGNAL",
        f"Cycle #{total} │ {action} │ H4:{data['h4_bias']} │ "
        f"${price} │ ATR:${atr} │ struct={'✓' if applied else '—'}"
    )
    return data


def load_intelligence_context() -> str:
    """Compact 1-line-per-cycle summary from SQLite DB (replaces verbose JSON files)."""
    snaps = market_cache.get_recent_snapshots(MEMORY_CYCLES)
    if not snaps:
        return ""
    lines = ["RECENT CYCLES (oldest→newest):"]
    for s in snaps:
        ts   = (s.get("timestamp") or "")[:16].replace("T", " ")
        act  = s.get("action")    or "?"
        h4   = s.get("h4_bias")   or "?"
        px   = s.get("price")     or 0
        atr  = s.get("atr")       or 0
        sess = (s.get("session")  or "?")[:12]
        long_r  = (s.get("long_setup")  or "?")[:50]
        short_r = (s.get("short_setup") or "?")[:50]
        lines.append(
            f"  {ts} | {act:<12} | H4:{h4} | ${px:.0f} | ATR:${atr:.1f}"
            f" | {sess} | L:{long_r} | S:{short_r}"
        )
    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
#  Main cycle
# ══════════════════════════════════════════════════════════════════════════════
def run_claude_cycle(cycle_num: int) -> bool:
    sess = detect_session()

    log.log("CYCLE",   "━" * 55)
    log.log("CYCLE",   f"CYCLE #{cycle_num}  (global:{get_total_cycles()+1})")
    log.log("SESSION", f"{sess['icon']} {sess['name']}  │  Priority: {sess['priority']}")
    log.log("SESSION", f"  {sess['wib_range']}  /  {sess['utc_range']}")
    log.log("SESSION", f"  Now: {ts_dual()}")
    log.log("CYCLE",   "━" * 55)

    if not TRADING_RULES_FILE.exists():
        log.error(f"Trading rules not found: {TRADING_RULES_FILE}")
        return False

    rules   = TRADING_RULES_FILE.read_text(encoding="utf-8")
    context = load_intelligence_context()
    cached  = market_cache.format_cached_structure()

    # Prepend session header so Claude knows current time/session immediately
    session_header = format_session_header() + "\n\n"
    prompt_parts   = [session_header + rules]
    if context: prompt_parts.append(context)
    if cached:  prompt_parts.append(cached)
    prompt = "\n\n".join(prompt_parts)

    log.info(
        f"Prompt │ rules={len(rules)}c "
        f"intel={len(context)}c "
        f"cache={len(cached)}c "
        f"total={len(prompt)}c"
    )
    stats = market_cache.get_stats()
    log.log("CACHE",
        f"DB │ cycles:{stats['total_cycles']}  "
        f"active_levels:{stats['active_levels']}  "
        f"trades:{stats['trades_taken']}"
    )

    cmd = [CLAUDE_CMD, "--model", MODEL, "--dangerously-skip-permissions", "-p"]

    try:
        t0  = time.time()
        env = {**os.environ}
        env.pop("CLAUDECODE", None)

        result = subprocess.run(
            cmd, input=prompt, capture_output=True,
            text=True, timeout=CLAUDE_TIMEOUT,
            encoding="utf-8", errors="replace",
            env=env, cwd=str(SCRIPT_DIR),
        )
        elapsed = time.time() - t0
        ok = result.returncode == 0
        log.info(f"Claude done │ {elapsed:.1f}s │ exit={result.returncode} │ {ts_dual()}")

        if result.stdout:
            log_block(result.stdout)
            save_cycle_memory(cycle_num, result.stdout, ok)

        if result.stderr:
            if not ok:
                log_block(result.stderr[:3000], top="┌─ Stderr (error)")
            else:
                for ln in result.stderr.splitlines():
                    if any(k in ln.lower() for k in ("error","fail","exception","traceback")):
                        log.warning(f"⚠  {ln}")
        return ok

    except subprocess.TimeoutExpired:
        log.error(f"⏱  Timed out after {CLAUDE_TIMEOUT}s at {ts_dual()}")
        return False
    except FileNotFoundError:
        log.error("Claude CLI not found")
        return False
    except Exception as exc:
        log.exception(f"Unexpected error: {exc}")
        return False


def sleep_interruptible(secs: float):
    end = time.time() + secs
    while running and time.time() < end:
        time.sleep(1)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    check_existing_instance()
    market_cache.init_db()
    _csv_init()

    prior = get_total_cycles()
    stats = market_cache.get_stats()
    sess  = detect_session()

    log.log("CYCLE", "═" * 55)
    log.log("CYCLE", "  CLAUDE AUTONOMOUS TRADER  —  STARTING")
    log.info(f"  Time    : {ts_full()}")
    log.info(f"  Session : {sess['icon']} {sess['name']}  ({sess['priority']})")
    log.info(f"  Model   : {MODEL}")
    log.info(f"  Account : Finex Demo 61045904  |  XAUUSD")
    log.info(f"  Timing  : H1 boundary (XX:00 UTC / WIB+7)")
    log.info(f"  Memory  : {MEMORY_CYCLES} cycles + SQLite structure cache")
    log.info(f"  PID     : {os.getpid()}")
    log.info(f"  History : {prior} intelligence files  │  {stats['total_cycles']} DB cycles")
    log.info(f"  Struct  : {stats['active_levels']} active levels in cache")
    log.info(f"  Trades  : {stats['trades_taken']} taken so far")
    log.info(f"  Logs    : {LOG_FILE.name}  │  {TRADES_CSV.name}")
    log.log("CYCLE", "═" * 55)

    cycle  = 0
    errors = 0
    first  = True

    while running:
        if first:
            log.info(f"First cycle — running immediately  [{ts_dual()}]")
            first = False
        else:
            wait    = seconds_until_next_h1()
            nxt     = next_boundary_str()
            sess_nxt = detect_session(
                datetime.now(timezone.utc) + timedelta(seconds=wait))
            log.info(
                f"⏳  Next boundary: {nxt}  ({wait:.0f}s)  "
                f"│  Upcoming session: {sess_nxt['icon']} {sess_nxt['name']}"
            )
            sleep_interruptible(wait)

        if not running:
            break

        cycle += 1
        if run_claude_cycle(cycle):
            errors = 0
        else:
            errors += 1
            log.warning(f"Cycle failed ({errors}/5 consecutive)")
            if errors >= 5:
                log.error("Too many consecutive failures — stopping")
                break

    log.log("CYCLE", "═" * 55)
    log.log("CYCLE",
        f"Stopped at {ts_full()}  │  session={cycle}  │  total={prior + cycle}")
    log.log("CYCLE", "═" * 55)


if __name__ == "__main__":
    main()
