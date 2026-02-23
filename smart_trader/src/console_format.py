"""
Console Format — colorized terminal output for Smart Trader.
Uses ANSI escape codes for rich, color-coded log output.
"""
import re
import sys
import os

# Enable ANSI/VT100 escape codes on Windows
if sys.platform == "win32":
    os.system("")  # triggers ENABLE_VIRTUAL_TERMINAL_PROCESSING via cmd
    # Also force via kernel32 for PowerShell
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # STD_OUTPUT_HANDLE = -11
        handle = kernel32.GetStdHandle(-11)
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass

# ── ANSI Escape Codes ────────────────────────────────────────────────────────

RST       = "\033[0m"
BOLD      = "\033[1m"
DIM       = "\033[2m"
ITALIC    = "\033[3m"
UNDERLINE = "\033[4m"

# Foreground
BLACK     = "\033[30m"
RED       = "\033[31m"
GREEN     = "\033[32m"
YELLOW    = "\033[33m"
BLUE      = "\033[34m"
MAGENTA   = "\033[35m"
CYAN      = "\033[36m"
WHITE     = "\033[37m"

# Bright foreground
BRED      = "\033[91m"
BGREEN    = "\033[92m"
BYELLOW   = "\033[93m"
BBLUE     = "\033[94m"
BMAGENTA  = "\033[95m"
BCYAN     = "\033[96m"
BWHITE    = "\033[97m"

# Background
BG_BLACK  = "\033[40m"
BG_RED    = "\033[41m"
BG_GREEN  = "\033[42m"
BG_BLUE   = "\033[44m"
BG_CYAN   = "\033[46m"
BG_WHITE  = "\033[47m"
BG_BYELLOW = "\033[103m"
BG_BGREEN  = "\033[102m"
BG_BRED    = "\033[101m"
BG_BBLUE   = "\033[104m"

# Level colors
_LEVEL_STYLE = {
    "DEBUG":    DIM,
    "INFO":     WHITE,
    "SUCCESS":  BGREEN + BOLD,
    "WARNING":  BYELLOW + BOLD,
    "ERROR":    BRED + BOLD,
    "CRITICAL": BRED + BOLD + UNDERLINE,
}

# ── Status prefixes (visual indicators for scan line types) ──────────────────

_PREFIX_SCAN     = f"{BBLUE}>>>{RST}"     # normal scan cycle
_PREFIX_WAIT     = f"{DIM}...{RST}"        # waiting for zone
_PREFIX_SETUP    = f"{BCYAN}<<<{RST}"      # setup found / evaluating
_PREFIX_TRADE    = f"{BGREEN}$$${RST}"     # trade execution
_PREFIX_EXIT     = f"{BMAGENTA}>>>{RST}"   # exit review
_PREFIX_OFF      = f"{DIM}zzz{RST}"        # off hours / market closed
_PREFIX_WARN     = f"{BYELLOW}!!{RST} "    # warning / skip
_PREFIX_CONNECT  = f"{BGREEN}>>>{RST}"     # connection


def _c(color: str, text: str) -> str:
    return f"{color}{text}{RST}"


def _badge(bg: str, fg: str, text: str) -> str:
    """Render a highlighted badge: [TEXT]"""
    return f"{bg}{fg}{BOLD} {text} {RST}"


def _rsi_color(val: float) -> str:
    if val > 80 or val < 20:
        return BRED + BOLD
    if val > 70 or val < 30:
        return BYELLOW
    return GREEN


def _zone_dist_color(dist: float) -> str:
    """Color based on how far nearest zone is."""
    if dist <= 5:
        return BGREEN + BOLD
    if dist <= 15:
        return BYELLOW
    if dist <= 30:
        return YELLOW
    return DIM + WHITE


def colorize_line(line: str) -> str:
    """Apply pattern-based ANSI colors to a log line."""

    # ── Banner / separator lines ─────────────────────────────────────────
    if "====" in line or line.strip().startswith("="):
        return _c(CYAN + BOLD, line)

    if "Smart Trader starting" in line:
        return _badge(BG_BGREEN, BLACK, "SMART TRADER STARTED")

    if "Smart Trader stopped" in line:
        return _badge(BG_BRED, WHITE, "SMART TRADER STOPPED")

    if "Logger configured" in line:
        return f"{_PREFIX_CONNECT} {_c(DIM, line)}"

    # ── MT5 connection ─────────────────────────────────────────────────
    if "MT5 connected" in line:
        line = re.sub(r'(MT5 connected)', _c(BGREEN + BOLD, 'MT5 connected'), line)
        line = re.sub(r'(login=)(\d+)', lambda m: f'login={BWHITE + BOLD}{m.group(2)}{RST}', line)
        line = re.sub(r'(\$[\d,.]+)', lambda m: f'{BGREEN + BOLD}{m.group(1)}{RST}', line)
        return f"{_PREFIX_CONNECT} {line}"

    if "MT5 reconnect" in line or "MT5 connection" in line:
        if "restored" in line or "successfully" in line:
            return f"{_PREFIX_CONNECT} {_c(BGREEN + BOLD, line)}"
        if "failed" in line:
            return f"{_PREFIX_WARN} {_c(BRED + BOLD, line)}"
        return f"{_PREFIX_WARN} {_c(BYELLOW, line)}"

    # ── Cycle headers ────────────────────────────────────────────────────
    if "Cycle " in line and ("--" in line or "-- " in line):
        return _c(DIM, line)

    if "Exit Review" in line and ("--" in line or "(" in line):
        return f"{_PREFIX_EXIT} {_c(BMAGENTA + BOLD, line)}"

    # ── Determine scan line type for prefix ────────────────────────────
    prefix = ""
    if "OFF_HOURS" in line:
        prefix = _PREFIX_OFF
    elif "Market closed" in line:
        prefix = _PREFIX_OFF
    elif "waiting for zone" in line or "no structure" in line:
        prefix = _PREFIX_WAIT
    elif "evaluating setup" in line or "nearby" in line:
        prefix = _PREFIX_SETUP
    elif "Validating" in line:
        prefix = _PREFIX_SETUP
    elif "ORDER FILLED" in line or "ENTRY" in line:
        prefix = _PREFIX_TRADE

    # ── Trade execution (high-priority matches) ──────────────────────────
    if "ORDER FILLED" in line:
        line = re.sub(r'ORDER FILLED', _badge(BG_BGREEN, BLACK, 'ORDER FILLED'), line)

    if "PARTIAL CLOSE" in line:
        line = re.sub(r'PARTIAL CLOSE', _badge(BG_BYELLOW, BLACK, 'PARTIAL CLOSE'), line)

    if "PROFIT LOCK" in line:
        line = re.sub(r'PROFIT LOCK', _badge(BG_BYELLOW, BLACK, 'PROFIT LOCK'), line)

    if "CLAUDE EXIT" in line:
        line = re.sub(r'CLAUDE EXIT', _badge(BG_BRED, WHITE, 'CLAUDE EXIT'), line)

    if "CLAUDE TAKE PROFIT" in line:
        line = re.sub(r'CLAUDE TAKE PROFIT', _badge(BG_BGREEN, BLACK, 'CLAUDE TAKE PROFIT'), line)

    if "SCRATCH EXIT" in line:
        line = re.sub(r'SCRATCH EXIT', _badge(BG_BYELLOW, BLACK, 'SCRATCH EXIT'), line)

    if "STALE TIGHTEN" in line:
        line = re.sub(r'STALE TIGHTEN', _c(BYELLOW + BOLD, 'STALE TIGHTEN'), line)

    if "CLAUDE TIGHTEN" in line:
        line = re.sub(r'CLAUDE TIGHTEN', _c(BYELLOW + BOLD, 'CLAUDE TIGHTEN'), line)

    if "CLOSED" in line and "ticket=" in line:
        line = re.sub(r'CLOSED', _badge(BG_BRED, WHITE, 'CLOSED'), line)

    # ── Market data patterns ─────────────────────────────────────────────

    # Price=XXXX.XX
    line = re.sub(
        r'(Price=)([\d.]+)',
        lambda m: f'{CYAN}Price={BCYAN + BOLD}{m.group(2)}{RST}',
        line,
    )

    # Balance / money ($XXXX.XX)
    line = re.sub(
        r'(\$[\d,.]+)',
        lambda m: f'{BGREEN}{m.group(1)}{RST}',
        line,
    )

    # RSI=XX (dynamic color based on value)
    def _fmt_rsi(m):
        val = float(m.group(2))
        return f'{YELLOW}RSI={_rsi_color(val)}{m.group(2)}{RST}'
    line = re.sub(r'(RSI=)([\d.]+)', _fmt_rsi, line)

    # ATR=XX.X
    line = re.sub(
        r'(ATR=)([\d.]+)',
        lambda m: f'{MAGENTA}ATR={BMAGENTA}{m.group(2)}{RST}',
        line,
    )

    # Spread=N.N
    line = re.sub(
        r'(Spread=)([\d.]+)',
        lambda m: f'{DIM}Spread={WHITE}{m.group(2)}{RST}',
        line,
    )

    # EMA=TREND
    line = re.sub(
        r'(EMA=)(BULLISH|BEARISH|NEUTRAL)',
        lambda m: (
            f'{MAGENTA}EMA='
            f'{BGREEN + BOLD if m.group(2) == "BULLISH" else BRED + BOLD if m.group(2) == "BEARISH" else BYELLOW}'
            f'{m.group(2)}{RST}'
        ),
        line,
    )

    # P/D=ZONE
    line = re.sub(
        r'(P/D=)(PREMIUM|DISCOUNT|EQUILIBRIUM)',
        lambda m: (
            f'{DIM}P/D='
            f'{BRED if m.group(2) == "PREMIUM" else BGREEN if m.group(2) == "DISCOUNT" else BYELLOW}'
            f'{m.group(2)}{RST}'
        ),
        line,
    )

    # H4=BIAS
    line = re.sub(
        r'(H4=)(BULLISH|BEARISH|RANGING)',
        lambda m: (
            f'{DIM}H4='
            f'{BGREEN if m.group(2) == "BULLISH" else BRED if m.group(2) == "BEARISH" else BYELLOW}'
            f'{m.group(2)}{RST}'
        ),
        line,
    )

    # Pos=N (position count)
    line = re.sub(
        r'(Pos=)(\d+)',
        lambda m: (
            f'{DIM}Pos={BWHITE + BOLD}{m.group(2)}{RST}'
            if m.group(2) == "0"
            else f'{BCYAN}Pos={BCYAN + BOLD}{m.group(2)}{RST}'
        ),
        line,
    )

    # Zones=N (zone count)
    line = re.sub(
        r'(Zones=)(\d+)',
        lambda m: (
            f'{BRED}Zones={BRED + BOLD}{m.group(2)}{RST}'
            if m.group(2) == "0"
            else f'{BBLUE}Zones={BBLUE + BOLD}{m.group(2)}{RST}'
        ),
        line,
    )

    # nearest=TYPE@LEVEL(NNpt away)
    def _fmt_nearest(m):
        ztype = m.group(1)
        level = m.group(2)
        dist = float(m.group(3))
        dist_c = _zone_dist_color(dist)
        # Color zone type
        if "BULL" in ztype:
            ztype_c = _c(BGREEN, ztype)
        elif "BEAR" in ztype:
            ztype_c = _c(BRED, ztype)
        else:
            ztype_c = _c(WHITE, ztype)
        return f'{CYAN}nearest={RST}{ztype_c}{DIM}@{BWHITE}{level}{RST}{DIM}({dist_c}{m.group(3)}pt away{RST}{DIM}){RST}'
    line = re.sub(
        r'nearest=(\w+)@([\d.]+)\(([\d.]+)pt away\)',
        _fmt_nearest,
        line,
    )

    # ── Status messages (end of scan lines) ────────────────────────────
    line = re.sub(
        r'waiting for zone proximity',
        _c(DIM + YELLOW, 'waiting for zone proximity'),
        line,
    )
    line = re.sub(
        r'no structure detected',
        _c(DIM + RED, 'no structure detected'),
        line,
    )
    line = re.sub(
        r'evaluating setup',
        _c(BCYAN + BOLD, 'evaluating setup'),
        line,
    )

    # ── Session names ────────────────────────────────────────────────────
    line = re.sub(
        r'\b(OVERLAP)\b',
        lambda m: _badge(BG_BBLUE, WHITE, m.group(1)),
        line,
    )
    line = re.sub(
        r'\b(LONDON)\b(?!_)',
        lambda m: _c(BBLUE + BOLD, m.group(1)),
        line,
    )
    line = re.sub(
        r'\b(NEW_YORK|LONDON_NY)\b',
        lambda m: _c(BBLUE + BOLD, m.group(1)),
        line,
    )
    line = re.sub(
        r'\b(ASIAN)\b',
        lambda m: _c(BLUE, m.group(1)),
        line,
    )
    line = re.sub(
        r'\bOFF_HOURS\b',
        lambda m: _c(DIM + YELLOW, 'OFF_HOURS'),
        line,
    )

    # ── Direction ────────────────────────────────────────────────────────
    line = re.sub(r'\bLONG\b', _c(BGREEN + BOLD, 'LONG'), line)
    line = re.sub(r'\bSHORT\b', _c(BRED + BOLD, 'SHORT'), line)
    line = re.sub(r'\bNO_TRADE\b', _c(BYELLOW + BOLD, 'NO_TRADE'), line)

    # ── Trade parameters ─────────────────────────────────────────────────

    # conf=X.XX
    line = re.sub(
        r'(conf=)([\d.]+)',
        lambda m: f'{YELLOW}conf={BYELLOW + BOLD}{m.group(2)}{RST}',
        line,
    )

    # RR=X.X
    line = re.sub(
        r'(RR=)([\d.]+)',
        lambda m: f'{CYAN}RR={BCYAN + BOLD}{m.group(2)}{RST}',
        line,
    )

    # SL=XXXX.XX or SL XXXX.XX
    line = re.sub(
        r'(SL[= ]+)([\d.]+)',
        lambda m: f'{RED}SL={BRED}{m.group(2)}{RST}',
        line,
    )

    # TP=XXXX.XX or TP XXXX.XX
    line = re.sub(
        r'(TP[= ]+)([\d.]+)',
        lambda m: f'{GREEN}TP={BGREEN}{m.group(2)}{RST}',
        line,
    )

    # ticket=XXXX
    line = re.sub(
        r'(ticket=)(\d+)',
        lambda m: f'ticket={BWHITE + BOLD}{m.group(2)}{RST}',
        line,
    )

    # lot=X.XX
    line = re.sub(
        r'(lot=)([\d.]+)',
        lambda m: f'{CYAN}lot={BCYAN}{m.group(2)}{RST}',
        line,
    )

    # signals=N [list]
    line = re.sub(
        r'(signals=)(\d+)',
        lambda m: f'{YELLOW}signals={BYELLOW + BOLD}{m.group(2)}{RST}',
        line,
    )

    # dist=N.Npt
    line = re.sub(
        r'(dist=)([\d.]+)(pt)',
        lambda m: f'{DIM}dist={WHITE}{m.group(2)}{DIM}pt{RST}',
        line,
    )

    # ── Profit / Loss ────────────────────────────────────────────────────

    # P/L=+N.Npt or P/L=-N.Npt
    def _fmt_pnl(m):
        val_str = m.group(1)
        if val_str.startswith('+'):
            return f'P/L={BGREEN + BOLD}{val_str}{RST}'
        elif val_str.startswith('-'):
            return f'P/L={BRED + BOLD}{val_str}{RST}'
        return f'P/L={WHITE}{val_str}{RST}'
    line = re.sub(r'P/L=([+-]?[\d.]+(?:pt)?)', _fmt_pnl, line)

    # profit_pts=+/-N.N
    def _fmt_profit(m):
        val_str = m.group(1)
        if float(val_str) >= 0:
            return f'{GREEN}profit_pts={BGREEN}{val_str}{RST}'
        return f'{RED}profit_pts={BRED}{val_str}{RST}'
    line = re.sub(r'profit_pts=([+-]?[\d.]+)', _fmt_profit, line)

    # ── Zone types ───────────────────────────────────────────────────────
    line = re.sub(
        r'\b(BEAR_(?:FVG|OB|BREAKER)|BOS_BEAR|CHOCH_BEAR)\b',
        lambda m: _c(BRED, m.group(1)),
        line,
    )
    line = re.sub(
        r'\b(BULL_(?:FVG|OB|BREAKER)|BOS_BULL|CHOCH_BULL)\b',
        lambda m: _c(BGREEN, m.group(1)),
        line,
    )

    # ── Exit review actions ──────────────────────────────────────────────
    line = re.sub(r'\bHOLD\b', _c(GREEN, 'HOLD'), line)
    # TIGHTEN (standalone, not CLAUDE TIGHTEN which is already colored)
    if "CLAUDE" not in line:
        line = re.sub(r'\bTIGHTEN\b', _c(BYELLOW, 'TIGHTEN'), line)

    # ── Connection info (fallback for lines not caught above) ──────────
    line = re.sub(
        r'(login=)(\d+)',
        lambda m: f'login={BWHITE + BOLD}{m.group(2)}{RST}',
        line,
    )
    line = re.sub(
        r'(?<!\033\[92m\033\[1m)(balance)',
        lambda m: _c(GREEN, 'balance'),
        line,
    )

    # ── Bias (catch remaining not in H4=/EMA= patterns) ────────────────
    # Avoid double-coloring by checking if already colored
    if '\033[92m' not in line or 'BULLISH' not in line.split('\033[92m')[-1][:20]:
        pass  # already handled by EMA=/H4= patterns above

    # ── Claude decisions ──────────────────────────────────────────────
    if "Claude >>" in line:
        line = re.sub(r'(Claude >>)', _c(BCYAN + BOLD, 'Claude >>'), line)
    if "APPROVED" in line:
        line = re.sub(r'(APPROVED)', _c(BGREEN + BOLD, 'APPROVED'), line)
    if "REJECTED" in line:
        line = re.sub(r'(REJECTED)', _c(BRED + BOLD, 'REJECTED'), line)

    # ── Skip / filter reasons ──────────────────────────────────────────
    if "Skip" in line:
        line = re.sub(r'(Skip)', _c(BYELLOW + BOLD, 'Skip'), line)
    if "counter-trend" in line:
        line = re.sub(r'(counter-trend)', _c(BRED, 'counter-trend'), line)
    if "cooldown" in line.lower():
        line = re.sub(r'(?i)(cooldown)', _c(BYELLOW, 'cooldown'), line)
    if "Spike window" in line:
        line = re.sub(r'(Spike window)', _c(BYELLOW, 'Spike window'), line)

    # ── Position stages ──────────────────────────────────────────────────
    line = re.sub(r'UNDERWATER', _c(BRED + BOLD, 'UNDERWATER'), line)
    line = re.sub(r'BREAK-EVEN', _c(BYELLOW, 'BREAK-EVEN'), line)
    line = re.sub(r'IN PROFIT', _c(BGREEN, 'IN PROFIT'), line)
    line = re.sub(r'TRAILING', _c(BCYAN, 'TRAILING'), line)

    # Prepend prefix if determined
    if prefix:
        line = f"{prefix} {line}"

    return line


def console_sink(message) -> None:
    """
    Custom loguru sink with colorized output.
    Receives a loguru Message object (str-like with .record attribute).
    Shows dual UTC + WIB timestamps when available (injected by logger_config patcher).
    """
    record = message.record
    level_name = record["level"].name
    level_style = _LEVEL_STYLE.get(level_name, WHITE)

    # Prefer patcher-injected UTC/WIB strings; fall back to record time
    utc_str = record["extra"].get("utc", record["time"].strftime("%H:%M:%S"))
    wib_str = record["extra"].get("wib", "")

    if wib_str:
        time_part = f"{DIM}{utc_str}{RST} {BWHITE}{wib_str}{RST}"
    else:
        time_part = f"{DIM}{utc_str}{RST}"

    raw_msg = record["message"]

    # Colorize the message content
    colored_msg = colorize_line(raw_msg)

    # Level badge (compact)
    if level_name == "WARNING":
        level_badge = f"{BG_BYELLOW}{BLACK}{BOLD} WARN {RST}"
    elif level_name == "ERROR":
        level_badge = f"{BG_BRED}{WHITE}{BOLD} ERR  {RST}"
    elif level_name == "CRITICAL":
        level_badge = f"{BG_BRED}{WHITE}{BOLD}{UNDERLINE} CRIT {RST}"
    else:
        level_badge = f"{level_style}{level_name:<5}{RST}"

    # Separator
    sep = f"{DIM}|{RST}"

    # Build the formatted line
    line = f"{time_part} {sep} {level_badge} {sep} {colored_msg}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
