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

# Level colors
_LEVEL_STYLE = {
    "DEBUG":    DIM,
    "INFO":     WHITE,
    "SUCCESS":  BGREEN + BOLD,
    "WARNING":  BYELLOW + BOLD,
    "ERROR":    BRED + BOLD,
    "CRITICAL": BRED + BOLD + UNDERLINE,
}


def _c(color: str, text: str) -> str:
    return f"{color}{text}{RST}"


def _rsi_color(val: float) -> str:
    if val > 80 or val < 20:
        return BRED + BOLD
    if val > 70 or val < 30:
        return BYELLOW
    return GREEN


def colorize_line(line: str) -> str:
    """Apply pattern-based ANSI colors to a log line."""

    # ── Banner / separator lines ─────────────────────────────────────────
    if "====" in line or line.strip().startswith("="):
        return _c(CYAN + BOLD, line)

    if "Smart Trader starting" in line:
        return _c(BGREEN + BOLD, line)

    if "Smart Trader stopped" in line:
        return _c(BRED + BOLD, line)

    # ── Cycle headers ────────────────────────────────────────────────────
    if "Cycle " in line and ("──" in line or "-- " in line):
        return _c(BWHITE + BOLD, line)

    if "Exit Review" in line and "──" in line:
        return _c(BMAGENTA + BOLD, line)

    # ── Trade execution (high-priority matches) ──────────────────────────
    if "ORDER FILLED" in line:
        line = re.sub(r'ORDER FILLED', _c(BGREEN + BOLD, 'ORDER FILLED'), line)

    if "PARTIAL CLOSE" in line:
        line = re.sub(r'PARTIAL CLOSE', _c(BYELLOW + BOLD, 'PARTIAL CLOSE'), line)

    if "PROFIT LOCK" in line:
        line = re.sub(r'PROFIT LOCK', _c(BYELLOW + BOLD, 'PROFIT LOCK'), line)

    if "CLAUDE EXIT" in line:
        line = re.sub(r'CLAUDE EXIT', _c(BRED + BOLD, 'CLAUDE EXIT'), line)

    if "CLAUDE TAKE PROFIT" in line:
        line = re.sub(r'CLAUDE TAKE PROFIT', _c(BGREEN + BOLD, 'CLAUDE TAKE PROFIT'), line)

    if "SCRATCH EXIT" in line:
        line = re.sub(r'SCRATCH EXIT', _c(BYELLOW + BOLD, 'SCRATCH EXIT'), line)

    if "STALE TIGHTEN" in line:
        line = re.sub(r'STALE TIGHTEN', _c(BYELLOW + BOLD, 'STALE TIGHTEN'), line)

    if "CLAUDE TIGHTEN" in line:
        line = re.sub(r'CLAUDE TIGHTEN', _c(BYELLOW + BOLD, 'CLAUDE TIGHTEN'), line)

    if "CLOSED" in line and "ticket=" in line:
        line = re.sub(r'CLOSED', _c(BRED + BOLD, 'CLOSED'), line)

    # ── Market data patterns ─────────────────────────────────────────────

    # Price=XXXX.XX
    line = re.sub(
        r'(Price=)([\d.]+)',
        lambda m: f'{CYAN}Price={BCYAN}{m.group(2)}{RST}',
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

    # Spread
    line = re.sub(
        r'(Spread=)([\d.]+)',
        lambda m: f'{DIM}Spread={WHITE}{m.group(2)}{RST}',
        line,
    )

    # EMA=TREND
    line = re.sub(
        r'(EMA=)(BULLISH|BEARISH|NEUTRAL)',
        lambda m: f'{MAGENTA}EMA={BGREEN + BOLD if m.group(2)=="BULLISH" else BRED + BOLD if m.group(2)=="BEARISH" else BYELLOW}{m.group(2)}{RST}',
        line,
    )

    # ── Session names ────────────────────────────────────────────────────
    line = re.sub(
        r'\b(ASIAN|LONDON|NEW_YORK|OVERLAP|LONDON_NY)\b',
        lambda m: _c(BBLUE + BOLD, m.group(1)),
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

    # ── Connection info ──────────────────────────────────────────────────
    line = re.sub(
        r'(login=)(\d+)',
        lambda m: f'login={BWHITE + BOLD}{m.group(2)}{RST}',
        line,
    )
    line = re.sub(
        r'(balance)',
        lambda m: _c(GREEN, 'balance'),
        line,
    )
    line = re.sub(
        r'(MT5 connected)',
        lambda m: _c(BGREEN, 'MT5 connected'),
        line,
    )

    # ── Bias ─────────────────────────────────────────────────────────────
    line = re.sub(r'\bBULLISH\b', _c(BGREEN, 'BULLISH'), line)
    line = re.sub(r'\bBEARISH\b', _c(BRED, 'BEARISH'), line)
    line = re.sub(r'\bRANGING\b', _c(BYELLOW, 'RANGING'), line)

    # ── Premium / Discount zones ─────────────────────────────────────────
    line = re.sub(r'\bPREMIUM\b', _c(BRED, 'PREMIUM'), line)
    line = re.sub(r'\bDISCOUNT\b', _c(BGREEN, 'DISCOUNT'), line)
    line = re.sub(r'\bEQUILIBRIUM\b', _c(BYELLOW, 'EQUILIBRIUM'), line)

    # ── Validation ───────────────────────────────────────────────────────
    if "Validating" in line:
        line = re.sub(r'(Validating)', _c(BCYAN + BOLD, 'Validating'), line)

    # ── Skip / filter reasons ──────────────────────────────────────────
    if "Skip" in line:
        line = re.sub(r'(Skip)', _c(BYELLOW, 'Skip'), line)
    if "counter-trend" in line:
        line = re.sub(r'(counter-trend)', _c(BRED, 'counter-trend'), line)
    if "cooldown" in line.lower():
        line = re.sub(r'(?i)(cooldown)', _c(BYELLOW, 'cooldown'), line)
    if "OFF_HOURS" in line and "no new entries" in line:
        line = re.sub(r'(no new entries)', _c(DIM + YELLOW, 'no new entries'), line)

    # ── Position stages ──────────────────────────────────────────────────
    line = re.sub(r'UNDERWATER', _c(BRED + BOLD, 'UNDERWATER'), line)
    line = re.sub(r'BREAK-EVEN', _c(BYELLOW, 'BREAK-EVEN'), line)
    line = re.sub(r'IN PROFIT', _c(BGREEN, 'IN PROFIT'), line)
    line = re.sub(r'TRAILING', _c(BCYAN, 'TRAILING'), line)

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
        time_part = f"UTC {utc_str} | WIB {wib_str}"
    else:
        time_part = utc_str

    raw_msg = record["message"]

    # Colorize the message content
    colored_msg = colorize_line(raw_msg)

    # Build the formatted line
    line = (
        f"{DIM}{time_part}{RST} "
        f"{level_style}{level_name:<8}{RST} "
        f"{colored_msg}\n"
    )
    sys.stdout.write(line)
    sys.stdout.flush()
