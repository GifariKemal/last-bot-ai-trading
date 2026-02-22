"""
Market Cache — SQLite-backed persistence for H1 structure analysis.

Stores:
  - structure_levels : OB / FVG / BOS / CHoCH identified by Claude each cycle
  - cycle_snapshots  : per-cycle summary (h4_bias, h1_trend, price, atr, action)

Benefits:
  - Structure levels survive restarts and are injected into the next cycle prompt
  - OBs/FVGs are tracked as ACTIVE → MITIGATED automatically (price vs zone)
  - Claude no longer needs to re-identify structure from 100 raw candles; it only
    checks new candles and verifies/updates the cached levels
  - Prompt shrinks ~60 %, response is faster and more consistent

Structure-update protocol:
  Claude must end each cycle report with a STRUCTURE_UPDATE JSON block.
  Python parses it and updates the DB. Next cycle prompt injects DB contents.
"""

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "data" / "market_cache.db"
DB_PATH.parent.mkdir(exist_ok=True)

# ── Schema ─────────────────────────────────────────────────────────────────────
_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS structure_levels (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at  TEXT    NOT NULL,          -- ISO timestamp UTC
    cycle        INTEGER NOT NULL,
    type         TEXT    NOT NULL,          -- BULL_OB | BEAR_OB | BULL_FVG | BEAR_FVG
                                            -- BOS_BULL | BOS_BEAR | CHOCH_BULL | CHOCH_BEAR
                                            -- LIQ_BUY  | LIQ_SELL
    low          REAL,                      -- zone bottom  (NULL for point levels)
    high         REAL,                      -- zone top     (NULL for point levels)
    level        REAL,                      -- for point levels (BOS/CHoCH/Liq)
    status       TEXT    NOT NULL DEFAULT 'ACTIVE',   -- ACTIVE | MITIGATED | EXPIRED
    mitigated_at TEXT                       -- ISO timestamp, NULL if still active
);

CREATE TABLE IF NOT EXISTS cycle_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT    NOT NULL,
    cycle        INTEGER NOT NULL,
    price        REAL,
    atr          REAL,
    h4_bias      TEXT,
    h1_trend     TEXT,
    session      TEXT,
    action       TEXT,
    long_setup   TEXT,
    short_setup  TEXT,
    reason       TEXT
);

CREATE INDEX IF NOT EXISTS idx_struct_status ON structure_levels(status);
CREATE INDEX IF NOT EXISTS idx_snap_cycle    ON cycle_snapshots(cycle);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.executescript(_SCHEMA)


# ── Structure levels ───────────────────────────────────────────────────────────

def _check_mitigation(price: Optional[float]):
    """Mark levels as MITIGATED if current price is inside / has passed them."""
    if price is None:
        return
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        rows = c.execute(
            "SELECT id, type, low, high, level FROM structure_levels WHERE status='ACTIVE'"
        ).fetchall()
        for r in rows:
            mitigated = False
            if r["low"] is not None and r["high"] is not None:
                # Zone: mitigated when price enters the zone
                if r["low"] <= price <= r["high"]:
                    mitigated = True
            elif r["level"] is not None:
                # Point level: mitigated when price crosses it
                if r["type"] in ("BOS_BULL", "CHOCH_BULL", "LIQ_BUY"):
                    mitigated = price >= r["level"]
                else:
                    mitigated = price <= r["level"]
            if mitigated:
                c.execute(
                    "UPDATE structure_levels SET status='MITIGATED', mitigated_at=? WHERE id=?",
                    (now, r["id"]),
                )


def store_structure(cycle: int, levels: list[dict], price: Optional[float] = None):
    """
    Persist structure levels from a cycle.
    levels = list of dicts with keys: type, low, high, level, status
    """
    now = datetime.now(timezone.utc).isoformat()
    # Check mitigation with latest price first
    _check_mitigation(price)

    # Expire OLD point levels of same type that are now far from price (> 5×ATR)
    # — done implicitly: we keep all ACTIVE levels unless mitigated

    with _conn() as c:
        for lv in levels:
            # Avoid exact duplicates (same type + same price range)
            existing = c.execute(
                """SELECT id FROM structure_levels
                   WHERE type=? AND status='ACTIVE'
                     AND ABS(COALESCE(low,level,0)  - COALESCE(?,0)) < 1.0
                     AND ABS(COALESCE(high,level,0) - COALESCE(?,0)) < 1.0""",
                (lv.get("type"), lv.get("low", lv.get("level")),
                 lv.get("high", lv.get("level"))),
            ).fetchone()
            if existing:
                continue  # already stored, skip

            status = lv.get("status", "ACTIVE").upper()
            c.execute(
                """INSERT INTO structure_levels
                   (detected_at, cycle, type, low, high, level, status)
                   VALUES (?,?,?,?,?,?,?)""",
                (now, cycle,
                 lv.get("type", "UNKNOWN"),
                 lv.get("low"),
                 lv.get("high"),
                 lv.get("level"),
                 status),
            )


def get_active_levels() -> list[dict]:
    """Return all ACTIVE structure levels as list of dicts."""
    with _conn() as c:
        rows = c.execute(
            """SELECT type, low, high, level, detected_at
               FROM structure_levels WHERE status='ACTIVE'
               ORDER BY detected_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def expire_old_levels(max_cycles: int = 50):
    """Expire levels not updated in the last max_cycles cycles."""
    with _conn() as c:
        max_cycle = c.execute("SELECT MAX(cycle) FROM cycle_snapshots").fetchone()[0]
        if max_cycle is None:
            return
        cutoff = max_cycle - max_cycles
        c.execute(
            "UPDATE structure_levels SET status='EXPIRED' WHERE status='ACTIVE' AND cycle < ?",
            (cutoff,),
        )


# ── Cycle snapshots ────────────────────────────────────────────────────────────

def store_snapshot(data: dict):
    """Store one cycle's analysis summary."""
    with _conn() as c:
        c.execute(
            """INSERT INTO cycle_snapshots
               (timestamp, cycle, price, atr, h4_bias, h1_trend, session,
                action, long_setup, short_setup, reason)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                data.get("timestamp_utc", datetime.now(timezone.utc).isoformat()),
                data.get("total_cycle", 0),
                data.get("price"),
                data.get("atr"),
                data.get("h4_bias"),
                data.get("h1_trend"),
                data.get("session"),
                data.get("action"),
                data.get("long_setup"),
                data.get("short_setup"),
                data.get("reason"),
            ),
        )


# ── Parse STRUCTURE_UPDATE block from Claude output ───────────────────────────

def parse_structure_update(output: str) -> Optional[dict]:
    """
    Extract and parse the STRUCTURE_UPDATE JSON block from Claude output.
    Returns the parsed dict, or None if not found.
    """
    match = re.search(
        r'STRUCTURE_UPDATE\s*\{(.+?)\}\s*END_STRUCTURE_UPDATE',
        output, re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return None
    try:
        return json.loads("{" + match.group(1) + "}")
    except json.JSONDecodeError:
        return None


def apply_structure_update(cycle: int, output: str, price: Optional[float] = None):
    """Parse STRUCTURE_UPDATE from Claude output and persist to DB."""
    su = parse_structure_update(output)
    if su is None:
        return False

    levels = su.get("active_levels", [])
    # Add liquidity levels
    for liq in su.get("liquidity", {}).get("buy_side", []):
        levels.append({"type": "LIQ_BUY", "level": liq})
    for liq in su.get("liquidity", {}).get("sell_side", []):
        levels.append({"type": "LIQ_SELL", "level": liq})

    store_structure(cycle, levels, price=price)
    expire_old_levels()
    return True


# ── Format cached structure for prompt injection ──────────────────────────────

def format_cached_structure() -> str:
    """
    Format active structure levels as a human-readable block
    to inject into the Claude prompt.
    """
    levels = get_active_levels()
    if not levels:
        return ""

    groups: dict[str, list] = {}
    for lv in levels:
        t = lv["type"]
        groups.setdefault(t, []).append(lv)

    lines = [
        "",
        "=" * 60,
        "CACHED STRUCTURE LEVELS (identified in previous cycles — verified still ACTIVE)",
        "=" * 60,
        "These levels were identified by previous analyses. Use them directly.",
        "Only identify NEW structure from candles since the last cycle.",
        "Mark a level as MITIGATED in STRUCTURE_UPDATE if price has passed through it.",
        "",
    ]

    label = {
        "BULL_OB":      "Bullish Order Blocks",
        "BEAR_OB":      "Bearish Order Blocks",
        "BULL_FVG":     "Bullish FVGs",
        "BEAR_FVG":     "Bearish FVGs",
        "BULL_BREAKER": "Bull Breaker Blocks (flipped Bear OB → support)",
        "BEAR_BREAKER": "Bear Breaker Blocks (flipped Bull OB → resistance)",
        "BOS_BULL":     "Bullish BOS Levels",
        "BOS_BEAR":     "Bearish BOS Levels",
        "CHOCH_BULL":   "Bullish CHoCH Levels",
        "CHOCH_BEAR":   "Bearish CHoCH Levels",
        "LIQ_BUY":      "Buy-side Liquidity",
        "LIQ_SELL":     "Sell-side Liquidity",
    }

    for t, lvs in groups.items():
        lines.append(f"  {label.get(t, t)}:")
        for lv in lvs:
            if lv["low"] is not None and lv["high"] is not None:
                lines.append(f"    • {lv['low']:.2f} – {lv['high']:.2f}")
            elif lv["level"] is not None:
                lines.append(f"    • {lv['level']:.2f}")
        lines.append("")

    lines += ["END CACHED STRUCTURE", "=" * 60, ""]
    return "\n".join(lines)


# ── Stats helper ───────────────────────────────────────────────────────────────

def get_recent_snapshots(n: int = 7) -> list[dict]:
    """Return last n cycle snapshots (oldest→newest) for compact intel injection."""
    with _conn() as c:
        rows = c.execute(
            """SELECT timestamp, cycle, price, atr, h4_bias, h1_trend,
                      session, action, long_setup, short_setup, reason
               FROM cycle_snapshots ORDER BY cycle DESC LIMIT ?""",
            (n,),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_stats() -> dict:
    """Return quick stats about the cache."""
    with _conn() as c:
        total_cycles   = c.execute("SELECT COUNT(*) FROM cycle_snapshots").fetchone()[0]
        active_levels  = c.execute("SELECT COUNT(*) FROM structure_levels WHERE status='ACTIVE'").fetchone()[0]
        total_levels   = c.execute("SELECT COUNT(*) FROM structure_levels").fetchone()[0]
        trades         = c.execute(
            "SELECT COUNT(*) FROM cycle_snapshots WHERE action LIKE '%ENTERED%'"
        ).fetchone()[0]
    return {
        "total_cycles":  total_cycles,
        "active_levels": active_levels,
        "total_levels":  total_levels,
        "trades_taken":  trades,
    }
