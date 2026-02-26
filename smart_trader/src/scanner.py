"""
Zone Scanner — reads active structure levels from claude_trader's SQLite cache
and detects when price is near a tradeable zone.
"""
import sqlite3
from datetime import datetime, timezone
from typing import Optional
from loguru import logger


_BULL_TYPES = {"BULL_OB", "BULL_FVG", "BULL_BREAKER", "BOS_BULL", "CHOCH_BULL", "BULL_LIQSWEEP"}
_BEAR_TYPES = {"BEAR_OB", "BEAR_FVG", "BEAR_BREAKER", "BOS_BEAR", "CHOCH_BEAR", "BEAR_LIQSWEEP"}
# _SPIKE_WINDOWS = [
#     (7, 45, 8,  0),   # London spike
#     (12, 45, 13, 0),   # NY spike
# ]
_SPIKE_WINDOWS = []

_SESSIONS = [
    (12, 0, 16, 0,  "OVERLAP",   "HIGHEST"),
    (7,  0, 12, 0,  "LONDON",    "HIGH"),
    (16, 0, 20, 0,  "NEW_YORK",  "HIGH"),
    (20, 0, 22, 0,  "LATE_NY",   "MEDIUM"),
    (22, 0, 23, 0,  "OFF_HOURS", "MEDIUM"),   # IC Markets daily reset / wide spread window
    (23, 0, 24, 0,  "ASIAN",     "LOW"),
    (0,  0, 7,  0,  "ASIAN",     "LOW"),
]


def get_active_zones(db_path: str) -> list[dict]:
    """Read ACTIVE structure levels from claude_trader's market_cache.db."""
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT type, low, high, level, detected_at "
            "FROM structure_levels WHERE status='ACTIVE' "
            "ORDER BY detected_at DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"Zone cache read error: {e}")
        return []


def get_last_h4_bias(db_path: str) -> str:
    """Get most recent H4 bias from cycle_snapshots."""
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        row = conn.execute(
            "SELECT h4_bias FROM cycle_snapshots ORDER BY cycle DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return row[0] if row and row[0] else "RANGING"
    except Exception:
        return "RANGING"


def find_nearby_zones(
    price: float,
    zones: list[dict],
    proximity_pts: float,
) -> list[dict]:
    """
    Return zones where price is within proximity_pts of the zone boundary.
    Sorted by distance (closest first).
    """
    nearby = []
    for z in zones:
        z_type = z.get("type", "")
        low    = z.get("low")
        high   = z.get("high")
        level  = z.get("level")

        if low is not None and high is not None:
            if low - proximity_pts <= price <= high + proximity_pts:
                dist = 0 if low <= price <= high else min(
                    abs(price - low), abs(price - high)
                )
                nearby.append({**z, "distance_pts": round(dist, 2)})
        elif level is not None:
            dist = abs(price - level)
            if dist <= proximity_pts:
                nearby.append({**z, "distance_pts": round(dist, 2)})

    return sorted(nearby, key=lambda x: x["distance_pts"])


def nearest_zone(price: float, zones: list[dict]) -> Optional[dict]:
    """Return the single closest zone to price (no proximity filter)."""
    best = None
    best_dist = float("inf")
    for z in zones:
        low = z.get("low")
        high = z.get("high")
        level = z.get("level")

        if low is not None and high is not None:
            dist = 0 if low <= price <= high else min(
                abs(price - low), abs(price - high)
            )
        elif level is not None:
            dist = abs(price - level)
        else:
            continue

        if dist < best_dist:
            best_dist = dist
            best = {**z, "distance_pts": round(dist, 2)}
    return best


def direction_for_zone(zone: dict) -> Optional[str]:
    """Return LONG, SHORT, or None based on zone type."""
    z_type = zone.get("type", "")
    if z_type in _BULL_TYPES:
        return "LONG"
    if z_type in _BEAR_TYPES:
        return "SHORT"
    return None


def is_spike_window(dt_utc: Optional[datetime] = None) -> bool:
    """Return True if current UTC time is inside a spike avoidance window."""
    if dt_utc is None:
        dt_utc = datetime.now(timezone.utc)
    hm = dt_utc.hour * 60 + dt_utc.minute
    for sh, sm, eh, em in _SPIKE_WINDOWS:
        if sh * 60 + sm <= hm < eh * 60 + em:
            return True
    return False


def current_session(dt_utc: Optional[datetime] = None) -> dict:
    """Return current session info."""
    if dt_utc is None:
        dt_utc = datetime.now(timezone.utc)
    hm = dt_utc.hour * 60 + dt_utc.minute
    for sh, sm, eh, em, name, priority in _SESSIONS:
        e_min = eh * 60 + em if eh < 24 else 1440
        if sh * 60 + sm <= hm < e_min:
            return {"name": name, "priority": priority}
    return {"name": "OFF_HOURS", "priority": "MEDIUM"}


def check_risk_filters(
    account: dict,
    positions: list[dict],
    direction: str,
    max_positions: int,
    max_per_direction: int,
    max_drawdown_pct: float,
    free_margin_pct: float,
    estimated_margin: float = 50.0,
) -> tuple[bool, str]:
    """
    Check risk management filters before calling Claude.
    Returns (pass, reason_if_fail).
    """
    balance = account.get("balance", 0)
    equity  = account.get("equity", 0)
    margin_free = account.get("margin_free", 0)

    # Only count OUR positions (magic=202602), ignore manual trades
    our_pos = [p for p in positions if p.get("_raw") and p["_raw"].magic == 202602]

    # Max positions
    if len(our_pos) >= max_positions:
        return False, f"Max positions {max_positions} reached ({len(our_pos)} open)"

    # Max per direction
    same_dir = [p for p in our_pos if p.get("type") == direction]
    if len(same_dir) >= max_per_direction:
        return False, f"Already have {len(same_dir)} {direction} position(s)"

    # Drawdown
    if balance > 0:
        dd_pct = (balance - equity) / balance * 100
        if dd_pct > max_drawdown_pct:
            return False, f"Drawdown {dd_pct:.1f}% > {max_drawdown_pct}%"

    # Free margin
    if balance > 0:
        margin_ratio = margin_free / balance * 100
        if margin_ratio < free_margin_pct:
            return False, f"Low free margin {margin_ratio:.0f}%"

    return True, ""
