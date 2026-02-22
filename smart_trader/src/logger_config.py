"""
Structured logging for Smart Trader — 5 separate daily log sinks.

Log sinks:
  bot_activity/bot_YYYY-MM-DD.log      — all INFO+  (general activity)
  market/market_YYYY-MM-DD.log         — per-cycle scan data  (kind="MARKET")
  trades/trades_YYYY-MM-DD.log         — trade events          (kind="TRADE")
  trade_journal/journal_YYYY-MM-DD.log — structured per-trade  (kind="JOURNAL")
  errors/errors_YYYY-MM-DD.log         — WARNING+ from all

Usage in code:
  logger.info("normal message")                  → bot_activity
  logger.bind(kind="MARKET").debug("...")        → market log
  logger.bind(kind="TRADE").info("...")          → trades log
  logger.bind(kind="JOURNAL").info("...")        → trade_journal log
"""

import sys
from pathlib import Path
from datetime import timezone, timedelta
from loguru import logger

import console_format as cfmt

WIB = timezone(timedelta(hours=7))

# ── Format strings ────────────────────────────────────────────────────────────

_FMT_FULL = (
    "{time:YYYY-MM-DD} UTC {extra[utc]} | WIB {extra[wib]} | "
    "{level:<8} | {message}"
)
_FMT_SHORT = (
    "{extra[utc_date]} UTC {extra[utc]} | WIB {extra[wib]} | "
    "{level:<8} | {message}"
)


# ── Patcher — injects dual UTC+WIB timestamps into every record ───────────────

def _patch_dual_time(record: dict) -> None:
    """Inject UTC and WIB time strings into record['extra'] before formatting."""
    utc_dt = record["time"].astimezone(timezone.utc)
    wib_dt = utc_dt.astimezone(WIB)
    record["extra"]["utc_date"] = utc_dt.strftime("%Y-%m-%d")
    record["extra"]["utc"]      = utc_dt.strftime("%H:%M:%S")
    record["extra"]["wib"]      = wib_dt.strftime("%H:%M:%S")


# ── Filter helpers ────────────────────────────────────────────────────────────

def _is_activity(r: dict) -> bool:
    """Bot-activity sink: all records except MARKET and JOURNAL kinds."""
    return r["extra"].get("kind", "") not in ("MARKET", "JOURNAL")


def _is_market(r: dict) -> bool:
    return r["extra"].get("kind", "") == "MARKET"


def _is_trade(r: dict) -> bool:
    return r["extra"].get("kind", "") == "TRADE"


def _is_journal(r: dict) -> bool:
    return r["extra"].get("kind", "") == "JOURNAL"


# ── Main setup ────────────────────────────────────────────────────────────────

def setup_logging(cfg: dict) -> None:
    """
    Configure all log sinks with daily rotation.
    Call once at startup, before any log messages are emitted.
    """
    base = Path(cfg.get("paths", {}).get("log_dir", "logs"))

    # Create subdirectories
    for subdir in ("bot_activity", "market", "trades", "trade_journal", "errors"):
        (base / subdir).mkdir(parents=True, exist_ok=True)

    # Remove default loguru handler
    logger.remove()

    # Inject dual-time extras into every record
    logger.configure(patcher=_patch_dual_time)

    # ── 1. Console — colorized, activity + trade events only ─────────────────
    logger.add(
        cfmt.console_sink,
        level="INFO",
        colorize=False,          # color handling is done inside console_sink
        filter=_is_activity,
    )

    # ── 2. Bot Activity — all INFO+ excluding market/journal noise ────────────
    logger.add(
        str(base / "bot_activity" / "bot_{time:YYYY-MM-DD}.log"),
        format=_FMT_FULL,
        level="INFO",
        rotation="00:00",        # new file at midnight UTC
        retention="30 days",
        encoding="utf-8",
        filter=_is_activity,
    )

    # ── 3. Market — per-cycle scan data (DEBUG so we capture full detail) ─────
    logger.add(
        str(base / "market" / "market_{time:YYYY-MM-DD}.log"),
        format=_FMT_SHORT,
        level="DEBUG",
        rotation="00:00",
        retention="14 days",
        encoding="utf-8",
        filter=_is_market,
    )

    # ── 4. Trades — entry / exit / BE / trail / partial / scratch / stale ─────
    logger.add(
        str(base / "trades" / "trades_{time:YYYY-MM-DD}.log"),
        format=_FMT_SHORT,
        level="INFO",
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
        filter=_is_trade,
    )

    # ── 5. Trade Journal — structured ENTRY/EXIT/MILESTONE records ────────────
    logger.add(
        str(base / "trade_journal" / "journal_{time:YYYY-MM-DD}.log"),
        format=_FMT_SHORT,
        level="INFO",
        rotation="00:00",
        retention="90 days",
        encoding="utf-8",
        filter=_is_journal,
    )

    # ── 6. Errors — WARNING+ from all records ─────────────────────────────────
    logger.add(
        str(base / "errors" / "errors_{time:YYYY-MM-DD}.log"),
        format=_FMT_FULL,
        level="WARNING",
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
    )

    logger.info(
        "Logger configured | sinks: bot_activity / market / trades / trade_journal / errors"
    )
