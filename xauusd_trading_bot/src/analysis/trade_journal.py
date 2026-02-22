"""
Trade Journal — per-ticket lifecycle tracker.

Writes two types of files:
  logs/trade_journal/YYYYMMDD_TICKET.json   — full lifecycle per ticket (updated live)
  logs/trade_journal/daily_YYYY-MM-DD.jsonl — append-only daily event stream

Snapshot interval: every 2 minutes (SNAPSHOT_INTERVAL_SECONDS).

Each ticket's JSON has the structure:
  {
    "ticket": 1488233709,
    "symbol": "XAUUSDm",
    "direction": "BUY",
    "volume": 0.01,
    "entry": { ts_utc, price, sl, tp, sl_pips, tp_pips, tp_rr, confluence,
               smc_signals, regime, session, balance },
    "snapshots": [
      { ts_utc, price, pnl_usd, rr, sl, stage, action }
      ...
    ],
    "exit": { ts_utc, price, pnl_usd, exit_reason, duration_minutes,
              mfe_usd, mae_usd, rr_final, stage_reached, snapshots_count }
  }

This is designed for post-trade analysis and future strategy improvement.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


class TradeJournal:
    """Per-ticket trade lifecycle journal with 2-minute snapshot throttle."""

    SNAPSHOT_INTERVAL_SECONDS = 120  # 2 minutes

    def __init__(self, log_dir: str = "logs/trade_journal"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # In-memory state (cleared on exit)
        self._journals: Dict[int, dict] = {}
        self._last_snapshot: Dict[int, datetime] = {}
        # Track file date for tickets opened near midnight
        self._ticket_date: Dict[int, str] = {}

    # ─── Public API ─────────────────────────────────────────────────────────

    def log_entry(self, ticket: int, data: dict) -> None:
        """
        Call immediately after a position is opened.

        Required keys in data:
          direction, price, sl, tp
        Optional keys:
          symbol, volume, confluence, smc_signals, regime, session, balance
        """
        now = datetime.now(timezone.utc)
        entry_price = float(data.get("price", 0))
        sl = float(data.get("sl", 0))
        tp = float(data.get("tp", 0))
        volume = float(data.get("volume", 0.01))

        sl_pips = round(abs(entry_price - sl), 2) if sl and entry_price else 0
        tp_pips = round(abs(tp - entry_price), 2) if tp and entry_price else 0
        tp_rr = round(tp_pips / sl_pips, 2) if sl_pips > 0 else 0

        date_str = now.strftime("%Y%m%d")
        journal = {
            "ticket": ticket,
            "symbol": data.get("symbol", "XAUUSDm"),
            "direction": data.get("direction", ""),
            "volume": volume,
            "entry": {
                "ts_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "price": round(entry_price, 2),
                "sl": round(sl, 2),
                "tp": round(tp, 2),
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "tp_rr": tp_rr,
                "confluence": round(float(data.get("confluence", 0)), 4),
                "smc_signals": data.get("smc_signals", ""),
                "regime": data.get("regime", ""),
                "session": data.get("session", ""),
                "balance": round(float(data.get("balance", 0)), 2),
            },
            "snapshots": [],
            "exit": None,
        }

        self._journals[ticket] = journal
        self._last_snapshot[ticket] = now
        self._ticket_date[ticket] = date_str

        self._save_journal(ticket)
        self._append_daily(date_str, {
            "event": "ENTRY",
            "ticket": ticket,
            **journal["entry"],
            "direction": journal["direction"],
            "volume": volume,
        })

    def log_snapshot(
        self,
        ticket: int,
        data: dict,
        force: bool = False,
    ) -> bool:
        """
        Call every main loop iteration. Returns True if snapshot was written.
        Throttled to SNAPSHOT_INTERVAL_SECONDS unless force=True.

        data keys:
          price     — current market price
          pnl_usd   — MT5 unrealized profit in USD
          rr        — current RR ratio (profit_distance / sl_distance)
          sl        — current SL price
          stage     — trade stage string (OPEN / BE_REACHED / TRAILING / ...)
          action    — optional: what action occurred this iteration (or None)
        """
        now = datetime.now(timezone.utc)

        if not force:
            last = self._last_snapshot.get(ticket)
            if last and (now - last).total_seconds() < self.SNAPSHOT_INTERVAL_SECONDS:
                return False

        journal = self._journals.get(ticket)
        if journal is None:
            return False

        snap = {
            "ts_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price": round(float(data.get("price", 0)), 2),
            "pnl_usd": round(float(data.get("pnl_usd", 0)), 2),
            "rr": round(float(data.get("rr", 0)), 3),
            "sl": round(float(data.get("sl", 0)), 2),
            "stage": data.get("stage", "OPEN"),
            "action": data.get("action"),  # None = no special action this interval
        }

        journal["snapshots"].append(snap)
        self._last_snapshot[ticket] = now

        self._save_journal(ticket)

        date_str = self._ticket_date.get(ticket, now.strftime("%Y%m%d"))
        self._append_daily(date_str, {"event": "SNAPSHOT", "ticket": ticket, **snap})
        return True

    def log_exit(self, ticket: int, data: dict) -> None:
        """
        Call immediately after a position is closed.

        data keys:
          price       — close price
          pnl_usd     — final realized P&L in USD
          exit_reason — string reason (e.g. "TP Hit", "SL Hit", "Near-SL early exit")
        """
        now = datetime.now(timezone.utc)
        journal = self._journals.get(ticket)

        final_pnl = round(float(data.get("pnl_usd", 0)), 2)
        close_price = round(float(data.get("price", 0)), 2)
        exit_reason = data.get("exit_reason", "")

        if journal is None:
            # Journal lost (e.g. bot restart) — write minimal exit record
            date_str = now.strftime("%Y%m%d")
            self._append_daily(date_str, {
                "event": "EXIT",
                "ticket": ticket,
                "ts_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "price": close_price,
                "pnl_usd": final_pnl,
                "exit_reason": exit_reason,
                "note": "journal_lost_on_restart",
            })
            return

        # Compute duration
        entry_ts = journal["entry"]["ts_utc"]
        try:
            entry_dt = datetime.strptime(entry_ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            duration_minutes = round((now - entry_dt).total_seconds() / 60)
        except Exception:
            duration_minutes = 0

        # MFE (max favorable) and MAE (max adverse) from snapshots + final close
        snapshots = journal["snapshots"]
        pnl_values = [s["pnl_usd"] for s in snapshots] + [final_pnl]
        mfe_usd = round(max(pnl_values), 2)
        mae_usd = round(min(pnl_values), 2)

        # Final RR using entry SL in price units
        # XAUUSD formula: pnl_per_price_unit = contract_size × volume = 100 × volume
        sl_pips = journal["entry"].get("sl_pips", 0)
        volume = journal.get("volume", 0.01)
        sl_usd = sl_pips * volume * 100  # expected loss if SL hit
        rr_final = round(final_pnl / sl_usd, 3) if sl_usd > 0 else 0

        last_stage = snapshots[-1]["stage"] if snapshots else "OPEN"

        exit_record = {
            "ts_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price": close_price,
            "pnl_usd": final_pnl,
            "exit_reason": exit_reason,
            "duration_minutes": duration_minutes,
            "mfe_usd": mfe_usd,
            "mae_usd": mae_usd,
            "rr_final": rr_final,
            "stage_reached": last_stage,
            "snapshots_count": len(snapshots),
        }

        journal["exit"] = exit_record
        self._save_journal(ticket)

        date_str = self._ticket_date.get(ticket, now.strftime("%Y%m%d"))
        self._append_daily(date_str, {"event": "EXIT", "ticket": ticket, **exit_record})

        # Cleanup memory
        self._journals.pop(ticket, None)
        self._last_snapshot.pop(ticket, None)
        self._ticket_date.pop(ticket, None)

    # ─── Queries ─────────────────────────────────────────────────────────────

    def get_open_tickets(self) -> list:
        """Return list of tickets currently tracked in-memory."""
        return list(self._journals.keys())

    def get_journal(self, ticket: int) -> Optional[dict]:
        """Return in-memory journal for a ticket, or None if not tracked."""
        return self._journals.get(ticket)

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _get_journal_path(self, ticket: int) -> Path:
        date_str = self._ticket_date.get(ticket, datetime.now(timezone.utc).strftime("%Y%m%d"))
        return self.log_dir / f"{date_str}_{ticket}.json"

    def _get_daily_path(self, date_str: str) -> Path:
        return self.log_dir / f"daily_{date_str}.jsonl"

    def _save_journal(self, ticket: int) -> None:
        """Atomically write the journal to disk (overwrite)."""
        journal = self._journals.get(ticket)
        if journal is None:
            return
        path = self._get_journal_path(ticket)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(journal, f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # Never crash the bot for journal I/O errors

    def _append_daily(self, date_str: str, record: dict) -> None:
        """Append one JSONL record to the daily event stream."""
        path = self._get_daily_path(date_str)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass
