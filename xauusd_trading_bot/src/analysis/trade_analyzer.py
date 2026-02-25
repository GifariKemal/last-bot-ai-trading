"""
Trade History Analyzer
Analyzes CSV trade history to compute actionable statistics.
Called at bot startup for summary logging — NOT used for live decisions yet.

Key design: CLOSE rows are enriched by joining with their matching OPEN row
(same ticket) so session, smc_signals, regime, and entry_price are always
available for stats, regardless of what was stored in the CLOSE row.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from ..bot_logger import get_logger

# Abbreviation map: normalise whatever format was stored in the CSV
# Old format: FVG+BOS+LiqSw|C:0.71  -> "FVG+BOS+LiqSweep"
# New format: CH|LS|0.85             -> "CHoCH+LiqSweep"
SIGNAL_ABBREV = {
    "CH": "CHoCH",
    "LS": "LiqSweep",
    "LiqSw": "LiqSweep",
    "BS": "BOS",
    "BOS": "BOS",
    "FG": "FVG",
    "FVG": "FVG",
    "OB": "OB",
    "CHoCH": "CHoCH",
    "LiqSweep": "LiqSweep",
}


def _normalize_smc_combo(smc_str: str) -> str:
    """
    Convert raw smc_signals string from CSV to a clean combo label.

    Examples:
      "CH|LS|0.85"           -> "CHoCH+LiqSweep"
      "FVG+BOS+LiqSw|C:0.71" -> "BOS+FVG+LiqSweep"
      "BOS"                   -> "BOS"
      ""                      -> "Unknown"
    """
    if not smc_str:
        return "Unknown"

    # Take everything before the first | or : that contains a float (confidence)
    # Strategy: split on | and take parts that are NOT pure floats / labels
    raw_parts = smc_str.replace("|", "+").split("+")
    signals = []
    for p in raw_parts:
        p = p.strip()
        if not p:
            continue
        # Skip confidence markers like "0.85", "C:0.71", "0.6"
        if p.startswith("C:"):
            continue
        try:
            float(p)
            continue  # pure float → skip
        except ValueError:
            pass
        # Normalise abbreviation
        normed = SIGNAL_ABBREV.get(p, p)
        if normed and normed not in signals:
            signals.append(normed)

    if not signals:
        return "Unknown"
    return "+".join(sorted(signals))  # sorted for consistency across formats


class TradeAnalyzer:
    """Analyze trade history from CSV files."""

    def __init__(self, trade_history_dir: str = "data/trade_history"):
        self.logger = get_logger()
        self.trade_history_dir = Path(trade_history_dir)
        self.trades: List[Dict] = []

    def load_trades(self, days_back: int = 90) -> int:
        """
        Load trades from CSV files.

        Returns:
            Number of raw rows loaded (OPEN + CLOSE)
        """
        self.trades = []

        if not self.trade_history_dir.exists():
            self.logger.info("No trade history directory found")
            return 0

        cutoff_date = datetime.utcnow() - timedelta(days=days_back)

        csv_files = sorted(self.trade_history_dir.glob("trades_*.csv"))
        files_loaded = 0
        for csv_path in csv_files:
            try:
                date_str = csv_path.stem.replace("trades_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < cutoff_date:
                    continue

                with open(csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.trades.append(row)
                files_loaded += 1
            except Exception as e:
                self.logger.debug(f"Error loading {csv_path}: {e}")

        self.logger.info(f"Loaded {len(self.trades)} trade records from {files_loaded} files")
        return len(self.trades)

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _build_open_map(self) -> Dict[str, dict]:
        """
        Build lookup: ticket -> OPEN row.
        Used to enrich CLOSE rows with session, signals, entry price, regime.
        """
        open_map: Dict[str, dict] = {}
        for t in self.trades:
            if t.get("type") == "OPEN":
                ticket = t.get("ticket", "")
                if ticket:
                    open_map[ticket] = t
        return open_map

    def get_closed_trades(self) -> List[Dict]:
        """
        Get CLOSE rows enriched with OPEN-row context (session, smc_signals, regime).
        Joins by ticket so stats are always complete even when CLOSE rows have
        empty columns (which was the bug causing 'Unknown' everywhere).
        """
        open_map = self._build_open_map()
        closed = []

        for trade in self.trades:
            if trade.get("type") != "CLOSE":
                continue

            profit_str = trade.get("profit", "")
            try:
                profit = float(profit_str) if profit_str else 0.0
            except (ValueError, TypeError):
                profit = 0.0

            ticket = trade.get("ticket", "")
            open_row = open_map.get(ticket, {})

            # Session: CLOSE row first (usually empty), fallback to OPEN row
            session = (trade.get("session") or "").strip() or \
                      (open_row.get("session") or "").strip() or "Unknown"

            # SMC signals: prefer OPEN row (always has it; CLOSE row is often empty)
            raw_smc = (open_row.get("smc_signals") or trade.get("smc_signals") or "").strip()
            smc_combo = _normalize_smc_combo(raw_smc)

            # Regime: from OPEN row (new CSV field added 2026-02-22+)
            regime = (open_row.get("regime") or "").strip() or "Unknown"

            # Confluence: from OPEN row
            try:
                confluence = float(open_row.get("confluence") or trade.get("confluence") or 0)
            except (ValueError, TypeError):
                confluence = 0.0

            # Entry / exit timestamps for duration calc
            open_date = open_row.get("date", "")
            open_time = open_row.get("time_utc", "")
            close_date = trade.get("date", "")
            close_time = trade.get("time_utc", "")

            duration_minutes = None
            if open_date and open_time and close_date and close_time:
                try:
                    open_dt = datetime.strptime(f"{open_date} {open_time}", "%Y-%m-%d %H:%M:%S")
                    close_dt = datetime.strptime(f"{close_date} {close_time}", "%Y-%m-%d %H:%M:%S")
                    duration_minutes = max(0, int((close_dt - open_dt).total_seconds() / 60))
                except Exception:
                    pass

            # Entry / exit prices for pips calculation
            try:
                entry_price = float(open_row.get("price") or 0)
            except (ValueError, TypeError):
                entry_price = 0.0
            try:
                exit_price = float(trade.get("price") or 0)
            except (ValueError, TypeError):
                exit_price = 0.0

            is_breakeven = abs(profit) < 0.50
            closed.append({
                **trade,
                "profit_num": profit,
                "is_win": profit > 0 and not is_breakeven,
                "is_loss": profit < 0 and not is_breakeven,
                "is_breakeven": is_breakeven,
                # Enriched from OPEN row
                "session": session,
                "smc_combo": smc_combo,
                "regime": regime,
                "confluence": confluence,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "duration_minutes": duration_minutes,
            })
        return closed

    # ─── Stats computation ────────────────────────────────────────────────────

    def compute_overall_stats(self) -> Dict:
        """Compute overall trading statistics."""
        closed = self.get_closed_trades()
        if not closed:
            return {"total_trades": 0, "message": "No closed trades found"}

        wins = [t for t in closed if t["is_win"]]
        losses = [t for t in closed if t["is_loss"]]
        breakevens = [t for t in closed if t["is_breakeven"]]

        total = len(closed)
        win_count = len(wins)
        loss_count = len(losses)

        win_rate = win_count / total if total > 0 else 0

        avg_win = sum(t["profit_num"] for t in wins) / win_count if wins else 0
        avg_loss = sum(t["profit_num"] for t in losses) / loss_count if losses else 0

        gross_profit = sum(t["profit_num"] for t in wins)
        gross_loss = abs(sum(t["profit_num"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        net_profit = sum(t["profit_num"] for t in closed)

        # Duration stats
        durations = [t["duration_minutes"] for t in closed if t["duration_minutes"] is not None]
        avg_duration = sum(durations) / len(durations) if durations else None

        # Best and worst trades
        best = max(closed, key=lambda t: t["profit_num"]) if closed else None
        worst = min(closed, key=lambda t: t["profit_num"]) if closed else None

        # Max consecutive losses
        max_consec_loss = 0
        cur_consec = 0
        for t in closed:
            if t["is_loss"]:
                cur_consec += 1
                max_consec_loss = max(max_consec_loss, cur_consec)
            else:
                cur_consec = 0

        return {
            "total_trades": total,
            "wins": win_count,
            "losses": loss_count,
            "breakevens": len(breakevens),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "net_profit": net_profit,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "avg_duration_minutes": avg_duration,
            "best_trade": {
                "ticket": best.get("ticket", ""),
                "profit": best["profit_num"],
                "combo": best.get("smc_combo", ""),
                "session": best.get("session", ""),
            } if best and best["profit_num"] > 0 else None,
            "worst_trade": {
                "ticket": worst.get("ticket", ""),
                "profit": worst["profit_num"],
                "combo": worst.get("smc_combo", ""),
                "session": worst.get("session", ""),
            } if worst and worst["profit_num"] < 0 else None,
            "max_consecutive_losses": max_consec_loss,
        }

    def compute_session_stats(self) -> Dict[str, Dict]:
        """Compute statistics per trading session."""
        closed = self.get_closed_trades()
        if not closed:
            return {}

        by_session = defaultdict(list)
        for trade in closed:
            by_session[trade["session"]].append(trade)

        stats = {}
        for session, trades in by_session.items():
            wins = [t for t in trades if t["is_win"]]
            losses = [t for t in trades if t["is_loss"]]

            total = len(trades)
            win_count = len(wins)

            gross_profit = sum(t["profit_num"] for t in wins)
            gross_loss = abs(sum(t["profit_num"] for t in losses))
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            stats[session] = {
                "total": total,
                "wins": win_count,
                "losses": len(losses),
                "win_rate": win_count / total if total > 0 else 0,
                "avg_win": sum(t["profit_num"] for t in wins) / win_count if wins else 0,
                "avg_loss": sum(t["profit_num"] for t in losses) / len(losses) if losses else 0,
                "profit_factor": pf,
                "net_profit": sum(t["profit_num"] for t in trades),
            }

        return stats

    def compute_smc_combo_stats(self) -> Dict[str, Dict]:
        """Compute win rate by SMC signal combination."""
        closed = self.get_closed_trades()
        if not closed:
            return {}

        by_combo = defaultdict(list)
        for trade in closed:
            by_combo[trade["smc_combo"]].append(trade)

        stats = {}
        for combo, trades in by_combo.items():
            wins = [t for t in trades if t["is_win"]]
            total = len(trades)
            net = sum(t["profit_num"] for t in trades)

            stats[combo] = {
                "total": total,
                "wins": len(wins),
                "win_rate": len(wins) / total if total > 0 else 0,
                "net_profit": net,
                "avg_profit": net / total if total > 0 else 0,
            }

        return dict(sorted(stats.items(), key=lambda x: x[1]["net_profit"], reverse=True))

    def compute_regime_stats(self) -> Dict[str, Dict]:
        """Compute win rate by market regime."""
        closed = self.get_closed_trades()
        if not closed:
            return {}

        by_regime = defaultdict(list)
        for trade in closed:
            r = trade.get("regime") or "Unknown"
            by_regime[r].append(trade)

        stats = {}
        for regime, trades in by_regime.items():
            wins = [t for t in trades if t["is_win"]]
            total = len(trades)
            gross_profit = sum(t["profit_num"] for t in wins)
            gross_loss = abs(sum(t["profit_num"] for t in trades if t["is_loss"]))
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            stats[regime] = {
                "total": total,
                "wins": len(wins),
                "losses": len([t for t in trades if t["is_loss"]]),
                "win_rate": len(wins) / total if total > 0 else 0,
                "profit_factor": pf,
                "net_profit": sum(t["profit_num"] for t in trades),
            }

        return dict(sorted(stats.items(), key=lambda x: x[1]["net_profit"], reverse=True))

    def compute_time_stats(self) -> Dict:
        """Compute time-of-day performance stats."""
        closed = self.get_closed_trades()
        if not closed:
            return {}

        by_hour = defaultdict(list)
        for trade in closed:
            time_utc = trade.get("time_utc", "")
            if time_utc:
                try:
                    hour = int(time_utc.split(":")[0])
                    by_hour[hour].append(trade)
                except (ValueError, IndexError):
                    pass

        hourly_stats = {}
        for hour in sorted(by_hour.keys()):
            trades = by_hour[hour]
            wins = [t for t in trades if t["is_win"]]
            net = sum(t["profit_num"] for t in trades)

            hourly_stats[f"{hour:02d}:00"] = {
                "total": len(trades),
                "wins": len(wins),
                "win_rate": len(wins) / len(trades) if trades else 0,
                "net_profit": net,
            }

        return hourly_stats

    def get_full_analysis(self) -> Dict:
        """Run complete trade analysis."""
        total_loaded = self.load_trades()
        if total_loaded == 0:
            return {"loaded": 0, "message": "No trade history available"}

        return {
            "loaded": total_loaded,
            "overall": self.compute_overall_stats(),
            "by_session": self.compute_session_stats(),
            "by_smc_combo": self.compute_smc_combo_stats(),
            "by_regime": self.compute_regime_stats(),
            "by_hour": self.compute_time_stats(),
        }

    def get_summary_text(self, analysis: dict = None) -> str:
        """
        Get human-readable summary for logging.

        Pass pre-computed analysis dict to avoid reloading CSV files.
        If None, will call get_full_analysis() internally.
        """
        if analysis is None:
            analysis = self.get_full_analysis()

        if analysis.get("loaded", 0) == 0:
            return "No trade history available yet."

        overall = analysis.get("overall", {})
        total = overall.get("total_trades", 0)
        if total == 0:
            return "No closed trades yet."

        wins = overall.get("wins", 0)
        losses = overall.get("losses", 0)
        bes = overall.get("breakevens", 0)
        wr = overall.get("win_rate", 0)
        pf = overall.get("profit_factor", 0)
        net = overall.get("net_profit", 0)
        avg_win = overall.get("avg_win", 0)
        avg_loss = abs(overall.get("avg_loss", 0))
        avg_dur = overall.get("avg_duration_minutes")
        max_cl = overall.get("max_consecutive_losses", 0)

        lines = [
            f"Total Trades: {total}",
            f"W/L/BE: {wins}/{losses}/{bes}",
            f"Win Rate: {wr:.0%}",
            f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}",
            f"Profit Factor: {pf:.2f}",
            f"Net P/L: ${net:.2f}",
        ]
        if avg_dur is not None:
            lines.append(f"Avg Duration: {avg_dur:.0f} min")
        if max_cl > 0:
            lines.append(f"Max Consec Losses: {max_cl}")

        # Per session
        by_session = analysis.get("by_session", {})
        if by_session:
            lines.append("\nPer Session:")
            for sname, s in by_session.items():
                pf_s = s["profit_factor"]
                pf_str = f"{pf_s:.2f}" if pf_s != float("inf") else "∞"
                lines.append(
                    f"  {sname}: {s['wins']}/{s['total']} "
                    f"({s['win_rate']:.0%}) | PF:{pf_str} | ${s['net_profit']:.2f}"
                )

        # SMC combos
        by_combo = analysis.get("by_smc_combo", {})
        if by_combo:
            lines.append("\nSMC Combos (top 5):")
            for combo, cs in list(by_combo.items())[:5]:
                lines.append(f"  {combo}: {cs['wins']}/{cs['total']} ({cs['win_rate']:.0%}) | ${cs['net_profit']:.2f}")

        return "\n".join(lines)
