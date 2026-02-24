"""
Trade Tracker — per-trade history with rolling performance metrics.
JSON persistence for adaptive engine feedback loop.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger


class TradeTracker:
    """Track trade history with rolling metrics for adaptive tuning."""

    def __init__(self, state_path: str = "data/trade_history.json", max_trades: int = 200):
        self.state_path = Path(state_path)
        self.max_trades = max_trades
        self.trades: list[dict] = []
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def record_entry(self, ticket: int, data: dict) -> None:
        """Append a new trade record on entry."""
        record = {
            "ticket": ticket,
            "status": "open",
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "exit_time": None,
            "direction": data.get("direction"),
            "entry_price": data.get("entry_price"),
            "sl": data.get("sl"),
            "tp": data.get("tp"),
            "lot": data.get("lot"),
            "zone_type": data.get("zone_type"),
            "signals": data.get("signals"),
            "signal_count": data.get("signal_count"),
            "confidence": data.get("confidence"),
            "claude_reason": data.get("claude_reason"),
            "session": data.get("session"),
            "regime": data.get("regime"),
            "ema_trend": data.get("ema_trend"),
            "rsi": data.get("rsi"),
            "atr": data.get("atr"),
            "pd_zone": data.get("pd_zone"),
            # Exit fields filled later
            "exit_price": None,
            "pnl_pts": None,
            "pnl_usd": None,
            "close_type": None,
            "duration_min": None,
        }
        self.trades.append(record)
        # Trim to max_trades
        if len(self.trades) > self.max_trades:
            self.trades = self.trades[-self.max_trades:]
        self._save()
        logger.debug(f"TradeTracker: recorded entry ticket={ticket}")

    def record_exit(self, ticket: int, data: dict) -> None:
        """Fill exit fields for an open trade by ticket."""
        for trade in reversed(self.trades):
            if trade["ticket"] == ticket and trade["status"] == "open":
                trade["status"] = "closed"
                trade["exit_time"] = datetime.now(timezone.utc).isoformat()
                trade["exit_price"] = data.get("exit_price")
                trade["pnl_pts"] = data.get("pnl_pts")
                trade["pnl_usd"] = data.get("pnl_usd")
                trade["close_type"] = data.get("close_type")
                trade["duration_min"] = data.get("duration_min")
                self._save()
                logger.debug(
                    f"TradeTracker: recorded exit ticket={ticket} | "
                    f"pnl={data.get('pnl_pts', 0):+.1f}pt"
                )
                return
        logger.debug(f"TradeTracker: exit ticket={ticket} not found in open trades")

    def get_closed_trades(self, last_n: int = 50) -> list[dict]:
        """Return most recent closed trades, newest first."""
        closed = [t for t in self.trades if t["status"] == "closed"]
        return list(reversed(closed[-last_n:]))

    def get_rolling_metrics(self, window: int = 50) -> dict:
        """Compute rolling performance metrics from last N closed trades."""
        closed = self.get_closed_trades(window)
        if not closed:
            return self._empty_metrics()

        total = len(closed)
        wins = [t for t in closed if (t.get("pnl_pts") or 0) > 0]
        losses = [t for t in closed if (t.get("pnl_pts") or 0) <= 0]
        win_rate = len(wins) / total if total > 0 else 0

        avg_winner = sum(t.get("pnl_pts", 0) for t in wins) / len(wins) if wins else 0
        avg_loser = abs(sum(t.get("pnl_pts", 0) for t in losses) / len(losses)) if losses else 0
        gross_profit = sum(t.get("pnl_pts", 0) for t in wins)
        gross_loss = abs(sum(t.get("pnl_pts", 0) for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.0

        avg_pnl = sum(t.get("pnl_pts", 0) for t in closed) / total if total > 0 else 0
        avg_rr = avg_winner / avg_loser if avg_loser > 0 else 0

        return {
            "total_trades": total,
            "win_rate": win_rate,
            "avg_rr": avg_rr,
            "profit_factor": profit_factor,
            "avg_pnl_pts": avg_pnl,
            "wins": len(wins),
            "losses": len(losses),
            "avg_winner_pts": avg_winner,
            "avg_loser_pts": avg_loser,
        }

    def get_metrics_by_key(self, key: str, window: int = 50) -> dict:
        """
        Group closed trades by a key (regime, session, zone_type, direction)
        and compute per-group metrics.
        """
        closed = self.get_closed_trades(window)
        groups: dict[str, list[dict]] = {}
        for t in closed:
            val = t.get(key) or "UNKNOWN"
            groups.setdefault(val, []).append(t)

        result = {}
        for group_name, trades in groups.items():
            total = len(trades)
            wins = [t for t in trades if (t.get("pnl_pts") or 0) > 0]
            losses = [t for t in trades if (t.get("pnl_pts") or 0) <= 0]
            gross_profit = sum(t.get("pnl_pts", 0) for t in wins)
            gross_loss = abs(sum(t.get("pnl_pts", 0) for t in losses))
            result[group_name] = {
                "count": total,
                "win_rate": len(wins) / total if total > 0 else 0,
                "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 99.0,
                "avg_pnl_pts": sum(t.get("pnl_pts", 0) for t in trades) / total if total > 0 else 0,
            }
        return result

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    self.trades = json.load(f)
                logger.info(f"TradeTracker: loaded {len(self.trades)} trades from {self.state_path}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"TradeTracker: failed to load {self.state_path}: {e}")
                self.trades = []

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(self.trades, f, indent=2, default=str)
        except IOError as e:
            logger.warning(f"TradeTracker: failed to save: {e}")

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "total_trades": 0, "win_rate": 0, "avg_rr": 0,
            "profit_factor": 0, "avg_pnl_pts": 0, "wins": 0,
            "losses": 0, "avg_winner_pts": 0, "avg_loser_pts": 0,
        }
