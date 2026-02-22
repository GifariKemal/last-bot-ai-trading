"""
Position Tracker
Tracks and monitors all open positions with detailed metrics.
"""

from typing import Dict, List, Optional
from datetime import datetime
from ..bot_logger import get_logger


class PositionTracker:
    """Track open positions with detailed metrics."""

    def __init__(self):
        """Initialize position tracker."""
        self.logger = get_logger()
        self.positions: Dict[int, Dict] = {}  # ticket_id -> position_data
        self.position_history: List[Dict] = []  # Closed positions
        self._recently_closed: Dict[int, datetime] = {}  # ticket -> close_time (debounce)

    def add_position(self, position: Dict) -> None:
        """
        Add a new position to tracking.

        Args:
            position: Position data from MT5
        """
        ticket = position.get("ticket")
        if not ticket:
            self.logger.warning("Cannot track position without ticket ID")
            return

        # Enrich position data
        tracked_position = {
            **position,
            "tracked_since": datetime.now(),
            "initial_price": position.get("open_price") or position.get("entry_price") or position.get("price"),
            "peak_profit": 0.0,
            "max_drawdown": 0.0,
            "updates": [],
            "status": "active",
        }

        self.positions[ticket] = tracked_position
        price = position.get("open_price") or position.get("entry_price") or position.get("price", 0)
        self.logger.info(
            f"Tracking new position: {ticket} ({position.get('type') or position.get('direction')}) "
            f"@ {price:.2f}"
        )

    def update_position(self, ticket: int, updates: Dict) -> None:
        """
        Update position data.

        Args:
            ticket: Position ticket ID
            updates: Dictionary of fields to update
        """
        if ticket not in self.positions:
            self.logger.warning(f"Position {ticket} not tracked")
            return

        # Record update
        update_entry = {
            "timestamp": datetime.now(),
            "updates": updates.copy(),
        }
        self.positions[ticket]["updates"].append(update_entry)

        # Apply updates
        self.positions[ticket].update(updates)

        self.logger.debug(f"Updated position {ticket}: {list(updates.keys())}")

    def update_position_metrics(
        self,
        ticket: int,
        current_price: float,
        current_profit: float = None
    ) -> None:
        """
        Update position metrics (profit, drawdown, etc.).

        Args:
            ticket: Position ticket ID
            current_price: Current market price
            current_profit: Current profit/loss
        """
        if ticket not in self.positions:
            return

        position = self.positions[ticket]
        entry_price = position.get("open_price", 0)
        position_type = position.get("type", "").upper()

        # Calculate profit in pips
        if position_type == "BUY":
            profit_pips = current_price - entry_price
        else:  # SELL
            profit_pips = entry_price - current_price

        # Update current price and profit
        position["current_price"] = current_price
        position["current_profit_pips"] = profit_pips

        if current_profit is not None:
            position["profit"] = current_profit

        # Update peak profit
        if profit_pips > position.get("peak_profit", 0):
            position["peak_profit"] = profit_pips
            position["peak_profit_time"] = datetime.now()

        # Update max drawdown
        peak = position.get("peak_profit", 0)
        if peak > 0:
            drawdown = peak - profit_pips
            if drawdown > position.get("max_drawdown", 0):
                position["max_drawdown"] = drawdown

        # Calculate time in position
        tracked_since = position.get("tracked_since", datetime.now())
        position["time_in_position"] = (datetime.now() - tracked_since).total_seconds()

    def remove_position(self, ticket: int, close_data: Dict = None) -> Optional[Dict]:
        """
        Remove position from tracking (when closed).

        Args:
            ticket: Position ticket ID
            close_data: Optional close information

        Returns:
            Final position data
        """
        if ticket not in self.positions:
            return None

        position = self.positions[ticket]
        position["status"] = "closed"
        position["closed_at"] = datetime.now()

        if close_data:
            position.update(close_data)

        # Move to history
        self.position_history.append(position)

        # Keep only last 100 in history
        if len(self.position_history) > 100:
            self.position_history = self.position_history[-100:]

        # Remove from active
        del self.positions[ticket]

        # Track as recently closed to prevent re-tracking
        self._recently_closed[ticket] = datetime.now()

        self.logger.info(
            f"Position {ticket} closed. Final P&L: ${position.get('profit', 0):.2f}"
        )

        return position

    def get_position(self, ticket: int) -> Optional[Dict]:
        """
        Get position data.

        Args:
            ticket: Position ticket ID

        Returns:
            Position data or None
        """
        return self.positions.get(ticket)

    def get_all_positions(self) -> List[Dict]:
        """
        Get all tracked positions.

        Returns:
            List of position data
        """
        return list(self.positions.values())

    def get_positions_by_type(self, position_type: str) -> List[Dict]:
        """
        Get positions by type.

        Args:
            position_type: "BUY" or "SELL"

        Returns:
            List of positions
        """
        return [
            p for p in self.positions.values()
            if p.get("type", "").upper() == position_type.upper()
        ]

    def get_position_count(self) -> int:
        """
        Get count of tracked positions.

        Returns:
            Number of positions
        """
        return len(self.positions)

    def get_total_exposure(self) -> Dict:
        """
        Get total exposure metrics.

        Returns:
            Exposure dictionary
        """
        total_lots = sum(p.get("volume", 0) for p in self.positions.values())
        total_profit = sum(p.get("profit", 0) for p in self.positions.values())

        buy_positions = self.get_positions_by_type("BUY")
        sell_positions = self.get_positions_by_type("SELL")

        buy_lots = sum(p.get("volume", 0) for p in buy_positions)
        sell_lots = sum(p.get("volume", 0) for p in sell_positions)

        return {
            "total_positions": len(self.positions),
            "total_lots": total_lots,
            "total_profit": total_profit,
            "buy_positions": len(buy_positions),
            "buy_lots": buy_lots,
            "sell_positions": len(sell_positions),
            "sell_lots": sell_lots,
            "net_direction": "LONG" if buy_lots > sell_lots else "SHORT" if sell_lots > buy_lots else "NEUTRAL",
            "net_exposure": abs(buy_lots - sell_lots),
        }

    def get_position_stats(self, ticket: int) -> Optional[Dict]:
        """
        Get detailed statistics for a position.

        Args:
            ticket: Position ticket ID

        Returns:
            Statistics dictionary
        """
        position = self.get_position(ticket)
        if not position:
            return None

        entry_price = position.get("open_price", 0)
        current_price = position.get("current_price", entry_price)
        position_type = position.get("type", "").upper()

        # Calculate metrics
        if position_type == "BUY":
            pips_from_entry = current_price - entry_price
        else:
            pips_from_entry = entry_price - current_price

        sl = position.get("sl", 0)
        tp = position.get("tp", 0)

        if position_type == "BUY":
            pips_to_sl = current_price - sl if sl else None
            pips_to_tp = tp - current_price if tp else None
        else:
            pips_to_sl = sl - current_price if sl else None
            pips_to_tp = current_price - tp if tp else None

        time_in_position = position.get("time_in_position", 0)

        return {
            "ticket": ticket,
            "type": position_type,
            "entry_price": entry_price,
            "current_price": current_price,
            "pips_from_entry": pips_from_entry,
            "pips_to_sl": pips_to_sl,
            "pips_to_tp": pips_to_tp,
            "current_profit": position.get("profit", 0),
            "peak_profit": position.get("peak_profit", 0),
            "max_drawdown": position.get("max_drawdown", 0),
            "time_in_position_seconds": time_in_position,
            "time_in_position_hours": time_in_position / 3600,
            "sl": sl,
            "tp": tp,
            "volume": position.get("volume", 0),
        }

    def get_losing_positions(self) -> List[Dict]:
        """
        Get positions currently in loss.

        Returns:
            List of losing positions
        """
        return [
            p for p in self.positions.values()
            if p.get("profit", 0) < 0
        ]

    def get_winning_positions(self) -> List[Dict]:
        """
        Get positions currently in profit.

        Returns:
            List of winning positions
        """
        return [
            p for p in self.positions.values()
            if p.get("profit", 0) > 0
        ]

    def get_oldest_position(self) -> Optional[Dict]:
        """
        Get the oldest open position.

        Returns:
            Position data or None
        """
        if not self.positions:
            return None

        return min(
            self.positions.values(),
            key=lambda p: p.get("tracked_since", datetime.now())
        )

    def get_position_summary(self) -> str:
        """
        Get human-readable summary of all positions.

        Returns:
            Summary string
        """
        if not self.positions:
            return "No open positions"

        exposure = self.get_total_exposure()

        lines = [
            f"Open Positions: {exposure['total_positions']}",
            f"  BUY: {exposure['buy_positions']} ({exposure['buy_lots']:.2f} lots)",
            f"  SELL: {exposure['sell_positions']} ({exposure['sell_lots']:.2f} lots)",
            f"Total P&L: ${exposure['total_profit']:.2f}",
        ]

        # Add individual positions
        for position in self.positions.values():
            ticket = position.get("ticket")
            pos_type = position.get("type")
            entry = position.get("open_price", 0)
            profit = position.get("profit", 0)
            lines.append(
                f"  #{ticket} {pos_type} @ {entry:.2f} | P&L: ${profit:.2f}"
            )

        return "\n".join(lines)

    def clear_all(self) -> None:
        """Clear all tracked positions (use with caution)."""
        count = len(self.positions)
        self.positions.clear()
        self.logger.warning(f"Cleared all {count} tracked positions")

    def sync_with_mt5(self, mt5_positions: List[Dict]) -> Dict:
        """
        Synchronize tracked positions with MT5 positions.

        Args:
            mt5_positions: List of positions from MT5

        Returns:
            Sync report
        """
        mt5_tickets = {p.get("ticket") for p in mt5_positions}
        tracked_tickets = set(self.positions.keys())

        # Clean up stale recently_closed entries (older than 60 seconds)
        now = datetime.now()
        stale_keys = [
            t for t, ts in self._recently_closed.items()
            if (now - ts).total_seconds() > 60
        ]
        for k in stale_keys:
            del self._recently_closed[k]

        # Find positions to add (in MT5 but not tracked, and not recently closed)
        to_add = mt5_tickets - tracked_tickets - set(self._recently_closed.keys())

        # Find positions to remove (tracked but not in MT5)
        to_remove = tracked_tickets - mt5_tickets

        # Add new positions
        for position in mt5_positions:
            if position.get("ticket") in to_add:
                self.add_position(position)

        # Remove closed positions — capture their P/L before removing
        # (Bug #36 fix: return closed positions so main loop can record them)
        closed_externally = []
        for ticket in to_remove:
            pos_data = self.positions.get(ticket, {})
            closed_externally.append({
                "ticket": ticket,
                "profit": pos_data.get("profit", 0),
                "direction": pos_data.get("type", ""),
                "entry_price": pos_data.get("open_price", 0),
                "volume": pos_data.get("volume", 0.01),
                "sl": pos_data.get("sl", 0),
                "tp": pos_data.get("tp", 0),
                "entry_session": pos_data.get("entry_session", ""),
                "entry_smc_signals": pos_data.get("entry_smc_signals", ""),
            })
            self.remove_position(ticket, {"reason": "closed_externally"})

        # Update existing positions
        for position in mt5_positions:
            ticket = position.get("ticket")
            if ticket in self.positions:
                self.update_position(ticket, {
                    "current_price": position.get("current_price"),
                    "profit": position.get("profit"),
                    "sl": position.get("sl"),
                    "tp": position.get("tp"),
                })

        report = {
            "added": len(to_add),
            "removed": len(to_remove),
            "updated": len(mt5_tickets & tracked_tickets),
            "total_tracked": len(self.positions),
            "closed_externally": closed_externally,
        }

        if to_add or to_remove:
            self.logger.info(
                f"Position sync: +{report['added']}, -{report['removed']}, "
                f"={report['updated']}"
            )

        return report
