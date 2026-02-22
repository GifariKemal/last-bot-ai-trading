"""
Exit Signal Generator
Generates exit signals for open positions based on multiple criteria.
"""

from typing import Dict, List
from datetime import datetime, timedelta, timezone
import polars as pl

from ..core.constants import SignalType, TrendDirection
from ..bot_logger import get_logger


class ExitSignalGenerator:
    """Generate exit signals for managing open positions."""

    def __init__(self, config: Dict):
        """
        Initialize exit signal generator.

        Args:
            config: Strategy configuration
        """
        self.logger = get_logger()
        self.config = config
        self.exit_config = config.get("strategy", {}).get("exit", {})

        # Recovery zone config
        self.recovery_zone_config = self.exit_config.get("recovery_zone", {})

        # Session close config (from session config)
        self.session_close_config = config.get("session", {}).get("ny_close_exit", {})

    def check_exit_conditions(
        self,
        position: Dict,
        current_price: float,
        smc_analysis: Dict,
        market_analysis: Dict,
        technical_indicators: Dict,
    ) -> Dict:
        """
        Check if position should be exited.

        Args:
            position: Open position data
            current_price: Current market price
            smc_analysis: SMC analysis
            market_analysis: Market condition analysis
            technical_indicators: Technical indicators

        Returns:
            Exit signal dictionary
        """
        try:
            position_type = position.get("type", "").upper()
            entry_price = position.get("open_price", 0)
            entry_time = position.get("open_time")
            current_profit = position.get("profit", 0)

            # Check various exit conditions
            exit_checks = []

            # Minimum hold period: skip structure/signal exits on freshly opened positions.
            # A bearish CHoCH that existed when the BUY was opened should not
            # immediately close it — wait at least 1 M15 bar (15 minutes).
            MIN_HOLD_MINUTES = 15
            if entry_time:
                entry_aware = entry_time if entry_time.tzinfo else entry_time.replace(tzinfo=timezone.utc)
                minutes_open = (datetime.now(timezone.utc) - entry_aware).total_seconds() / 60
                position_too_new = minutes_open < MIN_HOLD_MINUTES
            else:
                position_too_new = False

            # 1. Target hit (TP reached)
            if position.get("tp") and self._is_tp_hit(current_price, position):
                exit_checks.append({
                    "reason": "Take Profit hit",
                    "priority": 1,
                    "exit_now": True,
                })

            # 2. Stop loss hit
            if position.get("sl") and self._is_sl_hit(current_price, position):
                exit_checks.append({
                    "reason": "Stop Loss hit",
                    "priority": 1,
                    "exit_now": True,
                })

            # 3. Opposite structure break (skip for freshly opened positions)
            # Only exit if near breakeven or profit — let SL protect losing positions
            if self.exit_config.get("use_structure_exit", True) and not position_too_new:
                structure_exit = self._check_structure_exit(
                    position_type, smc_analysis
                )
                if structure_exit:
                    sl_distance = abs(entry_price - position.get("sl", entry_price)) or 1
                    if position_type == "BUY":
                        pnl_fraction = (current_price - entry_price) / sl_distance
                    else:
                        pnl_fraction = (entry_price - current_price) / sl_distance

                    if pnl_fraction >= -0.3:  # Within 30% of SL = near BE or better
                        exit_checks.append({
                            "reason": "Opposite structure break",
                            "priority": 2,
                            "exit_now": True,
                        })
                    else:
                        # Skip exit — position too deep in loss, let SL protect
                        self.logger.info(
                            f"Structure exit SKIPPED: position at {pnl_fraction:+.0%} of SL "
                            f"(need >= -30%). Letting SL protect."
                        )

            # 4. Opposite entry signal (skip for freshly opened positions)
            # Only exit if near breakeven or profit — let SL protect losing positions
            if self.exit_config.get("use_opposite_signal", True) and not position_too_new:
                opposite_signal = self._check_opposite_signal(
                    position_type, smc_analysis
                )
                if opposite_signal:
                    sl_distance = abs(entry_price - position.get("sl", entry_price)) or 1
                    if position_type == "BUY":
                        pnl_fraction = (current_price - entry_price) / sl_distance
                    else:
                        pnl_fraction = (entry_price - current_price) / sl_distance

                    if pnl_fraction >= -0.3:  # Within 30% of SL = near BE or better
                        exit_checks.append({
                            "reason": "Opposite entry signal detected",
                            "priority": 2,
                            "exit_now": True,
                        })
                    else:
                        # Skip exit — position too deep in loss, let SL protect
                        self.logger.info(
                            f"Opposite signal exit SKIPPED: position at {pnl_fraction:+.0%} of SL "
                            f"(need >= -30%). Letting SL protect."
                        )

            # 5. Time-based exit
            if self.exit_config.get("use_time_exit", True):
                time_exit = self._check_time_exit(entry_time)
                if time_exit:
                    exit_checks.append({
                        "reason": "Time limit exceeded (no progress)",
                        "priority": 3,
                        "exit_now": True,
                    })

            # 6. Session close exit (NY Close rule) — Features 1 & 3
            if self.exit_config.get("use_session_close_exit", False):
                session_exit = self._check_session_close_exit(position, current_price)
                if session_exit:
                    exit_checks.append(session_exit)

            # 7. Recovery Zone exit (damage control) — Feature 2
            if self.exit_config.get("use_recovery_zone_exit", False):
                recovery_exit = self._check_recovery_zone_exit(position, current_price)
                if recovery_exit:
                    exit_checks.append(recovery_exit)

            # 8. Profit protection (move to breakeven opportunity)
            if current_profit > 0:
                be_suggestion = self._check_breakeven_move(position, current_profit)
                if be_suggestion:
                    exit_checks.append({
                        "reason": "Move SL to breakeven",
                        "priority": 4,
                        "exit_now": False,
                        "action": "move_to_breakeven",
                    })

            # Determine if should exit
            should_exit = any(check.get("exit_now", False) for check in exit_checks)

            if should_exit:
                # Get highest priority reason
                exit_reason = sorted(exit_checks, key=lambda x: x.get("priority", 99))[0]

                return {
                    "type": (
                        SignalType.EXIT_LONG
                        if position_type == "BUY"
                        else SignalType.EXIT_SHORT
                    ),
                    "should_exit": True,
                    "reason": exit_reason["reason"],
                    "exit_type": exit_reason.get("exit_type"),
                    "price": current_price,
                    "timestamp": datetime.now(timezone.utc),
                    "all_reasons": [check["reason"] for check in exit_checks],
                }
            elif exit_checks:
                # Suggestions but no exit yet
                return {
                    "type": SignalType.NEUTRAL,
                    "should_exit": False,
                    "suggestions": exit_checks,
                    "price": current_price,
                    "timestamp": datetime.now(timezone.utc),
                }
            else:
                return self._no_exit()

        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return self._no_exit()

    def _is_tp_hit(self, current_price: float, position: Dict) -> bool:
        """Check if take profit is hit."""
        tp = position.get("tp", 0)
        position_type = position.get("type", "").upper()

        if not tp:
            return False

        if position_type == "BUY":
            return current_price >= tp
        else:  # SELL
            return current_price <= tp

    def _is_sl_hit(self, current_price: float, position: Dict) -> bool:
        """Check if stop loss is hit."""
        sl = position.get("sl", 0)
        position_type = position.get("type", "").upper()

        if not sl:
            return False

        if position_type == "BUY":
            return current_price <= sl
        else:  # SELL
            return current_price >= sl

    def _check_structure_exit(
        self, position_type: str, smc_analysis: Dict
    ) -> bool:
        """Check if opposite structure break suggests exit."""

        if position_type == "BUY":
            # Exit long if bearish CHoCH
            bearish_signals = smc_analysis.get("bearish", {})
            return bearish_signals.get("structure", {}).get("choch", False)
        else:  # SELL
            # Exit short if bullish CHoCH
            bullish_signals = smc_analysis.get("bullish", {})
            return bullish_signals.get("structure", {}).get("choch", False)

    def _check_opposite_signal(
        self, position_type: str, smc_analysis: Dict
    ) -> bool:
        """Check if opposite entry signal is present."""

        if position_type == "BUY":
            # Exit long if strong bearish signal
            bearish = smc_analysis.get("bearish", {})
            bearish_score = bearish.get("confluence_score", 0)
            return bearish_score >= 0.70  # High threshold for exit
        else:  # SELL
            # Exit short if strong bullish signal
            bullish = smc_analysis.get("bullish", {})
            bullish_score = bullish.get("confluence_score", 0)
            return bullish_score >= 0.70

    def _check_time_exit(self, entry_time: datetime) -> bool:
        """Check if position has been open too long without progress."""
        if not entry_time:
            return False

        max_hours = self.exit_config.get("time_exit_hours", 24)
        time_limit = timedelta(hours=max_hours)

        now = datetime.now(timezone.utc)
        # Handle both naive and aware datetimes
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        time_open = now - entry_time

        return time_open > time_limit

    def _check_breakeven_move(self, position: Dict, current_profit: float) -> bool:
        """Check if SL should be moved to breakeven."""
        # Check if profit is sufficient to move to BE
        entry_price = position.get("open_price", 0)
        sl = position.get("sl", 0)

        if not sl or not entry_price:
            return False

        # Calculate distance to entry
        position_type = position.get("type", "").upper()
        current_price = position.get("current_price", entry_price)

        if position_type == "BUY":
            distance_pips = current_price - entry_price
        else:
            distance_pips = entry_price - current_price

        # Suggest BE if price moved 50% toward target
        return distance_pips > abs(entry_price - sl) * 0.5

    def _check_recovery_zone_exit(self, position: Dict, current_price: float) -> Dict:
        """
        Check if position recovered from deep loss and should exit (damage control).

        Positions that dip >60% toward SL and recover have a high chance of
        dipping again. Taking a small loss/BE exit preserves capital.
        """
        cfg = self.recovery_zone_config
        if not cfg.get("enabled", False):
            return None

        entry_price = position.get("open_price", 0)
        sl = position.get("sl", 0)
        position_type = position.get("type", "").upper()
        entry_time = position.get("open_time")

        if not entry_price or not sl:
            return None

        sl_distance = abs(entry_price - sl)
        if sl_distance <= 0:
            return None

        # Calculate current adverse excursion as fraction of SL distance
        if position_type == "BUY":
            current_profit_pips = current_price - entry_price
        else:
            current_profit_pips = entry_price - current_price

        # Get max_drawdown from position tracker data (peak_profit - current lowest)
        max_drawdown = position.get("max_drawdown", 0)

        # max_drawdown is measured as peak - trough in pips
        # For a losing position that never went positive, max_drawdown tracks
        # the deepest adverse move from entry (since peak defaults to 0)
        # We need to calculate max adverse excursion differently:
        # Use the deepest the position went against us
        # peak_profit can be 0 if never profitable, and max_drawdown = peak - current_low
        peak_profit = position.get("peak_profit", 0)

        # Max adverse excursion = how far below entry the position went
        # If peak_profit=0 and max_drawdown=X, the deepest point was -X pips from entry
        # If peak_profit=P and max_drawdown=D, the deepest point was P-D pips from entry
        deepest_point_pips = peak_profit - max_drawdown  # This is the worst P&L in pips

        # Thresholds
        deep_loss_threshold = cfg.get("deep_loss_threshold", 0.60)
        recovery_threshold = cfg.get("recovery_threshold", 0.10)
        min_hold_minutes = cfg.get("min_hold_minutes", 30)

        # Was position in deep loss? (>60% of SL distance)
        if deepest_point_pips >= 0:
            return None  # Never went negative, no recovery needed

        max_adverse_fraction = abs(deepest_point_pips) / sl_distance
        if max_adverse_fraction < deep_loss_threshold:
            return None  # Never deep enough to qualify

        # Has it recovered to near breakeven? (within 10% of SL distance)
        current_fraction = current_profit_pips / sl_distance  # Positive = profitable
        if current_fraction < -recovery_threshold:
            return None  # Still too deep in loss

        # Check minimum hold time (avoid whipsaws)
        if entry_time:
            entry_aware = entry_time if entry_time.tzinfo else entry_time.replace(tzinfo=timezone.utc)
            minutes_open = (datetime.now(timezone.utc) - entry_aware).total_seconds() / 60
            if minutes_open < min_hold_minutes:
                return None

        self.logger.info(
            f"Recovery Zone: Position recovered from {max_adverse_fraction:.0%} drawdown "
            f"to {current_fraction:+.0%} of SL distance — exiting to avoid re-loss"
        )

        return {
            "reason": f"Recovery Zone (recovered from {max_adverse_fraction:.0%} drawdown)",
            "priority": 2,
            "exit_now": True,
            "exit_type": "RECOVERY_EXIT",
        }

    def _check_session_close_exit(self, position: Dict, current_price: float) -> Dict:
        """
        Check if position should be exited near NY Close (21:30 UTC).

        Logic:
        - If profit > 0: EARLY EXIT (take profit before session end)
        - If loss < $1: EARLY EXIT (near breakeven, save spread)
        - If profitable > 30% of TP and open > 2 hours: EARLY EXIT
        - If profitable 4+ hours but profit decaying from peak: EXIT (profit decay)
        - If loss > $1: HOLD (let SL protect, Asian may reverse)
        """
        cfg = self.session_close_config
        if not cfg.get("enabled", False):
            return None

        now = datetime.now(timezone.utc)
        buffer_minutes = cfg.get("buffer_minutes", 30)

        # Parse NY close time
        ny_close_str = cfg.get("ny_close_utc", "22:00")
        ny_close_parts = ny_close_str.split(":")
        ny_close_hour = int(ny_close_parts[0])
        ny_close_minute = int(ny_close_parts[1])

        # Calculate minutes until NY close
        current_minutes = now.hour * 60 + now.minute
        close_minutes = ny_close_hour * 60 + ny_close_minute
        minutes_until_close = close_minutes - current_minutes
        if minutes_until_close < 0:
            minutes_until_close += 24 * 60

        # Check if within buffer window (e.g., 30 min before close)
        in_close_window = 0 <= minutes_until_close <= buffer_minutes

        # --- Feature 3: Profit Decay Exit (independent of session window) ---
        entry_time = position.get("open_time")
        profit = position.get("profit", 0)
        peak_profit_val = position.get("peak_profit", 0)

        if entry_time and profit > 0 and peak_profit_val > 0:
            entry_aware = entry_time if entry_time.tzinfo else entry_time.replace(tzinfo=timezone.utc)
            hours_open = (now - entry_aware).total_seconds() / 3600
            decay_hours = cfg.get("profit_decay_hours", 4)
            decay_threshold = cfg.get("profit_decay_threshold", 0.30)

            entry_price = position.get("open_price", 0)
            position_type = position.get("type", "").upper()
            if position_type == "BUY":
                current_profit_dist = current_price - entry_price
            else:
                current_profit_dist = entry_price - current_price

            # If profitable for 4+ hours and profit decaying > 30% from peak
            if hours_open >= decay_hours and peak_profit_val > 0:
                decay_pct = (peak_profit_val - current_profit_dist) / peak_profit_val
                if decay_pct >= decay_threshold and current_profit_dist > 0:
                    self.logger.info(
                        f"Profit Decay: Profitable {hours_open:.1f}h but profit dropped "
                        f"{decay_pct:.0%} from peak ({peak_profit_val:.2f} → {current_profit_dist:.2f})"
                    )
                    return {
                        "reason": f"Profit Decay ({decay_pct:.0%} from peak after {hours_open:.1f}h)",
                        "priority": 3,
                        "exit_now": True,
                        "exit_type": "EARLY_PROFIT",
                    }

        # --- Feature 1: Session close window checks ---
        if not in_close_window:
            return None

        entry_price = position.get("open_price", 0)
        position_type = position.get("type", "").upper()
        sl = position.get("sl", 0)
        tp = position.get("tp", 0)

        near_be_loss = cfg.get("near_breakeven_loss", 1.0)
        min_hold_hours = cfg.get("min_hold_hours_for_profit_take", 2)

        # Calculate hours open for hold check
        hours_open = 0
        if entry_time:
            entry_aware = entry_time if entry_time.tzinfo else entry_time.replace(tzinfo=timezone.utc)
            hours_open = (now - entry_aware).total_seconds() / 3600

        # Position in profit → take it before session end
        if profit > 0:
            # Check if profit > 30% of TP distance AND held long enough
            if tp and entry_price and hours_open >= min_hold_hours:
                tp_distance = abs(tp - entry_price)
                if position_type == "BUY":
                    profit_distance = current_price - entry_price
                else:
                    profit_distance = entry_price - current_price

                profit_pct_of_tp = profit_distance / tp_distance if tp_distance > 0 else 0
                min_profit_for_early = cfg.get("min_profit_for_early_exit", 0.30)

                if profit_pct_of_tp >= min_profit_for_early:
                    self.logger.info(
                        f"Session Close: Taking profit ({profit_pct_of_tp:.0%} of TP) "
                        f"{minutes_until_close}min before NY Close"
                    )
                    return {
                        "reason": f"Session End Profit ({profit_pct_of_tp:.0%} of TP, {minutes_until_close}min to close)",
                        "priority": 2,
                        "exit_now": True,
                        "exit_type": "SESSION_EXIT",
                    }

            # Any profit in close window → exit
            self.logger.info(
                f"Session Close: Taking profit (${profit:.2f}) "
                f"{minutes_until_close}min before NY Close"
            )
            return {
                "reason": f"Session End Profit Take (${profit:.2f}, {minutes_until_close}min to close)",
                "priority": 2,
                "exit_now": True,
                "exit_type": "SESSION_EXIT",
            }

        # Near breakeven loss → exit (save the spread)
        if abs(profit) <= near_be_loss:
            self.logger.info(
                f"Session Close: Near breakeven (${profit:.2f}), exiting "
                f"{minutes_until_close}min before NY Close"
            )
            return {
                "reason": f"Session End Near-BE (${profit:.2f}, {minutes_until_close}min to close)",
                "priority": 2,
                "exit_now": True,
                "exit_type": "SESSION_EXIT",
            }

        # Loss > $1 → HOLD (let SL protect, Asian session may reverse)
        self.logger.debug(
            f"Session Close: Holding losing position (${profit:.2f}), "
            f"SL protecting — Asian may reverse"
        )
        return None

    def _no_exit(self) -> Dict:
        """Return no exit structure."""
        return {
            "type": SignalType.NEUTRAL,
            "should_exit": False,
            "reason": None,
            "price": None,
            "timestamp": datetime.now(timezone.utc),
        }

    def get_exit_summary(self, exit_signal: Dict) -> str:
        """
        Get human-readable exit signal summary.

        Args:
            exit_signal: Exit signal data

        Returns:
            Summary string
        """
        if not exit_signal.get("should_exit", False):
            suggestions = exit_signal.get("suggestions", [])
            if suggestions:
                return f"HOLD (Suggestions: {len(suggestions)})"
            return "HOLD (No exit conditions)"

        reason = exit_signal.get("reason", "Unknown")
        price = exit_signal.get("price", 0)

        return f"EXIT @ {price:.2f} - {reason}"
