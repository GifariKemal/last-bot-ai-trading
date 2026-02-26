"""
Order Executor
Executes orders on MT5 with comprehensive error handling and validation.
"""

from typing import Dict, Optional, List
from datetime import datetime
import time
from ..bot_logger import get_logger


class OrderExecutor:
    """Execute orders on MT5."""

    def __init__(self, mt5_connector, config: Dict):
        """
        Initialize order executor.

        Args:
            mt5_connector: MT5Connector instance
            config: Configuration dictionary
        """
        self.logger = get_logger()
        self.mt5 = mt5_connector
        self.config = config

        # Execution settings
        self.max_retries = 3
        self.retry_delay_seconds = 2
        self.max_slippage = 10  # Maximum slippage in pips

    def execute_entry(
        self,
        signal: Dict,
        lot_size: float,
        sl_price: float,
        tp_price: float,
        comment: str = "SMC Strategy"
    ) -> Dict:
        """
        Execute entry order (open new position).

        Args:
            signal: Trading signal
            lot_size: Position size in lots
            sl_price: Stop loss price
            tp_price: Take profit price
            comment: Order comment

        Returns:
            Execution result dictionary
        """
        try:
            direction = signal.get("direction", "").upper()
            entry_price = signal.get("price", 0)

            if direction not in ["BUY", "SELL"]:
                return {
                    "success": False,
                    "error": "Invalid direction",
                }

            # Pre-execution checks
            pre_check = self._pre_execution_checks(lot_size)
            if not pre_check["passed"]:
                return {
                    "success": False,
                    "error": pre_check["reason"],
                    "checks": pre_check,
                }

            # Execute order with retry logic
            for attempt in range(1, self.max_retries + 1):
                self.logger.info(
                    f"Executing {direction} order: {lot_size} lots @ {entry_price:.2f} "
                    f"(SL: {sl_price:.2f}, TP: {tp_price:.2f}) - Attempt {attempt}"
                )

                result = self.mt5.send_order(
                    symbol=self.mt5.symbol,   # dynamic — respects config/mt5.symbol
                    order_type=direction,
                    volume=lot_size,
                    sl=sl_price,
                    tp=tp_price,
                    comment=comment,
                    magic=self.config.get("magic_number", 123456)
                )

                if result is not None:
                    self.logger.info(
                        f"Order executed successfully: Ticket #{result.get('ticket')}"
                    )
                    return {
                        "success": True,
                        "ticket": result.get("ticket"),
                        "entry_price": result.get("price", entry_price),
                        "sl": sl_price,
                        "tp": tp_price,
                        "lot_size": lot_size,
                        "direction": direction,
                        "timestamp": datetime.now(),
                        "comment": comment,
                    }

                # Log failure
                error_msg = "Order returned None"
                self.logger.warning(
                    f"Order execution failed (attempt {attempt}): {error_msg}"
                )

                # Check if should retry
                if attempt < self.max_retries:
                    if self._is_retryable_error(error_msg):
                        self.logger.info(f"Retrying in {self.retry_delay_seconds}s...")
                        time.sleep(self.retry_delay_seconds)
                    else:
                        # Non-retryable error
                        break

            # All attempts failed
            return {
                "success": False,
                "error": "Max retries exceeded",
                "attempts": self.max_retries,
            }

        except Exception as e:
            self.logger.error(f"Error executing entry order: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def execute_exit(
        self,
        ticket: int,
        reason: str = "Exit signal"
    ) -> Dict:
        """
        Execute exit order (close position).

        Args:
            ticket: Position ticket ID
            reason: Reason for exit

        Returns:
            Execution result
        """
        try:
            self.logger.info(f"Closing position #{ticket}: {reason}")

            # Execute close with retry logic
            for attempt in range(1, self.max_retries + 1):
                closed = self.mt5.close_position(ticket)

                if closed:
                    self.logger.info(
                        f"Position closed successfully: #{ticket} - {reason}"
                    )
                    return {
                        "success": True,
                        "ticket": ticket,
                        "reason": reason,
                        "timestamp": datetime.now(),
                    }

                # Log failure
                self.logger.warning(
                    f"Position close failed (attempt {attempt})"
                )

                # Retry
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds)

            # All attempts failed
            return {
                "success": False,
                "error": "Max retries exceeded",
                "ticket": ticket,
            }

        except Exception as e:
            self.logger.error(f"Error closing position #{ticket}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticket": ticket,
            }

    def modify_position(
        self,
        ticket: int,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None,
        reason: str = "Position modification"
    ) -> Dict:
        """
        Modify position (update SL/TP).

        Args:
            ticket: Position ticket ID
            new_sl: New stop loss price (None = no change)
            new_tp: New take profit price (None = no change)
            reason: Reason for modification

        Returns:
            Modification result
        """
        try:
            if new_sl is None and new_tp is None:
                return {
                    "success": False,
                    "error": "No modifications specified",
                }

            self.logger.info(
                f"Modifying position #{ticket}: {reason} "
                f"(SL: {new_sl}, TP: {new_tp})"
            )

            # Execute modification with retry logic
            for attempt in range(1, self.max_retries + 1):
                modified = self.mt5.modify_position(
                    ticket=ticket,
                    sl=new_sl,
                    tp=new_tp
                )

                if modified:
                    self.logger.info(
                        f"Position modified successfully: #{ticket} - {reason}"
                    )
                    return {
                        "success": True,
                        "ticket": ticket,
                        "new_sl": new_sl,
                        "new_tp": new_tp,
                        "reason": reason,
                        "timestamp": datetime.now(),
                    }

                # Log failure
                self.logger.warning(
                    f"Position modification failed (attempt {attempt})"
                )

                # Retry
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds)

            # All attempts failed
            return {
                "success": False,
                "error": "Max retries exceeded",
                "ticket": ticket,
            }

        except Exception as e:
            self.logger.error(f"Error modifying position #{ticket}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticket": ticket,
            }

    def _pre_execution_checks(self, lot_size: float) -> Dict:
        """
        Pre-execution validation checks.

        Args:
            lot_size: Proposed lot size

        Returns:
            Validation result
        """
        checks = []

        # 1. Check MT5 connection
        if not self.mt5.connected:
            checks.append({
                "check": "connection",
                "passed": False,
                "reason": "MT5 not connected",
            })
            return {"passed": False, "reason": "MT5 not connected", "checks": checks}

        # 2. Get account info
        account_info = self.mt5.get_account_info()
        if not account_info:
            checks.append({
                "check": "account_info",
                "passed": False,
                "reason": "Cannot retrieve account info",
            })
            return {"passed": False, "reason": "Cannot get account info", "checks": checks}

        # 3. Check account balance
        balance = account_info.get("balance", 0)
        if balance <= 0:
            checks.append({
                "check": "balance",
                "passed": False,
                "reason": "Insufficient balance",
            })
            return {"passed": False, "reason": "Insufficient balance", "checks": checks}

        checks.append({
            "check": "balance",
            "passed": True,
            "balance": balance,
        })

        # 4. Check free margin
        # Bug #56: was < 100 — hardcoded $100 floor blocked all trades on
        # sub-$100 account ($94.31 balance). Actual margin for 0.01 lot
        # XAUUSDm at 1:100 leverage is ~$50-52, so the real floor is $50.
        margin_free = account_info.get("margin_free", 0)
        if margin_free < 50:  # Min margin for 0.01 lot gold at 1:100 leverage
            checks.append({
                "check": "margin",
                "passed": False,
                "reason": f"Insufficient margin (${margin_free:.2f})",
            })
            return {"passed": False, "reason": "Insufficient margin", "checks": checks}

        checks.append({
            "check": "margin",
            "passed": True,
            "margin_free": margin_free,
        })

        # 5. Check spread
        tick = self.mt5.get_tick(self.mt5.symbol)
        if tick:
            spread = tick.get("spread", 0)
            if spread > 50:  # Max 5.0 pips
                checks.append({
                    "check": "spread",
                    "passed": False,
                    "reason": f"Spread too high ({spread} points)",
                })
                return {"passed": False, "reason": "Spread too high", "checks": checks}

            checks.append({
                "check": "spread",
                "passed": True,
                "spread": spread,
            })

        # 6. Check trading allowed
        # (MT5 checks this internally, but we can add session checks here)

        # All checks passed
        return {
            "passed": True,
            "reason": "All pre-execution checks passed",
            "checks": checks,
        }

    def _is_retryable_error(self, error_msg: str) -> bool:
        """
        Check if error is retryable.

        Args:
            error_msg: Error message

        Returns:
            True if error is retryable
        """
        # Retryable errors
        retryable_keywords = [
            "timeout",
            "connection",
            "network",
            "requote",
            "price changed",
            "off quotes",
            "busy",
        ]

        error_lower = str(error_msg).lower()
        return any(keyword in error_lower for keyword in retryable_keywords)

    def close_all_positions(self, reason: str = "Close all") -> Dict:
        """
        Close all open positions.

        Args:
            reason: Reason for closing all

        Returns:
            Results dictionary
        """
        try:
            self.logger.warning(f"Closing all positions: {reason}")

            positions = self.mt5.get_positions()
            if not positions:
                return {
                    "success": True,
                    "message": "No positions to close",
                    "closed_count": 0,
                }

            results = []
            for position in positions:
                ticket = position.get("ticket")
                result = self.execute_exit(ticket, reason)
                results.append(result)

                # Small delay between closes
                time.sleep(0.5)

            # Count successes
            successful = [r for r in results if r.get("success")]
            failed = [r for r in results if not r.get("success")]

            self.logger.info(
                f"Closed {len(successful)}/{len(positions)} positions"
            )

            return {
                "success": len(failed) == 0,
                "total_positions": len(positions),
                "closed_count": len(successful),
                "failed_count": len(failed),
                "results": results,
            }

        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def get_execution_statistics(self) -> Dict:
        """
        Get execution statistics.

        Returns:
            Statistics dictionary
        """
        # This would track execution metrics over time
        # For now, return basic info
        return {
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay_seconds,
            "max_slippage": self.max_slippage,
        }

    def validate_order_parameters(
        self,
        direction: str,
        lot_size: float,
        sl_price: float,
        tp_price: float,
        entry_price: float
    ) -> Dict:
        """
        Validate order parameters before execution.

        Args:
            direction: BUY or SELL
            lot_size: Lot size
            sl_price: Stop loss price
            tp_price: Take profit price
            entry_price: Entry price

        Returns:
            Validation result
        """
        errors = []

        # Validate direction
        if direction not in ["BUY", "SELL"]:
            errors.append("Invalid direction")

        # Validate lot size
        if lot_size <= 0 or lot_size > 1.0:
            errors.append(f"Invalid lot size: {lot_size}")

        # Validate SL
        if direction == "BUY":
            if sl_price >= entry_price:
                errors.append("SL must be below entry for BUY")
        else:  # SELL
            if sl_price <= entry_price:
                errors.append("SL must be above entry for SELL")

        # Validate TP
        if direction == "BUY":
            if tp_price <= entry_price:
                errors.append("TP must be above entry for BUY")
        else:  # SELL
            if tp_price >= entry_price:
                errors.append("TP must be below entry for SELL")

        # Validate minimum distance in price units (dollars for XAUUSD, NOT pips).
        # XAUUSD: 1 pip = $0.10, so $1.0 min = 10 pips — a safe floor that our
        # typical 3×ATR SL ($30-60) always exceeds. Using raw price diff avoids the
        # "price-as-pips" unit confusion that would wrongly reject valid XAUUSD orders.
        sl_distance = abs(entry_price - sl_price)
        tp_distance = abs(tp_price - entry_price)
        MIN_DIST = 1.0  # $1.0 minimum (= 10 standard pips for XAUUSD)

        if sl_distance < MIN_DIST:
            errors.append(f"SL too close to entry (${sl_distance:.2f} < ${MIN_DIST:.2f})")

        if tp_distance < MIN_DIST:
            errors.append(f"TP too close to entry (${tp_distance:.2f} < ${MIN_DIST:.2f})")

        # Validate RR ratio
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
        if rr_ratio < 1.5:
            errors.append(f"RR ratio too low ({rr_ratio:.2f})")

        if errors:
            return {
                "valid": False,
                "errors": errors,
            }

        return {
            "valid": True,
            "rr_ratio": rr_ratio,
            "sl_distance": sl_distance,
            "tp_distance": tp_distance,
        }
