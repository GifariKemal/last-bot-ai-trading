"""
Drawdown Monitor
Monitors account drawdown and enforces protection rules.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from ..bot_logger import get_logger


class DrawdownMonitor:
    """Monitor account drawdown and enforce protection rules."""

    def __init__(self, config: Dict):
        """
        Initialize drawdown monitor.

        Args:
            config: Risk configuration
        """
        self.logger = get_logger()
        self.config = config

        self.protection_config = config.get("account_protection", {})

        # Limits
        self.max_daily_loss_percent = self.protection_config.get(
            "max_daily_loss_percent", 5.0
        )
        self.max_weekly_loss_percent = self.protection_config.get(
            "max_weekly_loss_percent", 10.0
        )
        self.max_monthly_loss_percent = self.protection_config.get(
            "max_monthly_loss_percent", 20.0
        )
        self.max_drawdown_percent = self.protection_config.get(
            "max_drawdown_percent", 15.0
        )
        self.max_consecutive_losses = self.protection_config.get(
            "max_consecutive_losses", 3
        )

        # Pause settings
        self.pause_on_limit = self.protection_config.get("pause_on_limit", True)
        self.pause_duration_minutes = self.protection_config.get(
            "pause_duration_minutes", 60
        )

        # Daily profit target (optional)
        self.daily_profit_target = self.protection_config.get("daily_profit_target", None)

        # Tracking
        self.daily_start_balance = None
        self.weekly_start_balance = None
        self.monthly_start_balance = None
        self.peak_balance = None
        self.consecutive_losses = 0
        self.last_reset_date = None
        self.pause_until = None
        self.trade_history = []  # Recent trades for tracking
        self._loss_limit_logged = False  # Bug #52: prevent repeated log spam for percent-based limits

    def initialize(self, account_info: Dict) -> None:
        """
        Initialize tracking with current account state.

        Args:
            account_info: Current account information
        """
        balance = account_info.get("balance", 0)
        equity = account_info.get("equity", 0)

        self.daily_start_balance = balance
        self.weekly_start_balance = balance
        self.monthly_start_balance = balance
        self.peak_balance = max(balance, equity)
        self.last_reset_date = datetime.now().date()

        self.logger.info(f"Drawdown monitor initialized with balance: ${balance:.2f}")

    def check_trading_allowed(self, account_info: Dict) -> Dict:
        """
        Check if trading is allowed based on protection rules.

        Args:
            account_info: Current account information

        Returns:
            Dictionary with allowed status and reasons
        """
        if not self.daily_start_balance:
            self.initialize(account_info)

        # Reset daily/weekly/monthly if needed
        balance = account_info.get("balance", 0)
        self._reset_periods_if_needed(balance)

        # Check if in pause period
        if self.pause_until and datetime.now() < self.pause_until:
            time_remaining = (self.pause_until - datetime.now()).total_seconds() / 60
            return {
                "allowed": False,
                "reason": f"Trading paused. {time_remaining:.0f} minutes remaining",
                "pause_until": self.pause_until,
            }

        # Bug #37 fix: pause window just expired — clear consecutive losses so trading
        # resumes. Without this, consecutive_losses stays >= limit and immediately
        # triggers another 60-minute pause, making the cooldown infinite.
        pause_just_expired = False
        if self.pause_until and datetime.now() >= self.pause_until:
            self.logger.info(
                f"Pause window expired. Resetting consecutive losses from {self.consecutive_losses} to 0."
            )
            self.consecutive_losses = 0
            self.pause_until = None
            pause_just_expired = True

        violations = []

        # Get current balance and equity (balance already extracted above for period reset)
        equity = account_info.get("equity", 0)

        # Update peak balance
        if equity > self.peak_balance:
            self.peak_balance = equity

        # Check daily loss
        daily_loss_percent = self._calculate_loss_percent(
            self.daily_start_balance, balance
        )
        if daily_loss_percent >= self.max_daily_loss_percent:
            violations.append({
                "type": "daily_loss",
                "current": daily_loss_percent,
                "limit": self.max_daily_loss_percent,
                "message": f"Daily loss limit exceeded: {daily_loss_percent:.2f}%",
            })

        # Check weekly loss
        weekly_loss_percent = self._calculate_loss_percent(
            self.weekly_start_balance, balance
        )
        if weekly_loss_percent >= self.max_weekly_loss_percent:
            violations.append({
                "type": "weekly_loss",
                "current": weekly_loss_percent,
                "limit": self.max_weekly_loss_percent,
                "message": f"Weekly loss limit exceeded: {weekly_loss_percent:.2f}%",
            })

        # Check monthly loss
        monthly_loss_percent = self._calculate_loss_percent(
            self.monthly_start_balance, balance
        )
        if monthly_loss_percent >= self.max_monthly_loss_percent:
            violations.append({
                "type": "monthly_loss",
                "current": monthly_loss_percent,
                "limit": self.max_monthly_loss_percent,
                "message": f"Monthly loss limit exceeded: {monthly_loss_percent:.2f}%",
            })

        # Check maximum drawdown
        drawdown_percent = self._calculate_drawdown(self.peak_balance, equity)
        if drawdown_percent >= self.max_drawdown_percent:
            violations.append({
                "type": "max_drawdown",
                "current": drawdown_percent,
                "limit": self.max_drawdown_percent,
                "message": f"Maximum drawdown exceeded: {drawdown_percent:.2f}%",
            })

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            violations.append({
                "type": "consecutive_losses",
                "current": self.consecutive_losses,
                "limit": self.max_consecutive_losses,
                "message": f"Max consecutive losses reached: {self.consecutive_losses}",
            })

        # Check daily profit target (if set)
        if self.daily_profit_target:
            daily_profit_percent = ((balance - self.daily_start_balance) / self.daily_start_balance) * 100
            if daily_profit_percent >= self.daily_profit_target:
                return {
                    "allowed": False,
                    "reason": f"Daily profit target reached: {daily_profit_percent:.2f}%",
                    "target_reached": True,
                }

        # If violations exist
        if violations:
            # Bug #52 fix: percent-based violations (daily/weekly/monthly loss, drawdown)
            # are permanent until the period resets — they can't recover without trading.
            # Only set a timed pause for consecutive_losses (which was just cleared).
            # For percent-based violations, block permanently (no new timed pause).
            has_percent_violation = any(
                v["type"] in ("daily_loss", "weekly_loss", "monthly_loss", "max_drawdown")
                for v in violations
            )
            only_consecutive = all(v["type"] == "consecutive_losses" for v in violations)

            if self.pause_on_limit and only_consecutive:
                # Consecutive losses: set timed pause (resumes after cooldown)
                self.pause_until = datetime.now() + timedelta(
                    minutes=self.pause_duration_minutes
                )
                self.logger.warning(
                    f"Trading paused until {self.pause_until} due to: "
                    f"{', '.join([v['message'] for v in violations])}"
                )
            elif has_percent_violation and not self._loss_limit_logged:
                # Percent-based: block permanently until period reset (no timed pause loop)
                self._loss_limit_logged = True
                self.logger.warning(
                    f"Trading BLOCKED until period reset: "
                    f"{', '.join([v['message'] for v in violations])}"
                )

            return {
                "allowed": False,
                "reason": "Protection rules triggered" if not has_percent_violation
                    else f"Loss limit exceeded — blocked until reset ({', '.join(v['type'] for v in violations)})",
                "violations": violations,
                "pause_until": self.pause_until if only_consecutive else None,
            }

        # All checks passed
        return {
            "allowed": True,
            "reason": "All protection checks passed",
            "stats": {
                "daily_loss_percent": daily_loss_percent,
                "weekly_loss_percent": weekly_loss_percent,
                "monthly_loss_percent": monthly_loss_percent,
                "drawdown_percent": drawdown_percent,
                "consecutive_losses": self.consecutive_losses,
            },
        }

    def record_trade_result(self, trade: Dict) -> None:
        """
        Record trade result for tracking.

        Args:
            trade: Trade data with profit/loss
        """
        profit = trade.get("profit", 0)
        is_win = profit > 0

        # Update consecutive losses
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        # Add to history
        self.trade_history.append({
            "timestamp": datetime.now(),
            "profit": profit,
            "is_win": is_win,
            "ticket": trade.get("ticket"),
        })

        # Keep only recent history (last 100 trades)
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]

        self.logger.info(
            f"Trade recorded: {'WIN' if is_win else 'LOSS'} ${profit:.2f}, "
            f"Consecutive losses: {self.consecutive_losses}"
        )

    def _calculate_loss_percent(self, start_balance: float, current_balance: float) -> float:
        """Calculate loss percentage."""
        if start_balance <= 0:
            return 0.0

        loss = start_balance - current_balance
        if loss <= 0:
            return 0.0

        return (loss / start_balance) * 100

    def _calculate_drawdown(self, peak_balance: float, current_equity: float) -> float:
        """Calculate drawdown from peak."""
        if peak_balance <= 0:
            return 0.0

        drawdown = peak_balance - current_equity
        if drawdown <= 0:
            return 0.0

        return (drawdown / peak_balance) * 100

    def _reset_periods_if_needed(self, balance: float) -> None:
        """Reset daily/weekly/monthly tracking if period changed."""
        now = datetime.now()
        current_date = now.date()

        if self.last_reset_date is None:
            self.last_reset_date = current_date
            return

        # Reset daily
        if current_date > self.last_reset_date:
            self.logger.info(f"Resetting daily tracking — new start balance: ${balance:.2f}")
            self.daily_start_balance = balance
            self.last_reset_date = current_date
            self.consecutive_losses = 0  # Reset on new day
            self._loss_limit_logged = False  # Bug #52: allow logging for new day's violations

        # Reset weekly (on Monday)
        if now.weekday() == 0 and (now - datetime.combine(self.last_reset_date, datetime.min.time())).days >= 7:
            self.logger.info(f"Resetting weekly tracking — new start balance: ${balance:.2f}")
            self.weekly_start_balance = balance

        # Reset monthly (on 1st of month)
        if now.day == 1 and current_date.month != self.last_reset_date.month:
            self.logger.info(f"Resetting monthly tracking — new start balance: ${balance:.2f}")
            self.monthly_start_balance = balance

    def get_account_status(self, account_info: Dict) -> Dict:
        """
        Get current account status and risk metrics.

        Args:
            account_info: Account information

        Returns:
            Status dictionary
        """
        if not self.daily_start_balance:
            self.initialize(account_info)

        balance = account_info.get("balance", 0)
        equity = account_info.get("equity", 0)

        return {
            "balance": balance,
            "equity": equity,
            "peak_balance": self.peak_balance,
            "daily_start": self.daily_start_balance,
            "weekly_start": self.weekly_start_balance,
            "monthly_start": self.monthly_start_balance,
            "daily_pnl": balance - self.daily_start_balance,
            "daily_pnl_percent": ((balance - self.daily_start_balance) / self.daily_start_balance) * 100 if self.daily_start_balance else 0,
            "drawdown_percent": self._calculate_drawdown(self.peak_balance, equity),
            "consecutive_losses": self.consecutive_losses,
            "paused": self.pause_until and datetime.now() < self.pause_until,
            "pause_until": self.pause_until,
        }

    def manual_pause(self, duration_minutes: int, reason: str = "Manual pause") -> None:
        """
        Manually pause trading.

        Args:
            duration_minutes: Pause duration in minutes
            reason: Reason for pause
        """
        self.pause_until = datetime.now() + timedelta(minutes=duration_minutes)
        self.logger.warning(f"Trading manually paused until {self.pause_until}: {reason}")

    def resume_trading(self) -> None:
        """Resume trading (clear pause)."""
        if self.pause_until:
            self.logger.info("Trading resumed manually")
            self.pause_until = None

    def reset_consecutive_losses(self) -> None:
        """Reset consecutive loss counter."""
        old_count = self.consecutive_losses
        self.consecutive_losses = 0
        self.logger.info(f"Consecutive losses reset from {old_count} to 0")
