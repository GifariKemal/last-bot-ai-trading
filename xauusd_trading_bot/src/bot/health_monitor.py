"""
Health Monitor
Monitors bot health and system status.
"""

from typing import Dict
from datetime import datetime, timedelta
from ..bot_logger import get_logger


class HealthMonitor:
    """Monitor bot health and performance."""

    def __init__(self):
        """Initialize health monitor."""
        self.logger = get_logger()

        # Health tracking
        self.start_time = datetime.now()
        self.last_loop_time = None
        self.loop_count = 0
        self.error_count = 0
        self.last_errors = []  # Track recent errors

        # Performance metrics
        self.total_signals_generated = 0
        self.total_orders_executed = 0
        self.total_orders_failed = 0

        # System health
        self.mt5_connected = False
        self.last_mt5_check = None

    def record_loop_iteration(self) -> None:
        """Record a successful loop iteration."""
        self.last_loop_time = datetime.now()
        self.loop_count += 1

    def record_error(self, error: Exception, context: str = "") -> None:
        """
        Record an error.

        Args:
            error: Exception that occurred
            context: Context where error occurred
        """
        self.error_count += 1

        error_entry = {
            "timestamp": datetime.now(),
            "error": str(error),
            "context": context,
        }

        self.last_errors.append(error_entry)

        # Keep only last 10 errors
        if len(self.last_errors) > 10:
            self.last_errors = self.last_errors[-10:]

        self.logger.error(f"Error in {context}: {error}")

    def record_signal(self, signal_valid: bool) -> None:
        """Record signal generation."""
        self.total_signals_generated += 1

    def record_order(self, success: bool) -> None:
        """Record order execution."""
        if success:
            self.total_orders_executed += 1
        else:
            self.total_orders_failed += 1

    def update_mt5_status(self, connected: bool) -> None:
        """Update MT5 connection status."""
        self.mt5_connected = connected
        self.last_mt5_check = datetime.now()

    def check_health(self) -> Dict:
        """
        Check overall bot health.

        Returns:
            Health status dictionary
        """
        issues = []
        warnings = []

        # Check if loop is running
        if self.last_loop_time:
            time_since_loop = (datetime.now() - self.last_loop_time).total_seconds()
            if time_since_loop > 60:  # No loop in last minute
                issues.append({
                    "issue": "loop_stalled",
                    "severity": "critical",
                    "message": f"No loop iteration for {time_since_loop:.0f}s",
                })

        # Check MT5 connection
        if not self.mt5_connected:
            issues.append({
                "issue": "mt5_disconnected",
                "severity": "critical",
                "message": "MT5 not connected",
            })

        # Check error rate
        if self.loop_count > 0:
            error_rate = (self.error_count / self.loop_count) * 100
            if error_rate > 10:  # More than 10% errors
                warnings.append({
                    "issue": "high_error_rate",
                    "severity": "warning",
                    "message": f"Error rate: {error_rate:.1f}%",
                })

        # Check order success rate
        total_orders = self.total_orders_executed + self.total_orders_failed
        if total_orders > 0:
            success_rate = (self.total_orders_executed / total_orders) * 100
            if success_rate < 80:  # Less than 80% success
                warnings.append({
                    "issue": "low_order_success",
                    "severity": "warning",
                    "message": f"Order success rate: {success_rate:.1f}%",
                })

        # Determine overall health
        if issues:
            status = "unhealthy"
        elif warnings:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "healthy": len(issues) == 0,
            "status": status,
            "issues": issues,
            "warnings": warnings,
        }

    def get_statistics(self) -> Dict:
        """
        Get bot statistics.

        Returns:
            Statistics dictionary
        """
        uptime = datetime.now() - self.start_time
        uptime_hours = uptime.total_seconds() / 3600

        # Calculate rates
        loops_per_second = self.loop_count / uptime.total_seconds() if uptime.total_seconds() > 0 else 0

        return {
            "uptime_seconds": uptime.total_seconds(),
            "uptime_hours": uptime_hours,
            "start_time": self.start_time,
            "loop_count": self.loop_count,
            "loops_per_second": loops_per_second,
            "error_count": self.error_count,
            "error_rate_percent": (self.error_count / self.loop_count * 100) if self.loop_count > 0 else 0,
            "signals_generated": self.total_signals_generated,
            "orders_executed": self.total_orders_executed,
            "orders_failed": self.total_orders_failed,
            "order_success_rate": (
                self.total_orders_executed / (self.total_orders_executed + self.total_orders_failed) * 100
                if (self.total_orders_executed + self.total_orders_failed) > 0 else 0
            ),
            "mt5_connected": self.mt5_connected,
            "last_loop_time": self.last_loop_time,
        }

    def get_health_summary(self) -> str:
        """
        Get human-readable health summary.

        Returns:
            Summary string
        """
        health = self.check_health()
        stats = self.get_statistics()

        lines = [
            f"Bot Health: {health['status'].upper()}",
            f"  Uptime: {stats['uptime_hours']:.1f}h",
            f"  Loops: {stats['loop_count']} ({stats['loops_per_second']:.2f}/s)",
            f"  Errors: {stats['error_count']} ({stats['error_rate_percent']:.1f}%)",
            f"  MT5: {'Connected' if stats['mt5_connected'] else 'Disconnected'}",
        ]

        # Add issues
        if health.get("issues"):
            lines.append("  ISSUES:")
            for issue in health["issues"]:
                lines.append(f"    - {issue['message']}")

        # Add warnings
        if health.get("warnings"):
            lines.append("  WARNINGS:")
            for warning in health["warnings"]:
                lines.append(f"    - {warning['message']}")

        # Add recent errors
        if self.last_errors:
            lines.append(f"  Recent Errors: {len(self.last_errors)}")

        return "\n".join(lines)

    def reset_statistics(self) -> None:
        """Reset statistics (for testing)."""
        self.loop_count = 0
        self.error_count = 0
        self.last_errors = []
        self.total_signals_generated = 0
        self.total_orders_executed = 0
        self.total_orders_failed = 0
        self.logger.info("Health monitor statistics reset")

    def get_recent_errors(self) -> list:
        """
        Get recent errors.

        Returns:
            List of recent errors
        """
        return self.last_errors.copy()
