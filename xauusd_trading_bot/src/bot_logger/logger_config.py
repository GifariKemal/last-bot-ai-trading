"""
Logging configuration using Loguru.
Provides structured, comprehensive logging for all bot activities.
"""

import sys
from pathlib import Path
from typing import Optional
from datetime import timezone, timedelta

from loguru import logger

from ..utils.config_loader import get_config_value

# WIB (GMT+7) timezone
WIB = timezone(timedelta(hours=7))


def _dual_time_format(record):
    """Format log record with dual timestamps: UTC + WIB."""
    utc_time = record["time"].astimezone(timezone.utc)
    wib_time = record["time"].astimezone(WIB)
    record["extra"]["utc"] = utc_time.strftime("%H:%M:%S")
    record["extra"]["wib"] = wib_time.strftime("%H:%M:%S")


class LoggerConfig:
    """Configure and manage logging for the trading bot."""

    def __init__(self):
        """Initialize logger configuration."""
        self.logger = logger
        self._configured = False

    def setup(self, config_dir: str = "config") -> None:
        """
        Setup logging based on configuration.

        Args:
            config_dir: Path to configuration directory
        """
        if self._configured:
            return

        # Load logging configuration
        log_level = get_config_value("settings", "logging.level", "INFO")
        console_output = get_config_value("settings", "logging.console_output", True)
        file_output = get_config_value("settings", "logging.file_output", True)
        rotation = get_config_value("settings", "logging.rotation", "100 MB")
        retention = get_config_value("settings", "logging.retention", "30 days")

        # Remove default handler
        self.logger.remove()

        # Patch logger to inject dual timezone extras on every log call
        self.logger = self.logger.patch(_dual_time_format)

        # Console handler
        if console_output:
            self.logger.add(
                sys.stderr,
                format="<green>{time:YYYY-MM-DD}</green> "
                "<green>UTC {extra[utc]}</green> | "
                "<yellow>WIB {extra[wib]}</yellow> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>",
                level=log_level,
                colorize=True,
            )

        # File handlers
        if file_output:
            # Create log directories
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            (log_dir / "trades").mkdir(exist_ok=True)
            (log_dir / "bot_activity").mkdir(exist_ok=True)
            (log_dir / "market").mkdir(exist_ok=True)

            # Dual timestamp format for file logs
            file_fmt = "{time:YYYY-MM-DD} UTC {extra[utc]} | WIB {extra[wib]} | {level: <8} | {name}:{function}:{line} | {message}"
            file_fmt_short = "{time:YYYY-MM-DD} UTC {extra[utc]} | WIB {extra[wib]} | {level: <8} | {message}"

            # Main bot activity log
            self.logger.add(
                log_dir / "bot_activity" / "bot_{time:YYYY-MM-DD}.log",
                format=file_fmt,
                level=log_level,
                rotation=rotation,
                retention=retention,
                compression="zip",
            )

            # Trade-specific log
            self.logger.add(
                log_dir / "trades" / "trades_{time:YYYY-MM-DD}.log",
                format=file_fmt_short,
                level="INFO",
                rotation=rotation,
                retention=retention,
                compression="zip",
                filter=lambda record: "trade" in record["extra"],
            )

            # Market data log
            self.logger.add(
                log_dir / "market" / "market_{time:YYYY-MM-DD}.log",
                format=file_fmt_short,
                level="DEBUG",
                rotation=rotation,
                retention=retention,
                compression="zip",
                filter=lambda record: "market" in record["extra"],
            )

            # Error log
            self.logger.add(
                log_dir / "errors_{time:YYYY-MM-DD}.log",
                format=file_fmt,
                level="ERROR",
                rotation=rotation,
                retention=retention,
                compression="zip",
            )

        self._configured = True
        self.logger.info("Logger configured successfully")

    def get_logger(self):
        """Get the configured logger instance."""
        if not self._configured:
            self.setup()
        return self.logger


# Global logger instance
_logger_config: Optional[LoggerConfig] = None


def get_logger():
    """
    Get the global logger instance.

    Returns:
        Configured loguru logger
    """
    global _logger_config
    if _logger_config is None:
        _logger_config = LoggerConfig()
        _logger_config.setup()
    return _logger_config.get_logger()


def setup_logger(config_dir: str = "config") -> None:
    """
    Setup the global logger.

    Args:
        config_dir: Path to configuration directory
    """
    global _logger_config
    if _logger_config is None:
        _logger_config = LoggerConfig()
    _logger_config.setup(config_dir)
