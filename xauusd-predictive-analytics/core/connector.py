"""
MetaTrader 5 connection handler.

Implements a context-manager-based connector so the MT5 terminal
is always shut down cleanly, even when an exception occurs downstream.

Usage
-----
    with MT5Connector(config.mt5) as conn:
        # terminal is initialised and logged-in here
        ...
    # mt5.shutdown() is called automatically on exit
"""

from __future__ import annotations

import MetaTrader5 as mt5
from loguru import logger

from config.settings import MT5Config


class MT5Connector:
    """Manages a single MetaTrader 5 terminal session."""

    def __init__(self, config: MT5Config) -> None:
        self.config = config
        self._connected: bool = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Initialise the MT5 terminal and log in with the configured account.

        Returns
        -------
        bool
            True on success, False on any failure.
        """
        logger.info("Initialising MetaTrader 5 terminal …")

        # Step 1 – initialise the terminal process
        init_kwargs: dict = {}
        if self.config.path:
            init_kwargs["path"] = self.config.path

        if not mt5.initialize(**init_kwargs):
            error = mt5.last_error()
            logger.error(
                "mt5.initialize() failed. "
                "Is MetaTrader 5 installed and running? "
                f"Error: {error}"
            )
            return False

        logger.success("MT5 terminal initialised.")

        # Step 2 – log in (skipped when no credentials are provided)
        if self.config.login:
            logger.info(
                f"Logging in as account #{self.config.login} "
                f"on {self.config.server} …"
            )
            authorised = mt5.login(
                login=self.config.login,
                password=self.config.password,
                server=self.config.server,
            )
            if not authorised:
                error = mt5.last_error()
                logger.error(f"mt5.login() failed. Error: {error}")
                mt5.shutdown()
                return False

            logger.success(f"Logged in as account #{self.config.login}.")

        self._connected = True
        return True

    def disconnect(self) -> None:
        """Shut down the MT5 terminal connection."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 terminal disconnected.")

    # ──────────────────────────────────────────────────────────────────────────
    # Context-manager support
    # ──────────────────────────────────────────────────────────────────────────

    def __enter__(self) -> "MT5Connector":
        if not self.connect():
            raise ConnectionError(
                "Failed to connect to MetaTrader 5. See logs above."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.disconnect()
