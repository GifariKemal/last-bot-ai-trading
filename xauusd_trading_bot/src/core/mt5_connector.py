"""
MetaTrader 5 connector with robust connection handling and retry logic.
Handles all interactions with MT5 platform.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import pandas as pd

from ..bot_logger import get_logger
from ..utils.config_loader import load_config


class MT5Connector:
    """Manages connection and operations with MetaTrader 5."""

    def __init__(self, config_path: str = "config"):
        """
        Initialize MT5 connector.

        Args:
            config_path: Path to configuration directory
        """
        self.logger = get_logger()
        self.config = load_config("mt5_config")
        self.settings = load_config("settings")

        self.connected = False
        self.symbol = self.settings.get("trading", {}).get("symbol", "XAUUSDm")
        self.login = self.config["connection"]["login"]
        self.password = self.config["connection"]["password"]
        self.server = self.config["connection"]["server"]
        self.terminal_path = self.config["connection"].get("terminal_path")
        self.timeout = self.config["connection"].get("timeout", 60000)

        # Retry configuration
        self.max_attempts = self.config["retry"]["max_attempts"]
        self.initial_delay = self.config["retry"]["initial_delay"]
        self.max_delay = self.config["retry"]["max_delay"]
        self.exponential_base = self.config["retry"]["exponential_base"]

        # Symbol info
        self.symbol_info = None
        self.point = 0.001  # XAUUSDm point value (Exness 3-digit)
        self.digits = 3  # XAUUSDm decimal places

        # Execution settings (configurable for live vs demo)
        _exec_cfg = self.config.get("execution", {})
        self._deviation = _exec_cfg.get("max_slippage_points", 50)
        _fp_map = {
            "FOK": mt5.ORDER_FILLING_FOK,
            "IOC": mt5.ORDER_FILLING_IOC,
            "RETURN": mt5.ORDER_FILLING_RETURN,
        }
        self._filling_type = _fp_map.get(
            _exec_cfg.get("fill_policy", "FOK"),
            mt5.ORDER_FILLING_FOK,
        )

    def connect(self) -> bool:
        """
        Connect to MT5 with retry logic.

        Returns:
            True if connected successfully, False otherwise
        """
        if self.connected and mt5.terminal_info() is not None:
            return True

        for attempt in range(1, self.max_attempts + 1):
            try:
                self.logger.info(f"Attempting MT5 connection (attempt {attempt}/{self.max_attempts})...")

                # Initialize MT5
                if self.terminal_path:
                    if not mt5.initialize(
                        path=self.terminal_path,
                        login=self.login,
                        password=self.password,
                        server=self.server,
                        timeout=self.timeout,
                    ):
                        error = mt5.last_error()
                        self.logger.error(f"MT5 initialize failed: {error}")
                        raise ConnectionError(f"MT5 initialize failed: {error}")
                else:
                    if not mt5.initialize(
                        login=self.login,
                        password=self.password,
                        server=self.server,
                        timeout=self.timeout,
                    ):
                        error = mt5.last_error()
                        self.logger.error(f"MT5 initialize failed: {error}")
                        raise ConnectionError(f"MT5 initialize failed: {error}")

                # Verify connection
                terminal_info = mt5.terminal_info()
                if terminal_info is None:
                    raise ConnectionError("Failed to get terminal info")

                account_info = mt5.account_info()
                if account_info is None:
                    raise ConnectionError("Failed to get account info")

                # Get symbol info
                self.symbol_info = mt5.symbol_info(self.symbol)
                if self.symbol_info is None:
                    self.logger.warning(f"Symbol {self.symbol} not found, trying to enable...")
                    if not mt5.symbol_select(self.symbol, True):
                        raise ConnectionError(f"Failed to select symbol {self.symbol}")
                    self.symbol_info = mt5.symbol_info(self.symbol)

                if self.symbol_info is not None:
                    self.point = self.symbol_info.point
                    self.digits = self.symbol_info.digits

                self.connected = True
                self.logger.info(
                    f"Connected to MT5 successfully | "
                    f"Account: {account_info.login} | "
                    f"Server: {account_info.server} | "
                    f"Balance: {account_info.balance:.2f} | "
                    f"Equity: {account_info.equity:.2f}"
                )
                return True

            except Exception as e:
                self.logger.error(f"Connection attempt {attempt} failed: {e}")
                self.connected = False

                if attempt < self.max_attempts:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.initial_delay * (self.exponential_base ** (attempt - 1)),
                        self.max_delay,
                    )
                    self.logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error("Max connection attempts reached. Giving up.")
                    return False

        return False

    def disconnect(self) -> None:
        """Disconnect from MT5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Disconnected from MT5")

    def ensure_connected(self) -> bool:
        """
        Ensure MT5 is connected, attempt reconnection if not.

        Returns:
            True if connected, False otherwise
        """
        if not self.connected or mt5.terminal_info() is None:
            self.logger.warning("MT5 not connected, attempting reconnection...")
            return self.connect()
        return True

    def get_bars(
        self,
        symbol: Optional[str] = None,
        timeframe: int = mt5.TIMEFRAME_M15,
        count: int = 1000,
        start_pos: int = 0,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical bars from MT5.

        Args:
            symbol: Trading symbol (default: configured symbol)
            timeframe: MT5 timeframe constant
            count: Number of bars to retrieve
            start_pos: Start position (0 = latest)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self.ensure_connected():
            return None

        symbol = symbol or self.symbol

        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)

            if rates is None or len(rates) == 0:
                self.logger.error(f"Failed to get bars for {symbol}: {mt5.last_error()}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")

            self.logger.bind(market=True).debug(
                f"Retrieved {len(df)} bars for {symbol} | "
                f"Timeframe: {timeframe} | "
                f"From: {df['time'].iloc[0]} | "
                f"To: {df['time'].iloc[-1]}"
            )

            return df

        except Exception as e:
            self.logger.error(f"Error getting bars: {e}")
            return None

    def get_bars_range(
        self,
        symbol: Optional[str] = None,
        timeframe: int = mt5.TIMEFRAME_M15,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical bars for a date range.

        Args:
            symbol: Trading symbol
            timeframe: MT5 timeframe constant
            date_from: Start date (UTC)
            date_to: End date (UTC)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self.ensure_connected():
            return None

        symbol = symbol or self.symbol

        try:
            # Default to last 30 days if dates not provided
            if date_to is None:
                date_to = datetime.now(timezone.utc)
            if date_from is None:
                date_from = datetime.now(timezone.utc).replace(day=1)

            rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

            if rates is None or len(rates) == 0:
                self.logger.error(f"Failed to get bars for date range: {mt5.last_error()}")
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")

            self.logger.bind(market=True).debug(
                f"Retrieved {len(df)} bars for {symbol} in date range"
            )

            return df

        except Exception as e:
            self.logger.error(f"Error getting bars for date range: {e}")
            return None

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ):
        """
        Get historical data as Polars DataFrame (for backtesting compatibility).

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (M1, M5, M15, etc.)
            start_date: Start date
            end_date: End date

        Returns:
            Polars DataFrame with OHLCV data
        """
        import polars as pl

        # Convert timeframe string to MT5 constant
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_M15)

        # Get bars using existing method
        df_pandas = self.get_bars_range(
            symbol=symbol,
            timeframe=mt5_timeframe,
            date_from=start_date,
            date_to=end_date
        )

        if df_pandas is None or len(df_pandas) == 0:
            return None

        # Convert to Polars
        df_polars = pl.from_pandas(df_pandas)

        return df_polars

    def get_tick(self, symbol: Optional[str] = None) -> Optional[Dict]:
        """
        Get latest tick data.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with tick data or None if failed
        """
        if not self.ensure_connected():
            return None

        symbol = symbol or self.symbol

        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None

            return {
                "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "volume": tick.volume,
                "spread": tick.ask - tick.bid,
            }

        except Exception as e:
            self.logger.error(f"Error getting tick: {e}")
            return None

    def get_bar_time(self, symbol: str, timeframe: int) -> Optional[datetime]:
        """
        Get the open time of the current (most recent) bar — fast path.

        Bypasses full DataFrame construction; used by is_new_bar() every second
        to avoid the 10-bar fetch + Pandas→Polars + feature-add overhead.

        Args:
            symbol: Trading symbol
            timeframe: MT5 timeframe constant

        Returns:
            Bar open time (UTC, timezone-naive to match pandas_to_polars output)
        """
        if not self.ensure_connected():
            return None
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
            if rates is None or len(rates) == 0:
                return None
            return datetime.utcfromtimestamp(int(rates[0]["time"]))
        except Exception as e:
            self.logger.error(f"Error getting bar time: {e}")
            return None

    def get_account_info(self) -> Optional[Dict]:
        """
        Get account information.

        Returns:
            Dictionary with account info or None if failed
        """
        if not self.ensure_connected():
            return None

        try:
            account = mt5.account_info()
            if account is None:
                return None

            return {
                "login": account.login,
                "server": account.server,
                "balance": account.balance,
                "equity": account.equity,
                "profit": account.profit,
                "margin": account.margin,
                "margin_free": account.margin_free,
                "margin_level": account.margin_level,
                "leverage": account.leverage,
                "currency": account.currency,
            }

        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get open positions.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of position dictionaries
        """
        if not self.ensure_connected():
            return []

        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()

            if positions is None:
                return []

            result = []
            for pos in positions:
                result.append(
                    {
                        "ticket": pos.ticket,
                        "symbol": pos.symbol,
                        "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                        "volume": pos.volume,
                        "open_price": pos.price_open,
                        "current_price": pos.price_current,
                        "sl": pos.sl,
                        "tp": pos.tp,
                        "profit": pos.profit,
                        "open_time": datetime.fromtimestamp(pos.time, tz=timezone.utc),
                        "comment": pos.comment,
                    }
                )

            return result

        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []

    def send_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "",
        magic: int = 0,
    ) -> Optional[Dict]:
        """
        Send a trade order to MT5.

        Args:
            symbol: Trading symbol
            order_type: "BUY" or "SELL"
            volume: Lot size
            price: Entry price (None for market order)
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment
            magic: Magic number

        Returns:
            Order result dictionary or None if failed
        """
        if not self.ensure_connected():
            return None

        try:
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
                "deviation": self._deviation,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._filling_type,
            }

            # Add price (or use current market price)
            if price is not None:
                request["price"] = price
            else:
                tick = self.get_tick(symbol)
                if tick is None:
                    self.logger.error("Failed to get current price")
                    return None
                request["price"] = tick["ask"] if order_type == "BUY" else tick["bid"]

            # Add SL/TP if provided
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp

            # Send order
            result = mt5.order_send(request)

            if result is None:
                self.logger.error(f"Order send failed: {mt5.last_error()}")
                return None

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return None

            self.logger.bind(trade=True).info(
                f"ORDER EXECUTED | "
                f"Ticket: {result.order} | "
                f"Type: {order_type} | "
                f"Volume: {volume} | "
                f"Price: {result.price:.2f} | "
                f"SL: {f'{sl:.2f}' if sl else 'None'} | "
                f"TP: {f'{tp:.2f}' if tp else 'None'}"
            )

            return {
                "ticket": result.order,
                "retcode": result.retcode,
                "deal": result.deal,
                "volume": result.volume,
                "price": result.price,
                "comment": result.comment,
            }

        except Exception as e:
            self.logger.error(f"Error sending order: {e}")
            return None

    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        """
        Modify an existing position's SL/TP.

        Args:
            ticket: Position ticket number
            sl: New stop loss price
            tp: New take profit price

        Returns:
            True if modification successful
        """
        if not self.ensure_connected():
            return False

        try:
            # Get position
            position = mt5.positions_get(ticket=ticket)
            if not position:
                self.logger.error(f"Position {ticket} not found")
                return False

            position = position[0]

            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": ticket,
                "sl": sl if sl is not None else position.sl,
                "tp": tp if tp is not None else position.tp,
            }

            # Send modification
            result = mt5.order_send(request)

            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                # Check if it's a "no change needed" scenario (not a real error)
                error_info = mt5.last_error()
                if result is not None and error_info and error_info[1] == 'Success':
                    self.logger.debug(f"Position modification: no change needed for ticket {ticket}")
                    return True
                self.logger.error(f"Position modification failed: {error_info}")
                return False

            actual_sl = sl if sl is not None else position.sl
            actual_tp = tp if tp is not None else position.tp
            self.logger.bind(trade=True).info(
                f"POSITION MODIFIED | Ticket: {ticket} | SL: {actual_sl} | TP: {actual_tp}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error modifying position: {e}")
            return False

    def close_position(self, ticket: int, volume: Optional[float] = None) -> bool:
        """
        Close a position.

        Args:
            ticket: Position ticket number
            volume: Volume to close (None = close all)

        Returns:
            True if closed successfully
        """
        if not self.ensure_connected():
            return False

        try:
            # Get position
            position = mt5.positions_get(ticket=ticket)
            if not position:
                self.logger.error(f"Position {ticket} not found")
                return False

            position = position[0]
            close_volume = volume if volume is not None else position.volume

            # Determine close order type (opposite of open)
            close_type = (
                mt5.ORDER_TYPE_SELL
                if position.type == mt5.ORDER_TYPE_BUY
                else mt5.ORDER_TYPE_BUY
            )

            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": close_volume,
                "type": close_type,
                "position": ticket,
                "deviation": self._deviation,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._filling_type,
            }

            # Get current price
            tick = self.get_tick(position.symbol)
            if tick is None:
                return False

            request["price"] = tick["bid"] if close_type == mt5.ORDER_TYPE_SELL else tick["ask"]

            # Send close order
            result = mt5.order_send(request)

            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error_info = mt5.last_error()
                # MT5 sometimes returns retcode != 10009 but last_error is (1, 'Success')
                if result is not None and error_info and error_info[1] == 'Success':
                    self.logger.bind(trade=True).info(
                        f"POSITION CLOSED | Ticket: {ticket} | Volume: {close_volume}"
                    )
                    return True
                self.logger.error(f"Position close failed: {error_info}")
                return False

            self.logger.bind(trade=True).info(
                f"POSITION CLOSED | Ticket: {ticket} | Volume: {close_volume}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False

    def close_all_positions(self, symbol: Optional[str] = None) -> int:
        """
        Close all open positions.

        Args:
            symbol: Close positions for specific symbol (None = all symbols)

        Returns:
            Number of positions closed
        """
        positions = self.get_positions(symbol)
        closed_count = 0

        for pos in positions:
            if self.close_position(pos["ticket"]):
                closed_count += 1

        self.logger.info(f"Closed {closed_count} positions")
        return closed_count
