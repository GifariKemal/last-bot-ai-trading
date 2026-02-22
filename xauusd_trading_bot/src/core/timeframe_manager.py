"""
Timeframe manager for multi-timeframe data synchronization.
Ensures all timeframes are properly aligned and up-to-date.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

import MetaTrader5 as mt5
import polars as pl

from .constants import Timeframe
from .data_manager import DataManager
from .mt5_connector import MT5Connector
from ..bot_logger import get_logger


class TimeframeManager:
    """Manage multiple timeframes and keep them synchronized."""

    def __init__(self, mt5_connector: MT5Connector, data_manager: DataManager):
        """
        Initialize timeframe manager.

        Args:
            mt5_connector: MT5 connector instance
            data_manager: Data manager instance
        """
        self.logger = get_logger()
        self.mt5 = mt5_connector
        self.data_manager = data_manager

        # Timeframe mapping (string to MT5 constant)
        self.timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
        }

        # Cache for timeframe data
        self.data_cache: Dict[str, pl.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}

    def get_mt5_timeframe(self, timeframe_str: str) -> int:
        """
        Convert timeframe string to MT5 constant.

        Args:
            timeframe_str: Timeframe string (e.g., "M15")

        Returns:
            MT5 timeframe constant
        """
        return self.timeframe_map.get(timeframe_str.upper(), mt5.TIMEFRAME_M15)

    def get_timeframe_minutes(self, timeframe_str: str) -> int:
        """
        Get timeframe duration in minutes.

        Args:
            timeframe_str: Timeframe string

        Returns:
            Duration in minutes
        """
        timeframe_minutes = {
            "M1": 1,
            "M5": 5,
            "M15": 15,
            "M30": 30,
            "H1": 60,
            "H4": 240,
            "D1": 1440,
            "W1": 10080,
        }
        return timeframe_minutes.get(timeframe_str.upper(), 15)

    def fetch_timeframe_data(
        self,
        symbol: str,
        timeframe_str: str,
        bars: int = 1000,
        use_cache: bool = True,
    ) -> Optional[pl.DataFrame]:
        """
        Fetch data for a specific timeframe.

        Args:
            symbol: Trading symbol
            timeframe_str: Timeframe string (e.g., "M15")
            bars: Number of bars to fetch
            use_cache: Use cached data if available

        Returns:
            Polars DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe_str}"

        # Check cache
        if use_cache and cache_key in self.data_cache:
            cached_df = self.data_cache[cache_key]
            last_update = self.last_update.get(cache_key)

            if last_update:
                # Check if cache is still fresh (within timeframe duration)
                tf_minutes = self.get_timeframe_minutes(timeframe_str)
                time_since_update = (
                    datetime.now(timezone.utc) - last_update
                ).total_seconds() / 60

                if time_since_update < tf_minutes:
                    self.logger.debug(
                        f"Using cached data for {symbol} {timeframe_str}",
                    )
                    return cached_df

        # Fetch fresh data
        mt5_timeframe = self.get_mt5_timeframe(timeframe_str)
        df_pandas = self.mt5.get_bars(
            symbol=symbol,
            timeframe=mt5_timeframe,
            count=bars,
        )

        if df_pandas is None or df_pandas.empty:
            self.logger.error(f"Failed to fetch data for {symbol} {timeframe_str}")
            return None

        # Convert to Polars
        df = self.data_manager.pandas_to_polars(df_pandas)

        # Add basic features
        df = self.data_manager.add_basic_features(df)
        df = self.data_manager.add_price_changes(df)

        # Cache the data
        self.data_cache[cache_key] = df
        self.last_update[cache_key] = datetime.now(timezone.utc)

        self.logger.bind(market=True).debug(
            f"Fetched {len(df)} bars for {symbol} {timeframe_str}"
        )

        return df

    def fetch_multiple_timeframes(
        self,
        symbol: str,
        timeframes: List[str],
        bars: int = 1000,
        use_cache: bool = True,
    ) -> Dict[str, pl.DataFrame]:
        """
        Fetch data for multiple timeframes.

        Args:
            symbol: Trading symbol
            timeframes: List of timeframe strings
            bars: Number of bars to fetch
            use_cache: Use cached data if available

        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        result = {}

        for tf in timeframes:
            df = self.fetch_timeframe_data(
                symbol=symbol,
                timeframe_str=tf,
                bars=bars,
                use_cache=use_cache,
            )

            if df is not None:
                result[tf] = df
            else:
                self.logger.warning(f"Failed to fetch {tf} data")

        self.logger.debug(
            f"Fetched data for {len(result)}/{len(timeframes)} timeframes"
        )

        return result

    def align_timeframes(
        self,
        data: Dict[str, pl.DataFrame],
        reference_timeframe: str = "M15",
    ) -> Dict[str, pl.DataFrame]:
        """
        Align multiple timeframes to the reference timeframe.

        Args:
            data: Dictionary of timeframe DataFrames
            reference_timeframe: Reference timeframe for alignment

        Returns:
            Aligned timeframe data
        """
        if reference_timeframe not in data:
            self.logger.error(f"Reference timeframe {reference_timeframe} not found")
            return data

        ref_df = data[reference_timeframe]
        aligned_data = {reference_timeframe: ref_df}

        # Align other timeframes
        for tf, df in data.items():
            if tf == reference_timeframe:
                continue

            # Merge higher timeframe data into reference
            if self.get_timeframe_minutes(tf) > self.get_timeframe_minutes(
                reference_timeframe
            ):
                aligned_data[tf] = df
            else:
                # For lower timeframes, just keep as is
                aligned_data[tf] = df

        return aligned_data

    def get_latest_bar(
        self, symbol: str, timeframe_str: str
    ) -> Optional[Dict]:
        """
        Get the latest bar for a timeframe.

        Args:
            symbol: Trading symbol
            timeframe_str: Timeframe string

        Returns:
            Dictionary with bar data
        """
        df = self.fetch_timeframe_data(
            symbol=symbol,
            timeframe_str=timeframe_str,
            bars=10,
            use_cache=False,  # Always fetch fresh for latest bar
        )

        if df is None or df.is_empty():
            return None

        return self.data_manager.get_latest_bar(df)

    def get_latest_bars(
        self, symbol: str, timeframe_str: str, n: int = 10
    ) -> Optional[pl.DataFrame]:
        """
        Get the latest N bars for a timeframe.

        Args:
            symbol: Trading symbol
            timeframe_str: Timeframe string
            n: Number of bars to get

        Returns:
            DataFrame with latest N bars
        """
        df = self.fetch_timeframe_data(
            symbol=symbol,
            timeframe_str=timeframe_str,
            bars=n * 2,  # Fetch extra to ensure we have enough
            use_cache=False,
        )

        if df is None or df.is_empty():
            return None

        return self.data_manager.get_latest_bars(df, n)

    def update_all_timeframes(
        self, symbol: str, timeframes: List[str]
    ) -> Dict[str, pl.DataFrame]:
        """
        Update all timeframes with fresh data.

        Args:
            symbol: Trading symbol
            timeframes: List of timeframe strings

        Returns:
            Updated timeframe data
        """
        return self.fetch_multiple_timeframes(
            symbol=symbol,
            timeframes=timeframes,
            use_cache=False,
        )

    def clear_cache(self, timeframe: Optional[str] = None) -> None:
        """
        Clear timeframe cache.

        Args:
            timeframe: Specific timeframe to clear (None = all)
        """
        if timeframe:
            keys_to_remove = [k for k in self.data_cache.keys() if timeframe in k]
            for key in keys_to_remove:
                del self.data_cache[key]
                if key in self.last_update:
                    del self.last_update[key]
        else:
            self.data_cache.clear()
            self.last_update.clear()

        self.logger.debug("Timeframe cache cleared")

    def get_cache_info(self) -> Dict:
        """
        Get information about cached data.

        Returns:
            Dictionary with cache information
        """
        info = {}

        for key, df in self.data_cache.items():
            info[key] = {
                "rows": len(df),
                "columns": len(df.columns),
                "last_update": self.last_update.get(key),
                "memory_mb": df.estimated_size() / 1024 / 1024,
            }

        return info

    def is_new_bar(
        self, symbol: str, timeframe_str: str, last_known_time: datetime
    ) -> bool:
        """
        Check if a new bar has formed since last known time.

        Uses a fast-path MT5 call (1 bar, raw timestamp) instead of the full
        10-bar fetch + Pandas→Polars + feature-add pipeline.  This runs every
        ~1 second so the old pipeline was wasting ~900 full fetches per candle.

        Args:
            symbol: Trading symbol
            timeframe_str: Timeframe string
            last_known_time: Last known bar open time

        Returns:
            True if new bar has formed
        """
        mt5_tf = self.get_mt5_timeframe(timeframe_str)
        bar_time = self.mt5.get_bar_time(symbol, mt5_tf)

        if bar_time is None:
            return False

        return bar_time > last_known_time
