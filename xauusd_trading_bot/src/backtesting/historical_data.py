"""
Historical Data Manager
Manages loading and caching of historical market data for backtesting.
"""

import polars as pl
from typing import Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
from ..core.mt5_connector import MT5Connector
from ..bot_logger import get_logger


class HistoricalDataManager:
    """Manage historical data for backtesting."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize historical data manager.

        Args:
            data_dir: Directory for storing historical data
        """
        self.logger = get_logger()

        # Data directory
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent.parent / "data" / "market_history"

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Cache
        self.cache = {}

    def fetch_historical_data(
        self,
        mt5: MT5Connector,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> Optional[pl.DataFrame]:
        """
        Fetch historical data from MT5.

        Args:
            mt5: MT5 connector instance
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, M15, H1, etc.)
            start_date: Start date
            end_date: End date
            use_cache: Use cached data if available

        Returns:
            Polars DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}_{start_date.date()}_{end_date.date()}"

        # Check cache
        if use_cache and cache_key in self.cache:
            self.logger.debug(f"Using cached data for {cache_key}")
            return self.cache[cache_key]

        # Check file cache
        cache_file = self.data_dir / f"{cache_key}.parquet"
        if use_cache and cache_file.exists():
            self.logger.info(f"Loading cached data from {cache_file}")
            df = pl.read_parquet(cache_file)
            self.cache[cache_key] = df
            return df

        # Fetch from MT5
        self.logger.info(
            f"Fetching historical data: {symbol} {timeframe} "
            f"from {start_date.date()} to {end_date.date()}"
        )

        df = mt5.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if df is None or len(df) == 0:
            self.logger.error("Failed to fetch historical data")
            return None

        # Save to cache
        self.logger.info(f"Saving {len(df)} bars to cache")
        df.write_parquet(cache_file)
        self.cache[cache_key] = df

        return df

    def fetch_multiple_timeframes(
        self,
        mt5: MT5Connector,
        symbol: str,
        timeframes: list,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> Dict[str, pl.DataFrame]:
        """
        Fetch multiple timeframes.

        Args:
            mt5: MT5 connector instance
            symbol: Trading symbol
            timeframes: List of timeframes
            start_date: Start date
            end_date: End date
            use_cache: Use cached data

        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        result = {}

        for tf in timeframes:
            df = self.fetch_historical_data(
                mt5, symbol, tf, start_date, end_date, use_cache
            )
            if df is not None:
                result[tf] = df
            else:
                self.logger.warning(f"Failed to fetch {tf} data")

        return result

    def prepare_backtest_data(
        self,
        mt5: MT5Connector,
        symbol: str,
        primary_timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> Optional[pl.DataFrame]:
        """
        Prepare data for backtesting.

        Args:
            mt5: MT5 connector
            symbol: Trading symbol
            primary_timeframe: Primary timeframe for backtesting
            start_date: Start date
            end_date: End date
            use_cache: Use cached data

        Returns:
            Prepared DataFrame
        """
        # Fetch primary timeframe data
        df = self.fetch_historical_data(
            mt5, symbol, primary_timeframe, start_date, end_date, use_cache
        )

        if df is None:
            return None

        # Validate data
        if len(df) < 100:
            self.logger.error(f"Insufficient data: only {len(df)} bars")
            return None

        # Sort by time
        df = df.sort("time")

        # Check for gaps
        time_diffs = df["time"].diff()
        if time_diffs is not None and len(time_diffs) > 1:
            # Skip first None value
            valid_diffs = time_diffs[1:]
            if len(valid_diffs) > 0:
                max_diff = valid_diffs.max()
                expected_diff = self._get_expected_interval(primary_timeframe)

                if max_diff and expected_diff and max_diff > expected_diff * 3:
                    self.logger.warning(f"Large time gap detected: {max_diff}")

        self.logger.info(
            f"Prepared {len(df)} bars from {df['time'][0]} to {df['time'][-1]}"
        )

        return df

    def _get_expected_interval(self, timeframe: str) -> timedelta:
        """Get expected time interval for timeframe."""
        intervals = {
            "M1": timedelta(minutes=1),
            "M5": timedelta(minutes=5),
            "M15": timedelta(minutes=15),
            "M30": timedelta(minutes=30),
            "H1": timedelta(hours=1),
            "H4": timedelta(hours=4),
            "D1": timedelta(days=1),
        }
        return intervals.get(timeframe, timedelta(minutes=1))

    def get_data_info(self, df: pl.DataFrame) -> Dict:
        """
        Get information about dataset.

        Args:
            df: Data DataFrame

        Returns:
            Information dictionary
        """
        if df is None or len(df) == 0:
            return {
                "bars": 0,
                "start": None,
                "end": None,
                "duration_days": 0,
            }

        return {
            "bars": len(df),
            "start": df["time"][0],
            "end": df["time"][-1],
            "duration_days": (df["time"][-1] - df["time"][0]).days,
            "columns": df.columns,
        }

    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self.cache.clear()
        self.logger.info("Cache cleared")

    def delete_cached_files(self, symbol: Optional[str] = None) -> int:
        """
        Delete cached parquet files.

        Args:
            symbol: If provided, only delete files for this symbol

        Returns:
            Number of files deleted
        """
        count = 0
        pattern = f"{symbol}_*.parquet" if symbol else "*.parquet"

        for file in self.data_dir.glob(pattern):
            file.unlink()
            count += 1
            self.logger.debug(f"Deleted {file.name}")

        self.logger.info(f"Deleted {count} cached files")
        return count
