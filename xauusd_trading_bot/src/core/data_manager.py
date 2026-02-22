"""
Data manager using Polars for fast data processing.
Converts MT5 data to Polars DataFrames for efficient analysis.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import polars as pl

from ..bot_logger import get_logger


class DataManager:
    """Manage and process market data using Polars."""

    def __init__(self):
        """Initialize data manager."""
        self.logger = get_logger()
        self._cache: Dict[str, pl.DataFrame] = {}

    def pandas_to_polars(self, df: pd.DataFrame) -> pl.DataFrame:
        """
        Convert pandas DataFrame to Polars DataFrame.

        Args:
            df: Pandas DataFrame with OHLCV data

        Returns:
            Polars DataFrame
        """
        try:
            # Convert to Polars
            pl_df = pl.from_pandas(df)

            # Ensure correct data types
            pl_df = pl_df.with_columns(
                [
                    pl.col("time").cast(pl.Datetime),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("tick_volume").cast(pl.Int64),
                ]
            )

            # Sort by time
            pl_df = pl_df.sort("time")

            return pl_df

        except Exception as e:
            self.logger.error(f"Error converting DataFrame: {e}")
            return pl.DataFrame()

    def add_basic_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add basic price features to DataFrame.

        Args:
            df: Polars DataFrame with OHLCV data

        Returns:
            DataFrame with additional features
        """
        try:
            df = df.with_columns(
                [
                    # Price ranges
                    (pl.col("high") - pl.col("low")).alias("range"),
                    (pl.col("close") - pl.col("open")).alias("body"),
                    (pl.col("high") - pl.max_horizontal("open", "close")).alias("upper_wick"),
                    (pl.min_horizontal("open", "close") - pl.col("low")).alias("lower_wick"),
                    # Typical price (HLC/3)
                    ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias(
                        "typical_price"
                    ),
                    # Median price (HL/2)
                    ((pl.col("high") + pl.col("low")) / 2).alias("median_price"),
                    # Weighted close (HLCC/4)
                    (
                        (pl.col("high") + pl.col("low") + pl.col("close") * 2) / 4
                    ).alias("weighted_close"),
                    # Bullish/Bearish candle
                    (pl.col("close") > pl.col("open")).alias("is_bullish"),
                    (pl.col("close") < pl.col("open")).alias("is_bearish"),
                    # Body percentage of total range
                    (
                        (pl.col("close") - pl.col("open")).abs()
                        / (pl.col("high") - pl.col("low"))
                        * 100
                    ).alias("body_percent"),
                ]
            )

            return df

        except Exception as e:
            self.logger.error(f"Error adding basic features: {e}")
            return df

    def add_price_changes(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add price change calculations.

        Args:
            df: Polars DataFrame

        Returns:
            DataFrame with price changes
        """
        try:
            df = df.with_columns(
                [
                    # Price changes
                    (pl.col("close") - pl.col("close").shift(1)).alias("close_change"),
                    (
                        (pl.col("close") - pl.col("close").shift(1))
                        / pl.col("close").shift(1)
                        * 100
                    ).alias("close_change_pct"),
                    # High/Low changes
                    (pl.col("high") - pl.col("high").shift(1)).alias("high_change"),
                    (pl.col("low") - pl.col("low").shift(1)).alias("low_change"),
                ]
            )

            return df

        except Exception as e:
            self.logger.error(f"Error adding price changes: {e}")
            return df

    def add_rolling_features(
        self, df: pl.DataFrame, windows: List[int] = [5, 10, 20, 50]
    ) -> pl.DataFrame:
        """
        Add rolling window features.

        Args:
            df: Polars DataFrame
            windows: List of window sizes

        Returns:
            DataFrame with rolling features
        """
        try:
            for window in windows:
                df = df.with_columns(
                    [
                        # Rolling means
                        pl.col("close").rolling_mean(window).alias(f"sma_{window}"),
                        pl.col("high").rolling_max(window).alias(f"high_{window}"),
                        pl.col("low").rolling_min(window).alias(f"low_{window}"),
                        # Rolling volatility
                        pl.col("close").rolling_std(window).alias(f"std_{window}"),
                    ]
                )

            return df

        except Exception as e:
            self.logger.error(f"Error adding rolling features: {e}")
            return df

    def resample_timeframe(
        self, df: pl.DataFrame, timeframe: str = "15m"
    ) -> pl.DataFrame:
        """
        Resample data to different timeframe.

        Args:
            df: Polars DataFrame with time column
            timeframe: Target timeframe (e.g., "5m", "15m", "1h")

        Returns:
            Resampled DataFrame
        """
        try:
            # Group by timeframe and aggregate
            resampled = (
                df.group_by_dynamic("time", every=timeframe)
                .agg(
                    [
                        pl.col("open").first(),
                        pl.col("high").max(),
                        pl.col("low").min(),
                        pl.col("close").last(),
                        pl.col("tick_volume").sum(),
                    ]
                )
                .sort("time")
            )

            return resampled

        except Exception as e:
            self.logger.error(f"Error resampling timeframe: {e}")
            return df

    def merge_timeframes(
        self, df_primary: pl.DataFrame, df_higher: pl.DataFrame, suffix: str = "_htf"
    ) -> pl.DataFrame:
        """
        Merge higher timeframe data into primary timeframe.

        Args:
            df_primary: Primary (lower) timeframe DataFrame
            df_higher: Higher timeframe DataFrame
            suffix: Suffix for higher timeframe columns

        Returns:
            Merged DataFrame
        """
        try:
            # Join dataframes based on time
            # Using asof join to align timestamps
            merged = df_primary.join_asof(
                df_higher.select(
                    [
                        pl.col("time"),
                        pl.col("close").alias(f"close{suffix}"),
                        pl.col("high").alias(f"high{suffix}"),
                        pl.col("low").alias(f"low{suffix}"),
                    ]
                ),
                on="time",
                strategy="backward",
            )

            return merged

        except Exception as e:
            self.logger.error(f"Error merging timeframes: {e}")
            return df_primary

    def get_latest_bars(self, df: pl.DataFrame, n: int = 1) -> pl.DataFrame:
        """
        Get latest N bars from DataFrame.

        Args:
            df: Polars DataFrame
            n: Number of bars to retrieve

        Returns:
            DataFrame with latest N bars
        """
        return df.tail(n)

    def get_bar_at_index(self, df: pl.DataFrame, index: int) -> Optional[Dict]:
        """
        Get bar at specific index.

        Args:
            df: Polars DataFrame
            index: Bar index (negative for from end)

        Returns:
            Dictionary with bar data or None
        """
        try:
            if index < 0:
                index = len(df) + index

            if index < 0 or index >= len(df):
                return None

            row = df[index]
            return row.to_dicts()[0]

        except Exception as e:
            self.logger.error(f"Error getting bar at index: {e}")
            return None

    def get_latest_bar(self, df: pl.DataFrame) -> Optional[Dict]:
        """
        Get the most recent bar.

        Args:
            df: Polars DataFrame

        Returns:
            Dictionary with bar data or None
        """
        return self.get_bar_at_index(df, -1)

    def cache_dataframe(self, key: str, df: pl.DataFrame) -> None:
        """
        Cache a DataFrame.

        Args:
            key: Cache key
            df: DataFrame to cache
        """
        self._cache[key] = df

    def get_cached_dataframe(self, key: str) -> Optional[pl.DataFrame]:
        """
        Get cached DataFrame.

        Args:
            key: Cache key

        Returns:
            Cached DataFrame or None
        """
        return self._cache.get(key)

    def clear_cache(self, key: Optional[str] = None) -> None:
        """
        Clear cache.

        Args:
            key: Specific key to clear (None = clear all)
        """
        if key:
            if key in self._cache:
                del self._cache[key]
        else:
            self._cache.clear()

    def validate_dataframe(self, df: pl.DataFrame) -> bool:
        """
        Validate DataFrame has required columns and data.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid
        """
        required_columns = ["time", "open", "high", "low", "close"]

        # Check if DataFrame is empty
        if df.is_empty():
            self.logger.warning("DataFrame is empty")
            return False

        # Check required columns
        for col in required_columns:
            if col not in df.columns:
                self.logger.error(f"Missing required column: {col}")
                return False

        # Check for null values in critical columns
        for col in required_columns:
            null_count = df[col].null_count()
            if null_count > 0:
                self.logger.warning(f"Column {col} has {null_count} null values")

        return True

    def get_dataframe_info(self, df: pl.DataFrame) -> Dict:
        """
        Get information about a DataFrame.

        Args:
            df: Polars DataFrame

        Returns:
            Dictionary with DataFrame info
        """
        if df.is_empty():
            return {"empty": True}

        return {
            "empty": False,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns,
            "start_time": df["time"][0] if "time" in df.columns else None,
            "end_time": df["time"][-1] if "time" in df.columns else None,
            "memory_usage": df.estimated_size(),
        }
