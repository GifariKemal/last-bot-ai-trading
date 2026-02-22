"""
Base indicator class providing common interface for all indicators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import polars as pl

from ..bot_logger import get_logger


class BaseIndicator(ABC):
    """Abstract base class for all technical indicators."""

    def __init__(self, name: str):
        """
        Initialize base indicator.

        Args:
            name: Indicator name
        """
        self.name = name
        self.logger = get_logger()
        self._cache: Dict[str, Any] = {}

    @abstractmethod
    def calculate(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """
        Calculate indicator values.

        Args:
            df: Polars DataFrame with OHLCV data
            **kwargs: Additional parameters

        Returns:
            DataFrame with indicator columns added
        """
        pass

    def validate_dataframe(self, df: pl.DataFrame, required_columns: list) -> bool:
        """
        Validate that DataFrame has required columns.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            True if valid, raises ValueError otherwise
        """
        if df.is_empty():
            raise ValueError(f"{self.name}: DataFrame is empty")

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"{self.name}: Missing required columns: {missing_columns}"
            )

        return True

    def get_latest_value(self, df: pl.DataFrame, column: str) -> Optional[float]:
        """
        Get the latest value from a column.

        Args:
            df: DataFrame
            column: Column name

        Returns:
            Latest value or None
        """
        if df.is_empty() or column not in df.columns:
            return None

        value = df[column][-1]
        return float(value) if value is not None else None

    def get_values(self, df: pl.DataFrame, column: str, n: int = 10) -> list:
        """
        Get last N values from a column.

        Args:
            df: DataFrame
            column: Column name
            n: Number of values to get

        Returns:
            List of values
        """
        if df.is_empty() or column not in df.columns:
            return []

        values = df[column].tail(n).to_list()
        return [float(v) if v is not None else None for v in values]

    def cache_value(self, key: str, value: Any) -> None:
        """
        Cache a value.

        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = value

    def get_cached_value(self, key: str) -> Optional[Any]:
        """
        Get cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        return self._cache.get(key)

    def clear_cache(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
