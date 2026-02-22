"""
Technical indicators implementation.
Includes: ATR, EMA, RSI, MACD, Bollinger Bands
"""

from typing import Dict, List, Optional

import polars as pl

from .base_indicator import BaseIndicator


class ATR(BaseIndicator):
    """Average True Range indicator for volatility measurement."""

    def __init__(self, period: int = 14, smoothing: str = "ema"):
        """
        Initialize ATR indicator.

        Args:
            period: ATR period
            smoothing: Smoothing method ('ema', 'sma', 'wilder')
        """
        super().__init__(name=f"ATR_{period}")
        self.period = period
        self.smoothing = smoothing

    def calculate(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """
        Calculate ATR.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with ATR column
        """
        self.validate_dataframe(df, ["high", "low", "close"])

        # Calculate True Range
        df = df.with_columns(
            [
                # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
                pl.max_horizontal(
                    pl.col("high") - pl.col("low"),
                    (pl.col("high") - pl.col("close").shift(1)).abs(),
                    (pl.col("low") - pl.col("close").shift(1)).abs(),
                ).alias("true_range")
            ]
        )

        # Calculate ATR based on smoothing method
        if self.smoothing == "ema":
            df = df.with_columns(
                [
                    pl.col("true_range")
                    .ewm_mean(span=self.period, adjust=False)
                    .alias(f"atr_{self.period}")
                ]
            )
        elif self.smoothing == "sma":
            df = df.with_columns(
                [
                    pl.col("true_range")
                    .rolling_mean(window_size=self.period)
                    .alias(f"atr_{self.period}")
                ]
            )
        elif self.smoothing == "wilder":
            # Wilder's smoothing (similar to EMA with alpha = 1/period)
            df = df.with_columns(
                [
                    pl.col("true_range")
                    .ewm_mean(alpha=1 / self.period, adjust=False)
                    .alias(f"atr_{self.period}")
                ]
            )

        # Drop temporary column
        df = df.drop("true_range")

        return df


class EMA(BaseIndicator):
    """Exponential Moving Average indicator."""

    def __init__(self, periods: List[int] = [20, 50, 100, 200]):
        """
        Initialize EMA indicator.

        Args:
            periods: List of EMA periods to calculate
        """
        super().__init__(name="EMA")
        self.periods = periods

    def calculate(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """
        Calculate EMAs for all periods.

        Args:
            df: DataFrame with close prices

        Returns:
            DataFrame with EMA columns
        """
        self.validate_dataframe(df, ["close"])

        for period in self.periods:
            df = df.with_columns(
                [
                    pl.col("close")
                    .ewm_mean(span=period, adjust=False)
                    .alias(f"ema_{period}")
                ]
            )

        return df


class RSI(BaseIndicator):
    """Relative Strength Index indicator."""

    def __init__(self, period: int = 14):
        """
        Initialize RSI indicator.

        Args:
            period: RSI period
        """
        super().__init__(name=f"RSI_{period}")
        self.period = period

    def calculate(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """
        Calculate RSI.

        Args:
            df: DataFrame with close prices

        Returns:
            DataFrame with RSI column
        """
        self.validate_dataframe(df, ["close"])

        # Calculate price changes
        df = df.with_columns(
            [(pl.col("close") - pl.col("close").shift(1)).alias("price_change")]
        )

        # Separate gains and losses
        df = df.with_columns(
            [
                pl.when(pl.col("price_change") > 0)
                .then(pl.col("price_change"))
                .otherwise(0)
                .alias("gain"),
                pl.when(pl.col("price_change") < 0)
                .then(pl.col("price_change").abs())
                .otherwise(0)
                .alias("loss"),
            ]
        )

        # Calculate average gains and losses using EMA
        df = df.with_columns(
            [
                pl.col("gain")
                .ewm_mean(span=self.period, adjust=False)
                .alias("avg_gain"),
                pl.col("loss")
                .ewm_mean(span=self.period, adjust=False)
                .alias("avg_loss"),
            ]
        )

        # Calculate RS and RSI
        df = df.with_columns(
            [
                (pl.col("avg_gain") / pl.col("avg_loss")).alias("rs"),
            ]
        )

        df = df.with_columns(
            [
                (100 - (100 / (1 + pl.col("rs")))).alias(f"rsi_{self.period}"),
            ]
        )

        # Drop temporary columns
        df = df.drop(["price_change", "gain", "loss", "avg_gain", "avg_loss", "rs"])

        return df


class MACD(BaseIndicator):
    """Moving Average Convergence Divergence indicator."""

    def __init__(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ):
        """
        Initialize MACD indicator.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
        """
        super().__init__(name="MACD")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """
        Calculate MACD.

        Args:
            df: DataFrame with close prices

        Returns:
            DataFrame with MACD, signal, and histogram columns
        """
        self.validate_dataframe(df, ["close"])

        # Calculate fast and slow EMAs
        df = df.with_columns(
            [
                pl.col("close")
                .ewm_mean(span=self.fast_period, adjust=False)
                .alias("ema_fast"),
                pl.col("close")
                .ewm_mean(span=self.slow_period, adjust=False)
                .alias("ema_slow"),
            ]
        )

        # Calculate MACD line
        df = df.with_columns(
            [(pl.col("ema_fast") - pl.col("ema_slow")).alias("macd_line")]
        )

        # Calculate signal line
        df = df.with_columns(
            [
                pl.col("macd_line")
                .ewm_mean(span=self.signal_period, adjust=False)
                .alias("macd_signal")
            ]
        )

        # Calculate histogram
        df = df.with_columns(
            [(pl.col("macd_line") - pl.col("macd_signal")).alias("macd_histogram")]
        )

        # Drop temporary columns
        df = df.drop(["ema_fast", "ema_slow"])

        return df


class BollingerBands(BaseIndicator):
    """Bollinger Bands indicator."""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands indicator.

        Args:
            period: Moving average period
            std_dev: Number of standard deviations
        """
        super().__init__(name=f"BB_{period}")
        self.period = period
        self.std_dev = std_dev

    def calculate(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """
        Calculate Bollinger Bands.

        Args:
            df: DataFrame with close prices

        Returns:
            DataFrame with BB upper, middle, and lower bands
        """
        self.validate_dataframe(df, ["close"])

        # Calculate middle band (SMA)
        df = df.with_columns(
            [pl.col("close").rolling_mean(window_size=self.period).alias("bb_middle")]
        )

        # Calculate standard deviation
        df = df.with_columns(
            [pl.col("close").rolling_std(window_size=self.period).alias("bb_std")]
        )

        # Calculate upper and lower bands
        df = df.with_columns(
            [
                (pl.col("bb_middle") + (pl.col("bb_std") * self.std_dev)).alias(
                    "bb_upper"
                ),
                (pl.col("bb_middle") - (pl.col("bb_std") * self.std_dev)).alias(
                    "bb_lower"
                ),
            ]
        )

        # Calculate bandwidth (for volatility measurement)
        df = df.with_columns(
            [
                (
                    (pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_middle")
                    * 100
                ).alias("bb_bandwidth")
            ]
        )

        # Calculate %B (position within bands)
        df = df.with_columns(
            [
                (
                    (pl.col("close") - pl.col("bb_lower"))
                    / (pl.col("bb_upper") - pl.col("bb_lower"))
                    * 100
                ).alias("bb_percent")
            ]
        )

        # Drop temporary column
        df = df.drop("bb_std")

        return df


class TechnicalIndicators:
    """
    Wrapper class to calculate all technical indicators at once.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize technical indicators.

        Args:
            config: Configuration dictionary
        """
        self.logger = get_logger()

        # Default configuration
        if config is None:
            config = {
                "atr": {"period": 14, "smoothing": "ema"},
                "ema": {"periods": [20, 50, 100, 200]},
                "rsi": {"period": 14},
                "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "bollinger_bands": {"period": 20, "std_dev": 2.0},
            }

        self.config = config

        # Initialize indicators
        self.atr = ATR(**config.get("atr", {}))
        self.ema = EMA(**config.get("ema", {}))
        self.rsi = RSI(**config.get("rsi", {}))
        self.macd = MACD(**config.get("macd", {}))
        self.bb = BollingerBands(**config.get("bollinger_bands", {}))

    def calculate_all(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate all technical indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all indicator columns
        """
        try:
            # Calculate each indicator
            df = self.atr.calculate(df)
            df = self.ema.calculate(df)
            df = self.rsi.calculate(df)
            df = self.macd.calculate(df)
            df = self.bb.calculate(df)

            self.logger.debug("All technical indicators calculated successfully")
            return df

        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            raise

    def get_indicator_summary(self, df: pl.DataFrame) -> Dict:
        """
        Get summary of current indicator values.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Dictionary with indicator values
        """
        if df.is_empty():
            return {}

        latest = df.tail(1).to_dicts()[0]

        summary = {
            "atr": latest.get(f"atr_{self.atr.period}"),
            "ema": {
                period: latest.get(f"ema_{period}") for period in self.ema.periods
            },
            "rsi": latest.get(f"rsi_{self.rsi.period}"),
            "macd": {
                "line": latest.get("macd_line"),
                "signal": latest.get("macd_signal"),
                "histogram": latest.get("macd_histogram"),
            },
            "bollinger_bands": {
                "upper": latest.get("bb_upper"),
                "middle": latest.get("bb_middle"),
                "lower": latest.get("bb_lower"),
                "bandwidth": latest.get("bb_bandwidth"),
                "percent": latest.get("bb_percent"),
            },
        }

        return summary


def get_logger():
    """Import logger to avoid circular import."""
    from ..bot_logger import get_logger as _get_logger

    return _get_logger()
