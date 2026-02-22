"""
Structure Detector
Identifies market structure: swing highs/lows, BOS (Break of Structure),
and CHoCH (Change of Character).
"""

from typing import List, Dict, Optional
import polars as pl

from .base_indicator import BaseIndicator
from ..core.constants import StructureType, TrendDirection


class StructureDetector(BaseIndicator):
    """Detect market structure patterns (BOS, CHoCH, swing points)."""

    def __init__(
        self, swing_lookback: int = 10, bos_confirmation_bars: int = 2, **kwargs
    ):
        """
        Initialize Structure detector.

        Args:
            swing_lookback: Bars to each side for swing detection
            bos_confirmation_bars: Bars to confirm structure break
        """
        super().__init__(name="Structure_Detector")
        self.swing_lookback = swing_lookback
        self.bos_confirmation_bars = bos_confirmation_bars

    def calculate(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """
        Calculate market structure.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with structure columns
        """
        self.validate_dataframe(df, ["open", "high", "low", "close"])

        # Detect swing highs and lows
        df = self._detect_swing_points(df)

        # Detect structure breaks
        df = self._detect_structure_breaks(df)

        # Determine current trend
        df = self._determine_trend(df)

        return df

    def _detect_swing_points(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect swing highs and swing lows.

        Swing High: High higher than N bars on each side
        Swing Low: Low lower than N bars on each side
        """
        n = self.swing_lookback

        # Build conditions for swing high
        high_conditions = []
        for i in range(1, n + 1):
            high_conditions.append(pl.col("high") > pl.col("high").shift(i))
            high_conditions.append(pl.col("high") > pl.col("high").shift(-i))

        # Build conditions for swing low
        low_conditions = []
        for i in range(1, n + 1):
            low_conditions.append(pl.col("low") < pl.col("low").shift(i))
            low_conditions.append(pl.col("low") < pl.col("low").shift(-i))

        df = df.with_columns(
            [
                pl.all_horizontal(high_conditions).alias("is_swing_high"),
                pl.all_horizontal(low_conditions).alias("is_swing_low"),
            ]
        )

        # Store swing levels
        df = df.with_columns(
            [
                pl.when(pl.col("is_swing_high"))
                .then(pl.col("high"))
                .otherwise(None)
                .alias("swing_high_level"),
                pl.when(pl.col("is_swing_low"))
                .then(pl.col("low"))
                .otherwise(None)
                .alias("swing_low_level"),
            ]
        )

        return df

    def _detect_structure_breaks(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect Break of Structure (BOS) and Change of Character (CHoCH).

        BOS: Price breaks recent swing in same trend direction (continuation)
        CHoCH: Price breaks recent swing in opposite direction (reversal)
        """
        # Get previous swing levels using forward fill
        df = df.with_columns(
            [
                pl.col("swing_high_level").fill_null(strategy="forward").alias("prev_swing_high"),
                pl.col("swing_low_level").fill_null(strategy="forward").alias("prev_swing_low"),
            ]
        )

        # Detect breaks
        df = df.with_columns(
            [
                # Bullish BOS: close above recent swing high while in uptrend
                (pl.col("close") > pl.col("prev_swing_high")).alias("broke_swing_high"),
                # Bearish BOS: close below recent swing low while in downtrend
                (pl.col("close") < pl.col("prev_swing_low")).alias("broke_swing_low"),
            ]
        )

        # Determine if break is BOS or CHoCH based on previous trend
        # We'll compute this in a rolling manner

        return df

    def _determine_trend(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Determine current trend based on swing highs and lows.

        Uptrend: Higher highs and higher lows
        Downtrend: Lower highs and lower lows
        """
        # Step 1: Forward-fill swing levels to get the LATEST swing at each bar
        df = df.with_columns(
            [
                pl.col("swing_high_level").fill_null(strategy="forward").alias("last_swing_high"),
                pl.col("swing_low_level").fill_null(strategy="forward").alias("last_swing_low"),
            ]
        )

        # Step 2: At each swing point, record what the PREVIOUS swing level was
        # At a swing high bar, the forward-filled value from the bar before is the previous swing high
        # At a swing low bar, the forward-filled value from the bar before is the previous swing low
        # Then forward-fill these to persist until the next swing updates them
        df = df.with_columns(
            [
                pl.when(pl.col("is_swing_high"))
                .then(pl.col("last_swing_high").shift(1))
                .otherwise(None)
                .fill_null(strategy="forward")
                .alias("prev_swing_high_2"),
                pl.when(pl.col("is_swing_low"))
                .then(pl.col("last_swing_low").shift(1))
                .otherwise(None)
                .fill_null(strategy="forward")
                .alias("prev_swing_low_2"),
            ]
        )

        # Step 3: Determine trend - use OR logic (HH or HL = bullish bias)
        # This is more lenient than requiring both, which is too strict for M15
        df = df.with_columns(
            [
                pl.when(
                    # Strong uptrend: HH and HL
                    (pl.col("last_swing_high") > pl.col("prev_swing_high_2"))
                    & (pl.col("last_swing_low") > pl.col("prev_swing_low_2"))
                )
                .then(pl.lit("BULLISH"))
                .when(
                    # Strong downtrend: LH and LL
                    (pl.col("last_swing_high") < pl.col("prev_swing_high_2"))
                    & (pl.col("last_swing_low") < pl.col("prev_swing_low_2"))
                )
                .then(pl.lit("BEARISH"))
                .when(
                    # Weak uptrend: HH only (price making new highs)
                    (pl.col("last_swing_high") > pl.col("prev_swing_high_2"))
                )
                .then(pl.lit("BULLISH"))
                .when(
                    # Weak downtrend: LL only (price making new lows)
                    (pl.col("last_swing_low") < pl.col("prev_swing_low_2"))
                )
                .then(pl.lit("BEARISH"))
                .otherwise(pl.lit("NEUTRAL"))
                .alias("market_structure_trend")
            ]
        )

        # Step 4: Classify structure breaks as BOS or CHoCH
        df = df.with_columns(
            [
                # Bullish BOS: break high in uptrend (continuation)
                (
                    pl.col("broke_swing_high")
                    & (pl.col("market_structure_trend") == "BULLISH")
                ).alias("is_bullish_bos"),
                # Bearish BOS: break low in downtrend (continuation)
                (
                    pl.col("broke_swing_low")
                    & (pl.col("market_structure_trend") == "BEARISH")
                ).alias("is_bearish_bos"),
                # Bullish CHoCH: break high in downtrend (reversal to bullish)
                (
                    pl.col("broke_swing_high")
                    & (pl.col("market_structure_trend") == "BEARISH")
                ).alias("is_bullish_choch"),
                # Bearish CHoCH: break low in uptrend (reversal to bearish)
                (
                    pl.col("broke_swing_low")
                    & (pl.col("market_structure_trend") == "BULLISH")
                ).alias("is_bearish_choch"),
            ]
        )

        return df

    def get_swing_points(self, df: pl.DataFrame, n: int = 10) -> Dict:
        """
        Get recent swing highs and lows.

        Args:
            df: DataFrame with structure data
            n: Number of recent swings to get

        Returns:
            Dictionary with swing points
        """
        recent_df = df.tail(100)  # Look at recent bars

        swing_highs = [
            {"time": row["time"], "level": row["high"], "type": "SWING_HIGH"}
            for row in recent_df.iter_rows(named=True)
            if row.get("is_swing_high", False)
        ]

        swing_lows = [
            {"time": row["time"], "level": row["low"], "type": "SWING_LOW"}
            for row in recent_df.iter_rows(named=True)
            if row.get("is_swing_low", False)
        ]

        return {
            "swing_highs": swing_highs[-n:],  # Last N swings
            "swing_lows": swing_lows[-n:],
        }

    def get_recent_structure_breaks(
        self, df: pl.DataFrame, bars: int = 20
    ) -> Dict:
        """
        Get recent structure breaks (BOS and CHoCH).

        Args:
            df: DataFrame with structure data
            bars: Number of recent bars to check

        Returns:
            Dictionary with recent breaks
        """
        recent_df = df.tail(bars)

        bullish_bos = [
            {"time": row["time"], "level": row["close"], "type": "BULLISH_BOS"}
            for row in recent_df.iter_rows(named=True)
            if row.get("is_bullish_bos", False)
        ]

        bearish_bos = [
            {"time": row["time"], "level": row["close"], "type": "BEARISH_BOS"}
            for row in recent_df.iter_rows(named=True)
            if row.get("is_bearish_bos", False)
        ]

        bullish_choch = [
            {"time": row["time"], "level": row["close"], "type": "BULLISH_CHOCH"}
            for row in recent_df.iter_rows(named=True)
            if row.get("is_bullish_choch", False)
        ]

        bearish_choch = [
            {"time": row["time"], "level": row["close"], "type": "BEARISH_CHOCH"}
            for row in recent_df.iter_rows(named=True)
            if row.get("is_bearish_choch", False)
        ]

        return {
            "bullish_bos": bullish_bos,
            "bearish_bos": bearish_bos,
            "bullish_choch": bullish_choch,
            "bearish_choch": bearish_choch,
        }

    def get_current_trend(self, df: pl.DataFrame) -> TrendDirection:
        """
        Get current market structure trend.

        Args:
            df: DataFrame with structure data

        Returns:
            Current trend direction
        """
        latest_bar = df.tail(1).to_dicts()[0]
        trend_str = latest_bar.get("market_structure_trend", "NEUTRAL")

        if trend_str == "BULLISH":
            return TrendDirection.BULLISH
        elif trend_str == "BEARISH":
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL

    def has_recent_choch(
        self, df: pl.DataFrame, direction: TrendDirection, bars: int = 10
    ) -> bool:
        """
        Check if there was a recent CHoCH in specified direction.

        Args:
            df: DataFrame with structure data
            direction: Direction to check
            bars: Recent bars to check

        Returns:
            True if recent CHoCH occurred
        """
        breaks = self.get_recent_structure_breaks(df, bars)

        if direction == TrendDirection.BULLISH:
            return len(breaks["bullish_choch"]) > 0
        elif direction == TrendDirection.BEARISH:
            return len(breaks["bearish_choch"]) > 0

        return False

    def has_recent_bos(
        self, df: pl.DataFrame, direction: TrendDirection, bars: int = 10
    ) -> bool:
        """
        Check if there was a recent BOS in specified direction.

        Args:
            df: DataFrame with structure data
            direction: Direction to check
            bars: Recent bars to check

        Returns:
            True if recent BOS occurred
        """
        breaks = self.get_recent_structure_breaks(df, bars)

        if direction == TrendDirection.BULLISH:
            return len(breaks["bullish_bos"]) > 0
        elif direction == TrendDirection.BEARISH:
            return len(breaks["bearish_bos"]) > 0

        return False

    def get_structure_summary(self, df: pl.DataFrame) -> Dict:
        """
        Get summary of market structure.

        Args:
            df: DataFrame with structure data

        Returns:
            Summary dictionary
        """
        latest_bar = df.tail(1).to_dicts()[0]

        swing_points = self.get_swing_points(df, 5)
        recent_breaks = self.get_recent_structure_breaks(df, 20)
        current_trend = self.get_current_trend(df)

        return {
            "latest_bar": {
                "is_swing_high": latest_bar.get("is_swing_high", False),
                "is_swing_low": latest_bar.get("is_swing_low", False),
                "is_bullish_bos": latest_bar.get("is_bullish_bos", False),
                "is_bearish_bos": latest_bar.get("is_bearish_bos", False),
                "is_bullish_choch": latest_bar.get("is_bullish_choch", False),
                "is_bearish_choch": latest_bar.get("is_bearish_choch", False),
            },
            "current_trend": current_trend.value,
            "swing_points": swing_points,
            "recent_breaks": recent_breaks,
            "has_bullish_choch": len(recent_breaks["bullish_choch"]) > 0,
            "has_bearish_choch": len(recent_breaks["bearish_choch"]) > 0,
            "has_bullish_bos": len(recent_breaks["bullish_bos"]) > 0,
            "has_bearish_bos": len(recent_breaks["bearish_bos"]) > 0,
        }
