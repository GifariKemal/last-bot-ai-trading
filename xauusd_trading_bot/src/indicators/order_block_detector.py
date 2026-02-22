"""
Order Block Detector
Identifies the last opposite-colored candle before a strong move.
Order blocks represent institutional buying/selling zones.
"""

from typing import List, Dict, Optional
import polars as pl

from .base_indicator import BaseIndicator
from ..core.constants import OrderBlockType


class OrderBlockDetector(BaseIndicator):
    """Detect Order Blocks in price action."""

    def __init__(
        self,
        lookback_bars: int = 50,
        min_body_percent: float = 60.0,
        strong_move_bars: int = 5,
        strong_move_percent: float = 1.0,
        **kwargs,
    ):
        """
        Initialize Order Block detector.

        Args:
            lookback_bars: How many bars to look back
            min_body_percent: Minimum candle body as % of total range
            strong_move_bars: Bars to check for strong move
            strong_move_percent: Minimum % move to qualify as strong
        """
        super().__init__(name="OrderBlock_Detector")
        self.lookback_bars = lookback_bars
        self.min_body_percent = min_body_percent
        self.strong_move_bars = strong_move_bars
        self.strong_move_percent = strong_move_percent

    def calculate(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """
        Calculate Order Blocks.

        Order Block identification:
        1. Find strong moves (significant price change)
        2. Identify the last opposite candle before the move
        3. That candle's range is the Order Block zone

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with Order Block columns
        """
        self.validate_dataframe(df, ["open", "high", "low", "close"])

        # Calculate price change over next N bars
        for i in range(1, self.strong_move_bars + 1):
            df = df.with_columns(
                [
                    (
                        (pl.col("close").shift(-i) - pl.col("close"))
                        / pl.col("close")
                        * 100
                    ).alias(f"forward_change_{i}")
                ]
            )

        # Detect strong bullish moves (significant upward price change)
        move_conditions = []
        for i in range(1, self.strong_move_bars + 1):
            move_conditions.append(
                pl.col(f"forward_change_{i}") >= self.strong_move_percent
            )

        df = df.with_columns(
            [
                pl.any_horizontal(move_conditions).alias("has_bullish_move"),
            ]
        )

        # Detect strong bearish moves
        move_conditions = []
        for i in range(1, self.strong_move_bars + 1):
            move_conditions.append(
                pl.col(f"forward_change_{i}") <= -self.strong_move_percent
            )

        df = df.with_columns(
            [
                pl.any_horizontal(move_conditions).alias("has_bearish_move"),
            ]
        )

        # Identify Order Blocks
        # Bullish OB: bearish candle (close < open) before bullish move
        df = df.with_columns(
            [
                (
                    pl.col("has_bullish_move")
                    & pl.col("is_bearish")
                    & (pl.col("body_percent") >= self.min_body_percent)
                ).alias("is_bullish_ob"),
            ]
        )

        # Bearish OB: bullish candle (close > open) before bearish move
        df = df.with_columns(
            [
                (
                    pl.col("has_bearish_move")
                    & pl.col("is_bullish")
                    & (pl.col("body_percent") >= self.min_body_percent)
                ).alias("is_bearish_ob"),
            ]
        )

        # Store OB zones - only at actual OB bars, then forward-fill
        df = df.with_columns(
            [
                # Bullish OB zone (only at actual bullish OB candles)
                pl.when(pl.col("is_bullish_ob")).then(pl.col("low")).otherwise(None)
                .fill_null(strategy="forward").alias("bullish_ob_low"),
                pl.when(pl.col("is_bullish_ob")).then(pl.col("high")).otherwise(None)
                .fill_null(strategy="forward").alias("bullish_ob_high"),
                # Bearish OB zone (only at actual bearish OB candles)
                pl.when(pl.col("is_bearish_ob")).then(pl.col("low")).otherwise(None)
                .fill_null(strategy="forward").alias("bearish_ob_low"),
                pl.when(pl.col("is_bearish_ob")).then(pl.col("high")).otherwise(None)
                .fill_null(strategy="forward").alias("bearish_ob_high"),
            ]
        )

        # Check if current price is in the most recent OB zone
        df = df.with_columns(
            [
                (
                    pl.col("bullish_ob_low").is_not_null()
                    & (pl.col("close") >= pl.col("bullish_ob_low"))
                    & (pl.col("close") <= pl.col("bullish_ob_high"))
                ).alias("in_bullish_ob_zone"),
                (
                    pl.col("bearish_ob_low").is_not_null()
                    & (pl.col("close") >= pl.col("bearish_ob_low"))
                    & (pl.col("close") <= pl.col("bearish_ob_high"))
                ).alias("in_bearish_ob_zone"),
            ]
        )

        return df

    def get_active_order_blocks(
        self, df: pl.DataFrame, ob_type: Optional[OrderBlockType] = None
    ) -> List[Dict]:
        """
        Get list of active (unmitigated) Order Blocks.

        Args:
            df: DataFrame with OB data
            ob_type: Filter by OB type (None = both)

        Returns:
            List of active OB dictionaries
        """
        active_obs = []

        # Get recent bars
        recent_df = df.tail(self.lookback_bars)

        for i, row in enumerate(recent_df.iter_rows(named=True)):
            # Check bullish OBs
            if (ob_type is None or ob_type == OrderBlockType.BULLISH) and row.get(
                "is_bullish_ob"
            ):
                # Check if OB is still valid (price hasn't closed below it)
                future_bars = recent_df[i + 1 :]
                if not future_bars.is_empty():
                    mitigated = (future_bars["close"] < row["bullish_ob_low"]).any()
                    if not mitigated:
                        active_obs.append(
                            {
                                "type": "BULLISH",
                                "time": row["time"],
                                "low": row["bullish_ob_low"],
                                "high": row["bullish_ob_high"],
                                "open": row["open"],
                                "close": row["close"],
                                "age_bars": len(recent_df) - i - 1,
                                "strength": row["body_percent"],
                            }
                        )

            # Check bearish OBs
            if (ob_type is None or ob_type == OrderBlockType.BEARISH) and row.get(
                "is_bearish_ob"
            ):
                # Check if OB is still valid (price hasn't closed above it)
                future_bars = recent_df[i + 1 :]
                if not future_bars.is_empty():
                    mitigated = (future_bars["close"] > row["bearish_ob_high"]).any()
                    if not mitigated:
                        active_obs.append(
                            {
                                "type": "BEARISH",
                                "time": row["time"],
                                "low": row["bearish_ob_low"],
                                "high": row["bearish_ob_high"],
                                "open": row["open"],
                                "close": row["close"],
                                "age_bars": len(recent_df) - i - 1,
                                "strength": row["body_percent"],
                            }
                        )

        return active_obs

    def is_price_at_order_block(
        self, df: pl.DataFrame, ob_type: OrderBlockType, current_price: float
    ) -> bool:
        """
        Check if current price is at an active Order Block.

        Args:
            df: DataFrame with OB data
            ob_type: Type of OB to check
            current_price: Current price

        Returns:
            True if price is at OB
        """
        active_obs = self.get_active_order_blocks(df, ob_type)

        for ob in active_obs:
            if ob["low"] <= current_price <= ob["high"]:
                return True

        return False

    def get_nearest_order_block(
        self, df: pl.DataFrame, ob_type: OrderBlockType, current_price: float
    ) -> Optional[Dict]:
        """
        Get the nearest active Order Block to current price.

        Args:
            df: DataFrame with OB data
            ob_type: Type of OB to find
            current_price: Current price

        Returns:
            Nearest OB dictionary or None
        """
        active_obs = self.get_active_order_blocks(df, ob_type)

        if not active_obs:
            return None

        # Find nearest by distance from current price to OB midpoint
        nearest = min(
            active_obs,
            key=lambda ob: abs(current_price - (ob["high"] + ob["low"]) / 2),
        )

        return nearest

    def get_strongest_order_block(
        self, df: pl.DataFrame, ob_type: OrderBlockType
    ) -> Optional[Dict]:
        """
        Get the strongest (highest body %) active Order Block.

        Args:
            df: DataFrame with OB data
            ob_type: Type of OB to find

        Returns:
            Strongest OB dictionary or None
        """
        active_obs = self.get_active_order_blocks(df, ob_type)

        if not active_obs:
            return None

        # Find strongest by body percentage
        strongest = max(active_obs, key=lambda ob: ob["strength"])

        return strongest

    def get_ob_summary(self, df: pl.DataFrame) -> Dict:
        """
        Get summary of Order Block status.

        Args:
            df: DataFrame with OB data

        Returns:
            Summary dictionary
        """
        latest_bar = df.tail(1).to_dicts()[0]

        active_bullish = self.get_active_order_blocks(df, OrderBlockType.BULLISH)
        active_bearish = self.get_active_order_blocks(df, OrderBlockType.BEARISH)

        return {
            "latest_bar": {
                "is_bullish_ob": latest_bar.get("is_bullish_ob", False),
                "is_bearish_ob": latest_bar.get("is_bearish_ob", False),
                "in_bullish_zone": latest_bar.get("in_bullish_ob_zone", False),
                "in_bearish_zone": latest_bar.get("in_bearish_ob_zone", False),
            },
            "active_obs": {
                "bullish_count": len(active_bullish),
                "bearish_count": len(active_bearish),
                "bullish": active_bullish[:5],  # Top 5 most recent
                "bearish": active_bearish[:5],
            },
            "strongest": {
                "bullish": self.get_strongest_order_block(df, OrderBlockType.BULLISH),
                "bearish": self.get_strongest_order_block(df, OrderBlockType.BEARISH),
            },
        }
