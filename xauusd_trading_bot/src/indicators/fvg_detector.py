"""
Fair Value Gap (FVG) Detector
Identifies imbalances in the market where price moved too quickly,
leaving gaps that often get filled later.
"""

from typing import List, Dict, Optional
import polars as pl

from .base_indicator import BaseIndicator
from ..core.constants import FVGType


class FVGDetector(BaseIndicator):
    """Detect Fair Value Gaps (imbalances) in price action."""

    def __init__(self, min_gap_pips: float = 5.0, max_age_bars: int = 100, **kwargs):
        """
        Initialize FVG detector.

        Args:
            min_gap_pips: Minimum gap size in pips to consider
            max_age_bars: Maximum age of FVG to track
        """
        super().__init__(name="FVG_Detector")
        self.min_gap_pips = min_gap_pips
        self.max_age_bars = max_age_bars

    def calculate(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """
        Calculate Fair Value Gaps.

        FVG occurs when:
        - Bullish FVG: high[i-2] < low[i] (gap between candle i-2 high and candle i low)
        - Bearish FVG: low[i-2] > high[i] (gap between candle i-2 low and candle i high)

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with FVG columns added
        """
        self.validate_dataframe(df, ["open", "high", "low", "close"])

        # Detect bullish FVGs (gap up)
        df = df.with_columns(
            [
                # Bullish FVG: high 2 bars ago < low current bar
                (pl.col("high").shift(2) < pl.col("low")).alias("is_bullish_fvg"),
                # Gap size for bullish FVG
                (pl.col("low") - pl.col("high").shift(2)).alias("bullish_fvg_gap"),
                # FVG zone boundaries
                pl.col("high").shift(2).alias("bullish_fvg_low"),
                pl.col("low").alias("bullish_fvg_high"),
            ]
        )

        # Detect bearish FVGs (gap down)
        df = df.with_columns(
            [
                # Bearish FVG: low 2 bars ago > high current bar
                (pl.col("low").shift(2) > pl.col("high")).alias("is_bearish_fvg"),
                # Gap size for bearish FVG
                (pl.col("low").shift(2) - pl.col("high")).alias("bearish_fvg_gap"),
                # FVG zone boundaries
                pl.col("high").alias("bearish_fvg_low"),
                pl.col("low").shift(2).alias("bearish_fvg_high"),
            ]
        )

        # Filter by minimum gap size
        df = df.with_columns(
            [
                (
                    pl.col("is_bullish_fvg")
                    & (pl.col("bullish_fvg_gap") >= self.min_gap_pips)
                ).alias("is_valid_bullish_fvg"),
                (
                    pl.col("is_bearish_fvg")
                    & (pl.col("bearish_fvg_gap") >= self.min_gap_pips)
                ).alias("is_valid_bearish_fvg"),
            ]
        )

        # Check if price is inside FVG zone
        df = df.with_columns(
            [
                # Price inside bullish FVG zone
                (
                    (pl.col("close") >= pl.col("bullish_fvg_low"))
                    & (pl.col("close") <= pl.col("bullish_fvg_high"))
                ).alias("in_bullish_fvg_zone"),
                # Price inside bearish FVG zone
                (
                    (pl.col("close") >= pl.col("bearish_fvg_low"))
                    & (pl.col("close") <= pl.col("bearish_fvg_high"))
                ).alias("in_bearish_fvg_zone"),
            ]
        )

        return df

    def get_active_fvgs(
        self, df: pl.DataFrame, fvg_type: Optional[FVGType] = None
    ) -> List[Dict]:
        """
        Get list of active (unfilled) FVGs.

        Args:
            df: DataFrame with FVG data
            fvg_type: Filter by FVG type (None = both)

        Returns:
            List of active FVG dictionaries
        """
        active_fvgs = []

        # Get only recent bars
        recent_df = df.tail(self.max_age_bars)

        for i, row in enumerate(recent_df.iter_rows(named=True)):
            # Check bullish FVGs
            if (fvg_type is None or fvg_type == FVGType.BULLISH) and row.get(
                "is_valid_bullish_fvg"
            ):
                # Check if FVG still unfilled (price hasn't closed below it)
                future_bars = recent_df[i + 1 :]
                if not future_bars.is_empty():
                    filled = (future_bars["close"] < row["bullish_fvg_low"]).any()
                    if not filled:
                        active_fvgs.append(
                            {
                                "type": "BULLISH",
                                "time": row["time"],
                                "low": row["bullish_fvg_low"],
                                "high": row["bullish_fvg_high"],
                                "gap_size": row["bullish_fvg_gap"],
                                "age_bars": len(recent_df) - i - 1,
                            }
                        )

            # Check bearish FVGs
            if (fvg_type is None or fvg_type == FVGType.BEARISH) and row.get(
                "is_valid_bearish_fvg"
            ):
                # Check if FVG still unfilled (price hasn't closed above it)
                future_bars = recent_df[i + 1 :]
                if not future_bars.is_empty():
                    filled = (future_bars["close"] > row["bearish_fvg_high"]).any()
                    if not filled:
                        active_fvgs.append(
                            {
                                "type": "BEARISH",
                                "time": row["time"],
                                "low": row["bearish_fvg_low"],
                                "high": row["bearish_fvg_high"],
                                "gap_size": row["bearish_fvg_gap"],
                                "age_bars": len(recent_df) - i - 1,
                            }
                        )

        return active_fvgs

    def is_price_in_fvg(
        self, df: pl.DataFrame, fvg_type: FVGType, current_price: float
    ) -> bool:
        """
        Check if current price is within an active FVG zone.

        Args:
            df: DataFrame with FVG data
            fvg_type: Type of FVG to check
            current_price: Current price to check

        Returns:
            True if price is in FVG zone
        """
        active_fvgs = self.get_active_fvgs(df, fvg_type)

        for fvg in active_fvgs:
            if fvg["low"] <= current_price <= fvg["high"]:
                return True

        return False

    def get_nearest_fvg(
        self, df: pl.DataFrame, fvg_type: FVGType, current_price: float
    ) -> Optional[Dict]:
        """
        Get the nearest active FVG to current price.

        Args:
            df: DataFrame with FVG data
            fvg_type: Type of FVG to find
            current_price: Current price

        Returns:
            Nearest FVG dictionary or None
        """
        active_fvgs = self.get_active_fvgs(df, fvg_type)

        if not active_fvgs:
            return None

        # Find nearest by distance from current price to FVG midpoint
        nearest = min(
            active_fvgs,
            key=lambda fvg: abs(current_price - (fvg["high"] + fvg["low"]) / 2),
        )

        return nearest

    def get_fvg_summary(self, df: pl.DataFrame) -> Dict:
        """
        Get summary of FVG status.

        Args:
            df: DataFrame with FVG data

        Returns:
            Summary dictionary
        """
        latest_bar = df.tail(1).to_dicts()[0]

        active_bullish = self.get_active_fvgs(df, FVGType.BULLISH)
        active_bearish = self.get_active_fvgs(df, FVGType.BEARISH)

        return {
            "latest_bar": {
                "is_bullish_fvg": latest_bar.get("is_valid_bullish_fvg", False),
                "is_bearish_fvg": latest_bar.get("is_valid_bearish_fvg", False),
                "in_bullish_zone": latest_bar.get("in_bullish_fvg_zone", False),
                "in_bearish_zone": latest_bar.get("in_bearish_fvg_zone", False),
            },
            "active_fvgs": {
                "bullish_count": len(active_bullish),
                "bearish_count": len(active_bearish),
                "bullish": active_bullish[:5],  # Top 5 most recent
                "bearish": active_bearish[:5],
            },
        }
