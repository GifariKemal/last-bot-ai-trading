"""
Liquidity Detector
Identifies equal highs/lows and liquidity sweeps (stop hunts).
Liquidity zones are where retail stops are typically placed.
"""

from typing import List, Dict, Optional, Tuple
import polars as pl

from .base_indicator import BaseIndicator
from ..core.constants import LiquidityType


class LiquidityDetector(BaseIndicator):
    """Detect liquidity zones and sweeps."""

    def __init__(
        self,
        lookback_bars: int = 30,
        equal_level_tolerance_pips: float = 3.0,
        min_touches: int = 2,
        sweep_confirmation_bars: int = 20,
        **kwargs,
    ):
        """
        Initialize Liquidity detector.

        Args:
            lookback_bars: How many bars to look back
            equal_level_tolerance_pips: Tolerance for equal levels (pips)
            min_touches: Minimum touches to form liquidity level
            sweep_confirmation_bars: Bars to confirm sweep
        """
        super().__init__(name="Liquidity_Detector")
        self.lookback_bars = lookback_bars
        self.equal_level_tolerance = equal_level_tolerance_pips
        self.min_touches = min_touches
        self.sweep_confirmation_bars = sweep_confirmation_bars

    def calculate(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """
        Calculate liquidity levels and sweeps.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with liquidity columns
        """
        self.validate_dataframe(df, ["open", "high", "low", "close"])

        # Find local highs and lows (swing points)
        df = df.with_columns(
            [
                # Local high: higher than N bars on each side
                (
                    (pl.col("high") > pl.col("high").shift(1))
                    & (pl.col("high") > pl.col("high").shift(2))
                    & (pl.col("high") > pl.col("high").shift(-1))
                    & (pl.col("high") > pl.col("high").shift(-2))
                ).alias("is_local_high"),
                # Local low: lower than N bars on each side
                (
                    (pl.col("low") < pl.col("low").shift(1))
                    & (pl.col("low") < pl.col("low").shift(2))
                    & (pl.col("low") < pl.col("low").shift(-1))
                    & (pl.col("low") < pl.col("low").shift(-2))
                ).alias("is_local_low"),
            ]
        )

        # Detect liquidity sweeps
        # Sweep occurs when price breaks above/below level then reverses quickly
        df = df.with_columns(
            [
                # High liquidity sweep: broke above recent high then closed lower
                (
                    (pl.col("high") > pl.col("high").shift(1).rolling_max(5))
                    & (pl.col("close") < pl.col("open"))
                ).alias("swept_high_liquidity"),
                # Low liquidity sweep: broke below recent low then closed higher
                (
                    (pl.col("low") < pl.col("low").shift(1).rolling_min(5))
                    & (pl.col("close") > pl.col("open"))
                ).alias("swept_low_liquidity"),
            ]
        )

        return df

    def find_equal_highs(self, df: pl.DataFrame) -> List[Dict]:
        """
        Find equal highs (liquidity pools above market).

        Args:
            df: DataFrame with OHLC data

        Returns:
            List of equal high zones
        """
        recent_df = df.tail(self.lookback_bars)
        equal_highs = []

        # Get all local highs
        local_highs = [
            (i, row["high"], row["time"])
            for i, row in enumerate(recent_df.iter_rows(named=True))
            if row.get("is_local_high", False)
        ]

        # Group highs by similar levels
        for i, (idx1, high1, time1) in enumerate(local_highs):
            matches = [(idx1, high1, time1)]

            for idx2, high2, time2 in local_highs[i + 1 :]:
                if abs(high1 - high2) <= self.equal_level_tolerance:
                    matches.append((idx2, high2, time2))

            # If enough touches at same level, it's a liquidity zone
            if len(matches) >= self.min_touches:
                avg_level = sum(h for _, h, _ in matches) / len(matches)
                equal_highs.append(
                    {
                        "type": "EQUAL_HIGH",
                        "level": avg_level,
                        "touches": len(matches),
                        "first_touch": matches[0][2],
                        "last_touch": matches[-1][2],
                        "strength": len(matches),  # More touches = stronger
                    }
                )

        # Remove duplicates and sort by level
        unique_levels = {}
        for eh in equal_highs:
            level_key = round(eh["level"], 1)
            if (
                level_key not in unique_levels
                or eh["touches"] > unique_levels[level_key]["touches"]
            ):
                unique_levels[level_key] = eh

        return sorted(unique_levels.values(), key=lambda x: x["level"], reverse=True)

    def find_equal_lows(self, df: pl.DataFrame) -> List[Dict]:
        """
        Find equal lows (liquidity pools below market).

        Args:
            df: DataFrame with OHLC data

        Returns:
            List of equal low zones
        """
        recent_df = df.tail(self.lookback_bars)
        equal_lows = []

        # Get all local lows
        local_lows = [
            (i, row["low"], row["time"])
            for i, row in enumerate(recent_df.iter_rows(named=True))
            if row.get("is_local_low", False)
        ]

        # Group lows by similar levels
        for i, (idx1, low1, time1) in enumerate(local_lows):
            matches = [(idx1, low1, time1)]

            for idx2, low2, time2 in local_lows[i + 1 :]:
                if abs(low1 - low2) <= self.equal_level_tolerance:
                    matches.append((idx2, low2, time2))

            # If enough touches at same level, it's a liquidity zone
            if len(matches) >= self.min_touches:
                avg_level = sum(l for _, l, _ in matches) / len(matches)
                equal_lows.append(
                    {
                        "type": "EQUAL_LOW",
                        "level": avg_level,
                        "touches": len(matches),
                        "first_touch": matches[0][2],
                        "last_touch": matches[-1][2],
                        "strength": len(matches),
                    }
                )

        # Remove duplicates and sort by level
        unique_levels = {}
        for el in equal_lows:
            level_key = round(el["level"], 1)
            if (
                level_key not in unique_levels
                or el["touches"] > unique_levels[level_key]["touches"]
            ):
                unique_levels[level_key] = el

        return sorted(unique_levels.values(), key=lambda x: x["level"])

    def detect_recent_sweeps(
        self, df: pl.DataFrame, bars: int = 10
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect recent liquidity sweeps.

        Args:
            df: DataFrame with liquidity data
            bars: Number of recent bars to check

        Returns:
            Tuple of (high_sweeps, low_sweeps)
        """
        recent_df = df.tail(bars)

        high_sweeps = []
        low_sweeps = []

        for row in recent_df.iter_rows(named=True):
            if row.get("swept_high_liquidity"):
                high_sweeps.append(
                    {
                        "type": "HIGH_SWEEP",
                        "time": row["time"],
                        "high": row["high"],
                        "close": row["close"],
                        "reversal_size": row["high"] - row["close"],
                    }
                )

            if row.get("swept_low_liquidity"):
                low_sweeps.append(
                    {
                        "type": "LOW_SWEEP",
                        "time": row["time"],
                        "low": row["low"],
                        "close": row["close"],
                        "reversal_size": row["close"] - row["low"],
                    }
                )

        return high_sweeps, low_sweeps

    def is_liquidity_sweep_active(
        self, df: pl.DataFrame, sweep_type: LiquidityType
    ) -> bool:
        """
        Check if there was a recent liquidity sweep.

        Args:
            df: DataFrame with liquidity data
            sweep_type: Type of sweep to check

        Returns:
            True if recent sweep occurred
        """
        high_sweeps, low_sweeps = self.detect_recent_sweeps(
            df, self.sweep_confirmation_bars
        )

        if sweep_type == LiquidityType.HIGH:
            return len(high_sweeps) > 0
        elif sweep_type == LiquidityType.LOW:
            return len(low_sweeps) > 0

        return False

    def get_nearest_liquidity_level(
        self, df: pl.DataFrame, current_price: float, direction: str = "both"
    ) -> Optional[Dict]:
        """
        Get nearest liquidity level to current price.

        Args:
            df: DataFrame with liquidity data
            current_price: Current price
            direction: 'above', 'below', or 'both'

        Returns:
            Nearest liquidity level or None
        """
        equal_highs = self.find_equal_highs(df)
        equal_lows = self.find_equal_lows(df)

        candidates = []

        if direction in ["above", "both"]:
            candidates.extend(
                [
                    {"distance": eh["level"] - current_price, **eh}
                    for eh in equal_highs
                    if eh["level"] > current_price
                ]
            )

        if direction in ["below", "both"]:
            candidates.extend(
                [
                    {"distance": current_price - el["level"], **el}
                    for el in equal_lows
                    if el["level"] < current_price
                ]
            )

        if not candidates:
            return None

        return min(candidates, key=lambda x: x["distance"])

    def get_liquidity_summary(self, df: pl.DataFrame) -> Dict:
        """
        Get summary of liquidity status.

        Args:
            df: DataFrame with liquidity data

        Returns:
            Summary dictionary
        """
        latest_bar = df.tail(1).to_dicts()[0]
        current_price = latest_bar["close"]

        equal_highs = self.find_equal_highs(df)
        equal_lows = self.find_equal_lows(df)
        high_sweeps, low_sweeps = self.detect_recent_sweeps(df, 10)

        # Find nearest levels
        nearest_above = self.get_nearest_liquidity_level(df, current_price, "above")
        nearest_below = self.get_nearest_liquidity_level(df, current_price, "below")

        return {
            "latest_bar": {
                "is_local_high": latest_bar.get("is_local_high", False),
                "is_local_low": latest_bar.get("is_local_low", False),
                "swept_high": latest_bar.get("swept_high_liquidity", False),
                "swept_low": latest_bar.get("swept_low_liquidity", False),
            },
            "equal_levels": {
                "highs": equal_highs[:5],  # Top 5
                "lows": equal_lows[:5],
                "highs_count": len(equal_highs),
                "lows_count": len(equal_lows),
            },
            "recent_sweeps": {
                "high_sweeps": high_sweeps,
                "low_sweeps": low_sweeps,
                "high_sweeps_count": len(high_sweeps),
                "low_sweeps_count": len(low_sweeps),
            },
            "nearest_liquidity": {
                "above": nearest_above,
                "below": nearest_below,
            },
        }
