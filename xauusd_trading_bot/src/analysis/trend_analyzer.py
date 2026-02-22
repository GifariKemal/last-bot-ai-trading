"""
Trend Analyzer
Identifies and measures trend strength using multiple methods.
"""

from typing import Dict
import polars as pl

from ..core.constants import TrendDirection
from ..bot_logger import get_logger


class TrendAnalyzer:
    """Analyze market trend direction and strength."""

    def __init__(self):
        """Initialize trend analyzer."""
        self.logger = get_logger()

    def analyze(self, df: pl.DataFrame) -> Dict:
        """
        Comprehensive trend analysis.

        Args:
            df: DataFrame with indicators

        Returns:
            Trend analysis dictionary
        """
        try:
            # Multiple trend detection methods
            ema_trend = self._analyze_ema_trend(df)
            structure_trend = self._analyze_structure_trend(df)
            momentum_trend = self._analyze_momentum_trend(df)

            # Determine overall trend
            trend = self._determine_overall_trend(ema_trend, structure_trend, momentum_trend)

            # Calculate trend strength
            strength = self._calculate_trend_strength(df, trend)

            return {
                "direction": trend,
                "strength": strength,
                "ema_trend": ema_trend,
                "structure_trend": structure_trend,
                "momentum_trend": momentum_trend,
                "aligned": self._are_trends_aligned(ema_trend, structure_trend, momentum_trend),
            }

        except Exception as e:
            self.logger.error(f"Error analyzing trend: {e}")
            return {
                "direction": TrendDirection.NEUTRAL,
                "strength": 0.0,
            }

    def _analyze_ema_trend(self, df: pl.DataFrame) -> TrendDirection:
        """Analyze trend using EMAs."""
        if df.is_empty() or "ema_20" not in df.columns:
            return TrendDirection.NEUTRAL

        latest = df.tail(1).to_dicts()[0]

        ema_20 = latest.get("ema_20")
        ema_50 = latest.get("ema_50")
        ema_100 = latest.get("ema_100")
        ema_200 = latest.get("ema_200")
        close = latest.get("close")

        if not all([ema_20, ema_50, close]):
            return TrendDirection.NEUTRAL

        # Check EMA alignment
        if ema_100 and ema_200:
            # Strong trend
            if ema_20 > ema_50 > ema_100 > ema_200 and close > ema_20:
                return TrendDirection.BULLISH
            elif ema_20 < ema_50 < ema_100 < ema_200 and close < ema_20:
                return TrendDirection.BEARISH

        # Weak trend
        if ema_20 > ema_50 and close > ema_20:
            return TrendDirection.BULLISH
        elif ema_20 < ema_50 and close < ema_20:
            return TrendDirection.BEARISH

        return TrendDirection.NEUTRAL

    def _analyze_structure_trend(self, df: pl.DataFrame) -> TrendDirection:
        """Analyze trend using market structure."""
        if df.is_empty() or "market_structure_trend" not in df.columns:
            return TrendDirection.NEUTRAL

        latest = df.tail(1).to_dicts()[0]
        structure_trend = latest.get("market_structure_trend", "NEUTRAL")

        if structure_trend == "BULLISH":
            return TrendDirection.BULLISH
        elif structure_trend == "BEARISH":
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL

    def _analyze_momentum_trend(self, df: pl.DataFrame) -> TrendDirection:
        """Analyze trend using momentum indicators."""
        if df.is_empty():
            return TrendDirection.NEUTRAL

        latest = df.tail(1).to_dicts()[0]

        # MACD
        macd_line = latest.get("macd_line")
        macd_signal = latest.get("macd_signal")

        # RSI
        rsi = latest.get("rsi_14")

        bullish_votes = 0
        bearish_votes = 0

        # MACD vote
        if macd_line is not None and macd_signal is not None:
            if macd_line > macd_signal and macd_line > 0:
                bullish_votes += 1
            elif macd_line < macd_signal and macd_line < 0:
                bearish_votes += 1

        # RSI vote
        if rsi is not None:
            if rsi > 55:
                bullish_votes += 1
            elif rsi < 45:
                bearish_votes += 1

        if bullish_votes > bearish_votes:
            return TrendDirection.BULLISH
        elif bearish_votes > bullish_votes:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL

    def _determine_overall_trend(
        self,
        ema_trend: TrendDirection,
        structure_trend: TrendDirection,
        momentum_trend: TrendDirection,
    ) -> TrendDirection:
        """
        Determine overall trend from multiple sources.

        Args:
            ema_trend: EMA-based trend
            structure_trend: Structure-based trend
            momentum_trend: Momentum-based trend

        Returns:
            Overall trend direction
        """
        # Count votes
        bullish_votes = sum([
            ema_trend == TrendDirection.BULLISH,
            structure_trend == TrendDirection.BULLISH,
            momentum_trend == TrendDirection.BULLISH,
        ])

        bearish_votes = sum([
            ema_trend == TrendDirection.BEARISH,
            structure_trend == TrendDirection.BEARISH,
            momentum_trend == TrendDirection.BEARISH,
        ])

        # Majority vote with structure having priority
        if bullish_votes >= 2:
            return TrendDirection.BULLISH
        elif bearish_votes >= 2:
            return TrendDirection.BEARISH
        elif structure_trend != TrendDirection.NEUTRAL:
            return structure_trend  # Structure has priority
        else:
            return TrendDirection.NEUTRAL

    def _calculate_trend_strength(
        self, df: pl.DataFrame, trend: TrendDirection
    ) -> float:
        """
        Calculate trend strength (0 to 1).

        Args:
            df: DataFrame
            trend: Current trend direction

        Returns:
            Trend strength score
        """
        if trend == TrendDirection.NEUTRAL:
            return 0.0

        strength = 0.0
        factors = 0

        # EMA alignment strength
        if "ema_20" in df.columns and "ema_50" in df.columns:
            latest = df.tail(1).to_dicts()[0]
            ema_20 = latest.get("ema_20", 0)
            ema_50 = latest.get("ema_50", 0)
            close = latest.get("close", 0)

            if ema_50 > 0:
                ema_separation = abs(ema_20 - ema_50) / ema_50
                strength += min(ema_separation * 10, 1.0)  # Normalize
                factors += 1

            # Price distance from EMA
            if ema_20 > 0:
                price_separation = abs(close - ema_20) / ema_20
                strength += min(price_separation * 20, 1.0)
                factors += 1

        # Momentum strength
        if "rsi_14" in df.columns:
            latest = df.tail(1).to_dicts()[0]
            rsi = latest.get("rsi_14")
            if rsi:
                # Distance from neutral (50)
                rsi_strength = abs(rsi - 50) / 50
                strength += rsi_strength
                factors += 1

        # Average strength
        if factors > 0:
            strength = strength / factors

        # Clamp to 0-1
        return max(0.0, min(1.0, strength))

    def _are_trends_aligned(
        self,
        ema_trend: TrendDirection,
        structure_trend: TrendDirection,
        momentum_trend: TrendDirection,
    ) -> bool:
        """
        Check if all trend methods are aligned.

        Returns:
            True if all trends agree
        """
        trends = [ema_trend, structure_trend, momentum_trend]

        # Remove neutral votes
        non_neutral = [t for t in trends if t != TrendDirection.NEUTRAL]

        if not non_neutral:
            return False

        # All non-neutral trends must be same
        return len(set(non_neutral)) == 1
