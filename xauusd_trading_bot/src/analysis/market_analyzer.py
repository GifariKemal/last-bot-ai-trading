"""
Market Analyzer
Analyzes overall market conditions: trending, ranging, volatile.
Determines the current market state to adjust strategy parameters.
"""

from typing import Dict, Optional
import polars as pl

from ..core.constants import MarketCondition, VolatilityLevel, TrendDirection
from ..bot_logger import get_logger


class MarketAnalyzer:
    """Analyze overall market conditions."""

    def __init__(self, lookback_bars: int = 50):
        """
        Initialize market analyzer.

        Args:
            lookback_bars: Bars to analyze for market condition
        """
        self.logger = get_logger()
        self.lookback_bars = lookback_bars

    def analyze(self, df: pl.DataFrame) -> Dict:
        """
        Analyze current market condition.

        Args:
            df: DataFrame with OHLC and indicator data

        Returns:
            Dictionary with market analysis
        """
        try:
            # Get recent data
            recent_df = df.tail(self.lookback_bars)

            # Analyze components
            trend = self._analyze_trend(recent_df)
            volatility = self._analyze_volatility(recent_df)
            condition = self._determine_market_condition(recent_df, trend, volatility)
            momentum = self._analyze_momentum(recent_df)
            range_info = self._analyze_range(recent_df)

            analysis = {
                "condition": condition,
                "trend": trend,
                "volatility": volatility,
                "momentum": momentum,
                "range_info": range_info,
                "is_favorable": self._is_market_favorable(condition, volatility),
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing market: {e}")
            return {
                "condition": MarketCondition.UNKNOWN,
                "trend": TrendDirection.NEUTRAL,
                "volatility": VolatilityLevel.MEDIUM,
                "momentum": 0.0,
                "range_info": {},
                "is_favorable": False,
            }

    def _analyze_trend(self, df: pl.DataFrame) -> TrendDirection:
        """
        Analyze trend direction using EMAs and price action.

        Args:
            df: DataFrame with EMA data

        Returns:
            Trend direction
        """
        latest = df.tail(1).to_dicts()[0]

        # Check if we have EMA data
        if "ema_20" not in df.columns or "ema_50" not in df.columns:
            return TrendDirection.NEUTRAL

        ema_20 = latest.get("ema_20")
        ema_50 = latest.get("ema_50")
        ema_100 = latest.get("ema_100")
        ema_200 = latest.get("ema_200")
        close = latest.get("close")

        if not all([ema_20, ema_50, close]):
            return TrendDirection.NEUTRAL

        # Strong trend criteria
        if ema_100 and ema_200:
            if ema_20 > ema_50 > ema_100 > ema_200 and close > ema_20:
                return TrendDirection.BULLISH
            elif ema_20 < ema_50 < ema_100 < ema_200 and close < ema_20:
                return TrendDirection.BEARISH

        # Medium trend criteria
        if ema_20 > ema_50 and close > ema_20:
            return TrendDirection.BULLISH
        elif ema_20 < ema_50 and close < ema_20:
            return TrendDirection.BEARISH

        return TrendDirection.NEUTRAL

    def _analyze_volatility(self, df: pl.DataFrame) -> VolatilityLevel:
        """
        Analyze volatility level using ATR percentile.

        Args:
            df: DataFrame with ATR data

        Returns:
            Volatility level
        """
        if "atr_14" not in df.columns:
            return VolatilityLevel.MEDIUM

        # Get current ATR
        latest = df.tail(1).to_dicts()[0]
        current_atr = latest.get("atr_14")

        if current_atr is None:
            return VolatilityLevel.MEDIUM

        # Calculate ATR percentile
        atr_values = df["atr_14"].drop_nulls().to_list()
        if not atr_values:
            return VolatilityLevel.MEDIUM

        # Sort and find percentile
        sorted_atr = sorted(atr_values)
        n = len(sorted_atr)
        current_percentile = (sorted_atr.index(min(sorted_atr, key=lambda x: abs(x - current_atr))) / n) * 100

        # Classify volatility
        if current_percentile < 30:
            return VolatilityLevel.LOW
        elif current_percentile > 70:
            return VolatilityLevel.HIGH
        else:
            return VolatilityLevel.MEDIUM

    def _determine_market_condition(
        self, df: pl.DataFrame, trend: TrendDirection, volatility: VolatilityLevel
    ) -> MarketCondition:
        """
        Determine overall market condition.

        Args:
            df: DataFrame
            trend: Current trend
            volatility: Current volatility

        Returns:
            Market condition
        """
        # High volatility = volatile market
        if volatility == VolatilityLevel.HIGH:
            return MarketCondition.VOLATILE

        # Check price range vs movement
        latest_50 = df.tail(50)
        if latest_50.is_empty():
            return MarketCondition.UNKNOWN

        high_max = latest_50["high"].max()
        low_min = latest_50["low"].min()
        price_range = high_max - low_min

        # Calculate actual movement
        close_changes = latest_50["close_change"].drop_nulls().abs().sum()

        if price_range == 0:
            return MarketCondition.RANGING

        # If movement is much larger than range, it's trending
        movement_ratio = close_changes / price_range

        if movement_ratio > 2.0 and trend != TrendDirection.NEUTRAL:
            if trend == TrendDirection.BULLISH:
                return MarketCondition.TRENDING_UP
            else:
                return MarketCondition.TRENDING_DOWN
        elif movement_ratio < 1.0:
            return MarketCondition.RANGING
        else:
            # Medium movement - check trend
            if trend == TrendDirection.BULLISH:
                return MarketCondition.TRENDING_UP
            elif trend == TrendDirection.BEARISH:
                return MarketCondition.TRENDING_DOWN
            else:
                return MarketCondition.RANGING

    def _analyze_momentum(self, df: pl.DataFrame) -> float:
        """
        Analyze momentum using MACD and RSI.

        Args:
            df: DataFrame with MACD and RSI

        Returns:
            Momentum score (-1 to +1)
        """
        latest = df.tail(1).to_dicts()[0]

        momentum = 0.0
        factors = 0

        # MACD momentum
        if "macd_histogram" in df.columns:
            macd_hist = latest.get("macd_histogram", 0)
            if macd_hist:
                momentum += macd_hist / 10  # Normalize
                factors += 1

        # RSI momentum
        if "rsi_14" in df.columns:
            rsi = latest.get("rsi_14")
            if rsi:
                # Convert RSI to -1 to +1 scale
                rsi_momentum = (rsi - 50) / 50
                momentum += rsi_momentum
                factors += 1

        # Average momentum
        if factors > 0:
            momentum = momentum / factors
            # Clamp to -1 to +1
            momentum = max(-1.0, min(1.0, momentum))

        return momentum

    def _analyze_range(self, df: pl.DataFrame) -> Dict:
        """
        Analyze price range information.

        Args:
            df: DataFrame

        Returns:
            Range information dictionary
        """
        if df.is_empty():
            return {}

        high_max = df["high"].max()
        low_min = df["low"].min()
        current_close = df["close"][-1]

        price_range = high_max - low_min
        if price_range == 0:
            position_in_range = 0.5
        else:
            position_in_range = (current_close - low_min) / price_range

        return {
            "high": high_max,
            "low": low_min,
            "range": price_range,
            "current_position": position_in_range,  # 0 = bottom, 1 = top
            "near_high": position_in_range > 0.8,
            "near_low": position_in_range < 0.2,
            "in_middle": 0.3 <= position_in_range <= 0.7,
        }

    def _is_market_favorable(
        self, condition: MarketCondition, volatility: VolatilityLevel
    ) -> bool:
        """
        Determine if market conditions are favorable for trading.

        Args:
            condition: Market condition
            volatility: Volatility level

        Returns:
            True if favorable
        """
        # Allow VOLATILE markets — XAUUSD is naturally volatile, HIGH ATR is normal
        # SMC confluence scoring (FVG, OB, BOS/CHoCH, RSI checks) already gate quality
        # Only block UNKNOWN condition (no data / error state)
        favorable_conditions = [
            MarketCondition.TRENDING_UP,
            MarketCondition.TRENDING_DOWN,
            MarketCondition.RANGING,
            MarketCondition.VOLATILE,  # Allow — filtered by confluence + RSI checks
        ]

        return condition in favorable_conditions

    def get_market_summary(self, df: pl.DataFrame) -> str:
        """
        Get human-readable market summary.

        Args:
            df: DataFrame with analysis

        Returns:
            Market summary string
        """
        analysis = self.analyze(df)

        condition = analysis["condition"].value
        trend = analysis["trend"].value
        volatility = analysis["volatility"].value
        momentum = analysis["momentum"]
        favorable = "FAVORABLE" if analysis["is_favorable"] else "UNFAVORABLE"

        summary = (
            f"Market: {condition} | "
            f"Trend: {trend} | "
            f"Volatility: {volatility} | "
            f"Momentum: {momentum:.2f} | "
            f"Status: {favorable}"
        )

        return summary
