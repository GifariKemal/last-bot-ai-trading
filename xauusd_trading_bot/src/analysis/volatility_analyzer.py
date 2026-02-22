"""
Volatility Analyzer
Detailed volatility analysis and classification.
Adjusts strategy parameters based on volatility regime.
"""

from typing import Dict
import polars as pl

from ..core.constants import VolatilityLevel
from ..bot_logger import get_logger


class VolatilityAnalyzer:
    """Analyze and classify market volatility."""

    def __init__(self, lookback_bars: int = 100):
        """
        Initialize volatility analyzer.

        Args:
            lookback_bars: Bars to analyze for volatility
        """
        self.logger = get_logger()
        self.lookback_bars = lookback_bars

    def analyze(self, df: pl.DataFrame) -> Dict:
        """
        Analyze volatility comprehensively.

        Args:
            df: DataFrame with ATR and Bollinger Bands data

        Returns:
            Volatility analysis dictionary
        """
        try:
            recent_df = df.tail(self.lookback_bars)

            # Multiple volatility measures
            atr_analysis = self._analyze_atr(recent_df)
            bb_analysis = self._analyze_bollinger_bands(recent_df)
            range_analysis = self._analyze_price_range(recent_df)

            # Determine overall volatility level
            level = self._determine_volatility_level(
                atr_analysis, bb_analysis, range_analysis
            )

            # Get adjustments
            adjustments = self._get_volatility_adjustments(level)

            return {
                "level": level,
                "atr_analysis": atr_analysis,
                "bollinger_analysis": bb_analysis,
                "range_analysis": range_analysis,
                "adjustments": adjustments,
                "is_expanding": self._is_volatility_expanding(recent_df),
                "is_contracting": self._is_volatility_contracting(recent_df),
            }

        except Exception as e:
            self.logger.error(f"Error analyzing volatility: {e}")
            return {
                "level": VolatilityLevel.MEDIUM,
                "adjustments": {},
            }

    def _analyze_atr(self, df: pl.DataFrame) -> Dict:
        """Analyze ATR for volatility."""
        if "atr_14" not in df.columns:
            return {"current": None, "percentile": 50}

        atr_values = df["atr_14"].drop_nulls().to_list()
        if not atr_values:
            return {"current": None, "percentile": 50}

        current_atr = atr_values[-1]

        # Calculate percentile
        sorted_atr = sorted(atr_values)
        percentile = (sorted_atr.index(min(sorted_atr, key=lambda x: abs(x - current_atr))) / len(sorted_atr)) * 100

        # Calculate average
        avg_atr = sum(atr_values) / len(atr_values)

        return {
            "current": current_atr,
            "average": avg_atr,
            "percentile": percentile,
            "above_average": current_atr > avg_atr,
        }

    def _analyze_bollinger_bands(self, df: pl.DataFrame) -> Dict:
        """Analyze Bollinger Bands for volatility."""
        if "bb_bandwidth" not in df.columns:
            return {"bandwidth": None, "squeeze": False}

        latest = df.tail(1).to_dicts()[0]
        bandwidth = latest.get("bb_bandwidth")

        if bandwidth is None:
            return {"bandwidth": None, "squeeze": False}

        # Check for squeeze (low volatility)
        bandwidth_values = df["bb_bandwidth"].drop_nulls().to_list()
        if bandwidth_values:
            avg_bandwidth = sum(bandwidth_values) / len(bandwidth_values)
            squeeze = bandwidth < avg_bandwidth * 0.5  # Less than 50% of average
        else:
            squeeze = False

        return {
            "bandwidth": bandwidth,
            "squeeze": squeeze,
            "expansion": bandwidth > 4.0,  # High bandwidth = expansion
        }

    def _analyze_price_range(self, df: pl.DataFrame) -> Dict:
        """Analyze recent price range volatility."""
        if df.is_empty():
            return {}

        # Calculate average range
        ranges = df["range"].drop_nulls().to_list()
        if not ranges:
            return {}

        avg_range = sum(ranges) / len(ranges)
        current_range = ranges[-1] if ranges else 0

        return {
            "current_range": current_range,
            "average_range": avg_range,
            "range_ratio": current_range / avg_range if avg_range > 0 else 1.0,
        }

    def _determine_volatility_level(
        self, atr_analysis: Dict, bb_analysis: Dict, range_analysis: Dict
    ) -> VolatilityLevel:
        """
        Determine overall volatility level from multiple sources.

        Args:
            atr_analysis: ATR analysis results
            bb_analysis: Bollinger Bands analysis
            range_analysis: Price range analysis

        Returns:
            Volatility level
        """
        votes = []

        # ATR vote
        if atr_analysis.get("percentile"):
            percentile = atr_analysis["percentile"]
            if percentile < 30:
                votes.append(VolatilityLevel.LOW)
            elif percentile > 70:
                votes.append(VolatilityLevel.HIGH)
            else:
                votes.append(VolatilityLevel.MEDIUM)

        # Bollinger Bands vote
        if bb_analysis.get("squeeze"):
            votes.append(VolatilityLevel.LOW)
        elif bb_analysis.get("expansion"):
            votes.append(VolatilityLevel.HIGH)
        else:
            votes.append(VolatilityLevel.MEDIUM)

        # Range vote
        if range_analysis.get("range_ratio"):
            ratio = range_analysis["range_ratio"]
            if ratio < 0.7:
                votes.append(VolatilityLevel.LOW)
            elif ratio > 1.3:
                votes.append(VolatilityLevel.HIGH)
            else:
                votes.append(VolatilityLevel.MEDIUM)

        # Majority vote
        if not votes:
            return VolatilityLevel.MEDIUM

        low_count = votes.count(VolatilityLevel.LOW)
        high_count = votes.count(VolatilityLevel.HIGH)
        medium_count = votes.count(VolatilityLevel.MEDIUM)

        if low_count >= 2:
            return VolatilityLevel.LOW
        elif high_count >= 2:
            return VolatilityLevel.HIGH
        else:
            return VolatilityLevel.MEDIUM

    def _is_volatility_expanding(self, df: pl.DataFrame) -> bool:
        """Check if volatility is expanding."""
        if "atr_14" not in df.columns or len(df) < 20:
            return False

        recent_atr = df["atr_14"].tail(10).drop_nulls().to_list()
        older_atr = df["atr_14"].tail(20).head(10).drop_nulls().to_list()

        if not recent_atr or not older_atr:
            return False

        recent_avg = sum(recent_atr) / len(recent_atr)
        older_avg = sum(older_atr) / len(older_atr)

        # Expanding if recent ATR > 20% higher than older
        return recent_avg > older_avg * 1.2

    def _is_volatility_contracting(self, df: pl.DataFrame) -> bool:
        """Check if volatility is contracting."""
        if "atr_14" not in df.columns or len(df) < 20:
            return False

        recent_atr = df["atr_14"].tail(10).drop_nulls().to_list()
        older_atr = df["atr_14"].tail(20).head(10).drop_nulls().to_list()

        if not recent_atr or not older_atr:
            return False

        recent_avg = sum(recent_atr) / len(recent_atr)
        older_avg = sum(older_atr) / len(older_atr)

        # Contracting if recent ATR < 20% lower than older
        return recent_avg < older_avg * 0.8

    def _get_volatility_adjustments(self, level: VolatilityLevel) -> Dict:
        """
        Get strategy adjustments based on volatility level.

        Args:
            level: Volatility level

        Returns:
            Adjustment parameters
        """
        adjustments = {
            VolatilityLevel.LOW: {
                "sl_multiplier": 0.8,  # Tighter stops
                "tp_multiplier": 0.8,  # Closer targets
                "confluence_adjustment": 0.0,  # No change
                "position_size_multiplier": 1.0,
                "description": "Low volatility - tighter stops and targets",
            },
            VolatilityLevel.MEDIUM: {
                "sl_multiplier": 1.0,  # Normal
                "tp_multiplier": 1.0,
                "confluence_adjustment": 0.0,
                "position_size_multiplier": 1.0,
                "description": "Normal volatility - standard parameters",
            },
            VolatilityLevel.HIGH: {
                "sl_multiplier": 1.5,  # Wider stops
                "tp_multiplier": 1.3,  # Larger targets
                "confluence_adjustment": 0.10,  # Require higher confluence
                "position_size_multiplier": 0.6,  # Reduce size
                "description": "High volatility - wider stops, reduced size",
            },
        }

        return adjustments.get(level, adjustments[VolatilityLevel.MEDIUM])
