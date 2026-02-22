"""
Multi-Timeframe (MTF) Analyzer
Analyzes multiple timeframes and checks for alignment.
"""

from typing import Dict, List
import polars as pl

from ..core.constants import TrendDirection
from ..bot_logger import get_logger


class MTFAnalyzer:
    """Multi-timeframe analysis and alignment."""

    def __init__(self):
        """Initialize MTF analyzer."""
        self.logger = get_logger()

    def analyze(self, timeframe_data: Dict[str, pl.DataFrame]) -> Dict:
        """
        Analyze multiple timeframes for alignment.

        Args:
            timeframe_data: Dictionary mapping timeframe to DataFrame

        Returns:
            MTF analysis dictionary
        """
        try:
            # Analyze each timeframe
            tf_analysis = {}
            for tf, df in timeframe_data.items():
                tf_analysis[tf] = self._analyze_single_timeframe(df)

            # Check alignment
            alignment = self._check_alignment(tf_analysis)

            # Determine dominant trend
            dominant_trend = self._determine_dominant_trend(tf_analysis)

            return {
                "timeframes": tf_analysis,
                "alignment": alignment,
                "dominant_trend": dominant_trend,
                "is_aligned": alignment["aligned"],
                "alignment_score": alignment["score"],
            }

        except Exception as e:
            self.logger.error(f"Error in MTF analysis: {e}")
            return {
                "timeframes": {},
                "alignment": {"aligned": False, "score": 0.0},
                "dominant_trend": TrendDirection.NEUTRAL,
                "is_aligned": False,
                "alignment_score": 0.0,
            }

    def _analyze_single_timeframe(self, df: pl.DataFrame) -> Dict:
        """
        Analyze a single timeframe.

        Args:
            df: DataFrame for timeframe

        Returns:
            Analysis dictionary
        """
        if df.is_empty():
            return {
                "trend": TrendDirection.NEUTRAL,
                "structure_trend": TrendDirection.NEUTRAL,
            }

        latest = df.tail(1).to_dicts()[0]

        # Get trend from structure
        structure_trend_str = latest.get("market_structure_trend", "NEUTRAL")
        if structure_trend_str == "BULLISH":
            structure_trend = TrendDirection.BULLISH
        elif structure_trend_str == "BEARISH":
            structure_trend = TrendDirection.BEARISH
        else:
            structure_trend = TrendDirection.NEUTRAL

        # Get trend from EMAs
        ema_20 = latest.get("ema_20")
        ema_50 = latest.get("ema_50")
        close = latest.get("close")

        if ema_20 and ema_50 and close:
            if ema_20 > ema_50 and close > ema_20:
                ema_trend = TrendDirection.BULLISH
            elif ema_20 < ema_50 and close < ema_20:
                ema_trend = TrendDirection.BEARISH
            else:
                ema_trend = TrendDirection.NEUTRAL
        else:
            ema_trend = TrendDirection.NEUTRAL

        # Overall trend (structure has priority)
        if structure_trend != TrendDirection.NEUTRAL:
            trend = structure_trend
        else:
            trend = ema_trend

        return {
            "trend": trend,
            "structure_trend": structure_trend,
            "ema_trend": ema_trend,
        }

    def _check_alignment(self, tf_analysis: Dict) -> Dict:
        """
        Check if timeframes are aligned.

        Args:
            tf_analysis: Timeframe analysis results

        Returns:
            Alignment information
        """
        if not tf_analysis:
            return {"aligned": False, "score": 0.0}

        # Get all trends
        trends = [analysis["trend"] for analysis in tf_analysis.values()]

        # Remove neutral
        non_neutral = [t for t in trends if t != TrendDirection.NEUTRAL]

        if not non_neutral:
            return {
                "aligned": False,
                "score": 0.0,
                "reason": "All timeframes neutral",
            }

        # Count bullish and bearish
        bullish_count = non_neutral.count(TrendDirection.BULLISH)
        bearish_count = non_neutral.count(TrendDirection.BEARISH)

        total = len(non_neutral)

        # Perfect alignment
        if bullish_count == total or bearish_count == total:
            return {
                "aligned": True,
                "score": 1.0,
                "direction": TrendDirection.BULLISH if bullish_count == total else TrendDirection.BEARISH,
                "reason": "Perfect alignment",
            }

        # Majority alignment
        majority = max(bullish_count, bearish_count)
        score = majority / total

        if score >= 0.66:  # 2/3 or more
            direction = TrendDirection.BULLISH if bullish_count > bearish_count else TrendDirection.BEARISH
            return {
                "aligned": True,
                "score": score,
                "direction": direction,
                "reason": f"Majority alignment ({majority}/{total})",
            }

        return {
            "aligned": False,
            "score": score,
            "reason": f"Insufficient alignment ({majority}/{total})",
        }

    def _determine_dominant_trend(self, tf_analysis: Dict) -> TrendDirection:
        """
        Determine dominant trend across timeframes.
        Higher timeframes have more weight.

        Args:
            tf_analysis: Timeframe analysis

        Returns:
            Dominant trend direction
        """
        # Timeframe weights (higher timeframe = more weight)
        weights = {
            "H1": 3.0,
            "M15": 2.0,
            "M5": 1.0,
            "M1": 0.5,
        }

        bullish_weight = 0.0
        bearish_weight = 0.0

        for tf, analysis in tf_analysis.items():
            weight = weights.get(tf, 1.0)
            trend = analysis["trend"]

            if trend == TrendDirection.BULLISH:
                bullish_weight += weight
            elif trend == TrendDirection.BEARISH:
                bearish_weight += weight

        if bullish_weight > bearish_weight:
            return TrendDirection.BULLISH
        elif bearish_weight > bullish_weight:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL

    def get_mtf_summary(self, timeframe_data: Dict[str, pl.DataFrame]) -> str:
        """
        Get human-readable MTF summary.

        Args:
            timeframe_data: Dictionary of timeframe DataFrames

        Returns:
            Summary string
        """
        analysis = self.analyze(timeframe_data)

        aligned_status = "ALIGNED" if analysis["is_aligned"] else "NOT ALIGNED"
        dominant = analysis["dominant_trend"].value
        score = analysis["alignment_score"]

        summary = f"MTF: {aligned_status} | Dominant: {dominant} | Score: {score:.2f}"

        return summary
