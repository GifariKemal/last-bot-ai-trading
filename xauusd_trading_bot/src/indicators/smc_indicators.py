"""
Smart Money Concepts (SMC) Indicators Aggregator
Combines all SMC indicators: FVG, Order Blocks, Liquidity, Structure
"""

from typing import Dict, Optional
import polars as pl

from .fvg_detector import FVGDetector
from .order_block_detector import OrderBlockDetector
from .liquidity_detector import LiquidityDetector
from .structure_detector import StructureDetector
from ..core.constants import FVGType, OrderBlockType, LiquidityType, TrendDirection
from ..bot_logger import get_logger


class SMCIndicators:
    """
    Aggregates all Smart Money Concepts indicators.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize SMC indicators.

        Args:
            config: Configuration dictionary
        """
        self.logger = get_logger()

        # Default configuration
        if config is None:
            config = {
                "fair_value_gaps": {
                    "min_gap_pips": 5.0,
                    "max_age_bars": 100,
                },
                "order_blocks": {
                    "lookback_bars": 50,
                    "min_body_percent": 60.0,
                    "strong_move_bars": 5,
                    "strong_move_percent": 1.0,
                },
                "liquidity": {
                    "lookback_bars": 30,
                    "equal_level_tolerance_pips": 3.0,
                    "min_touches": 2,
                    "sweep_confirmation_bars": 3,
                },
                "structure": {
                    "swing_lookback": 10,
                    "bos_confirmation_bars": 2,
                },
            }

        self.config = config

        # Signal weights from config (with defaults matching original hardcoded values)
        fvg_cfg    = config.get("fair_value_gaps", {})
        ob_cfg     = config.get("order_blocks", {})
        liq_cfg    = config.get("liquidity", {})
        struct_cfg = config.get("structure", {})
        self._w_fvg   = fvg_cfg.get("weight", 0.20)
        self._w_ob    = ob_cfg.get("weight", 0.25)
        self._w_liq   = liq_cfg.get("weight", 0.20)
        self._w_choch = struct_cfg.get("choch_weight", 0.30)
        self._w_bos   = struct_cfg.get("bos_weight", 0.15)

        # Initialize detectors
        self.fvg = FVGDetector(**config.get("fair_value_gaps", {}))
        self.order_blocks = OrderBlockDetector(**config.get("order_blocks", {}))
        self.liquidity = LiquidityDetector(**config.get("liquidity", {}))
        self.structure = StructureDetector(**config.get("structure", {}))

    def calculate_all(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate all SMC indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all SMC indicator columns
        """
        try:
            # Calculate each SMC indicator
            self.logger.debug("Calculating Fair Value Gaps...")
            df = self.fvg.calculate(df)

            self.logger.debug("Calculating Order Blocks...")
            df = self.order_blocks.calculate(df)

            self.logger.debug("Calculating Liquidity levels...")
            df = self.liquidity.calculate(df)

            self.logger.debug("Calculating Market Structure...")
            df = self.structure.calculate(df)

            self.logger.debug("All SMC indicators calculated successfully")
            return df

        except Exception as e:
            self.logger.error(f"Error calculating SMC indicators: {e}")
            raise

    def get_comprehensive_summary(self, df: pl.DataFrame) -> Dict:
        """
        Get comprehensive summary of all SMC indicators.

        Args:
            df: DataFrame with calculated SMC indicators

        Returns:
            Dictionary with all SMC indicator summaries
        """
        return {
            "fair_value_gaps": self.fvg.get_fvg_summary(df),
            "order_blocks": self.order_blocks.get_ob_summary(df),
            "liquidity": self.liquidity.get_liquidity_summary(df),
            "structure": self.structure.get_structure_summary(df),
        }

    def get_bullish_signals(self, df: pl.DataFrame, current_price: float) -> Dict:
        """
        Get bullish SMC signals.

        Args:
            df: DataFrame with SMC data
            current_price: Current price

        Returns:
            Dictionary with bullish signals and scores
        """
        signals = {
            "fvg": {
                "in_zone": self.fvg.is_price_in_fvg(
                    df, FVGType.BULLISH, current_price
                ),
                "nearest": self.fvg.get_nearest_fvg(
                    df, FVGType.BULLISH, current_price
                ),
            },
            "order_block": {
                "at_zone": self.order_blocks.is_price_at_order_block(
                    df, OrderBlockType.BULLISH, current_price
                ),
                "nearest": self.order_blocks.get_nearest_order_block(
                    df, OrderBlockType.BULLISH, current_price
                ),
            },
            "liquidity": {
                "swept": self.liquidity.is_liquidity_sweep_active(
                    df, LiquidityType.LOW
                ),
            },
            "structure": {
                "choch": self.structure.has_recent_choch(
                    df, TrendDirection.BULLISH, bars=50
                ),
                "bos": self.structure.has_recent_bos(
                    df, TrendDirection.BULLISH, bars=50
                ),
                "trend": self.structure.get_current_trend(df),
                **self._get_dealing_range(df),
            },
            "current_price": current_price,
        }

        # Calculate confluence score using config-driven weights
        score = 0.0
        if signals["fvg"]["in_zone"]:
            score += self._w_fvg
        if signals["order_block"]["at_zone"]:
            score += self._w_ob
        if signals["liquidity"]["swept"]:
            score += self._w_liq
        if signals["structure"]["choch"]:
            score += self._w_choch
        elif signals["structure"]["bos"]:
            score += self._w_bos

        signals["confluence_score"] = score

        return signals

    def get_bearish_signals(self, df: pl.DataFrame, current_price: float) -> Dict:
        """
        Get bearish SMC signals.

        Args:
            df: DataFrame with SMC data
            current_price: Current price

        Returns:
            Dictionary with bearish signals and scores
        """
        signals = {
            "fvg": {
                "in_zone": self.fvg.is_price_in_fvg(
                    df, FVGType.BEARISH, current_price
                ),
                "nearest": self.fvg.get_nearest_fvg(
                    df, FVGType.BEARISH, current_price
                ),
            },
            "order_block": {
                "at_zone": self.order_blocks.is_price_at_order_block(
                    df, OrderBlockType.BEARISH, current_price
                ),
                "nearest": self.order_blocks.get_nearest_order_block(
                    df, OrderBlockType.BEARISH, current_price
                ),
            },
            "liquidity": {
                "swept": self.liquidity.is_liquidity_sweep_active(
                    df, LiquidityType.HIGH
                ),
            },
            "structure": {
                "choch": self.structure.has_recent_choch(
                    df, TrendDirection.BEARISH, bars=50
                ),
                "bos": self.structure.has_recent_bos(
                    df, TrendDirection.BEARISH, bars=50
                ),
                "trend": self.structure.get_current_trend(df),
                **self._get_dealing_range(df),
            },
            "current_price": current_price,
        }

        # Calculate confluence score using config-driven weights
        score = 0.0
        if signals["fvg"]["in_zone"]:
            score += self._w_fvg
        if signals["order_block"]["at_zone"]:
            score += self._w_ob
        if signals["liquidity"]["swept"]:
            score += self._w_liq
        if signals["structure"]["choch"]:
            score += self._w_choch
        elif signals["structure"]["bos"]:
            score += self._w_bos

        signals["confluence_score"] = score

        return signals

    def _get_dealing_range(self, df: pl.DataFrame) -> Dict:
        """
        Get the ICT dealing range (recent swing high/low) for Premium/Discount zone.

        Uses the most recent confirmed swing high and swing low as the dealing range
        boundaries. Midpoint = (swing_high + swing_low) / 2 = equilibrium (50% level).

        Returns:
            Dict with recent_swing_high and recent_swing_low (None if unavailable)
        """
        swing_pts = self.structure.get_swing_points(df, n=5)
        recent_swing_high = swing_pts["swing_highs"][-1]["level"] if swing_pts["swing_highs"] else None
        recent_swing_low = swing_pts["swing_lows"][-1]["level"] if swing_pts["swing_lows"] else None
        return {
            "recent_swing_high": recent_swing_high,
            "recent_swing_low": recent_swing_low,
        }

    def get_trade_context(self, df: pl.DataFrame, current_price: float) -> Dict:
        """
        Get complete trading context from SMC perspective.

        Args:
            df: DataFrame with SMC data
            current_price: Current price

        Returns:
            Complete trade context dictionary
        """
        bullish_signals = self.get_bullish_signals(df, current_price)
        bearish_signals = self.get_bearish_signals(df, current_price)

        # Determine bias
        if bullish_signals["confluence_score"] > bearish_signals["confluence_score"]:
            bias = "BULLISH"
            primary_signals = bullish_signals
        elif bearish_signals["confluence_score"] > bullish_signals["confluence_score"]:
            bias = "BEARISH"
            primary_signals = bearish_signals
        else:
            bias = "NEUTRAL"
            primary_signals = None

        return {
            "bias": bias,
            "bullish_score": bullish_signals["confluence_score"],
            "bearish_score": bearish_signals["confluence_score"],
            "primary_signals": primary_signals,
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
            "current_price": current_price,
        }

    def analyze(self, df: pl.DataFrame) -> Dict:
        """
        Analyze SMC patterns (alias for get_comprehensive_summary).

        Args:
            df: DataFrame with OHLC data

        Returns:
            Comprehensive SMC analysis dictionary
        """
        # First calculate all SMC indicators
        df_with_smc = self.calculate_all(df)
        # Then return comprehensive summary
        return self.get_comprehensive_summary(df_with_smc)
