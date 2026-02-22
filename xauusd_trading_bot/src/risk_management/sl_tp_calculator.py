"""
Stop Loss and Take Profit Calculator
Calculates dynamic SL/TP levels based on ATR and market conditions.
"""

from typing import Dict, Tuple
import polars as pl

from ..core.constants import VolatilityLevel
from ..bot_logger import get_logger


class SLTPCalculator:
    """Calculate dynamic stop loss and take profit levels."""

    def __init__(self, config: Dict):
        """
        Initialize SL/TP calculator.

        Args:
            config: Risk configuration
        """
        self.logger = get_logger()
        self.config = config

        # SL configuration
        self.sl_config = config.get("stop_loss", {})
        self.base_sl_multiplier = self.sl_config.get("atr_multiplier", 2.5)
        self.min_sl_pips = self.sl_config.get("min_pips", 10.0)
        self.max_sl_pips = self.sl_config.get("max_pips", 200.0)

        # Volatility adjustments
        vol_adj = self.sl_config.get("volatility_adjustment", {})
        self.vol_multipliers = {
            VolatilityLevel.LOW: vol_adj.get("low_volatility_multiplier", 0.8),
            VolatilityLevel.MEDIUM: vol_adj.get("medium_volatility_multiplier", 1.0),
            VolatilityLevel.HIGH: vol_adj.get("high_volatility_multiplier", 1.5),
        }

        # TP configuration
        self.tp_config = config.get("take_profit", {})
        self.base_tp_multiplier = self.tp_config.get("atr_multiplier", 5.0)
        self.min_rr_ratio = self.tp_config.get("min_rr_ratio", 2.0)
        self.max_tp_pips = self.tp_config.get("max_pips", 500.0)

        # Structure buffer (pips beyond swing point) — configurable to avoid magic number
        self.structure_buffer_pips = self.sl_config.get("structure_buffer_pips", 2.0)

    def calculate_sl_tp(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        volatility_level: VolatilityLevel,
        smc_context: Dict = None,
    ) -> Dict:
        """
        Calculate stop loss and take profit levels.

        Args:
            entry_price: Entry price
            direction: "BUY" or "SELL"
            atr: Current ATR value
            volatility_level: Current volatility level
            smc_context: Optional SMC context for structure-based levels

        Returns:
            Dictionary with SL, TP, and metadata
        """
        try:
            # Calculate SL
            sl_price = self._calculate_stop_loss(
                entry_price, direction, atr, volatility_level, smc_context
            )

            # Calculate TP
            tp_price = self._calculate_take_profit(
                entry_price, direction, atr, sl_price, smc_context
            )

            # Calculate distances and RR
            sl_distance = abs(entry_price - sl_price)
            tp_distance = abs(tp_price - entry_price)
            rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0

            result = {
                "sl": sl_price,
                "tp": tp_price,
                "sl_distance_pips": sl_distance,
                "tp_distance_pips": tp_distance,
                "rr_ratio": rr_ratio,
                "atr_used": atr,
                "volatility_level": volatility_level.value,
                "method": "atr_based",
            }

            self.logger.debug(
                f"Calculated SL/TP: SL={sl_price:.2f} ({sl_distance:.1f} pips), "
                f"TP={tp_price:.2f} ({tp_distance:.1f} pips), RR={rr_ratio:.2f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error calculating SL/TP: {e}")
            # Return conservative defaults
            return self._get_default_sl_tp(entry_price, direction)

    def _calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        volatility_level: VolatilityLevel,
        smc_context: Dict = None,
    ) -> float:
        """Calculate stop loss price."""

        # Get volatility adjustment
        vol_multiplier = self.vol_multipliers.get(volatility_level, 1.0)

        # Calculate SL distance
        sl_distance = atr * self.base_sl_multiplier * vol_multiplier

        # Apply limits
        sl_distance = max(self.min_sl_pips, min(sl_distance, self.max_sl_pips))

        # Calculate SL price
        if direction == "BUY":
            sl_price = entry_price - sl_distance
        else:  # SELL
            sl_price = entry_price + sl_distance

        # Optional: Adjust based on SMC structure
        if smc_context:
            sl_price = self._adjust_sl_for_structure(
                sl_price, direction, entry_price, smc_context
            )

        return round(sl_price, 2)

    def _calculate_take_profit(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        sl_price: float,
        smc_context: Dict = None,
    ) -> float:
        """Calculate take profit price."""

        # Calculate TP distance based on ATR
        tp_distance = atr * self.base_tp_multiplier

        # Ensure minimum RR ratio
        sl_distance = abs(entry_price - sl_price)
        min_tp_distance = sl_distance * self.min_rr_ratio
        tp_distance = max(tp_distance, min_tp_distance)

        # Apply max limit
        tp_distance = min(tp_distance, self.max_tp_pips)

        # Calculate TP price
        if direction == "BUY":
            tp_price = entry_price + tp_distance
        else:  # SELL
            tp_price = entry_price - tp_distance

        # Optional: Adjust based on SMC structure
        if smc_context and self.tp_config.get("use_structure_tp", True):
            tp_price = self._adjust_tp_for_structure(
                tp_price, direction, entry_price, smc_context
            )

        return round(tp_price, 2)

    def _adjust_sl_for_structure(
        self,
        sl_price: float,
        direction: str,
        entry_price: float,
        smc_context: Dict,
    ) -> float:
        """Adjust SL based on recent structure (swing points)."""

        # If we have recent swing low/high, place SL beyond it
        buf = self.structure_buffer_pips
        if direction == "BUY":
            recent_low = smc_context.get("recent_swing_low")
            if recent_low and recent_low < entry_price:
                adjusted_sl = recent_low - buf
                # Only use if it's tighter than current SL but not too tight
                if adjusted_sl > sl_price and (entry_price - adjusted_sl) >= self.min_sl_pips:
                    return adjusted_sl
        else:  # SELL
            recent_high = smc_context.get("recent_swing_high")
            if recent_high and recent_high > entry_price:
                adjusted_sl = recent_high + buf
                # Only use if it's tighter than current SL but not too tight
                if adjusted_sl < sl_price and (adjusted_sl - entry_price) >= self.min_sl_pips:
                    return adjusted_sl

        return sl_price

    def _adjust_tp_for_structure(
        self,
        tp_price: float,
        direction: str,
        entry_price: float,
        smc_context: Dict,
    ) -> float:
        """Adjust TP based on next key structure level."""

        # If we have next key level, consider it as TP target
        buf = self.structure_buffer_pips
        if direction == "BUY":
            next_high = smc_context.get("next_swing_high")
            if next_high and next_high > entry_price:
                adjusted_tp = next_high - buf
                # Only use if it's reasonable
                if adjusted_tp > entry_price and adjusted_tp < tp_price * 1.2:
                    return adjusted_tp
        else:  # SELL
            next_low = smc_context.get("next_swing_low")
            if next_low and next_low < entry_price:
                adjusted_tp = next_low + buf
                # Only use if it's reasonable
                if adjusted_tp < entry_price and adjusted_tp > tp_price * 0.8:
                    return adjusted_tp

        return tp_price

    def calculate_breakeven_level(
        self,
        position: Dict,
        buffer_pips: float = 2.0,
    ) -> float:
        """
        Calculate breakeven level with buffer.

        Args:
            position: Position data
            buffer_pips: Buffer above/below entry

        Returns:
            Breakeven price
        """
        entry_price = position.get("open_price", 0)
        direction = position.get("type", "").upper()

        if direction == "BUY":
            return entry_price + buffer_pips
        else:  # SELL
            return entry_price - buffer_pips

    def _get_default_sl_tp(self, entry_price: float, direction: str) -> Dict:
        """Return conservative default SL/TP in case of error."""
        default_sl_distance = 20.0  # 20 pips
        default_tp_distance = 40.0  # 40 pips (1:2 RR)

        if direction == "BUY":
            sl = entry_price - default_sl_distance
            tp = entry_price + default_tp_distance
        else:
            sl = entry_price + default_sl_distance
            tp = entry_price - default_tp_distance

        return {
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "sl_distance_pips": default_sl_distance,
            "tp_distance_pips": default_tp_distance,
            "rr_ratio": 2.0,
            "method": "default_fallback",
        }
