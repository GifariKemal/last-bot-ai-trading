"""
Position Sizer
Calculates position size based on risk parameters and account state.
"""

from typing import Dict
from ..bot_logger import get_logger


class PositionSizer:
    """Calculate position sizes for trades."""

    def __init__(self, config: Dict):
        """
        Initialize position sizer.

        Args:
            config: Risk configuration
        """
        self.logger = get_logger()
        self.config = config

        self.sizing_config = config.get("position_sizing", {})
        self.method = self.sizing_config.get("method", "fixed")
        self.fixed_lot = self.sizing_config.get("fixed_lot", 0.01)
        self.min_lot = self.sizing_config.get("min_lot", 0.01)
        self.max_lot = self.sizing_config.get("max_lot", 0.10)

    def calculate_position_size(
        self,
        account_info: Dict,
        sl_distance_pips: float,
        market_analysis: Dict = None,
        volatility_level: str = "MEDIUM",
    ) -> Dict:
        """
        Calculate position size for trade.

        Args:
            account_info: Account information (balance, equity, margin)
            sl_distance_pips: Stop loss distance in pips
            market_analysis: Market condition analysis
            volatility_level: Current volatility level

        Returns:
            Dictionary with position size and metadata
        """
        try:
            if self.method == "fixed":
                lot_size = self._calculate_fixed_size(
                    market_analysis, volatility_level
                )
            elif self.method == "percent_balance":
                lot_size = self._calculate_percent_size(
                    account_info, sl_distance_pips
                )
            elif self.method == "kelly":
                lot_size = self._calculate_kelly_size(
                    account_info, sl_distance_pips
                )
            else:
                lot_size = self.fixed_lot

            # Apply limits
            lot_size = max(self.min_lot, min(lot_size, self.max_lot))

            result = {
                "lot_size": lot_size,
                "method": self.method,
                "min_lot": self.min_lot,
                "max_lot": self.max_lot,
            }

            self.logger.debug(f"Calculated position size: {lot_size} lots ({self.method})")

            return result

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return {
                "lot_size": self.min_lot,
                "method": "fallback",
                "error": str(e),
            }

    def _calculate_fixed_size(
        self,
        market_analysis: Dict = None,
        volatility_level: str = "MEDIUM",
    ) -> float:
        """
        Calculate fixed position size with optional adjustments.

        Args:
            market_analysis: Market condition analysis
            volatility_level: Current volatility level

        Returns:
            Lot size
        """
        base_size = self.fixed_lot

        # Reduce size in unfavorable market conditions
        if market_analysis:
            if not market_analysis.get("is_favorable", True):
                multiplier = self.config.get("risk_adjustments", {}).get(
                    "unfavorable_market_multiplier", 0.5
                )
                base_size *= multiplier
                self.logger.info("Reduced position size due to unfavorable market")

        # Reduce size in high volatility
        if volatility_level == "HIGH":
            base_size *= 0.8
            self.logger.info("Reduced position size due to high volatility")

        return base_size

    def _calculate_percent_size(
        self,
        account_info: Dict,
        sl_distance_pips: float,
    ) -> float:
        """
        Calculate position size based on % of balance to risk.

        Args:
            account_info: Account information
            sl_distance_pips: SL distance in pips

        Returns:
            Lot size
        """
        # Risk % of balance
        risk_percent = self.sizing_config.get("percent_per_trade", 1.0)
        balance = account_info.get("balance", 0)

        if balance <= 0 or sl_distance_pips <= 0:
            return self.min_lot

        # Amount to risk
        risk_amount = balance * (risk_percent / 100)

        # For XAUUSD: contract size = 100 troy oz per lot
        # P&L formula: price_move × lot_size × contract_size
        # Therefore: lot_size = risk_amount / (sl_distance_dollars × contract_size)
        # sl_distance_pips is in dollar terms (e.g., 3×ATR = $26)
        # Example: $104.96 / ($26 × 100) = 0.04 lots → risk = $26 × 0.04 × 100 = $104 ✓
        contract_size = 100.0  # XAUUSD: 100 oz per standard lot

        # Calculate lot size
        lot_size = risk_amount / (sl_distance_pips * contract_size)

        return lot_size

    def _calculate_kelly_size(
        self,
        account_info: Dict,
        sl_distance_pips: float,
    ) -> float:
        """
        Calculate position size using Kelly Criterion.

        Kelly % = (Win% * Avg Win / Avg Loss) - (1 - Win%)

        For now, uses conservative Kelly fraction.

        Args:
            account_info: Account information
            sl_distance_pips: SL distance

        Returns:
            Lot size
        """
        # This requires tracking of win rate and avg win/loss
        # For now, use conservative fixed fraction
        kelly_fraction = self.sizing_config.get("kelly_fraction", 0.25)

        # Would calculate full Kelly here with historical data
        # For now, just apply fraction to balance-based sizing
        base_size = self._calculate_percent_size(account_info, sl_distance_pips)

        return base_size * kelly_fraction

    def check_position_limits(
        self,
        new_lot_size: float,
        current_positions: list,
        direction: str,
    ) -> Dict:
        """
        Check if new position violates limits.

        Args:
            new_lot_size: Proposed lot size
            current_positions: List of open positions
            direction: "BUY" or "SELL"

        Returns:
            Dictionary with allowed status and reason
        """
        limits = self.config.get("position_limits", {})
        max_positions = limits.get("max_open_positions", 3)
        max_per_direction = limits.get("max_positions_per_direction", 2)
        max_total_lots = limits.get("max_total_lots", 0.05)

        # Check max positions
        if len(current_positions) >= max_positions:
            return {
                "allowed": False,
                "reason": f"Max open positions ({max_positions}) reached",
            }

        # Check max per direction
        same_direction = [
            p for p in current_positions
            if p.get("type", "").upper() == direction.upper()
        ]
        if len(same_direction) >= max_per_direction:
            return {
                "allowed": False,
                "reason": f"Max positions in {direction} ({max_per_direction}) reached",
            }

        # Check total exposure
        current_total_lots = sum(p.get("volume", 0) for p in current_positions)
        if current_total_lots + new_lot_size > max_total_lots:
            return {
                "allowed": False,
                "reason": f"Total exposure would exceed {max_total_lots} lots",
            }

        return {
            "allowed": True,
            "reason": "Position limits OK",
            "current_positions": len(current_positions),
            "current_lots": current_total_lots,
        }

    def calculate_margin_required(
        self,
        lot_size: float,
        current_price: float,
        leverage: int = 100,
    ) -> float:
        """
        Calculate margin required for position.

        Args:
            lot_size: Position size in lots
            current_price: Current price
            leverage: Account leverage

        Returns:
            Margin required in account currency
        """
        # For XAUUSD: 1 lot = 100 oz
        contract_size = 100
        position_value = lot_size * contract_size * current_price
        margin_required = position_value / leverage

        return margin_required

    def check_margin_available(
        self,
        account_info: Dict,
        margin_required: float,
    ) -> Dict:
        """
        Check if sufficient margin is available.

        Args:
            account_info: Account information
            margin_required: Required margin

        Returns:
            Dictionary with availability status
        """
        margin_free = account_info.get("margin_free", 0)
        margin_level = account_info.get("margin_level", 0)

        # Minimum margin level from config
        min_margin_level = self.config.get("account_protection", {}).get(
            "min_margin_level", 200.0
        )

        if margin_free < margin_required:
            return {
                "available": False,
                "reason": "Insufficient free margin",
                "margin_free": margin_free,
                "margin_required": margin_required,
            }

        if margin_level > 0 and margin_level < min_margin_level:
            return {
                "available": False,
                "reason": f"Margin level below minimum ({min_margin_level}%)",
                "margin_level": margin_level,
            }

        return {
            "available": True,
            "margin_free": margin_free,
            "margin_required": margin_required,
            "margin_level": margin_level,
        }
