"""
Structure-Based SL/TP Calculator
Hybrid ATR + structure placement using swing points and liquidity levels.
Dynamic RR targets based on market regime.
"""

from typing import Dict, Optional
from ..core.constants import MarketRegime, VolatilityLevel
from ..bot_logger import get_logger


# Session spread model (XAUUSD typical spreads in $ per unit)
SESSION_SPREADS = {
    "asian":   {"mean": 0.40, "max": 0.60},
    "london":  {"mean": 0.15, "max": 0.30},
    "new_york": {"mean": 0.20, "max": 0.35},
    "overlap": {"mean": 0.10, "max": 0.20},
    "default": {"mean": 0.25, "max": 0.40},
}

# Minimum RR targets per regime
# M15 Fast Mode: all set to 1.5 — consistent target, faster TP hit (~1h hold)
# Previously 1.5–3.0 which (combined with wide SL) forced TP 40–48pt = 2h+ hold
REGIME_MIN_RR = {
    MarketRegime.STRONG_TREND_UP: 1.5,
    MarketRegime.STRONG_TREND_DOWN: 1.5,
    MarketRegime.WEAK_TREND_UP: 1.5,    # was 1.8
    MarketRegime.WEAK_TREND_DOWN: 1.5,  # was 1.8
    MarketRegime.RANGE_TIGHT: 1.5,      # was 2.5
    MarketRegime.RANGE_WIDE: 1.5,       # was 2.0
    MarketRegime.VOLATILE_BREAKOUT: 1.5, # was 2.0
    MarketRegime.REVERSAL: 1.5,         # was 3.0
}


class StructureSLTPCalculator:
    """Calculate SL/TP using hybrid ATR + market structure."""

    def __init__(self, config: Dict):
        self.logger = get_logger()
        self.config = config

        sl_cfg = config.get("stop_loss", {})
        self.base_sl_mult = sl_cfg.get("atr_multiplier", 3.0)
        self.min_sl_pips = sl_cfg.get("min_pips", 10.0)
        self.max_sl_pips = sl_cfg.get("max_pips", 200.0)

        tp_cfg = config.get("take_profit", {})
        self.base_tp_mult = tp_cfg.get("atr_multiplier", 5.0)
        self.max_tp_pips = tp_cfg.get("max_pips", 500.0)

        # Structure SL/TP config
        struct_cfg = config.get("structure_sl_tp", {})
        self.use_structure = struct_cfg.get("enabled", True)
        self.sl_buffer_atr = struct_cfg.get("sl_buffer_atr_fraction", 0.2)
        self.prefer_wider_sl = struct_cfg.get("prefer_wider_sl", True)

        # Volatility multipliers
        vol_adj = sl_cfg.get("volatility_adjustment", {})
        self.vol_multipliers = {
            VolatilityLevel.LOW: vol_adj.get("low_volatility_multiplier", 0.8),
            VolatilityLevel.MEDIUM: vol_adj.get("medium_volatility_multiplier", 1.0),
            VolatilityLevel.HIGH: vol_adj.get("high_volatility_multiplier", 1.5),
        }

        # Configurable exit stage thresholds (defaults match backtest)
        exit_cfg = config.get("exit_stages", {})
        self.be_trigger_rr = exit_cfg.get("be_trigger_rr", 1.0)
        self.partial_close_rr = exit_cfg.get("partial_close_rr", 1.5)
        self.trail_activation_rr = exit_cfg.get("trail_activation_rr", 2.0)

    def calculate_sl_tp(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        volatility_level: VolatilityLevel,
        regime: MarketRegime = MarketRegime.RANGE_WIDE,
        swing_points: Optional[Dict] = None,
        liquidity_levels: Optional[Dict] = None,
        active_obs: Optional[Dict] = None,
        session: str = "default",
    ) -> Dict:
        """
        Calculate SL/TP with hybrid ATR + structure placement.

        Args:
            entry_price: Trade entry price
            direction: "BUY" or "SELL"
            atr: Current ATR value
            volatility_level: Current volatility level
            regime: Current market regime
            swing_points: Dict with swing_highs and swing_lows lists
            liquidity_levels: Dict with liquidity levels
            active_obs: Dict with active order blocks
            session: Current trading session name

        Returns:
            SL/TP dictionary with all metadata
        """
        try:
            vol_mult = self.vol_multipliers.get(volatility_level, 1.0)

            # --- STOP LOSS ---
            atr_sl = atr * self.base_sl_mult * vol_mult
            atr_sl = max(self.min_sl_pips, min(atr_sl, self.max_sl_pips))

            if direction == "BUY":
                atr_sl_price = entry_price - atr_sl
            else:
                atr_sl_price = entry_price + atr_sl

            sl_method = "atr_based"
            sl_price = atr_sl_price

            # Structure-based SL: use swing point if available
            if self.use_structure and swing_points:
                struct_sl = self._structure_sl(
                    entry_price, direction, atr, swing_points
                )
                if struct_sl is not None:
                    if self.prefer_wider_sl:
                        # Take the WIDER SL for safety
                        if direction == "BUY":
                            sl_price = min(sl_price, struct_sl)
                        else:
                            sl_price = max(sl_price, struct_sl)
                    else:
                        sl_price = struct_sl
                    sl_method = "hybrid_structure"

            # Enforce limits
            sl_distance = abs(entry_price - sl_price)
            if sl_distance < self.min_sl_pips:
                sl_price = entry_price - self.min_sl_pips if direction == "BUY" else entry_price + self.min_sl_pips
                sl_distance = self.min_sl_pips
            elif sl_distance > self.max_sl_pips:
                sl_price = entry_price - self.max_sl_pips if direction == "BUY" else entry_price + self.max_sl_pips
                sl_distance = self.max_sl_pips

            # --- TAKE PROFIT ---
            min_rr = REGIME_MIN_RR.get(regime, 2.0)
            min_tp_distance = sl_distance * min_rr

            atr_tp_distance = atr * self.base_tp_mult
            tp_distance = max(atr_tp_distance, min_tp_distance)
            tp_distance = min(tp_distance, self.max_tp_pips)

            if direction == "BUY":
                tp_price = entry_price + tp_distance
            else:
                tp_price = entry_price - tp_distance

            tp_method = "atr_based"

            # Structure-based TP: use liquidity level or OB if closer than ATR
            if self.use_structure:
                struct_tp = self._structure_tp(
                    entry_price, direction, liquidity_levels, active_obs
                )
                if struct_tp is not None:
                    struct_tp_dist = abs(struct_tp - entry_price)
                    # Use structure TP if it meets min RR and is reasonable
                    if struct_tp_dist >= sl_distance * min_rr:
                        tp_price = struct_tp
                        tp_distance = struct_tp_dist
                        tp_method = "hybrid_structure"

            # Spread impact check
            spread = SESSION_SPREADS.get(session, SESSION_SPREADS["default"])["mean"]
            spread_pct_of_sl = (spread / sl_distance * 100) if sl_distance > 0 else 0

            rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0

            return {
                "sl": round(sl_price, 2),
                "tp": round(tp_price, 2),
                "sl_distance_pips": round(sl_distance, 2),
                "tp_distance_pips": round(tp_distance, 2),
                "rr_ratio": round(rr_ratio, 2),
                "atr_used": atr,
                "volatility_level": volatility_level.value,
                "sl_method": sl_method,
                "tp_method": tp_method,
                "regime": regime.value,
                "min_rr_target": min_rr,
                "spread": spread,
                "spread_pct_of_sl": round(spread_pct_of_sl, 2),
                # Exit stage thresholds (configurable for optimizer)
                "be_trigger_rr": self.be_trigger_rr,
                "partial_close_rr": self.partial_close_rr,
                "trail_activation_rr": self.trail_activation_rr,
            }

        except Exception as e:
            self.logger.error(f"Error in structure SL/TP: {e}")
            return self._fallback_sl_tp(entry_price, direction)

    def _structure_sl(
        self, entry: float, direction: str, atr: float,
        swing_points: Dict
    ) -> Optional[float]:
        """
        Calculate structure-based SL from swing points.
        BUY: SL below recent swing low - buffer
        SELL: SL above recent swing high + buffer
        """
        buffer = atr * self.sl_buffer_atr

        if direction == "BUY":
            lows = swing_points.get("swing_lows", [])
            if not lows:
                return None
            # Find the most recent swing low below entry
            candidates = [s["level"] for s in lows if s["level"] < entry]
            if not candidates:
                return None
            nearest_low = max(candidates)  # Closest swing low below
            return nearest_low - buffer
        else:
            highs = swing_points.get("swing_highs", [])
            if not highs:
                return None
            candidates = [s["level"] for s in highs if s["level"] > entry]
            if not candidates:
                return None
            nearest_high = min(candidates)  # Closest swing high above
            return nearest_high + buffer

    def _structure_tp(
        self, entry: float, direction: str,
        liquidity_levels: Optional[Dict] = None,
        active_obs: Optional[Dict] = None,
    ) -> Optional[float]:
        """
        Calculate structure-based TP from liquidity levels or opposing OBs.
        BUY: nearest liquidity level or bearish OB above entry
        SELL: nearest liquidity level or bullish OB below entry
        """
        candidates = []

        # Liquidity levels
        if liquidity_levels:
            levels = liquidity_levels.get("levels", [])
            for lvl in levels:
                price = lvl.get("level") or lvl.get("price")
                if price is None:
                    continue
                if direction == "BUY" and price > entry:
                    candidates.append(price - 2.0)  # Buffer below
                elif direction == "SELL" and price < entry:
                    candidates.append(price + 2.0)  # Buffer above

        # Opposing order blocks
        if active_obs:
            obs = active_obs.get("blocks", [])
            for ob in obs:
                ob_price = ob.get("level") or ob.get("zone_start")
                if ob_price is None:
                    continue
                if direction == "BUY" and ob_price > entry:
                    candidates.append(ob_price - 2.0)
                elif direction == "SELL" and ob_price < entry:
                    candidates.append(ob_price + 2.0)

        if not candidates:
            return None

        # Return nearest target
        if direction == "BUY":
            return min(candidates)
        else:
            return max(candidates)

    def get_session_spread(self, session: str = "default") -> float:
        """Get typical spread for a session."""
        return SESSION_SPREADS.get(session, SESSION_SPREADS["default"])["mean"]

    def _fallback_sl_tp(self, entry: float, direction: str) -> Dict:
        """Conservative fallback SL/TP."""
        sl_dist = 20.0
        tp_dist = 40.0
        if direction == "BUY":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist
        return {
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "sl_distance_pips": sl_dist,
            "tp_distance_pips": tp_dist,
            "rr_ratio": 2.0,
            "sl_method": "fallback",
            "tp_method": "fallback",
            "be_trigger_rr": self.be_trigger_rr,
            "partial_close_rr": self.partial_close_rr,
            "trail_activation_rr": self.trail_activation_rr,
        }
