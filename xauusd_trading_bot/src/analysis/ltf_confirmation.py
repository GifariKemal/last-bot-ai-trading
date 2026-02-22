"""
Lower-Timeframe (M5) Entry Confirmation
Scores M5 momentum, structure, and rejection patterns to confirm M15 signals.
"""

from typing import Dict
from datetime import timedelta

import polars as pl

from ..bot_logger import get_logger


class LTFConfirmation:
    """Calculate M5 lower-timeframe confirmation score for entry timing."""

    def __init__(self, config: Dict = None):
        self.logger = get_logger()
        config = config or {}
        ltf_cfg = config.get("mtf_analysis", {}).get("ltf_confirmation", {})
        self.lookback_bars = ltf_cfg.get("lookback_bars", 6)
        self.momentum_weight = ltf_cfg.get("momentum_weight", 0.40)
        self.structure_weight = ltf_cfg.get("structure_weight", 0.30)
        self.rejection_weight = ltf_cfg.get("rejection_weight", 0.30)

    def calculate_confirmation(
        self,
        direction: str,
        m5_df: pl.DataFrame,
        current_m15_time,
    ) -> Dict:
        """
        Calculate LTF confirmation score.

        Args:
            direction: "BULLISH" or "BEARISH" (or TrendDirection enum)
            m5_df: M5 DataFrame with technical indicators pre-calculated
            current_m15_time: The M15 bar's timestamp (candle open time)

        Returns:
            Dict with "score" (0.0-1.0) and "details"
        """
        try:
            dir_str = direction.value if hasattr(direction, "value") else str(direction)
            is_buy = dir_str.upper() in ("BULLISH", "BUY")

            # Slice M5 data: only bars up to last M5 within the current M15 candle
            # M15 candle at time T covers T to T+15min. Last closed M5 = T+10min.
            cutoff = current_m15_time + timedelta(minutes=10)
            m5_slice = m5_df.filter(pl.col("time") <= cutoff).tail(self.lookback_bars)

            if len(m5_slice) < 3:
                return {"score": 0.0, "details": {"error": "insufficient_m5_data"}}

            # Score sub-components
            momentum = self._score_momentum(m5_slice, is_buy)
            structure = self._score_structure(m5_slice, is_buy)
            rejection = self._score_rejection(m5_slice, is_buy)

            total = (
                momentum * self.momentum_weight
                + structure * self.structure_weight
                + rejection * self.rejection_weight
            )

            return {
                "score": round(max(0.0, min(1.0, total)), 4),
                "details": {
                    "momentum": round(momentum, 4),
                    "structure": round(structure, 4),
                    "rejection": round(rejection, 4),
                },
            }

        except Exception as e:
            self.logger.debug(f"LTF confirmation error: {e}")
            return {"score": 0.0, "details": {"error": str(e)}}

    def _score_momentum(self, df: pl.DataFrame, is_buy: bool) -> float:
        """Score RSI slope + MACD histogram direction (last 3 bars)."""
        score = 0.0

        # RSI slope (weight 0.50)
        if "rsi_14" in df.columns:
            rsi_first = df["rsi_14"][-3]
            rsi_last = df["rsi_14"][-1]
            if rsi_first is not None and rsi_last is not None:
                if is_buy and rsi_last > rsi_first:
                    score += 0.50
                elif not is_buy and rsi_last < rsi_first:
                    score += 0.50

        # MACD histogram sign (weight 0.30)
        if "macd_histogram" in df.columns:
            hist = df["macd_histogram"][-1]
            if hist is not None:
                if is_buy and hist > 0:
                    score += 0.30
                elif not is_buy and hist < 0:
                    score += 0.30

            # MACD histogram slope (weight 0.20)
            hist_prev = df["macd_histogram"][-2]
            if hist is not None and hist_prev is not None:
                if is_buy and hist > hist_prev:
                    score += 0.20
                elif not is_buy and hist < hist_prev:
                    score += 0.20

        return score

    def _score_structure(self, df: pl.DataFrame, is_buy: bool) -> float:
        """Score EMA alignment + price position (latest bar)."""
        score = 0.0

        # EMA alignment (weight 0.60)
        if "ema_20" in df.columns and "ema_50" in df.columns:
            ema_20 = df["ema_20"][-1]
            ema_50 = df["ema_50"][-1]
            if ema_20 is not None and ema_50 is not None:
                if is_buy and ema_20 > ema_50:
                    score += 0.60
                elif not is_buy and ema_20 < ema_50:
                    score += 0.60

        # Price position relative to EMA20 (weight 0.40)
        if "ema_20" in df.columns:
            close = df["close"][-1]
            ema_20 = df["ema_20"][-1]
            if close is not None and ema_20 is not None:
                if is_buy and close > ema_20:
                    score += 0.40
                elif not is_buy and close < ema_20:
                    score += 0.40

        return score

    def _score_rejection(self, df: pl.DataFrame, is_buy: bool) -> float:
        """Score rejection wicks + engulfing patterns (last 3 bars)."""
        score = 0.0
        has_wick_cols = "lower_wick" in df.columns and "upper_wick" in df.columns and "body" in df.columns

        # Rejection wick (weight 0.60) - any of last 3 bars
        if has_wick_cols:
            for offset in [-1, -2, -3]:
                body_val = abs(df["body"][offset]) if df["body"][offset] is not None else 0
                if body_val == 0:
                    continue
                if is_buy:
                    lower_wick = df["lower_wick"][offset]
                    if lower_wick is not None and lower_wick > 1.5 * body_val:
                        score += 0.60
                        break
                else:
                    upper_wick = df["upper_wick"][offset]
                    if upper_wick is not None and upper_wick > 1.5 * body_val:
                        score += 0.60
                        break

        # Engulfing pattern (weight 0.40) - last bar engulfs previous
        if len(df) >= 2:
            close_curr = df["close"][-1]
            open_prev = df["open"][-2]
            close_prev = df["close"][-2]
            open_curr = df["open"][-1]

            if all(v is not None for v in [close_curr, open_prev, close_prev, open_curr]):
                if is_buy:
                    # Bullish engulfing: current close > prev open, current is bullish
                    if close_curr > open_prev and close_curr > open_curr:
                        score += 0.40
                else:
                    # Bearish engulfing: current close < prev open, current is bearish
                    if close_curr < open_prev and close_curr < open_curr:
                        score += 0.40

        return score
