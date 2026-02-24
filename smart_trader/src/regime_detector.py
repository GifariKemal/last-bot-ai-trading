"""
Regime Detector — classify current market conditions into one of 8 regimes.
Ported from xauusd_trading_bot's RegimeDetector with component scoring.
"""
import enum
import numpy as np
import pandas as pd
from loguru import logger


class MarketRegime(enum.Enum):
    STRONG_TREND_UP    = "STRONG_TREND_UP"
    STRONG_TREND_DOWN  = "STRONG_TREND_DOWN"
    WEAK_TREND_UP      = "WEAK_TREND_UP"
    WEAK_TREND_DOWN    = "WEAK_TREND_DOWN"
    RANGE_TIGHT        = "RANGE_TIGHT"
    RANGE_WIDE         = "RANGE_WIDE"
    VOLATILE_BREAKOUT  = "VOLATILE_BREAKOUT"
    REVERSAL           = "REVERSAL"

    @property
    def category(self) -> str:
        if self in (MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN,
                    MarketRegime.WEAK_TREND_UP, MarketRegime.WEAK_TREND_DOWN):
            return "trending"
        if self in (MarketRegime.RANGE_TIGHT, MarketRegime.RANGE_WIDE):
            return "ranging"
        if self == MarketRegime.VOLATILE_BREAKOUT:
            return "breakout"
        return "reversal"

    @property
    def is_trending(self) -> bool:
        return self.category == "trending"

    @property
    def is_bullish(self) -> bool:
        return self in (MarketRegime.STRONG_TREND_UP, MarketRegime.WEAK_TREND_UP)

    @property
    def is_bearish(self) -> bool:
        return self in (MarketRegime.STRONG_TREND_DOWN, MarketRegime.WEAK_TREND_DOWN)

    @property
    def short_label(self) -> str:
        _labels = {
            "STRONG_TREND_UP":   "STR_UP",
            "STRONG_TREND_DOWN": "STR_DN",
            "WEAK_TREND_UP":     "WK_UP",
            "WEAK_TREND_DOWN":   "WK_DN",
            "RANGE_TIGHT":       "RNG_T",
            "RANGE_WIDE":        "RNG_W",
            "VOLATILE_BREAKOUT": "VOL_BK",
            "REVERSAL":          "REVRS",
        }
        return _labels.get(self.value, self.value[:6])


class RegimeDetector:
    """Detect market regime from H1 candle data using component scoring."""

    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    def detect(self, df_h1: pd.DataFrame) -> dict:
        """
        Classify current market regime from H1 candles.
        Returns {"regime": MarketRegime, "confidence": float, "components": dict, "short_label": str}
        """
        if len(df_h1) < self.lookback:
            return self._default_result()

        components = self._compute_components(df_h1)
        regime, confidence = self._classify(components)

        result = {
            "regime": regime,
            "confidence": confidence,
            "components": components,
            "short_label": regime.short_label,
        }
        logger.debug(
            f"Regime: {regime.short_label} ({confidence:.0%}) | "
            f"trend={components['trend_score']:.2f} | "
            f"ATR_pct={components['atr_percentile']:.0f} | "
            f"EMA_sp={components['ema_spread_pct']:.3f} | "
            f"BB_pct={components['bb_width_percentile']:.0f} | "
            f"choch={components['has_choch']}"
        )
        return result

    def _compute_components(self, df: pd.DataFrame) -> dict:
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)
        lb = min(self.lookback, n)

        # ATR
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        atr_14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
        atr_window = tr[-lb:] if len(tr) >= lb else tr
        atr_percentile = float(np.percentile(
            np.convolve(atr_window, np.ones(14) / 14, mode="valid"),
            [50],
        )[0])
        # Percentile of current ATR within lookback
        atr_series = np.convolve(tr, np.ones(14) / 14, mode="valid")
        if len(atr_series) > 1:
            atr_percentile = float(
                np.sum(atr_series[-lb:] <= atr_14) / min(lb, len(atr_series)) * 100
            )
        else:
            atr_percentile = 50.0

        # EMA20 and EMA50
        ema20 = self._ema(closes, 20)
        ema50 = self._ema(closes, 50)
        price = float(closes[-1])
        ema_spread_pct = (ema20 - ema50) / price * 100 if price > 0 else 0.0

        # Bollinger Bandwidth
        bb_period = 20
        if n >= bb_period:
            sma = np.mean(closes[-bb_period:])
            std = np.std(closes[-bb_period:])
            bb_width = (2 * std) / sma * 100 if sma > 0 else 0
            # Percentile within lookback
            bb_widths = []
            for i in range(max(0, n - lb), n - bb_period + 1):
                s = np.mean(closes[i:i + bb_period])
                d = np.std(closes[i:i + bb_period])
                bb_widths.append((2 * d) / s * 100 if s > 0 else 0)
            if bb_widths:
                bb_width_percentile = float(np.sum(np.array(bb_widths) <= bb_width) / len(bb_widths) * 100)
            else:
                bb_width_percentile = 50.0
        else:
            bb_width_percentile = 50.0

        # Trend score (-1 to +1)
        trend_score = self._trend_score(closes, ema20, ema50, price)

        # CHoCH detection (recent 10 bars)
        has_choch = self._detect_recent_choch(df, lookback=10)

        # Volatility expanding
        if len(tr) >= 28:
            recent_atr = float(np.mean(tr[-14:]))
            older_atr = float(np.mean(tr[-28:-14]))
            vol_expanding = recent_atr > older_atr * 1.2
        else:
            vol_expanding = False

        return {
            "atr_percentile": atr_percentile,
            "ema_spread_pct": ema_spread_pct,
            "bb_width_percentile": bb_width_percentile,
            "trend_score": trend_score,
            "has_choch": has_choch,
            "vol_expanding": vol_expanding,
            "atr_14": atr_14,
        }

    def _trend_score(self, closes: np.ndarray, ema20: float, ema50: float, price: float) -> float:
        """Compute trend score from -1 (strong bearish) to +1 (strong bullish)."""
        score = 0.0

        # EMA ordering: EMA20 > EMA50 = bullish
        if ema20 > ema50:
            score += 0.3
        elif ema20 < ema50:
            score -= 0.3

        # Price vs EMA50
        if price > ema50:
            score += 0.2
        elif price < ema50:
            score -= 0.2

        # EMA20 slope (last 5 bars vs 10 bars ago)
        n = len(closes)
        if n >= 25:
            ema20_now = self._ema(closes, 20)
            ema20_5ago = self._ema(closes[:-5], 20)
            slope = ema20_now - ema20_5ago
            if slope > 2.0:
                score += 0.3
            elif slope > 0.5:
                score += 0.15
            elif slope < -2.0:
                score -= 0.3
            elif slope < -0.5:
                score -= 0.15

        # Price momentum (last 5 bars)
        if n >= 6:
            mom = float(closes[-1] - closes[-6])
            if mom > 5:
                score += 0.2
            elif mom > 1:
                score += 0.1
            elif mom < -5:
                score -= 0.2
            elif mom < -1:
                score -= 0.1

        return max(-1.0, min(1.0, score))

    def _detect_recent_choch(self, df: pd.DataFrame, lookback: int = 10) -> bool:
        """Check if CHoCH occurred in the last `lookback` bars."""
        from zone_detector import detect_choch
        if len(df) < 15:
            return False
        choch_zones = detect_choch(df.tail(max(30, lookback * 3)), lookback=5)
        return len(choch_zones) > 0

    def _classify(self, c: dict) -> tuple:
        """Apply classification rules, return (MarketRegime, confidence)."""
        trend = c["trend_score"]
        atr_pct = c["atr_percentile"]
        ema_sp = abs(c["ema_spread_pct"])
        bb_pct = c["bb_width_percentile"]
        has_choch = c["has_choch"]
        vol_exp = c["vol_expanding"]

        # Rule 1: CHoCH + weak trend → REVERSAL
        if has_choch and abs(trend) < 0.5:
            return MarketRegime.REVERSAL, 0.75

        # Rule 2: High vol + expanding + wide BB → VOLATILE_BREAKOUT
        if atr_pct > 80 and vol_exp and bb_pct > 75:
            return MarketRegime.VOLATILE_BREAKOUT, 0.80

        # Rule 3: Strong trend
        if atr_pct > 50 and ema_sp > 0.15 and abs(trend) > 0.5:
            if trend > 0:
                return MarketRegime.STRONG_TREND_UP, min(0.65 + abs(trend) * 0.3, 0.95)
            else:
                return MarketRegime.STRONG_TREND_DOWN, min(0.65 + abs(trend) * 0.3, 0.95)

        # Rule 4: Weak trend
        if abs(trend) > 0.2 and ema_sp > 0.05:
            if trend > 0:
                return MarketRegime.WEAK_TREND_UP, 0.60 + abs(trend) * 0.2
            else:
                return MarketRegime.WEAK_TREND_DOWN, 0.60 + abs(trend) * 0.2

        # Rule 5: Tight range
        if abs(trend) <= 0.2 and atr_pct < 30 and bb_pct < 30:
            return MarketRegime.RANGE_TIGHT, 0.70

        # Rule 6: Default = wide range
        return MarketRegime.RANGE_WIDE, 0.55

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        """Compute EMA of the last value."""
        if len(data) < period:
            return float(data[-1]) if len(data) > 0 else 0.0
        alpha = 2 / (period + 1)
        ema_val = float(data[0])
        for v in data[1:]:
            ema_val = alpha * float(v) + (1 - alpha) * ema_val
        return ema_val

    @staticmethod
    def _default_result() -> dict:
        return {
            "regime": MarketRegime.RANGE_WIDE,
            "confidence": 0.50,
            "components": {
                "atr_percentile": 50, "ema_spread_pct": 0, "bb_width_percentile": 50,
                "trend_score": 0, "has_choch": False, "vol_expanding": False, "atr_14": 15,
            },
            "short_label": "RNG_W",
        }
