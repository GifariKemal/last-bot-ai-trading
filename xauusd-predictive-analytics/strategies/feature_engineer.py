"""
Feature engineering pipeline — regime-aware edition.

All technical indicators are computed via TA-Lib (C-extension, fast & accurate).
pandas_ta is used only for H1/H4 EMA resampling.

Candle pattern values use np.sign() normalisation:
  +1 = bullish pattern detected
  -1 = bearish pattern detected
   0 = no pattern

Features generated (49 total)
------------------------------
Momentum:
  rsi_14           – RSI(14)
  macd             – MACD line  (EMA12 − EMA26)
  macd_signal      – MACD signal (EMA9 of MACD line)
  macd_hist        – MACD histogram (macd − signal)
  stoch_k          – Stochastic %K (5,3,3)
  stoch_d          – Stochastic %D (signal of %K)
  cci_20           – Commodity Channel Index (20) — effective on Gold

Trend:
  close_vs_ema50   – (close − EMA50)  / EMA50   normalised distance
  close_vs_ema200  – (close − EMA200) / EMA200  long-term mean distance
  h1_ema50_bias    – +1 / -1  M15 close vs H1 EMA(50)
  h4_ema50_bias    – +1 / -1  M15 close vs H4 EMA(50)  ← new HTF layer

Regime:
  adx_14           – ADX(14): trend strength (>25 = trending, <20 = ranging)
  di_diff          – (+DI − −DI) / 100: directional bias within ADX context
  atr_regime       – ATR(14) / rolling-50-bar ATR mean: volatility regime
                     >1 = expanding volatility, <1 = contracting

Volatility:
  atr_14           – Average True Range (14)
  bb_pct           – Bollinger Band %B
  bb_width         – Bollinger Band width normalised
  vol_ratio        – tick_volume / rolling-20-bar mean
  return_std_5     – rolling std of 5-bar log-returns
  return_std_20    – rolling std of 20-bar log-returns

Candle structure:
  body_ratio       – |body| / candle_range
  upper_wick_ratio – upper shadow / candle_range
  lower_wick_ratio – lower shadow / candle_range

Candle patterns (TA-Lib, sign-normalised to -1 / 0 / +1):
  cdl_hammer, cdl_engulfing, cdl_doji, cdl_morning_star,
  cdl_evening_star, cdl_shooting_star, cdl_inv_hammer,
  cdl_harami, cdl_marubozu, cdl_piercing,
  cdl_dragonfly, cdl_gravestone

Time / calendar:
  day_of_week      – 0=Mon … 4=Fri (Gold reacts differently each day)
  hour_sin         – sin(2π × UTC_hour / 24)  cyclical hour encoding
  hour_cos         – cos(2π × UTC_hour / 24)

Session (UTC):
  is_asian         – 00:00–09:00
  is_london        – 07:00–16:00
  is_new_york      – 12:00–21:00
  is_overlap       – 12:00–16:00  (London + NY overlap)

Session High / Low proximity (ATR-normalised):
  dist_to_high_8h  – (rolling-32-bar high − close) / ATR  ← how far below 8-hour peak
  dist_to_low_8h   – (close − rolling-32-bar low)  / ATR  ← how far above 8-hour trough
  dist_to_high_24h – (rolling-96-bar high − close) / ATR  ← daily range upper distance
  dist_to_low_24h  – (close − rolling-96-bar low)  / ATR  ← daily range lower distance

Multi-Timeframe (H1 & H4) — "eyes of the smart money":
  h1_rsi      – RSI(14) on H1 candles  ← oversold/overbought at structure TF
  h1_adx      – ADX(14) on H1          ← trend strength at H1 (>25 = trending)
  h1_bb_width – Bollinger Band width on H1  ← volatility squeeze detection
  h1_bb_pos   – BB %B on H1  ← price position within H1 range (0=low, 1=high)
  h4_rsi      – RSI(14) on H4 candles  ← macro momentum context
  h4_adx      – ADX(14) on H4          ← macro trend strength
  h4_bb_width – Bollinger Band width on H4
  h4_bb_pos   – BB %B on H4

  All HTF features use shift(1)+ffill — leakage-proof by design.

Lagged log-returns:
  return_1  return_3  return_5

Classification label
--------------------
target = 1  → close[t+1] > close[t]
target = 0  → close[t+1] ≤ close[t]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta   # used only for HTF EMA resampling
import talib             # C-extension: all indicator computation
from loguru import logger


class FeatureEngineer:
    """
    Stateless feature-engineering and labelling helper.
    All methods return a new DataFrame; original data is never mutated.
    """

    RSI_PERIOD: int  = 14
    EMA_FAST: int    = 50
    EMA_SLOW: int    = 200
    ATR_PERIOD: int  = 14
    ADX_PERIOD: int  = 14
    BB_PERIOD: int   = 20
    BB_STD: float    = 2.0
    CCI_PERIOD: int  = 20

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all 45 feature columns to the OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV data: open, high, low, close, tick_volume.
            Index must be a timezone-aware UTC DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Original columns + 45 feature columns appended.
        """
        logger.info("Building regime-aware feature set (TA-Lib) …")
        out = df.copy()

        # Raw numpy arrays — TA-Lib operates on float64
        o = out["open"].to_numpy(dtype=float)
        h = out["high"].to_numpy(dtype=float)
        l = out["low"].to_numpy(dtype=float)
        c = out["close"].to_numpy(dtype=float)

        # ── 1. Momentum ───────────────────────────────────────────────────────
        out["rsi_14"] = talib.RSI(c, timeperiod=self.RSI_PERIOD)

        macd, macd_sig, macd_hist = talib.MACD(
            c, fastperiod=12, slowperiod=26, signalperiod=9
        )
        out["macd"]        = macd
        out["macd_signal"] = macd_sig
        out["macd_hist"]   = macd_hist

        stoch_k, stoch_d = talib.STOCH(
            h, l, c,
            fastk_period=5,
            slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0,
        )
        out["stoch_k"] = stoch_k
        out["stoch_d"] = stoch_d

        out["cci_20"] = talib.CCI(h, l, c, timeperiod=self.CCI_PERIOD)

        # ── 2. Trend ──────────────────────────────────────────────────────────
        ema50  = talib.EMA(c, timeperiod=self.EMA_FAST)
        ema200 = talib.EMA(c, timeperiod=self.EMA_SLOW)

        out["close_vs_ema50"]  = (c - ema50)  / ema50
        out["close_vs_ema200"] = (c - ema200) / ema200
        out["h1_ema50_bias"]   = self._htf_ema_bias(out, "1h",  50)
        out["h4_ema50_bias"]   = self._htf_ema_bias(out, "4h",  50)

        # ── 3. Regime ─────────────────────────────────────────────────────────
        out["adx_14"] = talib.ADX(h, l, c, timeperiod=self.ADX_PERIOD)

        plus_di  = talib.PLUS_DI(h, l, c,  timeperiod=self.ADX_PERIOD)
        minus_di = talib.MINUS_DI(h, l, c, timeperiod=self.ADX_PERIOD)
        out["di_diff"] = (plus_di - minus_di) / 100.0   # normalise to [-1, +1]

        atr_raw = talib.ATR(h, l, c, timeperiod=self.ATR_PERIOD)
        out["atr_14"]     = atr_raw
        atr_series        = pd.Series(atr_raw, index=out.index)
        out["atr_regime"] = atr_series / atr_series.rolling(50).mean()

        # ── 3b. Session High / Low proximity (ATR-normalised) ─────────────────
        # 32 bars × 15 min = 8 h  ≈ one trading session
        # 96 bars × 15 min = 24 h ≈ one full trading day
        _high_s = pd.Series(h, index=out.index)
        _low_s  = pd.Series(l, index=out.index)
        _atr_s  = atr_series.replace(0, np.nan)   # guard /0

        out["dist_to_high_8h"]  = (_high_s.rolling(32).max()  - out["close"]) / _atr_s
        out["dist_to_low_8h"]   = (out["close"] - _low_s.rolling(32).min())   / _atr_s
        out["dist_to_high_24h"] = (_high_s.rolling(96).max()  - out["close"]) / _atr_s
        out["dist_to_low_24h"]  = (out["close"] - _low_s.rolling(96).min())   / _atr_s

        # ── 3c. Multi-Timeframe features (H1 & H4) ───────────────────────────
        # Build a clean OHLCV frame for resampling (uses original numpy arrays)
        raw_frame = pd.DataFrame(
            {"open": o, "high": h, "low": l, "close": c,
             "tick_volume": out["tick_volume"].to_numpy()},
            index=out.index,
        )
        out = self._add_htf_features(out, raw_frame, "1h", "h1")
        out = self._add_htf_features(out, raw_frame, "4h", "h4")

        # ── 4. Volatility ─────────────────────────────────────────────────────
        bb_upper, bb_mid, bb_lower = talib.BBANDS(
            c,
            timeperiod=self.BB_PERIOD,
            nbdevup=self.BB_STD,
            nbdevdn=self.BB_STD,
            matype=0,
        )
        bb_range        = np.where(bb_upper - bb_lower != 0, bb_upper - bb_lower, np.nan)
        out["bb_pct"]   = (c - bb_lower) / bb_range
        out["bb_width"] = (bb_upper - bb_lower) / bb_mid

        out["vol_ratio"] = (
            out["tick_volume"] / out["tick_volume"].rolling(20).mean()
        )
        log_ret              = np.log(out["close"] / out["close"].shift(1))
        out["return_std_5"]  = log_ret.rolling(5).std()
        out["return_std_20"] = log_ret.rolling(20).std()

        # ── 5. Candle structure ───────────────────────────────────────────────
        candle_range            = (out["high"] - out["low"]).replace(0, np.nan)
        out["body_ratio"]       = (out["close"] - out["open"]).abs() / candle_range
        out["upper_wick_ratio"] = (
            out["high"] - out[["open", "close"]].max(axis=1)
        ) / candle_range
        out["lower_wick_ratio"] = (
            out[["open", "close"]].min(axis=1) - out["low"]
        ) / candle_range

        # ── 6. Candle patterns (TA-Lib, sign-normalised) ──────────────────────
        patterns: dict[str, np.ndarray] = {
            "cdl_hammer":        talib.CDLHAMMER(o, h, l, c),
            "cdl_engulfing":     talib.CDLENGULFING(o, h, l, c),
            "cdl_doji":          talib.CDLDOJI(o, h, l, c),
            "cdl_morning_star":  talib.CDLMORNINGSTAR(o, h, l, c),
            "cdl_evening_star":  talib.CDLEVENINGSTAR(o, h, l, c),
            "cdl_shooting_star": talib.CDLSHOOTINGSTAR(o, h, l, c),
            "cdl_inv_hammer":    talib.CDLINVERTEDHAMMER(o, h, l, c),
            "cdl_harami":        talib.CDLHARAMI(o, h, l, c),
            "cdl_marubozu":      talib.CDLMARUBOZU(o, h, l, c),
            "cdl_piercing":      talib.CDLPIERCING(o, h, l, c),
            "cdl_dragonfly":     talib.CDLDRAGONFLYDOJI(o, h, l, c),
            "cdl_gravestone":    talib.CDLGRAVESTONEDOJI(o, h, l, c),
        }
        for col, arr in patterns.items():
            out[col] = np.sign(arr).astype(int)

        hit_summary = ", ".join(
            f"{col.replace('cdl_', '')}={int((arr != 0).sum())}"
            for col, arr in patterns.items()
            if (arr != 0).sum() > 0
        )
        logger.info(f"  Pattern hits → {hit_summary}")

        # ── 7. Time / calendar ────────────────────────────────────────────────
        hour = out.index.hour
        out["day_of_week"] = out.index.dayofweek          # 0=Mon … 4=Fri
        out["hour_sin"]    = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"]    = np.cos(2 * np.pi * hour / 24)

        # ── 8. Session flags (UTC) ────────────────────────────────────────────
        out["is_asian"]    = ((hour >= 0)  & (hour < 9)).astype(int)
        out["is_london"]   = ((hour >= 7)  & (hour < 16)).astype(int)
        out["is_new_york"] = ((hour >= 12) & (hour < 21)).astype(int)
        out["is_overlap"]  = ((hour >= 12) & (hour < 16)).astype(int)

        # ── 9. Lagged log-returns ─────────────────────────────────────────────
        out["return_1"] = log_ret
        out["return_3"] = np.log(out["close"] / out["close"].shift(3))
        out["return_5"] = np.log(out["close"] / out["close"].shift(5))

        logger.success(
            f"Regime-aware + MTF features built. "
            f"DataFrame shape: {out.shape} | Feature columns: 57 (49 M15 + 8 HTF)"
        )
        return out

    def create_labels(
        self,
        df: pd.DataFrame,
        atr_threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        Append binary target column and drop NaN / neutral rows.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-enriched DataFrame; must contain 'close' and 'atr_14'.
        atr_threshold : float
            Minimum move size expressed as a multiple of ATR(14).
            Moves within ±(atr_threshold × ATR) are treated as noise and
            dropped (target = NaN).  Set to 0.0 to disable filtering.

        Labelling logic
        ---------------
        atr_threshold == 0  (legacy):
            target = 1  if close[t+1] > close[t]
            target = 0  otherwise
        atr_threshold > 0  (noise-filtered):
            target = 1  if move > +threshold × ATR
            target = 0  if move < −threshold × ATR
            dropped     if |move| ≤ threshold × ATR   (market noise)
        """
        logger.info(
            f"Creating classification labels … "
            f"(atr_threshold={atr_threshold:.2f})"
        )
        out = df.copy()
        future_close = out["close"].shift(-1)
        move         = future_close - out["close"]

        if atr_threshold > 0.0:
            noise_floor   = atr_threshold * out["atr_14"]
            out["target"] = np.where(
                move > noise_floor,  1.0,
                np.where(move < -noise_floor, 0.0, np.nan),
            )
        else:
            out["target"] = (future_close > out["close"]).astype(float)

        # Drop last bar (no future close) + NaN rows (neutral zone + warmup)
        pre_drop = len(out)
        out = out.iloc[:-1].dropna(subset=["target"])
        out["target"] = out["target"].astype(int)

        counts    = out["target"].value_counts()
        dropped   = pre_drop - len(out)
        noise_pct = dropped / pre_drop * 100
        logger.success(
            f"Labels created. "
            f"Total rows: {len(out):,} | "
            f"BUY (1): {counts.get(1, 0):,} | "
            f"SELL (0): {counts.get(0, 0):,} | "
            f"Noise dropped: {dropped:,} ({noise_pct:.1f}%)"
        )
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _add_htf_features(
        df: pd.DataFrame,
        raw: pd.DataFrame,
        resample_freq: str,
        prefix: str,
    ) -> pd.DataFrame:
        """
        Compute RSI(14), ADX(14), BB_width, BB_pos on HTF candles
        and forward-fill back to the base timeframe.

        shift(1) is applied before reindex so the HTF bar that closes
        at time t is only visible to base-TF bars AFTER t — leakage-proof.

        Parameters
        ----------
        df : pd.DataFrame
            Base-timeframe frame to augment.
        raw : pd.DataFrame
            OHLCV frame with same index as df (used for resampling).
        resample_freq : str
            Pandas offset alias, e.g. "1h", "4h".
        prefix : str
            Column prefix, e.g. "h1", "h4".
        """
        htf = raw.resample(resample_freq).agg({
            "open": "first", "high": "max",
            "low": "min",   "close": "last",
            "tick_volume": "sum",
        }).dropna()

        c = htf["close"].to_numpy(dtype=float)
        h = htf["high"].to_numpy(dtype=float)
        l = htf["low"].to_numpy(dtype=float)

        rsi = talib.RSI(c, timeperiod=14)
        adx = talib.ADX(h, l, c, timeperiod=14)

        upper, mid, lower = talib.BBANDS(
            c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
        )
        bb_range = np.where((upper - lower) != 0, upper - lower, np.nan)
        bb_width = (upper - lower) / np.where(mid != 0, mid, np.nan)
        bb_pos   = (c - lower) / bb_range

        feats = pd.DataFrame({
            f"{prefix}_rsi":      rsi,
            f"{prefix}_adx":      adx,
            f"{prefix}_bb_width": bb_width,
            f"{prefix}_bb_pos":   bb_pos,
        }, index=htf.index)

        # shift(1): HTF bar that closes at t visible only after t
        mapped = feats.shift(1).reindex(df.index, method="ffill")
        return pd.concat([df, mapped], axis=1)

    @staticmethod
    def _htf_ema_bias(
        df: pd.DataFrame,
        resample_freq: str,
        ema_length: int,
    ) -> pd.Series:
        """
        Higher-timeframe trend filter.

        Resamples M15 closes to resample_freq, computes EMA(ema_length),
        then maps +1 (close > EMA) / -1 (close < EMA) back to the M15
        index via forward-fill.

        Parameters
        ----------
        df : pd.DataFrame
            M15 frame with UTC DatetimeIndex.
        resample_freq : str
            Pandas offset alias, e.g. "1h", "4h".
        ema_length : int
            EMA period on the higher-timeframe series.
        """
        htf_close = df["close"].resample(resample_freq).last().dropna()
        htf_ema   = ta.ema(htf_close, length=ema_length)

        bias = pd.Series(
            np.where(htf_close > htf_ema, 1, -1),
            index=htf_close.index,
        )

        # CRITICAL: shift(1) so the H1/H4 bar that closes at e.g. 09:59
        # is only visible to M15 bars AFTER 10:00 — not during the bar.
        # Without this, resample('1h').last() labels the 09:00 bucket with
        # the 09:45 close, leaking future price into earlier M15 rows.
        return (
            bias
            .shift(1)                          # ← leak fix
            .reindex(df.index, method="ffill")
            .fillna(0)
            .astype(int)
        )
