"""
SMC Indicators V4 Adapter
Replaces the 4 custom SMC detectors (FVG, OB, Liquidity, Structure) with
the `smartmoneyconcepts` library (joshyattridge), keeping the same public
interface as SMCIndicators (V3) so all downstream code is unchanged.

Architecture:
    V3:  OHLC → [Custom Detectors] → smc_signals dict → AdaptiveScorer → Entry
    V4:  OHLC → [SMCIndicatorsV4]  → smc_signals dict → AdaptiveScorer → Entry
                 (library-based)      (same interface)    (unchanged)

Usage:
    # In settings.yaml: use_smc_v4: true
    # In backtest_engine.py / trading_bot.py: already branched by use_smc_v4 flag

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PANDAS ↔ POLARS BOUNDARY — INTENTIONAL DESIGN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The main bot pipeline uses Polars throughout for performance. Pandas
exists HERE ONLY because the `smartmoneyconcepts` library requires it.

Boundary contract:
  IN  → Polars DataFrame  (from trading_bot / backtest_engine)
  OUT → Python dict       (smc_signals, zero pandas leaks out)

Conversion happens at ONE point only: _to_pandas(df) in calculate_all().
All pandas DataFrames (swing, bos, fvg, ob, liq) live only inside this
file and are never returned to callers. Public methods return plain dicts.

DO NOT import pandas anywhere outside this file or mt5_connector/data_manager
(those have their own MT5 ↔ Polars boundary).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd   # Required: smartmoneyconcepts library needs pandas DataFrames
import polars as pl   # Main pipeline format — all public methods accept/return Polars
from typing import Dict, Optional

from ..core.constants import TrendDirection
from ..bot_logger import get_logger

try:
    from smartmoneyconcepts import smc as _smc_lib
except ImportError as e:
    raise ImportError(
        "smartmoneyconcepts library not installed. "
        "Run: pip install smartmoneyconcepts>=0.0.26"
    ) from e


class SMCIndicatorsV4:
    """
    V4: Library-based SMC detection via smartmoneyconcepts.

    Identical public interface to SMCIndicators (V3):
      - calculate_all(df)         → Polars DataFrame (unchanged, library results cached)
      - get_bullish_signals(df, price) → same dict format as V3
      - get_bearish_signals(df, price) → same dict format as V3
      - get_trade_context(df, price)   → same format as V3

    Enhanced V4 fields added (backward-compatible, extras ignored by scorer):
      - order_block.strength    = OB Percentage (0-100, strength of OB)
      - order_block.ob_volume   = OBVolume (3-bar cumulative volume)
      - fvg.mitigated           = bool (True if FVG was already filled)
      - liquidity.swept_bars_ago = int (bars since sweep, None if not swept)
    """

    # ── Signal lookback windows (matching V3 design decisions) ───────────────

    BOS_LOOKBACK_BARS = 50      # 50 bars = 12.5h on M15 (matches backtest: sw=5 + BOS=50 → PF=4.61)
    SWEEP_LOOKBACK_BARS = 20    # same as V3: sweep_confirmation_bars=20

    def __init__(self, config: Optional[Dict] = None):
        self.logger = get_logger()

        if config is None:
            config = {}

        # Library parameters mapped from V3 config
        struct_cfg = config.get("structure", {})
        liq_cfg    = config.get("liquidity", {})

        self.swing_length   = struct_cfg.get("swing_lookback", 5)  # M15-optimized (was 10, too few swings)
        self.range_percent  = liq_cfg.get("range_percent", 0.01)   # ~1% (~$50)

        # Signal weights (same as V3 SMCIndicators defaults)
        fvg_cfg   = config.get("fair_value_gaps", {})
        ob_cfg    = config.get("order_blocks", {})
        struct_w  = config.get("structure", {})
        self._w_fvg   = fvg_cfg.get("weight", 0.20)
        self._w_ob    = ob_cfg.get("weight", 0.25)
        self._w_liq   = liq_cfg.get("weight", 0.20)
        self._w_choch = struct_w.get("choch_weight", 0.30)
        self._w_bos   = struct_w.get("bos_weight", 0.15)

        # Cache: populated once by calculate_all(), reused by get_*_signals()
        self._cache: Optional[Dict] = None
        self._cache_len: int = 0

    # ── Public interface (identical to SMCIndicators V3) ─────────────────────

    def calculate_all(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Run all library-based SMC calculations and cache results.

        Args:
            df: Polars DataFrame with OHLCV data (must have open/high/low/close)

        Returns:
            The same Polars DataFrame (unchanged — library results are cached
            internally and accessed via get_bullish/bearish_signals).
        """
        try:
            ohlc = self._to_pandas(df)

            self.logger.debug(f"V4: running library on {len(ohlc)} bars ...")

            swing   = _smc_lib.swing_highs_lows(ohlc, swing_length=self.swing_length)
            bos_df  = _smc_lib.bos_choch(ohlc, swing, close_break=True)
            fvg_df  = _smc_lib.fvg(ohlc, join_consecutive=False)
            ob_df   = _smc_lib.ob(ohlc, swing, close_mitigation=False)
            liq_df  = _smc_lib.liquidity(ohlc, swing, range_percent=self.range_percent)

            self._cache = {
                "swing":   swing,
                "bos":     bos_df,
                "fvg":     fvg_df,
                "ob":      ob_df,
                "liq":     liq_df,
            }
            self._cache_len = len(df)
            self.logger.debug("V4: library cache populated.")

        except Exception as e:
            self.logger.error(f"V4 calculate_all error: {e}")
            self._cache = None

        return df   # Return unchanged Polars df

    def get_bullish_signals(self, df_slice: pl.DataFrame, current_price: float) -> Dict:
        """
        Get bullish SMC signals. Mirrors V3 SMCIndicators.get_bullish_signals().

        Args:
            df_slice: Polars DataFrame slice (df[:i+1] in backtest)
            current_price: Current bid/ask mid price

        Returns:
            Dict with fvg/order_block/liquidity/structure/confluence_score keys
        """
        idx = len(df_slice) - 1
        return self._build_signals(idx, current_price, direction="bullish")

    def get_bearish_signals(self, df_slice: pl.DataFrame, current_price: float) -> Dict:
        """
        Get bearish SMC signals. Mirrors V3 SMCIndicators.get_bearish_signals().
        """
        idx = len(df_slice) - 1
        return self._build_signals(idx, current_price, direction="bearish")

    def get_trade_context(self, df: pl.DataFrame, current_price: float) -> Dict:
        """Get complete trade context (same structure as V3)."""
        bullish = self.get_bullish_signals(df, current_price)
        bearish = self.get_bearish_signals(df, current_price)

        if bullish["confluence_score"] > bearish["confluence_score"]:
            bias = "BULLISH"
            primary = bullish
        elif bearish["confluence_score"] > bullish["confluence_score"]:
            bias = "BEARISH"
            primary = bearish
        else:
            bias = "NEUTRAL"
            primary = None

        return {
            "bias": bias,
            "bullish_score":   bullish["confluence_score"],
            "bearish_score":   bearish["confluence_score"],
            "primary_signals": primary,
            "bullish_signals": bullish,
            "bearish_signals": bearish,
            "current_price":   current_price,
        }

    def get_comprehensive_summary(self, df: pl.DataFrame) -> Dict:
        """Minimal summary for compatibility (not used in core loop)."""
        if self._cache is None:
            return {}
        bos_df  = self._cache["bos"]
        fvg_df  = self._cache["fvg"]
        ob_df   = self._cache["ob"]
        liq_df  = self._cache["liq"]
        idx = len(df) - 1
        return {
            "library": "smartmoneyconcepts",
            "bars_processed": idx + 1,
            "bos_count":      int((bos_df["BOS"].notna()).sum()),
            "choch_count":    int((bos_df["CHOCH"].notna()).sum()),
            "fvg_count":      int((fvg_df["FVG"].notna()).sum()),
            "ob_count":       int((ob_df["OB"].notna()).sum()),
            "liq_count":      int((liq_df["Liquidity"].notna()).sum()),
        }

    def analyze(self, df: pl.DataFrame) -> Dict:
        """Alias for get_comprehensive_summary (V3 compatibility)."""
        df = self.calculate_all(df)
        return self.get_comprehensive_summary(df)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_signals(self, idx: int, current_price: float, direction: str) -> Dict:
        """
        Build the smc_signals dict for one direction at bar idx.

        direction: "bullish" or "bearish"
        """
        if self._cache is None:
            return self._empty_signals(current_price, direction)

        bos_df  = self._cache["bos"]
        fvg_df  = self._cache["fvg"]
        ob_df   = self._cache["ob"]
        liq_df  = self._cache["liq"]
        swing   = self._cache["swing"]

        is_bull = (direction == "bullish")
        bos_target  =  1.0 if is_bull else -1.0
        fvg_target  =  1.0 if is_bull else -1.0
        ob_target   =  1.0 if is_bull else -1.0
        # For bullish trade: swept bearish liquidity (lows) → liq==-1
        # For bearish trade: swept bullish liquidity (highs) → liq==1
        liq_target  = -1.0 if is_bull else  1.0

        # ── BOS / CHoCH ──────────────────────────────────────────────────────
        has_bos   = False
        has_choch = False
        start = max(0, idx - self.BOS_LOOKBACK_BARS + 1)
        end   = min(idx + 1, len(bos_df))
        if start < end:
            slice_bos = bos_df.iloc[start:end]
            has_bos   = bool((slice_bos["BOS"]   == bos_target).any())
            has_choch = bool((slice_bos["CHOCH"] == bos_target).any())

        # ── FVG in zone (unmitigated) ─────────────────────────────────────────
        fvg_in_zone = False
        nearest_fvg = None
        end_fvg = min(idx + 1, len(fvg_df))
        if end_fvg > 0:
            sf = fvg_df.iloc[:end_fvg]
            # FVG active: MitigatedIndex NaN or 0.0 (library sentinel for "no mitigation")
            fmi = sf["MitigatedIndex"]
            fvg_active = fmi.isna() | (fmi == 0.0)
            mask = (
                (sf["FVG"]    == fvg_target) &
                (sf["Bottom"] <= current_price) &
                (sf["Top"]    >= current_price) &
                fvg_active
            )
            if mask.any():
                fvg_in_zone = True
                row = sf[mask].iloc[-1]
                nearest_fvg = {"top": float(row["Top"]), "bottom": float(row["Bottom"])}

        # ── Order Block at zone (unmitigated) ─────────────────────────────────
        ob_at_zone = False
        nearest_ob = None
        end_ob = min(idx + 1, len(ob_df))
        if end_ob > 0:
            so = ob_df.iloc[:end_ob]
            # Active OB: library uses 0.0 as "no mitigation" sentinel (not NaN).
            # OB is active when MitigatedIndex is NaN, 0.0, or > idx (mitigated after now).
            mi = so["MitigatedIndex"]
            active_mask = mi.isna() | (mi == 0.0) | (mi > idx)
            mask = (
                (so["OB"]     == ob_target) &
                (so["Bottom"] <= current_price) &
                (so["Top"]    >= current_price) &
                active_mask
            )
            if mask.any():
                ob_at_zone = True
                row = so[mask].iloc[-1]
                nearest_ob = {
                    "top":      float(row["Top"]),
                    "bottom":   float(row["Bottom"]),
                    # pd.notna: library returns pandas NaN for missing values — boundary use
                    "strength": float(row["Percentage"]) if pd.notna(row["Percentage"]) else None,
                    "volume":   float(row["OBVolume"])   if pd.notna(row["OBVolume"])   else None,
                }

        # ── Liquidity swept (within last 20 bars) ────────────────────────────
        liq_swept      = False
        swept_bars_ago = None
        end_liq = min(idx + 1, len(liq_df))
        if end_liq > 0:
            sl = liq_df.iloc[:end_liq]
            mask = (
                (sl["Liquidity"] == liq_target) &
                sl["Swept"].notna()
            )
            if mask.any():
                candidates = sl[mask].copy()
                # Swept column = index of candle that swept the level
                swept_idx_series = candidates["Swept"].astype(float)
                bars_ago = idx - swept_idx_series
                recent = bars_ago[bars_ago <= self.SWEEP_LOOKBACK_BARS]
                if not recent.empty:
                    liq_swept      = True
                    swept_bars_ago = int(recent.min())

        # ── Recent swing highs/lows ───────────────────────────────────────────
        recent_swing_high = None
        recent_swing_low  = None
        end_sw = min(idx + 1, len(swing))
        if end_sw > 0:
            ss = swing.iloc[:end_sw]
            sh_rows = ss[ss["HighLow"] ==  1.0]
            sl_rows = ss[ss["HighLow"] == -1.0]
            if not sh_rows.empty:
                recent_swing_high = float(sh_rows["Level"].iloc[-1])
            if not sl_rows.empty:
                recent_swing_low  = float(sl_rows["Level"].iloc[-1])

        # ── Current market trend (from swing structure) ───────────────────────
        trend = self._derive_trend(swing, idx)

        # ── Build output dict (same keys as V3) ───────────────────────────────
        signals = {
            "fvg": {
                "in_zone":  fvg_in_zone,
                "nearest":  nearest_fvg,
                "mitigated": not fvg_in_zone,      # V4 enhanced field
            },
            "order_block": {
                "at_zone":  ob_at_zone,
                "nearest":  nearest_ob,
                "strength": nearest_ob["strength"] if nearest_ob else None,  # V4 extra
                "volume":   nearest_ob["volume"]   if nearest_ob else None,  # V4 extra
                "type":     direction,
            },
            "liquidity": {
                "swept":         liq_swept,
                "swept_bars_ago": swept_bars_ago,  # V4 extra
            },
            "structure": {
                "choch":             has_choch,
                "bos":               has_bos,
                "trend":             trend,
                "direction":         direction,
                "recent_swing_high": recent_swing_high,
                "recent_swing_low":  recent_swing_low,
            },
            "current_price": current_price,
        }

        # Confluence score (same formula as V3)
        score = 0.0
        if fvg_in_zone: score += self._w_fvg
        if ob_at_zone:  score += self._w_ob
        if liq_swept:   score += self._w_liq
        if has_choch:   score += self._w_choch
        elif has_bos:   score += self._w_bos

        signals["confluence_score"] = round(score, 4)
        return signals

    def _derive_trend(self, swing: pd.DataFrame, idx: int) -> TrendDirection:
        """
        Derive market structure trend from last 4 swing points.

        Logic: compare last 2 swing highs and last 2 swing lows.
          - HH + HL → BULLISH
          - LH + LL → BEARISH
          - Else    → NEUTRAL
        """
        end = min(idx + 1, len(swing))
        if end < 4:
            return TrendDirection.NEUTRAL

        ss = swing.iloc[:end]
        highs = ss[ss["HighLow"] ==  1.0]["Level"].values
        lows  = ss[ss["HighLow"] == -1.0]["Level"].values

        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1] > highs[-2]   # Higher High
            hl = lows[-1]  > lows[-2]    # Higher Low
            lh = highs[-1] < highs[-2]   # Lower High
            ll = lows[-1]  < lows[-2]    # Lower Low

            if hh and hl:
                return TrendDirection.BULLISH
            if lh and ll:
                return TrendDirection.BEARISH
            if hh:
                return TrendDirection.BULLISH
            if ll:
                return TrendDirection.BEARISH

        return TrendDirection.NEUTRAL

    def _to_pandas(self, df: pl.DataFrame) -> pd.DataFrame:
        """
        PANDAS BOUNDARY — single conversion point from Polars to pandas.

        Called ONLY from calculate_all(). The result (and all downstream
        pandas DataFrames from the library) stays inside this file.
        Nothing pandas ever leaves this class.
        """
        ohlc = df.to_pandas()
        # Normalize column names to lowercase
        ohlc.columns = [c.lower() for c in ohlc.columns]

        # Ensure 'volume' column exists (OB needs it)
        if "volume" not in ohlc.columns:
            # Use tick_volume if available, else synthetic 1.0
            if "tick_volume" in ohlc.columns:
                ohlc["volume"] = ohlc["tick_volume"].astype(float)
            else:
                ohlc["volume"] = 1.0

        # Ensure required columns exist
        for col in ("open", "high", "low", "close"):
            if col not in ohlc.columns:
                raise ValueError(f"V4: required column '{col}' missing from DataFrame")

        # Library needs float64
        for col in ("open", "high", "low", "close", "volume"):
            ohlc[col] = ohlc[col].astype(float)

        return ohlc

    def _empty_signals(self, current_price: float, direction: str = "bullish") -> Dict:
        """Return zero-signal dict when cache is unavailable."""
        return {
            "fvg":         {"in_zone": False, "nearest": None, "mitigated": True},
            "order_block": {"at_zone": False, "nearest": None, "strength": None, "volume": None, "type": direction},
            "liquidity":   {"swept": False, "swept_bars_ago": None},
            "structure":   {
                "choch": False, "bos": False,
                "trend": TrendDirection.NEUTRAL, "direction": direction,
                "recent_swing_high": None, "recent_swing_low": None,
            },
            "current_price":   current_price,
            "confluence_score": 0.0,
        }

    # ── V3 compatibility: structure sub-interface ─────────────────────────────

    @property
    def structure(self) -> "_V4StructureProxy":
        """
        Expose a V3-compatible .structure sub-object with get_swing_points().
        Used by backtest_engine.py: self.smc.structure.get_swing_points(df_slice, n=5)
        """
        return _V4StructureProxy(self)


class _V4StructureProxy:
    """
    Thin proxy that satisfies `smc.structure.get_swing_points(df_slice, n)`.
    Reads from SMCIndicatorsV4's swing cache to return the same dict format
    as V3's StructureDetector.get_swing_points().
    """

    def __init__(self, v4: "SMCIndicatorsV4"):
        self._v4 = v4

    def get_swing_points(self, df_slice: pl.DataFrame, n: int = 10) -> Dict:
        """
        Return recent swing highs / lows up to the current bar (len(df_slice)-1).

        Returns:
            {"swing_highs": [...], "swing_lows": [...]}
            Each entry: {"time": bar_index, "level": float, "type": "SWING_HIGH"|"SWING_LOW"}
        """
        if self._v4._cache is None:
            return {"swing_highs": [], "swing_lows": []}

        swing = self._v4._cache["swing"]
        idx   = len(df_slice) - 1
        end   = min(idx + 1, len(swing))
        if end <= 0:
            return {"swing_highs": [], "swing_lows": []}

        ss = swing.iloc[:end]

        # pd.notna: filter library NaN rows before converting to plain dicts (boundary)
        highs = [
            {"time": i, "level": float(row["Level"]), "type": "SWING_HIGH"}
            for i, row in ss[ss["HighLow"] == 1.0].tail(n).iterrows()
            if pd.notna(row["Level"])
        ]
        lows = [
            {"time": i, "level": float(row["Level"]), "type": "SWING_LOW"}
            for i, row in ss[ss["HighLow"] == -1.0].tail(n).iterrows()
            if pd.notna(row["Level"])
        ]

        return {"swing_highs": highs, "swing_lows": lows}
