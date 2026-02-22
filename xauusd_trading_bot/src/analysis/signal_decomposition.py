"""
Signal Decomposition Analyzer
Measures which SMC signals and combinations actually predict profitable price moves.
Produces empirical signal quality tiers for the adaptive scoring engine.
"""

import json
import math
from itertools import combinations
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from ..bot_logger import get_logger


SIGNAL_NAMES = ["fvg", "ob", "liq_sweep", "bos", "choch"]
MIN_SAMPLES = 30
FORWARD_BARS = 20  # 5 hours at M15


class SignalDecompositionAnalyzer:
    """Analyze individual and combined SMC signal quality."""

    def __init__(self, config: Optional[Dict] = None):
        self.logger = get_logger()
        self.config = config or {}
        self.forward_bars = self.config.get("forward_bars", FORWARD_BARS)
        self.min_samples = self.config.get("min_samples", MIN_SAMPLES)

    def analyze(self, df: pl.DataFrame) -> Dict:
        """
        Run full signal decomposition on pre-calculated DataFrame.

        Args:
            df: DataFrame with SMC + technical indicators pre-calculated.

        Returns:
            Decomposition results with signal tiers, combo analysis, session breakdown.
        """
        self.logger.info("Starting signal decomposition analysis...")
        total_bars = len(df)
        if total_bars < 200:
            self.logger.error("Insufficient data for decomposition")
            return {"error": "Insufficient data"}

        # Extract signal presence per bar
        signals_matrix = self._build_signal_matrix(df)
        # Calculate forward MFE/MAE for each bar
        forward_stats = self._calculate_forward_stats(df)

        # Individual signal analysis
        individual = self._analyze_individual_signals(signals_matrix, forward_stats, df)

        # Combination analysis (2-signal and 3-signal combos)
        combo_2 = self._analyze_combos(signals_matrix, forward_stats, df, k=2)
        combo_3 = self._analyze_combos(signals_matrix, forward_stats, df, k=3)

        # OB sensitivity sweep
        ob_sweep = self._ob_sensitivity_sweep(df, forward_stats)

        # Build tier rankings
        tiers = self._build_tiers(individual, combo_2, combo_3)

        results = {
            "individual_signals": individual,
            "two_signal_combos": combo_2,
            "three_signal_combos": combo_3,
            "ob_sensitivity_sweep": ob_sweep,
            "tiers": tiers,
            "metadata": {
                "total_bars": total_bars,
                "analyzed_bars": total_bars - 100 - self.forward_bars,
                "forward_bars": self.forward_bars,
                "min_samples": self.min_samples,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

        self.logger.info(f"Signal decomposition complete: {len(individual)} signals analyzed")
        return results

    def _build_signal_matrix(self, df: pl.DataFrame) -> Dict[str, List[bool]]:
        """
        Build a per-bar boolean matrix of which signals are active.
        Returns dict of signal_name -> list[bool] aligned with df index.
        """
        n = len(df)
        matrix = {name: [False] * n for name in SIGNAL_NAMES}

        # FVG: actual columns are is_bullish_fvg, is_bearish_fvg
        for col in ["is_bullish_fvg", "is_bearish_fvg"]:
            if col in df.columns:
                vals = df[col].to_list()
                for i, v in enumerate(vals):
                    if v and v is not None and v is True:
                        matrix["fvg"][i] = True

        # Order Block: is_bullish_ob, is_bearish_ob
        for col in ["is_bullish_ob", "is_bearish_ob"]:
            if col in df.columns:
                vals = df[col].to_list()
                for i, v in enumerate(vals):
                    if v and v is not None and v is True:
                        matrix["ob"][i] = True

        # Liquidity sweep: swept_high_liquidity, swept_low_liquidity
        for col in ["swept_high_liquidity", "swept_low_liquidity"]:
            if col in df.columns:
                vals = df[col].to_list()
                for i, v in enumerate(vals):
                    if v and v is not None and v is True:
                        matrix["liq_sweep"][i] = True

        # BOS: is_bullish_bos, is_bearish_bos
        for col in ["is_bullish_bos", "is_bearish_bos"]:
            if col in df.columns:
                vals = df[col].to_list()
                for i, v in enumerate(vals):
                    if v and v is not None and v is True:
                        matrix["bos"][i] = True

        # CHoCH: is_bullish_choch, is_bearish_choch
        for col in ["is_bullish_choch", "is_bearish_choch"]:
            if col in df.columns:
                vals = df[col].to_list()
                for i, v in enumerate(vals):
                    if v and v is not None and v is True:
                        matrix["choch"][i] = True

        return matrix

    def _calculate_forward_stats(self, df: pl.DataFrame) -> List[Optional[Dict]]:
        """
        For each bar, calculate forward MFE/MAE over the next N bars.
        Returns list aligned with df index, None if insufficient forward data.
        """
        n = len(df)
        close_vals = df["close"].to_list()
        high_vals = df["high"].to_list()
        low_vals = df["low"].to_list()
        results = [None] * n

        for i in range(100, n - self.forward_bars):
            entry = close_vals[i]
            if entry is None or entry == 0:
                continue

            # Look forward
            future_highs = high_vals[i + 1: i + 1 + self.forward_bars]
            future_lows = low_vals[i + 1: i + 1 + self.forward_bars]

            max_high = max(h for h in future_highs if h is not None)
            min_low = min(l for l in future_lows if l is not None)

            # Bullish perspective
            mfe_up = max_high - entry
            mae_up = entry - min_low

            # Bearish perspective
            mfe_down = entry - min_low
            mae_down = max_high - entry

            # Determine best direction
            up_rr = mfe_up / mae_up if mae_up > 0 else float("inf")
            down_rr = mfe_down / mae_down if mae_down > 0 else float("inf")

            best_dir = "bullish" if up_rr >= down_rr else "bearish"
            mfe = mfe_up if best_dir == "bullish" else mfe_down
            mae = mae_up if best_dir == "bullish" else mae_down

            profitable = mfe >= 2 * mae if mae > 0 else mfe > 0

            results[i] = {
                "mfe": mfe,
                "mae": mae,
                "rr": mfe / mae if mae > 0 else 0,
                "profitable": profitable,
                "best_direction": best_dir,
                "mfe_up": mfe_up,
                "mae_up": mae_up,
                "mfe_down": mfe_down,
                "mae_down": mae_down,
            }

        return results

    def _analyze_individual_signals(
        self, matrix: Dict, forward: List, df: pl.DataFrame
    ) -> Dict:
        """Analyze each signal individually."""
        results = {}
        # Get session info if available
        times = df["time"].to_list() if "time" in df.columns else [None] * len(df)

        for signal_name in SIGNAL_NAMES:
            present_stats = []
            absent_stats = []
            session_breakdown = {"asian": [], "london": [], "new_york": [], "overlap": []}

            for i in range(100, len(df) - self.forward_bars):
                fwd = forward[i]
                if fwd is None:
                    continue

                if matrix[signal_name][i]:
                    present_stats.append(fwd)
                    # Session breakdown
                    session = self._get_session(times[i])
                    if session in session_breakdown:
                        session_breakdown[session].append(fwd)
                else:
                    absent_stats.append(fwd)

            n_present = len(present_stats)
            n_absent = len(absent_stats)

            results[signal_name] = {
                "present": self._compute_stats(present_stats) if n_present >= self.min_samples else None,
                "absent": self._compute_stats(absent_stats) if n_absent >= self.min_samples else None,
                "n_present": n_present,
                "n_absent": n_absent,
                "session_breakdown": {
                    k: self._compute_stats(v) if len(v) >= self.min_samples else None
                    for k, v in session_breakdown.items()
                },
            }

            # Statistical significance (chi-squared approximation)
            if n_present >= self.min_samples and n_absent >= self.min_samples:
                results[signal_name]["significance"] = self._chi_squared_test(
                    present_stats, absent_stats
                )

        return results

    def _analyze_combos(
        self, matrix: Dict, forward: List, df: pl.DataFrame, k: int
    ) -> Dict:
        """Analyze k-signal combinations."""
        results = {}
        combos = list(combinations(SIGNAL_NAMES, k))

        for combo in combos:
            combo_stats = []
            for i in range(100, len(df) - self.forward_bars):
                fwd = forward[i]
                if fwd is None:
                    continue
                if all(matrix[sig][i] for sig in combo):
                    combo_stats.append(fwd)

            combo_key = "+".join(combo)
            n = len(combo_stats)
            results[combo_key] = {
                "stats": self._compute_stats(combo_stats) if n >= self.min_samples else None,
                "n_samples": n,
            }

        return results

    def _ob_sensitivity_sweep(self, df: pl.DataFrame, forward: List) -> List[Dict]:
        """
        Test OB detection at different strong_move_percent thresholds.
        Returns results for each threshold.
        """
        # We can't re-calculate OB here without the detector, so we analyze
        # the existing OB presence data and note the current config
        ob_present = sum(
            1 for i in range(100, len(df) - self.forward_bars)
            if any(
                df[col][i] for col in ["is_bullish_ob", "is_bearish_ob"]
                if col in df.columns
            )
        )
        total_analyzed = len(df) - 100 - self.forward_bars

        return [{
            "note": "Run separate OB sweep with run_signal_decomposition.py --ob-sweep",
            "current_ob_rate": ob_present / total_analyzed if total_analyzed > 0 else 0,
            "current_ob_count": ob_present,
            "total_bars_analyzed": total_analyzed,
        }]

    def _compute_stats(self, stats: List[Dict]) -> Dict:
        """Compute aggregate statistics from forward stat entries."""
        if not stats:
            return {}

        n = len(stats)
        profits = [s for s in stats if s["profitable"]]
        win_rate = len(profits) / n
        avg_rr = sum(s["rr"] for s in stats) / n
        avg_mfe = sum(s["mfe"] for s in stats) / n
        avg_mae = sum(s["mae"] for s in stats) / n

        # Profit factor approximation
        total_mfe = sum(s["mfe"] for s in profits) if profits else 0
        total_mae = sum(s["mae"] for s in stats if not s["profitable"])
        pf = total_mfe / total_mae if total_mae > 0 else float("inf")

        # Bootstrap 95% CI for win rate
        ci_low, ci_high = self._bootstrap_ci([s["profitable"] for s in stats])

        return {
            "n": n,
            "win_rate": round(win_rate, 4),
            "avg_rr": round(avg_rr, 3),
            "avg_mfe": round(avg_mfe, 2),
            "avg_mae": round(avg_mae, 2),
            "profit_factor": round(min(pf, 99.0), 2),
            "win_rate_ci_95": [round(ci_low, 4), round(ci_high, 4)],
        }

    def _chi_squared_test(self, present: List[Dict], absent: List[Dict]) -> Dict:
        """Simple chi-squared test for signal vs profitability independence."""
        n_p = len(present)
        n_a = len(absent)
        p_win = sum(1 for s in present if s["profitable"])
        a_win = sum(1 for s in absent if s["profitable"])

        total = n_p + n_a
        total_win = p_win + a_win
        total_loss = total - total_win

        if total == 0 or total_win == 0 or total_loss == 0:
            return {"chi2": 0, "p_value": 1.0, "significant": False}

        # Expected frequencies
        e_p_win = n_p * total_win / total
        e_p_loss = n_p * total_loss / total
        e_a_win = n_a * total_win / total
        e_a_loss = n_a * total_loss / total

        chi2 = 0
        for obs, exp in [(p_win, e_p_win), (n_p - p_win, e_p_loss),
                         (a_win, e_a_win), (n_a - a_win, e_a_loss)]:
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp

        # Approximate p-value for 1 df (chi-squared critical: 3.84 for p<0.05)
        significant = chi2 > 3.84

        return {
            "chi2": round(chi2, 3),
            "significant_at_005": significant,
        }

    def _bootstrap_ci(self, values: List[bool], n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Bootstrap 95% confidence interval for proportion."""
        if not values:
            return (0.0, 0.0)

        arr = np.array(values, dtype=float)
        means = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            sample = np.random.choice(arr, size=len(arr), replace=True)
            means[b] = sample.mean()

        return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

    def _get_session(self, ts) -> str:
        """Determine trading session from timestamp."""
        if ts is None:
            return "unknown"
        try:
            hour = ts.hour if hasattr(ts, "hour") else 0
        except Exception:
            return "unknown"

        # UTC sessions
        if 13 <= hour < 17:
            return "overlap"
        elif 8 <= hour < 13:
            return "london"
        elif 13 <= hour < 22:
            return "new_york"
        else:
            return "asian"

    def _build_tiers(self, individual: Dict, combo_2: Dict, combo_3: Dict) -> Dict:
        """
        Build signal quality tiers from analysis results.
        Tier 1 = best signals (multiplier 1.2x)
        Tier 2 = average (1.0x)
        Tier 3 = weak (0.5x)
        """
        ranked = []

        for sig_name, data in individual.items():
            if data.get("present") and data["present"].get("win_rate"):
                ranked.append({
                    "signal": sig_name,
                    "win_rate": data["present"]["win_rate"],
                    "profit_factor": data["present"]["profit_factor"],
                    "n": data["present"]["n"],
                    "significant": data.get("significance", {}).get("significant_at_005", False),
                })

        # Sort by composite score (PF * WR)
        for r in ranked:
            r["composite"] = r["profit_factor"] * r["win_rate"]
        ranked.sort(key=lambda x: x["composite"], reverse=True)

        # Assign tiers
        tiers = {"tier_1": [], "tier_2": [], "tier_3": []}
        for i, r in enumerate(ranked):
            if r["composite"] > 0.8 and r["significant"]:
                tiers["tier_1"].append({"signal": r["signal"], "multiplier": 1.2, **r})
            elif r["composite"] > 0.4:
                tiers["tier_2"].append({"signal": r["signal"], "multiplier": 1.0, **r})
            else:
                tiers["tier_3"].append({"signal": r["signal"], "multiplier": 0.5, **r})

        # Add best combos to tier 1 if they outperform individuals
        for combo_key, data in {**combo_2, **combo_3}.items():
            if data.get("stats") and data["stats"].get("win_rate", 0) > 0.6:
                tiers["tier_1"].append({
                    "signal": combo_key,
                    "multiplier": 1.3,
                    "win_rate": data["stats"]["win_rate"],
                    "profit_factor": data["stats"]["profit_factor"],
                    "n": data["stats"]["n"],
                    "is_combo": True,
                })

        return tiers

    def save_results(self, results: Dict, output_dir: str = "data/signal_analysis") -> str:
        """Save decomposition results to JSON."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        filepath = out_path / "decomposition_results.json"

        # Make numpy types serializable
        def convert(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=convert)

        self.logger.info(f"Signal decomposition results saved to {filepath}")
        return str(filepath)
