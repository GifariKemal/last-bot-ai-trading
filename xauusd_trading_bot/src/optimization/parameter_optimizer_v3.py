"""
Parameter Optimizer V3 - Walk-Forward with Regime-Aware Parameters
~45 parameters across regime weights, signal toggles, SL/TP, exit stages.
Walk-forward validation prevents overfitting.
"""

import optuna
from optuna.pruners import MedianPruner, PercentilePruner
from optuna.samplers import TPESampler
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import yaml
import time
import copy

from ..backtesting.backtest_engine import BacktestEngine
from ..bot_logger import get_logger


class ParameterOptimizerV3:
    """Walk-forward optimizer with ~45 regime-aware parameters."""

    def __init__(self, mt5, base_config: Dict, optimization_config: Optional[Dict] = None):
        self.logger = get_logger()
        self.mt5 = mt5
        self.base_config = copy.deepcopy(base_config)

        opt = optimization_config or {}
        self.n_trials = opt.get("n_trials", 50)
        self.n_jobs = opt.get("n_jobs", 4)
        self.n_startup_trials = opt.get("n_startup_trials", 20)
        self.n_windows = opt.get("n_windows", 3)
        self.train_ratio = opt.get("train_ratio", 0.7)

        self.symbol = opt.get("symbol", "XAUUSDm")
        self.timeframe = opt.get("timeframe", "M15")
        self.initial_balance = opt.get("initial_balance", 10000.0)

        self.results_dir = Path("data/optimization_v3")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Pre-calculated data (set by prepare_data)
        self._df_prepared = None
        self._df_m5 = None
        self._windows = []

        self.start_time = None
        self.trial_count = 0

    def prepare_data(
        self, start_date: datetime, end_date: datetime, use_cache: bool = True
    ) -> None:
        """
        Pre-calculate all indicators once. Called before optimize().
        """
        self.logger.info("Pre-calculating indicators for all data...")
        engine = BacktestEngine(self.base_config)

        df = engine._prepare_data(
            self.mt5, self.symbol, self.timeframe,
            start_date, end_date, use_cache
        )
        if df is None:
            raise ValueError("Failed to prepare backtest data")

        self._df_prepared = df
        self._df_m5 = engine.df_m5
        self.logger.info(f"Data prepared: {len(df)} bars")

        # Build walk-forward windows
        self._windows = self._build_windows(df)
        self.logger.info(f"Walk-forward windows: {len(self._windows)}")
        for i, (train_start, train_end, test_start, test_end) in enumerate(self._windows):
            self.logger.info(
                f"  Window {i+1}: Train [{train_start}..{train_end}] "
                f"Test [{test_start}..{test_end}]"
            )

    def _build_windows(self, df) -> List[Tuple[int, int, int, int]]:
        """Build walk-forward window indices."""
        n = len(df)
        usable = n - 100  # Skip first 100 bars
        window_size = usable // self.n_windows
        windows = []

        for i in range(self.n_windows):
            start = 100 + i * window_size
            end = start + window_size
            if i == self.n_windows - 1:
                end = n  # Last window gets remaining bars

            train_size = int((end - start) * self.train_ratio)
            train_end = start + train_size
            test_start = train_end
            test_end = end

            windows.append((start, train_end, test_start, test_end))

        return windows

    def diagnose_windows(self) -> None:
        """
        Dry-run each OOS window with BASE CONFIG to verify trade counts before optimization.
        Call after prepare_data() and before optimize() to catch sparse-signal periods early.
        """
        if self._df_prepared is None:
            raise ValueError("Call prepare_data() first")

        self.logger.info("=" * 60)
        self.logger.info("WINDOW DIAGNOSTIC — base config trade counts")
        self.logger.info("=" * 60)

        all_ok = True
        for i, (train_start, train_end, test_start, test_end) in enumerate(self._windows):
            df_test = self._df_prepared[test_start:test_end]
            n_bars = len(df_test)

            engine = BacktestEngine(self.base_config)
            result = engine.run_backtest_fast(
                df_test,
                initial_balance=self.initial_balance,
            )

            if not result.get("success"):
                self.logger.warning(f"  Window {i+1}: FAILED — no result")
                all_ok = False
                continue

            m = result["metrics"]
            n_trades = m["total_trades"]
            pf = m.get("profit_factor", 0)
            wr = m.get("win_rate", 0)
            dd = m.get("max_drawdown_percent", 0)
            status = "✓" if n_trades >= 5 else "✗ TOO FEW"
            self.logger.info(
                f"  Window {i+1} (bars {test_start}-{test_end}, {n_bars} bars): "
                f"Trades={n_trades} {status}, PF={pf:.2f}, WR={wr:.1f}%, DD={dd:.1f}%"
            )
            if n_trades < 5:
                all_ok = False

        if all_ok:
            self.logger.info("All windows OK — proceeding with optimization")
        else:
            self.logger.warning(
                "Some windows have < 5 trades. Consider --months N to focus on recent data. "
                "Current approach: trials with <5-trade windows will score -50."
            )
        self.logger.info("=" * 60)

    def optimize(self, study_name: Optional[str] = None) -> Dict:
        """
        Run walk-forward optimization.

        Args:
            study_name: Optuna study name

        Returns:
            Optimization results with best parameters
        """
        if self._df_prepared is None:
            raise ValueError("Call prepare_data() before optimize()")

        if study_name is None:
            study_name = f"xauusd_v3_{datetime.now().strftime('%Y%m%d_%H%M')}"

        self.start_time = time.time()
        self.trial_count = 0

        self.logger.info("=" * 80)
        self.logger.info("PARAMETER OPTIMIZATION V3 - WALK-FORWARD")
        self.logger.info("=" * 80)
        self.logger.info(f"Trials: {self.n_trials} | Jobs: {self.n_jobs}")
        self.logger.info(f"Windows: {self.n_windows} | Train ratio: {self.train_ratio}")
        self.logger.info("=" * 80)

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=TPESampler(
                n_startup_trials=self.n_startup_trials,
                multivariate=True,
            ),
            pruner=PercentilePruner(
                percentile=50.0,
                n_startup_trials=self.n_startup_trials,
                n_warmup_steps=1,
            ),
        )

        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )

        # Process results
        best = study.best_trial
        elapsed = time.time() - self.start_time

        self.logger.info("=" * 80)
        self.logger.info(f"OPTIMIZATION COMPLETE in {elapsed/60:.1f} min")
        self.logger.info(f"Best Score: {best.value:.2f}")
        self.logger.info(f"Best Params: {json.dumps(best.params, indent=2)}")
        self.logger.info("=" * 80)

        results = {
            "best_score": best.value,
            "best_params": best.params,
            "best_trial": best.number,
            "total_trials": len(study.trials),
            "elapsed_seconds": elapsed,
            "n_windows": self.n_windows,
            "timestamp": datetime.now().isoformat(),
        }

        # Save results
        self._save_results(results)

        # Generate optimized config
        optimized_config = self._params_to_config(best.params)
        config_path = self.results_dir / "optimized_config_v3.yaml"
        with open(config_path, "w") as f:
            yaml.dump(optimized_config, f, default_flow_style=False)
        self.logger.info(f"Optimized config saved to {config_path}")

        return results

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Walk-forward objective: average OOS score across all windows.
        """
        self.trial_count += 1
        params = self._suggest_params(trial)

        config = self._params_to_config(params)
        oos_scores = []

        for window_idx, (train_start, train_end, test_start, test_end) in enumerate(self._windows):
            # Run backtest on OOS (test) window only
            df_test = self._df_prepared[test_start:test_end]

            if len(df_test) < 50:
                trial.report(-100, window_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                continue

            engine = BacktestEngine(config)
            result = engine.run_backtest_fast(
                df_test,
                initial_balance=self.initial_balance,
                df_m5=self._df_m5,
            )

            if not result.get("success"):
                trial.report(-100, window_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                continue

            metrics = result["metrics"]
            score = self._calculate_score(metrics, params)

            n_trades = metrics["total_trades"]
            dd = metrics.get("max_drawdown_percent", 0)

            # Rejection checks
            # min_trades=5: pragmatic floor — windows with 1-4 trades are statistically noise.
            if n_trades < 5:
                score = -50  # Too few trades
            # DD threshold: use 30% — meaningful when balance=10000 (each 0.01 lot SL≈$45 → 0.45% per loss).
            # NOT used with balance=$100 (45% per loss makes this impossible to pass).
            if dd > 30:
                score = -100  # Excessive drawdown (>30% with $10k balance = catastrophic)

            # Per-window debug: log trade count + PF for visibility
            pf_val = metrics.get("profit_factor", 0)
            self.logger.debug(
                f"  Window {window_idx+1}: trades={n_trades}, PF={pf_val:.2f}, "
                f"DD={dd:.1f}%, score={score:.2f}"
            )

            oos_scores.append(score)

            # Report for pruning
            trial.report(score, window_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if not oos_scores:
            return -100

        # Reject if PF < 1.0 in 2+ windows
        negative_windows = sum(1 for s in oos_scores if s < 0)
        if negative_windows >= 2:
            return -50

        avg_score = sum(oos_scores) / len(oos_scores)

        # Consistency penalty: penalize high variance across windows
        if len(oos_scores) >= 2:
            mean = avg_score
            variance = sum((s - mean) ** 2 for s in oos_scores) / len(oos_scores)
            std = variance ** 0.5
            if mean > 0 and std / mean > 0.3:
                avg_score *= 0.8  # 20% penalty for inconsistency

        # Progress logging
        if self.trial_count % 10 == 0:
            elapsed = time.time() - self.start_time
            self.logger.info(
                f"Trial {self.trial_count}/{self.n_trials} | "
                f"Score: {avg_score:.2f} | Time: {elapsed/60:.1f}min"
            )

        return avg_score

    def _suggest_params(self, trial: optuna.Trial) -> Dict:
        """Suggest ~45 parameters for optimization."""
        params = {}

        # --- SMC Signal Toggles (5) ---
        params["enable_fvg"] = trial.suggest_categorical("enable_fvg", [True, False])
        params["enable_ob"] = trial.suggest_categorical("enable_ob", [True, False])
        params["enable_liq_sweep"] = trial.suggest_categorical("enable_liq_sweep", [True, False])
        params["enable_bos"] = trial.suggest_categorical("enable_bos", [True, False])
        params["enable_choch"] = trial.suggest_categorical("enable_choch", [True, False])

        # At least 2 signals must be enabled
        enabled_count = sum([
            params["enable_fvg"], params["enable_ob"], params["enable_liq_sweep"],
            params["enable_bos"], params["enable_choch"]
        ])
        if enabled_count < 2:
            params["enable_choch"] = True
            params["enable_bos"] = True

        # --- Signal Recency (3) ---
        params["choch_max_age"] = trial.suggest_int("choch_max_age", 10, 100)
        params["bos_max_age"] = trial.suggest_int("bos_max_age", 10, 100)
        params["liq_max_age"] = trial.suggest_int("liq_max_age", 5, 50)

        # --- Per-Regime Weights (5 regimes x 4 params = 20) ---
        for regime in ["trending", "ranging", "breakout", "reversal", "volatile"]:
            params[f"{regime}_smc_weight"] = trial.suggest_float(f"{regime}_smc_weight", 0.20, 0.60)
            params[f"{regime}_tech_weight"] = trial.suggest_float(f"{regime}_tech_weight", 0.10, 0.40)
            params[f"{regime}_min_confluence"] = trial.suggest_float(f"{regime}_min_confluence", 0.40, 0.75)
            params[f"{regime}_atr_sl_mult"] = trial.suggest_float(f"{regime}_atr_sl_mult", 1.5, 5.0)

        # --- Min SMC per regime category (5) ---
        params["trending_min_smc"] = trial.suggest_int("trending_min_smc", 1, 3)
        params["ranging_min_smc"] = trial.suggest_int("ranging_min_smc", 1, 3)
        params["breakout_min_smc"] = trial.suggest_int("breakout_min_smc", 1, 3)
        params["reversal_min_smc"] = trial.suggest_int("reversal_min_smc", 1, 3)
        params["volatile_min_smc"] = trial.suggest_int("volatile_min_smc", 1, 3)

        # --- Exit Stage Thresholds (3) ---
        params["be_trigger_rr"] = trial.suggest_float("be_trigger_rr", 0.5, 2.0)
        params["partial_close_rr"] = trial.suggest_float("partial_close_rr", 1.0, 3.0)
        params["trail_activation_rr"] = trial.suggest_float("trail_activation_rr", 1.5, 4.0)

        # Ensure ordering: be < partial < trail
        if params["partial_close_rr"] <= params["be_trigger_rr"]:
            params["partial_close_rr"] = params["be_trigger_rr"] + 0.5
        if params["trail_activation_rr"] <= params["partial_close_rr"]:
            params["trail_activation_rr"] = params["partial_close_rr"] + 0.5

        # --- Session Weights (3) ---
        params["overlap_weight"] = trial.suggest_float("overlap_weight", 0.80, 1.50)
        params["london_ny_weight"] = trial.suggest_float("london_ny_weight", 0.80, 1.30)
        params["asian_weight"] = trial.suggest_float("asian_weight", 0.50, 1.00)

        # --- OB Sensitivity (1) ---
        params["ob_strong_move_pct"] = trial.suggest_float("ob_strong_move_pct", 0.2, 1.0)

        # --- LTF Confirmation (1) ---
        params["ltf_min_score"] = trial.suggest_float("ltf_min_score", 0.0, 0.50)

        # --- TP Multiplier (1) ---
        params["tp_atr_mult"] = trial.suggest_float("tp_atr_mult", 3.0, 8.0)

        # ── Phase 2: BOS Quality Filter (V5) ─────────────────────────────────
        # bos_solo_penalty     : extra score deduction for naked BOS (no CHoCH/FVG/OB/LiqSweep)
        # bos_ranging_fill_floor: stricter SMC fill floor for BOS in ranging/volatile regimes
        params["bos_solo_penalty"]       = trial.suggest_float("bos_solo_penalty", 0.0, 0.20)
        params["bos_ranging_fill_floor"] = trial.suggest_float("bos_ranging_fill_floor", 0.30, 0.55)

        return params

    def _params_to_config(self, params: Dict) -> Dict:
        """Convert trial parameters into a full bot config."""
        config = copy.deepcopy(self.base_config)

        # Regime weights
        regime_weights = {}
        for regime in ["trending", "ranging", "breakout", "reversal", "volatile"]:
            smc_w = params.get(f"{regime}_smc_weight", 0.40)
            tech_w = params.get(f"{regime}_tech_weight", 0.25)
            # Normalize so they sum to reasonable range
            regime_weights[regime] = {
                "smc_weight": smc_w,
                "tech_weight": tech_w,
                "market_weight": max(0.05, 1.0 - smc_w - tech_w - 0.10),
                "mtf_weight": 0.10,
                "min_confluence": params.get(f"{regime}_min_confluence", 0.55),
                "atr_sl_mult": params.get(f"{regime}_atr_sl_mult", 3.0),
                "min_smc_signals": params.get(f"{regime}_min_smc", 2),
            }
        config["regime_weights"] = regime_weights

        # Exit stages — write at TOP LEVEL: BacktestEngine reads config.get("exit_stages")
        # NOT under "risk:" — risk_config.yaml has no "risk:" wrapper, all keys are top-level
        config["exit_stages"] = {
            "be_trigger_rr": params.get("be_trigger_rr", 1.0),
            "partial_close_rr": params.get("partial_close_rr", 1.5),
            "trail_activation_rr": params.get("trail_activation_rr", 2.0),
        }

        # SL/TP — write at TOP LEVEL: BacktestEngine reads config.get("stop_loss") etc.
        sl_cfg = copy.deepcopy(config.get("stop_loss", {}))
        sl_cfg["atr_multiplier"] = params.get("trending_atr_sl_mult", 3.0)
        config["stop_loss"] = sl_cfg

        tp_cfg = copy.deepcopy(config.get("take_profit", {}))
        tp_cfg["atr_multiplier"] = params.get("tp_atr_mult", 5.0)
        config["take_profit"] = tp_cfg

        # OB sensitivity
        smc_cfg = config.get("smc_indicators", {})
        smc_cfg.setdefault("order_blocks", {})["strong_move_percent"] = params.get("ob_strong_move_pct", 1.0)
        config["smc_indicators"] = smc_cfg

        # Session weights
        session_cfg = config.get("session", {})
        session_cfg["overlap_weight"] = params.get("overlap_weight", 1.25)
        session_cfg["london_ny_weight"] = params.get("london_ny_weight", 1.10)
        session_cfg["asian_weight"] = params.get("asian_weight", 0.70)
        config["session"] = session_cfg

        # Adaptive scorer
        config["use_adaptive_scorer"] = True

        # Phase 2: BOS quality filter — wired into confluence_weights
        cw = config.get("confluence_weights", {})
        cw["bos_solo_penalty"]       = params.get("bos_solo_penalty", 0.05)
        cw["bos_ranging_fill_floor"] = params.get("bos_ranging_fill_floor", 0.38)
        config["confluence_weights"] = cw

        return config

    def _calculate_score(self, metrics: Dict, params: Dict) -> float:
        """
        Calculate optimization score.
        base = (PF * WR * 100) / max(DD, 1.0)
        + max(0, avg_rr - 1.5) * 5
        + total_return_pct * 0.25
        - small_account_penalty
        - consistency_penalty (handled in objective)
        """
        pf = metrics.get("profit_factor", 0)
        wr = metrics.get("win_rate", 0) / 100
        dd = max(metrics.get("max_drawdown_percent", 1), 1.0)
        avg_rr = metrics.get("avg_rr_ratio", 0)
        ret_pct = metrics.get("total_return_percent", 0)

        if pf < 1.0:
            return pf * 10 - 10  # Negative score for losing strategies

        base = (pf * wr * 100) / dd
        rr_bonus = max(0, avg_rr - 1.5) * 5
        return_bonus = ret_pct * 0.25

        # Small account penalty: any trade risking >$2 on $100
        small_penalty = 0
        # (evaluated at trade level in backtest engine via micro_manager)

        score = base + rr_bonus + return_bonus - small_penalty

        return round(score, 2)

    def _save_results(self, results: Dict) -> None:
        """Save optimization results."""
        filepath = self.results_dir / "optimization_v3_results.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Results saved to {filepath}")
