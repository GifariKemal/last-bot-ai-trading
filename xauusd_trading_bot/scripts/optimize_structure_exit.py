"""
Structure Exit Optimizer
Optimizes 6 regime-adaptive structure exit parameters using Optuna + walk-forward.

Parameters:
  - strong_trend_min_rr (0.0–1.0)   — min RR before CHoCH exit in strong trends
  - weak_trend_min_rr   (0.0–1.5)   — min RR before CHoCH exit in weak trends
  - range_min_rr        (0.3–2.0)   — min RR before CHoCH exit in ranging markets
  - volatile_min_rr     (0.0–1.0)   — min RR before CHoCH exit in volatile/breakout
  - reversal_min_rr     (0.0–1.0)   — min RR before CHoCH exit in reversals
  - min_hold_bars       (1–4)       — minimum bars held before structure exit allowed

Run:
  cd xauusd_trading_bot
  python scripts/optimize_structure_exit.py --trials 50 --months 3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import json
import argparse
import copy
import time
from datetime import datetime, timedelta
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import PercentilePruner

from src.core.mt5_connector import MT5Connector
from src.backtesting.backtest_engine import BacktestEngine
from src.bot_logger import get_logger, setup_logger


def load_config():
    """Load merged config from all YAML files."""
    config = {}
    config_dir = Path("config")
    for yaml_file in config_dir.glob("*.yaml"):
        with open(yaml_file) as f:
            data = yaml.safe_load(f) or {}
            config.update(data)
    return config


def load_v3_optimized_config():
    """Load V3 optimized config as baseline."""
    path = Path("data/optimization_v3/optimized_config_v3.yaml")
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


class StructureExitOptimizer:
    """Optuna optimizer for structure exit thresholds."""

    def __init__(self, mt5, base_config: dict, opt_config: dict):
        self.logger = get_logger()
        self.mt5 = mt5
        self.base_config = copy.deepcopy(base_config)

        self.n_trials = opt_config.get("n_trials", 50)
        self.n_startup_trials = opt_config.get("n_startup_trials", 15)
        self.n_windows = opt_config.get("n_windows", 3)
        self.train_ratio = opt_config.get("train_ratio", 0.7)
        self.initial_balance = opt_config.get("initial_balance", 10000.0)

        self.results_dir = Path("data/optimization_structure_exit")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._df_prepared = None
        self._df_m5 = None
        self._windows = []
        self.start_time = None
        self.trial_count = 0

        # Baseline results (no structure exit)
        self.baseline_score = None
        self.baseline_metrics = None

    def prepare_data(self, start_date: datetime, end_date: datetime) -> None:
        """Pre-calculate indicators once."""
        self.logger.info("Pre-calculating indicators...")
        engine = BacktestEngine(self.base_config)
        df = engine._prepare_data(
            self.mt5, "XAUUSDm", "M15", start_date, end_date, use_cache=True
        )
        if df is None:
            raise ValueError("Failed to prepare backtest data")

        self._df_prepared = df
        self._df_m5 = engine.df_m5
        self.logger.info(f"Data prepared: {len(df)} bars")

        # Build walk-forward windows
        self._windows = self._build_windows(df)
        self.logger.info(f"Walk-forward windows: {len(self._windows)}")
        for i, (ts, te, vs, ve) in enumerate(self._windows):
            self.logger.info(f"  Window {i+1}: Train [{ts}..{te}] Test [{vs}..{ve}]")

    def _build_windows(self, df) -> list:
        """Build walk-forward window indices."""
        n = len(df)
        usable = n - 100
        window_size = usable // self.n_windows
        windows = []
        for i in range(self.n_windows):
            start = 100 + i * window_size
            end = start + window_size
            if i == self.n_windows - 1:
                end = n
            train_size = int((end - start) * self.train_ratio)
            train_end = start + train_size
            windows.append((start, train_end, train_end, end))
        return windows

    def run_baseline(self) -> dict:
        """Run baseline backtest (no structure exit) for comparison."""
        self.logger.info("\n--- BASELINE: No structure exit ---")
        config = copy.deepcopy(self.base_config)
        config.pop("structure_exit", None)  # Ensure disabled

        scores = []
        all_metrics = []
        for ws, we, ts, te in self._windows:
            df_test = self._df_prepared[ts:te]
            if len(df_test) < 50:
                continue
            engine = BacktestEngine(config)
            result = engine.run_backtest_fast(
                df_test, initial_balance=self.initial_balance, df_m5=self._df_m5
            )
            if result.get("success"):
                m = result["metrics"]
                s = self._calculate_score(m)
                scores.append(s)
                all_metrics.append(m)

        if scores:
            avg = sum(scores) / len(scores)
            self.baseline_score = avg
            self.baseline_metrics = all_metrics
            self.logger.info(f"Baseline score: {avg:.2f}")
            for i, m in enumerate(all_metrics):
                self.logger.info(
                    f"  Window {i+1}: PF={m.get('profit_factor',0):.2f} "
                    f"WR={m.get('win_rate',0):.1f}% "
                    f"DD={m.get('max_drawdown_percent',0):.1f}% "
                    f"Ret={m.get('total_return_percent',0):.1f}% "
                    f"Trades={m.get('total_trades',0)}"
                )
        else:
            self.baseline_score = 0
            self.logger.warning("Baseline produced no valid windows")

        return {"score": self.baseline_score, "metrics": all_metrics}

    def optimize(self) -> dict:
        """Run Optuna optimization."""
        if self._df_prepared is None:
            raise ValueError("Call prepare_data() first")

        self.start_time = time.time()
        self.trial_count = 0

        study_name = f"structure_exit_{datetime.now().strftime('%Y%m%d_%H%M')}"

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
            n_jobs=1,  # Sequential — SMC computation is CPU-heavy
            show_progress_bar=True,
        )

        best = study.best_trial
        elapsed = time.time() - self.start_time

        self.logger.info("=" * 70)
        self.logger.info(f"OPTIMIZATION COMPLETE in {elapsed/60:.1f} min")
        self.logger.info(f"Best Score: {best.value:.2f} (baseline: {self.baseline_score:.2f})")
        self.logger.info(f"Improvement: {best.value - self.baseline_score:+.2f}")
        self.logger.info(f"Best Params:")
        for k, v in sorted(best.params.items()):
            self.logger.info(f"  {k}: {v}")
        self.logger.info("=" * 70)

        results = {
            "best_score": best.value,
            "baseline_score": self.baseline_score,
            "improvement": best.value - self.baseline_score,
            "best_params": best.params,
            "best_trial": best.number,
            "total_trials": len(study.trials),
            "elapsed_seconds": elapsed,
            "n_windows": self.n_windows,
            "timestamp": datetime.now().isoformat(),
        }

        # Save
        results_path = self.results_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Results saved to {results_path}")

        # Print live exit_signals.py values
        self._print_live_values(best.params)

        return results

    def _objective(self, trial: optuna.Trial) -> float:
        """Walk-forward objective for structure exit params."""
        self.trial_count += 1

        params = {
            "strong_trend_min_rr": trial.suggest_float("strong_trend_min_rr", 0.0, 1.0, step=0.1),
            "weak_trend_min_rr": trial.suggest_float("weak_trend_min_rr", 0.0, 1.5, step=0.1),
            "range_min_rr": trial.suggest_float("range_min_rr", 0.3, 2.0, step=0.1),
            "volatile_min_rr": trial.suggest_float("volatile_min_rr", 0.0, 1.0, step=0.1),
            "reversal_min_rr": trial.suggest_float("reversal_min_rr", 0.0, 1.0, step=0.1),
            "min_hold_bars": trial.suggest_int("min_hold_bars", 1, 4),
        }

        config = copy.deepcopy(self.base_config)
        config["structure_exit"] = params

        oos_scores = []
        for window_idx, (ts, te, vs, ve) in enumerate(self._windows):
            df_test = self._df_prepared[vs:ve]
            if len(df_test) < 50:
                trial.report(-100, window_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                continue

            engine = BacktestEngine(config)
            result = engine.run_backtest_fast(
                df_test, initial_balance=self.initial_balance, df_m5=self._df_m5
            )

            if not result.get("success"):
                trial.report(-100, window_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                continue

            metrics = result["metrics"]
            score = self._calculate_score(metrics)

            if metrics["total_trades"] < 8:
                score = -50
            if metrics.get("max_drawdown_percent", 0) > 25:
                score = -100

            oos_scores.append(score)
            trial.report(score, window_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if not oos_scores:
            return -100

        avg_score = sum(oos_scores) / len(oos_scores)

        # Soft penalty for negative windows (scale down, don't hard reject)
        negative_windows = sum(1 for s in oos_scores if s < 0)
        if negative_windows >= 2:
            avg_score *= 0.5  # Penalize but don't hard-reject

        # Consistency penalty
        if len(oos_scores) >= 2:
            mean = avg_score
            variance = sum((s - mean) ** 2 for s in oos_scores) / len(oos_scores)
            std = variance ** 0.5
            if mean > 0 and std / mean > 0.3:
                avg_score *= 0.8

        # Progress logging
        if self.trial_count % 10 == 0:
            elapsed = time.time() - self.start_time
            self.logger.info(
                f"Trial {self.trial_count}/{self.n_trials} | "
                f"Score: {avg_score:.2f} | Time: {elapsed/60:.1f}min"
            )

        return avg_score

    def _calculate_score(self, metrics: dict) -> float:
        """Score: (PF * WR * 100) / DD + rr_bonus + return_bonus."""
        pf = metrics.get("profit_factor", 0)
        wr = metrics.get("win_rate", 0) / 100
        dd = max(metrics.get("max_drawdown_percent", 1), 1.0)
        avg_rr = metrics.get("avg_rr_ratio", 0)
        ret_pct = metrics.get("total_return_percent", 0)

        if pf < 1.0:
            return pf * 10 - 10

        base = (pf * wr * 100) / dd
        rr_bonus = max(0, avg_rr - 1.5) * 5
        return_bonus = ret_pct * 0.25

        return round(base + rr_bonus + return_bonus, 2)

    def _print_live_values(self, params: dict) -> None:
        """Print values ready to paste into exit_signals.py."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("PASTE INTO exit_signals.py (STRUCTURE_EXIT_MIN_RR):")
        self.logger.info("=" * 70)

        strong = params["strong_trend_min_rr"]
        weak = params["weak_trend_min_rr"]
        rng = params["range_min_rr"]
        vol = params["volatile_min_rr"]
        rev = params["reversal_min_rr"]
        hold = params["min_hold_bars"]

        self.logger.info(f"""
STRUCTURE_EXIT_MIN_RR = {{
    MarketRegime.STRONG_TREND_UP: {strong},
    MarketRegime.STRONG_TREND_DOWN: {strong},
    MarketRegime.WEAK_TREND_UP: {weak},
    MarketRegime.WEAK_TREND_DOWN: {weak},
    MarketRegime.RANGE_TIGHT: {rng},
    MarketRegime.RANGE_WIDE: {rng},
    MarketRegime.VOLATILE_BREAKOUT: {vol},
    MarketRegime.REVERSAL: {rev},
}}
# MIN_HOLD_MINUTES = {hold * 15}  (= {hold} M15 bars)
""")


def main():
    parser = argparse.ArgumentParser(description="Structure Exit Optimization")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials")
    parser.add_argument("--months", type=int, default=3, help="Months of history")
    parser.add_argument("--windows", type=int, default=3, help="Walk-forward windows")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    args = parser.parse_args()

    setup_logger()
    logger = get_logger()
    logger.info("=" * 70)
    logger.info("STRUCTURE EXIT OPTIMIZATION")
    logger.info("=" * 70)

    # Load base config only (already has live V3-tuned params applied)
    # Do NOT merge V3 optimized config — it has stale swing_lookback=10
    # and wrong symbol (XAUUSD vs XAUUSDm), which produces 0 signals
    config = load_config()
    config.pop("structure_exit", None)

    # Connect MT5
    mt5_config = config.get("mt5", {})
    mt5 = MT5Connector(mt5_config)
    if not mt5.connect():
        logger.error("Failed to connect to MT5")
        return

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.months * 30)

        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Trials: {args.trials} | Windows: {args.windows}")

        opt_config = {
            "n_trials": args.trials,
            "n_startup_trials": 15,
            "n_windows": args.windows,
            "train_ratio": 0.7,
            "initial_balance": args.balance,
        }

        optimizer = StructureExitOptimizer(mt5, config, opt_config)

        # Step 1: Prepare data
        logger.info("\n--- STEP 1: Preparing data ---")
        optimizer.prepare_data(start_date, end_date)

        # Step 2: Run baseline
        logger.info("\n--- STEP 2: Running baseline (no structure exit) ---")
        baseline = optimizer.run_baseline()

        # Step 3: Optimize
        logger.info("\n--- STEP 3: Optimizing structure exit thresholds ---")
        results = optimizer.optimize()

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Baseline Score: {baseline['score']:.2f}")
        logger.info(f"Best Score:     {results['best_score']:.2f}")
        logger.info(f"Improvement:    {results['improvement']:+.2f}")
        logger.info(f"Results saved:  data/optimization_structure_exit/results.json")

    finally:
        mt5.disconnect()


if __name__ == "__main__":
    main()
