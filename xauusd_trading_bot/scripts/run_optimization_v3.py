"""
Optimizer V3 Runner
Walk-forward optimization with ~45 regime-aware parameters.
Run: python scripts/run_optimization_v3.py [--trials 300] [--jobs 4] [--months 12]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from src.core.mt5_connector import MT5Connector
from src.optimization.parameter_optimizer_v3 import ParameterOptimizerV3
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


def main():
    parser = argparse.ArgumentParser(description="V3 Walk-Forward Optimization")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel jobs")
    parser.add_argument("--months", type=int, default=12, help="Months of history")
    parser.add_argument("--windows", type=int, default=3, help="Walk-forward windows")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    parser.add_argument("--study-name", type=str, default=None, help="Optuna study name")
    args = parser.parse_args()

    setup_logger()
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("V3 WALK-FORWARD OPTIMIZATION")
    logger.info("=" * 60)

    config = load_config()

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
        logger.info(f"Trials: {args.trials} | Jobs: {args.jobs} | Windows: {args.windows}")

        opt_config = {
            "n_trials": args.trials,
            "n_jobs": args.jobs,
            "n_windows": args.windows,
            "initial_balance": args.balance,
        }

        optimizer = ParameterOptimizerV3(mt5, config, opt_config)

        # Step 1: Pre-calculate all indicators (done once)
        logger.info("\n--- STEP 1: Pre-calculating indicators ---")
        optimizer.prepare_data(start_date, end_date, use_cache=True)

        # Step 1b: Diagnose windows — verify trade counts before burning Optuna trials
        logger.info("\n--- STEP 1b: Window diagnostic ---")
        optimizer.diagnose_windows()

        # Step 2: Run optimization
        logger.info("\n--- STEP 2: Running walk-forward optimization ---")
        results = optimizer.optimize(study_name=args.study_name)

        # Step 3: Summary
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best Score: {results['best_score']:.2f}")
        logger.info(f"Best Trial: #{results['best_trial']}")
        logger.info(f"Total Trials: {results['total_trials']}")
        logger.info(f"Time: {results['elapsed_seconds']/60:.1f} min")
        logger.info(f"\nBest Parameters:")
        for k, v in sorted(results["best_params"].items()):
            logger.info(f"  {k}: {v}")

        logger.info(f"\nOptimized config saved to: data/optimization_v3/optimized_config_v3.yaml")
        logger.info("Apply: copy optimized values to config/ files")

    finally:
        mt5.disconnect()


if __name__ == "__main__":
    main()
