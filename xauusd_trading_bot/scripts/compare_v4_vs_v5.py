"""
Phase 4: V4 Base vs V5 Optimized Comparison
Runs backtest on the same OOS window with both configs, prints side-by-side metrics.

Usage:
    python scripts/compare_v4_vs_v5.py [--months 6] [--balance 10000]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from src.core.mt5_connector import MT5Connector
from src.backtesting.backtest_engine import BacktestEngine
from src.optimization.parameter_optimizer_v3 import ParameterOptimizerV3
from src.bot_logger import get_logger, setup_logger


def load_base_config():
    config = {}
    for yaml_file in Path("config").glob("*.yaml"):
        with open(yaml_file) as f:
            data = yaml.safe_load(f) or {}
            config.update(data)
    return config


def load_optimized_config():
    path = Path("data/optimization_v3/optimized_config_v3.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Optimized config not found: {path}. Run run_optimization_v3.py first.")
    with open(path) as f:
        return yaml.safe_load(f)


def fmt(label, v4, v5, fmt_str=".2f", higher_better=True):
    """Format a comparison row."""
    try:
        diff = float(v5) - float(v4)
        sign = "+" if diff >= 0 else ""
        arrow = "▲" if (diff > 0) == higher_better else ("▼" if diff < 0 else "=")
        return f"  {label:<30} V4: {float(v4):{fmt_str}}  V5: {float(v5):{fmt_str}}  ({sign}{diff:{fmt_str}}) {arrow}"
    except Exception:
        return f"  {label:<30} V4: {v4}  V5: {v5}"


def run_oos_backtest(config, df_prepared, test_start, test_end, balance):
    engine = BacktestEngine(config)
    df_test = df_prepared[test_start:test_end]
    result = engine.run_backtest_fast(df_test, initial_balance=balance)
    return result


def main():
    parser = argparse.ArgumentParser(description="V4 vs V5 Backtest Comparison")
    parser.add_argument("--months", type=int, default=6, help="Months of history")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    parser.add_argument("--windows", type=int, default=1, help="Walk-forward windows")
    args = parser.parse_args()

    setup_logger()
    logger = get_logger()

    logger.info("=" * 70)
    logger.info("PHASE 4: V4 BASE vs V5 OPTIMIZED COMPARISON")
    logger.info("=" * 70)

    base_config = load_base_config()
    try:
        opt_config = load_optimized_config()
        logger.info("Loaded optimized config from data/optimization_v3/optimized_config_v3.yaml")
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    mt5_config = base_config.get("mt5", {})
    mt5 = MT5Connector(mt5_config)
    if not mt5.connect():
        logger.error("Failed to connect to MT5")
        return

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.months * 30)
        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Balance: ${args.balance:,.0f}")

        # Pre-calculate indicators once (using base config for data)
        logger.info("\nPre-calculating indicators...")
        opt_cfg = {"n_trials": 1, "n_jobs": 1, "n_windows": args.windows, "initial_balance": args.balance}
        optimizer = ParameterOptimizerV3(mt5, base_config, opt_cfg)
        optimizer.prepare_data(start_date, end_date, use_cache=True)

        df_prepared = optimizer._df_prepared
        windows = optimizer._windows

        logger.info(f"\nOOS windows:")
        for i, (_, _, ts, te) in enumerate(windows):
            logger.info(f"  Window {i+1}: bars {ts}-{te} ({te-ts} bars = ~{(te-ts)/95:.0f} days)")

        # Run both configs on all OOS windows
        logger.info("\nRunning backtests...")
        for win_i, (_, _, test_start, test_end) in enumerate(windows):
            logger.info(f"\n{'='*70}")
            logger.info(f"WINDOW {win_i+1} — OOS bars {test_start}:{test_end}")
            logger.info(f"{'='*70}")

            # V4 base
            engine_v4 = BacktestEngine(base_config)
            df_test = df_prepared[test_start:test_end]
            r4 = engine_v4.run_backtest_fast(df_test, initial_balance=args.balance)
            m4 = r4.get("metrics", {})

            # V5 optimized
            engine_v5 = BacktestEngine(opt_config)
            r5 = engine_v5.run_backtest_fast(df_test, initial_balance=args.balance)
            m5 = r5.get("metrics", {})

            logger.info("\n  METRIC                         V4 (Base)    V5 (Optimized)")
            logger.info("  " + "-" * 60)
            logger.info(fmt("Total Trades",         m4.get("total_trades",0),        m5.get("total_trades",0),        ".0f"))
            logger.info(fmt("Win Rate (%)",          m4.get("win_rate",0),             m5.get("win_rate",0),             ".1f"))
            logger.info(fmt("Profit Factor",         m4.get("profit_factor",0),        m5.get("profit_factor",0),        ".3f"))
            logger.info(fmt("Avg RR",                m4.get("avg_rr_ratio",0),         m5.get("avg_rr_ratio",0),         ".2f"))
            logger.info(fmt("Total Return (%)",      m4.get("total_return_percent",0), m5.get("total_return_percent",0), ".2f"))
            logger.info(fmt("Max DD (%)",            m4.get("max_drawdown_percent",0), m5.get("max_drawdown_percent",0), ".1f", higher_better=False))
            logger.info(fmt("Expectancy ($)",        m4.get("expectancy",0),           m5.get("expectancy",0),           ".2f"))
            logger.info(fmt("Total PnL ($)",         m4.get("total_pnl",0),            m5.get("total_pnl",0),            ".2f"))

            # Opt score for comparison (reuse optimizer's scoring formula)
            dummy_opt = ParameterOptimizerV3.__new__(ParameterOptimizerV3)
            s4 = dummy_opt._calculate_score(m4, {})
            s5 = dummy_opt._calculate_score(m5, {})
            logger.info(fmt("Opt Score",             s4, s5, ".2f"))

    finally:
        mt5.disconnect()


if __name__ == "__main__":
    main()
