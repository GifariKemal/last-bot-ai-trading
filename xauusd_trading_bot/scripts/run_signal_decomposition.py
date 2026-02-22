"""
Signal Decomposition Runner
Analyzes which SMC signals and combos predict profitable trades.
Run: python scripts/run_signal_decomposition.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from src.core.mt5_connector import MT5Connector
from src.backtesting.historical_data import HistoricalDataManager
from src.core.data_manager import DataManager
from src.indicators.technical import TechnicalIndicators
from src.indicators.smc_indicators import SMCIndicators
from src.analysis.signal_decomposition import SignalDecompositionAnalyzer
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
    parser = argparse.ArgumentParser(description="Signal Decomposition Analysis")
    parser.add_argument("--months", type=int, default=12, help="Months of history")
    parser.add_argument("--forward-bars", type=int, default=20, help="Forward look window")
    parser.add_argument("--min-samples", type=int, default=30, help="Min samples per signal")
    args = parser.parse_args()

    setup_logger()
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("SIGNAL DECOMPOSITION ANALYSIS")
    logger.info("=" * 60)

    config = load_config()

    # Connect to MT5
    mt5_config = config.get("mt5", {})
    mt5 = MT5Connector(mt5_config)
    if not mt5.connect():
        logger.error("Failed to connect to MT5")
        return

    try:
        # Prepare data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.months * 30)

        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Forward window: {args.forward_bars} bars")

        data_mgr = HistoricalDataManager()
        price_mgr = DataManager()

        df = data_mgr.prepare_backtest_data(
            mt5, "XAUUSDm", "M15", start_date, end_date, use_cache=True
        )
        if df is None or len(df) < 500:
            logger.error("Insufficient data")
            return

        logger.info(f"Loaded {len(df)} bars")

        # Calculate indicators
        tech = TechnicalIndicators(config.get("indicators", {}))
        smc = SMCIndicators(config.get("smc_indicators", {}))

        df = price_mgr.add_basic_features(df)
        df = price_mgr.add_price_changes(df)
        df = tech.calculate_all(df)
        df = smc.calculate_all(df)

        logger.info("Indicators calculated, starting decomposition...")

        # Run analysis
        analyzer = SignalDecompositionAnalyzer({
            "forward_bars": args.forward_bars,
            "min_samples": args.min_samples,
        })
        results = analyzer.analyze(df)

        # Save results
        output_dir = "data/signal_analysis"
        filepath = analyzer.save_results(results, output_dir)

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("SIGNAL QUALITY SUMMARY")
        logger.info("=" * 60)

        for sig_name, data in results.get("individual_signals", {}).items():
            present = data.get("present")
            if present:
                logger.info(
                    f"  {sig_name:12s} | "
                    f"WR: {present['win_rate']:.1%} | "
                    f"PF: {present['profit_factor']:5.2f} | "
                    f"AvgRR: {present['avg_rr']:.2f} | "
                    f"n={present['n']}"
                )
            else:
                logger.info(f"  {sig_name:12s} | Insufficient samples (n={data['n_present']})")

        # Tier summary
        tiers = results.get("tiers", {})
        for tier_name, signals in tiers.items():
            if signals:
                names = [s["signal"] for s in signals]
                logger.info(f"\n  {tier_name}: {', '.join(names)}")

        logger.info(f"\nResults saved to: {filepath}")

    finally:
        mt5.disconnect()


if __name__ == "__main__":
    main()
