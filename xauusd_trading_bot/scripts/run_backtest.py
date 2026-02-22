"""
Run Backtest Script
Execute strategy backtest on historical data.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtesting import BacktestEngine
from src.core.mt5_connector import MT5Connector
from src.bot_logger import setup_logger, get_logger
from src.utils.config_loader import ConfigLoader


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run backtest on historical XAUUSD data"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD), default: 1 year ago",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD), default: today",
    )

    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000.0,
        help="Initial balance (default: 10000)",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="M15",
        choices=["M1", "M5", "M15", "M30", "H1", "H4"],
        help="Primary timeframe (default: M15)",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached data",
    )

    parser.add_argument(
        "--save",
        type=str,
        help="Save results to file (JSON)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Setup logging
    setup_logger()
    logger = get_logger()

    try:
        # Parse dates
        if args.end_date:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        else:
            end_date = datetime.now()

        if args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        else:
            start_date = end_date - timedelta(days=365)

        # Load configuration
        logger.info("Loading configuration...")
        config_loader = ConfigLoader()

        settings = config_loader.load("settings")
        mt5_config = config_loader.load("mt5_config")
        trading_rules = config_loader.load("trading_rules")
        risk_config = config_loader.load("risk_config")
        session_config = config_loader.load("session_config")

        # Merge configurations
        config = {
            **settings,
            "mt5": mt5_config,
            "strategy": trading_rules.get("strategy", {}),
            "indicators": trading_rules.get("indicators", {}),
            "smc_indicators": trading_rules.get("smc_indicators", {}),
            "technical_indicators": trading_rules.get("technical_indicators", {}),
            "confluence_weights": trading_rules.get("confluence_weights", {}),
            "market_conditions": trading_rules.get("market_conditions", {}),
            "mtf_analysis": trading_rules.get("mtf_analysis", {}),
            "signal_validation": trading_rules.get("signal_validation", {}),
            "risk": risk_config,
            "session": session_config,
        }

        # Connect to MT5 (for data fetching)
        logger.info("Connecting to MT5...")
        mt5 = MT5Connector(config["mt5"])
        connection = mt5.connect()

        if not connection:
            logger.error("Failed to connect to MT5")
            return 1

        logger.info("MT5 connected")

        # Initialize backtest engine
        logger.info("Initializing backtest engine...")
        backtest = BacktestEngine(config)

        # Run backtest
        results = backtest.run_backtest(
            mt5=mt5,
            symbol="XAUUSDm",
            start_date=start_date,
            end_date=end_date,
            initial_balance=args.initial_balance,
            timeframe=args.timeframe,
            use_cache=not args.no_cache,
        )

        # Save results if requested
        if args.save and results.get("success"):
            backtest.save_results(results, args.save)

        # Disconnect MT5
        mt5.disconnect()

        # Return based on success
        if results.get("success"):
            logger.info("\n✓ Backtest completed successfully")
            return 0
        else:
            logger.error("\n✗ Backtest failed")
            return 1

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
