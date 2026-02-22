"""
Test MT5 Connection
Quick script to test MT5 connection and fetch sample data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.mt5_connector import MT5Connector
from src.bot_logger import setup_logger, get_logger
from src.utils.config_loader import ConfigLoader


def main():
    """Test MT5 connection."""
    setup_logger()
    logger = get_logger()

    try:
        logger.info("=" * 80)
        logger.info("TESTING MT5 CONNECTION")
        logger.info("=" * 80)

        # Load MT5 config
        logger.info("\n1. Loading MT5 configuration...")
        config_loader = ConfigLoader()
        mt5_config = config_loader.load("mt5_config")

        logger.info(f"   Server: {mt5_config['connection']['server']}")
        logger.info(f"   Login: {mt5_config['connection']['login']}")
        logger.info(f"   Terminal: {mt5_config['connection']['terminal_path']}")

        # Initialize connector
        logger.info("\n2. Initializing MT5 connector...")
        mt5 = MT5Connector(mt5_config)

        # Connect
        logger.info("\n3. Connecting to MT5...")
        result = mt5.connect()

        if not result:
            logger.error("   ✗ Connection failed")
            return 1

        logger.info("   ✓ Connected successfully!")

        # Get account info
        logger.info("\n4. Fetching account information...")
        account = mt5.get_account_info()

        if account:
            logger.info(f"   Server: {account['server']}")
            logger.info(f"   Login: {account['login']}")
            logger.info(f"   Name: {account.get('name', 'N/A')}")
            logger.info(f"   Balance: ${account['balance']:.2f}")
            logger.info(f"   Equity: ${account['equity']:.2f}")
            logger.info(f"   Margin Free: ${account['margin_free']:.2f}")
            logger.info(f"   Leverage: 1:{account['leverage']}")
            logger.info(f"   Currency: {account.get('currency', 'USD')}")

        # Get current tick
        logger.info("\n5. Fetching XAUUSD current price...")
        tick = mt5.get_tick("XAUUSDm")

        if tick:
            logger.info(f"   Symbol: XAUUSD")
            logger.info(f"   Bid: {tick['bid']:.2f}")
            logger.info(f"   Ask: {tick['ask']:.2f}")
            logger.info(f"   Spread: {tick['spread']:.5f}")
            logger.info(f"   Time: {tick['time']}")

        # Fetch sample data
        logger.info("\n6. Fetching sample M15 data (100 bars)...")
        from datetime import datetime, timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        df = mt5.get_historical_data(
            symbol="XAUUSDm",
            timeframe="M15",
            start_date=start_date,
            end_date=end_date
        )

        if df is not None and len(df) > 0:
            logger.info(f"   ✓ Fetched {len(df)} bars")
            logger.info(f"   Date range: {df['time'][0]} to {df['time'][-1]}")
            logger.info(f"   Latest close: {df['close'][-1]:.2f}")
        else:
            logger.warning("   ⚠ No data received")

        # Get open positions
        logger.info("\n7. Checking open positions...")
        positions = mt5.get_positions("XAUUSDm")
        logger.info(f"   Open positions: {len(positions)}")

        if positions:
            for pos in positions:
                logger.info(f"\n   Position #{pos['ticket']}")
                logger.info(f"     Type: {pos['type']}")
                logger.info(f"     Volume: {pos['volume']}")
                logger.info(f"     Open: {pos['open_price']:.2f}")
                logger.info(f"     Current: {pos['current_price']:.2f}")
                logger.info(f"     Profit: ${pos['profit']:.2f}")

        # Disconnect
        logger.info("\n8. Disconnecting...")
        mt5.disconnect()

        logger.info("\n" + "=" * 80)
        logger.info("✓ MT5 CONNECTION TEST SUCCESSFUL")
        logger.info("=" * 80)
        logger.info("\nYou can now proceed with:")
        logger.info("1. Optimization: python scripts/run_optimization.py --trials 200 --months 6")
        logger.info("2. Backtesting: python scripts/run_backtest.py --months 6")
        logger.info("3. Demo Trading: python main.py --mode demo")

        return 0

    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
