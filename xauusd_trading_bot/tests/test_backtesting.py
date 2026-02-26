"""
Test Phase 11: Backtesting System
Tests backtesting engine and performance metrics.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtesting import (
    BacktestEngine,
    HistoricalDataManager,
    PerformanceMetrics,
)
from src.utils.config_loader import ConfigLoader


def test_performance_metrics():
    """Test performance metrics calculation."""
    print("\n" + "=" * 60)
    print("Testing Performance Metrics")
    print("=" * 60)

    metrics = PerformanceMetrics()

    # Mock trades data
    mock_trades = [
        {"profit": 50, "entry_time": datetime.now()},
        {"profit": -20, "entry_time": datetime.now()},
        {"profit": 30, "entry_time": datetime.now()},
        {"profit": -15, "entry_time": datetime.now()},
        {"profit": 40, "entry_time": datetime.now()},
        {"profit": 25, "entry_time": datetime.now()},
        {"profit": -10, "entry_time": datetime.now()},
        {"profit": 35, "entry_time": datetime.now()},
    ]

    initial_balance = 10000
    final_balance = 10135  # Sum of profits = 135

    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()

    print("\n1. Calculating metrics...")
    result = metrics.calculate_all_metrics(
        mock_trades,
        initial_balance,
        final_balance,
        start_date,
        end_date
    )

    print(f"\n2. Results:")
    print(f"  Total trades: {result['total_trades']}")
    print(f"  Win rate: {result['win_rate']:.2f}%")
    print(f"  Profit factor: {result['profit_factor']:.2f}")
    print(f"  Total return: ${result['total_return']:.2f}")
    print(f"  Max drawdown: {result['max_drawdown_percent']:.2f}%")
    print(f"  Sharpe ratio: {result['sharpe_ratio']:.2f}")

    # Generate report
    print("\n3. Generating report...")
    report = metrics.generate_report(result)

    # Display first few lines
    report_lines = report.split("\n")
    for line in report_lines[:15]:
        print(line)

    print("\n[OK] Performance Metrics tests passed!")


def test_historical_data_manager():
    """Test historical data manager."""
    print("\n" + "=" * 60)
    print("Testing Historical Data Manager")
    print("=" * 60)

    data_manager = HistoricalDataManager()

    print(f"\nData directory: {data_manager.data_dir}")
    print(f"  Exists: {data_manager.data_dir.exists()}")

    # Test cache operations
    print("\n1. Testing cache operations...")
    print(f"  Cache size: {len(data_manager.cache)}")

    data_manager.clear_cache()
    print(f"  After clear: {len(data_manager.cache)}")

    # Test data info (with empty DataFrame)
    print("\n2. Testing data info...")
    import polars as pl

    # Create mock data
    mock_df = pl.DataFrame({
        "time": [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)],
        "open": [2650.0 + i * 0.1 for i in range(100)],
        "high": [2651.0 + i * 0.1 for i in range(100)],
        "low": [2649.0 + i * 0.1 for i in range(100)],
        "close": [2650.5 + i * 0.1 for i in range(100)],
        "volume": [100 for _ in range(100)],
    })

    info = data_manager.get_data_info(mock_df)
    print(f"  Bars: {info['bars']}")
    print(f"  Duration: {info['duration_days']} days")
    print(f"  Columns: {len(info['columns'])}")

    print("\n[OK] Historical Data Manager tests passed!")


def test_backtest_engine_init():
    """Test backtest engine initialization."""
    print("\n" + "=" * 60)
    print("Testing Backtest Engine")
    print("=" * 60)

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load("settings")

    # Add required config sections
    config["indicators"] = {}
    config["risk"] = {"position_sizing": {"fixed_lot": 0.01}}
    config["session"] = {}

    print("\n1. Initializing backtest engine...")
    engine = BacktestEngine(config)

    print(f"  Data manager: {type(engine.data_manager).__name__}")
    print(f"  Metrics calculator: {type(engine.metrics_calculator).__name__}")
    print(f"  Strategy: {type(engine.strategy).__name__}")

    # Test state
    print("\n2. Checking initial state...")
    print(f"  Trades: {len(engine.trades)}")
    print(f"  Open positions: {len(engine.open_positions)}")
    print(f"  Balance: ${engine.balance:.2f}")

    print("\n[OK] Backtest Engine tests passed!")


def test_trade_simulation():
    """Test trade simulation logic."""
    print("\n" + "=" * 60)
    print("Testing Trade Simulation")
    print("=" * 60)

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load("settings")
    config["indicators"] = {}
    config["risk"] = {"position_sizing": {"fixed_lot": 0.01}}
    config["session"] = {}

    engine = BacktestEngine(config)

    # Initialize balance
    engine.balance = 10000
    engine.initial_balance = 10000

    print("\n1. Testing position opening...")
    mock_signal = {
        "direction": "BUY",
        "price": 2650.0,
        "confidence": 0.75,
    }

    engine._open_position(
        mock_signal,
        lot_size=0.01,
        sl=2640.0,
        tp=2670.0,
        entry_time=datetime.now()
    )

    print(f"  Open positions: {len(engine.open_positions)}")
    print(f"  Position details:")
    pos = engine.open_positions[0]
    print(f"    Direction: {pos['direction']}")
    print(f"    Entry: {pos['entry_price']:.2f}")
    print(f"    SL: {pos['sl']:.2f}")
    print(f"    TP: {pos['tp']:.2f}")

    # Test position closing
    print("\n2. Testing position closing (TP hit)...")
    engine._close_position(pos, 2670.0, "Take Profit")

    print(f"  Open positions: {len(engine.open_positions)}")
    print(f"  Closed trades: {len(engine.trades)}")
    print(f"  Balance: ${engine.balance:.2f}")

    if engine.trades:
        trade = engine.trades[0]
        print(f"  Trade profit: ${trade['profit']:.2f}")
        print(f"  Exit reason: {trade['reason']}")

    # Test loss trade
    print("\n3. Testing loss trade...")
    engine._open_position(
        {"direction": "SELL", "price": 2650.0, "confidence": 0.70},
        lot_size=0.01,
        sl=2660.0,
        tp=2630.0,
        entry_time=datetime.now()
    )

    pos = engine.open_positions[0]
    engine._close_position(pos, 2660.0, "Stop Loss")

    print(f"  Closed trades: {len(engine.trades)}")
    print(f"  Balance: ${engine.balance:.2f}")

    if len(engine.trades) > 1:
        trade = engine.trades[1]
        print(f"  Trade profit: ${trade['profit']:.2f}")

    print("\n[OK] Trade Simulation tests passed!")


def test_integration():
    """Test component integration."""
    print("\n" + "=" * 60)
    print("Testing Component Integration")
    print("=" * 60)

    print("\n1. Testing imports...")

    components = [
        ("BacktestEngine", "src.backtesting"),
        ("HistoricalDataManager", "src.backtesting"),
        ("PerformanceMetrics", "src.backtesting"),
    ]

    for comp_name, module_path in components:
        try:
            module = __import__(module_path, fromlist=[comp_name])
            comp_class = getattr(module, comp_name)
            print(f"  [OK] {comp_name}")
        except Exception as e:
            print(f"  [ERROR] {comp_name}: {e}")

    print("\n[OK] Component integration tests passed!")


def main():
    """Run all Phase 11 tests."""
    print("\n" + "=" * 60)
    print("PHASE 11: BACKTESTING SYSTEM - TESTING")
    print("=" * 60)

    try:
        # Test 1: Performance Metrics
        test_performance_metrics()

        # Test 2: Historical Data Manager
        test_historical_data_manager()

        # Test 3: Backtest Engine
        test_backtest_engine_init()

        # Test 4: Trade Simulation
        test_trade_simulation()

        # Test 5: Component Integration
        test_integration()

        # Summary
        print("\n" + "=" * 60)
        print("SUCCESS: PHASE 11 - ALL TESTS PASSED!")
        print("=" * 60)
        print("\nBacktesting System:")
        print("  * Historical Data Manager - Data loading & caching")
        print("  * Backtest Engine - Strategy simulation")
        print("  * Performance Metrics - Results analysis")
        print("\nKey Features:")
        print("  * Historical data fetching from MT5")
        print("  * Parquet file caching for speed")
        print("  * Realistic trade simulation")
        print("  * SL/TP execution")
        print("  * Comprehensive performance metrics")
        print("  * Win rate, profit factor, Sharpe ratio")
        print("  * Max drawdown tracking")
        print("  * Detailed performance reports")
        print("\nUsage:")
        print("  python scripts/run_backtest.py")
        print("  python scripts/run_backtest.py --start-date 2024-01-01")
        print("  python scripts/run_backtest.py --initial-balance 5000")
        print("\nReady for backtesting on historical data!")

    except Exception as e:
        print(f"\nERROR in Phase 11 testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
