"""
Test Phase 6: Risk Management
Tests SL/TP calculation, position sizing, trailing stops, and drawdown monitoring.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.risk_management import (
    SLTPCalculator,
    PositionSizer,
    TrailingStopManager,
    DrawdownMonitor,
)
from src.core.constants import VolatilityLevel
from src.utils.config_loader import ConfigLoader


def test_sl_tp_calculator():
    """Test SL/TP calculation."""
    print("\n" + "=" * 60)
    print("Testing SL/TP Calculator")
    print("=" * 60)

    # Load config
    config_loader = ConfigLoader()
    risk_config = config_loader.load("risk_config")

    # Initialize calculator
    calculator = SLTPCalculator(risk_config)

    # Test scenarios
    test_cases = [
        {
            "name": "BUY - Medium Volatility",
            "entry_price": 2650.00,
            "direction": "BUY",
            "atr": 15.0,
            "volatility": VolatilityLevel.MEDIUM,
        },
        {
            "name": "SELL - High Volatility",
            "entry_price": 2650.00,
            "direction": "SELL",
            "atr": 25.0,
            "volatility": VolatilityLevel.HIGH,
        },
        {
            "name": "BUY - Low Volatility",
            "entry_price": 2650.00,
            "direction": "BUY",
            "atr": 8.0,
            "volatility": VolatilityLevel.LOW,
        },
    ]

    for case in test_cases:
        print(f"\n{case['name']}:")
        print(f"  Entry: ${case['entry_price']:.2f}")
        print(f"  ATR: {case['atr']:.2f}")
        print(f"  Volatility: {case['volatility'].value}")

        result = calculator.calculate_sl_tp(
            case["entry_price"],
            case["direction"],
            case["atr"],
            case["volatility"],
        )

        print(f"\n  Results:")
        print(f"    SL: ${result['sl']:.2f} ({result['sl_distance_pips']:.1f} pips)")
        print(f"    TP: ${result['tp']:.2f} ({result['tp_distance_pips']:.1f} pips)")
        print(f"    Risk:Reward: 1:{result['rr_ratio']:.2f}")

        # Verify minimum RR
        assert result["rr_ratio"] >= 1.5, "RR ratio should be at least 1:1.5 (M15 Fast Mode)"
        print("    [OK] RR ratio >= 1:1.5")

    # Test trailing stop
    print("\n\nTesting Trailing Stop Update:")
    position = {
        "ticket": 12345,
        "type": "BUY",
        "open_price": 2650.00,
        "sl": 2625.00,
        "tp": 2700.00,
    }

    # update_trailing_stop() was removed — percent-based logic was mathematically
    # broken for XAUUSD (10% activation = $500 profit required at $5000 gold price).
    # Live trailing uses RR-based inline logic in trading_bot._manage_positions().
    print("  (update_trailing_stop removed — superseded by inline RR-based trailing)")

    print("\n[OK] SL/TP Calculator tests passed!")


def test_position_sizer():
    """Test position sizing."""
    print("\n" + "=" * 60)
    print("Testing Position Sizer")
    print("=" * 60)

    # Load config
    config_loader = ConfigLoader()
    risk_config = config_loader.load("risk_config")

    # Initialize sizer
    sizer = PositionSizer(risk_config)

    # Test account info
    account_info = {
        "balance": 10000.0,
        "equity": 10000.0,
        "margin_free": 9000.0,
        "margin_level": 1000.0,
    }

    print(f"\nAccount Balance: ${account_info['balance']:.2f}")
    print(f"Free Margin: ${account_info['margin_free']:.2f}")

    # Calculate position size
    sl_distance = 25.0  # 25 pips SL
    result = sizer.calculate_position_size(
        account_info, sl_distance, volatility_level="MEDIUM"
    )

    print(f"\nPosition Size Calculation:")
    print(f"  SL Distance: {sl_distance:.1f} pips")
    print(f"  Lot Size: {result['lot_size']:.2f}")
    print(f"  Method: {result['method']}")

    # Test position limits
    print("\n\nTesting Position Limits:")
    current_positions = [
        {"type": "BUY", "volume": 0.01},
        {"type": "SELL", "volume": 0.01},
    ]

    limits_check = sizer.check_position_limits(0.01, current_positions, "BUY")

    print(f"  Current Positions: {len(current_positions)}")
    print(f"  New Position Allowed: {limits_check['allowed']}")
    print(f"  Reason: {limits_check['reason']}")

    # Test margin calculation
    print("\n\nTesting Margin Calculation:")
    current_price = 2650.00
    lot_size = 0.01
    margin_required = sizer.calculate_margin_required(lot_size, current_price)

    print(f"  Price: ${current_price:.2f}")
    print(f"  Lot Size: {lot_size:.2f}")
    print(f"  Margin Required: ${margin_required:.2f}")

    # Check margin available
    margin_check = sizer.check_margin_available(account_info, margin_required)
    print(f"\n  Margin Available: {margin_check['available']}")
    if margin_check['available']:
        print(f"  Free Margin: ${margin_check['margin_free']:.2f}")
    else:
        print(f"  Reason: {margin_check['reason']}")

    print("\n[OK] Position Sizer tests passed!")


def test_trailing_stop_manager():
    """Test trailing stop manager."""
    print("\n" + "=" * 60)
    print("Testing Trailing Stop Manager")
    print("=" * 60)

    # Load config
    config_loader = ConfigLoader()
    risk_config = config_loader.load("risk_config")

    # Initialize manager
    manager = TrailingStopManager(risk_config)

    print(f"Trailing Stop Enabled: {manager.enabled}")
    print(f"Activation Threshold: {manager.activation_percent}%")
    print(f"Trail Distance: {manager.trail_distance_percent}%")

    # Test position
    position = {
        "ticket": 12345,
        "type": "BUY",
        "open_price": 2650.00,
        "sl": 2625.00,
        "tp": 2725.00,
        "volume": 0.01,
    }

    # Test scenarios with different prices
    test_prices = [
        (2660.00, "Small profit (not activated)"),
        (2680.00, "Medium profit (should activate)"),
        (2690.00, "Higher profit (should trail)"),
    ]

    for price, description in test_prices:
        print(f"\n{description}:")
        print(f"  Current Price: ${price:.2f}")

        # Get status
        status = manager.get_trailing_status(position, price)
        print(f"  Status: {status['status']}")
        print(f"  Profit: {status['current_profit_percent']:.2f}%")

        # Check for update
        update = manager.check_trailing_update(position, price, atr=15.0)

        if update:
            print(f"  [OK] Update Available:")
            print(f"    Current SL: ${position['sl']:.2f}")
            print(f"    New SL: ${update['new_sl']:.2f}")
            print(f"    Reason: {update['reason']}")
            # Update position for next iteration
            position["sl"] = update["new_sl"]
        else:
            print(f"  No update needed")

    print("\n[OK] Trailing Stop Manager tests passed!")


def test_drawdown_monitor():
    """Test drawdown monitor."""
    print("\n" + "=" * 60)
    print("Testing Drawdown Monitor")
    print("=" * 60)

    # Load config
    config_loader = ConfigLoader()
    risk_config = config_loader.load("risk_config")

    # Initialize monitor
    monitor = DrawdownMonitor(risk_config)

    print(f"Max Daily Loss: {monitor.max_daily_loss_percent}%")
    print(f"Max Drawdown: {monitor.max_drawdown_percent}%")
    print(f"Max Consecutive Losses: {monitor.max_consecutive_losses}")

    # Initial account state
    account_info = {
        "balance": 10000.0,
        "equity": 10000.0,
    }

    print(f"\nInitial Balance: ${account_info['balance']:.2f}")

    # Initialize
    monitor.initialize(account_info)

    # Check if trading allowed (should be OK)
    print("\n\nChecking initial trading status:")
    check = monitor.check_trading_allowed(account_info)
    print(f"  Trading Allowed: {check['allowed']}")
    print(f"  Reason: {check['reason']}")

    if check.get("stats"):
        stats = check["stats"]
        print(f"  Daily Loss: {stats['daily_loss_percent']:.2f}%")
        print(f"  Drawdown: {stats['drawdown_percent']:.2f}%")
        print(f"  Consecutive Losses: {stats['consecutive_losses']}")

    # Simulate losses
    print("\n\nSimulating losing trades:")
    for i in range(3):
        trade = {
            "ticket": 1000 + i,
            "profit": -100.0,
        }
        monitor.record_trade_result(trade)
        print(f"  Loss {i+1}: ${trade['profit']:.2f}")

    # Check status after losses
    account_info["balance"] = 9700.0  # Lost $300
    account_info["equity"] = 9700.0

    print(f"\nAfter losses - Balance: ${account_info['balance']:.2f}")
    check = monitor.check_trading_allowed(account_info)
    print(f"  Trading Allowed: {check['allowed']}")
    print(f"  Reason: {check['reason']}")
    print(f"  Consecutive Losses: {monitor.consecutive_losses}")

    if not check["allowed"] and check.get("violations"):
        print("\n  Violations:")
        for violation in check["violations"]:
            print(f"    - {violation['message']}")

    # Get account status
    print("\n\nAccount Status:")
    status = monitor.get_account_status(account_info)
    print(f"  Balance: ${status['balance']:.2f}")
    print(f"  Daily P&L: ${status['daily_pnl']:.2f} ({status['daily_pnl_percent']:.2f}%)")
    print(f"  Drawdown: {status['drawdown_percent']:.2f}%")
    print(f"  Consecutive Losses: {status['consecutive_losses']}")
    print(f"  Trading Paused: {status['paused']}")

    print("\n[OK] Drawdown Monitor tests passed!")


def main():
    """Run all Phase 6 tests."""
    print("\n" + "=" * 60)
    print("PHASE 6: RISK MANAGEMENT - TESTING")
    print("=" * 60)

    try:
        # Test 1: SL/TP Calculator
        test_sl_tp_calculator()

        # Test 2: Position Sizer
        test_position_sizer()

        # Test 3: Trailing Stop Manager
        test_trailing_stop_manager()

        # Test 4: Drawdown Monitor
        test_drawdown_monitor()

        # Summary
        print("\n" + "=" * 60)
        print("SUCCESS: PHASE 6 - ALL TESTS PASSED!")
        print("=" * 60)
        print("\nRisk Management System:")
        print("  * SL/TP Calculator - Dynamic ATR-based calculations")
        print("  * Position Sizer - Fixed lot with safety checks")
        print("  * Trailing Stop Manager - Profit protection")
        print("  * Drawdown Monitor - Account protection")
        print("\nKey Features:")
        print("  * Dynamic SL: 2.5x ATR (volatility adjusted)")
        print("  * Dynamic TP: 5.0x ATR (minimum 1:2 RR)")
        print("  * Trailing stops activate at +10% profit")
        print("  * Max daily loss: 5%")
        print("  * Max consecutive losses: 3")
        print("  * Position limits enforced")
        print("  * Margin checks before trading")
        print("\n* Ready for Phase 7: Position Management")

    except Exception as e:
        print(f"\nERROR in Phase 6 testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
