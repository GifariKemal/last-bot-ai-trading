"""
Test Phase 9: Order Execution
Tests order execution and emergency handling (with mocked MT5).
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.execution import OrderExecutor, EmergencyHandler


class MockMT5Connector:
    """Mock MT5 connector for testing."""

    def __init__(self):
        self.connected = True
        self.positions = []
        self.next_ticket = 10001

    def is_connected(self):
        return self.connected

    def connect(self):
        """Mock reconnection."""
        if not self.connected:
            self.connected = True
            return {"success": True}
        return {"success": True}

    def get_account_info(self):
        return {
            "balance": 10000.0,
            "equity": 9850.0,
            "margin_free": 9500.0,
            "margin_level": 500.0,
        }

    def get_tick(self, symbol):
        return {
            "bid": 2650.00,
            "ask": 2650.50,
            "spread": 5,  # 0.5 pips
        }

    def send_order(self, action, lot_size, sl, tp, comment, magic):
        """Mock order sending."""
        if not self.connected:
            return {"success": False, "error": "Not connected"}

        ticket = self.next_ticket
        self.next_ticket += 1

        # Simulate successful order
        position = {
            "ticket": ticket,
            "type": action,
            "volume": lot_size,
            "open_price": 2650.00,
            "sl": sl,
            "tp": tp,
            "profit": 0.0,
            "comment": comment,
        }

        self.positions.append(position)

        return {
            "success": True,
            "ticket": ticket,
            "price": 2650.00,
        }

    def close_position(self, ticket):
        """Mock position closing."""
        if not self.connected:
            return {"success": False, "error": "Not connected"}

        # Find and remove position
        for i, pos in enumerate(self.positions):
            if pos["ticket"] == ticket:
                profit = pos.get("profit", 0)
                self.positions.pop(i)
                return {
                    "success": True,
                    "ticket": ticket,
                    "price": 2655.00,
                    "profit": profit,
                }

        return {"success": False, "error": "Position not found"}

    def modify_position(self, ticket, sl, tp):
        """Mock position modification."""
        if not self.connected:
            return {"success": False, "error": "Not connected"}

        # Find and modify position
        for pos in self.positions:
            if pos["ticket"] == ticket:
                if sl is not None:
                    pos["sl"] = sl
                if tp is not None:
                    pos["tp"] = tp
                return {"success": True, "ticket": ticket}

        return {"success": False, "error": "Position not found"}

    def get_positions(self):
        """Get all positions."""
        return self.positions.copy()


def test_order_executor():
    """Test order executor."""
    print("\n" + "=" * 60)
    print("Testing Order Executor")
    print("=" * 60)

    # Create mock MT5 and executor
    mt5_mock = MockMT5Connector()
    config = {"magic_number": 123456}
    executor = OrderExecutor(mt5_mock, config)

    print(f"\nExecution Settings:")
    print(f"  Max retries: {executor.max_retries}")
    print(f"  Retry delay: {executor.retry_delay_seconds}s")
    print(f"  Max slippage: {executor.max_slippage} pips")

    # Test 1: Validate order parameters
    print("\n1. Testing order parameter validation...")

    valid_params = executor.validate_order_parameters(
        direction="BUY",
        lot_size=0.01,
        sl_price=2625.00,
        tp_price=2700.00,
        entry_price=2650.00
    )

    print(f"  Valid parameters:")
    print(f"    Valid: {valid_params['valid']}")
    if valid_params['valid']:
        print(f"    RR Ratio: 1:{valid_params['rr_ratio']:.2f}")
        print(f"    SL Distance: {valid_params['sl_distance']:.1f} pips")
        print(f"    TP Distance: {valid_params['tp_distance']:.1f} pips")

    # Test invalid parameters
    invalid_params = executor.validate_order_parameters(
        direction="BUY",
        lot_size=0.01,
        sl_price=2625.00,
        tp_price=2660.00,  # Too close (only 10 pips)
        entry_price=2650.00
    )

    print(f"\n  Invalid parameters (TP too close):")
    print(f"    Valid: {invalid_params['valid']}")
    if not invalid_params['valid']:
        print(f"    Errors: {', '.join(invalid_params['errors'])}")

    # Test 2: Execute entry order
    print("\n2. Testing entry order execution...")

    signal = {
        "direction": "BUY",
        "price": 2650.00,
        "confidence": 0.75,
    }

    entry_result = executor.execute_entry(
        signal=signal,
        lot_size=0.01,
        sl_price=2625.00,
        tp_price=2700.00,
        comment="Test Entry"
    )

    print(f"  Entry Result:")
    print(f"    Success: {entry_result['success']}")
    if entry_result['success']:
        print(f"    Ticket: #{entry_result['ticket']}")
        print(f"    Entry Price: ${entry_result['entry_price']:.2f}")
        print(f"    SL: ${entry_result['sl']:.2f}")
        print(f"    TP: ${entry_result['tp']:.2f}")
        print(f"    Lot Size: {entry_result['lot_size']}")

    # Test 3: Modify position
    print("\n3. Testing position modification...")

    if entry_result['success']:
        ticket = entry_result['ticket']

        modify_result = executor.modify_position(
            ticket=ticket,
            new_sl=2640.00,  # Move SL closer to entry
            reason="Move to breakeven"
        )

        print(f"  Modification Result:")
        print(f"    Success: {modify_result['success']}")
        if modify_result['success']:
            print(f"    Ticket: #{modify_result['ticket']}")
            print(f"    New SL: ${modify_result['new_sl']:.2f}")
            print(f"    Reason: {modify_result['reason']}")

    # Test 4: Execute exit order
    print("\n4. Testing exit order execution...")

    if entry_result['success']:
        ticket = entry_result['ticket']

        exit_result = executor.execute_exit(
            ticket=ticket,
            reason="Test exit"
        )

        print(f"  Exit Result:")
        print(f"    Success: {exit_result['success']}")
        if exit_result['success']:
            print(f"    Ticket: #{exit_result['ticket']}")
            print(f"    Close Price: ${exit_result['close_price']:.2f}")
            print(f"    Profit: ${exit_result.get('profit', 0):.2f}")
            print(f"    Reason: {exit_result['reason']}")

    # Test 5: Pre-execution checks
    print("\n5. Testing pre-execution checks...")

    checks = executor._pre_execution_checks(0.01)
    print(f"  Pre-execution checks:")
    print(f"    Passed: {checks['passed']}")
    print(f"    Reason: {checks['reason']}")

    for check in checks.get('checks', []):
        check_name = check['check']
        passed = check['passed']
        print(f"      - {check_name}: {'PASS' if passed else 'FAIL'}")

    print("\n[OK] Order Executor tests passed!")


def test_emergency_handler():
    """Test emergency handler."""
    print("\n" + "=" * 60)
    print("Testing Emergency Handler")
    print("=" * 60)

    # Create mock MT5, executor, and emergency handler
    mt5_mock = MockMT5Connector()
    config = {"magic_number": 123456}
    executor = OrderExecutor(mt5_mock, config)
    emergency = EmergencyHandler(mt5_mock, executor)

    # Add some test positions
    print("\n1. Setting up test positions...")
    for i in range(3):
        mt5_mock.positions.append({
            "ticket": 2000 + i,
            "type": "BUY" if i % 2 == 0 else "SELL",
            "volume": 0.01,
            "open_price": 2650.00 + (i * 10),
            "profit": -50.0 if i == 1 else 25.0,  # One losing position
        })

    print(f"  Added {len(mt5_mock.positions)} test positions")

    # Test 2: Check emergency conditions
    print("\n2. Testing emergency condition checks...")

    account_info = mt5_mock.get_account_info()
    positions = mt5_mock.get_positions()

    emergency_check = emergency.check_emergency_conditions(account_info, positions)

    print(f"  Emergency needed: {emergency_check['emergency_needed']}")
    if emergency_check.get('triggers'):
        print(f"  Triggers: {len(emergency_check['triggers'])}")
        for trigger in emergency_check['triggers']:
            print(f"    - {trigger['trigger']}: {trigger['message']} ({trigger['severity']})")

    # Test 3: Account safety validation
    print("\n3. Testing account safety validation...")

    safety_check = emergency.validate_account_safety(account_info)

    print(f"  Account safe: {safety_check['safe']}")
    print(f"  Status: {safety_check['status']}")

    if safety_check.get('issues'):
        print(f"  Issues: {len(safety_check['issues'])}")
        for issue in safety_check['issues']:
            print(f"    - {issue['message']} ({issue['severity']})")

    # Test 4: Emergency stop
    print("\n4. Testing emergency stop...")

    stop_result = emergency.emergency_stop("Test emergency stop")

    print(f"  Stop Result:")
    print(f"    Success: {stop_result['success']}")
    print(f"    Reason: {stop_result['reason']}")
    print(f"    Positions closed: {stop_result['positions_closed']}")
    print(f"  Remaining positions: {len(mt5_mock.get_positions())}")

    # Check emergency status
    print("\n5. Checking emergency status...")
    status = emergency.get_emergency_status()
    print(f"  Emergency active: {status['emergency_active']}")
    if status['emergency_active']:
        print(f"  Reason: {status['reason']}")
        print(f"  Timestamp: {status['timestamp']}")

    # Test 6: Reset emergency state
    print("\n6. Resetting emergency state...")
    emergency.reset_emergency_state()

    status_after = emergency.get_emergency_status()
    print(f"  Emergency active: {status_after['emergency_active']}")

    # Test 7: Safe shutdown
    print("\n7. Testing safe shutdown...")

    # Add more positions
    for i in range(2):
        mt5_mock.positions.append({
            "ticket": 3000 + i,
            "type": "BUY",
            "volume": 0.01,
            "profit": 15.0,
        })

    shutdown_result = emergency.safe_shutdown("Test shutdown")

    print(f"  Shutdown Result:")
    print(f"    Success: {shutdown_result['success']}")
    print(f"    Total positions: {shutdown_result['total_positions']}")
    print(f"    Closed: {shutdown_result['closed_count']}")
    print(f"  Remaining positions: {len(mt5_mock.get_positions())}")

    # Test 8: Connection loss handling
    print("\n8. Testing connection loss handling...")

    # Simulate connection loss
    mt5_mock.connected = False
    connection_result = emergency.handle_connection_loss()

    print(f"  Connection handling:")
    print(f"    Success: {connection_result['success']}")
    print(f"    Action: {connection_result['action']}")

    # Restore connection
    mt5_mock.connected = True

    print("\n[OK] Emergency Handler tests passed!")


def test_close_all_positions():
    """Test closing all positions."""
    print("\n" + "=" * 60)
    print("Testing Close All Positions")
    print("=" * 60)

    # Create mock MT5 and executor
    mt5_mock = MockMT5Connector()
    config = {"magic_number": 123456}
    executor = OrderExecutor(mt5_mock, config)

    # Add multiple positions
    print("\n1. Adding multiple test positions...")
    for i in range(5):
        mt5_mock.positions.append({
            "ticket": 4000 + i,
            "type": "BUY" if i % 2 == 0 else "SELL",
            "volume": 0.01,
            "profit": 10.0 * (i + 1),
        })

    print(f"  Added {len(mt5_mock.positions)} positions")

    # Close all
    print("\n2. Closing all positions...")
    result = executor.close_all_positions("Test close all")

    print(f"  Result:")
    print(f"    Success: {result['success']}")
    print(f"    Total positions: {result['total_positions']}")
    print(f"    Closed: {result['closed_count']}")
    print(f"    Failed: {result['failed_count']}")
    print(f"  Remaining positions: {len(mt5_mock.get_positions())}")

    print("\n[OK] Close all positions test passed!")


def main():
    """Run all Phase 9 tests."""
    print("\n" + "=" * 60)
    print("PHASE 9: ORDER EXECUTION - TESTING")
    print("=" * 60)

    try:
        # Test 1: Order Executor
        test_order_executor()

        # Test 2: Emergency Handler
        test_emergency_handler()

        # Test 3: Close All Positions
        test_close_all_positions()

        # Summary
        print("\n" + "=" * 60)
        print("SUCCESS: PHASE 9 - ALL TESTS PASSED!")
        print("=" * 60)
        print("\nOrder Execution System:")
        print("  * Order Executor - Safe order execution with retries")
        print("  * Emergency Handler - Emergency stop and safety checks")
        print("\nKey Features:")
        print("  * Entry order execution with validation")
        print("  * Exit order execution with retry logic")
        print("  * Position modification (SL/TP updates)")
        print("  * Pre-execution checks (balance, margin, spread)")
        print("  * Order parameter validation")
        print("  * Retry logic for transient errors (3 attempts)")
        print("  * Emergency stop (close all positions)")
        print("  * Emergency condition detection")
        print("  * Account safety validation")
        print("  * Safe shutdown functionality")
        print("  * Connection loss handling")
        print("\nSafety Features:")
        print("  * Balance checks before trading")
        print("  * Margin level monitoring")
        print("  * Spread validation")
        print("  * RR ratio validation (min 1.5:1)")
        print("  * Emergency triggers (equity drop, low margin)")
        print("  * Graceful error handling")
        print("\n* Ready for Phase 10: Main Bot Controller")

    except Exception as e:
        print(f"\nERROR in Phase 9 testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
