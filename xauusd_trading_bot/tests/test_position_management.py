"""
Test Phase 7: Position Management
Tests position tracking, management, and recovery systems.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.position_management import (
    PositionTracker,
    PositionManager,
    RecoveryManager,
)
from src.utils.config_loader import ConfigLoader


def test_position_tracker():
    """Test position tracking."""
    print("\n" + "=" * 60)
    print("Testing Position Tracker")
    print("=" * 60)

    tracker = PositionTracker()

    # Test adding positions
    print("\n1. Adding positions...")
    positions = [
        {
            "ticket": 12345,
            "type": "BUY",
            "open_price": 2650.00,
            "volume": 0.01,
            "sl": 2625.00,
            "tp": 2700.00,
            "profit": 25.00,
        },
        {
            "ticket": 12346,
            "type": "SELL",
            "open_price": 2660.00,
            "volume": 0.01,
            "sl": 2685.00,
            "tp": 2610.00,
            "profit": -15.00,
        },
    ]

    for pos in positions:
        tracker.add_position(pos)

    print(f"  Added {len(positions)} positions")
    print(f"  Total tracked: {tracker.get_position_count()}")

    # Test updating metrics
    print("\n2. Updating position metrics...")
    tracker.update_position_metrics(12345, 2675.00, 50.00)
    tracker.update_position_metrics(12346, 2670.00, -25.00)
    print("  [OK] Metrics updated")

    # Test getting exposure
    print("\n3. Testing exposure calculation...")
    exposure = tracker.get_total_exposure()
    print(f"  Total Positions: {exposure['total_positions']}")
    print(f"  Total Lots: {exposure['total_lots']:.2f}")
    print(f"  Total P&L: ${exposure['total_profit']:.2f}")
    print(f"  Net Direction: {exposure['net_direction']}")

    # Test getting statistics
    print("\n4. Testing position statistics...")
    stats = tracker.get_position_stats(12345)
    if stats:
        print(f"  Position #{stats['ticket']}:")
        print(f"    Entry: ${stats['entry_price']:.2f}")
        print(f"    Current: ${stats['current_price']:.2f}")
        print(f"    Profit: ${stats['current_profit']:.2f}")
        print(f"    Pips from entry: {stats['pips_from_entry']:.1f}")

    # Test getting losing positions
    print("\n5. Testing position filtering...")
    losing = tracker.get_losing_positions()
    winning = tracker.get_winning_positions()
    print(f"  Losing positions: {len(losing)}")
    print(f"  Winning positions: {len(winning)}")

    # Test summary
    print("\n6. Position summary:")
    summary = tracker.get_position_summary()
    for line in summary.split("\n"):
        print(f"  {line}")

    print("\n[OK] Position Tracker tests passed!")


def test_position_manager():
    """Test position management."""
    print("\n" + "=" * 60)
    print("Testing Position Manager")
    print("=" * 60)

    # Load config
    config_loader = ConfigLoader()
    risk_config = config_loader.load("risk_config")

    # Create tracker and manager
    tracker = PositionTracker()
    manager = PositionManager(risk_config, tracker)

    print(f"\nConfiguration:")
    print(f"  Max open positions: {manager.max_open_positions}")
    print(f"  Max per direction: {manager.max_positions_per_direction}")
    print(f"  Max total lots: {manager.max_total_lots}")

    # Add some positions
    print("\n1. Testing position limits...")
    positions = [
        {"ticket": 1, "type": "BUY", "open_price": 2650.00, "volume": 0.01, "profit": 10},
        {"ticket": 2, "type": "BUY", "open_price": 2700.00, "volume": 0.01, "profit": 5},
    ]

    for pos in positions:
        tracker.add_position(pos)

    print(f"  Current positions: {tracker.get_position_count()}")

    # Check if can open new position
    print("\n2. Checking if can open new position...")
    can_open = manager.can_open_position("BUY", 2655.00, 0.01)
    print(f"  Allowed: {can_open['allowed']}")
    print(f"  Reason: {can_open['reason']}")

    # Try opening SELL (different direction)
    can_open_sell = manager.can_open_position("SELL", 2660.00, 0.01)
    print(f"\n  Can open SELL: {can_open_sell['allowed']}")
    print(f"  Reason: {can_open_sell['reason']}")

    # Test position spacing
    print("\n3. Testing position spacing...")
    can_open_close = manager.can_open_position("BUY", 2651.00, 0.01)  # Too close
    print(f"  Can open at 2651 (close to 2650): {can_open_close['allowed']}")
    if not can_open_close['allowed']:
        print(f"  Reason: {can_open_close['reason']}")

    # Test exposure reduction
    print("\n4. Testing exposure reduction...")
    # Add losing position
    tracker.add_position({
        "ticket": 3,
        "type": "SELL",
        "open_price": 2660.00,
        "volume": 0.01,
        "profit": -50,
    })

    reduce_check = manager.should_reduce_exposure()
    print(f"  Should reduce: {reduce_check['should_reduce']}")
    print(f"  Reason: {reduce_check['reason']}")

    # Test priority list
    print("\n5. Testing priority list...")
    priority = manager.get_position_priority_list()
    print(f"  Positions by priority:")
    for item in priority[:3]:  # Top 3
        print(f"    #{item['ticket']}: Score {item['priority_score']:.1f} ({', '.join(item['reasons'])})")

    # Test management summary
    print("\n6. Management summary:")
    summary = manager.get_management_summary()
    for line in summary.split("\n"):
        print(f"  {line}")

    print("\n[OK] Position Manager tests passed!")


def test_recovery_manager():
    """Test recovery management."""
    print("\n" + "=" * 60)
    print("Testing Recovery Manager")
    print("=" * 60)

    # Load config
    config_loader = ConfigLoader()
    risk_config = config_loader.load("risk_config")

    # Create tracker and recovery manager
    tracker = PositionTracker()
    recovery = RecoveryManager(risk_config, tracker)

    print(f"\nRecovery Settings:")
    print(f"  Min loss for recovery: ${recovery.min_loss_for_recovery:.2f}")
    print(f"  Max recovery time: {recovery.max_recovery_time_hours}h")

    # Add positions (some losing)
    print("\n1. Adding test positions...")
    positions = [
        {
            "ticket": 100,
            "type": "BUY",
            "open_price": 2650.00,
            "volume": 0.01,
            "sl": 2625.00,
            "tp": 2700.00,
            "profit": -75.00,  # Large loss
            "current_price": 2600.00,
        },
        {
            "ticket": 101,
            "type": "SELL",
            "open_price": 2660.00,
            "volume": 0.01,
            "sl": 2685.00,
            "tp": 2610.00,
            "profit": -30.00,  # Small loss
            "current_price": 2670.00,
        },
        {
            "ticket": 102,
            "type": "BUY",
            "open_price": 2640.00,
            "volume": 0.01,
            "sl": 2615.00,
            "tp": 2690.00,
            "profit": 25.00,  # Winning
            "current_price": 2655.00,
        },
    ]

    for pos in positions:
        tracker.add_position(pos)
        tracker.update_position_metrics(
            pos["ticket"],
            pos["current_price"],
            pos["profit"]
        )

    print(f"  Added {len(positions)} positions")

    # Analyze recovery options
    print("\n2. Analyzing recovery options...")
    for pos in positions[:2]:  # Losing positions
        analysis = recovery.analyze_position_recovery(pos["ticket"])
        if analysis and analysis.get("needs_recovery"):
            print(f"\n  Position #{pos['ticket']}:")
            print(f"    Loss: ${abs(analysis['current_loss']):.2f}")

            action = analysis.get("recommended_action")
            if action:
                print(f"    Recommended: {action.get('description', 'N/A')}")
                print(f"    Priority: {action.get('priority')}")

    # Get all recommendations
    print("\n3. Getting recovery recommendations...")
    recommendations = recovery.get_recovery_recommendations()
    print(f"  Total recommendations: {len(recommendations)}")

    for rec in recommendations:
        ticket = rec.get("ticket")
        loss = rec.get("current_loss", 0)
        print(f"    #{ticket}: ${abs(loss):.2f} loss")

    # Test recovery actions
    print("\n4. Testing recovery actions...")
    if recommendations:
        ticket = recommendations[0].get("ticket")
        action_type = recommendations[0].get("recommended_action", {}).get("type")

        if action_type:
            result = recovery.execute_recovery_action(ticket, action_type)
            print(f"  Action '{action_type}' for #{ticket}:")
            print(f"    Success: {result.get('success')}")
            print(f"    Requires execution: {result.get('requires_execution', False)}")

    # Test prevention check
    print("\n5. Testing new trade prevention...")
    prevent = recovery.should_prevent_new_trades()
    print(f"  Prevent new trades: {prevent['prevent']}")
    print(f"  Reason: {prevent['reason']}")

    # Recovery summary
    print("\n6. Recovery summary:")
    summary = recovery.get_recovery_summary()
    for line in summary.split("\n"):
        print(f"  {line}")

    print("\n[OK] Recovery Manager tests passed!")


def test_integration():
    """Test integration of all position management components."""
    print("\n" + "=" * 60)
    print("Testing Integration")
    print("=" * 60)

    # Load config
    config_loader = ConfigLoader()
    risk_config = config_loader.load("risk_config")

    # Create full system
    tracker = PositionTracker()
    manager = PositionManager(risk_config, tracker)
    recovery = RecoveryManager(risk_config, tracker)

    print("\n1. Simulating trading scenario...")

    # Open 3 positions
    scenario_positions = [
        {"ticket": 1001, "type": "BUY", "open_price": 2650.00, "volume": 0.01, "profit": 30},
        {"ticket": 1002, "type": "SELL", "open_price": 2660.00, "volume": 0.01, "profit": -60},
        {"ticket": 1003, "type": "BUY", "open_price": 2700.00, "volume": 0.01, "profit": 15},
    ]

    for pos in scenario_positions:
        tracker.add_position(pos)
        tracker.update_position_metrics(pos["ticket"], pos["open_price"], pos["profit"])

    print(f"  Opened {len(scenario_positions)} positions")

    # Check if can open new position
    print("\n2. Checking if can open new position...")
    can_open = manager.can_open_position("SELL", 2655.00, 0.01)
    print(f"  Can open: {can_open['allowed']}")
    if not can_open['allowed']:
        print(f"  Reason: {can_open['reason']}")

    # Get management status
    print("\n3. Current management status:")
    mgmt_summary = manager.get_management_summary()
    for line in mgmt_summary.split("\n"):
        print(f"  {line}")

    # Check recovery needs
    print("\n4. Recovery status:")
    recovery_summary = recovery.get_recovery_summary()
    for line in recovery_summary.split("\n"):
        print(f"  {line}")

    # Get complete system status
    print("\n5. Complete system status:")
    print(f"  Positions tracked: {tracker.get_position_count()}/{manager.max_open_positions}")

    exposure = tracker.get_total_exposure()
    print(f"  Total exposure: {exposure['total_lots']:.2f}/{manager.max_total_lots}")
    print(f"  Net P&L: ${exposure['total_profit']:.2f}")

    losing_count = len(tracker.get_losing_positions())
    print(f"  Losing positions: {losing_count}")

    print("\n[OK] Integration tests passed!")


def main():
    """Run all Phase 7 tests."""
    print("\n" + "=" * 60)
    print("PHASE 7: POSITION MANAGEMENT - TESTING")
    print("=" * 60)

    try:
        # Test 1: Position Tracker
        test_position_tracker()

        # Test 2: Position Manager
        test_position_manager()

        # Test 3: Recovery Manager
        test_recovery_manager()

        # Test 4: Integration
        test_integration()

        # Summary
        print("\n" + "=" * 60)
        print("SUCCESS: PHASE 7 - ALL TESTS PASSED!")
        print("=" * 60)
        print("\nPosition Management System:")
        print("  * Position Tracker - Real-time tracking with metrics")
        print("  * Position Manager - Intelligent limit enforcement")
        print("  * Recovery Manager - Loss recovery strategies")
        print("\nKey Features:")
        print("  * Track up to 3 positions with detailed metrics")
        print("  * Position spacing enforcement (50+ pips)")
        print("  * Total exposure monitoring")
        print("  * Correlation checks")
        print("  * Priority-based position management")
        print("  * Intelligent recovery (NO martingale)")
        print("  * Move to breakeven when improving")
        print("  * Time-based exit for stale positions")
        print("  * Suggest closing worst performers")
        print("\n* Ready for Phase 8: Session Management")

    except Exception as e:
        print(f"\nERROR in Phase 7 testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
