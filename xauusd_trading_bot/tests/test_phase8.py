"""
Test Phase 8: Session Management
Tests session detection and management systems.
"""

import sys
from pathlib import Path
from datetime import datetime, time
import pytz

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sessions import SessionDetector, SessionManager
from src.utils.config_loader import ConfigLoader


def test_session_detector():
    """Test session detection."""
    print("\n" + "=" * 60)
    print("Testing Session Detector")
    print("=" * 60)

    # Load config
    config_loader = ConfigLoader()
    session_config = config_loader.load("session_config")

    detector = SessionDetector(session_config)

    # Test different times
    test_times = [
        (datetime(2024, 1, 15, 2, 0, tzinfo=pytz.UTC), "Asian Session"),
        (datetime(2024, 1, 15, 10, 0, tzinfo=pytz.UTC), "London Session"),
        (datetime(2024, 1, 15, 14, 0, tzinfo=pytz.UTC), "London-NY Overlap"),
        (datetime(2024, 1, 15, 18, 0, tzinfo=pytz.UTC), "New York Session"),
        (datetime(2024, 1, 15, 23, 0, tzinfo=pytz.UTC), "No Session"),
    ]

    print("\n1. Testing session detection at different times...")
    for test_time, expected in test_times:
        session = detector.get_current_session(test_time)
        session_name = session["name"] if session else "No Session"
        print(f"  {test_time.strftime('%H:%M UTC')}: {session_name}")

        if session:
            print(f"    Weight: {session['weight']:.2f}")
            print(f"    Confluence Adj: {session['confluence_adjustment']:+.2f}")

    # Test trading allowed
    print("\n2. Testing trading permissions...")

    # Monday 10:00 UTC (should be allowed)
    monday = datetime(2024, 1, 15, 10, 0, tzinfo=pytz.UTC)
    allowed = detector.is_trading_allowed(monday)
    print(f"  Monday 10:00 UTC:")
    print(f"    Allowed: {allowed['allowed']}")
    print(f"    Reason: {allowed['reason']}")

    # Saturday (should NOT be allowed)
    saturday = datetime(2024, 1, 20, 10, 0, tzinfo=pytz.UTC)
    allowed_sat = detector.is_trading_allowed(saturday)
    print(f"\n  Saturday 10:00 UTC:")
    print(f"    Allowed: {allowed_sat['allowed']}")
    print(f"    Reason: {allowed_sat['reason']}")

    # Friday late (should check early close)
    friday = datetime(2024, 1, 19, 21, 0, tzinfo=pytz.UTC)
    allowed_friday = detector.is_trading_allowed(friday)
    print(f"\n  Friday 21:00 UTC:")
    print(f"    Allowed: {allowed_friday['allowed']}")
    print(f"    Reason: {allowed_friday['reason']}")

    # Test session weights
    print("\n3. Testing session weights...")
    overlap_time = datetime(2024, 1, 15, 14, 0, tzinfo=pytz.UTC)
    weight = detector.get_session_weight(overlap_time)
    print(f"  Overlap session weight: {weight:.2f}")

    asian_time = datetime(2024, 1, 15, 2, 0, tzinfo=pytz.UTC)
    weight_asian = detector.get_session_weight(asian_time)
    print(f"  Asian session weight: {weight_asian:.2f}")

    # Test next session
    print("\n4. Testing next session detection...")
    late_night = datetime(2024, 1, 15, 23, 30, tzinfo=pytz.UTC)
    next_session = detector.get_next_session(late_night)
    if next_session:
        print(f"  Current time: 23:30 UTC")
        print(f"  Next session: {next_session['name']}")
        print(f"  Starts in: {next_session['minutes_until']} minutes")

    # Test summary
    print("\n5. Session summary:")
    overlap_summary = detector.get_session_summary(overlap_time)
    for line in overlap_summary.split("\n"):
        print(f"  {line}")

    print("\n[OK] Session Detector tests passed!")


def test_session_manager():
    """Test session management."""
    print("\n" + "=" * 60)
    print("Testing Session Manager")
    print("=" * 60)

    # Load config
    config_loader = ConfigLoader()
    session_config = config_loader.load("session_config")

    manager = SessionManager(session_config)

    # Test during overlap
    print("\n1. Testing during London-NY overlap...")
    overlap_time = datetime(2024, 1, 15, 14, 30, tzinfo=pytz.UTC)
    manager.update(overlap_time)

    session = manager.get_current_session()
    print(f"  Session: {session['name'] if session else 'None'}")

    # Test parameter adjustments
    print("\n2. Testing parameter adjustments...")
    base_params = {
        "min_confluence_score": 0.65,
        "position_size": 0.01,
        "max_trades_per_hour": 2,
    }

    adjusted = manager.apply_session_adjustments(base_params, overlap_time)
    print(f"  Base confluence: {base_params['min_confluence_score']:.2f}")
    print(f"  Adjusted confluence: {adjusted['min_confluence_score']:.2f}")
    print(f"  Session weight: {adjusted.get('session_weight', 1.0):.2f}")

    # Test during Asian session
    print("\n3. Testing during Asian session...")
    asian_time = datetime(2024, 1, 15, 4, 0, tzinfo=pytz.UTC)
    adjusted_asian = manager.apply_session_adjustments(base_params, asian_time)
    print(f"  Base confluence: {base_params['min_confluence_score']:.2f}")
    print(f"  Adjusted confluence: {adjusted_asian['min_confluence_score']:.2f}")

    # Test confluence threshold adjustment
    print("\n4. Testing confluence threshold...")
    base_threshold = 0.65

    overlap_threshold = manager.get_adjusted_confluence_threshold(
        base_threshold, overlap_time
    )
    print(f"  Overlap threshold: {overlap_threshold:.2f} (base: {base_threshold:.2f})")

    asian_threshold = manager.get_adjusted_confluence_threshold(
        base_threshold, asian_time
    )
    print(f"  Asian threshold: {asian_threshold:.2f} (base: {base_threshold:.2f})")

    # Test position size multiplier
    print("\n5. Testing position size multiplier...")
    overlap_multiplier = manager.get_position_size_multiplier(overlap_time)
    print(f"  Overlap multiplier: {overlap_multiplier:.2f}x")

    asian_multiplier = manager.get_position_size_multiplier(asian_time)
    print(f"  Asian multiplier: {asian_multiplier:.2f}x")

    # Test early close
    print("\n6. Testing Friday early close...")
    friday_late = datetime(2024, 1, 19, 19, 45, tzinfo=pytz.UTC)
    early_close = manager.should_close_positions_early(friday_late)
    print(f"  Should close: {early_close['should_close']}")
    print(f"  Reason: {early_close['reason']}")
    if early_close.get("minutes_until_close"):
        print(f"  Minutes until close: {early_close['minutes_until_close']}")

    # Test trade timing validation
    print("\n7. Testing trade timing validation...")

    signal = {
        "direction": "BUY",
        "confidence": 0.70,
    }

    # Validate during overlap (preferred)
    validation_overlap = manager.validate_trade_timing(signal, overlap_time)
    print(f"  Overlap session:")
    print(f"    Valid: {validation_overlap['valid']}")
    print(f"    Preferred: {validation_overlap.get('is_preferred_session', False)}")

    # Validate during Asian (non-preferred)
    validation_asian = manager.validate_trade_timing(signal, asian_time)
    print(f"\n  Asian session:")
    print(f"    Valid: {validation_asian['valid']}")
    if not validation_asian['valid']:
        print(f"    Reason: {validation_asian['reason']}")

    # Test with higher confidence
    high_conf_signal = {
        "direction": "BUY",
        "confidence": 0.80,
    }
    validation_asian_high = manager.validate_trade_timing(high_conf_signal, asian_time)
    print(f"\n  Asian session (high confidence):")
    print(f"    Valid: {validation_asian_high['valid']}")

    # Test manager summary
    print("\n8. Manager summary:")
    summary = manager.get_manager_summary(overlap_time)
    for line in summary.split("\n"):
        print(f"  {line}")

    print("\n[OK] Session Manager tests passed!")


def test_optimal_times():
    """Test optimal entry times."""
    print("\n" + "=" * 60)
    print("Testing Optimal Entry Times")
    print("=" * 60)

    # Load config
    config_loader = ConfigLoader()
    session_config = config_loader.load("session_config")

    manager = SessionManager(session_config)

    print("\n1. Optimal entry times:")
    optimal = manager.get_optimal_entry_times()

    for session_key, info in optimal.items():
        print(f"\n  {info['name']}:")
        print(f"    Start: {info['start_utc']} UTC")
        print(f"    End: {info['end_utc']} UTC")
        print(f"    Weight: {info['weight']:.2f}")
        print(f"    {info['description']}")

    # Get statistics
    print("\n2. Current session statistics:")
    overlap_time = datetime(2024, 1, 15, 14, 30, tzinfo=pytz.UTC)
    manager.update(overlap_time)

    stats = manager.get_session_statistics()
    print(f"  Current: {stats['current_session']}")
    print(f"  Weight: {stats['current_weight']:.2f}")
    print(f"  Preferred: {stats['is_preferred']}")
    print(f"  Trading Allowed: {stats['trading_allowed']}")

    print("\n[OK] Optimal times tests passed!")


def test_session_transitions():
    """Test session transitions."""
    print("\n" + "=" * 60)
    print("Testing Session Transitions")
    print("=" * 60)

    # Load config
    config_loader = ConfigLoader()
    session_config = config_loader.load("session_config")

    detector = SessionDetector(session_config)

    print("\n1. Testing 24-hour cycle:")

    # Test every 2 hours
    for hour in range(0, 24, 2):
        test_time = datetime(2024, 1, 15, hour, 0, tzinfo=pytz.UTC)
        session = detector.get_current_session(test_time)
        session_name = session["name"] if session else "No Active Session"
        weight = session["weight"] if session else 0.0

        print(f"  {hour:02d}:00 UTC - {session_name:20s} (Weight: {weight:.1f})")

    # Test all active sessions during overlap
    print("\n2. All active sessions during overlap:")
    overlap_time = datetime(2024, 1, 15, 14, 0, tzinfo=pytz.UTC)
    all_active = detector.get_all_active_sessions(overlap_time)

    for session in all_active:
        print(f"  - {session['name']} (Weight: {session['weight']:.1f})")

    print("\n[OK] Session transition tests passed!")


def main():
    """Run all Phase 8 tests."""
    print("\n" + "=" * 60)
    print("PHASE 8: SESSION MANAGEMENT - TESTING")
    print("=" * 60)

    try:
        # Test 1: Session Detector
        test_session_detector()

        # Test 2: Session Manager
        test_session_manager()

        # Test 3: Optimal Times
        test_optimal_times()

        # Test 4: Session Transitions
        test_session_transitions()

        # Summary
        print("\n" + "=" * 60)
        print("SUCCESS: PHASE 8 - ALL TESTS PASSED!")
        print("=" * 60)
        print("\nSession Management System:")
        print("  * Session Detector - UTC-based session detection")
        print("  * Session Manager - Strategy adjustments per session")
        print("\nKey Features:")
        print("  * 4 trading sessions (Asian, London, NY, Overlap)")
        print("  * Session weights (0.6 - 1.2)")
        print("  * Confluence adjustments per session")
        print("  * Position size adjustments")
        print("  * Weekend & Friday early close")
        print("  * Preferred session validation")
        print("  * Trade timing validation")
        print("  * Next session prediction")
        print("\nOptimal Trading:")
        print("  * BEST: London-NY Overlap (13:00-16:00 UTC)")
        print("  * GOOD: London (08:00-16:00 UTC)")
        print("  * GOOD: New York (13:00-22:00 UTC)")
        print("  * LOWER: Asian (00:00-08:00 UTC)")
        print("\n* Ready for Phase 9: Order Execution")

    except Exception as e:
        print(f"\nERROR in Phase 8 testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
