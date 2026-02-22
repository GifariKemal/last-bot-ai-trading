"""
Validation Loop Orchestrator
=============================
Run before every deploy to IC Markets live account.

Usage:
    python validate.py              # run once
    python validate.py --loop 3     # retry up to 3 times on failure (auto-debug hint)

Exit codes:
    0 = all tests passed → safe to deploy
    1 = tests failed     → DO NOT deploy
"""
import subprocess
import sys
import argparse
from datetime import datetime


BANNER = """
==============================================
  SMART TRADER - VALIDATION LOOP
  Run before every live deployment
==============================================
"""

TEST_SUITES = [
    ("Claude Parser",  "tests/test_claude_parser.py"),
    ("Indicators",     "tests/test_indicators.py"),
    ("Scanner",        "tests/test_scanner.py"),
    ("Zone Detector",  "tests/test_zone_detector.py"),
]


def run_suite(name: str, path: str, verbose: bool = False) -> tuple[bool, str]:
    """Run a single test suite. Returns (passed, output)."""
    flags = ["-v"] if verbose else ["-q"]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", path, "--tb=short", *flags],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    return result.returncode == 0, output


def main():
    parser = argparse.ArgumentParser(description="Smart Trader validation loop")
    parser.add_argument("--loop", type=int, default=1,
                        help="Max retry attempts on failure (default: 1)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show verbose test output")
    args = parser.parse_args()

    print(BANNER)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  Started: {timestamp}")
    print(f"  Suites:  {len(TEST_SUITES)}\n")

    for attempt in range(1, args.loop + 1):
        if args.loop > 1:
            print(f"── Attempt {attempt}/{args.loop} ─────────────────────────────")

        results = []
        total_passed = total_failed = 0

        for name, path in TEST_SUITES:
            passed, output = run_suite(name, path, verbose=args.verbose)

            # Extract counts from pytest output
            counts = _parse_counts(output)
            results.append((name, passed, counts, output))

            status = "[PASS]" if passed else "[FAIL]"
            print(f"  {status}  {name:<20} {counts}")

            if passed:
                total_passed += 1
            else:
                total_failed += 1

        print()

        if total_failed == 0:
            print("  ==========================================")
            print(f"  ALL {total_passed} SUITES PASSED")
            print("  ==========================================")
            print("  -> Safe to deploy to IC Markets live account\n")
            _update_task_status(6, "completed")
            sys.exit(0)
        else:
            print("  ==========================================")
            print(f"  {total_failed} SUITE(S) FAILED")
            print("  ==========================================")

            # Show failure details
            for name, passed, counts, output in results:
                if not passed:
                    print(f"\n  -- {name} failure details --")
                    # Print only the FAILURES section
                    lines = output.split("\n")
                    in_failures = False
                    for line in lines:
                        if "FAILURES" in line or "FAILED" in line:
                            in_failures = True
                        if in_failures:
                            print(f"  {line}")

            if attempt < args.loop:
                print(f"\n  Retrying... (attempt {attempt+1}/{args.loop})\n")
            else:
                print("\n  -> DO NOT deploy. Fix failing tests first.\n")
                sys.exit(1)


def _parse_counts(output: str) -> str:
    """Extract 'X passed, Y failed' from pytest output."""
    for line in reversed(output.split("\n")):
        if "passed" in line or "failed" in line or "error" in line:
            # Clean up the line
            line = line.strip()
            if line.startswith("="):
                line = line.strip("= ")
            return line
    return "no output"


def _update_task_status(task_id: int, status: str):
    """Placeholder for task tracking integration."""
    pass


if __name__ == "__main__":
    main()
