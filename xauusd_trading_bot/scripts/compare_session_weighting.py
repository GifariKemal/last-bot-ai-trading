"""
Compare Session Weighting Scenarios
Tests whether activating dead session.weight configs improves trading performance.

Currently dead config in session_config.yaml:
  asian.weight=0.947, london.weight=1.16, overlap.weight=1.442, new_york.weight=1.16
  asian.min_confluence_adjustment=0.05, overlap.min_confluence_adjustment=0.05

Scenarios:
  00_BASELINE     — Current bot (no session weight, flat threshold for all sessions)
  01_SCORE_MULT   — Multiply final score by session.weight per bar timestamp
  02_THRESH_ADJ   — Add min_confluence_adjustment to threshold per session
  03_BOTH         — Score multiplication + threshold adjustment combined (full activation)

Usage:
    python scripts/compare_session_weighting.py [--days 180] [--balance 10000]
"""

import sys
import io
import copy
from pathlib import Path
from datetime import datetime, timedelta

# Fix Windows cp1252 console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtesting import BacktestEngine
from src.core.mt5_connector import MT5Connector
from src.bot_logger import setup_logger, get_logger
from src.utils.config_loader import ConfigLoader


# ─────────────────────────────────────────────────────────────────────────────
# Session parameters (from session_config.yaml)
# (start_hour, end_hour): (weight, min_confluence_adjustment)
# ─────────────────────────────────────────────────────────────────────────────
SESSION_PARAMS = {
    (0,  8):  (0.947, 0.05),   # Asian: weight < 1 = reduces score, adj = tighter threshold
    (8,  13): (1.16,  0.00),   # London: weight > 1 = boosts score
    (13, 17): (1.442, 0.05),   # Overlap: biggest boost, but threshold also tighter
    (17, 22): (1.16,  0.00),   # New York: same as London
    (22, 24): (1.00,  0.00),   # Off hours / maintenance
}


def get_session_params(ts) -> tuple:
    """Return (weight, threshold_adj) for a given timestamp."""
    try:
        hour = ts.hour if hasattr(ts, "hour") else 12
        for (start, end), params in SESSION_PARAMS.items():
            if start <= hour < end:
                return params
    except Exception:
        pass
    return 1.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Session-Weighted Scorer Wrapper
# Wraps AdaptiveConfluenceScorer to apply session weighting per bar.
# Reads engine._current_bar_index (already set before calculate_score is called)
# and df_ref (set by runner before run_backtest_fast).
# ─────────────────────────────────────────────────────────────────────────────

class SessionWeightedScorer:
    """
    Drop-in replacement for AdaptiveConfluenceScorer that applies session
    weights to the final score and/or threshold adjustment per bar timestamp.

    Uses engine._current_bar_index (set at line 402 in _simulate_trading,
    before calculate_score is called at line 485) to look up current_time.
    """

    def __init__(
        self,
        base_scorer,
        engine_ref: BacktestEngine,
        weight_mode: bool = False,
        threshold_mode: bool = False,
    ):
        self._base = base_scorer
        self._engine_ref = engine_ref
        self._weight_mode = weight_mode
        self._threshold_mode = threshold_mode
        self._df_ref = None  # set by runner after data is prepared

    def __getattr__(self, name):
        """Delegate everything else to the base scorer."""
        return getattr(self._base, name)

    def calculate_score(self, direction, smc_signals, technical_indicators,
                        market_analysis, mtf_analysis, **kwargs):
        result = self._base.calculate_score(
            direction, smc_signals, technical_indicators,
            market_analysis, mtf_analysis, **kwargs
        )

        if self._df_ref is None or not (self._weight_mode or self._threshold_mode):
            return result

        try:
            bar_idx = getattr(self._engine_ref, "_current_bar_index", None)
            if bar_idx is None:
                return result

            current_time = self._df_ref["time"][bar_idx]
            weight, adj = get_session_params(current_time)

            result = dict(result)

            if self._weight_mode:
                result["score"] = round(result.get("score", 0.0) * weight, 4)

            if self._threshold_mode:
                result["min_confluence"] = round(
                    result.get("min_confluence", 0.55) + adj, 4
                )

            # Recompute passing after adjustments
            result["passing"] = result.get("score", 0.0) >= result.get("min_confluence", 0.55)

        except Exception:
            pass  # If anything fails, return unmodified result

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Runner
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id":    "00_BASELINE",
        "label": "Baseline (no session weighting)",
        "weight_mode":    False,
        "threshold_mode": False,
    },
    {
        "id":    "01_SCORE_MULT",
        "label": "Score x session.weight only",
        "weight_mode":    True,
        "threshold_mode": False,
    },
    {
        "id":    "02_THRESH_ADJ",
        "label": "Threshold + min_confluence_adjustment only",
        "weight_mode":    False,
        "threshold_mode": True,
    },
    {
        "id":    "03_BOTH",
        "label": "Both: score_mult + threshold_adj (full activation)",
        "weight_mode":    True,
        "threshold_mode": True,
    },
]


def run_scenario(scenario: dict, base_config: dict, df_prepared, initial_balance: float, logger) -> dict:
    """Run one backtest scenario and return metrics."""
    label = scenario["label"]
    logger.info(f"\n{'='*70}")
    logger.info(f"  SCENARIO: {label}")
    logger.info(f"{'='*70}")

    cfg = copy.deepcopy(base_config)
    engine = BacktestEngine(cfg)

    # Replace confluence_scorer with session-weighted wrapper
    base_scorer = engine.confluence_scorer
    wrapped_scorer = SessionWeightedScorer(
        base_scorer=base_scorer,
        engine_ref=engine,
        weight_mode=scenario["weight_mode"],
        threshold_mode=scenario["threshold_mode"],
    )
    # Give wrapper access to the prepared DataFrame for time lookup
    wrapped_scorer._df_ref = df_prepared
    engine.confluence_scorer = wrapped_scorer

    result = engine.run_backtest_fast(df_prepared, initial_balance=initial_balance)
    if not result.get("success"):
        logger.error(f"Backtest failed for {label}")
        return {"id": scenario["id"], "label": label, "success": False}

    m = result["metrics"]
    return {
        "id":            scenario["id"],
        "label":         label,
        "success":       True,
        "trades":        m.get("total_trades", 0),
        "win_rate":      m.get("win_rate", 0),
        "profit_factor": m.get("profit_factor", 0),
        "total_return":  m.get("total_return_percent", 0),
        "max_drawdown":  m.get("max_drawdown_percent", 0),
        "sharpe":        m.get("sharpe_ratio", 0),
        "expectancy":    m.get("expectancy", 0),
        "net_profit":    m.get("net_profit", 0),
    }


def print_session_breakdown(results: list, df_prepared, logger) -> None:
    """Show trade distribution across sessions for each scenario."""
    # Count bars per session to give context
    session_counts = {name: 0 for name in ["asian", "london", "overlap", "new_york", "off"]}
    for i in range(len(df_prepared)):
        try:
            hour = df_prepared["time"][i].hour
        except Exception:
            hour = 12
        if 0 <= hour < 8:
            session_counts["asian"] += 1
        elif 8 <= hour < 13:
            session_counts["london"] += 1
        elif 13 <= hour < 17:
            session_counts["overlap"] += 1
        elif 17 <= hour < 22:
            session_counts["new_york"] += 1
        else:
            session_counts["off"] += 1

    total_bars = len(df_prepared)
    print("\n Session Distribution in Data:")
    print(f"   Asian    {session_counts['asian']:5d} bars ({session_counts['asian']/total_bars*100:.1f}%)  weight=0.947 adj=+0.05")
    print(f"   London   {session_counts['london']:5d} bars ({session_counts['london']/total_bars*100:.1f}%)  weight=1.16  adj=+0.00")
    print(f"   Overlap  {session_counts['overlap']:5d} bars ({session_counts['overlap']/total_bars*100:.1f}%)  weight=1.442 adj=+0.05")
    print(f"   NY       {session_counts['new_york']:5d} bars ({session_counts['new_york']/total_bars*100:.1f}%)  weight=1.16  adj=+0.00")
    print(f"   Off hrs  {session_counts['off']:5d} bars ({session_counts['off']/total_bars*100:.1f}%)  (blackout)")

    print("\n Expected Effect of Session Weighting:")
    print("   Example: score=0.60, threshold=0.60")
    print("   BASELINE:     0.60 >= 0.60 -> PASS (all sessions equal)")
    print("   SCORE_MULT:   Asian  0.60x0.947=0.568 < 0.60 -> FAIL (harder)")
    print("                 London 0.60x1.16 =0.696 >= 0.60 -> PASS (easier)")
    print("                 Overlap 0.60x1.442=0.865 >= 0.60 -> PASS (much easier)")
    print("   THRESH_ADJ:   Asian  0.60 < 0.60+0.05=0.65 -> FAIL (harder)")
    print("                 Overlap 0.60 < 0.60+0.05=0.65 -> FAIL (harder)")
    print("   BOTH:         Asian  0.568 < 0.65 -> FAIL (much harder)")
    print("                 Overlap 0.865 >= 0.65 -> PASS (still easy)")


def print_results_table(results: list) -> None:
    """Print comparison table."""
    ok = [r for r in results if r.get("success")]
    if not ok:
        print("No results to compare.")
        return

    baseline = ok[0]

    print("\n" + "=" * 110)
    print("SESSION WEIGHTING — BACKTEST COMPARISON")
    print("=" * 110)

    col_w, num_w = 45, 9
    print(f"\n{'Scenario':<{col_w}} {'Trades':>{num_w}} {'WinRate':>{num_w}} {'PF':>{num_w}} "
          f"{'Return%':>{num_w}} {'MaxDD%':>{num_w}} {'Sharpe':>{num_w}} {'Net$':>{num_w}}")
    print("-" * 110)

    for r in ok:
        is_base = r["id"] == "00_BASELINE"
        label = r["label"] if is_base else f"  {r['label']}"

        pf  = r["profit_factor"]
        ret = r["total_return"]
        dd  = r["max_drawdown"]
        wr  = r["win_rate"]
        sh  = r["sharpe"]
        net = r["net_profit"]
        tr  = r["trades"]

        print(f"{label:<{col_w}} {tr:>{num_w}} {wr:>{num_w-1}.1f}% {pf:>{num_w}.2f} "
              f"{ret:>{num_w-1}.2f}% {dd:>{num_w-1}.2f}% {sh:>{num_w}.2f} ${net:>{num_w-1}.2f}")

        if not is_base:
            # Delta
            dpf  = pf  - baseline["profit_factor"]
            dret = ret - baseline["total_return"]
            ddd  = dd  - baseline["max_drawdown"]
            dwr  = wr  - baseline["win_rate"]
            dtr  = tr  - baseline["trades"]

            def fmt(v, inv=False):
                if abs(v) < 0.005:
                    return "  same"
                s = "+" if v > 0 else "-"
                if inv:
                    s = "-" if v > 0 else "+"
                return f"{s}{abs(v):.2f}"

            print(f"{'  vs Baseline':<{col_w}} "
                  f"{dtr:>+{num_w}} {fmt(dwr):>{num_w}} {fmt(dpf):>{num_w}} "
                  f"{fmt(dret):>{num_w}} {fmt(ddd, inv=True):>{num_w}}")
            print()

    print("=" * 110)

    # Verdict
    print("\n VERDICT:")
    ranked = sorted(ok[1:], key=lambda r: r["profit_factor"], reverse=True)
    best = ranked[0] if ranked else None
    both = next((r for r in ok if r["id"] == "03_BOTH"), None)
    base_pf = baseline["profit_factor"]

    if best and best["profit_factor"] > base_pf:
        print(f"  Best improvement: {best['label']}")
        print(f"    PF {base_pf:.2f} -> {best['profit_factor']:.2f} ({best['profit_factor']-base_pf:+.2f})")
        print(f"    Trades {baseline['trades']} -> {best['trades']} ({best['trades']-baseline['trades']:+d})")
        print(f"  RECOMMENDATION: ACTIVATE this session weighting in trading_bot.py")
    else:
        best_scenario = ranked[0] if ranked else None
        print(f"  No scenario beats baseline PF={base_pf:.2f}")
        if best_scenario:
            print(f"  Best was {best_scenario['label']}: PF={best_scenario['profit_factor']:.2f}")
        print(f"  RECOMMENDATION: Keep session weights as DEAD CONFIG (current state)")

    if both:
        print(f"\n  Full activation (BOTH): PF={both['profit_factor']:.2f} | "
              f"Return={both['total_return']:.1f}% | DD={both['max_drawdown']:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Compare session weighting scenarios")
    p.add_argument("--days",    type=int,   default=180,     help="Lookback days")
    p.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    p.add_argument("--no-cache", action="store_true",        help="Force re-fetch from MT5")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logger()
    logger = get_logger()

    print(f"\n{'='*70}")
    print(f"  SESSION WEIGHTING COMPARISON BACKTEST")
    print(f"  Period: {args.days} days  |  Balance: ${args.balance:.0f}")
    print(f"  Scenarios: {len(SCENARIOS)}")
    print(f"{'='*70}\n")

    # Load config
    config_loader = ConfigLoader()
    settings      = config_loader.load("settings")
    mt5_config    = config_loader.load("mt5_config")
    trading_rules = config_loader.load("trading_rules")
    risk_config   = config_loader.load("risk_config")
    session_cfg   = config_loader.load("session_config")

    base_config = {**settings, **trading_rules, **risk_config}
    base_config["session"] = session_cfg

    initial_balance = args.balance
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    # Connect MT5 and prepare data ONCE
    mt5 = MT5Connector(mt5_config)
    if not mt5.connect():
        print("ERROR: Cannot connect to MT5. Is MetaTrader 5 running?")
        sys.exit(1)

    symbol    = settings.get("trading", {}).get("symbol", "XAUUSDm")
    timeframe = settings.get("trading", {}).get("primary_timeframe", "M15")

    print(f"Preparing data: {symbol} {timeframe}  {start_date.date()} to {end_date.date()}")
    print("(Data prepared ONCE, reused across all scenarios)\n")

    # Prepare data using baseline engine
    baseline_cfg = copy.deepcopy(base_config)
    prep_engine  = BacktestEngine(baseline_cfg)

    use_cache = not args.no_cache
    df_prepared = prep_engine._prepare_data(mt5, symbol, timeframe, start_date, end_date, use_cache)
    mt5.disconnect()

    if df_prepared is None:
        print("ERROR: Failed to prepare data")
        sys.exit(1)

    print(f"Data ready: {len(df_prepared)} bars\n")

    # Show session breakdown
    print_session_breakdown(SCENARIOS, df_prepared, logger)
    print()

    # Run all scenarios
    results = []
    for scenario in SCENARIOS:
        r = run_scenario(scenario, base_config, df_prepared, initial_balance, logger)
        results.append(r)

    # Print final comparison
    print_results_table(results)


if __name__ == "__main__":
    main()
