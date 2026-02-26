"""
Compare Improvements Backtest
Backtests all proposed system improvements vs baseline and shows side-by-side results.

Usage:
    python scripts/compare_improvements.py [--days 180] [--balance 10000]
"""

import sys
import io
import copy
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from types import MethodType

# Fix Windows cp1252 console encoding
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
from src.core.constants import TrendDirection


# ──────────────────────────────────────────────────────────────────────────────
# MONKEY-PATCH HELPERS
# Each patch_* function modifies a BacktestEngine instance in-place.
# ──────────────────────────────────────────────────────────────────────────────

def patch_rsi_range_80(engine: BacktestEngine) -> None:
    """Fix A: Extend RSI confirmation range from <70 to <80 for BUY."""
    scorer = engine.confluence_scorer
    original_fn = scorer._score_technical_factors.__func__ if hasattr(
        scorer._score_technical_factors, "__func__") else scorer._score_technical_factors

    def patched(self_s, direction, indicators):
        result = original_fn(self_s, direction, indicators)
        rsi = indicators.get("rsi")
        if rsi:
            if direction == TrendDirection.BULLISH and 70 <= rsi < 80:
                # RSI in 70-79: valid trending zone, was incorrectly excluded
                base_w = self_s.base_weights.get("rsi_confirmation", 0.08)
                result["total"] += base_w
                result["details"]["rsi_trending_zone"] = round(base_w, 4)
            elif direction == TrendDirection.BEARISH and 20 < rsi <= 30:
                # Mirror for bearish (20-30 trending zone)
                base_w = self_s.base_weights.get("rsi_confirmation", 0.08)
                result["total"] += base_w
                result["details"]["rsi_trending_zone"] = round(base_w, 4)
        return result

    scorer._score_technical_factors = MethodType(patched, scorer)


def patch_session_bonus(engine: BacktestEngine) -> None:
    """Fix B: Apply session weight as bonus in scorer bonus factors."""
    scorer = engine.confluence_scorer
    backtest_engine_ref = engine  # capture for use in closure
    original_fn = scorer._score_bonus_factors.__func__ if hasattr(
        scorer._score_bonus_factors, "__func__") else scorer._score_bonus_factors

    # Session weight mapping (from session_config.yaml)
    SESSION_WEIGHTS = {
        "asian":    0.947,
        "london":   1.16,
        "overlap":  1.442,
        "new_york": 1.16,
        "default":  1.0,
    }
    SESSION_BONUS_WEIGHT = 0.10  # from trading_rules.yaml

    def patched(self_s, mtf_analysis, direction=None, ltf_data=None, mtf_weight_scale=1.0):
        result = original_fn(self_s, mtf_analysis, direction, ltf_data, mtf_weight_scale)
        # Get session from LTF data time or use default
        session = "default"
        if ltf_data and "current_m15_time" in ltf_data:
            t = ltf_data["current_m15_time"]
            try:
                hour = t.hour if hasattr(t, "hour") else 12
                if 13 <= hour < 17:
                    session = "overlap"
                elif 8 <= hour < 13:
                    session = "london"
                elif 17 <= hour < 22:
                    session = "new_york"
                elif 0 <= hour < 8:
                    session = "asian"
            except Exception:
                pass

        weight = SESSION_WEIGHTS.get(session, 1.0)
        # Apply session bonus proportional to weight deviation from 1.0
        # Sessions above 1.0 get positive bonus, below 1.0 get minor reduction
        session_adj = (weight - 1.0) * SESSION_BONUS_WEIGHT
        result["total"] += session_adj
        result["details"]["session_bonus"] = round(session_adj, 4)
        return result

    scorer._score_bonus_factors = MethodType(patched, scorer)


def patch_bollinger_scoring(engine: BacktestEngine) -> None:
    """Fix F: Add Bollinger Bands confirmation to technical scoring."""
    scorer = engine.confluence_scorer
    original_fn = scorer._score_technical_factors.__func__ if hasattr(
        scorer._score_technical_factors, "__func__") else scorer._score_technical_factors

    def patched(self_s, direction, indicators):
        result = original_fn(self_s, direction, indicators)
        # BB: price near lower band = bullish; near upper band = bearish
        bb = indicators.get("bollinger_bands", {})
        if not bb:
            bb = indicators.get("bb", {})
        if bb:
            upper = bb.get("upper")
            lower = bb.get("lower")
            middle = bb.get("middle") or bb.get("basis")
            price = indicators.get("close") or indicators.get("current_price")
            if upper and lower and middle and price:
                band_width = upper - lower
                if band_width > 0:
                    bb_weight = self_s.base_weights.get("bollinger_confirmation", 0.05)
                    if direction == TrendDirection.BULLISH and price < middle:
                        # Price below middle band = potential buy zone
                        result["total"] += bb_weight
                        result["details"]["bollinger_buy_zone"] = bb_weight
                    elif direction == TrendDirection.BEARISH and price > middle:
                        result["total"] += bb_weight
                        result["details"]["bollinger_sell_zone"] = bb_weight
        return result

    scorer._score_technical_factors = MethodType(patched, scorer)


def apply_rsi_bounce_78(engine: BacktestEngine) -> None:
    """Fix D: Raise RSI bounce protection threshold from 75 to 78."""
    engine.strategy.entry_generator.RSI_EXTREME_OVERBOUGHT = 78
    engine.strategy.entry_generator.RSI_EXTREME_OVERSOLD = 22  # mirror


def apply_config_trail_gap(config: dict) -> dict:
    """Fix C: Reduce trail activation from 2.71R to 1.8R to close BE→trail gap."""
    cfg = copy.deepcopy(config)
    cfg.setdefault("exit_stages", {})["trail_activation_rr"] = 1.8
    return cfg


def apply_config_time_exit(config: dict, hours: int = 24) -> dict:
    """Fix E: Reduce time exit from 48h to 24h."""
    cfg = copy.deepcopy(config)
    cfg.setdefault("strategy", {}).setdefault("exit", {})["time_exit_hours"] = hours
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# SCENARIO DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id":    "00_BASELINE",
        "label": "Baseline (current)",
        "config_mod": None,
        "patches":    [],
    },
    {
        "id":    "01_RSI_RANGE",
        "label": "Fix A: RSI range 40-70→40-80",
        "config_mod": None,
        "patches":    [patch_rsi_range_80],
    },
    {
        "id":    "02_SESSION_BONUS",
        "label": "Fix B: Session bonus in scorer",
        "config_mod": None,
        "patches":    [patch_session_bonus],
    },
    {
        "id":    "03_TRAIL_GAP",
        "label": "Fix C: Trail activation 2.71R→1.8R",
        "config_mod": apply_config_trail_gap,
        "patches":    [],
    },
    {
        "id":    "04_RSI_BOUNCE",
        "label": "Fix D: RSI bounce threshold 75→78",
        "config_mod": None,
        "patches":    [apply_rsi_bounce_78],
    },
    {
        "id":    "05_TIME_EXIT",
        "label": "Fix E: Time exit 48h→24h",
        "config_mod": lambda c: apply_config_time_exit(c, 24),
        "patches":    [],
    },
    {
        "id":    "06_BOLLINGER",
        "label": "Fix F: Bollinger confirmation added",
        "config_mod": None,
        "patches":    [patch_bollinger_scoring],
    },
    {
        "id":    "07_RSI_PLUS_SESSION",
        "label": "Fix A+B: RSI range + Session bonus",
        "config_mod": None,
        "patches":    [patch_rsi_range_80, patch_session_bonus],
    },
    {
        "id":    "08_ALL_FIXES",
        "label": "ALL FIXES COMBINED",
        "config_mod": lambda c: apply_config_time_exit(apply_config_trail_gap(c), 24),
        "patches":    [patch_rsi_range_80, patch_session_bonus, apply_rsi_bounce_78, patch_bollinger_scoring],
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_scenario(scenario: dict, base_config: dict, df_prepared, initial_balance: float, logger) -> dict:
    """Run one backtest scenario and return metrics."""
    label = scenario["label"]
    logger.info(f"\n{'='*70}")
    logger.info(f"  SCENARIO: {label}")
    logger.info(f"{'='*70}")

    # Build config
    if scenario["config_mod"]:
        cfg = scenario["config_mod"](base_config)
    else:
        cfg = copy.deepcopy(base_config)

    # Create engine
    engine = BacktestEngine(cfg)

    # Apply code-level patches
    for patch_fn in scenario["patches"]:
        patch_fn(engine)

    # Run fast backtest (data already prepared)
    result = engine.run_backtest_fast(df_prepared, initial_balance=initial_balance)
    if not result.get("success"):
        logger.error(f"Backtest failed for {label}")
        return {"id": scenario["id"], "label": label, "success": False}

    m = result["metrics"]
    return {
        "id":              scenario["id"],
        "label":           label,
        "success":         True,
        "trades":          m.get("total_trades", 0),
        "win_rate":        m.get("win_rate", 0),
        "profit_factor":   m.get("profit_factor", 0),
        "total_return":    m.get("total_return_percent", 0),
        "max_drawdown":    m.get("max_drawdown_percent", 0),
        "avg_win":         m.get("avg_win", 0),
        "avg_loss":        m.get("avg_loss", 0),
        "sharpe":          m.get("sharpe_ratio", 0),
        "calmar":          m.get("calmar_ratio", 0),
        "expectancy":      m.get("expectancy", 0),
        "net_profit":      m.get("net_profit", 0),
        "regime_breakdown": m.get("regime_breakdown", {}),
    }


def print_comparison_table(results: list) -> None:
    """Print a formatted comparison table."""
    ok = [r for r in results if r.get("success")]
    if not ok:
        print("No successful results to compare.")
        return

    baseline = ok[0]

    print("\n" + "=" * 120)
    print("IMPROVEMENT COMPARISON RESULTS")
    print("=" * 120)

    # Header
    col_w = 38
    num_w = 9
    print(f"\n{'Scenario':<{col_w}} {'Trades':>{num_w}} {'WinRate':>{num_w}} {'PF':>{num_w}} "
          f"{'Return%':>{num_w}} {'MaxDD%':>{num_w}} {'Sharpe':>{num_w}} {'Expect$':>{num_w}} {'Net$':>{num_w}}")
    print("-" * 120)

    for r in ok:
        is_baseline = r["id"] == "00_BASELINE"
        label = r["label"]
        if not is_baseline:
            label = f"  {label}"

        pf_val    = r["profit_factor"]
        ret_val   = r["total_return"]
        dd_val    = r["max_drawdown"]
        wr_val    = r["win_rate"]
        sh_val    = r["sharpe"]
        ex_val    = r["expectancy"]
        net_val   = r["net_profit"]
        tr_val    = r["trades"]

        # Delta vs baseline (+ = better)
        def delta(cur, base, higher_is_better=True):
            if is_baseline:
                return ""
            diff = cur - base
            if abs(diff) < 0.001:
                return "  ─"
            sign = "+" if (diff > 0) == higher_is_better else "-"
            return f"{sign}{abs(diff):.2f}"

        pf_d  = delta(pf_val,  baseline["profit_factor"])
        ret_d = delta(ret_val, baseline["total_return"])
        dd_d  = delta(dd_val,  baseline["max_drawdown"], higher_is_better=False)  # lower DD = better
        wr_d  = delta(wr_val,  baseline["win_rate"])
        sh_d  = delta(sh_val,  baseline["sharpe"])

        print(f"{label:<{col_w}} {tr_val:>{num_w}} "
              f"{wr_val:>{num_w-1}.1f}% {pf_val:>{num_w}.2f} "
              f"{ret_val:>{num_w-1}.2f}% {dd_val:>{num_w-1}.2f}% "
              f"{sh_val:>{num_w}.2f} "
              f"${ex_val:>{num_w-1}.2f} ${net_val:>{num_w-1}.2f}")

        if not is_baseline:
            print(f"{'  → vs Baseline':<{col_w}} {'':>{num_w}} "
                  f"{wr_d:>{num_w}} {pf_d:>{num_w}} "
                  f"{ret_d:>{num_w}} {dd_d:>{num_w}} "
                  f"{sh_d:>{num_w}}")

    print("=" * 120)

    # Ranking by Profit Factor
    ranked = sorted([r for r in ok if r["id"] != "00_BASELINE"],
                    key=lambda r: r["profit_factor"], reverse=True)

    print("\n RANKING BY PROFIT FACTOR (vs Baseline)")
    print("-" * 60)
    baseline_pf  = baseline["profit_factor"]
    baseline_ret = baseline["total_return"]
    baseline_dd  = baseline["max_drawdown"]
    for rank, r in enumerate(ranked, 1):
        pf_diff  = r["profit_factor"]  - baseline_pf
        ret_diff = r["total_return"]   - baseline_ret
        dd_diff  = r["max_drawdown"]   - baseline_dd
        pf_arrow  = "↑" if pf_diff  >= 0 else "↓"
        ret_arrow = "↑" if ret_diff >= 0 else "↓"
        dd_arrow  = "↓" if dd_diff  <= 0 else "↑"  # lower DD is better
        print(f"  #{rank:02d}  PF={r['profit_factor']:.2f} ({pf_arrow}{abs(pf_diff):.2f}) | "
              f"Return={r['total_return']:.1f}% ({ret_arrow}{abs(ret_diff):.1f}%) | "
              f"DD={r['max_drawdown']:.1f}% ({dd_arrow}{abs(dd_diff):.1f}%)  — {r['label']}")

    print("\n RECOMMENDATION:")
    if ranked:
        best = ranked[0]
        print(f"  Best single fix:  {best['label']}")
        all_fix = next((r for r in ok if r["id"] == "08_ALL_FIXES"), None)
        if all_fix:
            print(f"  All fixes combo:  PF={all_fix['profit_factor']:.2f} | "
                  f"Return={all_fix['total_return']:.1f}% | DD={all_fix['max_drawdown']:.1f}%")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Compare improvement scenarios via backtest")
    p.add_argument("--days",    type=int,   default=180,     help="Lookback days (default: 180)")
    p.add_argument("--balance", type=float, default=10000.0, help="Initial balance (default: 10000)")
    p.add_argument("--save",    type=str,   default="",      help="Save results to JSON file")
    p.add_argument("--no-cache",action="store_true",         help="Force re-fetch data from MT5")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logger()  # Backtest uses its own logging
    logger = get_logger()

    print(f"\n{'='*70}")
    print(f"  IMPROVEMENT COMPARISON BACKTEST")
    print(f"  Period: {args.days} days  |  Balance: ${args.balance:.0f}")
    print(f"  Scenarios: {len(SCENARIOS)}")
    print(f"{'='*70}\n")

    # ── Load configuration ──────────────────────────────────────────────────
    config_loader = ConfigLoader()
    settings      = config_loader.load("settings")
    mt5_config    = config_loader.load("mt5_config")
    trading_rules = config_loader.load("trading_rules")
    risk_config   = config_loader.load("risk_config")
    session_config= config_loader.load("session_config")

    # Merge risk_config at top-level AND under "risk" key
    # (BacktestEngine reads risk keys top-level; session_close uses config["session"])
    base_config = {
        **settings,
        **risk_config,                                       # top-level risk keys
        "mt5": mt5_config,
        "strategy":             trading_rules.get("strategy", {}),
        "indicators":           trading_rules.get("indicators", {}),
        "smc_indicators":       trading_rules.get("smc_indicators", {}),
        "technical_indicators": trading_rules.get("technical_indicators", {}),
        "confluence_weights":   trading_rules.get("confluence_weights", {}),
        "market_conditions":    trading_rules.get("market_conditions", {}),
        "mtf_analysis":         trading_rules.get("mtf_analysis", {}),
        "signal_validation":    trading_rules.get("signal_validation", {}),
        "risk":    risk_config,
        "session": session_config,
    }

    # ── Connect MT5 ─────────────────────────────────────────────────────────
    print("Connecting to MT5...")
    mt5 = MT5Connector(base_config["mt5"])
    if not mt5.connect():
        print("ERROR: Failed to connect to MT5")
        return 1
    print("MT5 connected.")

    # ── Prepare data ONCE ───────────────────────────────────────────────────
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    print(f"Fetching data: {start_date.date()} to {end_date.date()} (M15)...")
    seed_engine = BacktestEngine(base_config)
    df_prepared = seed_engine._prepare_data(
        mt5, "XAUUSDm", "M15", start_date, end_date,
        use_cache=not args.no_cache
    )
    mt5.disconnect()

    if df_prepared is None:
        print("ERROR: Failed to prepare data")
        return 1

    print(f"Data ready: {len(df_prepared)} bars  ({start_date.date()} – {end_date.date()})")
    print(f"\nRunning {len(SCENARIOS)} scenarios...\n")

    # ── Run all scenarios ───────────────────────────────────────────────────
    all_results = []
    for i, scenario in enumerate(SCENARIOS):
        print(f"[{i+1}/{len(SCENARIOS)}] {scenario['label']}...", flush=True)
        try:
            res = run_scenario(scenario, base_config, df_prepared, args.balance, logger)
            all_results.append(res)
            if res.get("success"):
                print(f"        PF={res['profit_factor']:.2f}  WR={res['win_rate']:.1f}%  "
                      f"Return={res['total_return']:.1f}%  DD={res['max_drawdown']:.1f}%  "
                      f"Trades={res['trades']}")
            else:
                print("        FAILED")
        except Exception as e:
            print(f"        ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"id": scenario["id"], "label": scenario["label"], "success": False, "error": str(e)})

    # ── Print comparison table ───────────────────────────────────────────────
    print_comparison_table(all_results)

    # ── Save results ─────────────────────────────────────────────────────────
    output_path = args.save or f"data/improvement_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
