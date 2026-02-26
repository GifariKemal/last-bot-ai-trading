"""
Compare RSI Extreme Zone Protection for SELL Signals
Tests whether blocking SHORT when RSI is currently in extreme oversold zone
(< RSI_EXTREME_OVERSOLD=25) improves or worsens performance.

Context:
  At 08:45 UTC 2026-02-26, bot generated SHORT at RSI=24 (score 0.63 > 0.60).
  RSI bounce protection did NOT fire because RSI was still FALLING (not yet crossing
  back above 25). By 09:00, RSI bounced 24→37 — SHORT would have lost.

  Gap: current code blocks SELL only when RSI was < 25 AND is NOW > 25 (bounce detected).
       Does NOT block when RSI IS currently < 25 and still falling.

Scenarios:
  00_BASELINE       - Current (HARD_OVERSOLD=15, EXTREME_OVERSOLD=25 bounce-only)
  01_HARD_20        - Raise hard block: RSI_HARD_OVERSOLD 15→20
  02_HARD_25        - Raise hard block: RSI_HARD_OVERSOLD 15→25 (block SELL at RSI<25)
  03_HARD_30        - Raise hard block: RSI_HARD_OVERSOLD 15→30 (wider protection)
  04_SYMMETRIC      - Both sides: SELL block RSI<25, BUY block RSI>75 (symmetric)
  05_HARD_25_TIGHT  - Hard 25 + BUY block RSI>75 (full symmetric at extreme thresholds)

Usage:
    python scripts/compare_rsi_extreme.py [--days 180] [--balance 10000]
"""

import sys
import io
import copy
from pathlib import Path
from datetime import datetime, timedelta
from types import MethodType

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
# PATCHES
# ─────────────────────────────────────────────────────────────────────────────

def patch_rsi_hard_oversold(engine: BacktestEngine, new_threshold: int) -> None:
    """Raise RSI_HARD_OVERSOLD threshold — blocks SELL when RSI < threshold."""
    eg = engine.strategy.entry_generator
    old = eg.RSI_HARD_OVERSOLD
    eg.RSI_HARD_OVERSOLD = new_threshold
    print(f"   RSI_HARD_OVERSOLD: {old} -> {new_threshold}")


def patch_rsi_hard_overbought(engine: BacktestEngine, new_threshold: int) -> None:
    """Lower RSI_HARD_OVERBOUGHT threshold — blocks BUY when RSI > threshold."""
    eg = engine.strategy.entry_generator
    old = eg.RSI_HARD_OVERBOUGHT
    eg.RSI_HARD_OVERBOUGHT = new_threshold
    print(f"   RSI_HARD_OVERBOUGHT: {old} -> {new_threshold}")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id":    "00_BASELINE",
        "label": "Baseline (HARD_OS=15, HARD_OB=85)",
        "patches": [],
    },
    {
        "id":    "01_HARD_20",
        "label": "HARD_OS 15→20 (block SELL at RSI<20)",
        "patches": [lambda e: patch_rsi_hard_oversold(e, 20)],
    },
    {
        "id":    "02_HARD_25",
        "label": "HARD_OS 15→25 (block SELL at RSI<25 = extreme zone)",
        "patches": [lambda e: patch_rsi_hard_oversold(e, 25)],
    },
    {
        "id":    "03_HARD_30",
        "label": "HARD_OS 15→30 (block SELL at RSI<30 = wider zone)",
        "patches": [lambda e: patch_rsi_hard_oversold(e, 30)],
    },
    {
        "id":    "04_SYMMETRIC_75",
        "label": "Symmetric: HARD_OS=25, HARD_OB=75",
        "patches": [
            lambda e: patch_rsi_hard_oversold(e, 25),
            lambda e: patch_rsi_hard_overbought(e, 75),
        ],
    },
    {
        "id":    "05_SYMMETRIC_80",
        "label": "Symmetric: HARD_OS=20, HARD_OB=80",
        "patches": [
            lambda e: patch_rsi_hard_oversold(e, 20),
            lambda e: patch_rsi_hard_overbought(e, 80),
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(scenario: dict, base_config: dict, df_prepared, initial_balance: float, logger) -> dict:
    label = scenario["label"]
    logger.info(f"\n{'='*60}")
    logger.info(f"  SCENARIO: {label}")
    logger.info(f"{'='*60}")

    cfg  = copy.deepcopy(base_config)
    engine = BacktestEngine(cfg)

    print(f"\n[{scenario['id']}] {label}")
    for patch_fn in scenario["patches"]:
        patch_fn(engine)

    result = engine.run_backtest_fast(df_prepared, initial_balance=initial_balance)
    if not result.get("success"):
        logger.error(f"Backtest failed for {label}")
        return {"id": scenario["id"], "label": label, "success": False}

    m = result["metrics"]

    # Count how many SELL/SHORT trades were taken vs blocked
    trades = result.get("trades", [])
    sell_trades = [t for t in trades if "SELL" in str(t.get("direction","")).upper()]
    buy_trades  = [t for t in trades if "BUY"  in str(t.get("direction","")).upper()]

    sell_wins = [t for t in sell_trades if t.get("profit", 0) > 0]
    buy_wins  = [t for t in buy_trades  if t.get("profit", 0) > 0]

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
        "sell_trades":   len(sell_trades),
        "sell_wr":       len(sell_wins)/len(sell_trades)*100 if sell_trades else 0,
        "buy_trades":    len(buy_trades),
        "buy_wr":        len(buy_wins)/len(buy_trades)*100 if buy_trades else 0,
    }


def print_results(results: list) -> None:
    ok = [r for r in results if r.get("success")]
    if not ok:
        print("No results.")
        return

    baseline = ok[0]

    print("\n" + "=" * 120)
    print("RSI EXTREME ZONE PROTECTION — BACKTEST COMPARISON")
    print("=" * 120)

    col_w, nw = 48, 8
    print(f"\n{'Scenario':<{col_w}} {'Trades':>{nw}} {'WR%':>{nw}} {'PF':>{nw}} "
          f"{'Ret%':>{nw}} {'DD%':>{nw}} {'Sharpe':>{nw}} {'Net$':>{nw}} {'SELL#':>{nw}} {'SWR%':>{nw}}")
    print("-" * 120)

    for r in ok:
        is_base = r["id"] == "00_BASELINE"
        label   = r["label"] if is_base else f"  {r['label']}"

        pf  = r["profit_factor"]
        ret = r["total_return"]
        dd  = r["max_drawdown"]
        wr  = r["win_rate"]
        sh  = r["sharpe"]
        net = r["net_profit"]
        tr  = r["trades"]
        sl  = r["sell_trades"]
        swr = r["sell_wr"]

        print(f"{label:<{col_w}} {tr:>{nw}} {wr:>{nw-1}.1f}% {pf:>{nw}.2f} "
              f"{ret:>{nw-1}.2f}% {dd:>{nw-1}.2f}% {sh:>{nw}.2f} "
              f"${net:>{nw-1}.2f} {sl:>{nw}} {swr:>{nw-1}.1f}%")

        if not is_base:
            dpf = pf - baseline["profit_factor"]
            dtr = tr - baseline["trades"]
            dsl = sl - baseline["sell_trades"]
            dwr = wr - baseline["win_rate"]
            dswr = swr - baseline["sell_wr"]

            def sig(v, inv=False):
                if abs(v) < 0.01: return "  same"
                s = "+" if v > 0 else ""
                if inv: s = ("-" if v > 0 else "+")
                return f"{s}{v:.2f}"

            print(f"{'  vs Baseline':<{col_w}} "
                  f"{dtr:>+{nw}} {sig(dwr):>{nw}} {sig(dpf):>{nw}} "
                  f"{'':>{nw}} {'':>{nw}} {'':>{nw}} "
                  f"{'':>{nw}} {dsl:>+{nw}} {sig(dswr):>{nw}}")
            print()

    print("=" * 120)

    # Ranking
    ranked = sorted(ok[1:], key=lambda r: r["profit_factor"], reverse=True)
    base_pf = baseline["profit_factor"]

    print("\n RANKING BY PROFIT FACTOR vs Baseline")
    print("-" * 70)
    for i, r in enumerate(ranked, 1):
        dpf = r["profit_factor"] - base_pf
        dtr = r["trades"] - baseline["trades"]
        dsl = r["sell_trades"] - baseline["sell_trades"]
        arrow = "+" if dpf >= 0 else ""
        print(f"  #{i:02d}  PF={r['profit_factor']:.2f} ({arrow}{dpf:.2f})  "
              f"WR={r['win_rate']:.1f}%  "
              f"Trades={r['trades']} ({dtr:+d})  "
              f"SELLs={r['sell_trades']} ({dsl:+d})  "
              f"SELL-WR={r['sell_wr']:.1f}%  — {r['label']}")

    print("\n SELL DIRECTION DEEP DIVE (Baseline):")
    b = baseline
    print(f"  BUY  trades: {b['buy_trades']:3d}  WR: {b['buy_wr']:.1f}%")
    print(f"  SELL trades: {b['sell_trades']:3d}  WR: {b['sell_wr']:.1f}%")

    print("\n VERDICT:")
    if ranked and ranked[0]["profit_factor"] > base_pf:
        best = ranked[0]
        print(f"  Best: {best['label']}")
        print(f"    PF {base_pf:.2f} -> {best['profit_factor']:.2f} ({best['profit_factor']-base_pf:+.2f})")
        print(f"    Trades {baseline['trades']} -> {best['trades']} ({best['trades']-baseline['trades']:+d})")
        print(f"    SELL WR {baseline['sell_wr']:.1f}% -> {best['sell_wr']:.1f}%")
        print(f"  RECOMMENDATION: Apply this RSI threshold change to entry_signals.py")
    else:
        print(f"  No scenario beats baseline PF={base_pf:.2f}")
        print(f"  RECOMMENDATION: Keep RSI_HARD_OVERSOLD=15 (current)")
        if ranked:
            print(f"  Closest: {ranked[0]['label']}: PF={ranked[0]['profit_factor']:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Compare RSI extreme protection scenarios")
    p.add_argument("--days",    type=int,   default=180,     help="Lookback days")
    p.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    p.add_argument("--no-cache", action="store_true",        help="Force re-fetch from MT5")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logger()
    logger = get_logger()

    print(f"\n{'='*70}")
    print(f"  RSI EXTREME ZONE PROTECTION BACKTEST")
    print(f"  Period: {args.days} days  |  Balance: ${args.balance:.0f}")
    print(f"  Scenarios: {len(SCENARIOS)}")
    print(f"{'='*70}")
    print(f"\n  Context: At 08:45 UTC 26-Feb, RSI=24 SHORT signal fired.")
    print(f"  Bounce protection missed it (RSI still falling, not crossed above 25 yet).")
    print(f"  By 09:00 RSI bounced 24→37 — would have been a losing SELL.")
    print(f"  Testing whether tighter RSI_HARD_OVERSOLD improves overall SELL quality.\n")

    config_loader = ConfigLoader()
    settings      = config_loader.load("settings")
    mt5_config    = config_loader.load("mt5_config")
    trading_rules = config_loader.load("trading_rules")
    risk_config   = config_loader.load("risk_config")
    session_cfg   = config_loader.load("session_config")

    base_config = {**settings, **trading_rules, **risk_config}
    base_config["session"] = session_cfg

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    mt5 = MT5Connector(mt5_config)
    if not mt5.connect():
        print("ERROR: Cannot connect to MT5")
        sys.exit(1)

    symbol    = settings.get("trading", {}).get("symbol", "XAUUSDm")
    timeframe = settings.get("trading", {}).get("primary_timeframe", "M15")

    print(f"Preparing data: {symbol} {timeframe}  {start_date.date()} to {end_date.date()}")

    prep_engine = BacktestEngine(copy.deepcopy(base_config))
    use_cache   = not args.no_cache
    df_prepared = prep_engine._prepare_data(mt5, symbol, timeframe, start_date, end_date, use_cache)
    mt5.disconnect()

    if df_prepared is None:
        print("ERROR: Failed to prepare data")
        sys.exit(1)

    print(f"Data ready: {len(df_prepared)} bars\n")

    results = []
    for scenario in SCENARIOS:
        r = run_scenario(scenario, base_config, df_prepared, args.balance, logger)
        results.append(r)

    print_results(results)


if __name__ == "__main__":
    main()
