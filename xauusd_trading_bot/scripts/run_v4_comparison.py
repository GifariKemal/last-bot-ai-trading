"""
V4 A/B Comparison: Custom SMC Detectors (V3) vs Library-based SMC (V4)
=======================================================================
Runs two identical backtests on the same date range with the ONLY difference
being the SMC detection layer:
  - V3: Custom FVG/OB/Liquidity/Structure detectors
  - V4: smartmoneyconcepts library (joshyattridge)

All other components are IDENTICAL:
  - AdaptiveConfluenceScorer   (unchanged)
  - EntrySignalGenerator       (unchanged)
  - StructureSLTPCalculator    (unchanged)
  - Multi-stage exits (BE/partial/trail) (unchanged)

Usage:
    cd xauusd_trading_bot
    python scripts/run_v4_comparison.py
    python scripts/run_v4_comparison.py --months 6    # 6-month window
    python scripts/run_v4_comparison.py --months 12   # full year
    python scripts/run_v4_comparison.py --months 3    # quick test

Results saved to: data/v4_comparison/v3_vs_v4_results.json
"""

import sys
import os
import json
import argparse
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import ConfigLoader
from src.core.mt5_connector import MT5Connector
from src.backtesting.backtest_engine import BacktestEngine
from src.bot_logger import setup_logger, get_logger


# ── Config loader ─────────────────────────────────────────────────────────────

def _load_base_config() -> dict:
    """Load the current live config (same for both V3 and V4 runs)."""
    config_loader = ConfigLoader()
    settings      = config_loader.load("settings")
    mt5_config    = config_loader.load("mt5_config")
    trading_rules = config_loader.load("trading_rules")
    risk_config   = config_loader.load("risk_config")
    session_config = config_loader.load("session_config")

    return {
        **settings,
        "mt5":                  mt5_config,
        "strategy":             trading_rules.get("strategy", {}),
        "indicators":           trading_rules.get("indicators", {}),
        "smc_indicators":       trading_rules.get("smc_indicators", {}),
        "technical_indicators": trading_rules.get("technical_indicators", {}),
        "confluence_weights":   trading_rules.get("confluence_weights", {}),
        "market_conditions":    trading_rules.get("market_conditions", {}),
        "mtf_analysis":         trading_rules.get("mtf_analysis", {}),
        "signal_validation":    trading_rules.get("signal_validation", {}),
        "risk":                 risk_config,
        "session":              session_config,
        "exit_stages":          risk_config.get("exit_stages", {}),
    }


def _v3_config(base: dict) -> dict:
    cfg = dict(base)
    cfg["use_smc_v4"]        = False  # V3: custom detectors
    cfg["use_adaptive_scorer"] = True   # Keep V3 scorer
    return cfg


def _v4_config(base: dict) -> dict:
    cfg = dict(base)
    cfg["use_smc_v4"]        = True   # V4: library-based SMC
    cfg["use_adaptive_scorer"] = True   # Same scorer
    return cfg


# ── Formatting ────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def _print_results(label: str, metrics: dict) -> None:
    _section(label)
    pf     = metrics.get("profit_factor", 0)
    wr     = metrics.get("win_rate", 0)
    dd     = metrics.get("max_drawdown_percent", metrics.get("max_drawdown_pct", 0))
    ret    = metrics.get("total_return_percent", metrics.get("total_return_pct", 0))
    trades = metrics.get("total_trades", 0)
    rr     = metrics.get("avg_rr", metrics.get("avg_risk_reward", 0))
    sharpe = metrics.get("sharpe_ratio", 0)
    print(f"  Profit Factor:    {pf:.3f}")
    print(f"  Win Rate:         {wr:.1f}%")
    print(f"  Max Drawdown:     {dd:.2f}%")
    print(f"  Total Return:     {ret:.2f}%")
    print(f"  Total Trades:     {trades}")
    print(f"  Avg RR:           {rr:.2f}")
    print(f"  Sharpe Ratio:     {sharpe:.2f}")


def _print_comparison(m3: dict, m4: dict) -> None:
    _section("V3 vs V4 — Delta Comparison")

    metrics_to_compare = [
        ("profit_factor",       "Profit Factor",   True),
        ("win_rate",            "Win Rate %",      True),
        ("max_drawdown_percent","Max Drawdown %",  False),  # lower is better
        ("total_return_percent","Total Return %",  True),
        ("total_trades",        "Total Trades",    None),   # neutral
        ("avg_rr",              "Avg RR",          True),
        ("sharpe_ratio",        "Sharpe Ratio",    True),
    ]

    print(f"  {'Metric':<20} {'V3':>10}  {'V4':>10}  {'Delta':>10}  Verdict")
    print(f"  {'-'*60}")

    for key, label, higher_better in metrics_to_compare:
        v3  = m3.get(key, 0) or 0
        v4  = m4.get(key, 0) or 0
        delta = v4 - v3
        if higher_better is True:
            verdict = "[+] V4 BETTER" if delta > 0.01 else "[-] V3 BETTER" if delta < -0.01 else "[=] EQUAL"
        elif higher_better is False:
            verdict = "[+] V4 BETTER" if delta < -0.01 else "[-] V3 BETTER" if delta > 0.01 else "[=] EQUAL"
        else:
            verdict = f"{'more' if delta > 0 else 'fewer'} trades"
        print(f"  {label:<20} {v3:>10.2f}  {v4:>10.2f}  {delta:>+10.2f}  {verdict}")

    print(f"  {'-'*60}")
    v4_wins = sum(
        1 for key, _, hb in metrics_to_compare
        if hb is not None and (
            (hb is True  and (m4.get(key, 0) or 0) > (m3.get(key, 0) or 0) + 0.01) or
            (hb is False and (m4.get(key, 0) or 0) < (m3.get(key, 0) or 0) - 0.01)
        )
    )
    v3_wins = sum(
        1 for key, _, hb in metrics_to_compare
        if hb is not None and (
            (hb is True  and (m3.get(key, 0) or 0) > (m4.get(key, 0) or 0) + 0.01) or
            (hb is False and (m3.get(key, 0) or 0) < (m4.get(key, 0) or 0) - 0.01)
        )
    )
    print(f"\n  V4 wins on {v4_wins}/6 metrics, V3 wins on {v3_wins}/6 metrics")
    if v4_wins > v3_wins:
        print("  RECOMMENDATION: V4 library is BETTER — consider enabling use_smc_v4=true for live")
    elif v3_wins > v4_wins:
        print("  RECOMMENDATION: V3 custom detectors are BETTER — keep use_smc_v4=false")
    else:
        print("  RECOMMENDATION: Results are SIMILAR — investigate per-regime breakdown")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V3 vs V4 SMC A/B comparison backtest")
    parser.add_argument("--months", type=int, default=12,
                        help="Number of months to backtest (default: 12)")
    parser.add_argument("--symbol", default="XAUUSDm")
    parser.add_argument("--timeframe", default="M15")
    args = parser.parse_args()

    setup_logger()
    logger = get_logger()

    end_date   = datetime.utcnow()
    start_date = end_date - timedelta(days=30 * args.months)

    print(f"\nXAUUSD Bot - V3 vs V4 SMC A/B Comparison")
    print(f"Started:    {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Period:     {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Symbol:     {args.symbol} | TF: {args.timeframe}")
    print(f"Months:     {args.months}")

    # Load config
    print("\nLoading config...")
    try:
        base = _load_base_config()
    except Exception as e:
        print(f"ERROR loading config: {e}")
        sys.exit(1)

    # Connect MT5
    mt5 = MT5Connector(base.get("mt5", {}))
    print("Connecting to MT5...")
    if not mt5.connect():
        print("ERROR: Could not connect to MT5. Is the terminal running?")
        sys.exit(1)
    print("MT5 connected.")

    m3, m4 = {}, {}

    try:
        # ── V3 run ──────────────────────────────────────────────────────────
        print(f"\n{'='*62}")
        print("  [V3] Running backtest with custom SMC detectors...")
        print(f"{'='*62}")
        t0 = time.time()
        cfg_v3 = _v3_config(base)
        engine_v3 = BacktestEngine(cfg_v3)
        res3 = engine_v3.run_backtest(
            mt5=mt5,
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            initial_balance=10000.0,
            timeframe=args.timeframe,
            use_cache=True,
        )
        elapsed3 = time.time() - t0
        print(f"  [V3] Completed in {elapsed3:.1f}s")

        if res3.get("success"):
            m3 = res3.get("metrics", res3.get("performance", {}))
            _print_results("V3 — Custom SMC Detectors", m3)
        else:
            print(f"  [V3] FAILED: {res3.get('error', 'unknown error')}")

        # ── V4 run ──────────────────────────────────────────────────────────
        print(f"\n{'='*62}")
        print("  [V4] Running backtest with smartmoneyconcepts library...")
        print(f"{'='*62}")
        t0 = time.time()
        cfg_v4 = _v4_config(base)
        engine_v4 = BacktestEngine(cfg_v4)
        res4 = engine_v4.run_backtest(
            mt5=mt5,
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            initial_balance=10000.0,
            timeframe=args.timeframe,
            use_cache=True,   # reuse same data as V3
        )
        elapsed4 = time.time() - t0
        print(f"  [V4] Completed in {elapsed4:.1f}s")

        if res4.get("success"):
            m4 = res4.get("metrics", res4.get("performance", {}))
            _print_results("V4 — Library-based SMC (smartmoneyconcepts)", m4)
        else:
            print(f"  [V4] FAILED: {res4.get('error', 'unknown error')}")

    finally:
        mt5.disconnect()

    # ── Comparison ──────────────────────────────────────────────────────────
    if m3 and m4:
        _print_comparison(m3, m4)

    # ── Save results ─────────────────────────────────────────────────────────
    os.makedirs("data/v4_comparison", exist_ok=True)
    output = {
        "timestamp":     datetime.utcnow().isoformat(),
        "period":        f"{start_date.date()} to {end_date.date()}",
        "months":        args.months,
        "symbol":        args.symbol,
        "timeframe":     args.timeframe,
        "v3_metrics":    m3,
        "v4_metrics":    m4,
    }
    out_path = "data/v4_comparison/v3_vs_v4_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    print(f"\nDone. V3 took {elapsed3:.1f}s | V4 took {elapsed4:.1f}s")


if __name__ == "__main__":
    main()
