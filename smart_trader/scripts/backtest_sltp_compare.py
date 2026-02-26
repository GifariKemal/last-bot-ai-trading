"""
Backtest comparison: Current SL/TP vs Proposed Tighter SL/TP.
Runs 6-month simulation for each config and prints side-by-side results.

Usage:
  cd smart_trader && python scripts/backtest_sltp_compare.py
  python scripts/backtest_sltp_compare.py --months 3
"""
import sys, os, copy, argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from backtest_smart import (
    load_config, connect_mt5, fetch_historical, run_simulation,
    compute_results, print_results, print_comparison,
    write_equity_csv, write_trades_csv,
)
import MetaTrader5 as mt5


# ── Config variants ─────────────────────────────────────────────────────────

CURRENT_REGIME_PARAMS = {
    "trending": {
        "sl_atr_mult": 2.75, "tp_atr_mult": 7.0, "min_confidence": 0.70,
        "be_trigger_mult": 0.3, "lock_trigger_mult": 1.4, "trail_keep_pct": 0.70,
        "stale_tighten_min": 120, "scratch_exit_min": 330,
    },
    "ranging": {
        "sl_atr_mult": 2.50, "tp_atr_mult": 6.0, "min_confidence": 0.70,
        "be_trigger_mult": 0.4, "lock_trigger_mult": 1.5, "trail_keep_pct": 0.30,
        "stale_tighten_min": 90, "scratch_exit_min": 210,
    },
    "breakout": {
        "sl_atr_mult": 3.25, "tp_atr_mult": 5.5, "min_confidence": 0.70,
        "be_trigger_mult": 0.4, "lock_trigger_mult": 1.4, "trail_keep_pct": 0.40,
        "stale_tighten_min": 90, "scratch_exit_min": 120,
    },
    "reversal": {
        "sl_atr_mult": 3.00, "tp_atr_mult": 5.0, "min_confidence": 0.70,
        "be_trigger_mult": 0.6, "lock_trigger_mult": 1.3, "trail_keep_pct": 0.55,
        "stale_tighten_min": 180, "scratch_exit_min": 60,
    },
}

PROPOSED_REGIME_PARAMS = {
    "trending": {
        "sl_atr_mult": 1.5, "tp_atr_mult": 2.5, "min_confidence": 0.70,
        "be_trigger_mult": 0.5, "lock_trigger_mult": 1.2, "trail_keep_pct": 0.55,
        "stale_tighten_min": 60, "scratch_exit_min": 150,
    },
    "ranging": {
        "sl_atr_mult": 1.2, "tp_atr_mult": 2.0, "min_confidence": 0.70,
        "be_trigger_mult": 0.5, "lock_trigger_mult": 1.0, "trail_keep_pct": 0.50,
        "stale_tighten_min": 45, "scratch_exit_min": 120,
    },
    "breakout": {
        "sl_atr_mult": 2.0, "tp_atr_mult": 3.5, "min_confidence": 0.70,
        "be_trigger_mult": 0.5, "lock_trigger_mult": 1.3, "trail_keep_pct": 0.50,
        "stale_tighten_min": 75, "scratch_exit_min": 180,
    },
    "reversal": {
        "sl_atr_mult": 1.2, "tp_atr_mult": 2.0, "min_confidence": 0.70,
        "be_trigger_mult": 0.5, "lock_trigger_mult": 1.0, "trail_keep_pct": 0.50,
        "stale_tighten_min": 45, "scratch_exit_min": 120,
    },
}

# Also test a middle-ground option
MODERATE_REGIME_PARAMS = {
    "trending": {
        "sl_atr_mult": 2.0, "tp_atr_mult": 3.5, "min_confidence": 0.70,
        "be_trigger_mult": 0.4, "lock_trigger_mult": 1.3, "trail_keep_pct": 0.60,
        "stale_tighten_min": 75, "scratch_exit_min": 210,
    },
    "ranging": {
        "sl_atr_mult": 1.5, "tp_atr_mult": 2.5, "min_confidence": 0.70,
        "be_trigger_mult": 0.4, "lock_trigger_mult": 1.2, "trail_keep_pct": 0.45,
        "stale_tighten_min": 60, "scratch_exit_min": 150,
    },
    "breakout": {
        "sl_atr_mult": 2.5, "tp_atr_mult": 4.0, "min_confidence": 0.70,
        "be_trigger_mult": 0.5, "lock_trigger_mult": 1.4, "trail_keep_pct": 0.45,
        "stale_tighten_min": 90, "scratch_exit_min": 150,
    },
    "reversal": {
        "sl_atr_mult": 1.5, "tp_atr_mult": 2.5, "min_confidence": 0.70,
        "be_trigger_mult": 0.5, "lock_trigger_mult": 1.2, "trail_keep_pct": 0.50,
        "stale_tighten_min": 60, "scratch_exit_min": 120,
    },
}

# OPTIMAL: keep SL wide enough to survive noise, but bring TP closer to realistic range
# Key insight from round 1: TIGHT SL=1.2x (29pt) gets stopped too easily on XAUUSD
# But CURRENT TP=5-7x (120-168pt) basically never gets hit (only 4.3%)
# Solution: SL similar to CURRENT/MODERATE, TP = MODERATE level
OPTIMAL_REGIME_PARAMS = {
    "trending": {
        "sl_atr_mult": 2.0, "tp_atr_mult": 3.0, "min_confidence": 0.70,
        "be_trigger_mult": 0.4, "lock_trigger_mult": 1.2, "trail_keep_pct": 0.55,
        "stale_tighten_min": 75, "scratch_exit_min": 180,
    },
    "ranging": {
        "sl_atr_mult": 1.8, "tp_atr_mult": 2.5, "min_confidence": 0.70,
        "be_trigger_mult": 0.4, "lock_trigger_mult": 1.0, "trail_keep_pct": 0.50,
        "stale_tighten_min": 60, "scratch_exit_min": 150,
    },
    "breakout": {
        "sl_atr_mult": 2.5, "tp_atr_mult": 4.0, "min_confidence": 0.70,
        "be_trigger_mult": 0.5, "lock_trigger_mult": 1.3, "trail_keep_pct": 0.50,
        "stale_tighten_min": 90, "scratch_exit_min": 180,
    },
    "reversal": {
        "sl_atr_mult": 1.8, "tp_atr_mult": 2.5, "min_confidence": 0.70,
        "be_trigger_mult": 0.4, "lock_trigger_mult": 1.0, "trail_keep_pct": 0.50,
        "stale_tighten_min": 60, "scratch_exit_min": 120,
    },
}

CONFIGS = {
    "CURRENT":  {"params": CURRENT_REGIME_PARAMS,  "bounds": {"sl_atr_mult": (1.0, 3.5), "tp_atr_mult": (2.0, 7.0)}},
    "MODERATE": {"params": MODERATE_REGIME_PARAMS,  "bounds": {"sl_atr_mult": (0.8, 3.0), "tp_atr_mult": (1.5, 5.0)}},
    "TIGHT":    {"params": PROPOSED_REGIME_PARAMS,  "bounds": {"sl_atr_mult": (0.8, 2.5), "tp_atr_mult": (1.5, 4.0)}},
    "OPTIMAL":  {"params": OPTIMAL_REGIME_PARAMS,   "bounds": {"sl_atr_mult": (1.0, 3.0), "tp_atr_mult": (1.5, 5.0)}},
}


def make_cfg(base_cfg: dict, regime_params: dict, bounds: dict) -> dict:
    """Create config with custom regime params."""
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("adaptive", {})
    cfg["adaptive"]["regime_params"] = regime_params
    # Merge bounds
    existing_bounds = cfg["adaptive"].get("bounds", {})
    existing_bounds.update(bounds)
    cfg["adaptive"]["bounds"] = existing_bounds
    return cfg


def print_sltp_table():
    """Print SL/TP comparison table (assumes ATR ~24pt)."""
    atr = 24.0
    print(f"\n{'=' * 90}")
    print(f"  SL/TP MULTIPLIER COMPARISON (ATR ~ {atr:.0f}pt)")
    print(f"{'=' * 90}")
    header = f"  {'Regime':<10s}"
    for name in CONFIGS:
        header += f" | {name:^24s}"
    print(header)
    print(f"  {'-' * 10}" + ("-+-" + "-" * 24) * len(CONFIGS))

    for regime in ["trending", "ranging", "breakout", "reversal"]:
        row = f"  {regime:<10s}"
        for name, conf in CONFIGS.items():
            p = conf["params"][regime]
            sl = p["sl_atr_mult"]
            tp = p["tp_atr_mult"]
            row += f" | SL={sl:.1f}({sl*atr:.0f}pt) TP={tp:.1f}({tp*atr:.0f}pt)"
        print(row)
    print()


def print_mega_comparison(all_results: dict):
    """Print all configs side by side."""
    print(f"\n{'=' * 90}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{'=' * 90}")

    metrics = [
        ("Trades",        "total",          "d"),
        ("Win Rate %",    "win_rate",       ".1f"),
        ("Profit Factor", "profit_factor",  ".2f"),
        ("Net PnL pts",   "net_pnl_pts",    "+.1f"),
        ("Net PnL $",     "net_pnl_usd",    "+.2f"),
        ("Return %",      "return_pct",     "+.1f"),
        ("Max DD %",      "max_dd_pct",     ".1f"),
        ("Avg Winner pt", "avg_win_pts",    ".1f"),
        ("Avg Loser pt",  "avg_loss_pts",   ".1f"),
        ("Avg RR",        "avg_rr",         ".2f"),
        ("Sharpe",        "sharpe",         ".2f"),
        ("Final Equity",  "final_equity",   ".2f"),
    ]

    header = f"  {'Metric':<16s}"
    for name in all_results:
        header += f" | {name:>14s}"
    print(header)
    print(f"  {'-' * 16}" + ("-+-" + "-" * 14) * len(all_results))

    for label, key, fmt in metrics:
        row = f"  {label:<16s}"
        for name, res in all_results.items():
            s = res.get("summary", {})
            val = s.get(key, 0)
            formatted = f"{val:{fmt}}"
            row += f" | {formatted:>14s}"
        print(row)

    # TP hit rate
    row = f"  {'TP Hit Rate':<16s}"
    for name, res in all_results.items():
        ct = res.get("by_close_type", {})
        tp_trades = ct.get("TP", {}).get("trades", 0)
        total = res.get("summary", {}).get("total", 1)
        pct = tp_trades / total * 100 if total > 0 else 0
        row += f" | {pct:>13.1f}%"
    print(row)

    # Avg hold time (from close types)
    print()

    # Close type breakdown per config
    for name, res in all_results.items():
        ct = res.get("by_close_type", {})
        print(f"  {name} close types: ", end="")
        parts = []
        for ctype, data in sorted(ct.items()):
            parts.append(f"{ctype}={data['trades']}({data['avg_pnl']:+.1f}pt)")
        print(" | ".join(parts))
    print()


def main():
    parser = argparse.ArgumentParser(description="SL/TP comparison backtest")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--balance", type=float, default=100.0)
    parser.add_argument("--spread", type=float, default=3.0)
    parser.add_argument("--pre-score-min", type=float, default=0.35)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 90)
    print("  SL/TP COMPARISON BACKTEST")
    print(f"  Months: {args.months} | Balance: ${args.balance} | Spread: {args.spread}pt")
    print(f"  Configs: {', '.join(CONFIGS.keys())}")
    print("=" * 90)

    print_sltp_table()

    base_cfg = load_config()
    if not connect_mt5(base_cfg):
        sys.exit(1)

    try:
        symbol = base_cfg.get("mt5", {}).get("symbol", "XAUUSD")
        df_m15, df_h1, df_h4 = fetch_historical(symbol, args.months)
        if df_m15.empty or df_h1.empty:
            print("ERROR: Failed to fetch data")
            sys.exit(1)

        period = f"{df_m15['time'].iloc[0].date()} -> {df_m15['time'].iloc[-1].date()}"
        print(f"\n  Period: {period}")
        print(f"  Bars: M15={len(df_m15)}, H1={len(df_h1)}, H4={len(df_h4)}")

        all_results = {}

        for name, conf in CONFIGS.items():
            print(f"\n{'-' * 90}")
            print(f"  Running {name}...")
            cfg = make_cfg(base_cfg, conf["params"], conf["bounds"])

            trades = run_simulation(
                df_m15, df_h1, df_h4, cfg,
                use_adaptive=True,
                spread=args.spread,
                balance=args.balance,
                pre_score_min=args.pre_score_min,
                verbose=args.verbose,
            )
            results = compute_results(trades, args.balance)
            all_results[name] = results

            print_results(results, f"{name} | {period}")

            # Save CSVs
            out_dir = _ROOT / "logs"
            out_dir.mkdir(exist_ok=True)
            write_trades_csv(trades, str(out_dir / f"bt_trades_{name.lower()}.csv"))

        # ── Side-by-side comparison ──────────────────────────────────────
        print_mega_comparison(all_results)

        # ── Recommendation ───────────────────────────────────────────────
        best_name = max(all_results, key=lambda n: all_results[n].get("summary", {}).get("net_pnl_pts", -9999))
        best = all_results[best_name]["summary"]
        print(f"  BEST CONFIG: {best_name}")
        print(f"    Net={best['net_pnl_pts']:+.1f}pt | PF={best['profit_factor']:.2f} | "
              f"WR={best['win_rate']:.1f}% | DD={best['max_dd_pct']:.1f}% | "
              f"Return={best['return_pct']:+.1f}%")

    finally:
        mt5.shutdown()
        print("\nMT5 disconnected.")


if __name__ == "__main__":
    main()
