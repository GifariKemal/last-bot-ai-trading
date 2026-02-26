"""
Signal Combination Optimizer -- Find optimal entry gate configurations.

Philosophy (from 8 legendary traders mapped to SMC):
  Soros:        Reflexive confluence (BOS+OB+FVG aligned) = highest conviction
  Dennis:       Structure signals (BOS) = trend following, ride it till CHoCH
  Kotegawa:     CHoCH = reversal opportunity, different gate than trend
  Lipschutz:    Skip weak setups, wait for A+ confluence only
  Druckenmiller: Asymmetric R:R -- fewer trades, bigger conviction
  Simons:       Let data decide -- no emotion, trust the backtest

Methodology:
  Phase 1: Run "collector" backtest (min_signals=1) -- captures ALL possible trades
  Phase 2: Analyze signal combinations post-hoc (which combos win?)
  Phase 3: Analyze individual signal impact (does X improve results?)
  Phase 4: Analyze supporting params (RSI, EMA, regime, session, P/D)
  Phase 5: Run head-to-head backtests for candidate gate configs
  Phase 6: Output ranked results + save JSON

Run:
  cd smart_trader && python optimize_signals.py
  python optimize_signals.py --months 3 --verbose
  python optimize_signals.py --months 6 --top 20
"""
import os, sys, json, math, argparse, warnings
warnings.filterwarnings("ignore", message="no explicit representation of timezones")
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "src"))

import numpy as np
import pandas as pd

# Import from existing backtest engine (100% synced with live)
from backtest_smart import (
    load_config, connect_mt5, fetch_historical,
    run_simulation, compute_results, print_results,
    SimPosition, write_trades_csv,
)

# Silence loguru
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")


# ===============================================================================
# Gate Configurations to Test
# ===============================================================================

TIER1_SIGNALS = {"BOS", "OB", "LiqSweep"}
TIER2_SIGNALS = {"FVG", "CHoCH", "Breaker", "M15", "OTE"}
TIER3_SIGNALS = {"Premium", "Discount"}

GATE_CONFIGS = [
    # -- Baseline (current live system) ------------------------------------
    {"name": "LIVE_CURRENT",     "min_signals": 1, "pd_zone_gate": False},

    # -- Signal count variations -------------------------------------------
    {"name": "MIN_SIG_1",        "min_signals": 1, "pd_zone_gate": True},
    {"name": "MIN_SIG_2",        "min_signals": 2, "pd_zone_gate": True},
    {"name": "MIN_SIG_4",        "min_signals": 4, "pd_zone_gate": True},
    {"name": "MIN_SIG_5",        "min_signals": 5, "pd_zone_gate": True},

    # -- Structure-required (Dennis/Soros: trade WITH structure) -----------
    {"name": "STRUCT_1",         "min_signals": 1, "require_structure": True, "pd_zone_gate": True},
    {"name": "STRUCT_2",         "min_signals": 2, "require_structure": True, "pd_zone_gate": True},
    {"name": "STRUCT_3",         "min_signals": 3, "require_structure": True, "pd_zone_gate": True},

    # -- CHoCH-focused (Kotegawa: reversal = CHoCH) -----------------------
    {"name": "CHOCH_ONLY",       "min_signals": 1, "require_choch": True, "pd_zone_gate": True},
    {"name": "CHOCH_PLUS_1",     "min_signals": 2, "require_choch": True, "pd_zone_gate": True},
    {"name": "CHOCH_PLUS_2",     "min_signals": 3, "require_choch": True, "pd_zone_gate": True},

    # -- BOS-focused (Dennis: trend following = BOS) ----------------------
    {"name": "BOS_ONLY",         "min_signals": 1, "require_bos": True, "pd_zone_gate": True},
    {"name": "BOS_PLUS_1",       "min_signals": 2, "require_bos": True, "pd_zone_gate": True},
    {"name": "BOS_PLUS_2",       "min_signals": 3, "require_bos": True, "pd_zone_gate": True},

    # -- Tier-1 focused (Druckenmiller: high conviction only) -------------
    {"name": "TIER1_MIN_1",      "min_signals": 2, "tier1_min": 1, "pd_zone_gate": True},
    {"name": "TIER1_MIN_2",      "min_signals": 3, "tier1_min": 2, "pd_zone_gate": True},

    # -- Without P/D gate (test if P/D gate helps or hurts) ---------------
    {"name": "NO_PD_3",          "min_signals": 3, "pd_zone_gate": False},
    {"name": "NO_PD_2",          "min_signals": 2, "pd_zone_gate": False},
    {"name": "NO_PD_STRUCT_2",   "min_signals": 2, "require_structure": True, "pd_zone_gate": False},

    # -- Aggressive (Soros: max confluence, few but powerful trades) -------
    {"name": "MAX_CONF_4S",      "min_signals": 4, "require_structure": True, "pd_zone_gate": True},
    {"name": "MAX_CONF_5S",      "min_signals": 5, "require_structure": True, "pd_zone_gate": True},
]


# ===============================================================================
# Phase 2: Signal Combination Analysis
# ===============================================================================

def analyze_signal_combos(trades: list[SimPosition], min_trades: int = 3) -> list[dict]:
    """Group trades by exact signal combination. Find which combos are profitable."""
    combos = defaultdict(list)
    for t in trades:
        key = frozenset(s for s in t.signals if s not in TIER3_SIGNALS)
        combos[key].append(t)

    results = []
    for key, group in combos.items():
        n = len(group)
        if n < min_trades:
            continue
        wins = sum(1 for t in group if t.pnl_pts > 0)
        gross_p = sum(t.pnl_pts for t in group if t.pnl_pts > 0)
        gross_l = abs(sum(t.pnl_pts for t in group if t.pnl_pts <= 0))
        pf = gross_p / gross_l if gross_l > 0 else 99.99
        net = sum(t.pnl_pts for t in group)
        avg = net / n
        wr = wins / n * 100

        # Composite score: PF × WR_fraction × sqrt(sample_size), capped
        score = min(pf, 10) * (wr / 100) * min(math.sqrt(n), 5)

        results.append({
            "signals": "+".join(sorted(key)) if key else "(empty)",
            "count": len(key),
            "trades": n,
            "wins": wins,
            "wr": round(wr, 1),
            "pf": round(min(pf, 99.99), 2),
            "avg_pnl": round(avg, 1),
            "net_pts": round(net, 1),
            "score": round(score, 2),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ===============================================================================
# Phase 3: Individual Signal Impact
# ===============================================================================

def analyze_individual_signals(trades: list[SimPosition]) -> list[dict]:
    """For each signal type: performance WITH vs WITHOUT it."""
    all_sigs = ["BOS", "CHoCH", "OB", "FVG", "LiqSweep", "Breaker", "M15", "OTE"]
    results = []

    for sig in all_sigs:
        with_sig = [t for t in trades if sig in t.signals]
        without = [t for t in trades if sig not in t.signals]

        def _stats(group):
            if not group:
                return {"n": 0, "wr": 0, "pf": 0, "avg": 0, "net": 0}
            n = len(group)
            w = sum(1 for t in group if t.pnl_pts > 0)
            gp = sum(t.pnl_pts for t in group if t.pnl_pts > 0)
            gl = abs(sum(t.pnl_pts for t in group if t.pnl_pts <= 0))
            return {
                "n": n,
                "wr": round(w / n * 100, 1),
                "pf": round(gp / gl, 2) if gl > 0 else 99.99,
                "avg": round(sum(t.pnl_pts for t in group) / n, 1),
                "net": round(sum(t.pnl_pts for t in group), 1),
            }

        ws = _stats(with_sig)
        wos = _stats(without)
        edge = ws["avg"] - wos["avg"] if ws["n"] > 0 and wos["n"] > 0 else 0

        tier = "T1" if sig in TIER1_SIGNALS else "T2"
        results.append({
            "signal": sig,
            "tier": tier,
            "with": ws,
            "without": wos,
            "edge": round(edge, 1),
        })

    results.sort(key=lambda x: x["edge"], reverse=True)
    return results


# ===============================================================================
# Phase 4: Supporting Parameter Analysis
# ===============================================================================

def analyze_supporting_params(trades: list[SimPosition]) -> dict:
    """Analyze performance by regime, session, EMA, RSI range, P/D zone."""

    def _group_stats(group):
        if not group:
            return None
        n = len(group)
        w = sum(1 for t in group if t.pnl_pts > 0)
        gp = sum(t.pnl_pts for t in group if t.pnl_pts > 0)
        gl = abs(sum(t.pnl_pts for t in group if t.pnl_pts <= 0))
        pf = gp / gl if gl > 0 else 99.99
        return {
            "trades": n, "wr": round(w / n * 100, 1),
            "pf": round(min(pf, 99.99), 2),
            "avg_pnl": round(sum(t.pnl_pts for t in group) / n, 1),
            "net_pts": round(sum(t.pnl_pts for t in group), 1),
        }

    # By regime
    by_regime = {}
    for t in trades:
        by_regime.setdefault(t.regime, []).append(t)
    regime_stats = {k: _group_stats(v) for k, v in sorted(by_regime.items())}

    # By session
    by_session = {}
    for t in trades:
        by_session.setdefault(t.session, []).append(t)
    session_stats = {k: _group_stats(v) for k, v in sorted(by_session.items())}

    # By EMA trend
    by_ema = {}
    for t in trades:
        by_ema.setdefault(t.ema_trend, []).append(t)
    ema_stats = {k: _group_stats(v) for k, v in sorted(by_ema.items())}

    # By P/D zone
    by_pd = {}
    for t in trades:
        by_pd.setdefault(t.pd_zone, []).append(t)
    pd_stats = {k: _group_stats(v) for k, v in sorted(by_pd.items())}

    # By RSI bucket
    rsi_buckets = {"0-30": [], "30-45": [], "45-55": [], "55-70": [], "70-85": [], "85+": []}
    for t in trades:
        r = t.rsi_val
        if r < 30:
            rsi_buckets["0-30"].append(t)
        elif r < 45:
            rsi_buckets["30-45"].append(t)
        elif r < 55:
            rsi_buckets["45-55"].append(t)
        elif r < 70:
            rsi_buckets["55-70"].append(t)
        elif r < 85:
            rsi_buckets["70-85"].append(t)
        else:
            rsi_buckets["85+"].append(t)
    rsi_stats = {k: _group_stats(v) for k, v in rsi_buckets.items() if v}

    # By direction
    by_dir = {}
    for t in trades:
        by_dir.setdefault(t.direction, []).append(t)
    dir_stats = {k: _group_stats(v) for k, v in sorted(by_dir.items())}

    # EMA alignment (direction matches trend?)
    aligned = [t for t in trades if
               (t.direction == "LONG" and t.ema_trend == "BULLISH") or
               (t.direction == "SHORT" and t.ema_trend == "BEARISH")]
    counter = [t for t in trades if
               (t.direction == "LONG" and t.ema_trend == "BEARISH") or
               (t.direction == "SHORT" and t.ema_trend == "BULLISH")]
    neutral = [t for t in trades if t.ema_trend == "NEUTRAL"]
    alignment_stats = {
        "ALIGNED": _group_stats(aligned),
        "COUNTER": _group_stats(counter),
        "NEUTRAL": _group_stats(neutral),
    }

    return {
        "by_regime": regime_stats,
        "by_session": session_stats,
        "by_ema": ema_stats,
        "by_pd_zone": pd_stats,
        "by_rsi": rsi_stats,
        "by_direction": dir_stats,
        "ema_alignment": alignment_stats,
    }


# ===============================================================================
# Phase 5: Grid Search -- Run Each Gate Config
# ===============================================================================

def run_grid_search(
    df_m15, df_h1, df_h4, cfg,
    configs: list[dict],
    spread: float, balance: float, pre_score_min: float,
) -> list[dict]:
    """Run backtest for each gate config and collect results."""
    all_results = []
    total = len(configs)

    for idx, gc in enumerate(configs, 1):
        name = gc.get("name", f"config_{idx}")
        print(f"\n  [{idx}/{total}] Running: {name} ...", end="", flush=True)

        trades = run_simulation(
            df_m15, df_h1, df_h4, cfg,
            use_adaptive=True, spread=spread,
            balance=balance, pre_score_min=pre_score_min,
            verbose=False, gate_config=gc,
        )
        results = compute_results(trades, balance)
        s = results.get("summary", {})

        # Composite score: PF × WR × sqrt(trades) × (1 - DD_frac)
        pf = s.get("profit_factor", 0)
        wr = s.get("win_rate", 0) / 100
        n = s.get("total", 0)
        dd = s.get("max_dd_pct", 100) / 100
        composite = min(pf, 10) * wr * min(math.sqrt(n), 10) * max(1 - dd, 0.01)

        entry = {
            "name": name,
            "config": {k: v for k, v in gc.items() if k != "name"},
            "trades": n,
            "wins": s.get("wins", 0),
            "wr": s.get("win_rate", 0),
            "pf": min(pf, 99.99),
            "net_pts": s.get("net_pnl_pts", 0),
            "net_usd": s.get("net_pnl_usd", 0),
            "return_pct": s.get("return_pct", 0),
            "max_dd_pct": s.get("max_dd_pct", 0),
            "avg_rr": s.get("avg_rr", 0),
            "sharpe": s.get("sharpe", 0),
            "composite": round(composite, 2),
        }
        all_results.append(entry)

        status = f" {n} trades | WR={entry['wr']:.0f}% | PF={pf:.2f} | Net={entry['net_pts']:+.0f}pt"
        print(status)

    all_results.sort(key=lambda x: x["composite"], reverse=True)
    return all_results


# ===============================================================================
# Output Formatting
# ===============================================================================

def print_signal_combos(combos: list[dict], top_n: int = 15):
    print(f"\n{'=' * 90}")
    print(f"  PHASE 2: SIGNAL COMBINATION ANALYSIS (top {top_n})")
    print(f"  Trader insight: Which signal combos produce the best results?")
    print(f"{'=' * 90}")
    print(f"  {'Signals':<35s} | {'N':>4s} | {'WR':>5s} | {'PF':>6s} | {'AvgPnL':>7s} | {'NetPts':>8s} | {'Score':>5s}")
    print(f"  {'-' * 35}-+-{'-' * 4}-+-{'-' * 5}-+-{'-' * 6}-+-{'-' * 7}-+-{'-' * 8}-+-{'-' * 5}")
    for c in combos[:top_n]:
        sig_str = c["signals"][:35]
        pf_str = f"{c['pf']:.2f}" if c["pf"] < 99 else "  inf"
        print(f"  {sig_str:<35s} | {c['trades']:>4d} | {c['wr']:>4.0f}% | {pf_str:>6s} | {c['avg_pnl']:>+6.1f}pt | {c['net_pts']:>+7.1f}pt | {c['score']:>5.1f}")


def print_individual_signals(sig_analysis: list[dict]):
    print(f"\n{'=' * 90}")
    print(f"  PHASE 3: INDIVIDUAL SIGNAL IMPACT")
    print(f"  Q: Does having signal X improve results?")
    print(f"{'=' * 90}")
    print(f"  {'Signal':<10s} {'Tier':>4s} | {'WITH':^30s} | {'WITHOUT':^30s} | {'Edge':>6s}")
    print(f"  {'':10s} {'':4s} | {'N':>5s} {'WR':>5s} {'PF':>6s} {'Avg':>7s} {'Net':>7s} | "
          f"{'N':>5s} {'WR':>5s} {'PF':>6s} {'Avg':>7s} {'Net':>7s} | {'':>6s}")
    print(f"  {'-' * 14}-+-{'-' * 30}-+-{'-' * 30}-+-{'-' * 6}")
    for s in sig_analysis:
        w = s["with"]
        wo = s["without"]
        edge_str = f"{s['edge']:>+5.1f}pt"
        marker = " **" if s["edge"] > 2.0 else " *" if s["edge"] > 0.5 else ""
        w_pf = f"{w['pf']:.2f}" if w["pf"] < 99 else "  inf"
        wo_pf = f"{wo['pf']:.2f}" if wo["pf"] < 99 else "  inf"
        print(f"  {s['signal']:<10s} {s['tier']:>4s} | "
              f"{w['n']:>5d} {w['wr']:>4.0f}% {w_pf:>6s} {w['avg']:>+6.1f}pt {w['net']:>+6.0f}pt | "
              f"{wo['n']:>5d} {wo['wr']:>4.0f}% {wo_pf:>6s} {wo['avg']:>+6.1f}pt {wo['net']:>+6.0f}pt | "
              f"{edge_str}{marker}")


def print_supporting_params(params: dict):
    print(f"\n{'=' * 90}")
    print(f"  PHASE 4: SUPPORTING PARAMETER ANALYSIS")
    print(f"{'=' * 90}")

    sections = [
        ("BY REGIME", params["by_regime"]),
        ("BY SESSION", params["by_session"]),
        ("EMA ALIGNMENT", params["ema_alignment"]),
        ("BY P/D ZONE", params["by_pd_zone"]),
        ("BY RSI RANGE", params["by_rsi"]),
        ("BY DIRECTION", params["by_direction"]),
    ]

    for title, data in sections:
        print(f"\n  {title}:")
        print(f"  {'Key':<14s} | {'Trades':>6s} | {'WR':>5s} | {'PF':>6s} | {'AvgPnL':>7s} | {'NetPts':>8s}")
        print(f"  {'-' * 14}-+-{'-' * 6}-+-{'-' * 5}-+-{'-' * 6}-+-{'-' * 7}-+-{'-' * 8}")
        for k, v in data.items():
            if v is None:
                continue
            pf_str = f"{v['pf']:.2f}" if v["pf"] < 99 else "  inf"
            print(f"  {k:<14s} | {v['trades']:>6d} | {v['wr']:>4.0f}% | {pf_str:>6s} | {v['avg_pnl']:>+6.1f}pt | {v['net_pts']:>+7.1f}pt")


def print_grid_results(results: list[dict], top_n: int = 15):
    print(f"\n{'=' * 110}")
    print(f"  PHASE 5: GATE CONFIGURATION COMPARISON (ranked by composite score)")
    print(f"{'=' * 110}")
    print(f"  {'#':>2s} {'Config':<18s} | {'Trades':>6s} | {'WR':>5s} | {'PF':>6s} | {'NetPts':>8s} | {'Return':>7s} | "
          f"{'DD':>5s} | {'RR':>4s} | {'Sharpe':>6s} | {'Score':>6s}")
    print(f"  {'-' * 2} {'-' * 18}-+-{'-' * 6}-+-{'-' * 5}-+-{'-' * 6}-+-{'-' * 8}-+-{'-' * 7}-+-"
          f"{'-' * 5}-+-{'-' * 4}-+-{'-' * 6}-+-{'-' * 6}")
    for idx, r in enumerate(results[:top_n], 1):
        pf_str = f"{r['pf']:.2f}" if r["pf"] < 99 else "  inf"
        marker = " <-- LIVE" if r["name"] == "LIVE_CURRENT" else ""
        print(f"  {idx:>2d} {r['name']:<18s} | {r['trades']:>6d} | {r['wr']:>4.0f}% | {pf_str:>6s} | "
              f"{r['net_pts']:>+7.1f}pt | {r['return_pct']:>+6.1f}% | {r['max_dd_pct']:>4.1f}% | "
              f"{r['avg_rr']:>4.2f} | {r['sharpe']:>6.2f} | {r['composite']:>6.2f}{marker}")


def print_recommendations(grid_results: list[dict], sig_analysis: list[dict], combos: list[dict]):
    """Print actionable recommendations based on all analyses."""
    print(f"\n{'=' * 90}")
    print(f"  RECOMMENDATIONS -- Trading Philosophy Applied to Data")
    print(f"{'=' * 90}")

    # Find live baseline
    live = next((r for r in grid_results if r["name"] == "LIVE_CURRENT"), None)
    best = grid_results[0] if grid_results else None

    if live and best:
        print(f"\n  Current live system: {live['trades']} trades, WR={live['wr']:.0f}%, "
              f"PF={live['pf']:.2f}, Net={live['net_pts']:+.1f}pt")
        print(f"  Best config found:  {best['name']} -- {best['trades']} trades, WR={best['wr']:.0f}%, "
              f"PF={best['pf']:.2f}, Net={best['net_pts']:+.1f}pt")

        if best["composite"] > (live.get("composite", 0) * 1.1):
            print(f"\n  ** {best['name']} outperforms LIVE by "
                  f"{(best['composite'] / max(live['composite'], 0.01) - 1) * 100:.0f}% composite **")

    # Signal impact insights
    print(f"\n  SIGNAL INSIGHTS (from individual impact analysis):")
    pos_edge = [s for s in sig_analysis if s["edge"] > 0.5]
    neg_edge = [s for s in sig_analysis if s["edge"] < -0.5]

    if pos_edge:
        print(f"  Positive edge signals (having these IMPROVES results):")
        for s in pos_edge:
            print(f"    + {s['signal']} ({s['tier']}): edge = {s['edge']:+.1f}pt/trade")
    if neg_edge:
        print(f"  Negative edge signals (having these HURTS results):")
        for s in neg_edge:
            print(f"    - {s['signal']} ({s['tier']}): edge = {s['edge']:+.1f}pt/trade")

    # Top combos
    if combos:
        print(f"\n  TOP 3 SIGNAL COMBINATIONS:")
        for i, c in enumerate(combos[:3], 1):
            print(f"    {i}. {c['signals']} -- WR={c['wr']:.0f}%, PF={c['pf']:.2f}, "
                  f"N={c['trades']}, Avg={c['avg_pnl']:+.1f}pt")

    print()


# ===============================================================================
# Main
# ===============================================================================

def main():
    parser = argparse.ArgumentParser(description="Signal Combination Optimizer")
    parser.add_argument("--months", type=int, default=3, help="Backtest period (default: 3)")
    parser.add_argument("--balance", type=float, default=100.0, help="Starting balance (default: 100)")
    parser.add_argument("--spread", type=float, default=3.0, help="Spread in points (default: 3.0)")
    parser.add_argument("--pre-score-min", type=float, default=0.35, help="Min algo pre-score (default: 0.35)")
    parser.add_argument("--top", type=int, default=15, help="Top N results to show (default: 15)")
    parser.add_argument("--min-combo-trades", type=int, default=3, help="Min trades per combo (default: 3)")
    parser.add_argument("--verbose", action="store_true", help="Show individual trades")
    parser.add_argument("--symbol", default=None, help="Override symbol")
    parser.add_argument("--output", default="data/signal_optimization", help="Output directory")
    args = parser.parse_args()

    print("=" * 90)
    print("  Signal Combination Optimizer")
    print(f"  Period: {args.months} months | Balance: ${args.balance} | Spread: {args.spread}pt")
    print(f"  Pre-score min: {args.pre_score_min} | Min combo trades: {args.min_combo_trades}")
    print("=" * 90)

    cfg = load_config()
    symbol = args.symbol or cfg.get("mt5", {}).get("symbol", "XAUUSD")

    if not connect_mt5(cfg):
        sys.exit(1)

    try:
        import MetaTrader5 as mt5
        df_m15, df_h1, df_h4 = fetch_historical(symbol, args.months)
        if df_m15.empty or df_h1.empty:
            print("ERROR: Failed to fetch data")
            sys.exit(1)

        period = f"{df_m15['time'].iloc[0].date()} -> {df_m15['time'].iloc[-1].date()}"
        print(f"\n  Period: {period}")
        print(f"  Bars: M15={len(df_m15)}, H1={len(df_h1)}, H4={len(df_h4)}")

        # -- Phase 1: Collector Run (min_signals=1, no P/D gate) -----------
        print(f"\n{'-' * 90}")
        print("  PHASE 1: Collector run (min_signals=1, no P/D gate, no structure req)")
        print("  Purpose: capture ALL possible trades for signal combination analysis")
        print(f"{'-' * 90}")

        collector_config = {"min_signals": 1, "pd_zone_gate": False}
        all_trades = run_simulation(
            df_m15, df_h1, df_h4, cfg,
            use_adaptive=True, spread=args.spread,
            balance=args.balance, pre_score_min=args.pre_score_min,
            verbose=args.verbose, gate_config=collector_config,
        )
        print(f"\n  Collector captured: {len(all_trades)} trades")

        if not all_trades:
            print("  ERROR: No trades generated. Check data/config.")
            sys.exit(1)

        # Quick summary
        wins = sum(1 for t in all_trades if t.pnl_pts > 0)
        net = sum(t.pnl_pts for t in all_trades)
        print(f"  Quick stats: {wins}/{len(all_trades)} wins ({wins/len(all_trades)*100:.0f}%), "
              f"Net={net:+.1f}pt")

        # -- Phase 2: Signal Combination Analysis --------------------------
        combos = analyze_signal_combos(all_trades, min_trades=args.min_combo_trades)
        print_signal_combos(combos, top_n=args.top)

        # -- Phase 3: Individual Signal Impact -----------------------------
        sig_analysis = analyze_individual_signals(all_trades)
        print_individual_signals(sig_analysis)

        # -- Phase 4: Supporting Parameter Analysis ------------------------
        params = analyze_supporting_params(all_trades)
        print_supporting_params(params)

        # -- Phase 5: Grid Search ------------------------------------------
        print(f"\n{'-' * 90}")
        print(f"  PHASE 5: Grid Search -- {len(GATE_CONFIGS)} gate configurations")
        print(f"{'-' * 90}")

        grid_results = run_grid_search(
            df_m15, df_h1, df_h4, cfg,
            configs=GATE_CONFIGS,
            spread=args.spread, balance=args.balance,
            pre_score_min=args.pre_score_min,
        )
        print_grid_results(grid_results, top_n=args.top)

        # -- Phase 6: Recommendations -------------------------------------
        print_recommendations(grid_results, sig_analysis, combos)

        # -- Save Results --------------------------------------------------
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        output = {
            "metadata": {
                "period": period,
                "months": args.months,
                "symbol": symbol,
                "balance": args.balance,
                "spread": args.spread,
                "pre_score_min": args.pre_score_min,
                "total_collector_trades": len(all_trades),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "signal_combos": combos[:50],
            "individual_signals": sig_analysis,
            "supporting_params": params,
            "grid_results": grid_results,
        }

        json_path = out_dir / "signal_optimization_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Results saved: {json_path}")

        # Save collector trades CSV
        csv_path = str(out_dir / "collector_trades.csv")
        write_trades_csv(all_trades, csv_path)

        print(f"\n  Done! Review results at: {out_dir}/")

    finally:
        import MetaTrader5 as mt5
        mt5.shutdown()
        print("  MT5 disconnected.")


if __name__ == "__main__":
    main()
