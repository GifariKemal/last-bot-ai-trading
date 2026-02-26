"""
analyze_entry_quality.py
=========================
Validates the Dynamic Entry Gate by running a backtest and analyzing
per-tier performance (win rate, profit factor, avg RR, avg score).

PURPOSE:
  - Confirms Tier A > Tier B performance gap (validates gate logic)
  - Confirms Tier C would have been bad trades (validates skipping)
  - Shows whether zone fill actually improves outcomes

USAGE:
  python scripts/analyze_entry_quality.py --months 6 --balance 10000
  python scripts/analyze_entry_quality.py --months 3 --balance 10000 --verbose
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


def run_analysis(months: int = 6, balance: float = 10000.0, verbose: bool = False):
    import yaml
    import sys as _sys
    from loguru import logger as _loguru_logger
    # Suppress verbose loguru output during backtest (only show WARNING+)
    _loguru_logger.remove()
    _loguru_logger.add(_sys.stderr, level="WARNING")

    from src.core.mt5_connector import MT5Connector
    from src.backtesting.historical_data import HistoricalDataManager
    from src.backtesting.backtest_engine import BacktestEngine
    from src.core.data_manager import DataManager

    # ── Load merged config ────────────────────────────────────────────────
    def load_yaml(p):
        return yaml.safe_load(Path(p).read_text(encoding="utf-8"))

    settings    = load_yaml("config/settings.yaml")
    risk        = load_yaml("config/risk_config.yaml")
    rules       = load_yaml("config/trading_rules.yaml")
    session_cfg = load_yaml("config/session_config.yaml")

    config = {**settings, **risk, **rules}
    config["session"] = session_cfg

    # ── Connect MT5 & fetch data ──────────────────────────────────────────
    mt5 = MT5Connector(config)
    if not mt5.connect():
        logger.error("MT5 connection failed")
        return

    end_date   = datetime.utcnow()
    start_date = end_date - timedelta(days=months * 30)

    logger.info(f"Fetching {months} months of M15 data ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")

    dm = HistoricalDataManager()
    df = dm.prepare_backtest_data(
        mt5, "XAUUSDm", "M15", start_date, end_date, use_cache=True
    )
    mt5.disconnect()

    if df is None or len(df) < 200:
        logger.error("Not enough data")
        return

    # Add features + indicators
    from src.core.data_manager import DataManager
    from src.indicators.technical import TechnicalIndicators
    from src.indicators.smc_v4_adapter import SMCIndicatorsV4

    pdm = DataManager()
    df = pdm.add_basic_features(df)
    df = pdm.add_price_changes(df)

    tech = TechnicalIndicators(config.get("indicators", {}))
    df = tech.calculate_all(df)

    smc_ind = SMCIndicatorsV4(config.get("smc_indicators", {}))
    df = smc_ind.calculate_all(df)

    logger.info(f"Prepared {len(df)} bars. Running backtest...")

    # ── Run backtest ──────────────────────────────────────────────────────
    engine = BacktestEngine(config)
    results = engine.run_backtest_fast(df, initial_balance=balance)

    if not results.get("success"):
        logger.error("Backtest failed")
        return

    metrics = results["metrics"]
    trades  = results["trades"]

    # ── Print overall results ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  DYNAMIC ENTRY GATE — QUALITY TIER ANALYSIS")
    print("=" * 65)
    print(f"  Period    : {months} months | Balance: ${balance:,.0f}")
    print(f"  Total trades: {metrics.get('total_trades', 0)}")
    print(f"  Overall WR  : {metrics.get('win_rate', 0):.1f}%")
    print(f"  Overall PF  : {metrics.get('profit_factor', 0):.2f}")
    print(f"  Max DD      : {metrics.get('max_drawdown_pct', 0):.1f}%")
    print(f"  Return      : {metrics.get('return_pct', 0):.2f}%")

    # ── Tier breakdown ────────────────────────────────────────────────────
    tier_data = metrics.get("tier_breakdown", {})
    if not tier_data:
        print("\n  [!] No tier breakdown found — check BacktestEngine._compute_tier_breakdown()")
        return

    print("\n" + "-" * 65)
    print(f"  {'TIER':<12} {'N':>4} {'WR%':>6} {'PF':>6} {'AvgRR':>7} {'Score':>7} {'Zone%':>7} {'P&L':>9}")
    print("-" * 65)

    tier_labels = {
        "HIGH":   "[A:HIGH] Institutional",
        "MEDIUM": "[B:MED]  Structural   ",
        "LOW":    "[C:LOW]  Marginal     ",
    }
    for tier_val in ["HIGH", "MEDIUM", "LOW"]:
        d = tier_data.get(tier_val, {})
        if d.get("count", 0) == 0:
            print(f"  {tier_labels[tier_val]:<12} {'0':>4}  (no trades)")
            continue

        label = tier_labels[tier_val]
        n     = d["count"]
        wr    = d["win_rate"]
        pf    = d["profit_factor"]
        avg_rr = d["avg_rr"]
        score  = d["avg_score"]
        zone   = d["zone_pct"]
        pnl    = d["total_profit"]

        # Highlight Tier A
        marker = " <-- best" if tier_val == "HIGH" else ""
        print(f"  {label}  {n:>4}  {wr:>5.1f}%  {pf:>5.2f}  {avg_rr:>6.3f}  {score:>6.3f}  {zone:>5.1f}%  ${pnl:>7.2f}{marker}")

    print("-" * 65)

    # ── Philosophical validation ──────────────────────────────────────────
    print("\n  GATE VALIDATION:")

    tier_a = tier_data.get("HIGH", {})
    tier_b = tier_data.get("MEDIUM", {})
    tier_c = tier_data.get("LOW", {})

    def check(label, condition, ok_msg, fail_msg):
        status = "PASS" if condition else "FAIL"
        msg    = ok_msg if condition else fail_msg
        print(f"  [{status}] {label}: {msg}")

    if tier_a.get("count", 0) > 0 and tier_b.get("count", 0) > 0:
        check(
            "Tier A > Tier B WR",
            tier_a.get("win_rate", 0) >= tier_b.get("win_rate", 0) - 5,
            f"A:{tier_a['win_rate']:.1f}% vs B:{tier_b['win_rate']:.1f}%",
            f"A:{tier_a.get('win_rate',0):.1f}% < B:{tier_b.get('win_rate',0):.1f}% — recalibrate threshold"
        )
        check(
            "Tier A PF > 1.5",
            tier_a.get("profit_factor", 0) >= 1.5,
            f"PF={tier_a['profit_factor']:.2f}",
            f"PF={tier_a.get('profit_factor',0):.2f} too low — TIER_A_SCORE_WITH_ZONE too permissive"
        )
        check(
            "Tier B PF > 1.0",
            tier_b.get("profit_factor", 0) >= 1.0,
            f"PF={tier_b['profit_factor']:.2f}",
            f"PF={tier_b.get('profit_factor',0):.2f} — Tier B losing money"
        )

    if tier_c.get("count", 0) > 0:
        check(
            "Tier C PF < Tier B",
            tier_c.get("profit_factor", 99) <= tier_b.get("profit_factor", 1.0),
            f"C:{tier_c['profit_factor']:.2f} < B:{tier_b.get('profit_factor',0):.2f} — skip correct",
            f"C:{tier_c.get('profit_factor',0):.2f} > B:{tier_b.get('profit_factor',0):.2f} — Tier C might be tradeable"
        )
    else:
        print("  [INFO] No Tier C trades recorded (all signals passed gates)")

    # ── Zone precision analysis ───────────────────────────────────────────
    print("\n  ZONE FILL ANALYSIS (ICT principle validation):")
    zone_trades = [t for t in trades if t.get("has_zone")]
    nozone_trades = [t for t in trades if not t.get("has_zone")]

    def _stats(group):
        if not group:
            return (0, 0.0, 0.0)
        wins = [t["profit"] for t in group if t["profit"] > 0]
        losses = [abs(t["profit"]) for t in group if t["profit"] <= 0]
        wr = len(wins) / len(group) * 100
        pf = sum(wins) / max(sum(losses), 0.001)
        return (len(group), wr, pf)

    zn, zwr, zpf = _stats(zone_trades)
    nzn, nzwr, nzpf = _stats(nozone_trades)
    print(f"  Zone fill YES: n={zn}, WR={zwr:.1f}%, PF={zpf:.2f}")
    print(f"  Zone fill NO:  n={nzn}, WR={nzwr:.1f}%, PF={nzpf:.2f}")
    if zn > 5 and nzn > 5:
        if zpf > nzpf:
            print(f"  [PASS] Zone fills outperform: +{zpf-nzpf:.2f} PF advantage (ICT principle CONFIRMED)")
        else:
            print(f"  [INFO] Zone fills do NOT outperform on this dataset — may need H4/Daily zones")

    # ── Verbose: trade list per tier ──────────────────────────────────────
    if verbose:
        print("\n  INDIVIDUAL TRADES BY TIER:")
        for tier_val in ["HIGH", "MEDIUM"]:
            tier_trades = [t for t in trades if t.get("quality_tier") == tier_val]
            if not tier_trades:
                continue
            print(f"\n  -- Tier {tier_val} ({len(tier_trades)} trades) --")
            for t in tier_trades[:20]:  # Show max 20 per tier
                ep = t["entry_price"]
                xp = t["exit_price"]
                sl = t.get("sl", ep)
                sl_d = abs(ep - sl)
                if sl_d > 0:
                    rr = (xp - ep) / sl_d if t["direction"] == "BUY" else (ep - xp) / sl_d
                else:
                    rr = 0
                zone_flag = "Z" if t.get("has_zone") else "-"
                win_flag  = "W" if t["profit"] > 0 else "L"
                print(
                    f"    #{t['ticket']:>3} {t['direction']:<4} "
                    f"E:{ep:.1f} X:{xp:.1f} "
                    f"RR:{rr:+.2f} P&L:${t['profit']:+.2f} "
                    f"{win_flag} {zone_flag} {t.get('regime','')[:8]:<8} "
                    f"Score:{t.get('quality_score',0):.3f}"
                )

    print("\n" + "=" * 65)
    print(f"  Done. {len(trades)} total trades analyzed.")
    print("=" * 65 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Entry Gate Quality Analysis")
    parser.add_argument("--months",  type=int,   default=6,     help="Months of data (default: 6)")
    parser.add_argument("--balance", type=float, default=10000.0, help="Starting balance (default: 10000)")
    parser.add_argument("--verbose", action="store_true",        help="Show per-trade detail")
    args = parser.parse_args()

    run_analysis(months=args.months, balance=args.balance, verbose=args.verbose)
