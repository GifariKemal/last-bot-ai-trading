"""
Event Backtest Comparison Runner
=================================
Runs 4 tier configurations + Optuna optimization to find optimal event bias parameters.

Configurations:
  1. baseline  - Pure SMC (no events)
  2. tier_a    - SMC + Holiday/Economic calendar + DXY
  3. tier_b    - SMC + News/GDELT sentiment
  4. tier_ab   - SMC + All sources combined

For each tier, Optuna finds optimal:
  - boost_multiplier      (0.0 to 0.20)
  - penalty_multiplier    (0.0 to 0.15)
  - economic_weight       (0.0 to 0.60)
  - dxy_weight            (0.0 to 0.50)
  - news_weight           (0.0 to 0.40)
  - event_lookback_hours  (4 to 24)
  - dxy_bars_back         (2 to 8)

Usage:
  cd xauusd_trading_bot/data/event_backtest
  python run_comparison.py
  python run_comparison.py --no-fetch   # Skip data fetch if already done
  python run_comparison.py --no-optuna  # Just run default configs, no optimization
  python run_comparison.py --trials 50  # Override Optuna trial count
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from event_fetcher import fetch_and_store_all
from event_backtest import run_tier, TIER_CONFIGS
from database import initialize_database, save_backtest_run, get_best_results

# ─── Backtest Period ──────────────────────────────────────────────────────────

BACKTEST_START = datetime(2025, 11, 17, tzinfo=timezone.utc)
BACKTEST_END   = datetime(2026, 2, 17, tzinfo=timezone.utc)
INITIAL_BALANCE = 10000.0

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─── Score Function (matches run_backtest.py Optuna scoring) ──────────────────

def compute_score(metrics: dict) -> float:
    """
    Compute composite score from backtest metrics.
    Higher is better. Penalizes poor win rate and high drawdown.
    Matches the scoring formula used in the main Optuna optimization.
    """
    pf   = metrics.get('profit_factor', 0)
    wr   = metrics.get('win_rate', 0)
    dd   = metrics.get('max_drawdown_percent', metrics.get('max_drawdown', 100))
    ret  = metrics.get('total_return_percent', metrics.get('total_return', 0))
    rr   = metrics.get('avg_rr_ratio', 1.0)

    # Guard against degenerate results
    if pf < 0.5 or wr < 30 or dd > 20:
        return 0.0

    score = (
        pf * 20 +
        wr * 0.5 +
        (10 - dd) * 2 +
        ret * 0.3 +
        rr * 5
    )
    return max(0.0, score)


# ─── MT5 Connector Helper ─────────────────────────────────────────────────────

def get_mt5_connector(config: dict):
    """Create a minimal MT5 connector for backtesting."""
    from src.core.mt5_connector import MT5Connector
    connector = MT5Connector(config.get('mt5', {}))
    success = connector.connect()
    if not success:
        raise RuntimeError("Failed to connect to MT5")
    return connector


def load_bot_config() -> dict:
    """Load standard bot configuration directly from YAML files."""
    import yaml

    # Change to project root so all relative imports work
    os.chdir(PROJECT_ROOT)

    config_dir = os.path.join(PROJECT_ROOT, 'config')
    config = {}
    for name in ['settings', 'mt5_config', 'trading_rules', 'risk_config', 'session_config']:
        path = os.path.join(config_dir, f'{name}.yaml')
        if os.path.exists(path):
            try:
                with open(path) as f:
                    loaded = yaml.safe_load(f)
                    if loaded:
                        config.update(loaded)
                print(f"  Loaded {name}.yaml")
            except Exception as ex:
                print(f"  [Config] Warning: {name}: {ex}")
        else:
            print(f"  [Config] Not found: {path}")

    return config


# ─── Single Tier Run ──────────────────────────────────────────────────────────

def run_single_tier(config: dict, mt5, tier_label: str, scorer_override: dict = None) -> dict:
    """Run one tier and save result to DB."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {tier_label.upper()}")
    if scorer_override:
        print(f"  Scorer override: {scorer_override}")
    print(f"{'='*60}")

    try:
        result = run_tier(
            config=config,
            mt5=mt5,
            tier_label=tier_label,
            start_date=BACKTEST_START,
            end_date=BACKTEST_END,
            scorer_override=scorer_override,
            initial_balance=INITIAL_BALANCE,
        )
    except Exception as ex:
        print(f"[ERROR] {tier_label} failed: {ex}")
        import traceback
        traceback.print_exc()
        return {}

    if not result.get("success"):
        print(f"[FAIL] {tier_label}: {result.get('error', 'unknown')}")
        return {}

    metrics = result.get("metrics", {})
    score = compute_score(metrics)

    # Save to DB
    save_backtest_run({
        'tier_label':    tier_label,
        'tier_config':   scorer_override or TIER_CONFIGS.get(tier_label, {}),
        'start_date':    BACKTEST_START.strftime('%Y-%m-%d'),
        'end_date':      BACKTEST_END.strftime('%Y-%m-%d'),
        'profit_factor': metrics.get('profit_factor', 0),
        'win_rate':      metrics.get('win_rate', 0),
        'total_return':  metrics.get('total_return_percent', 0),
        'max_drawdown':  metrics.get('max_drawdown_percent', 0),
        'total_trades':  metrics.get('total_trades', 0),
        'avg_rr':        metrics.get('avg_rr_ratio', 0),
        'score':         score,
        'optuna_trial':  -1,  # Will be updated by Optuna callback
    })

    print(f"\n[{tier_label}] Score={score:.2f} | PF={metrics.get('profit_factor',0):.2f} | "
          f"WR={metrics.get('win_rate',0):.1f}% | DD={metrics.get('max_drawdown_percent',0):.1f}% | "
          f"Trades={metrics.get('total_trades',0)} | Return={metrics.get('total_return_percent',0):.2f}%")

    return {
        'tier_label': tier_label,
        'score':      score,
        'metrics':    metrics,
        'tier_config': scorer_override or TIER_CONFIGS.get(tier_label, {}),
        'event_impact': result.get('event_impact', {}),
    }


# ─── Optuna Optimization Per Tier ─────────────────────────────────────────────

def optimize_tier(config: dict, mt5, tier_label: str, n_trials: int = 50) -> dict:
    """
    Use Optuna to find optimal scorer parameters for a given tier.
    Returns the best scorer config found.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[Optuna] Not installed. Skipping optimization.")
        return {}

    print(f"\n{'='*60}")
    print(f"OPTUNA: Optimizing {tier_label} ({n_trials} trials)")
    print(f"{'='*60}")

    tier_base = TIER_CONFIGS.get(tier_label, {})
    best_score = [0.0]
    best_config = [{}]

    def objective(trial):
        scorer_config = dict(tier_base)  # Start from tier defaults

        # Only suggest parameters that are relevant for this tier
        if tier_base.get('use_events') or tier_base.get('use_dxy'):
            scorer_config['economic_weight']  = trial.suggest_float('economic_weight', 0.1, 0.60)
            scorer_config['dxy_weight']       = trial.suggest_float('dxy_weight',      0.0, 0.50)
            scorer_config['event_lookback_hours'] = trial.suggest_int('event_lookback_hours', 4, 24)
            scorer_config['dxy_bars_back']    = trial.suggest_int('dxy_bars_back',     2, 8)

        if tier_base.get('use_news'):
            scorer_config['news_weight']      = trial.suggest_float('news_weight',     0.0, 0.40)
            scorer_config['news_lookback_hours'] = trial.suggest_int('news_lookback_hours', 4, 24)

        scorer_config['boost_multiplier']   = trial.suggest_float('boost_multiplier',   0.0, 0.20)
        scorer_config['penalty_multiplier'] = trial.suggest_float('penalty_multiplier', 0.0, 0.15)
        scorer_config['min_confidence_to_boost']    = trial.suggest_float('min_conf_boost',   0.20, 0.70)
        scorer_config['min_confidence_to_penalize'] = trial.suggest_float('min_conf_penalty', 0.15, 0.60)

        try:
            result = run_tier(
                config=config,
                mt5=mt5,
                tier_label=tier_label,
                start_date=BACKTEST_START,
                end_date=BACKTEST_END,
                scorer_override=scorer_config,
                initial_balance=INITIAL_BALANCE,
            )
        except Exception as ex:
            print(f"  Trial failed: {ex}")
            return 0.0

        if not result.get("success"):
            return 0.0

        metrics = result.get("metrics", {})
        score = compute_score(metrics)

        # Track best
        if score > best_score[0]:
            best_score[0] = score
            best_config[0] = scorer_config.copy()
            print(f"  Trial #{trial.number}: NEW BEST score={score:.2f} | "
                  f"PF={metrics.get('profit_factor',0):.2f} | "
                  f"WR={metrics.get('win_rate',0):.1f}% | "
                  f"Trades={metrics.get('total_trades',0)}")

        # Save to DB
        save_backtest_run({
            'tier_label':    f"{tier_label}_optuna",
            'tier_config':   scorer_config,
            'start_date':    BACKTEST_START.strftime('%Y-%m-%d'),
            'end_date':      BACKTEST_END.strftime('%Y-%m-%d'),
            'profit_factor': metrics.get('profit_factor', 0),
            'win_rate':      metrics.get('win_rate', 0),
            'total_return':  metrics.get('total_return_percent', 0),
            'max_drawdown':  metrics.get('max_drawdown_percent', 0),
            'total_trades':  metrics.get('total_trades', 0),
            'avg_rr':        metrics.get('avg_rr_ratio', 0),
            'score':         score,
            'optuna_trial':  trial.number,
        })

        return score

    study = optuna.create_study(direction='maximize',
                                 study_name=f"event_backtest_{tier_label}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[Optuna/{tier_label}] Best score={best_score[0]:.2f}")
    if best_config[0]:
        print(f"  Best config: {json.dumps(best_config[0], indent=2)}")

    return best_config[0]


# ─── Report Generation ────────────────────────────────────────────────────────

def generate_report(results: list) -> str:
    """Generate a comparison report from all tier results."""
    lines = [
        "",
        "=" * 70,
        "EVENT BACKTEST COMPARISON REPORT",
        f"Period: {BACKTEST_START.date()} to {BACKTEST_END.date()}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 70,
        "",
        f"{'Tier':<20} {'Score':>7} {'PF':>6} {'WR%':>7} {'DD%':>7} "
        f"{'Return%':>9} {'Trades':>7} {'Boosts':>7}",
        "-" * 70,
    ]

    # Sort by score descending
    sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)

    baseline_score = next((r['score'] for r in results if r['tier_label'] == 'baseline'), 0)

    for r in sorted_results:
        m     = r.get('metrics', {})
        score = r.get('score', 0)
        delta = score - baseline_score
        delta_str = f"(+{delta:.1f})" if delta > 0 else f"({delta:.1f})" if delta != 0 else ""

        boosts = r.get('event_impact', {}).get('boosts', '-')

        lines.append(
            f"{r['tier_label']:<20} {score:>7.2f} "
            f"{m.get('profit_factor',0):>6.2f} "
            f"{m.get('win_rate',0):>6.1f}% "
            f"{m.get('max_drawdown_percent',0):>6.1f}% "
            f"{m.get('total_return_percent',0):>8.2f}% "
            f"{m.get('total_trades',0):>7} "
            f"{str(boosts):>7}  {delta_str}"
        )

    lines += [
        "-" * 70,
        "",
        "VERDICT:",
    ]

    best = sorted_results[0] if sorted_results else None
    if best:
        if best['tier_label'] == 'baseline':
            lines.append("  Baseline wins — event data does NOT improve performance.")
            lines.append("  Consider higher Optuna trial count or different event weights.")
        else:
            improvement = best['score'] - baseline_score
            lines.append(f"  WINNER: {best['tier_label'].upper()} "
                        f"(+{improvement:.2f} score vs baseline)")
            lines.append(f"  Optimal config: {json.dumps(best.get('tier_config', {}), indent=4)}")

    lines.append("")
    lines.append("Full results saved to: data/event_backtest/results/comparison_report.json")
    lines.append("=" * 70)

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Event Backtest Comparison")
    parser.add_argument('--no-fetch',  action='store_true', help='Skip data fetching')
    parser.add_argument('--no-optuna', action='store_true', help='Skip Optuna optimization')
    parser.add_argument('--trials',    type=int, default=50,  help='Optuna trials per tier')
    parser.add_argument('--tiers',     nargs='+',
                        default=['baseline', 'tier_a', 'tier_b', 'tier_ab'],
                        help='Which tiers to run')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("XAUUSD EVENT BACKTEST COMPARISON SYSTEM")
    print(f"Period: {BACKTEST_START.date()} to {BACKTEST_END.date()}")
    print(f"Tiers:  {args.tiers}")
    print(f"Optuna: {'OFF' if args.no_optuna else f'{args.trials} trials per tier'}")
    print("=" * 70)

    # ── Step 1: Initialize DB and fetch event data ───────────────────────────
    initialize_database()

    if not args.no_fetch:
        print("\n[Step 1/3] Fetching event data into database...")
        fetch_and_store_all(BACKTEST_START, BACKTEST_END)
    else:
        print("\n[Step 1/3] Skipping fetch (--no-fetch)")

    # ── Step 2: Load bot config and connect to MT5 ───────────────────────────
    print("\n[Step 2/3] Loading config and connecting to MT5...")
    config = load_bot_config()
    mt5 = get_mt5_connector(config)
    print("  MT5 connected.")

    # ── Step 3: Run tier comparisons ─────────────────────────────────────────
    print(f"\n[Step 3/3] Running {len(args.tiers)} tier backtests...")
    all_results = []
    start_time = time.time()

    for tier_label in args.tiers:
        if tier_label not in TIER_CONFIGS:
            print(f"  Unknown tier: {tier_label}, skipping")
            continue

        # Run with default config
        result = run_single_tier(config, mt5, tier_label)
        if result:
            all_results.append(result)

        # Optuna optimization for non-baseline tiers
        if not args.no_optuna and tier_label != 'baseline':
            best_scorer = optimize_tier(config, mt5, tier_label, n_trials=args.trials)
            if best_scorer:
                print(f"\n  Running optimized {tier_label} with best params...")
                opt_result = run_single_tier(config, mt5, f"{tier_label}_best", best_scorer)
                if opt_result:
                    opt_result['tier_label'] = f"{tier_label}_best"
                    opt_result['tier_config'] = best_scorer
                    all_results.append(opt_result)

    # Disconnect MT5
    try:
        mt5.disconnect()
    except Exception:
        pass

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    # ── Generate and save report ──────────────────────────────────────────────
    report = generate_report(all_results)
    print(report)

    # Save JSON results
    report_path = RESULTS_DIR / "comparison_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            'generated_at':  datetime.now().isoformat(),
            'backtest_start': str(BACKTEST_START.date()),
            'backtest_end':   str(BACKTEST_END.date()),
            'results':        all_results,
            'report':         report,
        }, f, indent=2, default=str)

    print(f"\nReport saved: {report_path}")

    # Show DB best results
    print("\nTop results from database:")
    best = get_best_results()
    for r in best[:10]:
        print(f"  {r['tier_label']:<22} Score={r['score']:.2f} | "
              f"PF={r['profit_factor']:.2f} | WR={r['win_rate']:.1f}% | "
              f"Trades={r['total_trades']}")


if __name__ == '__main__':
    main()
