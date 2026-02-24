"""
Optuna hyperparameter optimization for Smart Trader backtest.

Tunes entry/exit parameters across 4 regime categories to maximize
a composite score (profit factor + net PnL).  Data is fetched once,
then reused across all trials.

Run:
  cd smart_trader && python optimize_backtest.py
  python optimize_backtest.py --trials 50 --months 6
  python optimize_backtest.py --trials 100 --months 3 --metric pf
"""
import os, sys, copy, json, warnings, argparse, time
warnings.filterwarnings("ignore", message="no explicit representation of timezones")
from pathlib import Path
from datetime import datetime, timezone

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "src"))

import optuna
from optuna.trial import Trial

# Reuse backtest infrastructure
from backtest_smart import (
    load_config, connect_mt5, fetch_historical,
    run_simulation, compute_results,
)


# ============================================================================
# Objective function
# ============================================================================

# Global data cache (fetched once)
_DATA_CACHE: dict = {}


def objective(trial: Trial, base_cfg: dict, args) -> float:
    """
    Single Optuna trial.  Samples hyperparameters, patches config,
    runs simulation, returns objective value.
    """

    # -- Global (non-regime) params ------------------------------------------
    pre_score_min = trial.suggest_float("pre_score_min", 0.15, 0.55, step=0.05)
    proximity = trial.suggest_float("zone_proximity_pts", 2.0, 15.0, step=1.0)

    # -- Per-regime params (4 categories) ------------------------------------
    regime_params = {}
    for cat in ["trending", "ranging", "breakout", "reversal"]:
        sl = trial.suggest_float(f"{cat}_sl_atr_mult", 1.0, 3.5, step=0.25)
        tp = trial.suggest_float(f"{cat}_tp_atr_mult", 2.0, 7.0, step=0.5)

        # Enforce TP > SL (minimum 1.5x ratio)
        if tp / sl < 1.5:
            tp = round(sl * 1.5, 1)
            tp = min(tp, 7.0)

        be = trial.suggest_float(f"{cat}_be_trigger_mult", 0.3, 1.2, step=0.1)
        lock = trial.suggest_float(f"{cat}_lock_trigger_mult", 0.8, 2.5, step=0.1)

        # lock should be > be
        if lock <= be:
            lock = round(be + 0.3, 1)
            lock = min(lock, 2.5)

        trail = trial.suggest_float(f"{cat}_trail_keep_pct", 0.30, 0.70, step=0.05)
        stale = trial.suggest_int(f"{cat}_stale_tighten_min", 30, 180, step=15)
        scratch = trial.suggest_int(f"{cat}_scratch_exit_min", 60, 360, step=30)

        # scratch should be > stale
        if scratch <= stale:
            scratch = stale + 30

        regime_params[cat] = {
            "sl_atr_mult": sl,
            "tp_atr_mult": tp,
            "min_confidence": 0.70,  # fixed (not relevant without Claude)
            "be_trigger_mult": be,
            "lock_trigger_mult": lock,
            "trail_keep_pct": trail,
            "stale_tighten_min": stale,
            "scratch_exit_min": scratch,
        }

    # -- Patch config --------------------------------------------------------
    cfg = copy.deepcopy(base_cfg)
    cfg["trading"]["zone_proximity_pts"] = proximity
    cfg["adaptive"]["regime_params"] = regime_params

    # -- Run simulation ------------------------------------------------------
    trades = run_simulation(
        df_m15=_DATA_CACHE["m15"],
        df_h1=_DATA_CACHE["h1"],
        df_h4=_DATA_CACHE["h4"],
        cfg=cfg,
        use_adaptive=True,
        spread=args.spread,
        balance=args.balance,
        pre_score_min=pre_score_min,
        verbose=False,
    )

    results = compute_results(trades, args.balance)
    s = results.get("summary", {})

    if not s or s.get("total", 0) < 10:
        return -999.0  # not enough trades

    pf = s.get("profit_factor", 0)
    wr = s.get("win_rate", 0)
    net_pts = s.get("net_pnl_pts", 0)
    max_dd = s.get("max_dd_pct", 100)
    total = s.get("total", 0)
    sharpe = s.get("sharpe", 0)

    # Store stats as user attrs for later analysis
    trial.set_user_attr("total_trades", total)
    trial.set_user_attr("win_rate", wr)
    trial.set_user_attr("profit_factor", pf)
    trial.set_user_attr("net_pnl_pts", net_pts)
    trial.set_user_attr("max_dd_pct", max_dd)
    trial.set_user_attr("sharpe", sharpe)
    trial.set_user_attr("final_equity", s.get("final_equity", 0))

    # -- Compute objective ---------------------------------------------------
    if args.metric == "pf":
        return pf
    elif args.metric == "net":
        return net_pts
    elif args.metric == "sharpe":
        return sharpe
    else:
        # Composite: weighted combination (default)
        # Reward: profit factor + positive net pnl + sharpe
        # Penalize: excessive drawdown, too few trades
        score = 0.0
        score += pf * 30.0                              # PF=1.5 -> 45 pts
        score += max(net_pts, -500) * 0.1                # Net +100pt -> 10 pts
        score += sharpe * 10.0                           # Sharpe=0.5 -> 5 pts
        score += min(wr, 60) * 0.5                       # WR=50% -> 25 pts
        score -= max(max_dd - 50, 0) * 0.5               # Penalize DD > 50%
        if total < 50:
            score -= (50 - total) * 0.5                  # Penalize too few trades
        return score


# ============================================================================
# Result display
# ============================================================================

def print_best_params(study: optuna.Study, base_cfg: dict, args):
    """Print best trial parameters in a readable format."""
    best = study.best_trial

    print(f"\n{'=' * 70}")
    print(f"  OPTUNA OPTIMIZATION RESULTS")
    print(f"  Trials: {len(study.trials)} | Metric: {args.metric}")
    print(f"  Best trial: #{best.number} | Score: {best.value:.2f}")
    print(f"{'=' * 70}")

    # Stats
    ua = best.user_attrs
    print(f"\n  Performance:")
    print(f"    Trades:        {ua.get('total_trades', '?')}")
    print(f"    Win rate:      {ua.get('win_rate', '?'):.1f}%")
    print(f"    Profit factor: {ua.get('profit_factor', '?'):.2f}")
    print(f"    Net PnL:       {ua.get('net_pnl_pts', '?'):+.1f} pts")
    print(f"    Max DD:        {ua.get('max_dd_pct', '?'):.1f}%")
    print(f"    Sharpe:        {ua.get('sharpe', '?'):.2f}")
    print(f"    Final equity:  ${ua.get('final_equity', '?'):.2f}")

    # Global params
    bp = best.params
    print(f"\n  Global Params:")
    print(f"    pre_score_min:      {bp['pre_score_min']:.2f}")
    print(f"    zone_proximity_pts: {bp['zone_proximity_pts']:.1f}")

    # Per-regime params
    for cat in ["trending", "ranging", "breakout", "reversal"]:
        print(f"\n  {cat.upper()}:")
        print(f"    sl_atr_mult:      {bp[f'{cat}_sl_atr_mult']:.2f}")
        print(f"    tp_atr_mult:      {bp[f'{cat}_tp_atr_mult']:.1f}")
        print(f"    be_trigger_mult:  {bp[f'{cat}_be_trigger_mult']:.1f}")
        print(f"    lock_trigger_mult:{bp[f'{cat}_lock_trigger_mult']:.1f}")
        print(f"    trail_keep_pct:   {bp[f'{cat}_trail_keep_pct']:.2f}")
        print(f"    stale_tighten_min:{bp[f'{cat}_stale_tighten_min']}")
        print(f"    scratch_exit_min: {bp[f'{cat}_scratch_exit_min']}")

    # Comparison with baseline
    print(f"\n{'-' * 70}")
    print(f"  BASELINE vs OPTIMIZED comparison:")
    print(f"{'-' * 70}")

    # Run baseline
    print("  Running baseline simulation...")
    trades_base = run_simulation(
        _DATA_CACHE["m15"], _DATA_CACHE["h1"], _DATA_CACHE["h4"],
        base_cfg, use_adaptive=True, spread=args.spread,
        balance=args.balance, pre_score_min=0.35, verbose=False,
    )
    rb = compute_results(trades_base, args.balance).get("summary", {})

    metrics = [
        ("Total trades", "total_trades", "total", "d"),
        ("Win rate %", "win_rate", "win_rate", ".1f"),
        ("Profit factor", "profit_factor", "profit_factor", ".2f"),
        ("Net PnL pts", "net_pnl_pts", "net_pnl_pts", "+.1f"),
        ("Max DD %", "max_dd_pct", "max_dd_pct", ".1f"),
        ("Sharpe", "sharpe", "sharpe", ".2f"),
    ]

    print(f"  {'Metric':<16s} | {'Baseline':>10s} | {'Optimized':>10s} | {'Delta':>10s}")
    print(f"  {'-' * 16}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")
    for label, ua_key, rb_key, fmt in metrics:
        vb = rb.get(rb_key, 0)
        vo = ua.get(ua_key, 0)
        delta = vo - vb
        vb_s = f"{vb:{fmt}}"
        vo_s = f"{vo:{fmt}}"
        d_s = f"{delta:{fmt}}"
        print(f"  {label:<16s} | {vb_s:>10s} | {vo_s:>10s} | {d_s:>10s}")

    print()


def save_results(study: optuna.Study, args):
    """Save best params and top trials to JSON."""
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)

    best = study.best_trial

    # Build optimized config snippet
    bp = best.params
    optimized_cfg = {
        "pre_score_min": bp["pre_score_min"],
        "zone_proximity_pts": bp["zone_proximity_pts"],
        "regime_params": {},
    }
    for cat in ["trending", "ranging", "breakout", "reversal"]:
        optimized_cfg["regime_params"][cat] = {
            "sl_atr_mult": bp[f"{cat}_sl_atr_mult"],
            "tp_atr_mult": bp[f"{cat}_tp_atr_mult"],
            "min_confidence": 0.70,
            "be_trigger_mult": bp[f"{cat}_be_trigger_mult"],
            "lock_trigger_mult": bp[f"{cat}_lock_trigger_mult"],
            "trail_keep_pct": bp[f"{cat}_trail_keep_pct"],
            "stale_tighten_min": bp[f"{cat}_stale_tighten_min"],
            "scratch_exit_min": bp[f"{cat}_scratch_exit_min"],
        }

    # Top 10 trials
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -9999, reverse=True)[:10]
    top_list = []
    for t in top_trials:
        top_list.append({
            "number": t.number,
            "score": round(t.value, 2) if t.value is not None else None,
            "attrs": t.user_attrs,
            "params": {k: round(v, 4) if isinstance(v, float) else v for k, v in t.params.items()},
        })

    output = {
        "best_trial": best.number,
        "best_score": round(best.value, 2),
        "best_attrs": best.user_attrs,
        "optimized_config": optimized_cfg,
        "top_10_trials": top_list,
        "total_trials": len(study.trials),
        "metric": args.metric,
        "months": args.months,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    path = out_dir / "optuna_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {path}")

    # Also save as YAML snippet for easy config.yaml patching
    yaml_path = out_dir / "optimized_params.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# Optuna-optimized parameters\n")
        f.write(f"# Trial #{best.number} | Score: {best.value:.2f}\n")
        f.write(f"# PF={best.user_attrs.get('profit_factor', '?'):.2f} ")
        f.write(f"WR={best.user_attrs.get('win_rate', '?'):.1f}% ")
        f.write(f"Net={best.user_attrs.get('net_pnl_pts', '?'):+.1f}pt\n\n")
        f.write("trading:\n")
        f.write(f"  zone_proximity_pts: {bp['zone_proximity_pts']:.1f}\n\n")
        f.write("adaptive:\n")
        f.write("  regime_params:\n")
        for cat in ["trending", "ranging", "breakout", "reversal"]:
            f.write(f"    {cat}:\n")
            f.write(f"      sl_atr_mult: {bp[f'{cat}_sl_atr_mult']:.2f}\n")
            f.write(f"      tp_atr_mult: {bp[f'{cat}_tp_atr_mult']:.1f}\n")
            f.write(f"      min_confidence: 0.70\n")
            f.write(f"      be_trigger_mult: {bp[f'{cat}_be_trigger_mult']:.1f}\n")
            f.write(f"      lock_trigger_mult: {bp[f'{cat}_lock_trigger_mult']:.1f}\n")
            f.write(f"      trail_keep_pct: {bp[f'{cat}_trail_keep_pct']:.2f}\n")
            f.write(f"      stale_tighten_min: {bp[f'{cat}_stale_tighten_min']}\n")
            f.write(f"      scratch_exit_min: {bp[f'{cat}_scratch_exit_min']}\n")
    print(f"  YAML snippet saved: {yaml_path}")


# ============================================================================
# Trial progress callback
# ============================================================================

class TrialCallback:
    """Print compact progress after each trial."""

    def __init__(self):
        self.start_time = time.time()
        self.best_score = -9999

    def __call__(self, study: optuna.Study, trial):
        if trial.value is None:
            return
        elapsed = time.time() - self.start_time
        is_best = trial.value > self.best_score
        if is_best:
            self.best_score = trial.value

        ua = trial.user_attrs
        marker = " ** BEST" if is_best else ""
        print(
            f"  Trial #{trial.number:>3d} | Score: {trial.value:>7.1f} | "
            f"PF={ua.get('profit_factor', 0):>5.2f} WR={ua.get('win_rate', 0):>4.1f}% "
            f"Net={ua.get('net_pnl_pts', 0):>+7.1f}pt "
            f"DD={ua.get('max_dd_pct', 0):>5.1f}% "
            f"T={ua.get('total_trades', 0):>3d} | "
            f"{elapsed:.0f}s{marker}"
        )


# ============================================================================
# Main
# ============================================================================

def main():
    import MetaTrader5 as mt5

    parser = argparse.ArgumentParser(
        description="Optuna parameter optimization for Smart Trader backtest"
    )
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials (default: 50)")
    parser.add_argument("--months", type=int, default=6, help="Backtest period in months (default: 6)")
    parser.add_argument("--balance", type=float, default=100.0, help="Starting balance USD (default: 100)")
    parser.add_argument("--spread", type=float, default=3.0, help="Fixed spread in points (default: 3.0)")
    parser.add_argument("--metric", choices=["composite", "pf", "net", "sharpe"], default="composite",
                        help="Optimization metric (default: composite)")
    parser.add_argument("--symbol", default=None, help="Override symbol")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Smart Trader -- Optuna Parameter Optimization")
    print(f"  Trials: {args.trials} | Months: {args.months} | Metric: {args.metric}")
    print(f"  Balance: ${args.balance} | Spread: {args.spread}pt | Seed: {args.seed}")
    print("=" * 70)

    # Load config
    cfg = load_config()
    symbol = args.symbol or cfg.get("mt5", {}).get("symbol", "XAUUSD")

    # Connect MT5 and fetch data once
    if not connect_mt5(cfg):
        sys.exit(1)

    try:
        df_m15, df_h1, df_h4 = fetch_historical(symbol, args.months)
        if df_m15.empty or df_h1.empty:
            print("ERROR: Failed to fetch required data")
            sys.exit(1)

        print(f"\n  Period: {df_m15['time'].iloc[0].date()} -> {df_m15['time'].iloc[-1].date()}")
        print(f"  Bars: M15={len(df_m15)}, H1={len(df_h1)}, H4={len(df_h4)}")

        # Cache data globally
        _DATA_CACHE["m15"] = df_m15
        _DATA_CACHE["h1"] = df_h1
        _DATA_CACHE["h4"] = df_h4

        # Suppress backtest progress output during optimization
        # (run_simulation prints progress on stderr via \r)
        import io
        original_stdout = sys.stdout

        # Create Optuna study
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(
            study_name="smart_trader_optuna",
            direction="maximize",
            sampler=sampler,
        )

        # Suppress optuna default logging (we have our own callback)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(f"\n  Starting {args.trials} trials...")
        print(f"  {'=' * 65}")

        callback = TrialCallback()

        # Redirect stdout during simulation to suppress progress bars
        class SimQuiet:
            """Context manager to suppress run_simulation print output."""
            def __enter__(self):
                self._orig = sys.stdout
                sys.stdout = open(os.devnull, "w")
                return self
            def __exit__(self, *_):
                sys.stdout.close()
                sys.stdout = self._orig

        def wrapped_objective(trial):
            with SimQuiet():
                return objective(trial, cfg, args)

        study.optimize(
            wrapped_objective,
            n_trials=args.trials,
            callbacks=[callback],
            show_progress_bar=False,
        )

        print(f"  {'=' * 65}")
        elapsed = time.time() - callback.start_time
        print(f"  Optimization complete in {elapsed:.0f}s ({elapsed / args.trials:.1f}s/trial)")

        # Print and save results
        print_best_params(study, cfg, args)
        save_results(study, args)

    finally:
        mt5.shutdown()
        print("MT5 disconnected.")


if __name__ == "__main__":
    main()
