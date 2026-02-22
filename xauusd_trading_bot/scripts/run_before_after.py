"""
V3 Integration Before/After Comparison
Runs backtest with current live config (V3 applied) and compares to V2 baseline.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import json
from datetime import datetime, timedelta
from src.core.mt5_connector import MT5Connector
from src.backtesting.backtest_engine import BacktestEngine
from src.bot_logger import setup_logger, get_logger


def load_live_config():
    """Load the CURRENT live config from individual config files (now V3)."""
    from src.utils.config_loader import ConfigLoader
    config_loader = ConfigLoader()
    settings = config_loader.load("settings")
    mt5_config = config_loader.load("mt5_config")
    trading_rules = config_loader.load("trading_rules")
    risk_config = config_loader.load("risk_config")
    session_config = config_loader.load("session_config")

    config = {
        **settings,
        "mt5": mt5_config,
        "strategy": trading_rules.get("strategy", {}),
        "indicators": trading_rules.get("indicators", {}),
        "smc_indicators": trading_rules.get("smc_indicators", {}),
        "technical_indicators": trading_rules.get("technical_indicators", {}),
        "confluence_weights": trading_rules.get("confluence_weights", {}),
        "market_conditions": trading_rules.get("market_conditions", {}),
        "mtf_analysis": trading_rules.get("mtf_analysis", {}),
        "signal_validation": trading_rules.get("signal_validation", {}),
        "risk": risk_config,
        "session": session_config,
    }
    return config


def load_v2_baseline():
    """Load V2 baseline config (same as live but with V3 features OFF)."""
    config = load_live_config()

    # Override V3 features to simulate V2
    config["use_adaptive_scorer"] = False

    # Restore V2 exit stages
    config["risk"]["exit_stages"] = {
        "be_trigger_rr": 1.0,
        "partial_close_rr": 1.5,
        "trail_activation_rr": 2.0,
    }

    # Restore V2 TP multiplier
    config["risk"]["take_profit"]["atr_multiplier"] = 5.0

    # Restore V2 OB sensitivity (disabled)
    config.setdefault("smc_indicators", {}).setdefault("order_blocks", {})["strong_move_percent"] = 1.0

    # Restore V2 session weights
    sessions = config.get("session", {}).get("sessions", {})
    if "asian" in sessions:
        sessions["asian"]["weight"] = 0.70
    if "london" in sessions:
        sessions["london"]["weight"] = 1.10
    if "new_york" in sessions:
        sessions["new_york"]["weight"] = 1.10
    if "overlap" in sessions:
        sessions["overlap"]["weight"] = 1.25

    return config


def print_results(label, metrics):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Profit Factor:    {metrics.get('profit_factor', 0):.2f}")
    print(f"  Win Rate:         {metrics.get('win_rate', 0):.1f}%")
    print(f"  Max Drawdown:     {metrics.get('max_drawdown_percent', metrics.get('max_drawdown_pct', 0)):.2f}%")
    print(f"  Total Return:     {metrics.get('total_return_percent', metrics.get('total_return_pct', 0)):.2f}%")
    print(f"  Total Trades:     {metrics.get('total_trades', 0)}")
    print(f"  Avg RR:           {metrics.get('avg_rr_ratio', metrics.get('avg_rr', 0)):.2f}")
    print(f"  Max Consec Loss:  {metrics.get('max_consecutive_losses', 0)}")
    print(f"  Max Consec Win:   {metrics.get('max_consecutive_wins', 0)}")
    print(f"  Sharpe:           {metrics.get('sharpe_ratio', 0):.2f}")
    if 'final_balance' in metrics:
        print(f"  Final Balance:    ${metrics['final_balance']:.2f}")
    print(f"{'='*60}")


def main():
    setup_logger()
    logger = get_logger()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print(f"\n{'#'*60}")
    print(f"  V3 INTEGRATION: BEFORE / AFTER COMPARISON")
    print(f"{'#'*60}")
    print(f"\nPeriod: {start_date.date()} to {end_date.date()}")
    print(f"Initial Balance: $10,000")

    # Connect MT5
    live_config = load_live_config()
    mt5_cfg = live_config.get("mt5", {})
    mt5 = MT5Connector(mt5_cfg)
    if not mt5.connect():
        print("ERROR: Failed to connect to MT5")
        return

    try:
        # ---- BEFORE (V2 Baseline) ----
        print("\n>>> Running BEFORE (V2 Baseline) backtest...")
        v2_config = load_v2_baseline()
        engine_v2 = BacktestEngine(v2_config)
        results_v2 = engine_v2.run_backtest(
            mt5=mt5, symbol="XAUUSDm",
            start_date=start_date, end_date=end_date,
            initial_balance=10000.0, timeframe="M15", use_cache=True
        )

        if results_v2.get("success"):
            m_v2 = results_v2.get("metrics", results_v2.get("performance", {}))
            print_results("BEFORE: V2 Baseline (Fixed Weights, BE@1.0R, PC@1.5R)", m_v2)
        else:
            print("V2 backtest failed!")
            m_v2 = {}

        # ---- AFTER (V3 Live Config) ----
        print("\n>>> Running AFTER (V3 Integrated Live Config) backtest...")
        engine_v3 = BacktestEngine(live_config)
        results_v3 = engine_v3.run_backtest(
            mt5=mt5, symbol="XAUUSDm",
            start_date=start_date, end_date=end_date,
            initial_balance=10000.0, timeframe="M15", use_cache=True
        )

        if results_v3.get("success"):
            m_v3 = results_v3.get("metrics", results_v3.get("performance", {}))
            print_results("AFTER: V3 Integrated (Adaptive Scorer, Regime Detection)", m_v3)
        else:
            print("V3 backtest failed!")
            m_v3 = {}

        # ---- COMPARISON TABLE ----
        if m_v3 and m_v2:
            print(f"\n{'#'*60}")
            print(f"  BEFORE vs AFTER COMPARISON")
            print(f"{'#'*60}")
            print(f"  {'Metric':<20s} {'BEFORE':>10s} {'AFTER':>10s} {'CHANGE':>10s}")
            print(f"  {'-'*50}")

            metrics_map = [
                ("Profit Factor", "profit_factor", ".2f"),
                ("Win Rate %", "win_rate", ".1f"),
                ("Max DD %", "max_drawdown_percent", ".2f", "max_drawdown_pct"),
                ("Return %", "total_return_percent", ".2f", "total_return_pct"),
                ("Trades", "total_trades", ".0f"),
                ("Avg RR", "avg_rr_ratio", ".2f", "avg_rr"),
                ("Sharpe", "sharpe_ratio", ".2f"),
                ("Max Consec Loss", "max_consecutive_losses", ".0f"),
                ("Final Balance $", "final_balance", ".2f"),
            ]

            for item in metrics_map:
                label = item[0]
                key = item[1]
                fmt = item[2]
                alt_key = item[3] if len(item) > 3 else None

                v2_val = m_v2.get(key, m_v2.get(alt_key, 0) if alt_key else 0)
                v3_val = m_v3.get(key, m_v3.get(alt_key, 0) if alt_key else 0)
                diff = v3_val - v2_val

                # Determine if improvement (lower DD is better, higher everything else)
                if "DD" in label or "Consec Loss" in label:
                    arrow = "+" if diff < 0 else "-" if diff > 0 else "="
                else:
                    arrow = "+" if diff > 0 else "-" if diff < 0 else "="

                v2_str = f"{v2_val:{fmt}}"
                v3_str = f"{v3_val:{fmt}}"
                diff_str = f"{arrow}{abs(diff):{fmt}}"

                print(f"  {label:<20s} {v2_str:>10s} {v3_str:>10s} {diff_str:>10s}")

            print(f"  {'-'*50}")

            # Summary
            pf_improve = ((m_v3.get("profit_factor", 0) / max(m_v2.get("profit_factor", 1), 0.01)) - 1) * 100
            ret_improve = m_v3.get("total_return_percent", m_v3.get("total_return_pct", 0)) - m_v2.get("total_return_percent", m_v2.get("total_return_pct", 0))
            dd_improve = m_v2.get("max_drawdown_percent", m_v2.get("max_drawdown_pct", 0)) - m_v3.get("max_drawdown_percent", m_v3.get("max_drawdown_pct", 0))

            print(f"\n  SUMMARY:")
            print(f"  - PF improvement:     {pf_improve:+.1f}%")
            print(f"  - Return improvement: {ret_improve:+.2f}%")
            print(f"  - DD reduction:       {dd_improve:+.2f}%")
            print(f"\n  V3 CHANGES APPLIED:")
            print(f"  - Adaptive Confluence Scorer (regime-conditional weights)")
            print(f"  - Market Regime Detection (8 regimes)")
            print(f"  - Structure-Based SL/TP (hybrid ATR + swing)")
            print(f"  - Micro Account Safety ($50-$100)")
            print(f"  - Exit stages: BE@0.77R, PC@2.73R (was BE@1.0R, PC@1.5R)")
            print(f"  - TP multiplier: 6.02 ATR (was 5.0)")
            print(f"  - OB detection: strong_move=0.45% (was 1.0% disabled)")
            print(f"{'#'*60}")

        # Save comparison
        comparison = {
            "before_v2": m_v2,
            "after_v3": m_v3,
            "v3_changes": {
                "adaptive_scorer": True,
                "regime_detection": True,
                "structure_sltp": True,
                "micro_account": True,
                "be_trigger_rr": 0.77,
                "partial_close_rr": 2.73,
                "tp_atr_multiplier": 6.02,
                "ob_strong_move_pct": 0.45,
            },
            "period": f"{start_date.date()} to {end_date.date()}",
            "timestamp": datetime.now().isoformat(),
        }
        os.makedirs("data/optimization_v3", exist_ok=True)
        with open("data/optimization_v3/before_after_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"\nComparison saved to data/optimization_v3/before_after_comparison.json")

    finally:
        mt5.disconnect()


if __name__ == "__main__":
    main()
