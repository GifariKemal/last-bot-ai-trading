"""
V3 Comparison Backtest
Run full 12-month backtest with V3 optimized config and compare to V2 baseline.
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


def load_v3_config():
    """Load the V3 optimized config."""
    with open("data/optimization_v3/optimized_config_v3.yaml") as f:
        return yaml.safe_load(f)


def load_v2_config():
    """Load the current V2 config from individual files."""
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
    # V2 doesn't use adaptive scorer
    config["use_adaptive_scorer"] = False
    return config


def print_results(label, metrics):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Profit Factor:    {metrics.get('profit_factor', 0):.2f}")
    print(f"  Win Rate:         {metrics.get('win_rate', 0):.1f}%")
    print(f"  Max Drawdown:     {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Total Return:     {metrics.get('total_return_pct', 0):.2f}%")
    print(f"  Total Trades:     {metrics.get('total_trades', 0)}")
    print(f"  Avg RR:           {metrics.get('avg_rr', 0):.2f}")
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

    print(f"\nPeriod: {start_date.date()} to {end_date.date()}")
    print(f"Initial Balance: $10,000")

    # Connect MT5
    v3_config = load_v3_config()
    mt5_cfg = v3_config.get("connection", v3_config.get("mt5", {}))
    mt5 = MT5Connector(mt5_cfg)
    if not mt5.connect():
        print("ERROR: Failed to connect to MT5")
        return

    try:
        # ---- V3 Backtest ----
        print("\n>>> Running V3 Optimized Backtest...")
        engine_v3 = BacktestEngine(v3_config)
        results_v3 = engine_v3.run_backtest(
            mt5=mt5, symbol="XAUUSDm",
            start_date=start_date, end_date=end_date,
            initial_balance=10000.0, timeframe="M15", use_cache=True
        )

        if results_v3.get("success"):
            m_v3 = results_v3.get("metrics", results_v3.get("performance", {}))
            print_results("V3 OPTIMIZED (Adaptive Scorer + Regime Detection)", m_v3)
        else:
            print("V3 backtest failed!")
            m_v3 = {}

        # ---- V2 Backtest ----
        print("\n>>> Running V2 Baseline Backtest...")
        v2_config = load_v2_config()
        engine_v2 = BacktestEngine(v2_config)
        results_v2 = engine_v2.run_backtest(
            mt5=mt5, symbol="XAUUSDm",
            start_date=start_date, end_date=end_date,
            initial_balance=10000.0, timeframe="M15", use_cache=True
        )

        if results_v2.get("success"):
            m_v2 = results_v2.get("metrics", results_v2.get("performance", {}))
            print_results("V2 BASELINE (Fixed Weights)", m_v2)
        else:
            print("V2 backtest failed!")
            m_v2 = {}

        # ---- Comparison ----
        if m_v3 and m_v2:
            print(f"\n{'='*60}")
            print(f"  COMPARISON: V3 vs V2")
            print(f"{'='*60}")
            for key, label in [
                ("profit_factor", "Profit Factor"),
                ("win_rate", "Win Rate %"),
                ("max_drawdown_pct", "Max DD %"),
                ("total_return_pct", "Return %"),
                ("total_trades", "Trades"),
                ("avg_rr", "Avg RR"),
            ]:
                v3_val = m_v3.get(key, 0)
                v2_val = m_v2.get(key, 0)
                diff = v3_val - v2_val
                arrow = "^" if diff > 0 else "v" if diff < 0 else "="
                print(f"  {label:18s} V3={v3_val:>8.2f}  V2={v2_val:>8.2f}  {arrow} {diff:+.2f}")
            print(f"{'='*60}")

        # Save comparison
        comparison = {
            "v3_metrics": m_v3,
            "v2_metrics": m_v2,
            "period": f"{start_date.date()} to {end_date.date()}",
            "timestamp": datetime.now().isoformat()
        }
        os.makedirs("data/optimization_v3", exist_ok=True)
        with open("data/optimization_v3/v3_vs_v2_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"\nComparison saved to data/optimization_v3/v3_vs_v2_comparison.json")

    finally:
        mt5.disconnect()


if __name__ == "__main__":
    main()
