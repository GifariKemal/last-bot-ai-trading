"""Debug script to trace exactly which entry conditions are failing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from src.core.mt5_connector import MT5Connector
from src.utils.config_loader import ConfigLoader
from src.indicators.technical import TechnicalIndicators
from src.indicators.smc_indicators import SMCIndicators
from src.core.data_manager import DataManager
from src.core.constants import TrendDirection
from src.backtesting.historical_data import HistoricalDataManager
from src.analysis import MarketAnalyzer, VolatilityAnalyzer, TrendAnalyzer, MTFAnalyzer, ConfluenceScorer
from src.strategy.smc_strategy import SMCStrategy
from src.sessions import SessionManager

# Load config
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
    "signal_validation": trading_rules.get("signal_validation", {}),
    "risk": risk_config,
    "session": session_config,
}

# Connect to MT5
mt5 = MT5Connector(mt5_config)
mt5.connect()

# Fetch data
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
data_mgr = HistoricalDataManager()
df = data_mgr.prepare_backtest_data(mt5, "XAUUSDm", "M15", start_date, end_date, use_cache=True)
mt5.disconnect()

# Process data
price_mgr = DataManager()
df = price_mgr.add_basic_features(df)
df = price_mgr.add_price_changes(df)
tech = TechnicalIndicators(config.get("indicators", {}))
df = tech.calculate_all(df)
smc_ind = SMCIndicators(config.get("indicators", {}))
df = smc_ind.calculate_all(df)

# Initialize analyzers
market_analyzer = MarketAnalyzer()
volatility_analyzer = VolatilityAnalyzer()
trend_analyzer = TrendAnalyzer()
mtf_analyzer = MTFAnalyzer()
confluence_scorer = ConfluenceScorer(config)
session_manager = SessionManager(config.get("session", {}))
strategy = SMCStrategy(config)

# Find bars with BOS/CHoCH
total_bars = len(df)
test_bars = []
for i in range(100, total_bars):
    bar = df[i]
    has_structure = (
        bar["is_bullish_bos"][0] or bar["is_bearish_bos"][0] or
        bar["is_bullish_choch"][0] or bar["is_bearish_choch"][0]
    )
    if has_structure:
        test_bars.append(i)

print(f"Found {len(test_bars)} bars with BOS/CHoCH out of {total_bars}")
print()

# Track failure reasons
failure_counts = {}

for idx, i in enumerate(test_bars):
    current_bar = df[i]
    current_price = current_bar["close"][0]
    current_time = current_bar["time"][0]
    df_slice = df[:i+1]

    # Session check
    session_check = session_manager.is_trading_allowed(current_time)
    session_ok = session_check.get("allowed", False)

    # SMC signals
    bullish_smc = smc_ind.get_bullish_signals(df_slice, current_price)
    bearish_smc = smc_ind.get_bearish_signals(df_slice, current_price)

    # Technical
    ema_20 = df_slice["ema_20"][-1] if "ema_20" in df_slice.columns else current_price
    ema_50 = df_slice["ema_50"][-1] if "ema_50" in df_slice.columns else current_price
    technical_indicators = {
        "atr": df_slice["atr_14"][-1] if "atr_14" in df_slice.columns else 15.0,
        "rsi": df_slice["rsi_14"][-1] if "rsi_14" in df_slice.columns else 50.0,
        "ema_20": ema_20,
        "ema": {20: ema_20, 50: ema_50},
        "macd": {"histogram": df_slice["macd_histogram"][-1] if "macd_histogram" in df_slice.columns else None},
    }

    # Market analysis
    market_analysis = market_analyzer.analyze(df_slice)
    volatility_analysis = volatility_analyzer.analyze(df_slice)
    market_analysis["volatility"] = volatility_analysis

    # MTF
    mtf_analysis = mtf_analyzer.analyze({"M15": df_slice})

    # Confluence scores
    bullish_conf = confluence_scorer.calculate_score(
        TrendDirection.BULLISH, bullish_smc, technical_indicators, market_analysis, mtf_analysis
    )
    bearish_conf = confluence_scorer.calculate_score(
        TrendDirection.BEARISH, bearish_smc, technical_indicators, market_analysis, mtf_analysis
    )

    # Strategy decision
    confluence_scores = {"bullish": bullish_conf, "bearish": bearish_conf}
    mock_account = {"balance": 10000, "equity": 10000}
    mock_market_data = {"bid": current_price, "ask": current_price, "spread": 0.02}

    decision = strategy.analyze_and_signal(
        current_price, {"bullish": bullish_smc, "bearish": bearish_smc},
        technical_indicators, market_analysis, mtf_analysis, confluence_scores,
        [], mock_account, mock_market_data
    )

    has_entry = decision.get("has_entry", False)

    # Print first 5 detailed
    if idx < 5:
        print(f"=== Bar {i} ({current_time}) price={current_price:.2f} ===")
        print(f"  Session: {'OK' if session_ok else 'BLOCKED'}")
        cond = market_analysis["condition"]
        print(f"  Market: condition={cond.value if hasattr(cond, 'value') else cond}, favorable={market_analysis['is_favorable']}")
        print(f"  MTF: aligned={mtf_analysis['is_aligned']}, dominant={mtf_analysis['dominant_trend'].value}")
        print(f"  Bull SMC: fvg={bullish_smc['fvg']['in_zone']}, ob={bullish_smc['order_block']['at_zone']}, bos={bullish_smc['structure']['bos']}, choch={bullish_smc['structure']['choch']}")
        print(f"  Bear SMC: fvg={bearish_smc['fvg']['in_zone']}, ob={bearish_smc['order_block']['at_zone']}, bos={bearish_smc['structure']['bos']}, choch={bearish_smc['structure']['choch']}")
        print(f"  Bull conf: score={bullish_conf['score']:.3f}, passing={bullish_conf['passing']}, breakdown={bullish_conf.get('breakdown', {})}")
        print(f"  Bear conf: score={bearish_conf['score']:.3f}, passing={bearish_conf['passing']}, breakdown={bearish_conf.get('breakdown', {})}")
        print(f"  Decision: has_entry={has_entry}")
        if not has_entry:
            print(f"  Reason: {decision.get('reason', 'unknown')}")
        print()

    # Track individual condition failures for both directions
    for direction, smc_sig, conf in [("BULL", bullish_smc, bullish_conf), ("BEAR", bearish_smc, bearish_conf)]:
        reasons = []
        if not conf.get("passing", False):
            reasons.append(f"confluence_low({conf['score']:.2f})")
        fvg_or_ob = smc_sig.get("fvg", {}).get("in_zone", False) or smc_sig.get("order_block", {}).get("at_zone", False)
        if not fvg_or_ob:
            reasons.append("no_fvg_or_ob")
        structure = smc_sig.get("structure", {}).get("bos", False) or smc_sig.get("structure", {}).get("choch", False)
        if not structure:
            reasons.append("no_structure")
        if not market_analysis.get("is_favorable", False):
            reasons.append("market_unfavorable")
        if not mtf_analysis.get("is_aligned", False):
            reasons.append("mtf_not_aligned")

        for r in reasons:
            failure_counts[r] = failure_counts.get(r, 0) + 1

total_checks = len(test_bars) * 2  # bull + bear for each bar
print(f"\n=== FAILURE REASON BREAKDOWN ({total_checks} direction-checks across {len(test_bars)} bars) ===")
for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
    pct = count / total_checks * 100
    print(f"  {reason}: {count}/{total_checks} ({pct:.1f}%)")
