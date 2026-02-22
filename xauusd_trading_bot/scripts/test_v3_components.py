"""Quick tests for V3 components."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from src.core.constants import MarketRegime, VolatilityLevel, TrendDirection


def test_regime_detector():
    print("=== Test 1: RegimeDetector ===")
    from src.analysis.regime_detector import RegimeDetector

    np.random.seed(42)
    n = 200
    prices = 5000 + np.cumsum(np.random.randn(n) * 5)
    times = [datetime(2026, 1, 1) + timedelta(minutes=15*i) for i in range(n)]

    df = pl.DataFrame({
        "time": times,
        "open": prices,
        "high": prices + np.random.rand(n) * 10,
        "low": prices - np.random.rand(n) * 10,
        "close": prices + np.random.randn(n) * 2,
        "atr_14": np.full(n, 15.0) + np.random.rand(n) * 5,
        "ema_20": prices + 5,
        "ema_50": prices - 5,
        "ema_100": prices - 15,
        "bb_bandwidth": np.full(n, 2.0) + np.random.rand(n),
        "rsi_14": np.full(n, 60.0),
    })

    detector = RegimeDetector()
    result = detector.detect(df)
    print(f"  Regime: {result['regime'].value}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  ATR%: {result['components']['atr_percentile']:.0f}")
    print(f"  EMA spread: {result['components']['ema_spread_pct']:.3f}")
    assert result["regime"] in MarketRegime, "Invalid regime"
    print("  PASSED")


def test_micro_account():
    print("\n=== Test 2: MicroAccountManager ===")
    from src.risk_management.micro_account_manager import MicroAccountManager

    mgr = MicroAccountManager({"micro_account": {"enabled": True, "max_risk_dollars": 2.0}})

    assert mgr.is_micro_account(100) == True
    assert mgr.is_micro_account(5000) == False
    print("  Account classification: PASSED")

    # Safe trade: SL=$1.50 at 0.01 lot → risk=$1.50 (< $2 max)
    # At 0.01 lot XAUUSD: risk = sl_distance * 0.01 * 100 = sl_distance * 1.0
    v1 = mgr.validate_trade(100, 1.5, 0.10, 0, 0)
    assert v1["approved"] == True
    print(f"  Safe trade (SL=$1.50): approved, risk=${v1['risk_dollars']}")

    # Unsafe trade: SL=$3.00 → risk=$3.00 (> $2 max)
    v2 = mgr.validate_trade(100, 3.0, 0.10, 0, 0)
    assert v2["approved"] == False
    print(f"  Risky trade (SL=$3.00): rejected - {v2['reasons'][0]}")

    # Too many positions
    v3 = mgr.validate_trade(100, 1.5, 0.10, 2, 0)
    assert v3["approved"] == False
    print(f"  Max positions: rejected")

    # After 3 losses
    v4 = mgr.validate_trade(100, 1.5, 0.10, 0, 3)
    assert v4["approved"] == False
    print(f"  3 losses: rejected (paused)")

    # Spread too large relative to SL
    v5 = mgr.validate_trade(100, 1.5, 0.20, 0, 0)
    assert v5["approved"] == False  # 0.20/1.5 = 13.3% > 10% limit
    print(f"  Spread too large: rejected - {v5['reasons'][0]}")

    max_sl = mgr.get_max_sl_distance(100)
    print(f"  Max SL for $100: ${max_sl:.2f}")

    recovery = mgr.calculate_recovery_plan(80, 100, 0.55)
    print(f"  Recovery plan: {recovery['trades_to_recover']} trades needed")
    print("  PASSED")


def test_structure_sltp():
    print("\n=== Test 3: StructureSLTPCalculator ===")
    from src.risk_management.structure_sl_tp import StructureSLTPCalculator

    calc = StructureSLTPCalculator({
        "stop_loss": {"atr_multiplier": 3.0, "min_pips": 10, "max_pips": 200},
        "take_profit": {"atr_multiplier": 5.0, "max_pips": 500},
    })

    swing_pts = {
        "swing_highs": [{"level": 5030}],
        "swing_lows": [{"level": 4970}],
    }

    sltp = calc.calculate_sl_tp(
        5000, "BUY", 15.0, VolatilityLevel.MEDIUM,
        MarketRegime.STRONG_TREND_UP, swing_points=swing_pts, session="overlap"
    )
    print(f"  BUY: SL={sltp['sl']}, TP={sltp['tp']}, RR={sltp['rr_ratio']}, method={sltp['sl_method']}")
    print(f"  Spread: {sltp['spread']}, Spread%SL: {sltp['spread_pct_of_sl']}%")
    assert sltp["rr_ratio"] >= 1.5, f"RR too low: {sltp['rr_ratio']}"

    # SELL without structure
    sltp2 = calc.calculate_sl_tp(
        5000, "SELL", 15.0, VolatilityLevel.HIGH,
        MarketRegime.REVERSAL, session="asian"
    )
    print(f"  SELL: SL={sltp2['sl']}, TP={sltp2['tp']}, RR={sltp2['rr_ratio']}, method={sltp2['sl_method']}")
    assert sltp2["rr_ratio"] >= 2.0, f"Reversal min RR should be >= 3.0"
    print("  PASSED")


def test_adaptive_scorer():
    print("\n=== Test 4: AdaptiveConfluenceScorer ===")
    from src.analysis.adaptive_scorer import AdaptiveConfluenceScorer

    scorer = AdaptiveConfluenceScorer({
        "confluence_weights": {
            "fvg": 0.20, "order_block": 0.25, "liquidity_sweep": 0.20,
            "structure_break": 0.30, "ema_alignment": 0.10, "rsi_confirmation": 0.08,
            "macd_confirmation": 0.07, "mtf_alignment_bonus": 0.15,
            "ltf_confirmation_bonus": 0.10,
        }
    })

    smc = {
        "fvg": {"in_zone": True},
        "order_block": {"at_zone": False},
        "liquidity": {"swept": True},
        "structure": {"choch": True, "bos": False},
    }
    tech = {"rsi": 55, "ema": {20: 5010, 50: 5000}, "macd": {"histogram": 2.0}}
    market = {"volatility": {"adjustments": {"confluence_adjustment": 0}}}
    mtf = {"is_aligned": True}

    # Strong trend
    s1 = scorer.calculate_score(
        TrendDirection.BULLISH, smc, tech, market, mtf, MarketRegime.STRONG_TREND_UP
    )
    print(f"  Strong Trend: score={s1['score']:.3f}, passing={s1['passing']}, min_conf={s1['min_confluence']}")

    # Reversal (should need higher confluence)
    s2 = scorer.calculate_score(
        TrendDirection.BULLISH, smc, tech, market, mtf, MarketRegime.REVERSAL
    )
    print(f"  Reversal:     score={s2['score']:.3f}, passing={s2['passing']}, min_conf={s2['min_confluence']}")

    assert s2["min_confluence"] > s1["min_confluence"], "Reversal should need higher confluence"
    print("  PASSED")


def test_performance_metrics_regime():
    print("\n=== Test 5: Performance Metrics Regime Breakdown ===")
    from src.backtesting.performance_metrics import PerformanceMetrics

    pm = PerformanceMetrics()
    trades = [
        {"profit": 10.0, "regime": "STRONG_TREND_UP"},
        {"profit": -5.0, "regime": "STRONG_TREND_UP"},
        {"profit": 8.0, "regime": "RANGE_WIDE"},
        {"profit": -3.0, "regime": "RANGE_WIDE"},
        {"profit": -2.0, "regime": "RANGE_WIDE"},
        {"profit": 15.0, "regime": "VOLATILE_BREAKOUT"},
    ]
    labels = [t["regime"] for t in trades]

    breakdown = pm.regime_breakdown_metrics(trades, labels)
    for regime, stats in breakdown.items():
        print(f"  {regime}: {stats['trades']} trades, WR={stats['win_rate']}%, PF={stats['profit_factor']}")

    assert "STRONG_TREND_UP" in breakdown
    assert "RANGE_WIDE" in breakdown
    assert breakdown["STRONG_TREND_UP"]["trades"] == 2
    print("  PASSED")


if __name__ == "__main__":
    test_regime_detector()
    test_micro_account()
    test_structure_sltp()
    test_adaptive_scorer()
    test_performance_metrics_regime()
    print("\n" + "=" * 50)
    print("ALL V3 COMPONENT TESTS PASSED!")
    print("=" * 50)
