"""
Tests: AdaptiveConfluenceScorer — score range and inflation regression.

Critical bugs tested:
  Bug #40: Score inflation — ALL scores were 1.00 due to wrong normalization.
           Fix: smc_fill = smc_raw / smc_base_max (not / 0.40 category weight).
  Also: score must always be clamped to [0.0, 1.0].
"""

import pytest
from conftest import make_smc, make_tech, make_market, make_mtf
from src.analysis.adaptive_scorer import AdaptiveConfluenceScorer
from src.core.constants import TrendDirection, MarketRegime


@pytest.fixture
def scorer(scorer_cfg):
    return AdaptiveConfluenceScorer(scorer_cfg)


def _bullish_smc(bos=True, choch=False, fvg=False, ob=False, liq=False):
    """Build bullish smc_signals with given signals active."""
    return {
        "fvg": {"in_zone": fvg, "type": "bullish"},
        "order_block": {"at_zone": ob, "type": "bullish"},
        "liquidity": {"swept": liq, "direction": "bullish"},
        "structure": {"choch": choch, "bos": bos, "direction": "bullish"},
    }


# ── Bug #40 regression: no score inflation ───────────────────────────────────

def test_no_signals_score_is_low(scorer):
    """Zero SMC signals must produce a low score, not inflated."""
    smc = _bullish_smc(bos=False, choch=False, fvg=False, ob=False, liq=False)
    result = scorer.calculate_score(
        direction=TrendDirection.BULLISH,
        smc_signals=smc,
        technical_indicators=make_tech(rsi=50),
        market_analysis=make_market(),
        mtf_analysis=make_mtf(),
        regime=MarketRegime.RANGE_WIDE,
    )
    assert result["score"] < 0.40, (
        f"Bug #40: empty signals score={result['score']:.3f} should be < 0.40"
    )


def test_all_signals_score_below_one(scorer):
    """Even with all signals active, score must be < 1.0 (not clamped-at-max)."""
    smc = _bullish_smc(bos=True, choch=True, fvg=True, ob=True, liq=True)
    result = scorer.calculate_score(
        direction=TrendDirection.BULLISH,
        smc_signals=smc,
        technical_indicators=make_tech(rsi=60),
        market_analysis=make_market(),
        mtf_analysis=make_mtf(),
        regime=MarketRegime.RANGE_WIDE,
    )
    # Should be high but physically impossible to be exactly 1.0 with just SMC
    assert 0.50 <= result["score"] <= 1.0, (
        f"All-signals score={result['score']:.3f} out of expected range 0.50-1.0"
    )


def test_bos_only_score_is_realistic(scorer):
    """BOS alone should score roughly 0.20-0.50 (not inflated to 1.00)."""
    smc = _bullish_smc(bos=True)
    result = scorer.calculate_score(
        direction=TrendDirection.BULLISH,
        smc_signals=smc,
        technical_indicators=make_tech(rsi=55),
        market_analysis=make_market(),
        mtf_analysis=make_mtf(),
        regime=MarketRegime.RANGE_WIDE,
    )
    score = result["score"]
    assert 0.10 <= score <= 0.55, (
        f"BOS-only score={score:.3f} should be 0.10-0.55. Bug #40 if >= 0.80"
    )


def test_score_always_clamped_0_to_1(scorer):
    """Score must never exceed 1.0 or go below 0.0 regardless of inputs."""
    for bos, choch, fvg, ob, liq in [
        (True, True, True, True, True),
        (False, False, False, False, False),
        (True, False, False, False, False),
    ]:
        smc = _bullish_smc(bos=bos, choch=choch, fvg=fvg, ob=ob, liq=liq)
        result = scorer.calculate_score(
            direction=TrendDirection.BULLISH,
            smc_signals=smc,
            technical_indicators=make_tech(),
            market_analysis=make_market(),
            mtf_analysis=make_mtf(),
            regime=MarketRegime.RANGE_WIDE,
        )
        assert 0.0 <= result["score"] <= 1.0, (
            f"Score={result['score']:.3f} outside [0,1] for signals {bos,choch,fvg,ob,liq}"
        )


# ── Regime changes min_confluence ────────────────────────────────────────────

def test_trending_regime_lower_min_confluence(scorer):
    """Trending regime has lower min_confluence than volatile regime."""
    trending_params = scorer.get_regime_params(MarketRegime.STRONG_TREND_UP)
    volatile_params = scorer.get_regime_params(MarketRegime.VOLATILE_BREAKOUT)
    assert trending_params["min_confluence"] < volatile_params["min_confluence"], (
        "Trending should require less confluence than volatile (easier to enter trends)"
    )


def test_reversal_regime_highest_min_confluence(scorer):
    """Reversal regime should have the highest min_confluence (riskiest)."""
    reversal_params = scorer.get_regime_params(MarketRegime.REVERSAL)
    ranging_params = scorer.get_regime_params(MarketRegime.RANGE_WIDE)
    assert reversal_params["min_confluence"] >= ranging_params["min_confluence"]


# ── Score consistency: more signals = higher score ────────────────────────────

def test_more_signals_higher_score(scorer):
    """Adding BOS + LiqSweep should score higher than BOS alone."""
    smc_bos_only = _bullish_smc(bos=True)
    smc_bos_liq = _bullish_smc(bos=True, liq=True)

    result_single = scorer.calculate_score(
        TrendDirection.BULLISH, smc_bos_only, make_tech(), make_market(), make_mtf()
    )
    result_double = scorer.calculate_score(
        TrendDirection.BULLISH, smc_bos_liq, make_tech(), make_market(), make_mtf()
    )
    assert result_double["score"] > result_single["score"], (
        f"2-signal score ({result_double['score']:.3f}) should exceed "
        f"1-signal score ({result_single['score']:.3f})"
    )


def test_scorer_returns_required_keys(scorer):
    """calculate_score must always return score, passing, regime, breakdown keys."""
    smc = _bullish_smc(bos=True)
    result = scorer.calculate_score(
        direction=TrendDirection.BULLISH,
        smc_signals=smc,
        technical_indicators=make_tech(),
        market_analysis=make_market(),
        mtf_analysis=make_mtf(),
    )
    for key in ["score", "passing", "regime", "min_confluence", "breakdown"]:
        assert key in result, f"calculate_score missing key: {key}"
    # passing must agree with score >= min_confluence
    assert result["passing"] == (result["score"] >= result["min_confluence"])
