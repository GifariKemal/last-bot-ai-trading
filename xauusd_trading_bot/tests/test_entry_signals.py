"""
Tests: EntrySignalGenerator — RSI gates and SMC signal count.

Critical bugs tested:
  Bug #31: RSI_HARD_OVERBOUGHT was 70 → blocked ALL longs in trending markets.
           Fix: raised to 90. Test that RSI 85 does NOT block entry.
  Bug #38: require_all_positions_profitable blocked entries when any position at loss.
           (Tested indirectly via open_position_count gate.)
  Regression: CHoCH alone (1 signal) should fail min_smc=2 gate for pyramid.
"""

import pytest
from conftest import make_smc, make_tech, make_market, make_mtf, make_confluence
from src.strategy.entry_signals import EntrySignalGenerator
from src.core.constants import MarketRegime


@pytest.fixture
def gen(entry_cfg):
    return EntrySignalGenerator(entry_cfg)


def _base_signal(gen, direction="BUY", smc=None, rsi=55.0, open_positions=0, score=0.70):
    """Call generate_signal with minimal valid inputs."""
    from src.core.constants import MarketRegime
    smc = smc or make_smc(bos=True, choch=True)
    bullish_smc = {
        "fvg": smc["fvg"],
        "order_block": smc["order_block"],
        "liquidity": smc["liquidity"],
        "structure": smc["structure"],
    }
    bearish_smc = {k: {"in_zone": False, "at_zone": False, "swept": False, "choch": False, "bos": False}.get(
        next(iter(v)), v) for k, v in bullish_smc.items()
    }
    # For simplicity, provide same smc for both directions
    return gen.generate_signal(
        current_price=5000.0,
        confluence_scores={
            "bullish": make_confluence(score, passing=score >= 0.55) if direction == "BUY" else make_confluence(0.30, False),
            "bearish": make_confluence(score, passing=score >= 0.55) if direction == "SELL" else make_confluence(0.30, False),
        },
        smc_signals={
            "bullish": bullish_smc,
            "bearish": bullish_smc,
        },
        market_analysis=make_market(favorable=True),
        mtf_analysis=make_mtf(aligned=False),
        technical_indicators=make_tech(rsi=rsi),
        open_position_count=open_positions,
        regime=MarketRegime.RANGE_WIDE,
    )


# ── RSI hard block ────────────────────────────────────────────────────────────

def test_rsi_85_does_not_block_long(gen):
    """Bug #31 regression: RSI 85 must NOT block LONG (threshold was 70, then 85, now 90)."""
    signal = _base_signal(gen, direction="BUY", rsi=85.0)
    # With all gates OFF (entry_cfg) and RSI=85 < 90, should produce valid entry
    assert signal.get("valid") is True, (
        f"RSI 85 should NOT block long (hard block=90). Got: {signal}"
    )


def test_rsi_90_blocks_long(gen):
    """RSI >= 90 must hard-block LONG (extreme overbought)."""
    signal = _base_signal(gen, direction="BUY", rsi=92.0)
    assert signal.get("valid") is not True, (
        f"RSI 92 should block LONG (RSI_HARD_OVERBOUGHT=90). Got: {signal}"
    )


def test_rsi_10_blocks_short(gen):
    """RSI <= 10 must hard-block SHORT (extreme oversold)."""
    signal = _base_signal(gen, direction="SELL", rsi=8.0)
    assert signal.get("valid") is not True, (
        f"RSI 8 should block SHORT (RSI_HARD_OVERSOLD=10). Got: {signal}"
    )


def test_rsi_15_does_not_block_short(gen):
    """RSI 15 must NOT block SHORT (above hard-oversold threshold of 10)."""
    signal = _base_signal(gen, direction="SELL", rsi=15.0)
    assert signal.get("valid") is True, (
        f"RSI 15 should NOT block SHORT (hard block=10). Got: {signal}"
    )


# ── RSI bounce protection ─────────────────────────────────────────────────────

def test_rsi_bounce_from_overbought_blocks_long(gen):
    """If RSI was >75 recently and is now falling, BUY should be blocked."""
    # Feed history: RSI was 80 (extreme), now at 72 (falling)
    gen.update_rsi_history(80.0)
    gen.update_rsi_history(80.0)
    gen.update_rsi_history(72.0)  # Now falling below 75 threshold

    signal = _base_signal(gen, direction="BUY", rsi=72.0)
    assert signal.get("valid") is not True, (
        "RSI bouncing from overbought (80→72) should block BUY"
    )
    # Reset RSI history for next tests
    gen._recent_rsi_values = []


def test_normal_rsi_no_bounce_block(gen):
    """Normal RSI without bounce history must not trigger bounce protection."""
    gen._recent_rsi_values = []  # Clean slate
    for v in [55, 56, 57]:
        gen.update_rsi_history(v)
    signal = _base_signal(gen, direction="BUY", rsi=57.0)
    assert signal.get("valid") is True, (
        "Normal RSI 55-57 should not trigger bounce protection"
    )


# ── Min SMC signals gate ──────────────────────────────────────────────────────

def test_two_signals_pass_for_first_position(gen):
    """BOS + CHoCH = 2 signals. First position (open_positions=0): only 1 required → passes."""
    gen._recent_rsi_values = []
    smc = make_smc(bos=True, choch=True)
    signal = _base_signal(gen, direction="BUY", smc=smc, open_positions=0)
    assert signal.get("valid") is True


def test_one_signal_blocks_pyramid_entry(gen):
    """CHoCH alone = 1 signal. With open_position (pyramid), need 2 → must fail."""
    gen._recent_rsi_values = []
    smc = make_smc(choch=True)  # Only 1 signal
    signal = _base_signal(gen, direction="BUY", smc=smc, open_positions=1)
    assert signal.get("valid") is not True, (
        "1 SMC signal must fail for pyramid entry (need 2 when position open)"
    )


def test_two_signals_pass_for_pyramid(gen):
    """BOS + LiqSweep = 2 signals. Pyramid (open_positions=1) should pass."""
    gen._recent_rsi_values = []
    smc = make_smc(bos=True, liq=True)  # 2 signals
    signal = _base_signal(gen, direction="BUY", smc=smc, open_positions=1)
    assert signal.get("valid") is True, (
        "BOS + LiqSweep (2 signals) should pass pyramid gate"
    )


# ── Low confluence blocks entry ───────────────────────────────────────────────

def test_low_confluence_blocks_entry(gen):
    """Score below 0.55 must not produce a valid signal."""
    gen._recent_rsi_values = []
    signal = _base_signal(gen, direction="BUY", rsi=55.0, score=0.40)
    assert signal.get("valid") is not True, (
        "Confluence 0.40 must be blocked (min=0.55)"
    )


def test_confluence_at_threshold_passes(gen):
    """Score exactly at 0.55 should pass the confluence gate."""
    gen._recent_rsi_values = []
    signal = _base_signal(gen, direction="BUY", rsi=55.0, score=0.55)
    assert signal.get("valid") is True, (
        "Confluence 0.55 should pass the gate (min=0.55)"
    )
