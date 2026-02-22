"""
Tests: Config file integrity.

Critical bugs tested:
  Bug #35: trading_bot.py used config.get("indicators") → empty dict.
           Correct key is "smc_indicators" in trading_rules.yaml.
           Test that the key exists and has expected sub-keys.
  General: All 4 config files must load without error and have required fields.
"""

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(filename: str) -> dict:
    path = CONFIG_DIR / filename
    assert path.exists(), f"Config file not found: {path}"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── settings.yaml ─────────────────────────────────────────────────────────────

def test_settings_loads():
    cfg = load_yaml("settings.yaml")
    assert isinstance(cfg, dict)


def test_settings_has_required_sections():
    cfg = load_yaml("settings.yaml")
    for section in ["trading", "telegram", "emergency"]:
        assert section in cfg, f"settings.yaml missing section: {section}"


def test_settings_use_adaptive_scorer():
    """use_adaptive_scorer flag must exist for V3/V4 switching."""
    cfg = load_yaml("settings.yaml")
    assert "use_adaptive_scorer" in cfg, (
        "settings.yaml must have 'use_adaptive_scorer' flag for V3 activation"
    )


def test_settings_max_open_positions():
    """max_open_positions must be >= 3 (was raised from 2 in Bug #38 fix)."""
    cfg = load_yaml("settings.yaml")
    max_pos = cfg.get("trading", {}).get("max_open_positions", 0)
    assert max_pos >= 3, (
        f"Bug #38: max_open_positions={max_pos} should be >=3. "
        "Was reduced to 2 causing fewer entries than dynamic max_pos=3."
    )


def test_settings_require_all_profitable_is_false():
    """require_all_positions_profitable must be false (Bug #38)."""
    cfg = load_yaml("settings.yaml")
    val = cfg.get("trading", {}).get("require_all_positions_profitable", True)
    assert val is False, (
        "Bug #38: require_all_positions_profitable=true blocked ALL entries. Must be false."
    )


# ── trading_rules.yaml ────────────────────────────────────────────────────────

def test_trading_rules_loads():
    cfg = load_yaml("trading_rules.yaml")
    assert isinstance(cfg, dict)


def test_trading_rules_has_smc_indicators_not_indicators():
    """Bug #35: key must be 'smc_indicators', not 'indicators'."""
    cfg = load_yaml("trading_rules.yaml")
    assert "smc_indicators" in cfg, (
        "Bug #35: trading_rules.yaml must have 'smc_indicators' key "
        "(not 'indicators'). Wrong key → all SMC detectors use hardcoded defaults."
    )
    assert "indicators" not in cfg, (
        "Stale 'indicators' key present — remove it to avoid confusion with smc_indicators."
    )


def test_smc_indicators_has_order_block():
    """smc_indicators.order_blocks must have strong_move_percent ≤ 0.5."""
    cfg = load_yaml("trading_rules.yaml")
    # Key is 'order_blocks' (plural) in trading_rules.yaml
    ob = cfg.get("smc_indicators", {}).get("order_blocks", {})
    assert ob, "smc_indicators.order_blocks section must exist"
    assert "strong_move_percent" in ob, (
        "order_blocks.strong_move_percent must be configured "
        "(default 1.0 means OB never detected at XAUUSD ~$5000)"
    )
    # strong_move_percent must be ≤ 0.5 for XAUUSD (0.45 = V3 Optuna-optimized)
    assert ob["strong_move_percent"] <= 0.5, (
        f"strong_move_percent={ob['strong_move_percent']} too high. "
        "At $5000, 1.0% = $50 = rarely detected. V3 optimized value = 0.45."
    )


def test_smc_indicators_sweep_confirmation_bars():
    """sweep_confirmation_bars must be 20 (optimized design decision)."""
    cfg = load_yaml("trading_rules.yaml")
    # Key is 'liquidity' in smc_indicators section
    liq = cfg.get("smc_indicators", {}).get("liquidity", {})
    bars = liq.get("sweep_confirmation_bars", 3)
    assert bars >= 10, (
        f"sweep_confirmation_bars={bars} too low. "
        "Optimized value is 20 bars (5h). Low value = false sweeps."
    )


# ── risk_config.yaml ──────────────────────────────────────────────────────────

def test_risk_config_loads():
    cfg = load_yaml("risk_config.yaml")
    assert isinstance(cfg, dict)


def test_risk_config_exit_stages():
    cfg = load_yaml("risk_config.yaml")
    stages = cfg.get("exit_stages", {})
    assert "be_trigger_rr" in stages, "exit_stages.be_trigger_rr must exist"
    assert "partial_close_rr" in stages, "exit_stages.partial_close_rr must exist"


def test_risk_be_trigger_is_below_partial():
    """BE trigger must fire before partial close (otherwise trail bug persists)."""
    cfg = load_yaml("risk_config.yaml")
    stages = cfg.get("exit_stages", {})
    be_rr = stages.get("be_trigger_rr", 0.77)
    partial_rr = stages.get("partial_close_rr", 2.73)
    assert be_rr < partial_rr, (
        f"BE trigger ({be_rr}R) must be < partial_close ({partial_rr}R). "
        "Trail activates after BE, not after partial."
    )


# ── session_config.yaml ───────────────────────────────────────────────────────

def test_session_config_loads():
    cfg = load_yaml("session_config.yaml")
    assert isinstance(cfg, dict)


def test_session_maintenance_is_correct():
    """Maintenance window must be 00:00-01:00 UTC (verified from MT5 candle gaps)."""
    cfg = load_yaml("session_config.yaml")
    blackout = cfg.get("restrictions", {}).get("blackout_hours", [])
    assert "00:00-01:00" in blackout, (
        f"Maintenance blackout must be '00:00-01:00' (verified from MT5 data). "
        f"Got: {blackout}. Old wrong assumption was 21:00-22:00."
    )


def test_session_friday_close_is_late():
    """Friday close must be 23:30 UTC (not 21:00 old assumption)."""
    cfg = load_yaml("session_config.yaml")
    close_time = cfg.get("restrictions", {}).get("friday_close_time_utc", "")
    assert close_time == "23:30", (
        f"friday_close_time_utc='{close_time}' but must be '23:30'. "
        "IC Markets last candle is 23:45 Fri, so close at 23:30 is correct."
    )


def test_session_blackout_covers_maintenance_window():
    """blackout_hours must cover 00:00-01:00 so Asian's nominal 00:00 start is blocked."""
    cfg = load_yaml("session_config.yaml")
    blackout = cfg.get("restrictions", {}).get("blackout_hours", [])
    # The maintenance window 00:00-01:00 must be in blackout so that even if Asian
    # session nominally starts at 00:00, the detector blocks trading until 01:00.
    assert any("00:00" in b for b in blackout), (
        f"blackout_hours must include 00:00-01:00 maintenance window. Got: {blackout}"
    )
