"""
Shared pytest fixtures for XAUUSD Trading Bot test suite.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pytz
import pytest

# Add project root to path so all src imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Date/time helpers
# ---------------------------------------------------------------------------

def make_utc(weekday_offset: int, hour: int, minute: int = 0) -> datetime:
    """
    Return a UTC-aware datetime for testing.
    Base date: 2026-02-23 (Monday).
      weekday_offset 0 = Monday
      weekday_offset 1 = Tuesday
      ...
      weekday_offset 5 = Saturday
      weekday_offset 6 = Sunday
    """
    base = datetime(2026, 2, 23, 0, 0, 0, tzinfo=pytz.UTC)   # Monday
    return base + timedelta(days=weekday_offset, hours=hour, minutes=minute)


# ---------------------------------------------------------------------------
# Session config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def session_cfg():
    """Minimal session configuration matching session_config.yaml.

    Static times are winter fallbacks; DST auto_detect overrides them.
    Test base date (2026-02-23) is winter, so DST times == winter times.
    """
    return {
        "dst": {"auto_detect": True},
        "sessions": {
            "asian": {
                "name": "Asian Session",
                "start_utc": "00:00",
                "end_utc": "09:00",
                "weight": 0.75,
                "min_confluence_adjustment": 0.05,
            },
            "london": {
                "name": "London Session",
                "start_utc": "08:00",
                "end_utc": "17:00",
                "weight": 1.16,
                "min_confluence_adjustment": 0.0,
            },
            "new_york": {
                "name": "New York Session",
                "start_utc": "13:00",
                "end_utc": "22:00",
                "weight": 1.16,
                "min_confluence_adjustment": 0.0,
            },
            "overlap": {
                "name": "London-NY Overlap",
                "start_utc": "13:00",
                "end_utc": "17:00",
                "weight": 1.18,
                "min_confluence_adjustment": -0.05,
            },
        },
        "restrictions": {
            "trading_days": [0, 1, 2, 3, 4],
            "avoid_weekends": True,
            "friday_close_early": True,
            "friday_close_time_utc": "21:30",
            "blackout_hours": ["22:00-23:00"],
        },
        "preferences": {
            "preferred_sessions": ["overlap", "london", "new_york"],
            "avoid_sessions": [],
            "session_adjustments": {},
        },
    }


# ---------------------------------------------------------------------------
# Confluence scorer config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer_cfg():
    """Minimal config for AdaptiveConfluenceScorer."""
    return {
        "confluence_weights": {
            "fvg": 0.20,
            "order_block": 0.25,
            "liquidity_sweep": 0.20,
            "structure_break": 0.30,
            "ema_alignment": 0.10,
            "rsi_confirmation": 0.08,
            "macd_confirmation": 0.07,
        }
    }


# ---------------------------------------------------------------------------
# Entry signal config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def entry_cfg():
    """Minimal config for EntrySignalGenerator (gates all OFF = permissive)."""
    return {
        "strategy": {
            "entry": {
                "min_confluence_score": 0.55,
                "require_fvg_or_ob": False,
                "require_structure_support": False,
                "require_mtf_alignment": False,
            }
        }
    }


# ---------------------------------------------------------------------------
# SMC signal helpers
# ---------------------------------------------------------------------------

def make_smc(bos=False, choch=False, fvg=False, ob=False, liq=False) -> dict:
    """Build a minimal smc_signals dict for testing."""
    return {
        "fvg": {"in_zone": fvg},
        "order_block": {"at_zone": ob},
        "liquidity": {"swept": liq},
        "structure": {"choch": choch, "bos": bos},
    }


def make_confluence(score: float, passing: bool = None) -> dict:
    """Build a minimal confluence_score dict."""
    if passing is None:
        passing = score >= 0.55
    return {"score": score, "passing": passing}


def make_tech(rsi: float = 55.0) -> dict:
    """Build minimal technical_indicators dict.

    Matches the structure expected by AdaptiveConfluenceScorer:
      - ema: dict with integer keys {20: price, 50: price}
      - macd: dict with 'histogram' key (not a plain float)
      - rsi: float
    """
    return {
        "rsi": rsi,
        "ema": {20: 5060.0, 50: 5050.0},   # Scorer expects ema_data.get(20), .get(50)
        "macd": {"histogram": 1.5, "macd": 1.5, "signal": 0.8},
        "atr": 15.0,
    }


def make_market(favorable: bool = True) -> dict:
    return {"is_favorable": favorable, "trend": "bullish"}


def make_mtf(aligned: bool = False) -> dict:
    from src.core.constants import TrendDirection
    return {
        "is_aligned": aligned,
        "dominant_trend": TrendDirection.BULLISH if aligned else TrendDirection.NEUTRAL,
        "h1_bias": "neutral",
    }
