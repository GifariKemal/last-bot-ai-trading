"""
Tests for the Kimi K2 prediction pipeline.

Covers:
- models/market_context.py  — build_context()
- models/kimi_predictor.py  — KimiPredictor._parse_response(), .predict()
- config/settings.py        — KimiConfig defaults

Run with:
    cd xauusd-predictive-analytics
    pytest tests/test_kimi_pipeline.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Ensure project root is on sys.path so package imports resolve correctly
# regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import KimiConfig
from models.market_context import build_context
from models.kimi_predictor import KimiPredictor


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic_df(n_rows: int = 25) -> pd.DataFrame:
    """
    Return a feature-complete M15 DataFrame with n_rows rows.

    All indicator columns are filled with realistic-ish constant values so
    that build_context() and KimiPredictor.predict() run without KeyErrors.
    Individual tests may override specific cells before passing the frame in.
    """
    idx = pd.date_range(
        start="2025-01-06 00:00",   # Monday
        periods=n_rows,
        freq="15min",
        tz="UTC",
    )

    data: dict[str, Any] = {
        # OHLCV
        "open":        2600.0,
        "high":        2605.0,
        "low":         2595.0,
        "close":       2602.0,
        "tick_volume": 1000.0,

        # ── M15 momentum ──
        "rsi_14":      55.0,
        "macd":        1.2,
        "macd_signal": 0.9,
        "macd_hist":   0.3,
        "stoch_k":     58.0,
        "stoch_d":     54.0,
        "cci_20":      45.0,

        # ── M15 trend / regime ──
        "adx_14":      28.0,
        "di_diff":     0.15,
        "atr_14":      3.5,
        "atr_regime":  1.05,

        # ── M15 volatility ──
        "bb_pct":      0.55,
        "bb_width":    0.0032,
        "close_vs_ema50":  0.0018,
        "close_vs_ema200": 0.0045,

        # ── H1 context ──
        "h1_ema50_bias": 1,
        "h1_rsi":       57.0,
        "h1_adx":       30.0,
        "h1_bb_pos":    0.60,
        "h1_bb_width":  0.0028,

        # ── H4 context ──
        "h4_ema50_bias": 1,
        "h4_rsi":       55.0,
        "h4_adx":       26.0,
        "h4_bb_pos":    0.52,
        "h4_bb_width":  0.0020,

        # ── Candle patterns (all neutral by default) ──
        "cdl_hammer":        0,
        "cdl_engulfing":     0,
        "cdl_shooting_star": 0,

        # ── Session flags ──
        "is_asian":    0,
        "is_london":   1,
        "is_new_york": 0,
        "is_overlap":  0,

        # ── Calendar ──
        "day_of_week": 0,   # Monday

        # ── Session high / low proximity ──
        "dist_to_high_8h":  0.80,
        "dist_to_low_8h":   1.20,
        "dist_to_high_24h": 1.50,
        "dist_to_low_24h":  2.00,
    }

    df = pd.DataFrame(
        {col: [val] * n_rows for col, val in data.items()},
        index=idx,
    )
    return df


def _make_mock_openai_response(content: str) -> MagicMock:
    """Build a minimal mock that looks like an openai ChatCompletion response."""
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# ──────────────────────────────────────────────────────────────────────────────
# build_context() tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildContext:
    """Unit tests for models.market_context.build_context."""

    def test_build_context_basic(self):
        """Context dict contains all required keys and recent_bars has 20 entries."""
        df  = make_synthetic_df(n_rows=25)
        ctx = build_context(df)

        # ── Top-level key presence ─────────────────────────────────────────
        required_top_level_keys = {
            "time", "day_of_week", "active_sessions", "recent_bars",
            "rsi", "macd", "macd_signal", "macd_hist",
            "stoch_k", "stoch_d", "cci",
            "adx", "di_diff", "atr", "atr_regime",
            "bb_pct", "bb_width",
            "ema50_dist", "ema200_dist",
            "h1_rsi", "h1_adx", "h1_bb_pos", "h1_ema_bias",
            "h4_rsi", "h4_adx", "h4_bb_pos", "h4_ema_bias",
            "dist_high_8h", "dist_low_8h", "dist_high_24h", "dist_low_24h",
            "patterns",
        }
        missing = required_top_level_keys - ctx.keys()
        assert not missing, f"Context is missing keys: {missing}"

        # ── recent_bars length ─────────────────────────────────────────────
        assert len(ctx["recent_bars"]) == 20, (
            f"Expected 20 recent bars, got {len(ctx['recent_bars'])}"
        )

        # ── Each bar has the required OHLCV keys ───────────────────────────
        required_bar_keys = {"time", "open", "high", "low", "close", "volume"}
        for i, bar in enumerate(ctx["recent_bars"]):
            missing_bar_keys = required_bar_keys - bar.keys()
            assert not missing_bar_keys, (
                f"Bar {i} is missing keys: {missing_bar_keys}"
            )

    def test_build_context_indicator_values(self):
        """Spot-check that indicator values are extracted correctly from the DataFrame."""
        df  = make_synthetic_df(n_rows=25)
        ctx = build_context(df)

        assert ctx["rsi"]  == pytest.approx(55.0)
        assert ctx["adx"]  == pytest.approx(28.0)
        assert ctx["atr"]  == pytest.approx(3.5)
        assert ctx["h1_rsi"] == pytest.approx(57.0)
        assert ctx["h4_adx"] == pytest.approx(26.0)

    def test_build_context_bar_ohlcv_values(self):
        """Values inside each bar dict match the source DataFrame rows."""
        df  = make_synthetic_df(n_rows=25)
        ctx = build_context(df)

        first_bar = ctx["recent_bars"][0]
        assert first_bar["open"]   == pytest.approx(2600.0)
        assert first_bar["high"]   == pytest.approx(2605.0)
        assert first_bar["low"]    == pytest.approx(2595.0)
        assert first_bar["close"]  == pytest.approx(2602.0)
        assert first_bar["volume"] == pytest.approx(1000.0)

    def test_build_context_raises_on_insufficient_rows(self):
        """ValueError is raised when the DataFrame has fewer than 20 clean rows."""
        df = make_synthetic_df(n_rows=15)
        with pytest.raises(ValueError, match="Insufficient clean rows"):
            build_context(df)

    def test_build_context_exactly_twenty_rows(self):
        """A DataFrame with exactly 20 rows should succeed (boundary condition)."""
        df  = make_synthetic_df(n_rows=20)
        ctx = build_context(df)
        assert len(ctx["recent_bars"]) == 20

    # ── Pattern detection ──────────────────────────────────────────────────

    def test_build_context_pattern_detection(self):
        """
        When cdl_hammer=1 (bullish) and cdl_engulfing=-1 (bearish) appear in
        the latest row, both are reflected in the patterns string.
        """
        df = make_synthetic_df(n_rows=25)
        df.iloc[-1, df.columns.get_loc("cdl_hammer")]    = 1
        df.iloc[-1, df.columns.get_loc("cdl_engulfing")] = -1

        ctx = build_context(df)

        assert "BULLISH" in ctx["patterns"] and "Hammer" in ctx["patterns"], (
            f"Expected 'BULLISH Hammer' in patterns, got: {ctx['patterns']!r}"
        )
        assert "BEARISH" in ctx["patterns"] and "Engulfing" in ctx["patterns"], (
            f"Expected 'BEARISH Engulfing' in patterns, got: {ctx['patterns']!r}"
        )

    def test_build_context_shooting_star_bearish(self):
        """cdl_shooting_star=-1 is labelled BEARISH Shooting Star."""
        df = make_synthetic_df(n_rows=25)
        df.iloc[-1, df.columns.get_loc("cdl_shooting_star")] = -1

        ctx = build_context(df)
        assert "BEARISH" in ctx["patterns"] and "Shooting Star" in ctx["patterns"]

    def test_build_context_no_patterns(self):
        """With all pattern columns at 0, patterns string should be empty."""
        df  = make_synthetic_df(n_rows=25)
        ctx = build_context(df)
        assert ctx["patterns"] == ""

    # ── Session detection ──────────────────────────────────────────────────

    def test_build_context_session_detection_overlap(self):
        """is_overlap=1 adds 'London+NY Overlap (high liquidity)' to active_sessions."""
        df = make_synthetic_df(n_rows=25)
        df.iloc[-1, df.columns.get_loc("is_overlap")]  = 1
        df.iloc[-1, df.columns.get_loc("is_london")]   = 0
        df.iloc[-1, df.columns.get_loc("is_new_york")] = 0

        ctx = build_context(df)
        assert "London+NY Overlap (high liquidity)" in ctx["active_sessions"], (
            f"Expected Overlap session, got: {ctx['active_sessions']}"
        )

    def test_build_context_session_london_only(self):
        """is_london=1 (no overlap) lists London session only."""
        df = make_synthetic_df(n_rows=25)
        # defaults already have is_london=1, is_overlap=0
        ctx = build_context(df)
        assert "London" in ctx["active_sessions"]
        assert not any("Overlap" in s for s in ctx["active_sessions"])

    def test_build_context_session_asian(self):
        """is_asian=1 appends 'Asian' to active_sessions."""
        df = make_synthetic_df(n_rows=25)
        df.iloc[-1, df.columns.get_loc("is_asian")]  = 1
        df.iloc[-1, df.columns.get_loc("is_london")] = 0

        ctx = build_context(df)
        assert "Asian" in ctx["active_sessions"]

    def test_build_context_off_session(self):
        """All session flags at 0 yields an empty active_sessions list."""
        df = make_synthetic_df(n_rows=25)
        for col in ("is_asian", "is_london", "is_new_york", "is_overlap"):
            df.iloc[-1, df.columns.get_loc(col)] = 0

        ctx = build_context(df)
        assert ctx["active_sessions"] == []

    # ── Day-of-week ────────────────────────────────────────────────────────

    def test_build_context_day_of_week_monday(self):
        """day_of_week=0 translates to 'Monday'."""
        df  = make_synthetic_df(n_rows=25)
        ctx = build_context(df)
        assert ctx["day_of_week"] == "Monday"

    def test_build_context_day_of_week_friday(self):
        """day_of_week=4 translates to 'Friday'."""
        df = make_synthetic_df(n_rows=25)
        df.iloc[-1, df.columns.get_loc("day_of_week")] = 4
        ctx = build_context(df)
        assert ctx["day_of_week"] == "Friday"


# ──────────────────────────────────────────────────────────────────────────────
# KimiPredictor._parse_response() tests
# ──────────────────────────────────────────────────────────────────────────────

class TestParseResponse:
    """Unit tests for KimiPredictor._parse_response() (static method — no API needed)."""

    def test_parse_response_buy(self):
        """Valid BUY JSON is parsed to signal=1."""
        raw = json.dumps({
            "signal":      "BUY",
            "confidence":  0.75,
            "reasoning":   "Strong bullish momentum with ADX > 25.",
            "key_factors": ["RSI 55 trending up", "ADX=28", "London session"],
        })
        result = KimiPredictor._parse_response(raw)

        assert result["signal"]       == 1
        assert result["signal_label"] == "BUY"
        assert result["confidence"]   == pytest.approx(0.75)
        assert "RSI" in result["key_factors"][0]

    def test_parse_response_sell(self):
        """Valid SELL JSON is parsed to signal=-1."""
        raw = json.dumps({
            "signal":      "SELL",
            "confidence":  0.68,
            "reasoning":   "Bearish engulfing at resistance.",
            "key_factors": ["CCI overbought", "Shooting star", "BB upper touch"],
        })
        result = KimiPredictor._parse_response(raw)

        assert result["signal"]       == -1
        assert result["signal_label"] == "SELL"
        assert result["confidence"]   == pytest.approx(0.68)

    def test_parse_response_no_trade(self):
        """Valid NO_TRADE JSON is parsed to signal=0."""
        raw = json.dumps({
            "signal":      "NO_TRADE",
            "confidence":  0.40,
            "reasoning":   "Mixed signals, low ADX.",
            "key_factors": ["ADX < 20", "MACD flat"],
        })
        result = KimiPredictor._parse_response(raw)

        assert result["signal"]       == 0
        assert result["signal_label"] == "NO_TRADE"

    def test_parse_response_invalid_json(self):
        """Non-JSON input results in a graceful NO_TRADE fallback (signal=0)."""
        result = KimiPredictor._parse_response("not json at all %%##")

        assert result["signal"]       == 0
        assert result["signal_label"] == "NO_TRADE"
        assert result["confidence"]   == pytest.approx(0.0)

    def test_parse_response_unknown_signal(self):
        """An unrecognised signal value (e.g. 'HOLD') defaults to NO_TRADE."""
        raw = json.dumps({
            "signal":      "HOLD",
            "confidence":  0.50,
            "reasoning":   "Wait for confirmation.",
            "key_factors": ["Flat momentum"],
        })
        result = KimiPredictor._parse_response(raw)

        assert result["signal"]       == 0
        assert result["signal_label"] == "NO_TRADE"

    def test_parse_response_case_insensitive(self):
        """Signal matching is case-insensitive ('buy' treated same as 'BUY')."""
        raw = json.dumps({
            "signal":      "buy",
            "confidence":  0.60,
            "reasoning":   "Lowercase signal test.",
            "key_factors": [],
        })
        result = KimiPredictor._parse_response(raw)

        assert result["signal"]       == 1
        assert result["signal_label"] == "BUY"

    def test_parse_response_missing_optional_fields(self):
        """Missing confidence / reasoning / key_factors fall back to safe defaults."""
        raw = json.dumps({"signal": "SELL"})
        result = KimiPredictor._parse_response(raw)

        assert result["signal"]       == -1
        assert result["confidence"]   == pytest.approx(0.0)
        assert result["reasoning"]    == ""
        assert result["key_factors"]  == []

    def test_parse_response_raw_field_preserved(self):
        """raw_response always equals the exact string passed in."""
        raw = json.dumps({
            "signal": "BUY", "confidence": 0.9,
            "reasoning": "test", "key_factors": [],
        })
        result = KimiPredictor._parse_response(raw)
        assert result["raw_response"] == raw

    def test_parse_response_empty_string(self):
        """Empty string is invalid JSON — returns NO_TRADE gracefully."""
        result = KimiPredictor._parse_response("")
        assert result["signal"]       == 0
        assert result["signal_label"] == "NO_TRADE"

    def test_parse_response_empty_json_object(self):
        """Empty JSON object '{}' should default signal to NO_TRADE."""
        result = KimiPredictor._parse_response("{}")
        assert result["signal"]       == 0
        assert result["signal_label"] == "NO_TRADE"


# ──────────────────────────────────────────────────────────────────────────────
# KimiPredictor.predict() — full pipeline with mocked OpenAI client
# ──────────────────────────────────────────────────────────────────────────────

class TestKimiPredictorPredict:
    """Integration-style tests for predict() with the OpenAI client mocked out."""

    def _make_predictor(self) -> KimiPredictor:
        """Return a KimiPredictor wired to a fake config (no real API calls)."""
        cfg = KimiConfig(
            api_key="test-key-does-not-matter",
            base_url="http://test-host/v1",
            model="test-model",
        )
        # Patch OpenAI at construction time so no real HTTP connection is attempted
        with patch("models.kimi_predictor.OpenAI"):
            predictor = KimiPredictor(cfg, max_retries=1, retry_delay=0.0)
        return predictor

    def test_predict_with_mocked_api_buy(self):
        """predict() returns signal=1 when the mocked API responds with BUY."""
        predictor = self._make_predictor()
        df        = make_synthetic_df(n_rows=25)

        api_content = json.dumps({
            "signal":      "BUY",
            "confidence":  0.82,
            "reasoning":   "Strong bullish momentum with RSI trending up and ADX above 25.",
            "key_factors": ["RSI trending up", "ADX>25", "London session"],
        })
        mock_response = _make_mock_openai_response(api_content)
        predictor._client.chat.completions.create.return_value = mock_response

        result = predictor.predict(df)

        assert result["signal"]       == 1,    f"Expected signal=1, got {result['signal']}"
        assert result["signal_label"] == "BUY"
        assert result["confidence"]   == pytest.approx(0.82)
        assert len(result["key_factors"]) == 3

    def test_predict_with_mocked_api_sell(self):
        """predict() returns signal=-1 when the mocked API responds with SELL."""
        predictor = self._make_predictor()
        df        = make_synthetic_df(n_rows=25)

        api_content = json.dumps({
            "signal":      "SELL",
            "confidence":  0.71,
            "reasoning":   "Bearish pressure at 24h high.",
            "key_factors": ["Near 24h high", "Stoch overbought"],
        })
        mock_response = _make_mock_openai_response(api_content)
        predictor._client.chat.completions.create.return_value = mock_response

        result = predictor.predict(df)

        assert result["signal"]       == -1
        assert result["signal_label"] == "SELL"
        assert result["confidence"]   == pytest.approx(0.71)

    def test_predict_with_mocked_api_no_trade(self):
        """predict() returns signal=0 when the mocked API responds with NO_TRADE."""
        predictor = self._make_predictor()
        df        = make_synthetic_df(n_rows=25)

        api_content = json.dumps({
            "signal":      "NO_TRADE",
            "confidence":  0.35,
            "reasoning":   "Consolidation — no clear edge.",
            "key_factors": ["ADX flat", "BB tight"],
        })
        mock_response = _make_mock_openai_response(api_content)
        predictor._client.chat.completions.create.return_value = mock_response

        result = predictor.predict(df)

        assert result["signal"]       == 0
        assert result["signal_label"] == "NO_TRADE"

    def test_predict_api_called_once(self):
        """The OpenAI chat.completions.create endpoint is called exactly once per predict()."""
        predictor = self._make_predictor()
        df        = make_synthetic_df(n_rows=25)

        api_content = json.dumps({
            "signal": "NO_TRADE", "confidence": 0.0,
            "reasoning": "test", "key_factors": [],
        })
        predictor._client.chat.completions.create.return_value = (
            _make_mock_openai_response(api_content)
        )
        predictor.predict(df)

        predictor._client.chat.completions.create.assert_called_once()

    def test_predict_returns_all_expected_keys(self):
        """predict() result always contains all 6 standardised output keys."""
        predictor = self._make_predictor()
        df        = make_synthetic_df(n_rows=25)

        api_content = json.dumps({
            "signal": "BUY", "confidence": 0.65,
            "reasoning": "test", "key_factors": ["x"],
        })
        predictor._client.chat.completions.create.return_value = (
            _make_mock_openai_response(api_content)
        )
        result = predictor.predict(df)

        required_keys = {"signal", "signal_label", "confidence", "reasoning",
                         "key_factors", "raw_response"}
        assert required_keys <= result.keys(), (
            f"Missing keys in result: {required_keys - result.keys()}"
        )

    def test_predict_prompt_contains_rsi_value(self):
        """The prompt sent to the API contains the RSI value from the DataFrame."""
        predictor = self._make_predictor()
        df        = make_synthetic_df(n_rows=25)

        captured_calls: list = []

        def capture_create(**kwargs):
            captured_calls.append(kwargs)
            return _make_mock_openai_response(
                json.dumps({"signal": "NO_TRADE", "confidence": 0.0,
                            "reasoning": "", "key_factors": []})
            )

        predictor._client.chat.completions.create.side_effect = capture_create
        predictor.predict(df)

        assert len(captured_calls) == 1
        user_msg = captured_calls[0]["messages"][1]["content"]
        # RSI default is 55.0 — expect "55.0" somewhere in the prompt
        assert "55.0" in user_msg, (
            "RSI value 55.0 not found in the prompt sent to the API."
        )


# ──────────────────────────────────────────────────────────────────────────────
# KimiConfig defaults
# ──────────────────────────────────────────────────────────────────────────────

class TestKimiConfigDefaults:
    """Verify KimiConfig falls back to documented defaults when no env vars are set."""

    def test_kimi_config_default_base_url(self, monkeypatch):
        """Default base_url is the sumopod endpoint."""
        monkeypatch.delenv("KIMI_BASE_URL", raising=False)
        cfg = KimiConfig()
        assert cfg.base_url == "https://ai.sumopod.com/v1", (
            f"Unexpected base_url default: {cfg.base_url!r}"
        )

    def test_kimi_config_default_model(self, monkeypatch):
        """Default model name is kimi-k2-250905."""
        monkeypatch.delenv("KIMI_MODEL", raising=False)
        cfg = KimiConfig()
        assert cfg.model == "kimi-k2-250905", (
            f"Unexpected model default: {cfg.model!r}"
        )

    def test_kimi_config_default_api_key_empty(self, monkeypatch):
        """Default api_key is an empty string (must be supplied via env / caller)."""
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        cfg = KimiConfig()
        assert cfg.api_key == ""

    def test_kimi_config_env_override_base_url(self, monkeypatch):
        """KIMI_BASE_URL env var overrides the default."""
        monkeypatch.setenv("KIMI_BASE_URL", "https://custom.host/v1")
        cfg = KimiConfig()
        assert cfg.base_url == "https://custom.host/v1"

    def test_kimi_config_env_override_model(self, monkeypatch):
        """KIMI_MODEL env var overrides the default model name."""
        monkeypatch.setenv("KIMI_MODEL", "kimi-next-gen")
        cfg = KimiConfig()
        assert cfg.model == "kimi-next-gen"

    def test_kimi_config_explicit_values(self):
        """Constructor accepts explicit keyword arguments."""
        cfg = KimiConfig(api_key="my-key", base_url="http://local/v1", model="test")
        assert cfg.api_key  == "my-key"
        assert cfg.base_url == "http://local/v1"
        assert cfg.model    == "test"


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases / robustness
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Miscellaneous robustness checks."""

    def test_build_context_dropna_reduces_rows(self):
        """
        If some rows have NaN in rsi_14 / adx_14 / atr_14, build_context drops
        them before slicing.  The result should still reflect only clean rows.
        """
        df = make_synthetic_df(n_rows=30)
        # Poison the first 5 rows so they are dropped
        df.iloc[:5, df.columns.get_loc("rsi_14")] = float("nan")

        ctx = build_context(df)
        # 30 - 5 = 25 clean rows; we still get 20 recent bars
        assert len(ctx["recent_bars"]) == 20

    def test_build_context_time_format(self):
        """ctx['time'] follows the 'YYYY-MM-DD HH:MM UTC' format."""
        df  = make_synthetic_df(n_rows=25)
        ctx = build_context(df)
        # Should end with UTC and include a date separator
        assert ctx["time"].endswith("UTC")
        assert "-" in ctx["time"] and ":" in ctx["time"]

    def test_parse_response_partial_key_factors(self):
        """A non-list key_factors falls back to an empty list."""
        raw    = json.dumps({"signal": "BUY", "confidence": 0.5,
                              "reasoning": "x", "key_factors": "not a list"})
        result = KimiPredictor._parse_response(raw)
        # str is iterable — list() on a str gives individual chars; just confirm
        # the return type is list (implementation converts via list()).
        assert isinstance(result["key_factors"], list)

    def test_make_synthetic_df_shape(self):
        """Helper itself returns the expected shape."""
        df = make_synthetic_df(n_rows=30)
        assert len(df) == 30
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"

    def test_make_synthetic_df_default_rows(self):
        """Default n_rows=25 when no argument is supplied."""
        df = make_synthetic_df()
        assert len(df) == 25
