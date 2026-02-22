"""
Tests for pure Python parsing/validation logic in src/claude_validator.py.
Does NOT test call_claude() (requires subprocess/MT5).
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from claude_validator import (
    build_prompt,
    build_exit_prompt,
    _extract_json,
    _validate_response,
    _validate_exit_response,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

MINIMAL_SETUP = {
    "price": 2900.0,
    "spread": 1.5,
    "atr": 8.0,
    "session": "LONDON",
    "h1_structure": "BOS_BULL",
    "ema_trend": "BULLISH",
    "rsi": 55.0,
    "pd_zone": "DISCOUNT",
    "direction": "LONG",
    "zone_type": "BULL_OB",
    "zone_level": 2898.0,
    "distance_pts": 2.0,
    "m15_conf": "M15_ENGULF_BULL",
    "ote": (2895.0, 2897.0),
    "signal_count": 3,
    "signals": ["BOS", "OB", "M15"],
    "sl": 2890.0,
    "tp": 2940.0,
    "rr": 1.7,
    "lot": 0.01,
    "context": "prior cycles",
}

MINIMAL_POS_DATA = {
    "price": 2905.0,
    "atr": 8.0,
    "session": "LONDON",
    "ema_trend": "BULLISH",
    "rsi": 62.0,
    "pd_zone": "PREMIUM",
    "nearby_signals": "BOS_BULL",
    "direction": "LONG",
    "entry": 2900.0,
    "pnl_pts": 5.0,
    "pnl_usd": 5.0,
    "sl": 2890.0,
    "tp": 2940.0,
    "tp_remaining": 35.0,
    "duration_min": 30.0,
    "stage": "BE",
    "tighten_sl": 2897.0,
}


# ── build_prompt ───────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_returns_nonempty_string(self):
        result = build_prompt(MINIMAL_SETUP)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_direction(self):
        result = build_prompt(MINIMAL_SETUP)
        assert "LONG" in result

    def test_contains_price(self):
        result = build_prompt(MINIMAL_SETUP)
        assert "2900" in result

    def test_contains_session(self):
        result = build_prompt(MINIMAL_SETUP)
        assert "LONDON" in result

    def test_truncates_long_context_to_last_4_lines(self):
        # Each line is 25+ chars so 20 lines >> 300 char threshold.
        # Zero-pad index and add padding to avoid substring collisions.
        long_lines = [f"line {i:02d}: {'x' * 15}" for i in range(20)]
        ctx = "\n".join(long_lines)
        assert len(ctx) > 300, "Precondition: context must exceed 300 chars"
        setup = {**MINIMAL_SETUP, "context": ctx}
        result = build_prompt(setup)
        # Last 4 lines (indices 16-19) must be present
        assert "line 19:" in result
        assert "line 18:" in result
        assert "line 17:" in result
        assert "line 16:" in result
        # Early lines (indices 00-15) must NOT appear
        assert "line 00:" not in result
        assert "line 01:" not in result
        assert "line 10:" not in result
        assert "line 15:" not in result

    def test_context_under_300_chars_not_truncated(self):
        ctx = "short context"
        setup = {**MINIMAL_SETUP, "context": ctx}
        result = build_prompt(setup)
        assert "short context" in result

    def test_context_exactly_300_chars_not_truncated(self):
        ctx = "x" * 300
        setup = {**MINIMAL_SETUP, "context": ctx}
        result = build_prompt(setup)
        assert ctx in result

    def test_context_over_300_chars_truncated(self):
        lines = [f"L{i:02d}:" + "a" * 20 for i in range(15)]
        ctx = "\n".join(lines)
        assert len(ctx) > 300
        setup = {**MINIMAL_SETUP, "context": ctx}
        result = build_prompt(setup)
        # Only last 4 lines kept: L11, L12, L13, L14
        assert "L14:" in result
        assert "L13:" in result
        assert "L12:" in result
        assert "L11:" in result
        assert "L00:" not in result

    def test_missing_ote_renders_no(self):
        setup = {**MINIMAL_SETUP, "ote": None}
        result = build_prompt(setup)
        assert "OTE: no" in result

    def test_missing_m15_conf_renders_none(self):
        setup = {**MINIMAL_SETUP, "m15_conf": None}
        result = build_prompt(setup)
        assert "M15 Confirm: none" in result

    def test_empty_signals_renders_none(self):
        setup = {**MINIMAL_SETUP, "signals": []}
        result = build_prompt(setup)
        assert "none" in result

    def test_missing_optional_fields_do_not_raise(self):
        minimal = {
            "price": 2900.0,
            "direction": "SHORT",
            "sl": 2910.0,
            "tp": 2860.0,
        }
        result = build_prompt(minimal)
        assert isinstance(result, str)
        assert "SHORT" in result

    def test_signals_list_joined(self):
        setup = {**MINIMAL_SETUP, "signals": ["BOS", "OB", "M15"]}
        result = build_prompt(setup)
        assert "BOS, OB, M15" in result

    def test_ote_tuple_formatted(self):
        setup = {**MINIMAL_SETUP, "ote": (2895.0, 2897.0)}
        result = build_prompt(setup)
        assert "2895.0" in result
        assert "2897.0" in result


# ── _extract_json ──────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_clean_json_string_returned_as_is(self):
        raw = '{"decision":"LONG","confidence":0.8}'
        assert _extract_json(raw) == raw

    def test_json_wrapped_in_markdown_code_fence(self):
        raw = '```json\n{"decision":"LONG","confidence":0.8}\n```'
        result = _extract_json(raw)
        assert result == '{"decision":"LONG","confidence":0.8}'

    def test_text_before_and_after_json(self):
        raw = 'Here is the result: {"decision":"NO_TRADE"} and that is all.'
        result = _extract_json(raw)
        assert result == '{"decision":"NO_TRADE"}'

    def test_no_braces_returns_none(self):
        assert _extract_json("no braces here at all") is None

    def test_empty_string_returns_none(self):
        assert _extract_json("") is None

    def test_only_open_brace_returns_none(self):
        # end == start would be end <= start → None
        assert _extract_json("{no closing brace") is None

    def test_brace_end_before_start_returns_none(self):
        # Only closing brace before opening brace — rfind gives wrong position
        result = _extract_json("}text{")
        # start=5, end=0 → end <= start → None
        assert result is None

    def test_nested_json_extracts_outermost(self):
        raw = '{"outer":{"inner":"val"},"decision":"SHORT"}'
        result = _extract_json(raw)
        assert result == raw

    def test_multiline_json(self):
        raw = '{\n  "decision": "LONG",\n  "confidence": 0.75\n}'
        result = _extract_json(raw)
        assert result == raw

    def test_multiple_objects_returns_outermost_span(self):
        # finds first { and last } — spans both objects
        raw = '{"a":1} {"b":2}'
        result = _extract_json(raw)
        assert result == '{"a":1} {"b":2}'


# ── _validate_response ─────────────────────────────────────────────────────────

class TestValidateResponse:
    def test_valid_long_response_no_error(self):
        r = {"decision": "LONG", "confidence": 0.8, "reason": "good setup", "sl": 2890.0, "tp": 2940.0}
        _validate_response(r)  # should not raise

    def test_valid_short_response_no_error(self):
        r = {"decision": "SHORT", "confidence": 0.7, "reason": "bearish", "sl": 2910.0, "tp": 2860.0}
        _validate_response(r)

    def test_valid_no_trade_response_no_error(self):
        r = {"decision": "NO_TRADE", "confidence": 0.3, "reason": "weak", "sl": 0.0, "tp": 0.0}
        _validate_response(r)

    def test_missing_decision_key_raises_value_error(self):
        r = {"confidence": 0.8, "reason": "ok", "sl": 2890.0, "tp": 2940.0}
        with pytest.raises(ValueError, match="Missing"):
            _validate_response(r)

    def test_invalid_decision_value_raises_value_error(self):
        r = {"decision": "MAYBE", "confidence": 0.5, "sl": 0.0, "tp": 0.0}
        with pytest.raises(ValueError, match="Invalid decision"):
            _validate_response(r)

    def test_missing_confidence_defaults_to_0_5(self):
        r = {"decision": "LONG", "sl": 2890.0, "tp": 2940.0}
        _validate_response(r)
        assert r["confidence"] == 0.5

    def test_confidence_coerced_to_float(self):
        r = {"decision": "LONG", "confidence": "0.75", "sl": 2890.0, "tp": 2940.0}
        _validate_response(r)
        assert isinstance(r["confidence"], float)
        assert r["confidence"] == 0.75

    def test_sl_coerced_to_float(self):
        r = {"decision": "LONG", "confidence": 0.8, "sl": "2890", "tp": 2940.0}
        _validate_response(r)
        assert isinstance(r["sl"], float)

    def test_tp_coerced_to_float(self):
        r = {"decision": "LONG", "confidence": 0.8, "sl": 2890.0, "tp": "2940"}
        _validate_response(r)
        assert isinstance(r["tp"], float)

    def test_missing_sl_defaults_to_0(self):
        r = {"decision": "NO_TRADE", "confidence": 0.3}
        _validate_response(r)
        assert r["sl"] == 0.0

    def test_missing_tp_defaults_to_0(self):
        r = {"decision": "NO_TRADE", "confidence": 0.3}
        _validate_response(r)
        assert r["tp"] == 0.0

    def test_missing_reason_defaults_to_empty_string(self):
        r = {"decision": "LONG", "confidence": 0.8, "sl": 2890.0, "tp": 2940.0}
        _validate_response(r)
        assert r["reason"] == ""

    def test_response_with_action_key_delegates_to_exit_validator(self):
        # If "action" key present, _validate_exit_response is called instead
        r = {"action": "HOLD", "reason": "steady"}
        _validate_response(r)  # should not raise


# ── _validate_exit_response ────────────────────────────────────────────────────

class TestValidateExitResponse:
    def test_hold_is_valid(self):
        r = {"action": "HOLD", "reason": "no momentum shift"}
        _validate_exit_response(r)

    def test_take_profit_is_valid(self):
        r = {"action": "TAKE_PROFIT", "reason": "RSI reversing"}
        _validate_exit_response(r)

    def test_tighten_with_new_sl_is_valid(self):
        r = {"action": "TIGHTEN", "new_sl": 2895.0, "reason": "lock gains"}
        _validate_exit_response(r)
        assert isinstance(r["new_sl"], float)

    def test_close_mapped_to_take_profit(self):
        r = {"action": "CLOSE", "reason": "manual close"}
        _validate_exit_response(r)
        assert r["action"] == "TAKE_PROFIT"

    def test_invalid_action_raises_value_error(self):
        r = {"action": "REVERSE", "reason": "flip it"}
        with pytest.raises(ValueError, match="Invalid action"):
            _validate_exit_response(r)

    def test_missing_action_raises_value_error(self):
        r = {"reason": "no action key"}
        with pytest.raises(ValueError, match="Missing 'action' key"):
            _validate_exit_response(r)

    def test_tighten_missing_new_sl_defaults_to_0(self):
        r = {"action": "TIGHTEN", "reason": "partial lock"}
        _validate_exit_response(r)
        assert r["new_sl"] == 0.0

    def test_tighten_new_sl_coerced_to_float(self):
        r = {"action": "TIGHTEN", "new_sl": "2895", "reason": "lock"}
        _validate_exit_response(r)
        assert isinstance(r["new_sl"], float)
        assert r["new_sl"] == 2895.0

    def test_missing_reason_defaults_to_empty_string(self):
        r = {"action": "HOLD"}
        _validate_exit_response(r)
        assert r["reason"] == ""

    def test_hold_does_not_require_new_sl(self):
        r = {"action": "HOLD"}
        _validate_exit_response(r)
        assert "new_sl" not in r

    def test_take_profit_does_not_require_new_sl(self):
        r = {"action": "TAKE_PROFIT"}
        _validate_exit_response(r)
        assert "new_sl" not in r


# ── build_exit_prompt ──────────────────────────────────────────────────────────

class TestBuildExitPrompt:
    def test_returns_nonempty_string(self):
        result = build_exit_prompt(MINIMAL_POS_DATA)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_direction(self):
        result = build_exit_prompt(MINIMAL_POS_DATA)
        assert "LONG" in result

    def test_contains_entry_price(self):
        result = build_exit_prompt(MINIMAL_POS_DATA)
        assert "2900.00" in result

    def test_contains_session(self):
        result = build_exit_prompt(MINIMAL_POS_DATA)
        assert "LONDON" in result

    def test_contains_action_options(self):
        result = build_exit_prompt(MINIMAL_POS_DATA)
        assert "HOLD" in result
        assert "TAKE_PROFIT" in result
        assert "TIGHTEN" in result

    def test_short_direction_in_prompt(self):
        pos_data = {**MINIMAL_POS_DATA, "direction": "SHORT", "entry": 2920.0}
        result = build_exit_prompt(pos_data)
        assert "SHORT" in result
        assert "2920.00" in result

    def test_pnl_formatted(self):
        result = build_exit_prompt(MINIMAL_POS_DATA)
        assert "+5.0pt" in result or "5.0pt" in result

    def test_all_required_fields_render(self):
        result = build_exit_prompt(MINIMAL_POS_DATA)
        assert "BULLISH" in result   # ema_trend
        assert "62.0" in result      # rsi
        assert "PREMIUM" in result   # pd_zone
        assert "BE" in result        # stage
