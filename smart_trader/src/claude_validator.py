"""
Claude Validator — builds a compact ~600c prompt from pre-analyzed data
and calls `claude -p` for pure reasoning validation (no MCP tool calls).
Expected response time: 20-30 seconds.
"""
import json
import subprocess
import shlex
from typing import Optional
from loguru import logger


_PROMPT_TEMPLATE = """\
XAUUSD TRADE VALIDATION — all data already provided, NO tool calls needed.
Respond with ONLY the JSON decision object, nothing else.

YOU ARE AN ELITE INSTITUTIONAL TRADER. Think like:
- Soros: high conviction when BOS+OB+FVG align (reflexive confluence); CHoCH = loop broken
- PTJ/Kovner: defense first — capital preservation paramount; SL at invalidation, never widen
- Druckenmiller: asymmetric bets — it's not win rate, it's profit when right vs loss when wrong
- Simons: systematic — trust the signals, no emotional override, process > outcome
- Lipschutz: patience — skip weak setups, A+ confluence only; sitting out IS a position
- Dennis: trade with structure — BOS = trend continuation, CHoCH = potential reversal
- Kotegawa: discipline — if the level fails, the trade is wrong, zero exceptions

MARKET SNAPSHOT:
Price: {price} | Spread: {spread}pt | ATR: {atr:.1f}pt
Session: {session} | H1: {h1_structure} | EMA(50): {ema_trend}
RSI(14): {rsi:.1f} | P/D: {pd_zone}
Regime: {regime} ({regime_confidence:.0%} confidence){choch_note}

PROPOSED TRADE:
Direction: {direction}
Zone: {zone_type} @ {zone_level}  (distance: {distance:.1f}pt)
M15 Confirm: {m15_conf}
OTE: {ote}
Signals ({signal_count}): {signals}
SL: {sl:.2f} | TP: {tp:.2f} | RR: {rr:.1f}
Lot: {lot:.2f} | Est.margin: ~$50

RECENT CONTEXT:
{context}

RULES (mandatory):
- RR ~1.7 is NORMAL for this system (3x ATR SL, 5x ATR TP). Do NOT reject for low RR.
- RSI 60-80 is normal in XAUUSD trends. Only reject RSI above 85 or below 15.
- You are the PRIMARY decision maker. Evaluate signal quality and confluence freely.
- REJECT only if: SL/TP values are clearly wrong (e.g. SL above entry for LONG).
- CONFIDENCE SCALE: 0.85+ = A+ Soros-level confluence (multiple aligned signals)
  0.70-0.84 = solid setup worth trading. Below 0.70 = skip (Lipschutz patience rule).

Respond ONLY with valid JSON (no markdown):
{{"decision":"LONG or SHORT or NO_TRADE","confidence":0.0-1.0,"reason":"<20 words","sl":{sl:.2f},"tp":{tp:.2f}}}
"""


def build_prompt(setup: dict) -> str:
    """Build compact validation prompt from pre-analyzed setup dict."""
    ote_str = "yes ({:.1f}-{:.1f})".format(*setup["ote"]) if setup.get("ote") else "no"
    signals = ", ".join(setup.get("signals", [])) or "none"
    context = setup.get("context", "").strip() or "No prior cycles."

    # Truncate context if too long
    if len(context) > 300:
        lines = context.split("\n")
        context = "\n".join(lines[-4:])  # Keep last 4 lines

    # Regime fields (optional — absent for legacy callers)
    regime = setup.get("regime", "UNKNOWN")
    regime_confidence = setup.get("regime_confidence", 0.0)
    has_choch = setup.get("has_choch", False)
    choch_note = "\n** CHoCH DETECTED — potential reversal, counter-trend valid **" if has_choch else ""

    return _PROMPT_TEMPLATE.format(
        price       = setup["price"],
        spread      = setup.get("spread", 0),
        atr         = setup.get("atr", 0),
        session     = setup.get("session", "UNKNOWN"),
        h1_structure= setup.get("h1_structure", "UNKNOWN"),
        ema_trend   = setup.get("ema_trend", "NEUTRAL"),
        rsi         = setup.get("rsi", 50),
        pd_zone     = setup.get("pd_zone", "EQUILIBRIUM"),
        regime      = regime,
        regime_confidence = regime_confidence,
        choch_note  = choch_note,
        direction   = setup["direction"],
        zone_type   = setup.get("zone_type", "ZONE"),
        zone_level  = setup.get("zone_level", setup["price"]),
        distance    = setup.get("distance_pts", 0),
        m15_conf    = setup.get("m15_conf") or "none",
        ote         = ote_str,
        signal_count= setup.get("signal_count", 0),
        signals     = signals,
        sl          = setup["sl"],
        tp          = setup["tp"],
        rr          = setup.get("rr", 0),
        lot         = setup.get("lot", 0.01),
        context     = context,
    )


def _empty_metrics() -> dict:
    """Return empty metrics dict (used on failure paths)."""
    return {
        "latency_ms": 0, "prompt_chars": 0, "response_chars": 0,
        "est_input_tokens": 0, "est_output_tokens": 0,
    }


def call_claude(
    prompt: str, claude_cmd: str, model: str, timeout_sec: int,
) -> tuple[Optional[dict], dict]:
    """
    Call `claude -p <prompt>` and parse JSON response.
    Uses direct stdin input (no temp file) and clean env (no CLAUDECODE).
    Returns (parsed_dict, metrics) tuple.
    parsed_dict: dict with keys decision/action, confidence, reason, sl, tp — or None on failure.
    metrics: {"latency_ms", "prompt_chars", "response_chars", "est_input_tokens", "est_output_tokens"}
    """
    import os
    import time as _time

    metrics = _empty_metrics()
    metrics["prompt_chars"] = len(prompt)
    metrics["est_input_tokens"] = len(prompt) // 4

    # Build clean env — remove Claude Code session markers to prevent
    # "cannot be launched inside another Claude Code session" error
    clean_env = os.environ.copy()
    clean_env.pop("CLAUDECODE", None)
    clean_env.pop("CLAUDE_CODE_SESSION", None)

    # Use -p (print mode), --max-turns 1 (single response, no tool loops)
    cmd = (
        f'"{claude_cmd}" --model {model} '
        f'--dangerously-skip-permissions --max-turns 1 -p'
    )

    max_retries = 2
    result = None
    t_start = _time.time()

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                encoding="utf-8",
                errors="replace",
                env=clean_env,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"Claude validation timed out after {timeout_sec}s")
            break
        except Exception as e:
            logger.error(f"Claude subprocess error: {e}")
            break

        if result and result.stdout.strip():
            break  # success

        if attempt < max_retries - 1:
            logger.debug(f"Claude empty response, retrying ({attempt+1}/{max_retries})...")
            _time.sleep(3)

    metrics["latency_ms"] = round((_time.time() - t_start) * 1000)

    if result is None:
        return None, metrics

    stdout = result.stdout.strip()
    if not stdout:
        logger.warning(f"Claude returned empty output. stderr: {result.stderr[:200]}")
        return None, metrics

    metrics["response_chars"] = len(stdout)
    metrics["est_output_tokens"] = len(stdout) // 4

    logger.debug(f"Claude raw response ({len(stdout)}c): {stdout[:500]}")

    # Extract JSON from response (handle extra text/markdown)
    json_str = _extract_json(stdout)
    if not json_str:
        logger.warning(f"No JSON in Claude response: {stdout[:300]}")
        return None, metrics

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e} | raw: {json_str[:200]}")
        return None, metrics

    # If top-level has "decision" or "action" key, use directly
    if "decision" in parsed or "action" in parsed:
        try:
            _validate_response(parsed)
            return parsed, metrics
        except ValueError as e:
            logger.warning(f"Invalid response: {e}")
            return None, metrics

    # Otherwise Claude returned nested JSON (used MCP tools) — search for decision obj
    decision_obj = _find_decision_obj(parsed)
    if decision_obj:
        logger.debug(f"Found decision in nested JSON: {decision_obj}")
        try:
            _validate_response(decision_obj)
            return decision_obj, metrics
        except ValueError as e:
            logger.warning(f"Invalid nested decision: {e}")

    # Last resort: try to extract from any text in the response
    logger.warning(f"No decision/action key found. JSON keys: {list(parsed.keys())[:8]}")
    return None, metrics


def _find_decision_obj(d: dict) -> Optional[dict]:
    """Recursively search for a dict containing 'decision' or 'action' key."""
    if isinstance(d, dict):
        if "decision" in d or "action" in d:
            return d
        for v in d.values():
            result = _find_decision_obj(v)
            if result:
                return result
    elif isinstance(d, list):
        for item in d:
            result = _find_decision_obj(item)
            if result:
                return result
    return None


def _extract_json(text: str) -> Optional[str]:
    """Extract the outermost {...} block from text."""
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


def _validate_response(r: dict) -> None:
    """Validate required keys exist and have correct types."""
    if "decision" not in r and "action" not in r:
        raise ValueError("Missing 'decision' or 'action' key")
    if "action" in r:
        _validate_exit_response(r)
        return
    if r["decision"] not in ("LONG", "SHORT", "NO_TRADE"):
        raise ValueError(f"Invalid decision: {r['decision']}")
    r.setdefault("confidence", 0.5)
    r.setdefault("reason", "")
    r.setdefault("sl", 0.0)
    r.setdefault("tp", 0.0)
    r["confidence"] = float(r["confidence"])
    r["sl"]         = float(r["sl"])
    r["tp"]         = float(r["tp"])


# ── Exit Review ──────────────────────────────────────────────────────────────

_EXIT_PROMPT_TEMPLATE = """\
XAUUSD SMART EXIT ANALYSIS — all data provided, NO tool calls needed.
You are a PROFIT OPTIMIZER. Maximize profit from this open position.
Respond with ONLY the JSON object, nothing else.

THINK LIKE THE MASTERS:
- Dennis/Lipschutz: LET WINNERS RUN — never exit early if trend intact
- Soros: exit only when the reflexive feedback loop BREAKS (CHoCH, momentum reversal)
- Druckenmiller: protect asymmetric gains — lock profits when structure weakens
- Kotegawa: when exit signal is clear, ACT — no hesitation, no "hold a little longer"

MARKET NOW:
Price: {price} | ATR: {atr:.1f}pt | Session: {session} | EMA(50): {ema_trend}
RSI(14): {rsi:.1f} | P/D: {pd_zone} | Regime: {regime}
Nearby signals: {nearby_signals}

OPEN POSITION:
Direction: {direction} | Entry: {entry:.2f} | P/L: {pnl_pts:+.1f}pt (${pnl_usd:+.2f})
SL: {sl:.2f} | TP: {tp:.2f} (remaining: {tp_remaining:.0f}pt) | Duration: {duration_min:.0f}min
Stage: {stage}

RULES (mandatory):
- Default is HOLD. You need a STRONG reason to override.
- TAKE_PROFIT only when profit is above 15pt AND momentum clearly reversing:
  * LONG: RSI dropping from above 70, or price broke below EMA(50)
  * SHORT: RSI rising from below 30, or price broke above EMA(50)
- TIGHTEN when profit is above 15pt and momentum weakening but not reversing yet.
  * Move SL to lock at least 50% of current profit.
- If UNDERWATER (P/L negative): ALWAYS HOLD. Never close at a loss.
- If profit is below 15pt: ALWAYS HOLD. Let the trade develop.
- If EMA trend matches direction and RSI is not extreme: HOLD.

Respond ONLY with valid JSON (no markdown):
{{"action":"HOLD","reason":"<15 words"}}
OR {{"action":"TAKE_PROFIT","reason":"<15 words"}}
OR {{"action":"TIGHTEN","new_sl":{tighten_sl:.2f},"reason":"<15 words"}}
"""


def build_exit_prompt(pos_data: dict) -> str:
    """Build compact exit review prompt."""
    # Ensure regime field exists (optional for backward compat)
    data = dict(pos_data)
    data.setdefault("regime", "UNKNOWN")
    return _EXIT_PROMPT_TEMPLATE.format(**data)


def _validate_exit_response(r: dict) -> None:
    """Validate exit review response."""
    if "action" not in r:
        raise ValueError("Missing 'action' key")
    # Accept CLOSE as alias for TAKE_PROFIT (Claude sometimes uses CLOSE)
    if r["action"] == "CLOSE":
        r["action"] = "TAKE_PROFIT"
    if r["action"] not in ("HOLD", "TAKE_PROFIT", "TIGHTEN"):
        raise ValueError(f"Invalid action: {r['action']}")
    r.setdefault("reason", "")
    if r["action"] == "TIGHTEN":
        r.setdefault("new_sl", 0.0)
        r["new_sl"] = float(r["new_sl"])


def review_exit(pos_data: dict, cfg: dict) -> tuple[Optional[dict], dict]:
    """
    Ask Claude to analyze an open position for optimal exit timing.
    Returns (response_dict, metrics) tuple.
    response_dict has action: HOLD / TAKE_PROFIT / TIGHTEN, or None on failure.
    """
    prompt = build_exit_prompt(pos_data)
    logger.debug(f"Exit review prompt ({len(prompt)}c):\n{prompt}")

    claude_cmd  = cfg.get("cmd", "claude")
    model       = cfg.get("model", "claude-opus-4-6")
    timeout_sec = cfg.get("timeout_sec", 60)

    response, metrics = call_claude(prompt, claude_cmd, model, timeout_sec)
    if not response:
        return None, metrics

    # Re-parse since exit responses have different structure
    if "action" in response:
        try:
            _validate_exit_response(response)
            return response, metrics
        except ValueError as e:
            logger.warning(f"Invalid exit response: {e}")
            return None, metrics

    # Try extracting from nested
    if "decision" in response:
        # Map entry-style response to exit format
        return {"action": "HOLD", "reason": response.get("reason", "")}, metrics

    return None, metrics


def validate(setup: dict, cfg: dict) -> tuple[Optional[dict], dict]:
    """
    High-level entry point.
    setup: pre-analyzed trade setup dict (see build_prompt fields)
    cfg:   claude section from config.yaml
    Returns (response_dict, metrics) tuple.
    """
    prompt = build_prompt(setup)
    logger.debug(f"Claude prompt ({len(prompt)}c):\n{prompt}")

    claude_cmd  = cfg.get("cmd", "claude")
    model       = cfg.get("model", "claude-opus-4-6")
    timeout_sec = cfg.get("timeout_sec", 60)

    response, metrics = call_claude(prompt, claude_cmd, model, timeout_sec)
    if response:
        latency_s = metrics["latency_ms"] / 1000
        est_tokens = metrics["est_input_tokens"] + metrics["est_output_tokens"]
        logger.info(
            f"Claude decision: {response['decision']} "
            f"(conf={response['confidence']:.2f}) — {response['reason']} "
            f"| {latency_s:.1f}s | ~{est_tokens} tokens"
        )
    return response, metrics
