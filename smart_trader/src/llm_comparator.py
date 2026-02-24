"""
LLM Comparator — compare multiple LLMs against Claude Opus 4.6 benchmark.

Read-only observation layer: fires in background threads, sends results
to Telegram. Never blocks the main trading loop or interferes with execution.

Models compared:
  1. Claude Opus 4.6 — benchmark (response reused from main flow)
  2. Claude Sonnet 4.6 — via CLI subprocess
  3. Kimi K-2 — via OpenAI-compatible API
  4. GLM 4-7 — via OpenAI-compatible API
  5. DeepSeek V3-2 — via OpenAI-compatible API
"""
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from loguru import logger

import telegram_notifier as tg

# ── LLM Model Registry ──────────────────────────────────────────────────────

LLM_MODELS = {
    "Sonnet 4.6": {"type": "cli", "model": "claude-sonnet-4-6"},
    "Kimi K-2":   {"type": "api", "model": "kimi-k2"},
    "GLM 4-7":    {"type": "api", "model": "glm-4-7"},
    "DeepSeek V3-2": {"type": "api", "model": "deepseek-v3-2"},
}

# Defaults — overridden by config.yaml if present
_config = {
    "enabled": True,
    "api_base": "https://ai.sumopod.com/v1",
    "api_key": "sk-qg78Jy5EYqEdz9IsJLlCDw",
    "timeout_sec": 60,
}


def configure(cfg: dict) -> None:
    """Update comparator config from config.yaml llm_comparison section."""
    _config.update(cfg)


# ── API Callers ──────────────────────────────────────────────────────────────

def _call_openai_compatible(prompt: str, model: str, timeout: int = 60) -> tuple[Optional[dict], str]:
    """Call OpenAI-compatible API. Returns (parsed_dict, error_reason)."""
    try:
        resp = requests.post(
            f"{_config['api_base']}/chat/completions",
            headers={
                "Authorization": f"Bearer {_config['api_key']}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 500,
            },
            timeout=timeout,
        )
        if not resp.ok:
            body = resp.text[:200]
            logger.debug(f"LLM API error ({model}): {resp.status_code} {body}")
            # Detect specific error types
            if "ExceededBudget" in body or "over budget" in body.lower():
                return None, "BUDGET EXCEEDED"
            if resp.status_code == 401:
                return None, "AUTH ERROR"
            if resp.status_code == 429:
                return None, "RATE LIMITED"
            return None, f"HTTP {resp.status_code}"

        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        parsed = _extract_and_parse_json(content)
        if parsed is None:
            return None, "JSON PARSE FAIL"
        return parsed, ""
    except requests.Timeout:
        logger.debug(f"LLM API timeout ({model})")
        return None, "TIMEOUT"
    except Exception as e:
        logger.debug(f"LLM API call failed ({model}): {e}")
        return None, f"ERROR"


def _call_sonnet_cli(prompt: str, claude_cmd: str, timeout: int = 60) -> tuple[Optional[dict], str]:
    """Call Claude Sonnet via CLI subprocess. Returns (parsed_dict, error_reason)."""
    try:
        clean_env = os.environ.copy()
        clean_env.pop("CLAUDECODE", None)
        clean_env.pop("CLAUDE_CODE_SESSION", None)

        cmd = (
            f'"{claude_cmd}" --model claude-sonnet-4-6 '
            f'--dangerously-skip-permissions --max-turns 1 -p'
        )

        result = subprocess.run(
            cmd,
            shell=True,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=clean_env,
        )

        stdout = result.stdout.strip()
        if not stdout:
            return None, "EMPTY RESPONSE"

        parsed = _extract_and_parse_json(stdout)
        if parsed is None:
            return None, "JSON PARSE FAIL"
        return parsed, ""
    except subprocess.TimeoutExpired:
        logger.debug("Sonnet CLI timed out")
        return None, "TIMEOUT"
    except Exception as e:
        logger.debug(f"Sonnet CLI error: {e}")
        return None, "ERROR"


def _extract_and_parse_json(text: str) -> Optional[dict]:
    """Extract outermost {...} from text and parse as JSON."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# ── Scoring ──────────────────────────────────────────────────────────────────

def _keyword_set(text: str) -> set:
    """Extract lowercase keywords from a reason string."""
    if not text:
        return set()
    # Remove common stop words, keep meaningful tokens
    words = re.findall(r'[a-zA-Z_]{3,}', text.lower())
    stop = {"the", "and", "for", "with", "not", "from", "has", "was", "are", "but", "this", "that"}
    return set(words) - stop


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _score_entry(benchmark: dict, candidate: dict) -> dict:
    """Score entry candidate vs Opus benchmark. Returns dict with score 0-100."""
    score = 0
    notes = []

    # Decision match (40 pts)
    b_dec = benchmark.get("decision", "")
    c_dec = candidate.get("decision", "")
    decision_match = b_dec == c_dec
    if decision_match:
        score += 40
        notes.append("decision=match")
    else:
        notes.append(f"decision=MISMATCH({c_dec})")

    # Confidence proximity (30 pts)
    b_conf = float(benchmark.get("confidence", 0.5))
    c_conf = float(candidate.get("confidence", 0.5))
    conf_delta = abs(b_conf - c_conf)
    conf_pts = max(0, 30 - conf_delta * 200)  # 0.15 gap = 0 pts
    score += conf_pts

    # Reason keyword overlap (30 pts)
    b_kw = _keyword_set(benchmark.get("reason", ""))
    c_kw = _keyword_set(candidate.get("reason", ""))
    sim = _jaccard(b_kw, c_kw)
    reason_pts = sim * 30
    score += reason_pts

    return {
        "score": round(min(score, 100)),
        "decision_match": decision_match,
        "conf_delta": round(conf_delta, 3),
        "reason_sim": round(sim, 2),
        "notes": " | ".join(notes),
    }


def _score_exit(benchmark: dict, candidate: dict) -> dict:
    """Score exit candidate vs Opus benchmark. Returns dict with score 0-100."""
    score = 0
    notes = []

    # Action match (50 pts)
    b_act = benchmark.get("action", "HOLD")
    c_act = candidate.get("action", "HOLD")
    # Normalize CLOSE → TAKE_PROFIT
    if c_act == "CLOSE":
        c_act = "TAKE_PROFIT"
    action_match = b_act == c_act
    if action_match:
        score += 50
        notes.append("action=match")
    else:
        notes.append(f"action=MISMATCH({c_act})")

    # SL proximity for TIGHTEN (20 pts)
    if b_act == "TIGHTEN" and c_act == "TIGHTEN":
        b_sl = float(benchmark.get("new_sl", 0))
        c_sl = float(candidate.get("new_sl", 0))
        if b_sl > 0 and c_sl > 0:
            sl_delta = abs(b_sl - c_sl)
            sl_pts = max(0, 20 - sl_delta * 2)  # 10pt gap = 0
            score += sl_pts
        else:
            score += 10  # partial credit
    elif action_match:
        score += 20  # full credit if both agree and not TIGHTEN

    # Reason keyword overlap (30 pts)
    b_kw = _keyword_set(benchmark.get("reason", ""))
    c_kw = _keyword_set(candidate.get("reason", ""))
    sim = _jaccard(b_kw, c_kw)
    reason_pts = sim * 30
    score += reason_pts

    return {
        "score": round(min(score, 100)),
        "action_match": action_match,
        "reason_sim": round(sim, 2),
        "notes": " | ".join(notes),
    }


# ── Star rating for reason similarity ────────────────────────────────────────

def _score_bar(score: int) -> str:
    """Visual score bar: 85 → ████████░░"""
    filled = round(score / 10)
    return "█" * filled + "░" * (10 - filled)


def _score_medal(score: int) -> str:
    """Medal emoji based on score."""
    if score >= 80:
        return "🥇"
    if score >= 60:
        return "🥈"
    if score >= 40:
        return "🥉"
    return "💤"


def _dec_emoji(match: bool) -> str:
    return "✅" if match else "❌"


def _latency_str(secs: float) -> str:
    if secs < 10:
        return f"{secs:.1f}s"
    return f"{secs:.0f}s"


# ── Telegram Formatting ─────────────────────────────────────────────────────

def _format_entry_comparison(setup: dict, opus: dict, results: list) -> str:
    """Format entry comparison as HTML for Telegram."""
    direction = setup.get("direction", "?")
    dir_emoji = "🟢" if direction == "LONG" else "🔴"
    zone_type = setup.get("zone_type", "?")
    signals = "+".join(setup.get("signals", [])) or "none"
    price = setup.get("price", 0)
    sl = setup.get("sl", 0)
    tp = setup.get("tp", 0)

    opus_dec = opus.get("decision", "?")
    opus_conf = float(opus.get("confidence", 0))
    opus_reason = opus.get("reason", "")

    # Sort results by score descending
    sorted_results = sorted(
        results,
        key=lambda x: x.get("scoring", {}).get("score", 0),
        reverse=True,
    )

    lines = [
        f"🔬 <b>LLM COMPARISON — ENTRY</b>",
        "",
        f"📊 {dir_emoji} {direction} | {zone_type} | {signals}",
        f"💰 {price:.2f} | SL {sl:.2f} | TP {tp:.2f}",
        "",
        f"🏆 <b>Opus 4.6</b> (benchmark)",
        f"   └ {opus_dec} | conf {opus_conf:.2f}",
        f"   └ {opus_reason}",
    ]

    for r in sorted_results:
        name = r["name"]
        resp = r.get("response")
        scoring = r.get("scoring", {})
        sc = scoring.get("score", 0)
        lat = r.get("latency_s", 0)
        medal = _score_medal(sc)
        bar = _score_bar(sc)

        if resp is None:
            err = r.get("error", "TIMEOUT") or "TIMEOUT"
            lines.append(f"")
            lines.append(f"{medal} <b>{name}</b>  ⚠️ {err}")
            lines.append(f"   └ {bar} --/100 | {_latency_str(lat)}")
            continue

        c_dec = resp.get("decision", "?")
        c_conf = float(resp.get("confidence", 0))
        c_reason = resp.get("reason", "")[:80]
        match = scoring.get("decision_match", False)
        conf_gap = scoring.get("conf_delta", 0)

        lines.append(f"")
        lines.append(f"{medal} <b>{name}</b>  {bar} <b>{sc}</b>/100")
        lines.append(f"   ├ {_dec_emoji(match)} {c_dec} | conf {c_conf:.2f} | gap {conf_gap:.2f} | {_latency_str(lat)}")
        lines.append(f"   └ {c_reason}")

    # Consensus line
    lines.append(f"")
    valid = [r for r in sorted_results if r.get("response")]
    agree = sum(1 for r in valid if r.get("scoring", {}).get("decision_match"))
    total = len(valid)
    if total > 0:
        pct = agree / total * 100
        if pct >= 75:
            cons_emoji = "🟢"
        elif pct >= 50:
            cons_emoji = "🟡"
        else:
            cons_emoji = "🔴"
        avg_score = sum(r.get("scoring", {}).get("score", 0) for r in valid) / total
        lines.append(f"{cons_emoji} Consensus: {agree}/{total} agree | avg {avg_score:.0f}/100")

    return "\n".join(lines)


def _format_exit_comparison(pos_data: dict, opus: dict, results: list) -> str:
    """Format exit comparison as HTML for Telegram."""
    direction = pos_data.get("direction", "?")
    dir_emoji = "🟢" if direction == "LONG" else "🔴"
    entry = pos_data.get("entry", 0)
    pnl_pts = pos_data.get("pnl_pts", 0)
    stage = pos_data.get("stage", "?")

    opus_act = opus.get("action", "?")
    opus_reason = opus.get("reason", "")

    sorted_results = sorted(
        results,
        key=lambda x: x.get("scoring", {}).get("score", 0),
        reverse=True,
    )

    lines = [
        f"🔬 <b>LLM COMPARISON — EXIT</b>",
        "",
        f"📊 {dir_emoji} {direction} | Entry {entry:.2f} | P/L {pnl_pts:+.1f}pt",
        f"📍 {stage}",
        "",
        f"🏆 <b>Opus 4.6</b> (benchmark)",
        f"   └ {opus_act} | {opus_reason}",
    ]

    for r in sorted_results:
        name = r["name"]
        resp = r.get("response")
        scoring = r.get("scoring", {})
        sc = scoring.get("score", 0)
        lat = r.get("latency_s", 0)
        medal = _score_medal(sc)
        bar = _score_bar(sc)

        if resp is None:
            err = r.get("error", "TIMEOUT") or "TIMEOUT"
            lines.append(f"")
            lines.append(f"{medal} <b>{name}</b>  ⚠️ {err}")
            lines.append(f"   └ {bar} --/100 | {_latency_str(lat)}")
            continue

        c_act = resp.get("action", "?")
        if c_act == "CLOSE":
            c_act = "TAKE_PROFIT"
        c_reason = resp.get("reason", "")[:80]
        match = scoring.get("action_match", False)

        new_sl_str = ""
        if c_act == "TIGHTEN" and resp.get("new_sl"):
            new_sl_str = f" | SL→{float(resp['new_sl']):.2f}"

        lines.append(f"")
        lines.append(f"{medal} <b>{name}</b>  {bar} <b>{sc}</b>/100")
        lines.append(f"   ├ {_dec_emoji(match)} {c_act}{new_sl_str} | {_latency_str(lat)}")
        lines.append(f"   └ {c_reason}")

    lines.append(f"")
    valid = [r for r in sorted_results if r.get("response")]
    agree = sum(1 for r in valid if r.get("scoring", {}).get("action_match"))
    total = len(valid)
    if total > 0:
        pct = agree / total * 100
        if pct >= 75:
            cons_emoji = "🟢"
        elif pct >= 50:
            cons_emoji = "🟡"
        else:
            cons_emoji = "🔴"
        avg_score = sum(r.get("scoring", {}).get("score", 0) for r in valid) / total
        lines.append(f"{cons_emoji} Consensus: {agree}/{total} agree | avg {avg_score:.0f}/100")

    return "\n".join(lines)


def _send_telegram(text: str) -> None:
    """Send raw HTML message via Telegram singleton."""
    notifier = tg.get()
    if notifier:
        notifier._send(text)


# ── Core Comparison Logic ────────────────────────────────────────────────────

def _call_model(name: str, model_cfg: dict, prompt: str, claude_cmd: str, timeout: int) -> dict:
    """Call a single LLM and return result dict."""
    t0 = time.time()
    response = None
    error = ""

    if model_cfg["type"] == "cli":
        response, error = _call_sonnet_cli(prompt, claude_cmd, timeout)
    elif model_cfg["type"] == "api":
        response, error = _call_openai_compatible(prompt, model_cfg["model"], timeout)

    elapsed = time.time() - t0
    return {
        "name": name,
        "response": response,
        "error": error,
        "latency_s": round(elapsed, 1),
    }


def _run_entry_comparison(setup: dict, opus_response: dict, claude_cfg: dict) -> None:
    """Run entry comparison against all LLMs. Called in background thread."""
    try:
        # Import here to avoid circular imports
        import claude_validator as cv
        prompt = cv.build_prompt(setup)
        claude_cmd = claude_cfg.get("cmd", "claude")
        timeout = _config.get("timeout_sec", 60)

        results = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_call_model, name, cfg, prompt, claude_cmd, timeout): name
                for name, cfg in LLM_MODELS.items()
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    # Score against benchmark
                    if result["response"]:
                        result["scoring"] = _score_entry(opus_response, result["response"])
                    else:
                        result["scoring"] = {"score": 0, "decision_match": False, "conf_delta": 0, "reason_sim": 0}
                    results.append(result)
                except Exception as e:
                    name = futures[future]
                    logger.debug(f"LLM comparison error ({name}): {e}")
                    results.append({
                        "name": name,
                        "response": None,
                        "error": "ERROR",
                        "scoring": {"score": 0},
                        "latency_s": 0,
                    })

        # Format and send
        msg = _format_entry_comparison(setup, opus_response, results)
        _send_telegram(msg)

        # Log summary
        for r in results:
            sc = r.get("scoring", {}).get("score", 0)
            logger.info(
                f"  LLM comparison | {r['name']}: score={sc}/100 | "
                f"{r.get('latency_s', 0):.1f}s"
            )

    except Exception as e:
        logger.warning(f"LLM entry comparison failed: {e}")


def _run_exit_comparison(pos_data: dict, opus_response: dict, claude_cfg: dict) -> None:
    """Run exit comparison against all LLMs. Called in background thread."""
    try:
        import claude_validator as cv
        prompt = cv.build_exit_prompt(pos_data)
        claude_cmd = claude_cfg.get("cmd", "claude")
        timeout = _config.get("timeout_sec", 60)

        results = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_call_model, name, cfg, prompt, claude_cmd, timeout): name
                for name, cfg in LLM_MODELS.items()
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result["response"]:
                        result["scoring"] = _score_exit(opus_response, result["response"])
                    else:
                        result["scoring"] = {"score": 0, "action_match": False, "reason_sim": 0}
                    results.append(result)
                except Exception as e:
                    name = futures[future]
                    logger.debug(f"LLM exit comparison error ({name}): {e}")
                    results.append({
                        "name": name,
                        "response": None,
                        "error": "ERROR",
                        "scoring": {"score": 0},
                        "latency_s": 0,
                    })

        msg = _format_exit_comparison(pos_data, opus_response, results)
        _send_telegram(msg)

        for r in results:
            sc = r.get("scoring", {}).get("score", 0)
            logger.info(
                f"  LLM exit comparison | {r['name']}: score={sc}/100 | "
                f"{r.get('latency_s', 0):.1f}s"
            )

    except Exception as e:
        logger.warning(f"LLM exit comparison failed: {e}")


# ── Public API (async wrappers) ──────────────────────────────────────────────

def compare_entry_async(setup: dict, opus_response: dict, claude_cfg: dict) -> None:
    """Fire entry comparison in a background daemon thread. Never blocks."""
    if not _config.get("enabled", True):
        return
    threading.Thread(
        target=_run_entry_comparison,
        args=(setup, opus_response, claude_cfg),
        daemon=True,
        name="llm-cmp-entry",
    ).start()
    logger.debug("LLM entry comparison fired (background)")


def compare_exit_async(pos_data: dict, opus_response: dict, claude_cfg: dict) -> None:
    """Fire exit comparison in a background daemon thread. Never blocks."""
    if not _config.get("enabled", True):
        return
    threading.Thread(
        target=_run_exit_comparison,
        args=(pos_data, opus_response, claude_cfg),
        daemon=True,
        name="llm-cmp-exit",
    ).start()
    logger.debug("LLM exit comparison fired (background)")
