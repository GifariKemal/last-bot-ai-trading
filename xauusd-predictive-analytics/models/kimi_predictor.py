"""
Kimi K2 Predictor — LLM-based XAUUSD signal generator.

Replaces CatBoost training pipeline with a zero-shot inference call to
Kimi K2 (OpenAI-compatible API hosted at sumopod.com).

Flow
----
1. Build a structured market context from latest 20 bars + indicator snapshot.
2. POST to Kimi API with a detailed analyst prompt.
3. Parse JSON response → signal (+1 BUY, -1 SELL, 0 NO_TRADE) + metadata.

Model: kimi-k2-250905 (1T MoE, 128k context, strong financial reasoning)
Temperature: 0.1 (deterministic, low variance across repeated calls)
"""

from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
from loguru import logger
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from config.settings import KimiConfig
from models.market_context import build_context

# Signal mapping
SIGNAL_MAP: dict[str, int] = {"BUY": 1, "SELL": -1, "NO_TRADE": 0}


class KimiPredictor:
    """
    Zero-shot XAUUSD signal generator using Kimi K2.

    Parameters
    ----------
    config : KimiConfig
        API key, base URL and model name.
    max_retries : int
        Number of API call retries on transient errors.
    retry_delay : float
        Seconds to wait between retries.
    """

    SYSTEM_PROMPT = (
        "You are a professional XAUUSD (Gold spot) intraday trader with 20 years of experience. "
        "You specialise in M15 chart analysis combining technical indicators, price action, "
        "and multi-timeframe confluence. Your responses must be concise, data-driven, and "
        "expressed as valid JSON only — no markdown, no preamble."
    )

    def __init__(
        self,
        config: KimiConfig,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.config      = config
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client     = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        logger.info(
            f"KimiPredictor ready | model: {config.model} | "
            f"base_url: {config.base_url}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Generate a trade signal for the latest bar in df.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-enriched DataFrame (output of FeatureEngineer.build_features).
            Must have at least 20 rows after dropna().

        Returns
        -------
        dict with keys:
            signal      : int   (+1 BUY, -1 SELL, 0 NO_TRADE)
            signal_label: str   ("BUY" | "SELL" | "NO_TRADE")
            confidence  : float (0.0 – 1.0)
            reasoning   : str
            key_factors : list[str]
            raw_response: str   (full JSON string from API)
        """
        ctx = build_context(df)
        prompt = self._build_prompt(ctx)

        raw = self._call_api(prompt)
        return self._parse_response(raw)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_prompt(self, ctx: dict) -> str:
        bars_table = self._format_bars_table(ctx["recent_bars"])
        patterns   = ctx["patterns"] or "None detected"
        sessions   = ", ".join(ctx["active_sessions"]) or "Off-session"

        return f"""Analyze this XAUUSD M15 market snapshot and predict the NEXT BAR direction.

=== LAST 20 M15 BARS ===
{bars_table}

=== CURRENT INDICATOR SNAPSHOT ===
RSI(14)      : {ctx['rsi']:.1f}
MACD hist    : {ctx['macd_hist']:.4f}  |  MACD line: {ctx['macd']:.4f}  |  Signal: {ctx['macd_signal']:.4f}
Stoch K/D    : {ctx['stoch_k']:.1f} / {ctx['stoch_d']:.1f}
CCI(20)      : {ctx['cci']:.1f}
ADX(14)      : {ctx['adx']:.1f}  (>25 trending)  |  DI bias (±1): {ctx['di_diff']:.3f}
ATR(14)      : {ctx['atr']:.3f}  |  Volatility regime: {ctx['atr_regime']:.2f}x avg
BB%B         : {ctx['bb_pct']:.3f}  |  BB Width: {ctx['bb_width']:.4f}
EMA50 dist   : {ctx['ema50_dist']:.4f}  |  EMA200 dist: {ctx['ema200_dist']:.4f}

=== HIGHER TIMEFRAME CONTEXT ===
H1 : RSI={ctx['h1_rsi']:.1f}  ADX={ctx['h1_adx']:.1f}  BB_pos={ctx['h1_bb_pos']:.3f}  EMA_bias={ctx['h1_ema_bias']:+d}
H4 : RSI={ctx['h4_rsi']:.1f}  ADX={ctx['h4_adx']:.1f}  BB_pos={ctx['h4_bb_pos']:.3f}  EMA_bias={ctx['h4_ema_bias']:+d}

=== SESSION & TIME ===
UTC Time : {ctx['time']}  |  Day: {ctx['day_of_week']}
Active   : {sessions}

=== SESSION RANGE PROXIMITY (ATR-normalised) ===
8h  high dist : {ctx['dist_high_8h']:.2f} ATR  |  8h  low dist : {ctx['dist_low_8h']:.2f} ATR
24h high dist : {ctx['dist_high_24h']:.2f} ATR  |  24h low dist : {ctx['dist_low_24h']:.2f} ATR

=== CANDLE PATTERNS DETECTED ===
{patterns}

Based on ALL factors above, respond ONLY with valid JSON (no extra text):
{{
  "signal": "BUY" or "SELL" or "NO_TRADE",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<2-3 sentence explanation of the most important signals>",
  "key_factors": ["<factor1>", "<factor2>", "<factor3>"]
}}"""

    @staticmethod
    def _format_bars_table(bars: list[dict]) -> str:
        header = f"{'Time (UTC)':<20} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'Vol':>8}"
        sep    = "-" * 64
        rows   = [
            f"{b['time']:<20} {b['open']:>8.3f} {b['high']:>8.3f} {b['low']:>8.3f} {b['close']:>8.3f} {b['volume']:>8.0f}"
            for b in bars
        ]
        return "\n".join([header, sep] + rows)

    def _call_api(self, prompt: str) -> str:
        """Call Kimi API with retry logic. Returns raw JSON string."""
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Calling Kimi API ({self.config.model}) … "
                    f"attempt {attempt}/{self.max_retries}"
                )
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=512,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or "{}"
                logger.debug(f"Kimi raw response: {raw}")
                return raw

            except RateLimitError:
                logger.warning(f"Rate limited. Waiting {self.retry_delay * attempt}s …")
                time.sleep(self.retry_delay * attempt)
            except APITimeoutError:
                logger.warning(f"API timeout on attempt {attempt}.")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
            except APIError as e:
                logger.error(f"Kimi APIError: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        logger.error("All Kimi API retries exhausted. Returning NO_TRADE.")
        return '{"signal": "NO_TRADE", "confidence": 0.0, "reasoning": "API unavailable.", "key_factors": []}'

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        """Parse Kimi JSON response into a standardised result dict."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Kimi response as JSON: {raw!r}")
            data = {}

        signal_label = str(data.get("signal", "NO_TRADE")).upper()
        if signal_label not in SIGNAL_MAP:
            logger.warning(f"Unexpected signal value '{signal_label}' — defaulting to NO_TRADE.")
            signal_label = "NO_TRADE"

        confidence  = float(data.get("confidence", 0.0))
        reasoning   = str(data.get("reasoning", ""))
        key_factors = list(data.get("key_factors", []))

        return {
            "signal":       SIGNAL_MAP[signal_label],
            "signal_label": signal_label,
            "confidence":   confidence,
            "reasoning":    reasoning,
            "key_factors":  key_factors,
            "raw_response": raw,
        }
