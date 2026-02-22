# CHANGELOG — Smart Trader

![Version](https://img.shields.io/badge/version-1.0.0--beta-blue)
![Keep a Changelog](https://img.shields.io/badge/changelog-Keep%20a%20Changelog-orange)
![SemVer](https://img.shields.io/badge/versioning-SemVer-green)

> **Smart Trader** — AI-Powered XAUUSD Trading Bot
> Author: **Gifari K Suryo** — CEO & Founder, Lead R&D
> Company: **PT Surya Inovasi Prioritas (SURIOTA)**

---

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Types of changes: **Added** for new features, **Changed** for changes in existing functionality, **Deprecated** for soon-to-be removed features, **Removed** for now removed features, **Fixed** for bug fixes, **Security** for vulnerability fixes.

---

## [Unreleased]

> Placeholder for upcoming work. See [Planned Features (v1.1.0)](#planned-features-v110) at the bottom of this file.

---

## [1.0.0-beta] - 2026-02-22

Initial beta release of Smart Trader. This version establishes the full split-architecture pipeline: Python handles data acquisition, indicator calculation, zone detection, and trade execution while Claude Opus 4.6 acts as the primary AI reasoning layer for entry validation and exit optimization.

---

### 🏗️ Core Architecture

- **Split architecture**: Python handles all data/execution concerns; Claude Opus 4.6 handles all reasoning and trading decisions via CLI subprocess
- **MetaTrader 5 Python API integration** via dedicated `mt5_client.py` module — all broker interactions isolated from business logic
- **Source code organized into `src/` directory** with clean module separation (`indicators.py`, `scanner.py`, `executor.py`, `claude_validator.py`, `zone_detector.py`, `console_format.py`)
- **5-sink structured logging with loguru**: dedicated log streams for `bot_activity`, `market`, `trades`, `trade_journal`, and `errors` — each with independent rotation and retention policies
- **Colored console output with ANSI codes** via `console_format.py` with custom loguru sink for human-readable terminal monitoring

---

### 📈 Trading Engine

- **H1-based SMC zone detection**: Fair Value Gap (FVG), Order Block (OB), and Break of Structure (BOS) — all computed on the H1 timeframe for macro context
- **Dual zone source**: real-time H1 zone detector combined with SQLite zone cache from `claude_trader` — reduces redundant computation and provides persistence across restarts
- **Zone proximity filter**: configurable proximity threshold (default 5 pts) to determine if price is near a significant zone
- **H1 EMA(50) trend filter**: `indicators.h1_ema_trend()` returns BULLISH / BEARISH / NEUTRAL; counter-trend entries are blocked automatically
- **RSI extreme gate**: block LONG entries when RSI > 85 and SHORT entries when RSI < 15 to avoid chasing exhausted moves
- **M15 entry confirmation**: `m15_confirmation()` detects CHoCH (Change of Character) and engulfing candle patterns on the M15 timeframe for lower-timeframe entry timing
- **OTE (Optimal Trade Entry) zone detection**: identifies the 62–79% Fibonacci retracement range within the current swing for high-probability entry points
- **Premium/Discount zone classification**: price above the midpoint of a range = Premium (favor SELL), below = Discount (favor BUY)
- **Signal counting**: composite signal score from BOS, OB, FVG, CHoCH, Breaker, LiqSweep, M15 confirmation, OTE, and Discount/Premium alignment
- **Session awareness**: OVERLAP / LONDON / NEW_YORK / ASIAN / OFF_HOURS classification with priority weights; no new entries during OFF_HOURS (17:00–24:00 UTC)
- **Spike window guard**: new entries blocked during 07:45–08:00 UTC and 12:45–13:00 UTC (news/open spike windows)
- **Market closed detection via tick age**: if last tick is older than 5 minutes the market is considered closed — spread-only detection was found to be unreliable and removed

---

### 🤖 Claude AI Integration

- **Claude Opus 4.6 via CLI subprocess**: invoked as `claude -p --max-turns 1 --dangerously-skip-permissions` with structured stdin input — no temp files
- **Entry validation prompt**: full market snapshot (price, indicators, zones, session, signals) + proposed trade parameters + explicit trading rules → structured JSON decision
- **Exit optimization prompt**: open position state (entry, current price, SL, peak profit, hold time) → one of HOLD / TAKE_PROFIT / TIGHTEN
- **Claude is PRIMARY decision maker**: not a confirmer — the bot defers to Claude's judgment on whether to enter, hold, or take profit
- **Minimum confidence threshold**: 0.70 — entries below this threshold are rejected regardless of signal count
- **JSON response parsing**: with nested key fallback and alias handling (e.g., `CLOSE` mapped to `TAKE_PROFIT` for backward compatibility)
- **2-retry logic**: on empty Claude response, retry up to 2 times with a 3-second delay before treating the signal as rejected
- **Environment isolation**: `CLAUDECODE` and `CLAUDE_CODE_SESSION` environment variables are removed from the subprocess environment via `os.environ.copy()` + `pop()` to prevent subprocess conflicts on Windows
- **stdin input method**: prompt delivered via stdin pipe (no temp file creation) for clean process management

---

### 🔄 Exit Management

Multi-stage automated exit system implemented in `executor.py` — operates independently of Claude review and covers all risk scenarios:

- **Stage 1 — Breakeven (BE)**: move SL to entry price when floating profit reaches 0.7× the initial SL distance
- **Stage 2 — Profit Lock**: move SL to lock 50% of current profit at 1.5× RR; uses SL adjustment (not partial close) for 0.01 lot compatibility
- **Stage 3 — Trailing Stop**: trail the SL at 40% of peak profit once the position is in profit — tightens as new peaks are reached
- **Stage 4 — Scratch Exit**: close the position if it remains flat (< 5 pts from entry) after 60 minutes of holding — avoids capital lock-up on stale setups
- **Stage 5 — Stale Tighten**: halve the remaining SL risk after 90 minutes with no meaningful progress (< 50% of original SL distance in profit) — reduces exposure on slow movers
- **Stage 6 — Claude Exit Review**: periodic AI review of open positions; Claude may return HOLD, TAKE_PROFIT (execute close if > 3 pts profit), or TIGHTEN (adjust SL closer)

---

### 🛡️ Risk Management

- **Fixed lot size**: 0.01 lot — appropriate for $50–$100 demo balance at 1:500 leverage
- **Maximum open positions**: 1 (configurable) — prevents over-exposure from concurrent signals
- **Maximum drawdown guard**: 5% of account balance — bot pauses new entries if floating drawdown exceeds threshold
- **Free margin guard**: minimum 20% of balance must remain free before a new position is opened
- **Magic number filtering**: bot only manages positions opened with `magic=202602` — prevents interference with manually placed or externally managed trades

---

### 📱 Telegram Notifications

- **BOT STARTED / BOT STOPPED**: sent on bot lifecycle events with account info and configuration summary
- **SCAN REPORT heartbeat**: sent every 30 minutes with current market state, active zones, and session classification
- **ENTRY notification**: sent immediately on trade execution with direction, entry price, SL, TP, and signal rationale
- **EXIT notification**: sent on position close with outcome (profit/loss), hold duration, and exit reason
- **Position update notifications**: dedicated messages for BE trigger, PROFIT LOCK activation, and STALE TIGHTEN adjustment
- **Claude exit review notifications**: TAKE_PROFIT and TIGHTEN actions from Claude review are reported with reasoning
- **Market closed notification**: sent during weekend or off-hours heartbeat when market is detected as closed
- **Non-blocking sends**: all Telegram messages sent via daemon threads to avoid blocking the main trading loop
- **Dual UTC/WIB timestamps**: every message includes both UTC and WIB (UTC+7) timestamps for the team

---

### 🧪 Validation & Testing

- **`validate.py` orchestrator script**: single-command pre-deployment check; exit code 0 = safe to deploy, exit code 1 = do not deploy
- **243 automated pytest tests** across 4 test suites — full coverage of all core modules
- **`tests/test_indicators.py`** — 61 tests covering RSI, ATR, EMA, `m15_confirmation()`, `count_signals()`, OTE zone detection, and Premium/Discount zone classification
- **`tests/test_scanner.py`** — 58 tests covering zone proximity logic, entry direction validation, session classification, spike window guard, and risk filter gates
- **`tests/test_zone_detector.py`** — 68 tests covering FVG detection, OB detection, BOS detection, and zone merge/deduplication logic
- **`tests/test_claude_parser.py`** — 56 tests covering prompt construction, JSON extraction from Claude output, response validation, alias handling, and retry behavior
- **`tests/conftest.py`** — shared pytest fixtures with realistic synthetic OHLCV data used across all test suites

---

### 🔧 Design Decisions & Improvements

- **Removed Python hard gate for minimum signal count**: the previous minimum-3-signals gate was blocking quality 2-signal setups; signal quality assessment is now delegated entirely to Claude
- **`m15_conf` counted as a signal**: M15 confirmation (CHoCH / engulfing) is now included in `count_signals()` — it was previously computed but not counted, understating setup quality
- **Removed 30-minute loss cooldown**: this guard was designed for M15 scalping bots; for H1-setup bots the signal quality check (via Claude) is the correct gate, not a time-based cooldown
- **Claude role elevated to PRIMARY decision maker**: the prompt language and system design were updated to position Claude as the decision authority, not a passive confirmer
- **Entry REJECT rule narrowed**: Python-layer auto-rejection is reserved only for clearly invalid SL/TP values (e.g., SL on wrong side of entry); all other quality judgments are made by Claude

---

### 🐛 Bug Fixes

- **Fixed — market closed detection**: spread-only detection was unreliable (spread can spike during volatile open markets); replaced with tick-age check (> 5 minutes = market closed)
- **Fixed — log throttle during market closure**: the market-closed state was being logged every 30 seconds, flooding the log files; changed to log once per hour during sustained closure
- **Fixed — multiple process conflict**: stdout.log interleaving occurred when a previous bot instance was not fully terminated before restarting; added process lock check on startup
- **Fixed — Windows cp1252 encoding error**: Unicode box-drawing characters (e.g., `└`, `├`) caused `UnicodeEncodeError` on Windows terminals using cp1252; replaced with ASCII equivalents
- **Fixed — test collection crash**: legacy `test_trade.py` and `test_pipe.py` imported MT5 at module level, causing pytest collection to fail in CI environments without a live MT5 connection; moved to fixtures

---

## Planned Features (v1.1.0)

The following improvements are under consideration for the next release:

- **Performance dashboard**: equity curve visualization, per-session win rate tracking, drawdown timeline, and trade journal export to CSV/Excel
- **Auto-debug agent loop**: when `validate.py` exits with code 1, an autonomous agent loop attempts to identify the failing test, diagnose the root cause, apply a code fix, and re-run validation — fully automated development QA cycle
- **Backtest integration**: replay engine using historical OHLCV data from MT5 to validate strategy changes before live deployment
- **Multi-symbol support**: extend the trading engine beyond XAUUSD to support additional instruments (e.g., EURUSD, GBPUSD, NAS100) with per-symbol configuration
- **Web dashboard**: browser-based monitoring interface for live position tracking, bot status, Telegram log replay, and performance metrics — accessible remotely without terminal access

---

---

*Copyright &copy; 2026 **PT Surya Inovasi Prioritas (SURIOTA)**. All rights reserved.*
*Author: **Gifari K Suryo** — CEO & Founder, Lead R&D*
