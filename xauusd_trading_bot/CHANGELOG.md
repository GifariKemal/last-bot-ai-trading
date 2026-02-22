# Changelog

All notable changes to **XAUUSD SMC Trading Bot** are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) | Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

- Live performance dashboard (web UI)
- Google Sheets trade journal integration
- Automated forward-test report generator

---

## [4.0.0] — 2026-02-22

> **V4 SMC Library + Exness Migration + Full Telegram Integration**
> Testing begins: 2026-02-23 (market open, Exness Demo XAUUSDm)

### Added
- **V4 SMC Library** — `smartmoneyconcepts>=0.0.26` replaces custom V3 detectors
- `src/indicators/smc_v4_adapter.py` — Drop-in `SMCIndicatorsV4` class, backward-compatible with V3 interface
- **Exness XAUUSDm support** — Symbol `XAUUSDm` (3-digit, point=0.001), replaces `XAUUSD`
- **Telegram notifications** — Full integration across all bot events:
  - `BOT STARTED` / `BOT STOPPED`
  - `ENTRY SIGNAL` (after all gates pass)
  - `EXIT` (scratch close, claude_take_profit)
  - `POSITION UPDATE` (BE trigger, profit lock, stale tighten)
  - `SCAN REPORT` (30-min heartbeat)
  - `CLAUDE EXIT REVIEW` (TAKE_PROFIT + TIGHTEN only — HOLD suppressed)
- **Maintenance blackout** — 22:00–23:00 UTC (Exness XAUUSDm daily gap, verified from candle history)
- **Friday close logic** — No new entries after 21:30 UTC Friday (Exness weekend close)
- **TradingView Pine Script** (`docs/XAUUSD_SMC_Bot.pine`) — Fully synchronized visual reference:
  - All 16 discrepancies vs live bot corrected (see sync notes in file header)
  - Partial close level plot, maintenance/Friday backgrounds, hard RSI blocks, counter-trend penalty
- **`docs/` folder** — Moved all `.md` reference files and Pine script here
- `scripts/start_bot_loop.sh` and `stop_bot.sh` moved from root to `scripts/`

### Changed
- **Symbol**: `XAUUSD` → `XAUUSDm` throughout all configs and MT5 calls
- **BOS_LOOKBACK_BARS**: 50 → **20** (5 hours vs 12.5 hours — 50 was causing trivially satisfied gate)
- **SWEEP_LOOKBACK_BARS**: set to **20** (consistent with BOS lookback)
- **RSI_HARD_OVERBOUGHT**: 90 → **85** (`entry_signals.py` — 90 was dead code for XAUUSD which regularly reaches 70–80)
- **RSI_HARD_OVERSOLD**: 10 → **15**
- **Session weights** (V3 Optuna-finalized in `session_config.yaml`):
  - Overlap: 1.25 → **1.18**
  - London/NY: 1.10 → **1.16**
  - Asian: 0.70 → **0.75**
- **Stage 2 partial close**: Fixed 2.73R hardcoded → `effective_partial_rr = max(tp_rr * 0.65, 1.0)` per position's actual TP/SL (old value was > TP_RR in all regimes — never fired)
- **Duplicate confluence gate removed**: `checks["confluence_met"] = True`; adaptive scorer `passing` flag is sole gatekeeper (old `min_confluence_score=0.60` overrode regime-adaptive thresholds 0.44–0.70)
- **Windows Unicode patch** — `smartmoneyconcepts/__init__.py` catches `UnicodeEncodeError` on startup

### Fixed
- **Bug #28** — MTF alignment never fires live: `mtf_data["M15"]` was RAW (unassigned); fix: assign result back to `mtf_data`
- **Bug #35** — `config.get("indicators")` returned empty dict; fix: `config.get("smc_indicators")`
- **Bug #36a** — `execute_exit` profit injection: wrong variable used for realized P&L
- **Bug #36b** — MT5 race condition on position close: added position existence check before close
- **Bug #36c** — External SL/TP closes not detected: added polling for position removal by MT5
- **Bug #37** — Pause expires but `consecutive_losses` not reset → infinite pause loop; fix: reset counter on expiry
- **Bug #38** — `require_all_positions_profitable` blocked ALL entries; fix: set `false`
- **Bug #39** — Exit signals firing on same candle as entry; fix: `MIN_HOLD_MINUTES = 15`
- **Bug #40** — `AdaptiveScorer` score inflation: `smc_raw / 0.40` inflated all scores to 1.00; fix: normalize by actual max (`self._smc_base_max`)

---

## [3.0.0] — 2026-02-20

> **V3 Regime-Adaptive Scoring + Optuna Optimization**

### Added
- `src/analysis/adaptive_scorer.py` — `AdaptiveConfluenceScorer` with regime-conditional weights
- **5 market regimes**: Trending, Ranging, Breakout, Reversal, Volatile
- **Optuna optimization** — 25 trials, 3-window walk-forward validation
- **Multi-stage exits**:
  - BE at 0.77R (`be_trigger_rr: 0.77`)
  - Dynamic partial close (`effective_partial_rr`)
  - Trail activation at 2.72R (`trail_activation_rr: 2.72`)
- **SL-hit directional cooldown** — Block same direction for 2 M15 candles (30 min) after SL
- **Counter-trend penalty** — `-0.10` score for BUY in bearish regime or SELL in bullish
- **MTF bonus direction-aware** — Only applies MTF bonus if dominant trend matches trade direction
- H1 data fetch — `"H1"` added to `fetch_multiple_timeframes`
- `data/optimization_v3/optimized_config_v3.yaml` — V3 optimal parameters

### Changed
- **Entry gate**: Removed duplicate `min_confluence_score=0.60` override; adaptive scorer `passing` is sole gatekeeper
- **`require_fvg_or_ob`**: `true` → `false` (FVG/OB scored in confluence — double-gating was too restrictive)
- **`require_mtf_alignment`**: `true` → `false` (MTF is weighted bonus, not hard gate)
- **H1 bias check removed** from `entry_signals.py` (replaced by regime-adaptive scoring)
- **ICT P/D zone logic removed** (not relevant to SMC-first approach)
- **Loop interval fix**: `loop_interval_seconds` key → `loop_interval` (was defaulting to 1s instead of configured value)

### V3 Backtest Results

| Config | Profit Factor | Win Rate | Max DD | Return | Trades |
|--------|:-------------:|:--------:|:------:|:------:|:------:|
| V3 Full (all signals) | 1.12 | 51.5% | 12.80% | +13.63% | 478 |
| V3 Optimized (BOS-only) | 1.33 | 56.2% | 6.24% | +31.89% | — |

---

## [2.0.0] — 2026-02-17

> **V2 Fixed-Weight Confluence + Backtesting Framework**

### Added
- `ConfluenceScorer` — fixed-weight SMC confluence scoring
- `BacktestEngine` — historical simulation on M15 OHLCV data
- Signal decomposition analysis (`run_signal_decomposition.py`)
- `docs/OPTIMIZATION_GUIDE.md` — Optuna optimization walkthrough
- Optimization status tracking (`docs/OPTIMIZATION_STATUS.md`)
- V3 comparison scripts (`run_v3_comparison.py`, `run_v4_comparison.py`)

### Changed
- SL method: `fixed_pips` → `atr_based` (ATR × multiplier)
- TP: minimum 1:2 RR enforced

---

## [1.0.0] — 2026-02-15

> **Initial Release — Core SMC Bot**

### Added
- MT5 Python API integration (`MetaTrader5>=5.0.45`)
- SMC Indicator Suite:
  - Fair Value Gap (FVG) detector
  - Order Block detector
  - Liquidity Sweep detector
  - Break of Structure (BOS) detector
  - Change of Character (CHoCH) detector
- Technical indicators: ATR(14), EMA(20/50/100/200), RSI(14), MACD(12/26/9)
- Multi-timeframe data fetch: M1, M5, M15, H1
- Basic position management (open, modify SL/TP, close)
- Session filtering: Asian / London / NY / Overlap
- ATR-based SL/TP calculation
- Fixed lot sizing (0.01)
- Comprehensive logging with Loguru (rotation, retention)
- Emergency stop via `EMERGENCY_STOP.txt` file detection
- Configuration system: YAML configs + `.env` secrets

---

## Versioning Convention

| Increment | When |
|-----------|------|
| **MAJOR** (X.0.0) | Architecture change, new indicator engine, broker migration |
| **MINOR** (x.Y.0) | New feature, new signal type, new exit logic |
| **PATCH** (x.y.Z) | Bug fix, parameter tuning, config update |

---

*Maintained by **Gifari K Suryo** — Lead R&D, PT Surya Inovasi Prioritas (SURIOTA)*
*© 2026 PT Surya Inovasi Prioritas. All Rights Reserved.*
