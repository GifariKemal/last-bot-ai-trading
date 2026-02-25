# CLAUDE.md — Developer Instructions for Claude Code

> Project: **XAUUSD SMC Trading Bot v4.0.0**
> Owner: Gifari K Suryo — PT Surya Inovasi Prioritas (SURIOTA)
> Last updated: 2026-02-22

This file provides Claude Code with project-specific instructions that **override default behavior**.
Read this before touching any code in this project.

---

## 🗂️ Project Context

Algorithmic trading bot for XAUUSD (Gold) on MetaTrader 5 using Smart Money Concepts.
- **Broker**: Exness Demo — Login 413371434, Server `Exness-MT5Trial6`
- **Symbol**: `XAUUSDm` (NOT `XAUUSD` — Exness uses 3-digit precision, point=0.001)
- **Timeframe**: M15 primary, H1 context
- **Balance**: $100 demo | Leverage 1:100 | Max 1 position | 0.01 lot fixed
- **MT5 terminal**: `C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe`

---

## 🚨 Critical Rules — Never Violate

1. **NEVER close positions on bot restart or crash** — only close positions on explicit user request
2. **NEVER run `start_bot.sh` or `run_bot_stable.sh`** — these cause multiple instances; use `python main.py`
3. **NEVER run `run_trader.py` (Claude Autonomous Trader) simultaneously** — it conflicts with main bot
4. **NEVER use symbol `XAUUSD`** — always `XAUUSDm` for Exness
5. **NEVER commit `.env`** — contains live credentials; it is gitignored
6. **NEVER push to `main` branch** without user confirmation
7. **When a pause/cooldown ends, MUST clear the counter that triggered it** — see Bug #37

---

## 🧠 Trading Philosophy — Wisdom of Legendary Traders

This bot embodies the combined wisdom of **Soros, Paul Tudor Jones, Druckenmiller, Jim Simons, Bruce Kovner, Richard Dennis, Bill Lipschutz, Takashi Kotegawa, and Ken Griffin**. Every decision must align with these principles:

### Risk First (Kovner, PTJ, Kotegawa)
- **1% max risk per trade** — survival is non-negotiable; protect capital above all
- **Predetermined stop before entry** — no trade exists without a defined exit
- **Never average into losers** — adding to a losing position is how accounts die
- **Reduce size on losing streaks** — when wrong consecutively, get smaller not bigger
- **Daily/weekly drawdown circuit breaker** — hard stop when cumulative loss hits threshold

### Entry Quality Over Quantity (Lipschutz, Kotegawa, Soros)
- **Skip marginal setups — sit on hands 50% of the time** — patience IS the edge
- **Enter after liquidity sweeps** — smart money hunts stops before moving; wait for the sweep
- **Session-aware** — London/NY overlap is where volume lives; respect session weights

### Asymmetric R:R (PTJ, Lipschutz, Druckenmiller)
- **Minimum 3:1 R:R, target 5:1** — one winner must pay for multiple losers
- **Trail winners with ATR stops** — let profits run; never cut winners short
- **Variable conviction sizing** — when confluence is exceptional, lean in harder (future feature)

### Systematic Discipline (Simons, Dennis, Griffin)
- **No manual override — trust the algorithm** — emotions destroy edge; the system decides
- **Continuous data-driven improvement** — backtest, measure, iterate; opinions don't matter, data does
- **ATR-normalize all parameters** — volatility changes; absolute values become stale
- **Regime-adaptive parameters** — trending, ranging, volatile markets need different rules

> _"The secret to being successful from a trading perspective is to have an indefatigable and undying thirst for information and knowledge."_ — **PTJ**

---

## ▶️ How to Run

```bash
# Standard start (demo or live)
cd xauusd_trading_bot
python main.py --mode live -y

# Run backtest
python scripts/run_backtest.py

# Check MT5 connection
python scripts/test_mt5_connection.py

# Pre-deploy validation
python scripts/validate_deploy.py

# Debug signal pipeline
python scripts/debug_signals.py
python scripts/diagnose_signals.py
```

---

## 🏗️ Architecture Quick Reference

```
main.py
  └─ TradingBot (src/bot/trading_bot.py)
       ├─ DataManager (src/core/data_manager.py)        — MT5 OHLCV fetch
       ├─ SMCIndicatorsV4 (src/indicators/smc_v4_adapter.py)  — V4 detection
       ├─ MarketAnalyzer (src/analysis/market_analyzer.py)    — Regime detection
       ├─ AdaptiveConfluenceScorer (src/analysis/adaptive_scorer.py) — Scoring
       ├─ EntrySignalGenerator (src/strategy/entry_signals.py)  — Entry gates
       ├─ ExitSignalMonitor (src/strategy/exit_signals.py)      — Exit gates
       ├─ RiskManager (src/risk_management/)                    — SL/TP/lot
       ├─ OrderExecutor (src/execution/order_executor.py)       — MT5 API calls
       └─ TelegramNotifier (src/notifications/telegram_notifier.py)
```

---

## ⚙️ Key Configuration Files

| File | What to Change |
|------|---------------|
| `config/settings.yaml` | `use_smc_v4`, `use_adaptive_scorer`, `regime_weights`, `telegram` |
| `config/risk_config.yaml` | `fixed_lot`, `atr_multiplier`, `exit_stages` (`be_trigger_rr=0.77`, `trail_activation_rr=2.72`) |
| `config/session_config.yaml` | Session weights (Overlap=1.18, London/NY=1.16, Asian=0.75), `blackout_hours`, `friday_close_time_utc` |
| `config/trading_rules.yaml` | `require_structure_support`, `require_fvg_or_ob: false`, `require_mtf_alignment: false` |

---

## 📐 Current Parameter State (v4.0.0)

### Entry Gates (entry_signals.py)
- `MIN_SMC_SIGNALS = 2` (regime-overridden: trending=1, ranging/breakout/volatile=3)
- `RSI_BOUNCE_LOOKBACK = 5` bars
- `RSI_EXTREME_OVERBOUGHT = 75` (bounce protection)
- `RSI_EXTREME_OVERSOLD = 25`
- `RSI_HARD_OVERBOUGHT = 85` (hard block — was 90, fixed 2026-02-22)
- `RSI_HARD_OVERSOLD = 15` (hard block — was 10, fixed 2026-02-22)
- `checks["confluence_met"] = True` (adaptive scorer `passing` is sole gatekeeper)

### Adaptive Scorer Guards (adaptive_scorer.py)
- `MIN_SMC_FILL = 0.30` — if raw SMC fill < 30% of max, cap final score at 0.45 (below any regime threshold). Prevents tech indicators from rescuing a weak SMC signal.
- `MAX_SCORE_ON_WEAK_SMC = 0.45` — cap applied when SMC fill is below floor
- `OPPOSING_CHOCH_PENALTY = 0.15` — when the opposing direction has a CHoCH (reversal signal), subtract 0.15 from this direction's score. Passed via `opposing_smc` param from trading_bot.py.
- Log line now shows `smc_fill=XX%` and `[SMC_CAPPED]` tag when floor triggers

### Regime Weights (settings.yaml)
- Trending: min_conf=0.550, min_smc=1, sl_mult=2.60  (floor 0.55 — was 0.437, allowed marginal entries)
- Ranging: min_conf=0.550, min_smc=3, sl_mult=4.66
- Breakout: min_conf=0.614, min_smc=3, sl_mult=4.26
- Reversal: min_conf=0.589, min_smc=2, sl_mult=4.39
- Volatile: min_conf=0.704, min_smc=3, sl_mult=4.57

### Exit Stages (risk_config.yaml)
- `be_trigger_rr: 0.77` — Move SL to BE at 77% of SL distance profit
- `partial_close_rr: max(tp_rr * 0.65, 1.0)` — Dynamic partial (NOT 2.73R fixed)
- `trail_activation_rr: 2.72` — Start trailing at 2.72R

### Stale Trade Exit (trading_bot.py `_manage_positions`)
Auto-close positions with dead momentum: open `min_hours` but peak profit never reached `min_peak_rr`, and currently in loss.
- `stale_trade.enabled: true`
- `stale_trade.min_hours: 3` — minimum age before stale check applies
- `stale_trade.min_peak_rr: 0.3` — peak must reach 0.3R to be "not stale"
- Config: `risk_config.yaml` → `stale_trade:` section
- Fires BEFORE near-SL early exit; uses same Bug #48-safe entry_time parsing

### Regime-Adaptive Structure Exit (exit_signals.py `STRUCTURE_EXIT_MIN_RR`)
Min RR before structure/opposite-signal exit fires. Optuna-optimized (50 trials, 3-month WF, score 100.56 vs baseline 45.43).
- Strong Trend: **0.8R** — CHoCH in strong trends is often pullback, not reversal; wait for profit
- Weak Trend: **0.2R** — Weak trends reverse easily on CHoCH; exit quickly
- Range: **1.1R** — Simons: structure = noise, only 1.1R+ exits
- Volatile/Breakout: **0.6R** — Don't panic exit; require 0.6R before structure exit
- Reversal: **0.6R** — Reversals ARE CHoCH patterns; don't exit on entry signal noise
- `MIN_HOLD_MINUTES = 60` (4 M15 bars, was 30/2 bars — Optuna min_hold_bars=4)

### SMC Lookbacks
- `swing_lookback = 5` (was 10 — M15-optimized 2026-02-23, sw=5 → PF=4.61, WR=64.7%)
- `BOS_LOOKBACK_BARS = 50` (restored from 20 — needed with sw=5 for adequate BOS coverage)
- `SWEEP_LOOKBACK_BARS = 20`

---

## 🐛 Known Bugs Reference (Fixed)

| Bug | Description | Fix |
|-----|-------------|-----|
| #28 | MTF alignment never fires live | `mtf_data["M15"]` = assigned result back |
| #35 | `config.get("indicators")` empty | Use `config.get("smc_indicators")` |
| #36a/b/c | execute_exit profit, MT5 race, external SL/TP | Patched in order_executor.py |
| #37 | Infinite pause after counter reset | Reset `consecutive_losses` on pause expiry |
| #38 | `require_all_positions_profitable` blocked all | Set `false` in config |
| #39 | Exit fires on entry candle | `MIN_HOLD_MINUTES = 30` (was 15, raised to 2 M15 bars) |
| #40 | AdaptiveScorer inflation (smc_raw/0.40) | Normalize by `_smc_base_max` |
| #50 | Weak SMC (BOS-only 26%) inflated to 0.76 by tech+MTF | `MIN_SMC_FILL=0.30` cap + `OPPOSING_CHOCH_PENALTY=0.15` in adaptive_scorer.py |
| #51 | Stale exit skipped after restart (no `entry_time`) | `add_position()` derives `entry_time` from MT5 `open_time`; also preserves `entry_sl` |
| #52 | Infinite pause loop (daily loss re-triggers 60min pause) | Percent-based violations block permanently until period reset; timed pause only for consecutive losses |

---

## 🔔 Telegram Message Types

Module: `src/notifications/telegram_notifier.py`

| Event | Sent? | Notes |
|-------|:-----:|-------|
| BOT STARTED / STOPPED | ✅ | |
| SCAN REPORT | ✅ | Every 30 min heartbeat |
| ENTRY SIGNAL | ✅ | After all gates pass |
| EXIT (scratch / TP) | ✅ | |
| POSITION UPDATE (BE) | ✅ | |
| POSITION UPDATE (TRAIL) | ❌ | Suppressed — too frequent |
| CLAUDE REVIEW → HOLD | ❌ | Suppressed — no spam |
| CLAUDE REVIEW → TAKE_PROFIT / TIGHTEN | ✅ | |

**HTML gotcha**: Avoid `>` `<` in plain text fields — Telegram parses as HTML tags.
**Chat ID**: Must use `-100` prefix for supergroups (e.g., `-1003549733840`).

---

## 📁 File Organization

```
Root (keep here):     main.py, README.md, requirements.txt, .env, .gitignore
docs/:                CHANGELOG.md lives in root; *.md reference + .pine file in docs/
scripts/:             All utility / diagnostic / backtest scripts
config/:              All YAML configuration files
src/:                 All Python source modules
```

---

## 🧪 Testing Protocol

Before any live or demo run:
1. `python scripts/test_mt5_connection.py` — verify MT5 connects
2. `python scripts/validate_deploy.py` — pre-flight checks
3. `python scripts/debug_signals.py` — verify signal pipeline produces signals
4. Run in demo mode for minimum **2 weeks** before any live deployment

---

## 💡 Code Style

- Python 3.13+ — use modern syntax (`match`, `f-strings`, type hints)
- All configs read via `config.get(key, default)` — never hardcode broker values
- Log with `self.logger` (Loguru) — not `print()`
- Polars DataFrames preferred over Pandas for performance-critical paths
- Config keys use `snake_case`, YAML files use `snake_case`
- No over-engineering — prefer 3 clear lines over a premature abstraction

---

*© 2026 PT Surya Inovasi Prioritas. Proprietary & Confidential.*
