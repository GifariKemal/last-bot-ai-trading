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
| #39 | Exit fires on entry candle | `MIN_HOLD_MINUTES = 15` |
| #40 | AdaptiveScorer inflation (smc_raw/0.40) | Normalize by `_smc_base_max` |

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
