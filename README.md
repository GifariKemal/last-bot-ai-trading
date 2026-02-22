# Last Bot AI Trading

Kumpulan bot trading algorithmic berbasis Python + AI untuk XAUUSD (Gold), dikembangkan oleh **Gifari K Suryo** — PT Surya Inovasi Prioritas (SURIOTA).

---

## Struktur Proyek

```
Last Bot AI Trading/
├── xauusd_trading_bot/     # Bot utama — SMC V4 + Adaptive Scorer (Exness Demo)
├── smart_trader/           # Bot dengan Claude AI validation loop (IC Markets Demo)
├── claude_trader/          # Eksperimen Claude-driven trader
└── xauusd-predictive-analytics/  # Analitik prediktif XAUUSD
```

---

## Bot Utama

### 1. `xauusd_trading_bot`
Bot trading XAUUSD menggunakan **Smart Money Concepts (SMC) V4** dengan scoring adaptif berbasis market regime.

- **Broker**: Exness Demo (XAUUSDm)
- **Strategy**: FVG, Order Block, BOS/CHoCH, Liquidity Sweep
- **Exit**: Multi-stage (Breakeven → Partial → Trailing)
- **Notifikasi**: Telegram real-time

```bash
cd xauusd_trading_bot
python main.py --mode live -y
```

### 2. `smart_trader`
Bot trading dengan **Claude Opus 4.6** sebagai primary decision maker via validation loop.

- **Broker**: IC Markets Demo
- **Strategy**: H1 zone-based (FVG, OB, BOS) + Claude reasoning
- **Exit**: BE → Profit Lock → Trail → Scratch → Stale
- **Notifikasi**: Telegram real-time

```bash
cd smart_trader
python main.py
```

---

## Stack

- Python 3.11+
- MetaTrader 5 (`MetaTrader5` library)
- `smartmoneyconcepts` (SMC V4)
- Claude Opus 4.6 (Anthropic)
- Telegram Bot API

---

## Author

**Gifari K Suryo** — CEO & Founder, PT Surya Inovasi Prioritas (SURIOTA)
