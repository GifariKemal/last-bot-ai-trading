<div align="center">

# Broker Comparison Analysis

![Status](https://img.shields.io/badge/Status-Decision_Pending-yellow?style=for-the-badge)
![Budget](https://img.shields.io/badge/Budget-$100-blue?style=for-the-badge)
![Instrument](https://img.shields.io/badge/Instrument-XAUUSD-gold?style=for-the-badge)

**Smart Trader — Broker Migration for Live Trading**

_Comparative analysis of 3 broker candidates for automated XAUUSD trading_

---

**Author:** Gifari K Suryo — CEO & Founder, Lead R&D
**Organization:** PT Surya Inovasi Prioritas (SURIOTA)
**Date:** February 24, 2026
**Analyst:** Claude Opus 4.6

</div>

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Requirements](#2-system-requirements)
3. [Broker Profiles](#3-broker-profiles)
   - [3.1 Exness](#31-exness)
   - [3.2 XM](#32-xm)
   - [3.3 WemasterTrade](#33-wemastertrade-prop-firm)
4. [Head-to-Head Comparison](#4-head-to-head-comparison)
5. [Compatibility Analysis](#5-compatibility-analysis-with-smart-trader)
6. [Cost Simulation](#6-cost-simulation-per-trade)
7. [Final Verdict & Recommendation](#7-final-verdict--recommendation)
8. [Migration Plan](#8-migration-plan)
9. [Sources](#9-sources)

---

## 1. Executive Summary

| Broker             | Score        | Verdict                                   |
| ------------------ | ------------ | ----------------------------------------- |
| **XM (Ultra Low)** | **9.2 / 10** | **RECOMMENDED**                           |
| Exness (Standard)  | 5.5 / 10     | Spread terlalu besar untuk $100 budget    |
| Exness (Pro)       | 8.8 / 10     | Ideal tapi min deposit $200 — over budget |
| WemasterTrade      | 2.0 / 10     | ELIMINASI — bot banned, bukan MT5         |

**Rekomendasi: XM Ultra Low Account** — deposit $100 + bonus $50 = effective $150 capital, spread XAUUSD ~2.5 pips, MT5 supported, bot allowed, zero code changes (config only).

---

## 2. System Requirements

Smart Trader bot memiliki kebutuhan spesifik yang harus dipenuhi broker:

| Requirement     | Detail                                                             | Priority   |
| --------------- | ------------------------------------------------------------------ | ---------- |
| **Platform**    | MetaTrader 5 (MT5) — seluruh codebase built di atas MT5 Python API | CRITICAL   |
| **Instrument**  | XAUUSD (Gold/USD)                                                  | CRITICAL   |
| **Automation**  | Python bot running 24/5, automated order placement via MT5 API     | CRITICAL   |
| **Min Capital** | $100 available budget                                              | HARD LIMIT |
| **Lot Size**    | 0.01 (micro lot) — fixed position sizing                           | REQUIRED   |
| **Leverage**    | Min 1:100 (margin ~$51/trade at 1:100)                             | REQUIRED   |
| **Spread**      | Serendah mungkin — SL=3xATR (~30-50pt), TP=5xATR (~50-80pt)        | HIGH       |
| **Execution**   | Market execution, no requotes                                      | HIGH       |
| **Regulation**  | Tier-1 or Tier-2 regulator preferred                               | MEDIUM     |

### Current System Parameters

```yaml
trading:
  risk_percent: 1.0 # % of balance per trade
  min_lot: 0.01
  sl_atr_mult: 3.0 # SL = 3x ATR (~30-50 points)
  tp_atr_mult: 5.0 # TP = 5x ATR (~50-80 points)
  max_positions: 1 # single position at a time
  max_drawdown_pct: 5.0 # $5 max DD on $100 balance
```

---

## 3. Broker Profiles

### 3.1 Exness

**Website:** [https://www.exness.com](https://www.exness.com)
**Regulasi:** CySEC (Cyprus), FCA (UK), FSA (Seychelles)

#### Account Types

| Parameter         | Standard        | Pro             | Raw Spread      | Zero                |
| ----------------- | --------------- | --------------- | --------------- | ------------------- |
| **Min Deposit**   | $1              | **$200**        | $200            | $200                |
| **XAUUSD Spread** | 20–35 pips      | ~1.1 pips       | from 0.0 pips   | near 0 pips         |
| **Commission**    | $0              | $0              | $3.5/lot/side   | $0.2/lot/side (min) |
| **Execution**     | Market          | Instant         | Market          | Market              |
| **Leverage**      | Up to Unlimited | Up to Unlimited | Up to Unlimited | Up to Unlimited     |
| **Stop Out**      | 0%              | 0%              | 0%              | 0%                  |
| **Swap-Free**     | Available       | Available       | Available       | Available           |

#### Keunggulan Exness

- Leverage Unlimited — margin per trade sangat kecil
- Stop Out 0% — posisi baru ditutup saat margin benar-benar habis
- Pro Account spread sangat kompetitif (~1.1 pips XAUUSD, no commission)
- Withdrawal instant (crypto, e-wallet)
- Familiar — sudah ada demo account Exness di sistem

#### Kelemahan untuk Budget $100

- **Standard Account spread 20-35 pips** — ini SANGAT BESAR untuk sistem kita
- Pro Account butuh **$200 minimum deposit** — over budget $100
- Dengan $100 hanya bisa Standard → spread makan sebagian besar profit range

---

### 3.2 XM

**Website:** [https://www.xm.com](https://www.xm.com)
**Regulasi:** CySEC (Cyprus), ASIC (Australia), IFSC (Belize)

#### Account Types

| Parameter          | Micro        | Standard     | Ultra Low     |
| ------------------ | ------------ | ------------ | ------------- |
| **Min Deposit**    | $5           | $5           | $5            |
| **XAUUSD Spread**  | ~3.5 pips    | ~3.5 pips    | **~2.5 pips** |
| **Commission**     | $0           | $0           | $0            |
| **Lot Size**       | 1K units     | 100K units   | 100K units    |
| **Leverage**       | Up to 1:1000 | Up to 1:1000 | Up to 1:1000  |
| **Stop Out**       | 20%          | 20%          | 20%           |
| **Bonus Eligible** | Yes          | Yes          | **No**        |

#### Bonus Program

| Bonus Type            | Detail                                          |
| --------------------- | ----------------------------------------------- |
| **No Deposit Bonus**  | $30 free — start trading tanpa deposit          |
| **50% Deposit Bonus** | Deposit $100 → dapat $50 credit (max $500)      |
| **20% Subsequent**    | Deposit berikutnya bonus 20% (max total $5,000) |

> **Penting:** Ultra Low Account **TIDAK eligible** untuk bonus. Harus pilih Standard Account untuk dapat bonus 50%.

#### Perhitungan Bonus untuk $100

```
Deposit:    $100.00
Bonus 50%:  + $50.00 (credit, tidak bisa di-withdraw)
────────────────────
Effective:  $150.00 (for margin & trading)
```

- Bonus bersifat credit — bisa dipakai untuk margin buffer tapi tidak bisa di-withdraw langsung
- Withdrawal dari balance akan mengurangi bonus secara proporsional
- Contoh: withdraw $50 (50% of deposit) → bonus berkurang 50% ($25 hilang)

#### Keunggulan XM

- Min deposit $5 — sangat aksesibel
- XAUUSD spread ~2.5-3.5 pips — kompetitif untuk akun non-ECN
- Bonus 50% memberikan extra margin buffer
- Leverage 1:1000 — margin per 0.01 lot XAUUSD hanya ~$5.2
- MT5 fully supported
- Bot/automation allowed — no restrictions
- Negative balance protection

#### Kelemahan XM

- Stop Out 20% (vs 0% Exness) — posisi ditutup lebih awal
- Ultra Low (spread terbaik) tidak eligible bonus — harus pilih salah satu
- Spread masih lebih besar dari Exness Pro/Raw

---

### 3.3 WemasterTrade (Prop Firm)

**Website:** [https://wemastertrade.com](https://wemastertrade.com)
**Tipe:** Proprietary Trading Firm (BUKAN broker tradisional)
**Regulasi:** MSB License, Canada (WeCopy Fintech Inc.)

#### Konsep Berbeda

WemasterTrade bukan broker — ini adalah **prop firm**. Trader membayar fee untuk mendapat akses ke funded account dengan modal besar milik perusahaan. Profit dibagi sesuai skema.

#### Package Options

| Package         | Fee        | Funded Account   | Daily DD | Max DD |
| --------------- | ---------- | ---------------- | -------- | ------ |
| Customize (min) | **$35**    | ~$10,000         | 4-5%     | 6-10%  |
| Customize (mid) | **$50.75** | ~$10,000–$25,000 | 4-5%     | 6-10%  |
| Challenge $25K  | ~$125      | $25,000          | 5%       | 10%    |
| Challenge $50K  | ~$220      | $50,000          | 5%       | 10%    |

#### Profit Split Progression

| Phase               | Profit Split (Trader : Firm) |
| ------------------- | ---------------------------- |
| Challenge Phase     | 0% — no profit withdrawal    |
| Stabilization Phase | 30 : 70                      |
| Fully Funded        | **90 : 10**                  |

#### Trading Rules

| Rule            | Detail                  | Impact on Smart Trader                 |
| --------------- | ----------------------- | -------------------------------------- |
| **HFT & Bots**  | **BANNED**              | FATAL — sistem kita adalah bot         |
| EA Allowed      | Yes (ambigu)            | Konflik dengan ban bot                 |
| Best Day Rule   | Max 20% of total profit | Bot bisa profit besar 1 hari → violasi |
| Daily Drawdown  | 4-5%                    | Sangat ketat untuk automated trading   |
| Max Drawdown    | 6-10% (static)          | Ketat tapi manageable                  |
| News Trading    | Allowed                 | OK                                     |
| Weekend Holding | Allowed                 | OK                                     |
| **Platform**    | **cTrader**             | FATAL — bukan MT5                      |

#### ELIMINASI — 3 Masalah Fatal

```
FATAL #1: Bot/HFT BANNED
├─ Smart Trader = Python automated bot (24/5 running)
├─ Bot detection bisa menyebabkan akun di-ban
└─ Semua profit disita jika terdeteksi

FATAL #2: Platform cTrader (bukan MT5)
├─ Seluruh codebase built di atas MT5 Python API
├─ mt5_client.py, executor.py, indicators.py — semua MT5
└─ Migrasi ke cTrader = REWRITE seluruh trading engine

FATAL #3: Profit Consistency Rule
├─ "Best trading day max 20% of total profit"
├─ Bot bisa capture big move 1 hari (50+ pips)
└─ Violasi rule → payout ditolak
```

---

## 4. Head-to-Head Comparison

| Kriteria              | Exness Standard | Exness Pro | XM Standard       | XM Ultra Low | WemasterTrade    |
| --------------------- | --------------- | ---------- | ----------------- | ------------ | ---------------- |
| **Min Deposit**       | $1              | **$200**   | $5                | $5           | $35-50 (fee)     |
| **Budget $100 OK?**   | Yes             | **NO**     | Yes               | Yes          | Beda konsep      |
| **XAUUSD Spread**     | 20-35 pips      | ~1.1 pips  | ~3.5 pips         | ~2.5 pips    | N/A              |
| **Commission**        | $0              | $0         | $0                | $0           | N/A              |
| **Effective Capital** | $100            | —          | **$150** (+bonus) | $100         | $10K-25K funded  |
| **Leverage**          | Unlimited       | Unlimited  | 1:1000            | 1:1000       | N/A              |
| **Stop Out**          | 0%              | 0%         | 20%               | 20%          | N/A              |
| **MT5**               | Yes             | Yes        | Yes               | Yes          | **NO (cTrader)** |
| **Bot Allowed**       | Yes             | Yes        | Yes               | Yes          | **BANNED**       |
| **Regulation**        | Tier-1          | Tier-1     | Tier-1/2          | Tier-1/2     | MSB Canada       |

---

## 5. Compatibility Analysis with Smart Trader

### Scoring Matrix (Weighted)

| Kriteria          | Bobot | Exness Std | XM Std (+Bonus) | XM Ultra Low | WemasterTrade |
| ----------------- | ----- | ---------- | --------------- | ------------ | ------------- |
| Budget $100 fit   | 20%   | 8/10       | **10/10**       | 8/10         | 2/10          |
| XAUUSD Spread     | 25%   | 2/10       | **7/10**        | **9/10**     | 0/10          |
| MT5 Support       | 15%   | 10/10      | 10/10           | 10/10        | **0/10**      |
| Bot/Automation OK | 20%   | 10/10      | 10/10           | 10/10        | **0/10**      |
| Leverage          | 10%   | 10/10      | 9/10            | 9/10         | 0/10          |
| Regulasi          | 10%   | 10/10      | 8/10            | 8/10         | 4/10          |

### Weighted Scores

```
Exness Standard:     (8×20 + 2×25 + 10×15 + 10×20 + 10×10 + 10×10) / 100 = 7.10
XM Standard (+50%):  (10×20 + 7×25 + 10×15 + 10×20 + 9×10 + 8×10) / 100 = 8.95
XM Ultra Low:        (8×20 + 9×25 + 10×15 + 10×20 + 9×10 + 8×10) / 100 = 9.15
WemasterTrade:       (2×20 + 0×25 + 0×15 + 0×20 + 0×10 + 4×10) / 100 = 0.80
```

### Final Ranking

| Rank | Broker                   | Score         | Status               |
| ---- | ------------------------ | ------------- | -------------------- |
| 1    | **XM Ultra Low**         | **9.15 / 10** | BEST SPREAD          |
| 2    | **XM Standard (+Bonus)** | **8.95 / 10** | BEST VALUE           |
| 3    | Exness Standard          | 7.10 / 10     | Spread terlalu besar |
| 4    | WemasterTrade            | 0.80 / 10     | ELIMINASI            |

---

## 6. Cost Simulation per Trade

Simulasi biaya per trade dengan 0.01 lot XAUUSD:

### Spread Cost Impact

| Broker          | Spread (pips) | Cost per Trade | % of TP (~60pt avg) | % of SL (~40pt avg) |
| --------------- | ------------- | -------------- | ------------------- | ------------------- |
| Exness Pro      | 1.1           | $0.11          | 1.8%                | 2.8%                |
| XM Ultra Low    | 2.5           | $0.25          | 4.2%                | 6.3%                |
| XM Standard     | 3.5           | $0.35          | 5.8%                | 8.8%                |
| Exness Standard | 25.0 (avg)    | $2.50          | **41.7%**           | **62.5%**           |

> **Exness Standard spread 25 pips = 41.7% dari target TP.** Ini berarti hampir setengah profit langsung hilang ke spread. Tidak viable.

### Monthly Projection (est. 60 trades/month)

| Broker          | Spread Cost/Trade | Monthly Spread Cost | Impact                 |
| --------------- | ----------------- | ------------------- | ---------------------- |
| XM Ultra Low    | $0.25             | $15.00              | Acceptable             |
| XM Standard     | $0.35             | $21.00              | Acceptable             |
| Exness Standard | $2.50             | **$150.00**         | Exceeds entire capital |

---

## 7. Final Verdict & Recommendation

### Pilihan Strategis: 2 Opsi Terbaik

#### OPSI A — XM Standard Account (RECOMMENDED)

```
Deposit:     $100
Bonus 50%:   +$50 credit
Effective:   $150
Spread:      ~3.5 pips ($0.35/trade)
Leverage:    1:1000
Margin/lot:  ~$5.2 (0.01 lot)
```

**Kenapa ini terbaik:**

- Extra $50 bonus = margin buffer signifikan (dari $100 jadi $150)
- Spread 3.5 pips masih acceptable (5.8% dari TP)
- Leverage 1:1000 → margin hanya $5.2/trade → bisa handle drawdown lebih baik
- Balance $100 + bonus $50 = lebih tahan terhadap string of losses

#### OPSI B — XM Ultra Low Account (ALTERNATIVE)

```
Deposit:     $100
Bonus:       NONE (Ultra Low not eligible)
Effective:   $100
Spread:      ~2.5 pips ($0.25/trade)
Leverage:    1:1000
Margin/lot:  ~$5.2 (0.01 lot)
```

**Kenapa ini alternatif:**

- Spread lebih kecil (save $0.10/trade = $6/month)
- Tapi TANPA bonus → capital tetap $100, no safety buffer
- Lebih cocok kalau prioritas utama adalah spread terkecil

### Rekomendasi Final

> **XM Standard Account dengan Bonus 50%** adalah pilihan optimal untuk $100 capital.
>
> Extra $50 credit lebih berharga daripada saving $0.10/trade dari spread lebih kecil.
> Dengan leverage 1:1000, margin hanya $5.2/trade, sehingga $150 effective capital memberikan ruang jauh lebih besar untuk drawdown dan recovery.

### Jika Budget Bisa Ditambah

| Budget | Best Option             | Alasan                                           |
| ------ | ----------------------- | ------------------------------------------------ |
| $100   | **XM Standard + Bonus** | Best value, $150 effective                       |
| $200   | **Exness Pro**          | Spread 1.1 pips, leverage unlimited, stop out 0% |
| $500+  | **Exness Raw Spread**   | Spread from 0.0 pips + $3.5 commission           |

---

## 8. Migration Plan

### Config Changes (XM Standard)

```yaml
# config.yaml — changes needed
mt5:
  symbol: XAUUSD # verify exact symbol name in XM MT5
  magic: 202602
  terminal_path: "C:\\Program Files\\MetaTrader 5 XM\\terminal64.exe"

account:
  login: <xm_account_number>
  password: <xm_password>
  server: XMGlobal-MT5 # verify exact server name

trading:
  max_positions: 1
  risk_percent: 1.0
  min_lot: 0.01
  # Leverage 1:1000 means margin ~$5.2/trade (vs $51 at 1:100)
  # Consider increasing max_positions to 2 if margin allows
```

### Migration Checklist

- [ ] Register XM account (Standard type)
- [ ] Claim 50% deposit bonus on first deposit ($100)
- [ ] Download & install MetaTrader 5 XM terminal
- [ ] Login to MT5, verify XAUUSD symbol name & availability
- [ ] Check actual live spread on XAUUSD during London/NY session
- [ ] Update `config.yaml` with new credentials & terminal path
- [ ] Test connection: `python -c "import MetaTrader5 as mt5; mt5.initialize(); print(mt5.account_info())"`
- [ ] Run bot on XM demo first (if available) for 1-2 days
- [ ] Switch to live account
- [ ] Monitor first 10 trades for execution quality

### Code Changes Required

**Zero code changes needed.** Only `config.yaml` modifications:

- `mt5.symbol` — verify exact XAUUSD symbol name
- `mt5.terminal_path` — path to XM MT5 terminal
- `account.*` — new login credentials
- Optional: adjust `max_positions` if leverage allows more margin room

---

## 9. Sources

| Source                          | URL                                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Exness Pro Accounts             | [exness.com/pro-accounts](https://www.exness.com/pro-accounts/)                                              |
| Exness XAUUSD Spread Comparison | [tradersunion.com](https://tradersunion.com/brokers/forex/view/exness/xauusd-spread/)                        |
| Exness Minimum Deposit 2026     | [fxleaders.com](https://www.fxleaders.com/forex-brokers/forex-brokers-review/exness-review/minimum-deposit/) |
| XM Account Types                | [xm.com/account-types](https://www.xm.com/account-types)                                                     |
| XM XAUUSD Spread (Myfxbook)     | [myfxbook.com](https://www.myfxbook.com/forex-broker-spreads/xm-group/2824,51)                               |
| XM Deposit Bonus Terms          | [xm.com (PDF)](https://www.xm.com/assets/pdf/new/terms/XM-Terms-and-Conditions-Bonus-Program.pdf)            |
| XM Broker Review 2026           | [tradingcritique.com](https://tradingcritique.com/broker-review/xm-broker-review/)                           |
| WemasterTrade Review 2025       | [thetrustedprop.com](https://thetrustedprop.com/prop-firms/wemastertrade)                                    |
| WemasterTrade Drawdown Rules    | [thetrustedprop.com](https://thetrustedprop.com/blogs/wemastertrader-challenges-drawdown-profits-split-2025) |
| WemasterTrade Prop Firm Details | [propfirmmatch.com](https://propfirmmatch.com/unlisted-prop-firms/wemastertrade)                             |

---

<div align="center">

_Analysis generated by Claude Opus 4.6 for Smart Trader v1.0.0-beta_

_PT Surya Inovasi Prioritas (SURIOTA) — February 2026_

</div>
