# Broker Comparison — XAUUSD SMC Trading Bot

> **Author**: Gifari K Suryo — PT Surya Inovasi Prioritas (SURIOTA)
> **Date**: 2026-02-24
> **Budget**: $100 real money
> **Bot Version**: V4.0.0 (SMC + Adaptive Scoring)

---

## Table of Contents

- [Bot Requirements](#bot-requirements)
- [Broker Profiles](#broker-profiles)
  - [1. Exness](#1-exness)
  - [2. XM](#2-xm)
  - [3. WeMasterTrade (Prop Firm)](#3-wemastertrade-prop-firm)
- [Head-to-Head Comparison](#head-to-head-comparison)
- [Compatibility Analysis](#compatibility-analysis)
- [Cost Simulation (30 Days)](#cost-simulation-30-days)
- [Migration Effort](#migration-effort)
- [Risk Assessment](#risk-assessment)
- [Verdict & Roadmap](#verdict--roadmap)
- [Sources](#sources)

---

## Bot Requirements

Sebelum membandingkan broker, berikut parameter kritis yang dibutuhkan bot:

| Parameter | Requirement | Alasan |
|-----------|-------------|--------|
| **Instrument** | XAUUSD (Gold) | Satu-satunya pair yang di-trade |
| **Min Lot** | 0.01 | Fixed lot, tidak bisa lebih kecil |
| **Leverage** | >= 1:100 | Margin ~$51/trade di 1:100 pada Gold ~$5180 |
| **Platform** | MetaTrader 5 | Bot menggunakan MT5 Python API |
| **Execution** | Market order | Bot kirim order instan, bukan pending |
| **Stop Out** | Serendah mungkin | Akun $100 rentan margin call |
| **Spread** | < 3.0 pips ideal | SL rata-rata ~35 pips, spread > 10% SL = masalah |
| **Swap** | Rendah/free | Bot bisa hold overnight (rata-rata 150 menit per trade) |
| **EA/Bot** | Diizinkan | Bot berjalan 24/5 otomatis |

### Current Bot Config

```yaml
symbol: XAUUSDm          # Exness-specific (3-digit, point=0.001)
fixed_lot: 0.01           # ~$1/pip on XAUUSD
max_open_positions: 1      # Margin limit $100 account
max_drawdown_percent: 30   # Single trade risk ~21%
atr_sl_mult: 2.60-4.66    # Regime-adaptive SL
atr_tp_mult: 6.02          # Fixed TP multiplier
```

---

## Broker Profiles

### 1. Exness

**Website**: [exness.com](https://www.exness.com/)
**Regulasi**: CySEC, FCA, FSA (Seychelles), FSCA
**Status**: Currently used (Demo Account 413371434)

#### Account Types

| | Standard | Pro | Raw Spread | Zero |
|--|----------|-----|------------|------|
| **Min Deposit** | $10 | $200 | $200 | $200 |
| **XAUUSD Spread** | 1.8-2.5 pips | ~0.3 pips | from 0.0 pips | 0.0 pips |
| **Commission** | $0 | $0 | $3.5/side/lot | from $0.2/side |
| **Leverage** | Up to Unlimited | Up to Unlimited | Up to Unlimited | Up to Unlimited |
| **Stop Out** | **0%** | **0%** | 0% | 0% |
| **Margin Call** | 60% | 30% | 30% | 30% |
| **Execution** | Market | Instant | Market | Market |
| **Swap-Free** | Available | Available | Available | Available |

#### Key Highlights

- **0% Stop Out** — posisi tidak akan di-close otomatis sampai equity benar-benar habis
- **Unlimited Leverage** — margin requirement bisa sangat rendah (di atas equity threshold tertentu)
- **Swap-Free** tersedia — tidak ada biaya overnight untuk negara tertentu
- **Symbol**: `XAUUSDm` (3-digit precision, point = 0.001)
- **Spread stabil** bahkan saat high-impact news (verified study Exness Jan-Jun 2025)

#### Relevant for $100 Budget

Hanya **Standard Account** yang affordable ($10 min deposit). Pro/Raw/Zero butuh $200.

| Metric | Standard Account |
|--------|-----------------|
| Spread cost per trade (0.01 lot) | 2.0 pips x $0.01 = **$0.02** |
| Margin per trade (1:100) | ~$51.80 |
| Free margin after 1 trade | ~$48.20 |
| Max drawdown before stop out | **$100** (0% stop out!) |

---

### 2. XM

**Website**: [xm.com](https://www.xm.com/)
**Regulasi**: CySEC, ASIC, IFSC (Belize), DFSA
**Referral**: [xmidbroker.org (affiliate link)](https://www.xmidbroker.org/referral?token=KSiOv2zBn6N62xKz-EtbWg)

#### Account Types

| | Standard | Ultra Low | Zero |
|--|----------|-----------|------|
| **Min Deposit** | $5 | $5 | $5 |
| **XAUUSD Spread** | 3.0-4.5 pips | 1.6-2.0 pips | Lower + commission |
| **Commission** | $0 | $0 | $3.5/side/lot |
| **Leverage** | Up to 1:1000 | Up to 1:1000 | Up to 1:1000 |
| **Stop Out** | **20%** | **20%** | **20%** |
| **Margin Call** | 100% | 100% | 100% |
| **Execution** | Market | Market | Market |
| **Swap** | ~$0.08/day (0.01 lot) | ~$0.08/day | ~$0.08/day |

#### Bonus & Promotions (2026)

| Bonus | Detail |
|-------|--------|
| No-Deposit Bonus | **$30** gratis untuk akun baru |
| Deposit Bonus | Tiered: **100%** (first $500) + 50% + 20%, up to $10,600 |
| Weekly Promo (Feb-Mar 2026) | Up to $8,750/week, max $52,500 total |

**Contoh**: Deposit $100 → dapat +$100 bonus credit = **$200 effective capital**

> **PENTING**: Bonus credit umumnya **bukan equity**. Credit tidak melindungi dari margin call.
> Saat real equity turun di bawah threshold, bonus bisa di-revoke oleh broker.

#### Key Highlights

- **Spread Ultra Low lebih ketat** dari Exness Standard (1.6-2.0 vs 1.8-2.5 pips)
- **Bonus menarik** untuk menambah capital buffer
- **$5 minimum deposit** — sangat rendah
- **Symbol**: kemungkinan `GOLD`, `XAUUSDmicro`, atau `XAUUSD` (BUKAN `XAUUSDm`)
- **Point precision** perlu diverifikasi (2 digit vs 3 digit)

#### Relevant for $100 Budget

| Metric | Ultra Low Account |
|--------|------------------|
| Spread cost per trade (0.01 lot) | 1.8 pips x $0.01 = **$0.018** |
| Margin per trade (1:1000) | ~$5.18 |
| Free margin after 1 trade | ~$94.82 |
| Max drawdown before stop out | ~**$90** (20% stop out, margin $5.18) |
| Bonus impact | +$100 credit (doesn't protect margin call) |

> Dengan leverage 1:1000, margin per trade hanya ~$5. Namun stop out 20% tetap berlaku
> pada **total equity**, bukan per-posisi.

---

### 3. WeMasterTrade (Prop Firm)

**Website**: [wemastertrade.com](https://wemastertrade.com/package-comparison/)
**Tipe**: Proprietary Trading Firm (BUKAN broker)
**Model**: Challenge → Funded Account → Profit Split

#### Package Options

| | 51010 | 510zero | 51010-NoPC | Customize |
|--|-------|---------|------------|-----------|
| **Fee** | ~$50+ | ~$50+ | ~$50+ | **From $35** |
| **Account Size** | $10,000+ | $10,000+ | $10,000+ | $10,000+ |
| **Daily DD** | 5% | 5% | 5% | 5% |
| **Max DD** | 10% | 10% | 10% | Customizable |
| **Profit Target** | 10% | 0% | 10% (no PC) | Custom |
| **Profit Consistency** | Required | No | No | Custom |

#### Challenge Phases

```
Phase 1 (Challenge)
├── Profit Target: 8%
├── Daily Drawdown: 5% max
├── Max Drawdown: 10% max
├── Time Limit: None
└── Profit Split: 0% (evaluation only)

Phase 2 (Stabilization)
├── Profit Target: 6%
├── Daily Drawdown: 5% max
├── Max Drawdown: 10% max
├── Time Limit: None
└── Profit Split: 30%

Funded Account
├── No Profit Target
├── Daily Drawdown: 5% max
├── Max Drawdown: 10% max
├── Risk per Trade: 1% max
└── Profit Split: 50% → 70% → 90% (progressive)
```

#### Key Highlights

- **Trade with $10k+ using only $35-50 initial fee**
- **EA/Bot allowed** — news trading OK, overnight OK
- **Platform**: MetaTrader 5 & MatchTrader
- **Static drawdown** — max DD fixed, tidak trailing
- **Scaling** up to $1,000,000 virtual capital
- **Profit split progressive**: 50% (1st payout) → 70% (2nd) → 90% (3rd+)

#### Relevant for $100 Budget

| Metric | Customize ($35) on $10k Account |
|--------|-------------------------------|
| Initial cost | **$35** (challenge fee, non-refundable if fail) |
| Remaining budget | $65 for retry or other broker |
| Risk per trade | 1% = **$100** on $10k |
| Daily DD limit | 5% = **$500** on $10k |
| Max DD limit | 10% = **$1,000** on $10k |
| 0.01 lot risk (SL 35 pips) | ~$35 = 0.35% (within 1% limit) |
| Potential lot size | 0.02-0.03 (to utilize 1% = $100 risk) |

---

## Head-to-Head Comparison

### Critical Parameters

| Parameter | Exness Standard | XM Ultra Low | WeMasterTrade |
|-----------|----------------|--------------|---------------|
| **Type** | Retail Broker | Retail Broker | Prop Firm |
| **Your Capital at Risk** | $100 | $100 | $35-50 (fee only) |
| **Trading Capital** | $100 | $100 (+bonus credit) | $10,000+ |
| **XAUUSD Spread** | 1.8-2.5 pips | 1.6-2.0 pips | Depends on LP |
| **Commission** | $0 | $0 | $0 |
| **Stop Out Level** | **0%** | 20% | N/A (DD rules) |
| **Max Leverage** | Unlimited | 1:1000 | Set by firm |
| **Swap-Free** | Yes | No | Depends |
| **Bonus** | None | Up to 100% deposit | N/A |
| **MT5 Support** | Yes | Yes | Yes |
| **EA/Bot Allowed** | Yes | Yes | Yes |
| **Profit** | 100% yours | 100% yours | 50-90% yours |
| **Config Changes** | **None** | Moderate | Significant |

### Spread Cost Impact (per trade, 0.01 lot)

| Broker | Avg Spread | Cost/Trade | % of SL (35 pips) |
|--------|-----------|-----------|-------------------|
| Exness Standard | 2.0 pips | $0.020 | 5.7% |
| XM Ultra Low | 1.8 pips | $0.018 | 5.1% |
| Exness Pro ($200) | 0.3 pips | $0.003 | 0.9% |

> Perbedaan spread Exness Standard vs XM Ultra Low: **$0.002 per trade** — negligible.

---

## Compatibility Analysis

### Exness Standard — Compatibility Score: 10/10

```
[OK] Symbol XAUUSDm .............. Already configured
[OK] Point precision 0.001 ...... Already configured
[OK] Lot size 0.01 .............. Supported
[OK] MT5 API .................... Same as demo
[OK] Stop out 0% ............... Bot max_drawdown 30% safe
[OK] Leverage >= 1:100 ......... Unlimited available
[OK] EA/Bot allowed ............. Yes
[OK] Swap-free .................. Available
[OK] Market execution ........... Supported
[OK] Config changes needed ...... ZERO
```

### XM Ultra Low — Compatibility Score: 6/10

```
[OK] Lot size 0.01 .............. Supported
[OK] MT5 API .................... Supported
[OK] Leverage 1:1000 ............ More than enough
[OK] EA/Bot allowed ............. Yes
[OK] Market execution ........... Supported
[!!] Symbol name ................ MUST change (not XAUUSDm)
[!!] Point precision ............ MUST verify (2 vs 3 digit)
[!!] SL/TP calculation .......... MUST re-validate
[XX] Stop out 20% ............... Bot max_drawdown 30% CONFLICTS
[XX] Swap fee ................... ~$0.08/day (small but exists)
```

**Required Code Changes for XM:**
1. `config/settings.yaml` → `symbol: "GOLD"` (or verified name)
2. `src/core/mt5_connector.py` → verify `point` value
3. `src/risk_management/` → adjust for 20% stop out reality
4. `config/risk_config.yaml` → lower `max_drawdown_percent` to ~15%
5. Full regression test with new symbol
6. MT5 terminal installation for XM server

### WeMasterTrade — Compatibility Score: 3/10

```
[OK] MT5 API .................... Supported
[OK] EA/Bot allowed ............. Yes
[OK] Lot size 0.01 .............. Supported (but should be higher)
[XX] max_drawdown 30% ........... VIOLATES 10% rule
[XX] max_daily_loss 5% .......... Borderline (currently 5%)
[XX] Single trade risk ~21% ..... VIOLATES 1% per trade rule
[XX] No guaranteed profit ........ Challenge fee at risk
[XX] Profit split 50% start ..... Earnings halved
[!!] Symbol unknown .............. Need to verify
[!!] Risk config ................. COMPLETE overhaul needed
```

**Required Code Changes for WeMasterTrade:**
1. `max_drawdown_percent: 30` → `8` (safety buffer under 10%)
2. `max_daily_loss_percent: 5` → `4` (safety buffer under 5%)
3. `fixed_lot: 0.01` → calculate dynamically (1% of $10k = $100 risk)
4. `max_consecutive_losses: 3` → `2` (tighter protection)
5. Add daily P/L tracking with hard circuit breaker
6. Add per-trade risk validation against account balance
7. Symbol, point precision, server configuration
8. Full backtest with new risk parameters to verify profitability

---

## Cost Simulation (30 Days)

Asumsi: 4 trades/week, 16 trades/month, avg hold 150 min (~2.5 jam)

### Exness Standard

| Item | Cost |
|------|------|
| Spread (16 trades x $0.02) | $0.32 |
| Commission | $0 |
| Swap | $0 (swap-free) |
| **Total Monthly Cost** | **~$0.32** |

### XM Ultra Low

| Item | Cost |
|------|------|
| Spread (16 trades x $0.018) | $0.29 |
| Commission | $0 |
| Swap (~5 overnight holds x $0.08) | $0.40 |
| **Total Monthly Cost** | **~$0.69** |

### WeMasterTrade Customize

| Item | Cost |
|------|------|
| Challenge Fee (one-time) | $35-50 |
| Spread (depends on LP) | ~$0.30 |
| Commission | Varies |
| Profit split (50% of earnings) | **50% of all profit** |
| **If fail challenge** | **-$35 to -$50 lost** |

---

## Migration Effort

### Exness Standard (Demo → Real)

```
Effort Level: MINIMAL (< 1 jam)

1. Buka akun real Exness Standard ........... 10 min
2. Deposit $100 ............................. 5 min
3. Update .env (login, password, server) .... 2 min
4. Test MT5 connection ...................... 5 min
5. Run validate_deploy.py ................... 2 min
6. Start bot ................................ 1 min

Config changes: .env only (credentials)
Code changes: NONE
Risk: Very low
```

### XM Ultra Low

```
Effort Level: MODERATE (3-5 jam)

1. Buka akun XM Ultra Low .................. 15 min
2. Install MT5 terminal for XM server ....... 10 min
3. Deposit $100, claim bonus ................ 10 min
4. Identify correct XAUUSD symbol name ...... 15 min
5. Verify point precision & contract size ... 30 min
6. Update config (symbol, server, creds) .... 15 min
7. Update SL/TP calculation if needed ....... 1-2 hr
8. Run full test suite ...................... 30 min
9. Run validate_deploy.py ................... 10 min
10. Monitor first 24 hours closely .......... ongoing

Config changes: symbol, server, credentials, possibly risk params
Code changes: Possible (point precision, SL/TP math)
Risk: Medium (bugs from config mismatch)
```

### WeMasterTrade

```
Effort Level: SIGNIFICANT (1-2 minggu)

1. Register WeMasterTrade ................... 15 min
2. Purchase Customize challenge ($35-50) .... 10 min
3. Get MT5 credentials for challenge ........ 30 min
4. Install MT5 for WeMasterTrade server ..... 10 min
5. Identify symbol name & specifications .... 30 min
6. OVERHAUL risk_config.yaml ................ 2-3 hr
7. Add daily P/L circuit breaker code ....... 2-3 hr
8. Add per-trade risk % validator ........... 1-2 hr
9. Run backtest with new risk params ........ 1 hr
10. Verify PF still positive with tight DD .. 1 hr
11. Run challenge Phase 1 (8% target) ....... 1-4 weeks
12. Run challenge Phase 2 (6% target) ....... 1-2 weeks
13. Funded account setup .................... 30 min

Config changes: Almost everything (risk, symbol, server)
Code changes: Significant (risk management overhaul)
Risk: HIGH (fee lost if challenge fails, tight DD rules)
```

---

## Risk Assessment

### Scenario: Single Large Loss (SL Hit, ~$35-50 loss)

| Broker | Consequence |
|--------|------------|
| **Exness** | Balance drops to $50-65. Bot continues. 0% stop out = no auto-close. |
| **XM** | Balance drops to $50-65. If equity < margin (~$5 at 1:1000), next trade blocked. If equity < 20% margin, positions auto-closed. Bonus credit may be revoked. |
| **WeMasterTrade** | $35 loss on $10k = 0.35%. Safe. But 3 consecutive losses ($105) = 1.05% daily. Still within 5% daily limit. |

### Scenario: 3 Consecutive Losses (~$105 total loss)

| Broker | Consequence |
|--------|------------|
| **Exness** | Balance ~$0 (wiped out). Deposit ulang. |
| **XM** | Balance ~$0 (wiped out). Bonus revoked. Deposit ulang. |
| **WeMasterTrade** | $105 loss on $10k = 1.05%. Still safe. Account survives. |

### Scenario: Bot has 75% win rate, runs 1 month (16 trades)

| Broker | Expected P/L | Net to You |
|--------|-------------|------------|
| **Exness** | +$82 (based on historical) | **+$82** (100%) |
| **XM** | +$82 - $0.37 costs | **+$81.63** (100%) |
| **WeMasterTrade** | +$820 (10x capital) | **+$410** (50% split) |

> WeMasterTrade lebih profitable **jika** challenge berhasil dan bot maintain strict DD rules.

---

## Verdict & Roadmap

### Rekomendasi Final

| Phase | Action | Timeline |
|-------|--------|----------|
| **Phase 1 (NOW)** | **Exness Standard Real** — $100 deposit, zero config changes, start trading immediately | Minggu ini |
| **Phase 2 (After 2-4 weeks profit)** | Upgrade ke **Exness Pro** saat balance >= $200 (spread 0.3 pips, no commission) | 2-4 minggu |
| **Phase 3 (Optional)** | Coba **WeMasterTrade Challenge** ($35) dengan bot yang sudah proven profitable, tune risk params | 1-2 bulan |
| **Phase 4 (Scaling)** | Jika WeMasterTrade funded, run parallel: Exness real + WeMasterTrade funded | 2-3 bulan |

### Kenapa Exness #1?

1. **0% Stop Out** — faktor paling kritis untuk akun $100. Posisi survive sampai equity $0. Di XM, auto-close di 20% margin.

2. **Zero Migration** — bot langsung jalan. Setiap perubahan config = risk bug baru. Pada real money, stability > optimization.

3. **Proven Track Record** — demo kita sudah 4 trades: 3W/1L, PF 63.52, +$82.53. Environment yang sama = hasil predictable.

4. **Spread difference negligible** — Exness 2.0 vs XM 1.8 = $0.002/trade. Pada 16 trades/month = **$0.03 difference**. Tidak worth the migration risk.

### Kenapa BUKAN XM?

- **20% stop out** pada akun $100 = death sentence. Satu floating loss $50 + margin $51 = equity $49, margin level 96% → margin call triggered.
- Bonus credit umumnya **tidak** menambah margin level. Perlu baca terms sangat detail.
- Config migration risk pada real money = unacceptable saat bot belum battle-tested.

### Kenapa BUKAN WeMasterTrade (Sekarang)?

- Bot belum proven di real money. Challenge fee $35-50 = gambling.
- Risk params butuh complete overhaul. Untested config pada real challenge = kemungkinan fail tinggi.
- **TAPI**: setelah 1-2 bulan profit konsisten di Exness real, WeMasterTrade jadi opsi sangat menarik (trade $10k+ dengan modal $35).

---

## Sources

- [Exness Account Types](https://issuu.com/exnesstrading/docs/exness_account/s/64035309)
- [Exness XAUUSD Trading](https://www.exness.com/commodities/xauusd/)
- [Exness Margin Call & Stop Out](https://get.exness.help/hc/en-us/articles/360014693340-Margin-call-and-stop-out-levels-by-account-type)
- [Exness Swap-Free Status](https://get.exness.help/hc/en-us/articles/4402341895570-Swap-free-status)
- [XM Account Types](https://www.xm.com/account-types)
- [XM XAUUSD Spread Review](https://brokerchooser.com/broker-reviews/xm-review/xauusd-spread)
- [XM Margin & Leverage](https://www.xm.com/margin-and-leverage)
- [XM 2026 Bonus Promotion](https://www.prnewswire.com/news-releases/xm-launches-promotion-offering-traders-up-to-52-500-in-bonuses-302683721.html)
- [XM Fees & Spreads](https://www.bestbrokers.com/reviews/xm/spreads-fees-and-commissions/)
- [WeMasterTrade Review](https://thetrustedprop.com/prop-firms/wemastertrade)
- [WeMasterTrade Drawdown Rules](https://thetrustedprop.com/blogs/wemastertrader-challenges-drawdown-profits-split-2025)
- [WeMasterTrade Packages](https://reviewpropfirm.com/prop-firms/wemastertrade/)
- [WeMasterTrade Payouts](https://thetrustedprop.com/blogs/wemastertrade-payouts-profit-split-withdrawals-eligibility-tips)

---

*Document generated: 2026-02-24 | PT Surya Inovasi Prioritas (SURIOTA)*
