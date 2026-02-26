<div align="center">

# XM Broker — Migration Guide & Reference

![XM](https://img.shields.io/badge/Broker-XM_Global-red?style=for-the-badge)
![Account](https://img.shields.io/badge/Account-Standard-blue?style=for-the-badge)
![Bonus](https://img.shields.io/badge/Bonus-50%25_First_Deposit-green?style=for-the-badge)
![Budget](https://img.shields.io/badge/Budget-$100+$50-gold?style=for-the-badge)

**Smart Trader — XAUUSD Bot Migration dari IC Markets Demo ke XM Live**

---

**Author:** Gifari K Suryo — CEO & Founder, Lead R&D
**Organization:** PT Surya Inovasi Prioritas (SURIOTA)
**Date:** February 26, 2026
**Analyst:** Claude Opus 4.6

</div>

---

## Table of Contents

1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Profil XM Broker](#2-profil-xm-broker)
3. [Tipe Akun — Analisa Detail](#3-tipe-akun--analisa-detail)
4. [Bonus Program 50%](#4-bonus-program-50)
5. [XAUUSD (GOLD) — Spesifikasi Trading](#5-xauusd-gold--spesifikasi-trading)
6. [Leverage & Margin](#6-leverage--margin)
7. [Spread Analysis](#7-spread-analysis)
8. [Swap & Overnight Fees](#8-swap--overnight-fees)
9. [Trading Hours](#9-trading-hours)
10. [Deposit & Withdrawal](#10-deposit--withdrawal)
11. [Regulasi & Keamanan Dana](#11-regulasi--keamanan-dana)
12. [MT5 Platform — Setup](#12-mt5-platform--setup)
13. [XM vs IC Markets vs Exness — Perbandingan](#13-xm-vs-ic-markets-vs-exness--perbandingan)
14. [Bot Configuration — Migration Checklist](#14-bot-configuration--migration-checklist)
15. [Config Template (XM)](#15-config-template-xm)
16. [Risk Management untuk $150](#16-risk-management-untuk-150)
17. [Post-Migration Verification](#17-post-migration-verification)
18. [Strategi Bonus — Optimasi](#18-strategi-bonus--optimasi)
19. [Known Differences & Adjustments](#19-known-differences--adjustments)
20. [Troubleshooting](#20-troubleshooting)
21. [Sources](#21-sources)

---

## 1. Ringkasan Eksekutif

| Item | Detail |
|------|--------|
| **Broker** | XM Global (Trading Point of Financial Instruments) |
| **Akun Dipilih** | **Standard Account** (wajib untuk bonus) |
| **Deposit** | $100 |
| **Bonus 50%** | +$50 credit |
| **Total Equity** | **$150** |
| **Instrumen** | GOLD (XAUUSD) |
| **Platform** | MetaTrader 5 |
| **Leverage** | 1:1000 |
| **Spread GOLD** | ~3.5 pips avg (Standard) |
| **Commission** | $0 |
| **Margin per 0.01 lot** | ~$2.90 |
| **Code Changes** | **Nol** — config.yaml only |

### Alasan Memilih XM Standard (bukan Ultra Low)

```
Ultra Low: spread 2.0 pips, TAPI bonus = $0
Standard:  spread 3.5 pips, bonus = +$50

Selisih spread per trade (0.01 lot): $0.015
Untuk "ganti" $50 bonus: butuh 3,333 trades
Rata-rata 60 trades/bulan = 55 BULAN untuk break-even

>>> BONUS 50% JAUH LEBIH BERHARGA untuk modal $100
```

---

## 2. Profil XM Broker

### Company Info

| Item | Detail |
|------|--------|
| Nama Legal | Trading Point of Financial Instruments Ltd |
| Berdiri | 2009 |
| Kantor Pusat | Limassol, Cyprus |
| Total Client | 10+ juta (190+ negara) |
| Website | [xm.com](https://www.xm.com) |

### Regulasi (7 Yurisdiksi)

| Regulator | Entitas | License | Tier |
|-----------|---------|---------|------|
| **CySEC** (Cyprus/EU) | Trading Point of Financial Instruments Ltd | 120/10 | Tier-1 |
| **ASIC** (Australia) | Trading Point of Financial Instruments Pty Ltd | 443670 | Tier-1 |
| **DFSA** (Dubai) | Trading Point MENA Ltd | F003484 | Tier-2 |
| **FSA** (Seychelles) | XM Global Limited | SD010 | Tier-3 |
| **IFSC** (Belize) | XM Global Limited | 60/354/TS/19 | Tier-3 |
| **FSCA** (South Africa) | - | - | Tier-2 |
| **SCA** (UAE) | - | - | Tier-2 |

> **Indonesia**: Kemungkinan masuk di bawah entitas **IFSC Belize** atau **FSA Seychelles**.

### Keamanan Dana

- **Segregated Accounts** — dana client terpisah dari dana operasional
- **Negative Balance Protection** — tidak bisa rugi lebih dari deposit
- **Investor Compensation Fund (ICF)** — kompensasi hingga EUR 20,000 (CySEC entity)
- **Execution**: 99.35% order dieksekusi dalam < 1 detik, no requotes

---

## 3. Tipe Akun — Analisa Detail

### Perbandingan Lengkap

| Feature | **Micro** | **Standard** | **Ultra Low Std** | **Ultra Low Micro** |
|---------|-----------|-------------|-------------------|---------------------|
| Min Deposit | $5 | $5 | $5 | $5 |
| Spread GOLD | ~3.5 pips | ~3.5 pips | ~1.6-2.0 pips | ~1.6-2.0 pips |
| Commission | $0 | $0 | $0 | $0 |
| Leverage | 1:1000 | 1:1000 | 1:1000 | 1:1000 |
| Contract Size (GOLD) | **1 oz/lot** | **100 oz/lot** | 100 oz/lot | 100 oz/lot |
| Min Lot (MT4) | 0.01 | 0.01 | 0.01 | 0.01 |
| Min Lot (MT5) | **0.1** | **0.01** | 0.01 | 0.01 |
| Stop Out Level | 20% | 20% | 20% | 20% |
| **Bonus Eligible** | **YES** | **YES** | **NO** | **NO** |
| **Swap-Free** | NO | NO | **YES** | **YES** |
| Execution | Market | Market | Market | Market |
| Hedging | Yes | Yes | Yes | Yes |
| EA/Bot Allowed | Yes | Yes | Yes | Yes |

### Kenapa Standard (Bukan Micro)

| Aspek | Standard | Micro |
|-------|----------|-------|
| Contract size GOLD | 100 oz/lot | 1 oz/lot |
| 0.01 lot = | **1 oz** ($2.90 margin) | 0.01 oz ($0.029 margin) |
| Profit per pip (0.01 lot) | **~$0.01** | ~$0.0001 |
| Min lot di MT5 | **0.01** | 0.1 |
| Meaningful trading | Yes | Butuh 1.00 lot Micro = 0.01 Standard |

**Micro terlalu kecil** — 0.01 lot Micro = 0.01 oz gold, profit/loss hampir nol. Untuk trading yang bermakna di $150 balance, **Standard** adalah pilihan tepat.

### Kenapa Standard (Bukan Ultra Low)

| Aspek | Standard + Bonus | Ultra Low |
|-------|-----------------|-----------|
| Effective capital | **$150** (+50%) | $100 |
| Spread GOLD | 3.5 pips | 2.0 pips |
| Cost per trade | $0.035 | $0.020 |
| Swap | Ada (negatif) | **Swap-free** |
| Bonus | **+$50** | $0 |
| Break-even trades | - | **3,333** trades untuk ganti $50 |

**Verdict: Standard + Bonus 50%** untuk modal $100. Bonus $50 = 50% tambahan equity, jauh lebih impactful dari saving $0.015/trade.

---

## 4. Bonus Program 50%

### Struktur Bonus

| Tier | Deposit | Bonus | Max Bonus |
|------|---------|-------|-----------|
| **1st Deposit** | $5 - $1,000 | **50%** | **$500** |
| Subsequent | Any | 20% | $4,500 |
| **Total Lifetime** | - | - | **$5,000** |

### Kalkulasi untuk $100

```
Deposit:         $100.00
Bonus 50%:     +  $50.00 (credit)
─────────────────────────
Total Equity:    $150.00
```

### Syarat & Ketentuan (PENTING)

| Syarat | Detail |
|--------|--------|
| Eligible accounts | **Standard & Micro ONLY** (Ultra Low = NO) |
| Bonus withdrawable? | **TIDAK** — hanya sebagai margin credit |
| Profit withdrawable? | **YA** — profit dari trading bebas ditarik |
| Volume requirement | **TIDAK ADA** — tidak perlu trade X lot untuk withdraw profit |
| Proportional removal | Withdraw dana → bonus berkurang proporsional |
| Opt-out | Bisa kapan saja dari Members Area |

### Contoh Proportional Removal

```
Skenario: Deposit $100, Bonus $50, Total $150

Withdraw $30 (30% of $100 deposit):
├── Bonus dikurangi 30%: $50 × 30% = -$15
├── Sisa bonus: $35
├── Sisa balance: $70 + $35 credit
└── Total equity: $105

Withdraw $100 (100% of deposit):
├── Bonus dikurangi 100%: $50 × 100% = -$50
├── Sisa bonus: $0
└── SEMUA bonus hilang
```

### Rules Penting

1. **Setiap withdrawal** (berapapun) mengurangi bonus proporsional
2. Bonus **otomatis masuk** saat deposit pertama (jika claim di Members Area)
3. Internal transfer antar akun = **bonus dihapus**
4. Jika equity turun ke level bonus saja (balance = $0), **posisi TIDAK auto-close** — bonus masih bisa dipakai sebagai margin sampai stop out 20%
5. Bonus **tidak dihitung** dalam perhitungan profit/loss — hanya sebagai margin buffer

---

## 5. XAUUSD (GOLD) — Spesifikasi Trading

### Instrument Specification

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Symbol** | **GOLD** | Bukan XAUUSD/XAUUSDm — perlu verifikasi di MT5 |
| Description | Gold vs US Dollar |  |
| Contract Size | **100 oz per standard lot** | Sama dengan Exness/IC Markets |
| Min Lot | 0.01 (Standard) | = 1 oz gold |
| Max Lot | 50 | |
| Lot Step | 0.01 | |
| **Digits** | **2** (perlu verifikasi) | Exness XAUUSDm = 3 digits |
| **Point** | **0.01** (perlu verifikasi) | Exness = 0.001 |
| Pip Value (0.01 lot) | ~$0.01 per pip | |
| Tick Size | 0.01 | |
| Tick Value | $0.01 per 0.01 lot | |

> **KRITIS**: Digits dan Point HARUS diverifikasi setelah login MT5. Jika XM pakai 2 digit (xx.xx) vs Exness 3 digit (xx.xxx), semua kalkulasi SL/TP dalam point BERBEDA.

### Digit Impact pada Bot

```python
# Exness XAUUSDm (3 digits, point=0.001):
#   ATR = 20.0 (dalam "points" = 20.000 di 3 digit)
#   SL = 2.0 × ATR = 40 points = 4.0 pips

# XM GOLD (2 digits, point=0.01):
#   ATR = 20.0 (dalam pips langsung)
#   SL = 2.0 × ATR = 40 pips
#   PERLU VERIFIKASI — mungkin unit berbeda

# IMPORTANT: Cek mt5.symbol_info("GOLD").digits dan .point setelah login
```

### Spread Detail

| Kondisi | Typical Spread | Peak Spread |
|---------|---------------|-------------|
| London Session (07:00-16:00 UTC) | 2.5 - 3.5 pips | 5-8 pips |
| NY Session (12:00-21:00 UTC) | 2.5 - 3.5 pips | 5-8 pips |
| London-NY Overlap (12:00-16:00 UTC) | **2.0 - 3.0 pips** | 4-6 pips |
| Asian Session (00:00-07:00 UTC) | 4.0 - 6.0 pips | 8-12 pips |
| News Events (NFP, FOMC, CPI) | 8 - 25+ pips | 50+ pips |
| Daily Maintenance | **SPREAD WIDENED** | Market closed |

---

## 6. Leverage & Margin

### Leverage Tiers (berdasarkan Equity)

| Account Equity | Max Leverage |
|---------------|-------------|
| $5 — $40,000 | **1:1000** |
| $40,001 — $80,000 | 1:500 |
| $80,001 — $200,000 | 1:200 |
| $200,001+ | 1:100 |

> Dengan equity $150, leverage = **1:1000** (tier tertinggi).

### Margin Calculation (Gold ~$2,900/oz, Feb 2026)

```
Formula: Margin = (Lots × Contract_Size × Price) / Leverage

0.01 lot: (0.01 × 100 × 2900) / 1000 = $2.90
0.02 lot: (0.02 × 100 × 2900) / 1000 = $5.80
0.05 lot: (0.05 × 100 × 2900) / 1000 = $14.50
0.10 lot: (0.10 × 100 × 2900) / 1000 = $29.00
```

### Margin vs Balance Analysis

| Lot Size | Margin | % of $150 | Free Margin | Max Loss Before Stop Out |
|----------|--------|-----------|-------------|--------------------------|
| **0.01** | **$2.90** | **1.9%** | **$147.10** | ~$147 |
| 0.02 | $5.80 | 3.9% | $144.20 | ~$144 |
| 0.05 | $14.50 | 9.7% | $135.50 | ~$135 |
| 0.10 | $29.00 | 19.3% | $121.00 | ~$121 |

> **0.01 lot hanya butuh $2.90 margin** — sangat kecil. Secara teori bisa buka banyak posisi, TAPI risk management tetap batasi max 1-2 posisi.

### Stop Out Level

| Item | Value |
|------|-------|
| Margin Call | 50% |
| **Stop Out** | **20%** |

```
Stop Out calculation:
Equity = Balance + Floating P/L
Margin Level = (Equity / Used Margin) × 100%
Stop Out fires when Margin Level <= 20%

Contoh 0.01 lot:
Used Margin = $2.90
Stop Out when Equity = $2.90 × 20% = $0.58
Max floating loss = $150 - $0.58 = $149.42

>>> Dengan 0.01 lot, hampir mustahil kena stop out kecuali balance benar-benar habis
```

---

## 7. Spread Analysis

### Cost per Trade (0.01 lot Standard)

| Broker | Spread (pips) | Cost per Trade | % of TP (40pt) | % of SL (20pt) |
|--------|--------------|----------------|-----------------|-----------------|
| XM Standard | 3.5 | $0.035 | 8.75% | 17.5% |
| XM Ultra Low | 2.0 | $0.020 | 5.0% | 10.0% |
| IC Markets Demo | ~1.0 | $0.010 | 2.5% | 5.0% |
| Exness Standard | ~25.0 | $0.250 | 62.5% | 125% |

### Monthly Cost Projection

```
Asumsi: 60 trades/bulan, 0.01 lot

XM Standard: 60 × $0.035 = $2.10/bulan
XM Ultra Low: 60 × $0.020 = $1.20/bulan
Difference: $0.90/bulan

Bonus advantage: $50 / $0.90 = 55+ bulan untuk break-even
>>> Standard + Bonus MENANG untuk trading period < 4.5 tahun
```

### Spread Impact pada SL/TP

```
Bot Settings: SL = 2.0×ATR, TP = 4.0×ATR
Asumsi ATR = 20 pips

SL distance = 40 pips
TP distance = 80 pips
Spread = 3.5 pips (XM Standard)

Effective SL = 40 - 3.5 = 36.5 pips (spread memakan 8.75% SL room)
Effective TP = 80 - 3.5 = 76.5 pips (spread memakan 4.375% TP room)
Effective RR = 76.5 / 36.5 = 2.10 (actual) vs 2.0 (theoretical)

>>> Spread 3.5 pips ACCEPTABLE — tidak signifikan merusak RR
```

---

## 8. Swap & Overnight Fees

### XM Standard Account — Swap

| Direction | Swap Type | Approximate Rate |
|-----------|----------|-----------------|
| **LONG (BUY)** | **Negatif** | ~-$0.30 to -$0.50 per 0.01 lot/night |
| **SHORT (SELL)** | **Negatif** | ~-$0.10 to -$0.30 per 0.01 lot/night |
| **Triple Swap** | Wednesday → Thursday | 3× normal swap |

> **KEDUA arah kena swap negatif** di Standard account. Ini berbeda dari beberapa broker yang memberi swap positif untuk salah satu arah.

### Swap Calculation

```
Swap dikenakan pada rollover time: 00:00 server time (GMT+2)

Contoh LONG 0.01 lot, swap = -$0.35/night:
├── Holding 1 malam: -$0.35
├── Holding 1 minggu (5 nights + 1 triple): -$0.35 × 7 = -$2.45
├── Holding 1 bulan (22 nights + 4 triple): -$0.35 × 30 = -$10.50
└── Signifikan untuk small balance!

Smart Trader Strategy:
├── Rata-rata hold time: 1-6 jam
├── Jarang overnight (exit management aktif)
├── Expected swap cost: < $1/bulan
└── ACCEPTABLE
```

### Swap-Free Alternative

| Option | Available? | Trade-off |
|--------|-----------|-----------|
| XM Ultra Low | Swap-free built-in | NO bonus eligibility |
| Islamic Account | Request via support | May have other fees |
| Standard | **Swap applies** | Bonus 50% eligible |

> **Keputusan**: Tetap Standard + terima swap. Bot jarang hold overnight, swap cost minimal.

---

## 9. Trading Hours

### GOLD Trading Hours (Server Time = GMT+2)

| Day | Open | Close | Notes |
|-----|------|-------|-------|
| Monday | **01:05** | 23:55 | |
| Tuesday | 01:05 | 23:55 | |
| Wednesday | 01:05 | 23:55 | Triple swap night |
| Thursday | 01:05 | 23:55 | |
| Friday | 01:05 | **23:50** | Close 5 min lebih awal |
| Saturday | CLOSED | CLOSED | |
| Sunday | CLOSED | CLOSED | |

### Konversi ke WIB (GMT+7)

| Event | Server (GMT+2) | UTC (GMT+0) | **WIB (GMT+7)** |
|-------|---------------|-------------|-----------------|
| Market Open (Mon) | 01:05 | 23:05 (Sun) | **06:05 Senin** |
| Daily Break Start | ~23:55 | ~21:55 | **~04:55** |
| Daily Break End | ~01:05 | ~23:05 | **~06:05** |
| Friday Close | 23:50 | 21:50 | **04:50 Sabtu** |
| Rollover/Swap | 00:00 | 22:00 | **05:00** |

### Bot Config — Blackout Hours

```yaml
# config.yaml adjustments needed for XM
# XM daily break: ~21:55-23:05 UTC (maintenance window)
# Berbeda dari IC Markets (22:00-23:00 UTC) dan Exness (22:00-23:00 UTC)

scanner:
  blackout_hours_utc:
    - [21, 55, 23, 10]   # XM daily maintenance (wider buffer)
  friday_close_utc: "21:30"  # Close positions before Friday market close
```

### Session Mapping untuk Bot

| Session | UTC | WIB | Volatility |
|---------|-----|-----|-----------|
| Asian | 00:00-07:00 | 07:00-14:00 | Low |
| London | 07:00-16:00 | 14:00-23:00 | High |
| **London-NY Overlap** | **12:00-16:00** | **19:00-23:00** | **Highest** |
| New York | 12:00-21:00 | 19:00-04:00 | High |
| Late NY | 16:00-21:00 | 23:00-04:00 | Medium |
| Off Hours | 21:55-23:05 | 04:55-06:05 | CLOSED |

---

## 10. Deposit & Withdrawal

### Deposit Methods

| Method | Fee | Processing | Min Amount |
|--------|-----|-----------|-----------|
| **Credit/Debit Card** (Visa/MC) | **Gratis** | **Instant** | $5 |
| **Skrill** | Gratis | Instant | $5 |
| **Neteller** | Gratis | Instant | $5 |
| **Bank Wire Transfer** | Gratis (XM bayar) | 2-5 hari kerja | $5 |
| **Local Bank Transfer** | Gratis | 1-2 hari kerja | $5 |
| Cryptocurrency | Varies | Minutes | Varies |

### Withdrawal Methods

| Method | Fee | Processing (XM side) | Total Time |
|--------|-----|---------------------|-----------|
| **E-wallets** (Skrill/Neteller) | **Gratis** | **< 24 jam** | **Same day** |
| **Credit/Debit Card** | Gratis | < 24 jam | 2-5 hari kerja |
| **Bank Wire** (>$200) | **Gratis** | < 24 jam | 2-5 hari kerja |
| **Bank Wire** (<$200) | **Ada fee bank** | < 24 jam | 2-5 hari kerja |

### Withdrawal Rules

1. **FIFO Priority**: Withdrawal mengikuti urutan deposit method (profit bisa ke method manapun)
2. **Bonus berkurang proporsional** pada setiap withdrawal
3. XM memproses withdrawal dalam **24 jam kerja**
4. Min withdrawal: varies by method
5. **Tidak ada limit** jumlah withdrawal per bulan

### Rekomendasi untuk Indonesia

| Priority | Method | Alasan |
|----------|--------|--------|
| 1 | **Skrill/Neteller** | Instant deposit, fast withdrawal, no fee |
| 2 | Credit/Debit Card | Instant deposit, 2-5 day withdrawal |
| 3 | Local Bank Transfer | 1-2 day deposit, 2-5 day withdrawal |
| 4 | Bank Wire | Slowest, possible fee if < $200 |

---

## 11. Regulasi & Keamanan Dana

### Fund Safety

| Protection | Detail |
|-----------|--------|
| **Segregated Accounts** | Dana client TERPISAH dari dana operasional XM |
| **Negative Balance Protection** | Balance tidak bisa negatif — max loss = deposit |
| **ICF Coverage** | Hingga EUR 20,000 per client (CySEC entity) |
| **Audit** | Regulated entities diaudit oleh external auditors |
| **AML/KYC** | Wajib verifikasi identitas sebelum trading |

### Privacy & Data

| Aspek | Status |
|-------|--------|
| GDPR Compliant | Yes (CySEC entity) |
| SSL Encryption | Yes |
| 2FA | Available |
| Data Deletion Rights | Yes (GDPR Art. 17) |
| Cross-border Transfer | With GDPR Art. 46 safeguards |

### KYC Requirements

| Document | Type | Notes |
|----------|------|-------|
| **Proof of Identity** | KTP/Passport/SIM | Clear photo, not expired |
| **Proof of Address** | Utility bill / Bank statement | < 6 months old |
| **Processing Time** | Usually < 24 hours | Can be instant |

---

## 12. MT5 Platform — Setup

### Download & Install

| Item | Detail |
|------|--------|
| Download | [xm.com/platforms](https://www.xm.com/platforms) → MT5 for PC |
| Installer | `xmglobal5setup.exe` (XM-specific installer) |
| Atau | Standard MT5 installer → search "XM" in broker list |
| Install Path | `C:\Program Files\MetaTrader 5 XM\` (suggested) |

### Server Names

| Type | Server Name |
|------|-------------|
| **Live** | `XMGlobal-Real` (series: Real 1, Real 2, ...) |
| **Demo** | `XMGlobal-Demo` (series: Demo 1, Demo 2, ...) |

> Exact server name akan diberikan saat registrasi akun. Bisa berbeda per region.

### MT5 Terminal Verification Steps

Setelah install dan login, verifikasi:

```
1. Market Watch → klik kanan → Show All
2. Cari "GOLD" atau "XAUUSD"
3. Klik kanan pada symbol → Specification
4. Catat:
   - Symbol name (exact)
   - Digits (2 atau 3)
   - Point (0.01 atau 0.001)
   - Contract size (should be 100)
   - Minimum volume (should be 0.01)
   - Volume step (should be 0.01)
   - Spread (check live)
```

### Python MT5 Connection Test

```python
import MetaTrader5 as mt5

# Initialize with XM terminal path
mt5.initialize("C:\\Program Files\\MetaTrader 5 XM\\terminal64.exe")

# Login
mt5.login(
    login=<XM_ACCOUNT_NUMBER>,
    password="<XM_PASSWORD>",
    server="XMGlobal-Real X"  # exact server from registration
)

# Account info
info = mt5.account_info()
print(f"Balance: ${info.balance}")
print(f"Leverage: 1:{info.leverage}")
print(f"Server: {info.server}")

# Symbol verification
sym = mt5.symbol_info("GOLD")  # try "GOLD" first
if sym is None:
    sym = mt5.symbol_info("XAUUSD")  # fallback
    if sym is None:
        # List all symbols containing "gold" or "xau"
        symbols = mt5.symbols_get()
        gold_syms = [s.name for s in symbols if "gold" in s.name.lower() or "xau" in s.name.lower()]
        print(f"Available gold symbols: {gold_syms}")
else:
    print(f"Symbol: {sym.name}")
    print(f"Digits: {sym.digits}")
    print(f"Point: {sym.point}")
    print(f"Contract Size: {sym.trade_contract_size}")
    print(f"Min Lot: {sym.volume_min}")
    print(f"Spread: {sym.spread} points")

mt5.shutdown()
```

---

## 13. XM vs IC Markets vs Exness — Perbandingan

### Head-to-Head

| Feature | **XM Standard** | IC Markets (current demo) | Exness Standard |
|---------|-----------------|--------------------------|-----------------|
| Account Type | Live | **Demo** | Demo |
| Balance | **$150** (incl bonus) | $100 (virtual) | $100 |
| Symbol | GOLD | XAUUSD | XAUUSDm |
| Digits | TBD (2 or 3) | 2 | 3 |
| Spread GOLD | ~3.5 pips | ~0.5-1.5 pips | ~25 pips |
| Commission | $0 | $0 | $0 |
| Leverage | 1:1000 | 1:500 | Unlimited |
| Margin 0.01 lot | $2.90 | $5.80 | $1.45 |
| Stop Out | 20% | 50% | 0% |
| Swap | Negatif 2 arah | Standard | Available |
| Bonus | **+$50** | N/A | N/A |
| **Real Money?** | **YES** | **NO** | NO |

### Key Differences yang Mempengaruhi Bot

| Aspect | IC Markets Demo | XM Live | Impact |
|--------|----------------|---------|--------|
| **Execution** | Perfect (demo) | Real slippage possible | May see 0-2 pip slippage |
| **Spread** | Ultra-tight | Wider (~3.5 pips) | SL/TP perlu buffer lebih |
| **Requotes** | Never (demo) | Rare but possible | Need retry logic |
| **Fills** | Always full | Usually full | 0.01 lot = always fill |
| **Psychology** | No emotion | **Real money** | Different behavior |

---

## 14. Bot Configuration — Migration Checklist

### Pre-Migration

- [ ] Register XM account → pilih **Standard** (bukan Ultra Low)
- [ ] Complete KYC verification (ID + address proof)
- [ ] Deposit $100 → claim 50% bonus di Members Area
- [ ] Download & install MetaTrader 5 XM
- [ ] Login ke MT5, catat:
  - [ ] Account number (login)
  - [ ] Server name (exact)
  - [ ] Terminal path
- [ ] Verify GOLD symbol:
  - [ ] Exact symbol name
  - [ ] Digits (2 or 3)
  - [ ] Point value
  - [ ] Contract size
  - [ ] Minimum lot
  - [ ] Live spread
- [ ] Run Python MT5 connection test (script di Section 12)

### Config Migration

- [ ] Update `config.yaml` dengan data baru (template di Section 15)
- [ ] Verify blackout hours match XM schedule
- [ ] Verify session times correct for XM server timezone
- [ ] Update Telegram notifier message (opsional — "XM Live" label)
- [ ] Backup current config: `cp config.yaml config_icmarkets_backup.yaml`

### Post-Migration

- [ ] Run bot: `python main.py`
- [ ] Verify MT5 connection successful
- [ ] Verify symbol info logged correctly
- [ ] Wait for first scan cycle → check zone detection
- [ ] Monitor first Claude call → check prompt data correct
- [ ] Monitor first trade execution → verify SL/TP placed correctly
- [ ] Check Telegram notifications arriving
- [ ] Monitor for 24h before leaving unattended

---

## 15. Config Template (XM)

```yaml
# config.yaml — XM Standard Live Account
# BACKUP: config_icmarkets_backup.yaml (original IC Markets demo)

mt5:
  symbol: GOLD              # VERIFIKASI: exact symbol name dari MT5
  magic: 202603             # New magic number untuk XM (avoid conflict)
  terminal_path: "C:\\Program Files\\MetaTrader 5 XM\\terminal64.exe"

account:
  login: <XM_ACCOUNT_NUMBER>     # Dari registrasi XM
  password: "<XM_PASSWORD>"       # Dari registrasi XM
  server: "XMGlobal-Real X"      # Exact server dari email XM

trading:
  zone_proximity_pts: 7.0        # Keep same (Optuna-optimized)
  min_confidence: 0.70            # Keep same
  max_positions: 1                # Keep 1 — real money, conservative
  max_per_direction: 1
  risk_percent: 1.0               # Keep 1% — $1.50 risk per trade on $150
  min_lot: 0.01
  sl_atr_mult: 2.0               # Keep same
  tp_atr_mult: 4.0               # Keep same
  max_drawdown_pct: 5.0           # 5% of $150 = $7.50 max DD
  free_margin_pct: 20.0           # Keep same

scanner:
  interval_sec: 30                # Keep same
  spike_windows_utc:
    - [7, 45, 8, 0]              # London spike
    - [12, 45, 13, 0]            # NY spike
  # XM daily maintenance: ~21:55-23:05 UTC
  # Bot already handles off-hours via session detection

claude:
  model: claude-opus-4-6          # Keep same
  timeout_sec: 120
  cmd: "C:\\Users\\Administrator\\AppData\\Roaming\\npm\\claude.cmd"

indicators:
  rsi_period: 14                  # Keep same
  atr_period: 14
  m15_lookback: 10

telegram:
  enabled: true
  token: "8215295219:AAGwcevN5QKqYIgVnOogB9P1Lo-x6HCoatM"
  chat_id: "-1003549733840"
  scan_report_interval_min: 15

llm_comparison:
  enabled: true
  api_base: "https://ai.sumopod.com/v1"
  api_key: "sk-qg78Jy5EYqEdz9IsJLlCDw"
  timeout_sec: 60

adaptive:
  enabled: true
  min_trades_for_adaptation: 20
  max_shift_pct: 0.25
  regime_params:    # Keep all Optuna-optimized params
    trending:
      sl_atr_mult: 2.75
      tp_atr_mult: 7.0
      min_confidence: 0.70
      be_trigger_mult: 0.3
      lock_trigger_mult: 1.4
      trail_keep_pct: 0.70
      stale_tighten_min: 120
      scratch_exit_min: 330
    ranging:
      sl_atr_mult: 2.50
      tp_atr_mult: 6.0
      min_confidence: 0.70
      be_trigger_mult: 0.4
      lock_trigger_mult: 1.5
      trail_keep_pct: 0.30
      stale_tighten_min: 90
      scratch_exit_min: 210
    breakout:
      sl_atr_mult: 3.25
      tp_atr_mult: 5.5
      min_confidence: 0.70
      be_trigger_mult: 0.4
      lock_trigger_mult: 1.4
      trail_keep_pct: 0.40
      stale_tighten_min: 90
      scratch_exit_min: 120
    reversal:
      sl_atr_mult: 3.00
      tp_atr_mult: 5.0
      min_confidence: 0.70
      be_trigger_mult: 0.6
      lock_trigger_mult: 1.3
      trail_keep_pct: 0.55
      stale_tighten_min: 180
      scratch_exit_min: 60
  bounds:           # Keep same
    sl_atr_mult: [1.0, 3.5]
    tp_atr_mult: [2.0, 7.0]
    min_confidence: [0.60, 0.90]
    be_trigger_mult: [0.3, 1.0]
    lock_trigger_mult: [0.8, 2.5]
    trail_keep_pct: [0.30, 0.70]
    stale_tighten_min: [30, 180]
    scratch_exit_min: [60, 360]

paths:
  zone_cache_db: "data/market_cache.db"   # New local path for XM
  log_dir: "logs"
  log_file: "logs/smart_trader.log"
  trades_csv: "logs/trades.csv"
```

---

## 16. Risk Management untuk $150

### Position Sizing

| Parameter | Value | Calculation |
|-----------|-------|-------------|
| Balance | $150 ($100 real + $50 bonus) |  |
| Risk per trade | 1% | $1.50 |
| Lot size | **0.01** (fixed) |  |
| Margin per trade | $2.90 | 1.9% of balance |
| Max drawdown | 5% | $7.50 |
| Max consecutive losses | ~5 | before 5% DD hit |

### SL/TP in Dollar Terms (0.01 lot)

```
ATR ≈ 20 pips (typical XAUUSD)

SL = 2.0 × ATR = 40 pips
TP = 4.0 × ATR = 80 pips

Pip value (0.01 lot, 100 oz contract):
1 pip = $0.01 per 0.01 lot (VERIFIKASI di MT5)

Dollar risk per trade:
SL = 40 pips × $0.01 = $0.40  (0.27% of $150)
TP = 80 pips × $0.01 = $0.80  (0.53% of $150)

Max consecutive losses before 5% DD:
$7.50 / $0.40 = 18.75 → ~18 losses before pause

>>> SANGAT CONSERVATIVE — bot bisa handle banyak loss sebelum DD limit
```

### Bonus sebagai Safety Net

```
Skenario: Balance turun dari profit = $0, hanya sisa bonus $50

Bonus masih bisa dipakai sebagai margin:
- 0.01 lot margin = $2.90
- $50 bonus / $2.90 = 17 posisi (theoretically)
- Stop out at 20% margin level = $0.58

>>> Bonus memberikan "second life" jika balance habis
>>> TAPI: jika balance real = $0, profit dari bonus masih withdrawable
```

---

## 17. Post-Migration Verification

### Day 1 — Connection & Data

```bash
# 1. Start bot
cd "C:\Users\Administrator\Videos\Last Bot AI Trading\smart_trader"
python main.py

# Verify in logs:
# ✅ MT5 initialized successfully
# ✅ Logged in to XM account (balance: $150.00)
# ✅ Symbol GOLD loaded (digits=X, point=X.XX)
# ✅ Scanner started (interval: 30s)
```

### Day 1 — Check Points

| Check | What to Verify | Pass Criteria |
|-------|---------------|---------------|
| MT5 Connect | Account info shows correct balance | Balance = $150 |
| Symbol Info | GOLD loaded with correct specs | Digits, point, contract_size logged |
| Zone Detection | Zones detected on M15/H1 | At least 1-2 zones per scan |
| Indicator Calc | ATR, RSI, EMA values reasonable | ATR ~15-30, RSI 30-70, EMA near price |
| Spread | Live spread within expected range | 2-5 pips during London/NY |
| Claude Call | First setup triggers Claude validation | Response received in < 30s |
| Telegram | Notifications arriving | Bot start message received |

### Week 1 — Performance Monitoring

| Metric | Target | Action if Missed |
|--------|--------|-----------------|
| Trades executed | 5-15 | Check gates, spread filter |
| Win rate | > 40% | Normal — small sample |
| Max drawdown | < $7.50 (5%) | Bot auto-pauses |
| Slippage | < 2 pips avg | Monitor, acceptable up to 3 |
| Claude timeout | < 5% of calls | Check internet, CLI |
| Spread avg | < 5 pips | Avoid Asian session entries |

---

## 18. Strategi Bonus — Optimasi

### DO's

1. **Jangan withdraw** sampai profit menumpuk signifikan
2. **Compound** — biarkan profit menambah balance
3. **Gunakan bonus sebagai margin buffer** — bukan sebagai "uang tambahan untuk trade lebih besar"
4. Tetap **0.01 lot** — jangan naik lot karena merasa punya $150
5. **Monitor bonus amount** di Members Area secara berkala

### DON'Ts

1. **Jangan withdraw kecil-kecil** — setiap withdrawal mengurangi bonus
2. **Jangan internal transfer** antar akun — bonus hilang
3. **Jangan naikkan lot** hanya karena bonus — risk management tetap sama
4. **Jangan abaikan swap** — jika hold overnight sering, pertimbangkan Ultra Low di masa depan

### Optimal Withdrawal Strategy

```
Phase 1 (Bulan 1-3): ZERO withdrawal
├── Biarkan profit compound
├── Target: grow $150 → $200+
└── Setiap $1 profit = $1 real (bonus tidak berkurang)

Phase 2 (Balance > $250): Consider small withdrawal
├── Withdraw max 20-30% of PROFIT only
├── Contoh: profit $100 → withdraw $20-30
├── Bonus berkurang: ($20/$100 deposit) × $50 = $10 hilang
└── Remaining: balance $220 + bonus $40 = $260

Phase 3 (Balance > $500): Free withdrawal
├── $50 bonus menjadi insignifikan vs $500+ balance
├── Bahkan jika bonus hilang total, modal sudah 5x
└── Consider switching to Ultra Low (better spread, swap-free)
```

---

## 19. Known Differences & Adjustments

### Symbol Name & Digits

| Aspect | IC Markets Demo | XM Live | Bot Impact |
|--------|----------------|---------|-----------|
| Symbol | XAUUSD | GOLD (TBD) | config.yaml change |
| Digits | 2 | 2 (TBD) | Verify — affects point calc |
| Point | 0.01 | 0.01 (TBD) | Verify — affects SL/TP |
| Contract | 100 oz | 100 oz | Same — no change |

### Spread & Execution

| Aspect | Demo | Live | Adjustment |
|--------|------|------|-----------|
| Spread | ~0.5-1.5 pips | ~3.5 pips | Bot already uses ATR-based SL/TP — spread absorbed |
| Slippage | 0 | 0-2 pips possible | No code change — market execution |
| Fills | Always | Usually (0.01 lot = always) | No issue for our lot size |
| Requotes | Never | Very rare | Retry logic already in executor |

### Timing

| Aspect | IC Markets | XM | Adjustment |
|--------|-----------|-----|-----------|
| Server TZ | UTC+2/+3 | UTC+2/+3 (EET) | Same — no change |
| Maintenance | 22:00-23:00 UTC | ~21:55-23:05 UTC | Widen blackout buffer 5 min |
| Friday close | 21:45 UTC | 21:50 UTC | Adjust pre-close time |
| Rollover | 22:00 UTC | 22:00 UTC | Same |

### Code Changes Summary

| File | Change | Type |
|------|--------|------|
| `config.yaml` | Symbol, login, server, terminal path, magic | Config only |
| **NO** source code changes needed | - | - |

> **Semua perubahan hanya di config.yaml.** Zero code changes karena bot sudah generic — symbol, login, server semua dari config.

---

## 20. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|---------|
| "Symbol GOLD not found" | Wrong symbol name | Check Market Watch → Show All → search gold |
| "Login failed" | Wrong server or credentials | Verify exact server name from XM email |
| "Not enough money" | Bonus not credited yet | Check Members Area → claim bonus |
| "Trade disabled" | Symbol not enabled | Market Watch → enable GOLD |
| "Off quotes" | Market closed or maintenance | Check trading hours |
| High spread (>10 pips) | Asian session or news event | Bot should auto-skip via session guard |
| "Invalid stops" | SL/TP too close to price | Min stop distance may differ — check `sym.trade_stops_level` |
| MT5 won't initialize | Wrong terminal path | Update config with correct install path |

### MT5 Connection Debug

```python
import MetaTrader5 as mt5

# Step 1: Initialize
if not mt5.initialize("C:\\Program Files\\MetaTrader 5 XM\\terminal64.exe"):
    print(f"Init failed: {mt5.last_error()}")
    # Common: wrong path, MT5 not installed, terminal already running

# Step 2: Login
if not mt5.login(login=XXXXX, password="XXX", server="XMGlobal-Real X"):
    print(f"Login failed: {mt5.last_error()}")
    # Common: wrong server name, wrong password, account not activated

# Step 3: Symbol
sym = mt5.symbol_info("GOLD")
if sym is None:
    print("GOLD not found, trying alternatives...")
    for name in ["XAUUSD", "XAUUSDm", "GOLD.", "GOLDm", "GOLD.a"]:
        sym = mt5.symbol_info(name)
        if sym:
            print(f"Found: {name}")
            break

# Step 4: Enable symbol
mt5.symbol_select("GOLD", True)  # Enable in Market Watch
```

### Bonus Not Showing

1. Login ke [XM Members Area](https://my.xm.com)
2. Navigate ke **Promotions** → **Deposit Bonus**
3. Click **Claim Bonus** (jika belum claim)
4. Bonus otomatis masuk ke akun trading
5. Jika masih belum: contact XM Live Chat

---

## 21. Sources

| Source | URL | Content |
|--------|-----|---------|
| XM Account Types | [xm.com/account-types](https://www.xm.com/account-types) | Account comparison |
| XM Promotions | [xm.com/promotions](https://www.xm.com/promotions) | Bonus program |
| XM Bonus Terms PDF | [xm.com (PDF)](https://www.xm.com/assets/pdf/new/terms/XM-Terms-and-Conditions-Bonus-Program.pdf) | Terms & conditions |
| XM Precious Metals | [xm.com/precious-metals](https://www.xm.com/precious-metals-trading) | Gold specs |
| XM Platforms | [xm.com/platforms](https://www.xm.com/platforms) | MT5 download |
| XM Regulation | [xm.com/regulation](https://www.xm.com/regulation) | License info |
| XM Deposits | [xm.com/deposits-withdrawals](https://www.xm.com/deposits-withdrawals) | Payment methods |
| XM Privacy Policy | [xm.com/privacy-policy](https://www.xm.com/privacy-policy) | Data protection |
| XM Leverage (Gold) | [xem.fxsignup.com](https://xem.fxsignup.com/en/faq/2022083101.html) | Leverage tiers |
| XM Trading Hours | [xem.fxsignup.com](https://xem.fxsignup.com/en/faq/2022083102.html) | GOLD hours |
| XM Margin Calculator | [xem.fxsignup.com](https://xem.fxsignup.com/en/reason/margin.html) | Margin calc |
| BrokerChooser XM Spread | [brokerchooser.com](https://brokerchooser.com/broker-reviews/xm-review/xauusd-spread) | Live spread data |
| Myfxbook XM Spreads | [myfxbook.com](https://www.myfxbook.com/forex-broker-spreads/xm-group/2824,51) | Historical spreads |
| TradingCritique XM | [tradingcritique.com](https://tradingcritique.com/broker-review/xm-broker-review/) | Full review |
| InvestFox XM Bonuses | [investfox.com](https://investfox.com/education/advanced/xm-deposit-bonuses/) | Bonus analysis |
| FXEmpire Comparison | [fxempire.com](https://www.fxempire.com/brokers/compare/exness-vs-xm) | Exness vs XM |
| Existing Analysis | [broker_comparison.md](broker_comparison.md) | Internal comparison doc |

---

<div align="center">

**Status: READY FOR MIGRATION**

_Menunggu registrasi XM + detail akun untuk config update._

---

_Smart Trader v1.0.0-beta — XM Broker Migration Guide_

_PT Surya Inovasi Prioritas (SURIOTA) — February 2026_

_Generated by Claude Opus 4.6_

</div>
