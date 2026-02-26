"""Analyze trade #3932379641 — what happened and could we have exited earlier?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone

mt5.initialize(r'C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe')
mt5.login(413371434, server='Exness-MT5Trial6')

# Trade details (in UTC)
ENTRY_PRICE = 5203.48
SL = 5164.80
TP = 5272.01
SL_DISTANCE = ENTRY_PRICE - SL  # 38.68

print("=" * 70)
print("TRADE #3932379641 POST-MORTEM")
print("=" * 70)
print(f"Entry: BUY @ {ENTRY_PRICE:.2f} at 05:45 UTC (12:45 WIB)")
print(f"SL: {SL:.2f} ({SL_DISTANCE:.2f} pips) | TP: {TP:.2f} ({TP-ENTRY_PRICE:.2f} pips)")
print(f"Duration: ~5.4 hours | Loss: -$38.75")

# Fetch M15 data — use broad range to ensure coverage
# MT5 uses broker time (Exness = UTC+0), but copy_rates_range
# may interpret naive datetimes as local time. Use wide range.
start = datetime(2026, 2, 24, 12, 0)  # well before entry
end = datetime(2026, 2, 26, 0, 0)     # well after exit
rates = mt5.copy_rates_range("XAUUSDm", mt5.TIMEFRAME_M15, start, end)

if rates is None or len(rates) == 0:
    print("ERROR: No data")
    mt5.shutdown()
    sys.exit(1)

print(f"Total M15 bars fetched: {len(rates)}")

# Find entry candle (close to 5203)
entry_idx = None
for i, r in enumerate(rates):
    t = datetime.fromtimestamp(r[0], tz=timezone.utc)
    if abs(r[1] - 5203.0) < 2 and t.day == 25 and t.hour >= 4:
        entry_idx = i
        break

if entry_idx is None:
    # Fallback: find candle with high near 5210 (entry candle high)
    for i, r in enumerate(rates):
        t = datetime.fromtimestamp(r[0], tz=timezone.utc)
        if r[2] > 5209 and r[2] < 5212 and t.day == 25:
            entry_idx = i
            break

if entry_idx is None:
    print("Could not find entry candle, showing all Feb 25 data:")
    for r in rates:
        t = datetime.fromtimestamp(r[0], tz=timezone.utc)
        if t.day == 25:
            print(f"  {t.strftime('%H:%M')} O={r[1]:.2f} H={r[2]:.2f} L={r[3]:.2f} C={r[4]:.2f}")
    mt5.shutdown()
    sys.exit(1)

entry_t = datetime.fromtimestamp(rates[entry_idx][0], tz=timezone.utc)
print(f"Entry candle found: idx={entry_idx}, time={entry_t.strftime('%m/%d %H:%M UTC')}")

print(f"\n--- M15 CANDLES: 1h before entry to 2h after exit ---")
print(f"{'Time UTC':>12} {'WIB':>5} {'Open':>9} {'High':>9} {'Low':>9} {'Close':>9} {'RR':>6} {'Note'}")
print("-" * 85)

max_favorable = -99
worst_rr = 99
sl_hit_idx = None

# Show from 4 bars before entry to well after
show_start = max(0, entry_idx - 4)
show_end = min(len(rates), entry_idx + 50)  # ~12 hours of M15

for i in range(show_start, show_end):
    r = rates[i]
    t = datetime.fromtimestamp(r[0], tz=timezone.utc)
    wib_h = (t.hour + 7) % 24
    o, h, l, c = r[1], r[2], r[3], r[4]

    note = ""
    rr = 0

    if i >= entry_idx:
        rr = (c - ENTRY_PRICE) / SL_DISTANCE
        high_rr = (h - ENTRY_PRICE) / SL_DISTANCE
        low_rr = (l - ENTRY_PRICE) / SL_DISTANCE

        if high_rr > max_favorable:
            max_favorable = high_rr
        if low_rr < worst_rr:
            worst_rr = low_rr

        if i == entry_idx:
            note = " <-- ENTRY"
        elif l <= SL and sl_hit_idx is None:
            sl_hit_idx = i
            note = " <-- SL HIT!"
        elif rr < -0.7:
            note = f" danger"
        elif rr > 0:
            note = f" profit"

    print(f"{t.strftime('%H:%M'):>12} {wib_h:>2}:{t.strftime('%M')} {o:>9.2f} {h:>9.2f} {l:>9.2f} {c:>9.2f} {rr:>+6.2f}{note}")

    # Stop 4 candles after SL hit
    if sl_hit_idx and i > sl_hit_idx + 4:
        break

print(f"\n--- TRADE METRICS ---")
print(f"Max favorable: {max_favorable:+.2f}R (best price +${max_favorable * SL_DISTANCE:.2f} from entry)")
print(f"Max adverse:   {worst_rr:+.2f}R (worst price -${abs(worst_rr * SL_DISTANCE):.2f} from entry)")

if sl_hit_idx:
    bars_to_sl = sl_hit_idx - entry_idx
    sl_t = datetime.fromtimestamp(rates[sl_hit_idx][0], tz=timezone.utc)
    hours_to_sl = bars_to_sl * 0.25  # 15 min per bar
    print(f"Bars to SL hit: {bars_to_sl} ({hours_to_sl:.1f}h) at {sl_t.strftime('%H:%M')} UTC")

# Timeline of key RR levels
print(f"\n--- WHEN DID POSITION REACH KEY LEVELS? ---")
for thresh in [0.0, -0.25, -0.50, -0.75, -1.0]:
    for i in range(entry_idx, min(len(rates), entry_idx + 50)):
        r = rates[i]
        t = datetime.fromtimestamp(r[0], tz=timezone.utc)
        low_rr = (r[3] - ENTRY_PRICE) / SL_DISTANCE
        close_rr = (r[4] - ENTRY_PRICE) / SL_DISTANCE
        check_rr = low_rr if thresh < 0 else (r[2] - ENTRY_PRICE) / SL_DISTANCE

        if (thresh < 0 and check_rr <= thresh) or (thresh >= 0 and check_rr >= thresh):
            bars = i - entry_idx
            hours = bars * 0.25
            wib_h = (t.hour + 7) % 24
            if thresh >= 0:
                print(f"  Peak {thresh:+.2f}R at {t.strftime('%H:%M')} UTC ({wib_h}:{t.strftime('%M')} WIB) — {hours:.1f}h in")
            else:
                print(f"  Hit  {thresh:+.2f}R at {t.strftime('%H:%M')} UTC ({wib_h}:{t.strftime('%M')} WIB) — {hours:.1f}h in, close={r[4]:.2f}")
            break

# What-if: time-based exit
print(f"\n--- WHAT-IF: TIME-BASED EXIT ---")
for max_hours in [1, 2, 3, 4]:
    target_idx = entry_idx + int(max_hours / 0.25)
    if target_idx < len(rates):
        r = rates[target_idx]
        t = datetime.fromtimestamp(r[0], tz=timezone.utc)
        close_price = r[4]
        pnl = (close_price - ENTRY_PRICE) * 100 * 0.01
        rr = (close_price - ENTRY_PRICE) / SL_DISTANCE
        saved = 38.75 - abs(pnl) if pnl < 0 else 38.75 + pnl
        print(f"  Exit after {max_hours}h: Price={close_price:.2f}, RR={rr:+.2f}, P/L=${pnl:+.2f} (save ${saved:.2f} vs SL)")

# What-if: tighter SL
print(f"\n--- WHAT-IF: TIGHTER SL ---")
for alt_pips in [20, 25, 30, 35]:
    alt_sl = ENTRY_PRICE - alt_pips
    for i in range(entry_idx, min(len(rates), entry_idx + 50)):
        r = rates[i]
        t = datetime.fromtimestamp(r[0], tz=timezone.utc)
        if r[3] <= alt_sl:
            bars = i - entry_idx
            hours = bars * 0.25
            alt_loss = alt_pips * 0.01 * 100
            saved = 38.75 - alt_loss
            print(f"  SL={alt_sl:.0f} ({alt_pips}p): Hit at {t.strftime('%H:%M')} ({hours:.1f}h), Loss=${alt_loss:.2f}, Save=${saved:.2f}")
            break
    else:
        alt_loss = alt_pips * 0.01 * 100
        print(f"  SL={alt_sl:.0f} ({alt_pips}p): NEVER HIT — would have saved ${38.75 - alt_loss:.2f} vs actual")

# What-if: adverse RR cutoff (close if position drops below threshold without recovery)
print(f"\n--- WHAT-IF: ADVERSE RR CUTOFF (early loss cut) ---")
for cutoff_rr in [-0.3, -0.5, -0.7]:
    for i in range(entry_idx + 2, min(len(rates), entry_idx + 50)):  # skip first 2 bars
        r = rates[i]
        t = datetime.fromtimestamp(r[0], tz=timezone.utc)
        close_rr = (r[4] - ENTRY_PRICE) / SL_DISTANCE
        if close_rr <= cutoff_rr:
            pnl = (r[4] - ENTRY_PRICE) * 100 * 0.01
            saved = 38.75 - abs(pnl)
            bars = i - entry_idx
            print(f"  Cut at {cutoff_rr}R: Exit {t.strftime('%H:%M')} ({bars*15}min), Price={r[4]:.2f}, P/L=${pnl:+.2f}, Save=${saved:.2f}")
            break

# Post-SL recovery
if sl_hit_idx:
    print(f"\n--- POST-SL RECOVERY ---")
    for i in range(sl_hit_idx, min(len(rates), sl_hit_idx + 12)):
        r = rates[i]
        t = datetime.fromtimestamp(r[0], tz=timezone.utc)
        rr_from_entry = (r[4] - ENTRY_PRICE) / SL_DISTANCE
        recovery_from_sl = r[4] - SL
        bars_after = i - sl_hit_idx
        if bars_after > 0:
            print(f"  +{bars_after*15}min: Price={r[4]:.2f} (SL+{recovery_from_sl:.2f}, {rr_from_entry:+.2f}R from entry)")

mt5.shutdown()
