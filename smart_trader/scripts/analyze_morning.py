"""Quick morning market analysis script."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import indicators as ind
import zone_detector as zdet
import scanner as scan
import regime_detector as rdet

mt5.initialize()
symbol = 'XAUUSD'

h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 60)
m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 80)

df_h1 = pd.DataFrame(h1)
df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
df_h1['wib'] = df_h1['time'] + timedelta(hours=7)

df_m15 = pd.DataFrame(m15)
df_m15['time'] = pd.to_datetime(df_m15['time'], unit='s')
df_m15['wib'] = df_m15['time'] + timedelta(hours=7)

# Filter window
mask = (df_m15['wib'] >= '2026-02-26 06:00') & (df_m15['wib'] <= '2026-02-26 10:15')
window = df_m15[mask].copy()
price = df_m15.iloc[-1]['close']

print("=" * 70)
print("  XAUUSD MARKET ANALYSIS | 06:00 - 10:00 WIB (26 Feb 2026)")
print("=" * 70)

# ── Price Action ──
if len(window) > 0:
    op = window.iloc[0]['open']
    cl = window.iloc[-1]['close']
    hi = window['high'].max()
    lo = window['low'].min()
    total_range = hi - lo
    net = cl - op
    direction = 'BULLISH' if net > 0 else 'BEARISH'

    print(f"\n  PRICE ACTION:")
    print(f"    Open={op:.2f}  Close={cl:.2f}  High={hi:.2f}  Low={lo:.2f}")
    print(f"    Range={total_range:.2f}pt | Net={net:+.2f}pt ({direction})")

    print(f"\n  M15 CANDLES:")
    bc = 0
    sc = 0
    for _, r in window.iterrows():
        body = 'BULL' if r['close'] > r['open'] else 'BEAR'
        if body == 'BULL':
            bc += 1
        else:
            sc += 1
        crng = r['high'] - r['low']
        bsz = abs(r['close'] - r['open'])
        notes = []
        if crng > 15:
            notes.append('WIDE')
        if bsz > crng * 0.7 and bsz > 0:
            notes.append('STRONG')
        if bsz < crng * 0.2:
            notes.append('DOJI')
        nstr = ' '.join(notes)
        wib_str = r['wib'].strftime("%H:%M")
        print(f"    {wib_str} | O={r['open']:.2f} H={r['high']:.2f} L={r['low']:.2f} C={r['close']:.2f} | {body} {crng:.1f}pt | {nstr}")
    print(f"    => {bc} BULL / {sc} BEAR candles")
else:
    print("\n  No data in 06:00-10:00 WIB window")

# ── H1 Indicators ──
rsi_val = ind.rsi(df_h1, 14)
atr_val = ind.atr(df_h1, 14)
ema_trend = ind.h1_ema_trend(df_h1)
pd_zone = ind.premium_discount(df_h1)
de = ind.daily_range_consumed(df_h1, 1.2)

print(f"\n  H1 INDICATORS:")
rsi_note = "(sweet spot)" if 55 <= rsi_val <= 85 else "(overbought)" if rsi_val > 70 else "(oversold)" if rsi_val < 30 else ""
print(f"    RSI(14)={rsi_val:.1f} {rsi_note}")
print(f"    ATR(14)={atr_val:.1f}pt")
print(f"    EMA(50)={ema_trend}")
print(f"    P/D zone={pd_zone}")
print(f"    Daily range={'EXHAUSTED' if de else 'OK'}")

# ── Regime ──
det = rdet.RegimeDetector()
res = det.detect(df_h1)
comps = res['components']
print(f"\n  REGIME:")
print(f"    {res['short_label']} ({res['confidence']:.0%}) | category={res['regime'].category}")
print(f"    CHoCH={comps.get('has_choch', False)}")
for k, v in comps.items():
    if k not in ('has_choch',) and v and not isinstance(v, (int, float, bool)):
        print(f"    {k}={v}")

# ── Zones ──
zones = zdet.detect_all_zones(df_h1)
nearby = scan.find_nearby_zones(price, zones, 15.0) if zones else []
print(f"\n  SMC ZONES (near {price:.2f}):")
print(f"    Total={len(zones)} | Nearby(<15pt)={len(nearby)}")
for z in nearby:
    dd = scan.direction_for_zone(z)
    lvl = z.get('high') or z.get('level', 0)
    dist = z.get('distance_pts', 0)
    print(f"    - {z['type']:20s} @ {lvl:.2f} | dist={dist:.1f}pt | {dd}")

# ── Signal Analysis ──
print(f"\n  SIGNAL ANALYSIS:")
for direction in ['LONG', 'SHORT']:
    dz = [z for z in nearby if scan.direction_for_zone(z) == direction]
    if dz:
        nt = [z['type'] for z in dz]
        mc = ind.m15_confirmation(df_m15.tail(12), direction)
        ote = ind.ote_zone(df_h1, direction)
        cnt, sigs = ind.count_signals(direction, nt, mc, ote, price, pd_zone)
        t1 = sum(1 for s in sigs if s in ('BOS', 'OB', 'LiqSweep'))
        t2 = sum(1 for s in sigs if s in ('FVG', 'CHoCH', 'Breaker', 'M15', 'OTE'))
        hs = any(s in ('BOS', 'CHoCH') for s in sigs)
        ea = (direction == 'LONG' and ema_trend == 'BULLISH') or (direction == 'SHORT' and ema_trend == 'BEARISH')
        pa = (direction == 'LONG' and pd_zone == 'DISCOUNT') or (direction == 'SHORT' and pd_zone == 'PREMIUM')
        ema_tag = "ALIGNED" if ea else "COUNTER"
        pd_tag = "ALIGNED" if pa else pd_zone
        gate = "PASS -> Claude" if cnt >= 1 else "BLOCKED (0 signals)"
        print(f"    {direction}: {cnt} sigs [{' + '.join(sigs)}]")
        print(f"      T1={t1} T2={t2} | struct={'Y' if hs else 'N'} | m15={mc or 'none'}")
        print(f"      EMA={ema_tag} | P/D={pd_tag} | Gate: {gate}")
    else:
        print(f"    {direction}: no nearby zones")

# ── Session ──
now_utc = df_m15.iloc[-1]['time']
if hasattr(now_utc, 'to_pydatetime'):
    now_utc = now_utc.to_pydatetime()
if now_utc.tzinfo is None:
    now_utc = now_utc.replace(tzinfo=timezone.utc)
sess = scan.current_session(now_utc)
print(f"\n  SESSION: {sess['name']} | Current: {price:.2f}")

# ── H1 candles in the window ──
h1_mask = (df_h1['wib'] >= '2026-02-26 06:00') & (df_h1['wib'] <= '2026-02-26 10:00')
h1_window = df_h1[h1_mask]
if len(h1_window) > 0:
    print(f"\n  H1 CANDLES (06-10 WIB):")
    for _, r in h1_window.iterrows():
        body = 'BULL' if r['close'] > r['open'] else 'BEAR'
        crng = r['high'] - r['low']
        wib_str = r['wib'].strftime("%H:%M")
        print(f"    {wib_str} | O={r['open']:.2f} H={r['high']:.2f} L={r['low']:.2f} C={r['close']:.2f} | {body} {crng:.1f}pt")

print("\n" + "=" * 70)
mt5.shutdown()
