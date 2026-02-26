"""Quick market check script — analyze current conditions and open positions."""
import MetaTrader5 as mt5
import polars as pl
from datetime import datetime, timezone
import numpy as np

mt5.initialize(r'C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe')
mt5.login(413371434, server='Exness-MT5Trial6')

# Get last 30 M15 candles
rates = mt5.copy_rates_from_pos('XAUUSDm', mt5.TIMEFRAME_M15, 0, 30)
print("=" * 70)
print("LAST 20 M15 CANDLES (XAUUSDm)")
print("=" * 70)
print(f"{'Time':>6} {'Open':>9} {'High':>9} {'Low':>9} {'Close':>9} {'Vol':>6} {'Body':>6} {'Dir':>4}")
print("-" * 70)
for r in rates[-20:]:
    t = datetime.fromtimestamp(r[0], tz=timezone.utc).strftime('%H:%M')
    o, h, l, c, v = r[1], r[2], r[3], r[4], r[5]
    body = c - o
    d = "+" if body > 0 else "-"
    print(f"{t:>6} {o:>9.2f} {h:>9.2f} {l:>9.2f} {c:>9.2f} {v:>6} {body:>+6.2f} {d:>4}")

# Compute key indicators
closes = np.array([r[4] for r in rates])
highs = np.array([r[2] for r in rates])
lows = np.array([r[3] for r in rates])

# ATR
tr = np.maximum(highs[1:] - lows[1:], np.maximum(
    np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
atr14 = np.mean(tr[-14:])

# RSI
deltas = np.diff(closes)
gains = np.where(deltas > 0, deltas, 0)
losses = np.where(deltas < 0, -deltas, 0)
avg_gain = np.mean(gains[-14:])
avg_loss = np.mean(losses[-14:])
rs = avg_gain / avg_loss if avg_loss > 0 else 100
rsi = 100 - (100 / (1 + rs))

# EMA 20/50
def ema(data, period):
    e = np.zeros(len(data))
    e[0] = data[0]
    k = 2 / (period + 1)
    for i in range(1, len(data)):
        e[i] = data[i] * k + e[i-1] * (1 - k)
    return e

ema20 = ema(closes, 20)
ema50 = ema(closes, 50) if len(closes) >= 50 else None

# Swing points (sw=5)
swing_highs = []
swing_lows = []
for i in range(5, len(highs) - 5):
    if highs[i] == max(highs[i-5:i+6]):
        swing_highs.append((i, highs[i]))
    if lows[i] == min(lows[i-5:i+6]):
        swing_lows.append((i, lows[i]))

print(f"\n{'=' * 70}")
print("MARKET INDICATORS")
print(f"{'=' * 70}")
print(f"  Current Price : {closes[-1]:.2f}")
print(f"  ATR(14)       : {atr14:.2f}")
print(f"  RSI(14)       : {rsi:.1f}")
print(f"  EMA(20)       : {ema20[-1]:.2f}")
if ema50 is not None:
    print(f"  EMA(50)       : {ema50[-1]:.2f}")
print(f"  Price vs EMA20: {'ABOVE' if closes[-1] > ema20[-1] else 'BELOW'} ({closes[-1] - ema20[-1]:+.2f})")

# BOS detection
print(f"\n  Recent Swing Highs: {[(datetime.fromtimestamp(rates[i][0], tz=timezone.utc).strftime('%H:%M'), f'{v:.2f}') for i, v in swing_highs[-3:]]}")
print(f"  Recent Swing Lows:  {[(datetime.fromtimestamp(rates[i][0], tz=timezone.utc).strftime('%H:%M'), f'{v:.2f}') for i, v in swing_lows[-3:]]}")

# Check if last candle broke above a swing high (bullish BOS)
if swing_highs:
    last_sh = swing_highs[-1][1]
    if closes[-1] > last_sh:
        print(f"  BOS: Close {closes[-1]:.2f} > Last Swing High {last_sh:.2f} -> BULLISH BOS")
    else:
        print(f"  BOS: Close {closes[-1]:.2f} vs Last Swing High {last_sh:.2f} -> No bullish break")

if swing_lows:
    last_sl = swing_lows[-1][1]
    if closes[-1] < last_sl:
        print(f"  BOS: Close {closes[-1]:.2f} < Last Swing Low {last_sl:.2f} -> BEARISH BOS")
    else:
        print(f"  BOS: Close {closes[-1]:.2f} vs Last Swing Low {last_sl:.2f} -> No bearish break")

# Open positions
print(f"\n{'=' * 70}")
print("OPEN POSITIONS")
print(f"{'=' * 70}")
positions = mt5.positions_get(symbol='XAUUSDm')
if positions:
    for p in positions:
        direction = "BUY" if p.type == 0 else "SELL"
        sl_dist = abs(p.price_open - p.sl)
        tp_dist = abs(p.tp - p.price_open)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        curr_rr = (p.price_current - p.price_open) / sl_dist if direction == "BUY" and sl_dist > 0 else (p.price_open - p.price_current) / sl_dist if sl_dist > 0 else 0
        print(f"  #{p.ticket} {direction} @ {p.price_open:.2f}")
        print(f"  Current: {p.price_current:.2f} | P/L: ${p.profit:.2f} | RR: {curr_rr:.2f}")
        print(f"  SL: {p.sl:.2f} ({sl_dist:.2f} pts) | TP: {p.tp:.2f} ({tp_dist:.2f} pts) | Setup RR: {rr:.1f}:1")
else:
    print("  No open positions")

# Account
info = mt5.account_info()
print(f"\n  Balance: ${info.balance:.2f} | Equity: ${info.equity:.2f} | Margin: ${info.margin:.2f}")

tick = mt5.symbol_info_tick('XAUUSDm')
print(f"  Live: Bid={tick.bid:.2f} Ask={tick.ask:.2f} Spread={tick.ask-tick.bid:.2f}")

mt5.shutdown()
