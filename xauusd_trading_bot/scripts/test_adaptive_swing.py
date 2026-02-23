"""Quick test: adaptive swing_length vs fixed, on 2000 M15 bars."""
import sys; sys.path.insert(0, '.')
from src.core.mt5_connector import MT5Connector
import yaml, MetaTrader5 as mt5_lib, pandas as pd
from smartmoneyconcepts import smc as _smc_lib
from collections import Counter

with open('config/settings.yaml') as f:
    settings = yaml.safe_load(f)
mt5 = MT5Connector(settings); mt5.connect()
bars = mt5_lib.copy_rates_from_pos('XAUUSDm', mt5_lib.TIMEFRAME_M15, 0, 2000)
ohlc = pd.DataFrame({
    'open': [b[1] for b in bars], 'high': [b[2] for b in bars],
    'low': [b[3] for b in bars], 'close': [b[4] for b in bars],
    'volume': [float(b[5]) for b in bars],
})
ohlc['tr'] = ohlc.apply(lambda r: max(r['high']-r['low'], abs(r['high']-r['close']), abs(r['low']-r['close'])), axis=1)
ohlc['atr14'] = ohlc['tr'].rolling(14).mean()
print(f"Bars: {len(ohlc)} | Price: {ohlc['close'].iloc[0]:.0f} -> {ohlc['close'].iloc[-1]:.0f}")
print(f"ATR range: {ohlc['atr14'].min():.1f} - {ohlc['atr14'].max():.1f} | Avg: {ohlc['atr14'].mean():.1f}")

atr_vals = ohlc['atr14'].dropna()
p25 = atr_vals.quantile(0.25)
p75 = atr_vals.quantile(0.75)
print(f"ATR percentiles: P25={p25:.1f}, P50={atr_vals.median():.1f}, P75={p75:.1f}")
print(f"Adaptive: ATR<{p25:.0f}->sw=7, {p25:.0f}-{p75:.0f}->sw=5, >{p75:.0f}->sw=3")

BOS_LOOKBACK = 50

def check_entry(bos_df, fvg_df, ob_df, liq_df, i, price, direction):
    is_bull = direction == 'bull'
    bos_t = 1.0 if is_bull else -1.0
    start = max(0, i - BOS_LOOKBACK + 1)
    sb = bos_df.iloc[start:i+1]
    has_bos = bool((sb['BOS'] == bos_t).any())
    has_choch = bool((sb['CHOCH'] == bos_t).any())

    fvg_t = 1.0 if is_bull else -1.0
    sf = fvg_df.iloc[:i+1]
    fmi = sf['MitigatedIndex']
    fvg_active = fmi.isna() | (fmi == 0.0)
    fvg_in = bool(((sf['FVG']==fvg_t) & (sf['Bottom']<=price) & (sf['Top']>=price) & fvg_active).any())

    ob_t = 1.0 if is_bull else -1.0
    so = ob_df.iloc[:i+1]
    mi_o = so['MitigatedIndex']
    ob_active = mi_o.isna() | (mi_o == 0.0) | (mi_o > i)
    ob_in = bool(((so['OB']==ob_t) & (so['Bottom']<=price) & (so['Top']>=price) & ob_active).any())

    liq_t = -1.0 if is_bull else 1.0
    slq = liq_df.iloc[:i+1]
    lmask = (slq['Liquidity']==liq_t) & slq['Swept'].notna()
    liq_swept = False
    if lmask.any():
        swept_idx = slq[lmask]['Swept'].astype(float)
        recent = (i - swept_idx)[(i - swept_idx) <= 20]
        liq_swept = not recent.empty

    score = 0.0
    if fvg_in: score += 0.20
    if ob_in: score += 0.25
    if liq_swept: score += 0.20
    if has_choch: score += 0.30
    elif has_bos: score += 0.15

    has_structure = has_bos or has_choch
    smc_count = sum([has_structure, fvg_in, ob_in, liq_swept])
    return has_structure and smc_count >= 1 and score >= 0.44, score, smc_count

# Pre-compute library outputs for each swing_length
cache = {}
for sl in [3, 5, 7]:
    swing = _smc_lib.swing_highs_lows(ohlc, swing_length=sl)
    bos_df = _smc_lib.bos_choch(ohlc, swing, close_break=True)
    fvg_df = _smc_lib.fvg(ohlc, join_consecutive=False)
    ob_df = _smc_lib.ob(ohlc, swing, close_mitigation=False)
    liq_df = _smc_lib.liquidity(ohlc, swing, range_percent=0.01)
    cache[sl] = (bos_df, fvg_df, ob_df, liq_df)

def simulate(get_sl_fn, cooldown=4):
    trades = []; i = 300; equity = 100.0; max_eq = 100.0; max_dd = 0
    while i < len(ohlc):
        price = ohlc['close'].iloc[i]
        atr = ohlc['atr14'].iloc[i]
        if pd.isna(atr) or atr < 1:
            i += 1; continue

        sl_val = get_sl_fn(atr)
        bos_df, fvg_df, ob_df, liq_df = cache[sl_val]

        best = None
        for d in ['bull', 'bear']:
            ok, score, smc = check_entry(bos_df, fvg_df, ob_df, liq_df, i, price, d)
            if ok and (best is None or score > best[1]):
                best = (d, score, smc, sl_val)

        if best is None:
            i += 1; continue

        direction, score, smc, used_sl = best
        is_bull = direction == 'bull'
        sl_dist = atr * 2.6
        tp_dist = atr * 6.0

        if is_bull:
            sl_p = price - sl_dist; tp_p = price + tp_dist
        else:
            sl_p = price + sl_dist; tp_p = price - tp_dist

        result = 'OPEN'; exit_price = price; exit_bar = len(ohlc)-1
        for j in range(i+1, len(ohlc)):
            h = ohlc['high'].iloc[j]; l = ohlc['low'].iloc[j]
            if is_bull:
                if l <= sl_p: result='SL'; exit_price=sl_p; exit_bar=j; break
                if h >= tp_p: result='TP'; exit_price=tp_p; exit_bar=j; break
            else:
                if h >= sl_p: result='SL'; exit_price=sl_p; exit_bar=j; break
                if l <= tp_p: result='TP'; exit_price=tp_p; exit_bar=j; break

        pnl_price = (exit_price - price) if is_bull else (price - exit_price)
        pnl_dollar = pnl_price * 0.01 * 100

        if result != 'OPEN':
            equity += pnl_dollar
            max_eq = max(max_eq, equity)
            dd = (max_eq - equity) / max_eq * 100
            max_dd = max(max_dd, dd)

        trades.append({
            'result': result, 'pnl': pnl_dollar,
            'rr': pnl_price / sl_dist if sl_dist else 0,
            'dir': direction, 'swing': used_sl, 'atr': atr,
            'bar': i, 'price': price,
        })

        if result != 'OPEN':
            i = exit_bar + cooldown
        else:
            i = len(ohlc)

    return trades, equity, max_dd

def report(label, trades, equity, max_dd):
    wins = [t for t in trades if t['result']=='TP']
    losses = [t for t in trades if t['result']=='SL']
    closed = len(wins)+len(losses)
    gp = sum(t['pnl'] for t in wins)
    gl = abs(sum(t['pnl'] for t in losses))
    pf = gp/gl if gl > 0 else 0
    wr = len(wins)/closed*100 if closed > 0 else 0
    total_pnl = sum(t['pnl'] for t in trades if t['result']!='OPEN')
    return f"{label:>15} | {len(trades):>3} | {len(wins):>3} {len(losses):>3} | {wr:>5.1f} | {pf:>5.2f} | {total_pnl:>+10.2f} | {equity:>9.2f} | {max_dd:>6.1f}"

print()
hdr = f"{'Method':>15} | {'N':>3} | {'TP':>3} {'SL':>3} | {'WR%':>5} | {'PF':>5} | {'PnL($)':>10} | {'Equity':>9} | {'DD%':>6}"
print(hdr)
print("-" * len(hdr))

for sl_val in [3, 5, 7]:
    t, eq, dd = simulate(lambda atr, s=sl_val: s)
    print(report(f"Fixed sw={sl_val}", t, eq, dd))

# Adaptive
def adaptive_fn(atr):
    if atr < p25: return 7
    elif atr > p75: return 3
    else: return 5

t_ad, eq_ad, dd_ad = simulate(adaptive_fn)
print(report("Adaptive", t_ad, eq_ad, dd_ad))

# Show adaptive trade details
print(f"\n--- Adaptive trade details ---")
sl_usage = Counter(t['swing'] for t in t_ad)
print(f"Swing usage: {dict(sl_usage)}")
for t in t_ad:
    s = f"{t['result']:4s}" if t['result']!='OPEN' else 'OPEN'
    print(f"  bar={t['bar']:4d} {t['dir']:4s} @{t['price']:.0f} {s} pnl=${t['pnl']:+7.2f} RR={t['rr']:+5.2f} sw={t['swing']} atr={t['atr']:.1f}")

mt5.disconnect()
