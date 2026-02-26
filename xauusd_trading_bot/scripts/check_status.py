"""Quick status check: account, positions, recent deals, market."""
import MetaTrader5 as mt5
from datetime import datetime, timedelta

mt5.initialize(r'C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe')
mt5.login(413371434, server='Exness-MT5Trial6')

# Account info
info = mt5.account_info()
print(f'Balance: ${info.balance:.2f} | Equity: ${info.equity:.2f} | Margin: ${info.margin:.2f} | Free: ${info.margin_free:.2f}')
print(f'Profit: ${info.profit:.2f} | Leverage: 1:{info.leverage}')

# Open positions
positions = mt5.positions_get(symbol='XAUUSDm')
if positions:
    for p in positions:
        profit = p.profit + p.swap + p.commission
        d = "BUY" if p.type == 0 else "SELL"
        t = datetime.fromtimestamp(p.time)
        print(f'\nOPEN: Ticket #{p.ticket} | {d} {p.volume} XAUUSDm @ {t.strftime("%m/%d %H:%M")}')
        print(f'  Entry: {p.price_open:.2f} | Current: {p.price_current:.2f} | SL: {p.sl:.2f} | TP: {p.tp:.2f}')
        print(f'  Profit: ${profit:.2f} | Swap: ${p.swap:.2f}')
else:
    print('\nNo open positions')

# Recent deals (last 7 days)
from_date = datetime.now() - timedelta(days=7)
to_date = datetime.now() + timedelta(hours=1)
deals = mt5.history_deals_get(from_date, to_date, group='*XAUUSDm*')
if deals:
    print(f'\n--- RECENT DEALS (last 7 days) ---')
    for d in deals:
        if d.entry == 1:  # OUT deals only
            t = datetime.fromtimestamp(d.time)
            # close type is opposite of position direction
            direction = "BUY" if d.type == 1 else "SELL"
            profit = d.profit + d.swap + d.commission
            print(f'  {t.strftime("%m/%d %H:%M")} | Closed {direction} #{d.position_id} | P/L: ${profit:.2f} | Exit: {d.price:.2f} | Vol: {d.volume}')

    # Summary
    total_pnl = sum(d.profit + d.swap + d.commission for d in deals if d.entry == 1)
    wins = sum(1 for d in deals if d.entry == 1 and (d.profit + d.swap + d.commission) > 0)
    losses = sum(1 for d in deals if d.entry == 1 and (d.profit + d.swap + d.commission) <= 0)
    print(f'\n  Summary: {wins}W / {losses}L | Net P/L: ${total_pnl:.2f}')
else:
    print('\nNo recent deals')

# Current price
tick = mt5.symbol_info_tick('XAUUSDm')
if tick:
    print(f'\n--- CURRENT MARKET ---')
    print(f'XAUUSDm: Bid={tick.bid:.2f} Ask={tick.ask:.2f} Spread={tick.ask-tick.bid:.3f}')
    print(f'Time: {datetime.fromtimestamp(tick.time).strftime("%Y-%m-%d %H:%M:%S")}')

mt5.shutdown()
