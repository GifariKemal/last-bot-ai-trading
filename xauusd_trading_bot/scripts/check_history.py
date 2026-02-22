"""Check recent trade history."""
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone

mt5.initialize(r'C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe')
mt5.login(52725397, server='ICMarketsSC-Demo')

# Check open positions
positions = mt5.positions_get(symbol='XAUUSD')
print(f'Open positions: {len(positions) if positions else 0}')

# Check recent deals
now = datetime.now(timezone.utc)
deals = mt5.history_deals_get(now - timedelta(days=2), now, group='XAUUSD')
if deals:
    print(f'\nRecent deals ({len(deals)} total):')
    for d in deals:
        if d.entry > 0:  # Only actual trade exits
            deal_type = 'BUY' if d.type == 0 else 'SELL'
            time_str = datetime.fromtimestamp(d.time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
            print(f'  Deal #{d.ticket} | {deal_type} close | Price: {d.price} | '
                  f'Profit: ${d.profit:.2f} | Volume: {d.volume} | Time: {time_str} | {d.comment}')
else:
    print('No recent deals')

# Account
info = mt5.account_info()
print(f'\nBalance: ${info.balance:.2f} | Equity: ${info.equity:.2f}')

mt5.shutdown()
