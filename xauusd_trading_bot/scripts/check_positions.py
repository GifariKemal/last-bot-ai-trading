"""Quick Position Check Script"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import MetaTrader5 as mt5
from datetime import datetime

# Initialize MT5
if not mt5.initialize(path=r'C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe'):
    print('MT5 init failed')
    sys.exit(1)

# Get account info
account = mt5.account_info()
if account:
    print('=' * 60)
    print('ACCOUNT STATUS')
    print('=' * 60)
    print(f'Server: {account.server}')
    print(f'Account: {account.login}')
    print(f'Balance: ${account.balance:.2f}')
    print(f'Equity: ${account.equity:.2f}')
    print(f'Profit: ${account.profit:.2f}')
    print(f'Margin Free: ${account.margin_free:.2f}')
    if account.margin_level:
        print(f'Margin Level: {account.margin_level:.2f}%')
    print()

# Get current price
tick = mt5.symbol_info_tick('XAUUSD')
if tick:
    print(f'Current XAUUSD Price: {tick.bid:.2f} / {tick.ask:.2f}')
    print()

# Get open positions
positions = mt5.positions_get(symbol='XAUUSD')
print('=' * 60)
print(f'OPEN POSITIONS: {len(positions) if positions else 0}')
print('=' * 60)

if positions:
    for i, pos in enumerate(positions, 1):
        direction = 'BUY' if pos.type == 0 else 'SELL'

        # Calculate profit distance
        if direction == 'BUY':
            profit_pips = (pos.price_current - pos.price_open) * 10
            to_sl_pips = (pos.price_current - pos.sl) * 10 if pos.sl else 0
            to_tp_pips = (pos.tp - pos.price_current) * 10 if pos.tp else 0
        else:
            profit_pips = (pos.price_open - pos.price_current) * 10
            to_sl_pips = (pos.sl - pos.price_current) * 10 if pos.sl else 0
            to_tp_pips = (pos.price_current - pos.tp) * 10 if pos.tp else 0

        print(f'\nPosition #{i}:')
        print(f'  Ticket: {pos.ticket}')
        print(f'  Type: {direction}')
        print(f'  Volume: {pos.volume} lots')
        print(f'  Entry: {pos.price_open:.2f}')
        print(f'  Current: {pos.price_current:.2f}')
        print(f'  SL: {pos.sl:.2f} (distance: {abs(to_sl_pips):.1f} pips)')
        print(f'  TP: {pos.tp:.2f} (distance: {abs(to_tp_pips):.1f} pips)')
        print(f'  Profit: ${pos.profit:.2f} ({profit_pips:+.1f} pips)')

        # Opened time
        opened = datetime.fromtimestamp(pos.time)
        print(f'  Opened: {opened.strftime("%Y-%m-%d %H:%M:%S")}')

        # Comment
        if pos.comment:
            print(f'  Comment: {pos.comment}')
else:
    print('No open positions')

print()
print('=' * 60)

mt5.shutdown()
