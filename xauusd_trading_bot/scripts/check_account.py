"""Check which MT5 account is connected."""
import MetaTrader5 as mt5

mt5.initialize()

info = mt5.account_info()
print(f"Account: {info.login}")
print(f"Server: {info.server}")
print(f"Name: {info.name}")
print(f"Balance: ${info.balance:.2f}")
print(f"Equity: ${info.equity:.2f}")
print(f"Type: {'Demo' if info.trade_mode == 0 else 'Real' if info.trade_mode == 2 else info.trade_mode}")
print(f"Leverage: 1:{info.leverage}")
print(f"Currency: {info.currency}")

mt5.shutdown()
