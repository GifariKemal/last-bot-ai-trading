"""Check deals history + positions on ICMarkets account."""
import MetaTrader5 as mt5
from datetime import datetime

# Connect to ICMarkets account specifically
mt5.initialize(
    path=r"C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe",
    login=52725397,
    password="WlNE&S3ck65acj",
    server="ICMarketsSC-Demo",
)

info = mt5.account_info()
print(f"Account: {info.login} | Server: {info.server} | Balance: ${info.balance:.2f}\n")

# Open positions
positions = mt5.positions_get(symbol="XAUUSDm")
print(f"=== Open Positions: {len(positions) if positions else 0} ===")
if positions:
    for p in positions:
        direction = "BUY" if p.type == 0 else "SELL"
        print(
            f"  #{p.ticket} {direction} @ {p.price_open} | "
            f"SL:{p.sl} TP:{p.tp} | P/L:{p.profit:+.2f} | Vol:{p.volume}"
        )

# Deals history Feb 18-21
print(f"\n=== Deals History (Feb 18-20) ===")
deals = mt5.history_deals_get(datetime(2026, 2, 18), datetime(2026, 2, 21), group="XAUUSDm")
if deals:
    print(f"Total deals: {len(deals)}\n")
    for d in deals:
        deal_time = datetime.fromtimestamp(d.time)
        entry_map = {0: "IN", 1: "OUT", 2: "INOUT", 3: "OUT_BY"}
        type_map = {0: "BUY", 1: "SELL"}
        print(
            f"{deal_time} | #{d.position_id} | "
            f"{type_map.get(d.type, d.type)} {entry_map.get(d.entry, d.entry)} | "
            f"Vol:{d.volume} | Price:{d.price} | P/L:{d.profit:+.2f} | "
            f"Comment: {d.comment}"
        )
else:
    print("No deals found |", mt5.last_error())

# Orders history
print(f"\n=== Orders History (Feb 18-20) ===")
orders = mt5.history_orders_get(datetime(2026, 2, 18), datetime(2026, 2, 21), group="XAUUSDm")
if orders:
    print(f"Total orders: {len(orders)}")
    for o in orders:
        order_time = datetime.fromtimestamp(o.time_setup)
        type_map = {0: "BUY", 1: "SELL", 2: "BUY_LIMIT", 3: "SELL_LIMIT", 4: "BUY_STOP", 5: "SELL_STOP"}
        state_map = {0: "STARTED", 1: "PLACED", 2: "CANCELED", 3: "PARTIAL", 4: "FILLED", 5: "REJECTED"}
        print(
            f"{order_time} | #{o.position_id} | {type_map.get(o.type, o.type)} | "
            f"Vol:{o.volume_current}/{o.volume_initial} | Price:{o.price_open} | "
            f"SL:{o.sl} TP:{o.tp} | State:{state_map.get(o.state, o.state)} | "
            f"Comment: {o.comment}"
        )
else:
    print("No orders found |", mt5.last_error())

mt5.shutdown()
