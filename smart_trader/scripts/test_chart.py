"""Test chart visualization with current market data."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import MetaTrader5 as mt5
import pandas as pd
import yaml
from datetime import datetime, timezone, timedelta

import indicators as ind
import zone_detector as zdet
import scanner as scan
import regime_detector as rdet
import chart_visualizer as cv

# Connect
with open(_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)
acct = cfg["account"]
mt5_cfg = cfg["mt5"]
mt5.initialize(
    path=mt5_cfg.get("terminal_path", ""),
    login=acct["login"],
    password=acct["password"],
    server=acct["server"],
)

symbol = "XAUUSD"
tick = mt5.symbol_info_tick(symbol)
price = tick.ask

# Fetch data
h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 80)

df_h1 = pd.DataFrame(h1)
df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
df_m15 = pd.DataFrame(m15)
df_m15["time"] = pd.to_datetime(df_m15["time"], unit="s")

# Indicators
atr_val = ind.atr(df_h1, 14)
rsi_val = ind.rsi(df_h1, 14)
ema_trend = ind.h1_ema_trend(df_h1)
pd_zone = ind.premium_discount(df_h1)

# Regime
det = rdet.RegimeDetector()
res = det.detect(df_h1)

# Zones
zones = zdet.detect_all_zones(df_h1)
nearby = scan.find_nearby_zones(price, zones, 20.0) if zones else []

# Session
now_utc = datetime.now(timezone.utc)
sess = scan.current_session(now_utc)

# Simulate a SHORT entry for testing (based on current nearby zones)
direction = "SHORT"
sl_dist = atr_val * 2.0
tp_dist = atr_val * 3.0
sl = round(price + sl_dist, 2)
tp = round(price - tp_dist, 2)

entry_data = {
    "ticket": 9999999,
    "direction": direction,
    "price": price,
    "sl": sl,
    "tp": tp,
    "confidence": 0.78,
    "reason": "OB+FVG+LiqSweep confluence at premium zone, WK_UP regime",
    "signals": ["OB", "FVG", "LiqSweep", "OTE", "Premium"],
    "zone_type": "BEAR_OB",
    "regime": res["short_label"],
    "session": sess["name"],
    "rsi": rsi_val,
    "ema_trend": ema_trend,
    "pd_zone": pd_zone,
    "atr": atr_val,
    "pre_score": 0.62,
    "lot": 0.01,
}

print(f"Price: {price:.2f} | ATR: {atr_val:.1f} | RSI: {rsi_val:.1f}")
print(f"Regime: {res['short_label']} | Session: {sess['name']}")
print(f"Nearby zones: {len(nearby)}")
for z in nearby:
    print(f"  {z['type']} @ {z.get('high') or z.get('level', 0):.2f}")
print(f"\nSimulated {direction} entry:")
print(f"  SL={sl:.2f} ({sl_dist:.0f}pt) | TP={tp:.2f} ({tp_dist:.0f}pt)")

# Generate chart
output = str(_ROOT / "logs" / "charts" / "test_entry_chart.png")
path = cv.generate_entry_chart(df_m15, df_h1, entry_data, nearby, output)

if path:
    print(f"\nChart saved: {path}")
    # Try to send to Telegram
    import telegram_notifier as tg
    tg_cfg = cfg.get("telegram", {})
    if tg_cfg.get("enabled"):
        tg.init(tg_cfg["token"], tg_cfg["chat_id"], True)
        notifier = tg.get()
        caption = (
            "<b>TEST CHART</b> | XAUUSD %s\n"
            "Conf: %.2f | %s | %s\n"
            "Signals: %s"
        ) % (direction, 0.78, res["short_label"], sess["name"],
             " + ".join(entry_data["signals"]))
        notifier.send_chart(path, caption)
        print("Chart sent to Telegram!")
        import time
        time.sleep(3)  # wait for async send
else:
    print("Chart generation failed!")

mt5.shutdown()
