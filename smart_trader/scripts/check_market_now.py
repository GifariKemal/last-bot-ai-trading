"""Quick check: recent deals + bot log activity."""
import sys, os, glob
from pathlib import Path
from datetime import datetime, timezone, timedelta

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import MetaTrader5 as mt5
import yaml

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

# Recent deals
now = datetime.now(timezone.utc)
day_ago = now - timedelta(hours=24)
deals = mt5.history_deals_get(day_ago, now, group="*XAUUSD*")
if deals:
    print("=== DEALS LAST 24H ===")
    reason_map = {0:"CLIENT",1:"MOBILE",2:"WEB",3:"EXPERT",4:"SL",5:"TP",6:"SO",7:"ROLLOVER",8:"VMARGIN"}
    entry_map = {0:"IN", 1:"OUT", 2:"INOUT", 3:"CLOSE_BY"}
    for d in deals:
        t = datetime.fromtimestamp(d.time, tz=timezone.utc)
        wib = (t + timedelta(hours=7)).strftime("%H:%M WIB")
        dtype = "BUY" if d.type == 0 else "SELL" if d.type == 1 else "type=%d" % d.type
        reason = reason_map.get(d.reason, "r=%d" % d.reason)
        entry = entry_map.get(d.entry, "e=%d" % d.entry)
        print("  %s | %s %.2flot @ %.2f | %s | pnl=$%.2f | %s | #%d" % (
            wib, dtype, d.volume, d.price, entry, d.profit, reason, d.order))
else:
    print("No deals in last 24h")

mt5.shutdown()

# Bot log
print()
log_dir = _ROOT / "logs"
log_files = sorted(log_dir.glob("smart_trader*.log"), key=os.path.getmtime, reverse=True)
if log_files:
    print("=== RECENT BOT LOG ===")
    with open(log_files[0], "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    keywords = ["Claude", "CLAUDE", "Validating", "NO_TRADE", "APPROVED", "REJECTED",
                "Gate", "signal", "setup", "ENTRY", "EXIT", "market closed", "standing by",
                "Bot started", "SCAN"]
    relevant = [l.strip() for l in lines[-300:] if any(k in l for k in keywords)]
    for l in relevant[-30:]:
        if len(l) > 160:
            l = l[:160] + "..."
        print("  " + l)
else:
    print("No log files found")
