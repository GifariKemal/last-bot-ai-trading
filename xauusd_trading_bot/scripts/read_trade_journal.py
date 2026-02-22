"""
Trade Journal Reader — pretty-print a ticket's lifecycle from the journal file.

Usage:
  python scripts/read_trade_journal.py                   # latest trade today
  python scripts/read_trade_journal.py 1488233709        # specific ticket
  python scripts/read_trade_journal.py --daily           # daily summary stats
  python scripts/read_trade_journal.py --daily 20260220  # specific date (YYYYMMDD)
"""

import json
import sys
from pathlib import Path

LOG_DIR = Path("logs/trade_journal")
SEP = "─" * 70


def color(s, code): return f"\033[{code}m{s}\033[0m"
def green(s): return color(s, "32")
def red(s): return color(s, "31")
def yellow(s): return color(s, "33")
def cyan(s): return color(s, "36")
def bold(s): return color(s, "1")
def pnl_color(v): return green(f"+${v:.2f}") if v > 0 else red(f"${v:.2f}") if v < 0 else f"$0.00"


def stage_label(s: str) -> str:
    icons = {
        "OPEN": "🔵",
        "BE_REACHED": "🟡",
        "TRAILING": "🟢",
        "PRE_CLOSE_LOCKED": "🔒",
        "PARTIAL_CLOSED": "✂️",
        "PROFIT_LOCKED": "🔒",
    }
    return f"{icons.get(s, '•')} {s}"


def print_ticket(journal: dict):
    ticket = journal["ticket"]
    direction = journal["direction"]
    volume = journal["volume"]
    symbol = journal["symbol"]
    e = journal["entry"]
    snaps = journal["snapshots"]
    ex = journal.get("exit")

    dir_color = green(direction) if direction == "BUY" else red(direction)
    print(f"\n{bold(SEP)}")
    print(f"  {bold('TRADE JOURNAL')}  #{cyan(str(ticket))}  {dir_color}  {symbol}  {volume} lot")
    print(f"{bold(SEP)}")

    # Entry
    print(f"\n{bold('ENTRY')}  {e['ts_utc']}")
    print(f"  Price      {e['price']:.2f}   SL {red(str(e['sl']))}   TP {green(str(e['tp']))}")
    print(f"  SL pips    {e['sl_pips']:.1f}   TP pips {e['tp_pips']:.1f}   TP/RR {yellow(str(e['tp_rr']))}R")
    print(f"  Confluence {yellow(str(round(e['confluence'],3)))}   Regime {cyan(e['regime'])}   Session {e['session']}")
    print(f"  Signals    {e['smc_signals']}   Balance ${e['balance']:.2f}")

    # Snapshots
    if snaps:
        print(f"\n{bold('SNAPSHOTS')}  ({len(snaps)} @ 2-min intervals)")
        print(f"  {'Time':>10}  {'Price':>8}  {'P&L':>9}  {'RR':>6}  {'SL':>9}  Stage")
        print(f"  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*6}  {'-'*9}  {'-'*20}")
        for s in snaps:
            ts = s["ts_utc"][11:16]  # HH:MM
            pnl_str = f"+${s['pnl_usd']:.2f}" if s["pnl_usd"] >= 0 else f"${s['pnl_usd']:.2f}"
            pnl_c = green(f"{pnl_str:>9}") if s["pnl_usd"] > 0 else red(f"{pnl_str:>9}")
            rr_c = green(f"{s['rr']:>6.3f}") if s["rr"] > 0 else red(f"{s['rr']:>6.3f}")
            action = f"  ← {s['action']}" if s.get("action") else ""
            print(f"  {ts:>10}  {s['price']:>8.2f}  {pnl_c}  {rr_c}  {s['sl']:>9.2f}  {stage_label(s['stage'])}{action}")

        # Mini stats
        pnl_vals = [s["pnl_usd"] for s in snaps]
        mfe = max(pnl_vals)
        mae = min(pnl_vals)
        print(f"\n  In-trade: peak {green(f'${mfe:.2f}')}  trough {red(f'${mae:.2f}')}")
    else:
        print(f"\n  {yellow('(no snapshots yet — position opened but no 2-min interval passed)')}")

    # Exit
    if ex:
        print(f"\n{bold('EXIT')}  {ex['ts_utc']}")
        pnl_str = pnl_color(ex["pnl_usd"])
        print(f"  Price      {ex['price']:.2f}   P&L  {pnl_str}   RR {yellow(str(ex['rr_final']))}R")
        print(f"  Reason     {ex['exit_reason']}")
        print(f"  Duration   {ex['duration_minutes']} min")
        print(f"  MFE        {green(f'${ex[\"mfe_usd\"]:.2f}')}   MAE {red(f'${ex[\"mae_usd\"]:.2f}')}")
        print(f"  Stage      {stage_label(ex['stage_reached'])}   Snapshots {ex['snapshots_count']}")
    else:
        print(f"\n{bold('EXIT')}  {yellow('(position still open)')}")

    print(f"\n{bold(SEP)}\n")


def daily_summary(date_str: str):
    path = LOG_DIR / f"daily_{date_str}.jsonl"
    if not path.exists():
        print(f"No daily log for {date_str}")
        return

    entries, exits, snapshots = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r["event"] == "ENTRY":
                    entries.append(r)
                elif r["event"] == "EXIT":
                    exits.append(r)
                elif r["event"] == "SNAPSHOT":
                    snapshots.append(r)
            except Exception:
                pass

    wins = [e for e in exits if e.get("pnl_usd", 0) > 0]
    losses = [e for e in exits if e.get("pnl_usd", 0) < 0]
    total_pnl = sum(e.get("pnl_usd", 0) for e in exits)
    avg_duration = (
        sum(e.get("duration_minutes", 0) for e in exits) / len(exits)
        if exits else 0
    )

    print(f"\n{bold(SEP)}")
    print(f"  {bold('DAILY SUMMARY')}  {date_str}")
    print(f"{bold(SEP)}")
    print(f"  Entries    {len(entries)}")
    print(f"  Exits      {len(exits)}  ({len(wins)} wins / {len(losses)} losses)")
    pnl_str = pnl_color(total_pnl)
    print(f"  Total P&L  {pnl_str}")
    if exits:
        print(f"  Win Rate   {len(wins)/len(exits)*100:.0f}%")
        print(f"  Avg hold   {avg_duration:.0f} min")
        if wins:
            print(f"  Avg win    {green(f'${sum(e[\"pnl_usd\"] for e in wins)/len(wins):.2f}')}")
        if losses:
            print(f"  Avg loss   {red(f'${sum(e[\"pnl_usd\"] for e in losses)/len(losses):.2f}')}")

    print(f"\n  {bold('Ticket Details:')}")
    for ex in exits:
        t = ex["ticket"]
        pnl = ex.get("pnl_usd", 0)
        reason = ex.get("exit_reason", "")
        dur = ex.get("duration_minutes", 0)
        rr = ex.get("rr_final", 0)
        stage = ex.get("stage_reached", "")
        mfe = ex.get("mfe_usd", 0)
        mae = ex.get("mae_usd", 0)
        pnl_c = pnl_color(pnl)
        print(f"    #{t:<12} {pnl_c:>10}  RR={rr:>5.2f}  dur={dur:>4}m  stage={stage_label(stage):<20}  MFE={green(f'${mfe:.2f}'):>8}  MAE={red(f'${mae:.2f}'):>8}  {reason}")

    print(f"\n{bold(SEP)}\n")


def find_latest_ticket() -> str | None:
    """Find the most recently modified ticket JSON."""
    files = sorted(LOG_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None


def main():
    if not LOG_DIR.exists():
        print(f"No trade journal directory found at: {LOG_DIR}")
        print("Run the bot at least once to create journal entries.")
        return

    args = sys.argv[1:]

    if "--daily" in args:
        args.remove("--daily")
        date_str = args[0] if args else None
        if not date_str:
            # Find latest daily file
            files = sorted(LOG_DIR.glob("daily_*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
            if not files:
                print("No daily journal files found.")
                return
            # Extract date from filename: daily_YYYYMMDD.jsonl
            date_str = files[0].stem.replace("daily_", "")
        daily_summary(date_str)
        return

    if args:
        ticket_id = int(args[0])
        # Find the file for this ticket
        files = list(LOG_DIR.glob(f"*_{ticket_id}.json"))
        if not files:
            print(f"No journal file found for ticket #{ticket_id}")
            return
        with open(files[0], encoding="utf-8") as f:
            journal = json.load(f)
        print_ticket(journal)
        return

    # Default: show latest
    files = sorted(LOG_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        print("No trade journal files found yet. The bot will create them when trades are opened.")
        return

    print(f"Showing latest {min(3, len(files))} trades (most recent first):")
    for f in files[:3]:
        with open(f, encoding="utf-8") as fh:
            journal = json.load(fh)
        print_ticket(journal)


if __name__ == "__main__":
    main()
