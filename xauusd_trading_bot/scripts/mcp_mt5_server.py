#!/usr/bin/env python3
"""
MCP Server for MetaTrader 5
Exposes MT5 live data and bot status to Claude Code via Model Context Protocol.

Tools:
  mt5_price         - Current bid/ask for any symbol
  mt5_account       - Balance, equity, margin, profit
  mt5_positions     - All open positions
  mt5_orders        - Pending orders
  mt5_history       - Recent closed deals
  mt5_bars          - OHLCV bars (M1/M5/M15/H1/H4/D1)
  bot_status        - Latest M15 analysis from bot log
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import MetaTrader5 as mt5
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

PROJECT_DIR = Path(__file__).parent

server = Server("metatrader5")

# ── MT5 Helpers ──────────────────────────────────────────────────────────────

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

def _connect():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

def _fmt_time(ts: int) -> str:
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M UTC")


# ── Tool Definitions ─────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="mt5_price",
            description="Get current live bid/ask price and spread for a symbol",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "e.g. XAUUSD, EURUSD"}
                },
                "required": ["symbol"]
            }
        ),
        types.Tool(
            name="mt5_account",
            description="Get MT5 account info: balance, equity, margin, free margin, profit",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="mt5_positions",
            description="List all currently open positions with profit/loss",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="mt5_orders",
            description="List all pending orders",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="mt5_history",
            description="Get recent closed deals (trade history)",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 7, "description": "Days back to look"}
                }
            }
        ),
        types.Tool(
            name="mt5_bars",
            description="Get OHLCV candlestick bars for a symbol and timeframe",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol":    {"type": "string", "description": "e.g. XAUUSD"},
                    "timeframe": {"type": "string", "description": "M1 M5 M15 H1 H4 D1"},
                    "count":     {"type": "integer", "default": 20, "description": "Number of bars"}
                },
                "required": ["symbol", "timeframe"]
            }
        ),
        types.Tool(
            name="bot_status",
            description="Get latest M15 candle analysis from the live trading bot log",
            inputSchema={"type": "object", "properties": {}}
        ),
    ]


# ── Tool Handlers ─────────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        # ── mt5_price ──────────────────────────────────────────────
        if name == "mt5_price":
            _connect()
            symbol = arguments["symbol"].upper()
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return [types.TextContent(type="text", text=f"Cannot get price for {symbol}: {mt5.last_error()}")]
            info = mt5.symbol_info(symbol)
            digits = info.digits if info else 2
            return [types.TextContent(type="text", text=json.dumps({
                "symbol":  symbol,
                "bid":     round(tick.bid, digits),
                "ask":     round(tick.ask, digits),
                "spread":  round(tick.ask - tick.bid, digits),
                "time":    _fmt_time(tick.time),
            }, indent=2))]

        # ── mt5_account ────────────────────────────────────────────
        elif name == "mt5_account":
            _connect()
            a = mt5.account_info()
            if a is None:
                return [types.TextContent(type="text", text=f"Error: {mt5.last_error()}")]
            return [types.TextContent(type="text", text=json.dumps({
                "login":        a.login,
                "server":       a.server,
                "currency":     a.currency,
                "leverage":     f"1:{a.leverage}",
                "balance":      round(a.balance, 2),
                "equity":       round(a.equity, 2),
                "margin":       round(a.margin, 2),
                "free_margin":  round(a.margin_free, 2),
                "margin_level": f"{a.margin_level:.1f}%" if a.margin_level else "N/A",
                "profit":       round(a.profit, 2),
            }, indent=2))]

        # ── mt5_positions ──────────────────────────────────────────
        elif name == "mt5_positions":
            _connect()
            positions = mt5.positions_get()
            if not positions:
                return [types.TextContent(type="text", text="No open positions")]
            result = []
            for p in positions:
                result.append({
                    "ticket":        p.ticket,
                    "symbol":        p.symbol,
                    "type":          "BUY" if p.type == 0 else "SELL",
                    "volume":        p.volume,
                    "open_price":    p.price_open,
                    "current_price": p.price_current,
                    "sl":            p.sl,
                    "tp":            p.tp,
                    "profit":        round(p.profit, 2),
                    "swap":          round(p.swap, 2),
                    "open_time":     _fmt_time(p.time),
                    "comment":       p.comment,
                })
            total_profit = sum(p["profit"] for p in result)
            return [types.TextContent(type="text", text=json.dumps({
                "count": len(result),
                "total_profit": round(total_profit, 2),
                "positions": result
            }, indent=2))]

        # ── mt5_orders ─────────────────────────────────────────────
        elif name == "mt5_orders":
            _connect()
            orders = mt5.orders_get()
            if not orders:
                return [types.TextContent(type="text", text="No pending orders")]
            result = [{
                "ticket": o.ticket,
                "symbol": o.symbol,
                "type":   o.type_description if hasattr(o, 'type_description') else o.type,
                "volume": o.volume_current,
                "price":  o.price_open,
                "sl":     o.sl,
                "tp":     o.tp,
            } for o in orders]
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        # ── mt5_history ────────────────────────────────────────────
        elif name == "mt5_history":
            _connect()
            days = int(arguments.get("days", 7))
            date_from = datetime.now() - timedelta(days=days)
            deals = mt5.history_deals_get(date_from, datetime.now())
            if not deals:
                return [types.TextContent(type="text", text=f"No deals in last {days} days")]
            result = []
            total_profit = 0.0
            for d in deals:
                if d.type in (0, 1):  # buy or sell only
                    result.append({
                        "ticket":  d.ticket,
                        "time":    _fmt_time(d.time),
                        "symbol":  d.symbol,
                        "type":    "BUY" if d.type == 0 else "SELL",
                        "volume":  d.volume,
                        "price":   d.price,
                        "profit":  round(d.profit, 2),
                        "comment": d.comment,
                    })
                    total_profit += d.profit
            wins  = sum(1 for r in result if r["profit"] > 0)
            loses = sum(1 for r in result if r["profit"] < 0)
            return [types.TextContent(type="text", text=json.dumps({
                "period":       f"last {days} days",
                "total_trades": len(result),
                "wins":         wins,
                "losses":       loses,
                "win_rate":     f"{wins/len(result)*100:.1f}%" if result else "N/A",
                "total_profit": round(total_profit, 2),
                "deals":        result[-30:]  # max 30
            }, indent=2))]

        # ── mt5_bars ───────────────────────────────────────────────
        elif name == "mt5_bars":
            _connect()
            symbol = arguments["symbol"].upper()
            tf_str = arguments["timeframe"].upper()
            tf     = TIMEFRAME_MAP.get(tf_str, mt5.TIMEFRAME_M15)
            count  = int(arguments.get("count", 20))
            rates  = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            if rates is None:
                return [types.TextContent(type="text", text=f"Error: {mt5.last_error()}")]
            bars = []
            for r in rates:
                bars.append({
                    "time":   datetime.utcfromtimestamp(r[0]).strftime("%m-%d %H:%M"),
                    "open":   r[1], "high": r[2], "low": r[3], "close": r[4],
                    "volume": int(r[5])
                })
            return [types.TextContent(type="text", text=json.dumps({
                "symbol": symbol, "timeframe": tf_str, "count": len(bars), "bars": bars
            }, indent=2))]

        # ── bot_status ─────────────────────────────────────────────
        elif name == "bot_status":
            today    = datetime.now().strftime("%Y-%m-%d")
            log_file = PROJECT_DIR / "logs" / "bot_activity" / f"bot_{today}.log"
            if not log_file.exists():
                return [types.TextContent(type="text", text=f"No log file for today: {log_file}")]
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # Extract last M15 analysis block
            block = []
            collecting = False
            for line in reversed(lines):
                stripped = line.strip()
                if "=" * 20 in stripped and collecting:
                    block.append(line)
                    break
                if "NEW M15 CANDLE ANALYSIS" in stripped or collecting:
                    collecting = True
                    block.append(line)
            if not block:
                return [types.TextContent(type="text", text="No M15 analysis found in today's log")]
            # Also get last tick for current price
            _connect()
            tick = mt5.symbol_info_tick("XAUUSDm")
            current_price = f"Current XAUUSD: Bid {tick.bid} / Ask {tick.ask}" if tick else ""
            text = current_price + "\n\n" + "".join(reversed(block))
            return [types.TextContent(type="text", text=text)]

        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        import traceback
        return [types.TextContent(type="text", text=f"Error in {name}: {e}\n{traceback.format_exc()}")]


# ── Entry Point ───────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
