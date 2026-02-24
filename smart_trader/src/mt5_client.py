"""
MT5 Client — direct MetaTrader5 Python connection.
Wraps all MT5 calls so the rest of the code stays clean.
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
from loguru import logger


# ── Connection ────────────────────────────────────────────────────────────────

# Store connection params for auto-reconnect
_conn_params: dict = {}

def connect(
    terminal_path: str = "",
    login: int = 0,
    password: str = "",
    server: str = "",
) -> bool:
    global _conn_params
    _conn_params = {
        "terminal_path": terminal_path,
        "login": login,
        "password": password,
        "server": server,
    }

    kwargs = {}
    if terminal_path:
        kwargs["path"] = terminal_path
    if login:
        kwargs["login"] = login
    if password:
        kwargs["password"] = password
    if server:
        kwargs["server"] = server

    if not mt5.initialize(**kwargs):
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        return False

    info = mt5.account_info()
    if info is None:
        logger.error("MT5 account_info() returned None")
        mt5.shutdown()
        return False

    logger.info(
        f"MT5 connected | login={info.login} | {info.server} | "
        f"${info.balance:.2f} balance | leverage=1:{info.leverage}"
    )
    if login and info.login != login:
        logger.warning(f"Connected to account {info.login} but expected {login}")
    return True


def is_connected() -> bool:
    """Check if MT5 connection is alive by calling account_info()."""
    try:
        info = mt5.account_info()
        return info is not None
    except Exception:
        return False


def reconnect() -> bool:
    """Re-establish MT5 connection using stored params."""
    if not _conn_params:
        logger.error("No stored connection params — cannot reconnect")
        return False
    logger.warning("MT5 connection lost — attempting reconnect...")
    try:
        mt5.shutdown()
    except Exception:
        pass
    ok = connect(**_conn_params)
    if ok:
        logger.info("MT5 reconnected successfully")
    else:
        logger.error("MT5 reconnect failed")
    return ok


def disconnect():
    mt5.shutdown()
    logger.info("MT5 disconnected")


# ── Market data ───────────────────────────────────────────────────────────────

def is_market_open(symbol: str, max_tick_age_min: float = 5.0) -> bool:
    """
    Return True if market is open.
    Checks last tick age — if older than max_tick_age_min, market is closed.
    Spread alone is unreliable (IC Markets keeps spread non-zero on weekends).
    Returns False with reconnect attempt if MT5 connection is lost.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        # Distinguish: connection lost vs no tick data
        if not is_connected():
            reconnect()
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False
        else:
            return False
    tick_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
    age_min = (datetime.now(timezone.utc) - tick_time).total_seconds() / 60
    # IC Markets server is UTC+2/+3: tick.time may be ahead of system UTC.
    # Negative age = tick from "future" (server offset) → market is live.
    return age_min <= max_tick_age_min


def get_price(symbol: str) -> dict:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {}
    return {
        "bid": tick.bid,
        "ask": tick.ask,
        "mid": round((tick.bid + tick.ask) / 2, 2),
        "spread": round(tick.ask - tick.bid, 2),
    }


def get_candles(symbol: str, timeframe, count: int) -> pd.DataFrame:
    """
    timeframe: mt5.TIMEFRAME_H1, mt5.TIMEFRAME_M15, etc.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["time", "open", "high", "low", "close", "volume"]]


# ── Account info ──────────────────────────────────────────────────────────────

def get_account() -> dict:
    info = mt5.account_info()
    if info is None:
        return {}
    return {
        "login":       info.login,
        "balance":     info.balance,
        "equity":      info.equity,
        "margin_free": info.margin_free,
        "profit":      info.profit,
        "leverage":    info.leverage,
    }


# ── Positions ─────────────────────────────────────────────────────────────────

def get_positions(symbol: str = None) -> list[dict]:
    pos = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    if pos is None:
        return []
    result = []
    for p in pos:
        result.append({
            "ticket":       p.ticket,
            "symbol":       p.symbol,
            "type":         "LONG" if p.type == mt5.ORDER_TYPE_BUY else "SHORT",
            "volume":       p.volume,
            "price_open":   p.price_open,
            "price_current": p.price_current,
            "sl":           p.sl,
            "tp":           p.tp,
            "profit":       p.profit,
            "comment":      p.comment,
            "time_open":    datetime.fromtimestamp(p.time, tz=timezone.utc),
            "_raw":         p,
        })
    return result


# ── Order execution ───────────────────────────────────────────────────────────

def place_market_order(
    symbol: str,
    direction: str,
    lot: float,
    sl: float,
    tp: float,
    magic: int = 202602,
    comment: str = "smart_trader",
) -> dict:
    order_type = mt5.ORDER_TYPE_BUY if direction == "LONG" else mt5.ORDER_TYPE_SELL
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"success": False, "error": "No tick data"}

    price = tick.ask if direction == "LONG" else tick.bid

    request = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      symbol,
        "volume":      lot,
        "type":        order_type,
        "price":       price,
        "sl":          round(sl, 2),
        "tp":          round(tp, 2),
        "deviation":   30,
        "magic":       magic,
        "comment":     comment,
        "type_time":   mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = result.comment if result else str(mt5.last_error())
        return {"success": False, "error": err, "retcode": result.retcode if result else 0}

    return {"success": True, "ticket": result.order, "price": result.price}


def modify_sl_tp(ticket: int, sl: float, tp: float) -> bool:
    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        return False
    p = pos[0]
    result = mt5.order_send({
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   p.symbol,
        "position": ticket,
        "sl":       round(sl, 2),
        "tp":       round(tp, 2),
    })
    return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


def get_deal_close_info(position_ticket: int, magic: int = 202602) -> tuple:
    """
    Look up the closing deal for a position from MT5 deal history.
    Returns (close_price, pnl_usd) or (None, None) if not found.

    NOTE: IC Markets' position= filter is broken (returns ALL deals).
    We ONLY use manual position_id matching on the full deal list.
    """
    from datetime import timedelta
    try:
        now = datetime.now(timezone.utc)
        from_time = now - timedelta(days=7)

        all_deals = mt5.history_deals_get(from_time, now)
        if not all_deals:
            logger.info(f"deal_close({position_ticket}): no deals in last 7 days")
            return None, None

        # Manual filter: only deals with matching position_id AND magic number
        matched = [d for d in all_deals if d.position_id == position_ticket and d.magic == magic]
        if not matched:
            logger.info(
                f"deal_close({position_ticket}): no deals with position_id match "
                f"in {len(all_deals)} total deals"
            )
            return None, None

        logger.info(
            f"deal_close({position_ticket}): {len(matched)} deals matched position_id"
        )
        for i, d in enumerate(matched):
            logger.info(
                f"  matched[{i}] ticket={d.ticket} entry={d.entry} "
                f"type={d.type} price={d.price} profit={d.profit} "
                f"magic={d.magic} comment={d.comment} "
                f"time={datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat()}"
            )

        # Find OUT deal (entry=1)
        for d in reversed(matched):
            if d.entry == 1:
                logger.info(
                    f"deal_close({position_ticket}): SELECTED OUT deal "
                    f"ticket={d.ticket} price={d.price} profit={d.profit}"
                )
                return d.price, d.profit

        logger.info(
            f"deal_close({position_ticket}): {len(matched)} deals but none with entry=OUT"
        )
        return None, None
    except Exception as e:
        logger.info(f"deal_close({position_ticket}) error: {e}")
        return None, None


def close_partial(ticket: int, lot: float, symbol: str, is_long: bool) -> bool:
    order_type = mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)
    price = tick.bid if is_long else tick.ask
    result = mt5.order_send({
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      symbol,
        "volume":      lot,
        "type":        order_type,
        "position":    ticket,
        "price":       price,
        "deviation":   30,
        "magic":       202602,
        "comment":     "partial_close",
        "type_time":   mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    })
    return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
