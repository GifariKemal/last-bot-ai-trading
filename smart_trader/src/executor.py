"""
Executor — trade execution and position management.
Handles: order placement, BE, partial close / profit lock, RR trailing stop.
Time-based exits: stale trade tighten, scratch exit.
Also: Claude-powered smart exit review (profit-focused).
"""
import math
from datetime import datetime, timezone
from typing import Optional
from loguru import logger

import mt5_client as mt5c
import claude_validator as validator
import telegram_notifier as tg


# ── Lot sizing ────────────────────────────────────────────────────────────────

def calc_lot(
    balance: float,
    risk_pct: float,
    sl_distance: float,
    min_lot: float = 0.01,
    contract_size: float = 100.0,
) -> float:
    """
    Risk-based lot sizing for XAUUSD.
    lot = (balance × risk_pct/100) / (sl_distance × contract_size)
    Falls back to min_lot if result < min_lot.
    """
    if sl_distance <= 0 or balance <= 0:
        return min_lot
    risk_usd = balance * risk_pct / 100.0
    lot = risk_usd / (sl_distance * contract_size)
    lot = math.floor(lot / min_lot) * min_lot  # round down to nearest step
    return max(round(lot, 2), min_lot)


# ── Order placement ───────────────────────────────────────────────────────────

def place_trade(
    symbol: str,
    direction: str,
    sl: float,
    tp: float,
    balance: float,
    cfg: dict,
) -> Optional[dict]:
    """
    Calculate lot size and place a market order.
    Returns result dict from mt5_client or None on failure.
    """
    price_info = mt5c.get_price(symbol)
    if not price_info:
        logger.error("Cannot place trade — no price data")
        return None

    price = price_info["ask"] if direction == "LONG" else price_info["bid"]
    sl_dist = abs(price - sl)

    risk_pct = cfg.get("risk_percent", 1.0)
    min_lot  = cfg.get("min_lot", 0.01)
    lot      = calc_lot(balance, risk_pct, sl_dist, min_lot)

    logger.info(
        f"Placing {direction} | {symbol} | lot={lot} | SL={sl:.2f} | TP={tp:.2f} | "
        f"SL_dist={sl_dist:.1f}pt | balance=${balance:.2f}"
    )

    result = mt5c.place_market_order(
        symbol=symbol,
        direction=direction,
        lot=lot,
        sl=sl,
        tp=tp,
        magic=cfg.get("magic", 202602),
        comment="Opus-4.6",
    )

    if result.get("success"):
        logger.bind(kind="TRADE").info(
            f"ORDER FILLED | ticket={result['ticket']} | price={result['price']:.2f} | "
            f"lot={lot} | SL={sl:.2f} | TP={tp:.2f}"
        )
    else:
        logger.error(f"Order failed: {result.get('error')} (retcode={result.get('retcode')})")

    return result


# ── Constants ────────────────────────────────────────────────────────────────

BE_TRIGGER_MULT       = 0.7   # BE at 0.7× SL distance (was 1.0×)
LOCK_TRIGGER_MULT     = 1.5   # Partial close / profit lock at 1.5× SL
TRAIL_KEEP_PCT        = 0.40  # Trail SL keeps 40% of profit as cushion (was 50%)
TRAIL_ACTIVATE_PCT    = 0.40  # Trail activates when trail > 40% of SL dist (was 50%)
STALE_TIGHTEN_MIN     = 90    # After 90 min with no progress, tighten SL
STALE_PROGRESS_MULT   = 0.5   # "No progress" = profit < 50% of SL distance
STALE_SL_REDUCE       = 0.5   # Reduce max loss to 50% of original SL
SCRATCH_EXIT_MIN      = 60    # After 60 min flat, close at ~breakeven
SCRATCH_FLAT_PTS      = 5.0   # "Flat" = less than 5pt movement either way


# ── Position management ───────────────────────────────────────────────────────

_scratched_tickets: set[int] = set()  # prevent repeat scratch attempts


def manage_positions(symbol: str, cfg: dict) -> None:
    """
    Multi-stage exit management for all open positions.
    Stage 0: Scratch exit (flat after 60min)
    Stage 0b: Time-based SL tighten (stale after 90min)
    Stage 1: Move SL to BE at 0.7×SL profit
    Stage 2: Partial close 50% (or profit-lock SL) at 1.5×SL profit
    Stage 3: RR-based trailing stop on remainder (40% of peak profit)
    """
    positions = mt5c.get_positions(symbol)
    if not positions:
        return

    price_info = mt5c.get_price(symbol)
    if not price_info:
        return

    now = datetime.now(timezone.utc)

    for pos in positions:
        # Only manage OUR positions
        if pos.get("_raw") and pos["_raw"].magic != cfg.get("magic", 202602):
            continue
        _manage_one(pos, price_info, cfg, now)


def _manage_one(pos: dict, price_info: dict, cfg: dict, now: datetime) -> None:
    ticket    = pos["ticket"]
    direction = pos["type"]       # "LONG" or "SHORT"
    entry     = pos["price_open"]
    current_sl = pos["sl"]
    current_tp = pos["tp"]
    volume    = pos["volume"]
    symbol    = pos["symbol"]

    price = price_info["mid"]

    # SL distance from entry (original)
    sl_dist = abs(entry - current_sl)
    if sl_dist <= 0:
        # SL already at entry (BE) — use a reference distance
        # Estimate from TP distance / RR ratio
        tp_dist = abs(current_tp - entry)
        sl_dist = tp_dist / 1.67 if tp_dist > 0 else 20.0

    profit_pts = (price - entry) if direction == "LONG" else (entry - price)

    # Position age
    time_open = pos.get("time_open")
    if time_open:
        age_min = (now - time_open).total_seconds() / 60
    else:
        age_min = 0

    # ── Stage 0: Scratch exit (flat after 60min) ────────────────────────────
    if (ticket not in _scratched_tickets
            and age_min >= SCRATCH_EXIT_MIN
            and abs(profit_pts) < SCRATCH_FLAT_PTS
            and _sl_is_below_entry(direction, current_sl, entry)):
        # Close at ~breakeven rather than wait for SL
        _scratched_tickets.add(ticket)
        logger.bind(kind="TRADE").info(
            f"SCRATCH EXIT | ticket={ticket} | {direction} | "
            f"P/L={profit_pts:+.1f}pt after {age_min:.0f}min (flat)"
        )
        logger.bind(kind="JOURNAL").info(
            f"EXIT | ticket={ticket} | {direction} | pnl_pts={profit_pts:+.1f} | "
            f"age={age_min:.0f}min | reason=scratch_exit"
        )
        close_position(ticket, symbol, direction, volume)
        _notif = tg.get()
        if _notif:
            _notif.send_exit(
                direction=direction,
                ticket=ticket,
                entry_price=entry,
                exit_price=price,
                pnl_pts=profit_pts,
                pnl_usd=pos.get("profit", 0),
                age_min=age_min,
                reason="scratch_exit",
            )
        return

    # ── Stage 0b: Time-based SL tighten (stale after 90min) ────────────────
    if (age_min >= STALE_TIGHTEN_MIN
            and profit_pts < sl_dist * STALE_PROGRESS_MULT
            and _sl_is_below_entry(direction, current_sl, entry)):
        # Reduce max loss: move SL to half of original distance
        half_dist = sl_dist * STALE_SL_REDUCE
        if direction == "LONG":
            new_sl = round(entry - half_dist, 2)
            if new_sl > current_sl + 0.5:
                logger.bind(kind="TRADE").info(
                    f"STALE TIGHTEN | ticket={ticket} | {age_min:.0f}min | "
                    f"SL {current_sl:.2f}→{new_sl:.2f} (halved risk)"
                )
                mt5c.modify_sl_tp(ticket, new_sl, current_tp)
                _notif = tg.get()
                if _notif:
                    _notif.send_position_update(
                        ticket=ticket, direction=direction,
                        action="STALE_TIGHTEN", old_sl=current_sl, new_sl=new_sl,
                        pnl_pts=profit_pts, extra=f"{age_min:.0f}min stale",
                    )
                current_sl = new_sl
        else:
            new_sl = round(entry + half_dist, 2)
            if new_sl < current_sl - 0.5:
                logger.bind(kind="TRADE").info(
                    f"STALE TIGHTEN | ticket={ticket} | {age_min:.0f}min | "
                    f"SL {current_sl:.2f}→{new_sl:.2f} (halved risk)"
                )
                mt5c.modify_sl_tp(ticket, new_sl, current_tp)
                _notif = tg.get()
                if _notif:
                    _notif.send_position_update(
                        ticket=ticket, direction=direction,
                        action="STALE_TIGHTEN", old_sl=current_sl, new_sl=new_sl,
                        pnl_pts=profit_pts, extra=f"{age_min:.0f}min stale",
                    )
                current_sl = new_sl

    # ── Stage 1: Break-even at 0.7×SL ──────────────────────────────────────
    be_trigger = sl_dist * BE_TRIGGER_MULT
    if profit_pts >= be_trigger and _sl_is_below_entry(direction, current_sl, entry):
        new_sl = entry + 0.2 if direction == "LONG" else entry - 0.2  # tiny buffer
        logger.bind(kind="TRADE").info(
            f"BE | ticket={ticket} | SL {current_sl:.2f}→{new_sl:.2f}"
        )
        logger.bind(kind="JOURNAL").info(
            f"MILESTONE | ticket={ticket} | {direction} | BE | "
            f"pnl_pts={profit_pts:+.1f} | SL {current_sl:.2f}→{new_sl:.2f}"
        )
        mt5c.modify_sl_tp(ticket, new_sl, current_tp)
        _notif = tg.get()
        if _notif:
            _notif.send_position_update(
                ticket=ticket, direction=direction,
                action="BE", old_sl=current_sl, new_sl=new_sl,
                pnl_pts=profit_pts,
            )
        current_sl = new_sl  # update local for stage 2 check

    # ── Stage 2: Partial close / profit lock at 1.5×SL ───────────────────────
    lock_trigger = sl_dist * LOCK_TRIGGER_MULT
    lock_comment = pos.get("comment", "")
    if profit_pts >= lock_trigger and "locked" not in lock_comment:
        if volume > 0.01:
            # Partial close 50%
            close_vol = round(volume * 0.5, 2)
            is_long   = direction == "LONG"
            ok = mt5c.close_partial(ticket, close_vol, symbol, is_long)
            if ok:
                logger.bind(kind="TRADE").info(
                    f"PARTIAL CLOSE | ticket={ticket} | vol={close_vol:.2f} | "
                    f"profit_pts={profit_pts:.1f}"
                )
                logger.bind(kind="JOURNAL").info(
                    f"MILESTONE | ticket={ticket} | {direction} | PARTIAL_CLOSE | "
                    f"pnl_pts={profit_pts:+.1f} | vol={close_vol:.2f}"
                )
                _notif = tg.get()
                if _notif:
                    _notif.send_position_update(
                        ticket=ticket, direction=direction,
                        action="PROFIT_LOCK", old_sl=current_sl, new_sl=current_sl,
                        pnl_pts=profit_pts, extra=f"partial {close_vol:.2f} lot closed",
                    )
        else:
            # 0.01 lot — lock profit by moving SL to entry + 50% of profit
            lock_sl = (entry + profit_pts * 0.5) if direction == "LONG" \
                      else (entry - profit_pts * 0.5)
            lock_sl = round(lock_sl, 2)
            if (direction == "LONG" and lock_sl > current_sl) or \
               (direction == "SHORT" and lock_sl < current_sl):
                logger.bind(kind="TRADE").info(
                    f"PROFIT LOCK | ticket={ticket} | SL {current_sl:.2f}→{lock_sl:.2f}"
                )
                logger.bind(kind="JOURNAL").info(
                    f"MILESTONE | ticket={ticket} | {direction} | PROFIT_LOCK | "
                    f"pnl_pts={profit_pts:+.1f} | SL {current_sl:.2f}→{lock_sl:.2f}"
                )
                mt5c.modify_sl_tp(ticket, lock_sl, current_tp)
                _notif = tg.get()
                if _notif:
                    _notif.send_position_update(
                        ticket=ticket, direction=direction,
                        action="PROFIT_LOCK", old_sl=current_sl, new_sl=lock_sl,
                        pnl_pts=profit_pts,
                    )

    # ── Stage 3: RR-based trailing stop ──────────────────────────────────────
    _rr_trail(pos, price_info, sl_dist, direction, symbol)


def _sl_is_below_entry(direction: str, sl: float, entry: float) -> bool:
    """True if SL has NOT been moved to BE yet."""
    if direction == "LONG":
        return sl < entry
    return sl > entry


def _rr_trail(pos: dict, price_info: dict, sl_dist: float, direction: str, symbol: str) -> None:
    """RR-based trailing: SL = price - 40% of profit (tighter than before)."""
    ticket  = pos["ticket"]
    entry   = pos["price_open"]
    cur_sl  = pos["sl"]
    cur_tp  = pos["tp"]
    price   = price_info["mid"]

    profit_pts = (price - entry) if direction == "LONG" else (entry - price)
    trail_sl_dist = profit_pts * TRAIL_KEEP_PCT

    # Only activate when trail distance > activation threshold
    if trail_sl_dist < sl_dist * TRAIL_ACTIVATE_PCT:
        return

    if direction == "LONG":
        new_sl = round(price - trail_sl_dist, 2)
        if new_sl > cur_sl + 0.5:  # only move if meaningful improvement
            logger.bind(kind="TRADE").info(
                f"TRAIL | ticket={ticket} | SL {cur_sl:.2f}→{new_sl:.2f}"
            )
            mt5c.modify_sl_tp(ticket, new_sl, cur_tp)
    else:
        new_sl = round(price + trail_sl_dist, 2)
        if new_sl < cur_sl - 0.5:
            logger.bind(kind="TRADE").info(
                f"TRAIL | ticket={ticket} | SL {cur_sl:.2f}→{new_sl:.2f}"
            )
            mt5c.modify_sl_tp(ticket, new_sl, cur_tp)


# ── Close position ───────────────────────────────────────────────────────────

def close_position(ticket: int, symbol: str, direction: str, volume: float) -> bool:
    """Fully close an open position."""
    is_long = direction == "LONG"
    ok = mt5c.close_partial(ticket, volume, symbol, is_long)
    if ok:
        logger.bind(kind="TRADE").info(
            f"CLOSED | ticket={ticket} | {direction} {volume} lot"
        )
    else:
        logger.error(f"Failed to close ticket={ticket}")
    return ok


# ── Claude Smart Exit Review ─────────────────────────────────────────────────

def review_positions_with_claude(
    symbol: str,
    cfg: dict,
    claude_cfg: dict,
    market_data: dict,
) -> None:
    """
    Ask Claude to analyze each open position for OPTIMAL EXIT TIMING.
    Focus: should we take profit now, tighten SL, or let it run?
    NOT for panic-closing — the multi-stage exit handles risk management.
    """
    positions = mt5c.get_positions(symbol)
    if not positions:
        return

    price_info = mt5c.get_price(symbol)
    if not price_info:
        return

    price = price_info["mid"]
    now = datetime.now(timezone.utc)

    for pos in positions:
        # Only review OUR positions (magic=202602)
        if pos.get("_raw") and pos["_raw"].magic != cfg.get("magic", 202602):
            continue

        ticket    = pos["ticket"]
        direction = pos["type"]
        entry     = pos["price_open"]
        volume    = pos["volume"]
        sl        = pos["sl"]
        tp        = pos["tp"]

        # Position age
        time_open = pos.get("time_open")
        if time_open:
            duration_min = (now - time_open).total_seconds() / 60
        else:
            duration_min = 0

        # Calculate P/L
        pnl_pts = (price - entry) if direction == "LONG" else (entry - price)
        pnl_usd = pos.get("profit", 0)

        # SL distance
        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            tp_dist = abs(tp - entry)
            sl_dist = tp_dist / 1.67 if tp_dist > 0 else 20.0

        # Determine current stage
        if pnl_pts >= sl_dist * 1.5:
            stage = "TRAILING (profit locked)"
        elif not _sl_is_below_entry(direction, sl, entry):
            stage = "BREAK-EVEN (SL at entry)"
        elif pnl_pts > 0:
            stage = f"IN PROFIT +{pnl_pts:.1f}pt"
        else:
            stage = f"UNDERWATER {pnl_pts:.1f}pt"

        # TP distance remaining
        tp_remaining = abs(tp - price)

        # Suggested tighten SL (lock 60% of current profit)
        if pnl_pts > 5:
            if direction == "LONG":
                tighten_sl = round(entry + pnl_pts * 0.6, 2)
            else:
                tighten_sl = round(entry - pnl_pts * 0.6, 2)
        else:
            tighten_sl = sl

        pos_data = {
            "price":          price,
            "atr":            market_data.get("atr", 15),
            "session":        market_data.get("session", "UNKNOWN"),
            "ema_trend":      market_data.get("ema_trend", "NEUTRAL"),
            "rsi":            market_data.get("rsi", 50),
            "pd_zone":        market_data.get("pd_zone", "EQUILIBRIUM"),
            "nearby_signals": market_data.get("nearby_signals", "none"),
            "direction":      direction,
            "entry":          entry,
            "pnl_pts":        pnl_pts,
            "pnl_usd":        pnl_usd,
            "sl":             sl,
            "sl_dist":        sl_dist,
            "tp":             tp,
            "tp_remaining":   tp_remaining,
            "duration_min":   duration_min,
            "stage":          stage,
            "tighten_sl":     tighten_sl,
        }

        logger.info(
            f"  Exit review | {direction} ticket={ticket} | "
            f"P/L={pnl_pts:+.1f}pt (${pnl_usd:+.2f}) | {stage} | {duration_min:.0f}min"
        )

        response = validator.review_exit(pos_data, claude_cfg)
        if response is None:
            logger.warning(f"  Exit review failed for ticket={ticket}")
            continue

        action = response["action"]
        reason = response.get("reason", "")

        if action == "TAKE_PROFIT":
            # Only close if actually in profit
            if pnl_pts > 3:
                logger.bind(kind="TRADE").info(
                    f"CLAUDE TAKE PROFIT | ticket={ticket} | +{pnl_pts:.1f}pt — {reason}"
                )
                logger.bind(kind="JOURNAL").info(
                    f"EXIT | ticket={ticket} | {direction} | pnl_pts={pnl_pts:+.1f} | "
                    f"pnl_usd=${pnl_usd:+.2f} | age={duration_min:.0f}min | "
                    f"reason=claude_take_profit | {reason}"
                )
                close_position(ticket, symbol, direction, volume)
                _notif = tg.get()
                if _notif:
                    _notif.send_claude_exit_review(
                        ticket=ticket, direction=direction,
                        action="TAKE_PROFIT",
                        pnl_pts=pnl_pts, pnl_usd=pnl_usd,
                        claude_reason=reason,
                    )
            else:
                logger.info(f"  TAKE_PROFIT ignored — only {pnl_pts:+.1f}pt (need >3pt)")

        elif action == "TIGHTEN":
            new_sl = response.get("new_sl", 0)
            if new_sl > 0:
                # Validate: new SL must be tighter (better) than current
                valid = (direction == "LONG" and new_sl > sl) or \
                        (direction == "SHORT" and new_sl < sl)
                if valid:
                    logger.bind(kind="TRADE").info(
                        f"CLAUDE TIGHTEN | ticket={ticket} | SL {sl:.2f}→{new_sl:.2f} — {reason}"
                    )
                    logger.bind(kind="JOURNAL").info(
                        f"MILESTONE | ticket={ticket} | {direction} | CLAUDE_TIGHTEN | "
                        f"pnl_pts={pnl_pts:+.1f} | SL {sl:.2f}→{new_sl:.2f} | {reason}"
                    )
                    mt5c.modify_sl_tp(ticket, round(new_sl, 2), tp)
                    _notif = tg.get()
                    if _notif:
                        _notif.send_claude_exit_review(
                            ticket=ticket, direction=direction,
                            action="TIGHTEN",
                            pnl_pts=pnl_pts, pnl_usd=pnl_usd,
                            claude_reason=reason,
                            new_sl=round(new_sl, 2),
                            old_sl=sl,
                        )
                else:
                    logger.debug(f"  Tighten rejected — new SL {new_sl:.2f} not tighter than {sl:.2f}")

        else:  # HOLD
            logger.info(f"  HOLD ticket={ticket} — {reason}")
