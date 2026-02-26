"""
Smart Trader -- 6-Month Backtest Script.

Validates the full adaptive pipeline (regime detector, adaptive engine, zone
detector, indicators) over historical M15 data.  Claude is SKIPPED -- algo
pre-score acts as the entry filter instead.

Run:
  cd smart_trader && python backtest_smart.py
  python backtest_smart.py --months 1 --verbose
  python backtest_smart.py --months 6 --compare
"""
import os, sys, argparse, math, csv, warnings
warnings.filterwarnings("ignore", message="no explicit representation of timezones")
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# -- Path setup ---------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "src"))

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import yaml

# Reuse existing modules (read-only, no MT5 calls during sim)
import indicators as ind
import zone_detector as zdet
import scanner as scan
import regime_detector as rdet
import adaptive_engine as ae

# Silence loguru during backtest (zone_detector/regime_detector use it)
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")


# ===============================================================================
# SimPosition
# ===============================================================================

@dataclass
class SimPosition:
    ticket: int
    direction: str          # LONG / SHORT
    entry_price: float
    entry_time: datetime
    sl: float
    tp: float
    sl_original: float      # frozen for RR calculation
    lot: float
    zone_type: str
    signals: list
    signal_count: int
    regime: str
    regime_cat: str
    session: str
    pre_score: float
    atr: float
    # Entry context (for analysis)
    rsi_val: float = 50.0
    ema_trend: str = "NEUTRAL"
    pd_zone: str = "EQUILIBRIUM"
    # Mutable state
    be_triggered: bool = False
    lock_triggered: bool = False
    stale_tightened: bool = False
    peak_profit: float = 0.0
    # Exit
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    close_type: str = ""
    pnl_pts: float = 0.0


# ===============================================================================
# Data Fetching
# ===============================================================================

def load_config() -> dict:
    cfg_path = _SCRIPT_DIR / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def connect_mt5(cfg: dict) -> bool:
    mt5_cfg = cfg.get("mt5", {})
    acct = cfg.get("account", {})
    kwargs = {}
    if mt5_cfg.get("terminal_path"):
        kwargs["path"] = mt5_cfg["terminal_path"]
    if acct.get("login"):
        kwargs["login"] = acct["login"]
    if acct.get("password"):
        kwargs["password"] = acct["password"]
    if acct.get("server"):
        kwargs["server"] = acct["server"]
    if not mt5.initialize(**kwargs):
        print(f"MT5 init failed: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    print(f"MT5 connected | {info.login} @ {info.server} | ${info.balance:.2f}")
    return True


def fetch_historical(symbol: str, months: int) -> tuple:
    """Fetch M15, H1, H4 candles for the backtest period."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=months * 30)

    print(f"Fetching {symbol} data: {start.date()} to {now.date()} ({months} months)")

    def _fetch(tf, tf_name):
        rates = mt5.copy_rates_range(symbol, tf, start, now)
        if rates is None or len(rates) == 0:
            print(f"  {tf_name}: FAILED ({mt5.last_error()})")
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        df = df[["time", "open", "high", "low", "close", "volume"]]
        print(f"  {tf_name}: {len(df)} bars ({df['time'].iloc[0].date()} -> {df['time'].iloc[-1].date()})")
        return df

    df_m15 = _fetch(mt5.TIMEFRAME_M15, "M15")
    df_h1 = _fetch(mt5.TIMEFRAME_H1, "H1")
    df_h4 = _fetch(mt5.TIMEFRAME_H4, "H4")

    return df_m15, df_h1, df_h4


# ===============================================================================
# Position Manager (simulation)
# ===============================================================================

class PositionManager:
    """Simulates position lifecycle: SL/TP/BE/trail/scratch/stale."""

    def __init__(self, spread: float = 3.0):
        self.spread = spread
        self._next_ticket = 1

    def open_position(self, bar, direction, sl, tp, lot, zone_type, signals,
                      signal_count, regime, regime_cat, session, pre_score, atr,
                      rsi_val=50.0, ema_trend="NEUTRAL", pd_zone="EQUILIBRIUM") -> SimPosition:
        half_spread = self.spread / 2
        if direction == "LONG":
            entry = bar["close"] + half_spread
        else:
            entry = bar["close"] - half_spread

        pos = SimPosition(
            ticket=self._next_ticket,
            direction=direction,
            entry_price=entry,
            entry_time=bar["time"],
            sl=sl, tp=tp,
            sl_original=sl,
            lot=lot,
            zone_type=zone_type,
            signals=list(signals),
            signal_count=signal_count,
            regime=regime,
            regime_cat=regime_cat,
            session=session,
            pre_score=pre_score,
            atr=atr,
            rsi_val=rsi_val,
            ema_trend=ema_trend,
            pd_zone=pd_zone,
        )
        self._next_ticket += 1
        return pos

    def update(self, pos: SimPosition, bar, exit_params: dict) -> bool:
        """
        Update position with current bar. Returns True if position was closed.
        Checks: SL hit, TP hit, BE trigger, profit lock, trailing, scratch, stale.
        """
        if pos.exit_time is not None:
            return True  # already closed

        high = bar["high"]
        low = bar["low"]
        close = bar["close"]
        bar_time = bar["time"]

        half_spread = self.spread / 2
        # Simulated bid/ask from bar
        bid = close - half_spread
        ask = close + half_spread

        sl_dist = abs(pos.entry_price - pos.sl_original)
        if sl_dist <= 0:
            sl_dist = pos.atr * 2.0

        # Resolve exit params
        be_mult = exit_params.get("be_trigger_mult", 0.7)
        lock_mult = exit_params.get("lock_trigger_mult", 1.5)
        trail_pct = exit_params.get("trail_keep_pct", 0.50)
        stale_min = exit_params.get("stale_tighten_min", 90)
        scratch_min = exit_params.get("scratch_exit_min", 180)

        # -- Check SL hit -----------------------------------------------------
        if pos.direction == "LONG":
            if low <= pos.sl:
                return self._close(pos, pos.sl - half_spread, bar_time, "SL")
        else:
            if high >= pos.sl:
                return self._close(pos, pos.sl + half_spread, bar_time, "SL")

        # -- Check TP hit -----------------------------------------------------
        if pos.direction == "LONG":
            if high >= pos.tp:
                return self._close(pos, pos.tp - half_spread, bar_time, "TP")
        else:
            if low <= pos.tp:
                return self._close(pos, pos.tp + half_spread, bar_time, "TP")

        # -- Current profit (using close, not extreme) ------------------------
        if pos.direction == "LONG":
            profit_pts = bid - pos.entry_price
        else:
            profit_pts = pos.entry_price - ask

        # Track peak profit
        if profit_pts > pos.peak_profit:
            pos.peak_profit = profit_pts

        # -- Position age -----------------------------------------------------
        age_min = (bar_time - pos.entry_time).total_seconds() / 60

        # -- Scratch exit (flat after scratch_min) ----------------------------
        if age_min >= scratch_min and abs(profit_pts) < 10.0:
            if self._sl_below_entry(pos):
                exit_px = bid if pos.direction == "LONG" else ask
                return self._close(pos, exit_px, bar_time, "SCRATCH")

        # -- Stale tighten (low progress after stale_min, fires ONCE) --------
        if (age_min >= stale_min and not pos.stale_tightened
                and profit_pts < sl_dist * 0.5 and self._sl_below_entry(pos)):
            half_dist = sl_dist * 0.5
            if pos.direction == "LONG":
                new_sl = pos.entry_price - half_dist
                if new_sl > pos.sl + 0.5:
                    pos.sl = round(new_sl, 2)
            else:
                new_sl = pos.entry_price + half_dist
                if new_sl < pos.sl - 0.5:
                    pos.sl = round(new_sl, 2)
            pos.stale_tightened = True

        # -- BE trigger -------------------------------------------------------
        be_trigger = sl_dist * be_mult
        if profit_pts >= be_trigger and not pos.be_triggered and self._sl_below_entry(pos):
            if pos.direction == "LONG":
                pos.sl = round(pos.entry_price + 0.2, 2)
            else:
                pos.sl = round(pos.entry_price - 0.2, 2)
            pos.be_triggered = True

        # -- Profit lock at lock_mult x SL ------------------------------------
        lock_trigger = sl_dist * lock_mult
        if profit_pts >= lock_trigger and not pos.lock_triggered:
            # Move SL to entry + 50% of profit
            if pos.direction == "LONG":
                lock_sl = round(pos.entry_price + profit_pts * 0.5, 2)
                if lock_sl > pos.sl:
                    pos.sl = lock_sl
            else:
                lock_sl = round(pos.entry_price - profit_pts * 0.5, 2)
                if lock_sl < pos.sl:
                    pos.sl = lock_sl
            pos.lock_triggered = True

        # -- Trailing stop (keep trail_pct of profit as cushion) --------------
        trail_sl_dist = profit_pts * trail_pct
        trail_activate = sl_dist * 0.40
        if trail_sl_dist >= trail_activate and profit_pts > 0:
            if pos.direction == "LONG":
                new_sl = round(bid - trail_sl_dist, 2)
                if new_sl > pos.sl + 0.5:
                    pos.sl = new_sl
            else:
                new_sl = round(ask + trail_sl_dist, 2)
                if new_sl < pos.sl - 0.5:
                    pos.sl = new_sl

        return False  # still open

    def _close(self, pos: SimPosition, price: float, time, close_type: str) -> bool:
        pos.exit_price = price
        pos.exit_time = time
        pos.close_type = close_type
        if pos.direction == "LONG":
            pos.pnl_pts = price - pos.entry_price
        else:
            pos.pnl_pts = pos.entry_price - price
        return True

    def _sl_below_entry(self, pos: SimPosition) -> bool:
        if pos.direction == "LONG":
            return pos.sl < pos.entry_price
        return pos.sl > pos.entry_price


# ===============================================================================
# Entry Evaluation (replicates main.py gates, Claude -> algo_pre_score)
# ===============================================================================

def evaluate_entry(
    m15_bar, price, h1_slice, h4_slice, m15_window,
    regime_result, adaptive: ae.AdaptiveEngine, cfg: dict,
    spread: float, pre_score_min: float,
    gate_config: Optional[dict] = None,
) -> Optional[dict]:
    """
    Evaluate entry for both directions. Returns dict with entry info or None.
    Replicates the gate pipeline from main.py (100% sync, minus Claude).

    gate_config keys (all optional, defaults match live system):
      min_signals     (int)  : min Tier-1/2 signals for gate   [default: 3]
      pd_zone_gate    (bool) : enforce P/D directional gate    [default: True]
      require_structure (bool): must have BOS or CHoCH         [default: False]
      require_choch   (bool) : must have CHoCH                 [default: False]
      require_bos     (bool) : must have BOS                   [default: False]
      tier1_min       (int)  : min Tier-1 signals              [default: 0]
    """
    gc = gate_config or {}
    min_sig = gc.get("min_signals", 1)              # synced: main.py hybrid (was 3)
    pd_gate = gc.get("pd_zone_gate", False)          # synced: Claude decides (was True)
    req_structure = gc.get("require_structure", False) # synced: Claude decides (was True)
    req_choch = gc.get("require_choch", False)
    req_bos = gc.get("require_bos", False)
    tier1_min = gc.get("tier1_min", 0)

    t_cfg = cfg.get("trading", {})
    proximity = t_cfg.get("zone_proximity_pts", 5.0)
    sl_mult = t_cfg.get("sl_atr_mult", 2.0)
    tp_mult = t_cfg.get("tp_atr_mult", 4.0)

    bar_time = m15_bar["time"]
    now_utc = bar_time if isinstance(bar_time, datetime) else pd.Timestamp(bar_time).to_pydatetime()

    # Indicators from H1
    atr_val = ind.atr(h1_slice, 14) if len(h1_slice) >= 15 else 0.0
    rsi_val = ind.rsi(h1_slice, 14) if len(h1_slice) >= 15 else 50.0
    pd_zone = ind.premium_discount(h1_slice) if len(h1_slice) >= 10 else "EQUILIBRIUM"
    ema_trend = ind.h1_ema_trend(h1_slice) if len(h1_slice) >= 55 else "NEUTRAL"
    h4_b = ind.h4_bias(h4_slice) if len(h4_slice) >= 6 else "RANGING"
    session = scan.current_session(now_utc)

    regime = regime_result["regime"]
    regime_cat = regime.category
    regime_label = regime_result["short_label"]
    has_choch = regime_result["components"].get("has_choch", False)

    # Session guard
    if session["name"] == "OFF_HOURS":
        return None

    # Spike window
    if scan.is_spike_window(now_utc):
        return None

    # Daily range consumed
    if ind.daily_range_consumed(h1_slice, 1.20):
        return None

    # Zone detection
    zones = zdet.detect_all_zones(h1_slice)
    if not zones:
        return None

    nearby = scan.find_nearby_zones(price, zones, proximity)
    if not nearby:
        return None

    # Collect nearby zones by direction (synced with main.py line 553-562)
    long_zones = []
    short_zones = []
    for z in nearby:
        d = scan.direction_for_zone(z)
        if d == "LONG":
            long_zones.append(z)
        elif d == "SHORT":
            short_zones.append(z)

    best = None

    for direction, dir_zones in [("LONG", long_zones), ("SHORT", short_zones)]:
        if not dir_zones:
            continue

        primary = dir_zones[0]

        # Use NEARBY zone types only — synced with main.py line 576 (Bug E fix)
        nearby_zone_types = [z["type"] for z in dir_zones]

        m15_conf = ind.m15_confirmation(m15_window, direction) if len(m15_window) >= 6 else None
        ote = ind.ote_zone(h1_slice, direction)

        signal_count, signals = ind.count_signals(
            direction, nearby_zone_types, m15_conf, ote, price, pd_zone
        )

        # ── Signal gate (configurable) ────────────────────────────────
        if signal_count < min_sig:
            continue

        # Data capture (synced with main.py hybrid — no hard gates here)
        zone_has_choch = any("CHOCH" in s.upper() for s in signals)
        choch_detected = has_choch or zone_has_choch

        # Signal composition gates (only when optimizer sets them explicitly)
        if req_structure and not any(s in ("BOS", "CHoCH") for s in signals):
            continue
        if req_choch and "CHoCH" not in signals:
            continue
        if req_bos and "BOS" not in signals:
            continue
        if tier1_min > 0:
            t1_count = sum(1 for s in signals if s in ("BOS", "OB", "LiqSweep"))
            if t1_count < tier1_min:
                continue

        # P/D zone gate (only when optimizer sets pd_gate=True)
        if pd_gate and not choch_detected:
            if direction == "LONG" and pd_zone == "PREMIUM":
                continue
            if direction == "SHORT" and pd_zone == "DISCOUNT":
                continue

        # Algo pre-score (logging + soft filter for backtest — replaces Claude)
        if adaptive:
            pre_score, should_call = adaptive.algo_pre_score(
                signal_count, regime_cat, session["name"], direction,
                ema_trend, rsi_val, pd_zone, choch_detected, signals,
            )
            if pre_score < pre_score_min:
                continue
        else:
            pre_score = 0.50

        # Build SL/TP
        if atr_val <= 0:
            atr_val = 15.0

        entry_params = adaptive.get_entry_params(regime_cat) if adaptive else {}
        sl_mult_eff = entry_params.get("sl_atr_mult", sl_mult)
        tp_mult_eff = entry_params.get("tp_atr_mult", tp_mult)
        sl_dist = atr_val * sl_mult_eff
        tp_dist = atr_val * tp_mult_eff

        if direction == "LONG":
            entry_px = price + spread / 2
            sl = round(entry_px - sl_dist, 2)
            tp = round(entry_px + tp_dist, 2)
        else:
            entry_px = price - spread / 2
            sl = round(entry_px + sl_dist, 2)
            tp = round(entry_px - tp_dist, 2)

        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        # Pick best direction by pre-score
        if best is None or pre_score > best["pre_score"]:
            best = {
                "direction": direction,
                "sl": sl, "tp": tp,
                "lot": 0.01,
                "zone_type": primary.get("type", ""),
                "signals": signals,
                "signal_count": signal_count,
                "regime": regime_label,
                "regime_cat": regime_cat,
                "session": session["name"],
                "pre_score": pre_score,
                "atr": atr_val,
                "rr": rr,
                "rsi_val": rsi_val,
                "ema_trend": ema_trend,
                "pd_zone": pd_zone,
            }

    return best


# ===============================================================================
# Main Simulation Loop
# ===============================================================================

def run_simulation(
    df_m15: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_h4: pd.DataFrame,
    cfg: dict,
    use_adaptive: bool = True,
    spread: float = 3.0,
    balance: float = 100.0,
    pre_score_min: float = 0.35,
    verbose: bool = False,
    gate_config: Optional[dict] = None,
) -> list[SimPosition]:
    """Run bar-by-bar M15 simulation. Returns list of closed trades."""

    # Initialize modules
    regime_det = rdet.RegimeDetector(lookback=50)
    adaptive_cfg = cfg.get("adaptive", {})

    if use_adaptive:
        adaptive = ae.AdaptiveEngine(
            config_baselines=adaptive_cfg.get("regime_params"),
            bounds={k: tuple(v) for k, v in adaptive_cfg.get("bounds", {}).items()}
            if adaptive_cfg.get("bounds") else None,
            state_path="__backtest_no_save__.json",  # don't touch live state
            min_trades=999999,  # disable live adaptation during backtest
        )
    else:
        adaptive = None

    pm = PositionManager(spread=spread)
    open_pos: Optional[SimPosition] = None
    closed_trades: list[SimPosition] = []

    # Precompute H1/H4 time arrays for fast slicing (strip tz for numpy comparison)
    h1_times = df_h1["time"].values.astype("datetime64[ns]")
    h4_times = df_h4["time"].values.astype("datetime64[ns]")

    total_bars = len(df_m15)
    # Need at least 55 H1 bars of warmup
    start_idx = max(12, 1)  # start after at least 12 M15 bars

    last_h1_bar = None  # track H1 changes for zone-attempt reset
    attempted_zones: set = set()
    last_regime_result = rdet.RegimeDetector._default_result()

    progress_step = max(1, total_bars // 20)

    for i in range(start_idx, total_bars):
        bar = df_m15.iloc[i]
        bar_time = bar["time"]
        price = bar["close"]

        # Progress indicator
        if i % progress_step == 0:
            pct = i / total_bars * 100
            print(f"  Simulating... {pct:.0f}% ({i}/{total_bars}) | "
                  f"Trades: {len(closed_trades)} closed" +
                  (f" | 1 open" if open_pos else ""), end="\r")

        # -- Slice H1/H4 up to current time --------------------------------
        bt = np.datetime64(bar_time, "ns")
        h1_end = int((h1_times <= bt).sum())
        h1_slice = df_h1.iloc[max(0, h1_end - 55):h1_end]

        h4_end = int((h4_times <= bt).sum())
        h4_slice = df_h4.iloc[max(0, h4_end - 25):h4_end]

        m15_window = df_m15.iloc[max(0, i - 11):i + 1]

        if len(h1_slice) < 20:
            continue  # not enough data yet

        # -- Reset attempted zones each H1 bar -----------------------------
        current_h1 = h1_slice.iloc[-1]["time"] if len(h1_slice) > 0 else None
        if current_h1 != last_h1_bar:
            last_h1_bar = current_h1
            attempted_zones.clear()

        # -- Detect regime (every 4 M15 bars = 1 H1 bar, or first time) ----
        if i % 4 == 0 or i == start_idx:
            if len(h1_slice) >= 50:
                last_regime_result = regime_det.detect(h1_slice)

        # -- Manage open position ------------------------------------------
        if open_pos is not None:
            exit_params = adaptive.get_exit_params(open_pos.regime_cat) if adaptive else {}
            closed = pm.update(open_pos, bar, exit_params)
            if closed:
                closed_trades.append(open_pos)
                if verbose:
                    dur = (open_pos.exit_time - open_pos.entry_time).total_seconds() / 60
                    print(
                        f"  #{open_pos.ticket:>4d} {open_pos.direction:5s} | "
                        f"{open_pos.entry_price:.2f} -> {open_pos.exit_price:.2f} | "
                        f"{open_pos.pnl_pts:+6.1f}pt | {open_pos.close_type:7s} | "
                        f"{dur:.0f}min | {open_pos.regime} {open_pos.session} | "
                        f"signals={open_pos.signal_count} pre={open_pos.pre_score:.2f}"
                    )
                open_pos = None

        # -- Evaluate new entry (only if no open position) -----------------
        if open_pos is None:
            entry = evaluate_entry(
                m15_bar=bar,
                price=price,
                h1_slice=h1_slice,
                h4_slice=h4_slice,
                m15_window=m15_window,
                regime_result=last_regime_result,
                adaptive=adaptive,
                cfg=cfg,
                spread=spread,
                pre_score_min=pre_score_min,
                gate_config=gate_config,
            )

            if entry is not None:
                zone_key = f"{entry['direction']}_{entry['zone_type']}"
                if zone_key not in attempted_zones:
                    attempted_zones.add(zone_key)
                    open_pos = pm.open_position(
                        bar=bar,
                        direction=entry["direction"],
                        sl=entry["sl"],
                        tp=entry["tp"],
                        lot=entry["lot"],
                        zone_type=entry["zone_type"],
                        signals=entry["signals"],
                        signal_count=entry["signal_count"],
                        regime=entry["regime"],
                        regime_cat=entry["regime_cat"],
                        session=entry["session"],
                        pre_score=entry["pre_score"],
                        atr=entry["atr"],
                        rsi_val=entry.get("rsi_val", 50.0),
                        ema_trend=entry.get("ema_trend", "NEUTRAL"),
                        pd_zone=entry.get("pd_zone", "EQUILIBRIUM"),
                    )
                    if verbose:
                        print(
                            f"  #{open_pos.ticket:>4d} OPEN {open_pos.direction:5s} @ "
                            f"{open_pos.entry_price:.2f} | SL={open_pos.sl:.2f} TP={open_pos.tp:.2f} | "
                            f"RR={entry['rr']:.1f} | {open_pos.regime} {open_pos.session} | "
                            f"[{', '.join(open_pos.signals)}] pre={open_pos.pre_score:.2f}"
                        )

    # Close any remaining open position at last bar
    if open_pos is not None:
        last_bar = df_m15.iloc[-1]
        half_sp = spread / 2
        if open_pos.direction == "LONG":
            exit_px = last_bar["close"] - half_sp
        else:
            exit_px = last_bar["close"] + half_sp
        open_pos.exit_price = exit_px
        open_pos.exit_time = last_bar["time"]
        open_pos.close_type = "END"
        if open_pos.direction == "LONG":
            open_pos.pnl_pts = exit_px - open_pos.entry_price
        else:
            open_pos.pnl_pts = open_pos.entry_price - exit_px
        closed_trades.append(open_pos)

    print(f"  Simulation complete: {len(closed_trades)} trades" + " " * 40)
    return closed_trades


# ===============================================================================
# Results Computation
# ===============================================================================

def compute_results(trades: list[SimPosition], balance: float = 100.0) -> dict:
    """Compute summary stats, per-regime and per-session breakdowns."""
    if not trades:
        return {"total": 0, "summary": {}, "by_regime": {}, "by_session": {},
                "by_close_type": {}, "equity_curve": []}

    total = len(trades)
    wins = [t for t in trades if t.pnl_pts > 0]
    losses = [t for t in trades if t.pnl_pts <= 0]
    win_count = len(wins)
    loss_count = len(losses)

    gross_profit = sum(t.pnl_pts for t in wins)
    gross_loss = abs(sum(t.pnl_pts for t in losses))
    net_pnl_pts = sum(t.pnl_pts for t in trades)

    # PnL in USD (0.01 lot XAUUSD, contract_size=100)
    pnl_per_pt = 0.01 * 100  # $1 per point per 0.01 lot
    net_pnl_usd = net_pnl_pts * pnl_per_pt

    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    wr = win_count / total * 100

    avg_win = gross_profit / win_count if win_count > 0 else 0
    avg_loss = gross_loss / loss_count if loss_count > 0 else 0
    avg_rr = avg_win / avg_loss if avg_loss > 0 else 0

    # Max drawdown
    equity_curve = []
    equity = balance
    peak_equity = balance
    max_dd = 0
    max_dd_pct = 0

    for t in sorted(trades, key=lambda x: x.exit_time):
        pnl_usd = t.pnl_pts * pnl_per_pt
        equity += pnl_usd
        peak_equity = max(peak_equity, equity)
        dd = peak_equity - equity
        dd_pct = dd / peak_equity * 100 if peak_equity > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd = dd
        equity_curve.append({
            "time": t.exit_time.isoformat() if isinstance(t.exit_time, datetime) else str(t.exit_time),
            "equity": round(equity, 2),
            "drawdown_pct": round(dd_pct, 2),
            "pnl_pts": round(t.pnl_pts, 1),
            "close_type": t.close_type,
        })

    # Sharpe-like ratio (daily PnL / std)
    daily_pnl = {}
    for t in trades:
        day = str(t.exit_time.date()) if isinstance(t.exit_time, datetime) else str(t.exit_time)[:10]
        daily_pnl.setdefault(day, 0)
        daily_pnl[day] += t.pnl_pts * pnl_per_pt
    daily_vals = list(daily_pnl.values())
    if len(daily_vals) > 1:
        avg_daily = sum(daily_vals) / len(daily_vals)
        std_daily = (sum((v - avg_daily) ** 2 for v in daily_vals) / len(daily_vals)) ** 0.5
        sharpe = avg_daily / std_daily if std_daily > 0 else 0
    else:
        sharpe = 0

    summary = {
        "total": total,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 2),
        "net_pnl_pts": round(net_pnl_pts, 1),
        "net_pnl_usd": round(net_pnl_usd, 2),
        "return_pct": round(net_pnl_usd / balance * 100, 1),
        "max_dd_usd": round(max_dd, 2),
        "max_dd_pct": round(max_dd_pct, 1),
        "avg_win_pts": round(avg_win, 1),
        "avg_loss_pts": round(avg_loss, 1),
        "avg_rr": round(avg_rr, 2),
        "sharpe": round(sharpe, 2),
        "final_equity": round(equity, 2),
    }

    # Per-regime breakdown
    by_regime = _breakdown(trades, lambda t: t.regime)
    by_session = _breakdown(trades, lambda t: t.session)
    by_close_type = _breakdown(trades, lambda t: t.close_type)

    return {
        "total": total,
        "summary": summary,
        "by_regime": by_regime,
        "by_session": by_session,
        "by_close_type": by_close_type,
        "equity_curve": equity_curve,
    }


def _breakdown(trades: list[SimPosition], key_fn) -> dict:
    """Group trades by key and compute stats per group."""
    groups = {}
    for t in trades:
        k = key_fn(t)
        groups.setdefault(k, []).append(t)

    result = {}
    for k, group in sorted(groups.items()):
        total = len(group)
        wins = sum(1 for t in group if t.pnl_pts > 0)
        gross_p = sum(t.pnl_pts for t in group if t.pnl_pts > 0)
        gross_l = abs(sum(t.pnl_pts for t in group if t.pnl_pts <= 0))
        pf = gross_p / gross_l if gross_l > 0 else float("inf")
        avg = sum(t.pnl_pts for t in group) / total
        result[k] = {
            "trades": total,
            "wr": round(wins / total * 100, 0),
            "pf": round(pf, 2) if pf < 100 else 99.99,
            "avg_pnl": round(avg, 1),
            "net_pts": round(sum(t.pnl_pts for t in group), 1),
        }
    return result


# ===============================================================================
# Output Formatting
# ===============================================================================

def print_results(results: dict, label: str = "ADAPTIVE"):
    s = results["summary"]
    if not s:
        print(f"\n{'=' * 60}")
        print(f"  {label}: No trades generated")
        print(f"{'=' * 60}")
        return

    print(f"\n{'=' * 70}")
    print(f"  BACKTEST RESULTS -- {label}")
    print(f"{'=' * 70}")
    print(f"  Total trades:  {s['total']:>6d}   |  Win rate:     {s['win_rate']:>5.1f}%")
    print(f"  Winners:       {s['wins']:>6d}   |  Losers:       {s['losses']:>6d}")
    print(f"  Profit factor: {s['profit_factor']:>6.2f}   |  Avg RR:       {s['avg_rr']:>5.2f}")
    print(f"  Net PnL:    {s['net_pnl_pts']:>+7.1f}pt  |  ${s['net_pnl_usd']:>+8.2f} ({s['return_pct']:>+.1f}%)")
    print(f"  Max DD:      ${s['max_dd_usd']:>7.2f}   |  {s['max_dd_pct']:>5.1f}%")
    print(f"  Avg winner:  {s['avg_win_pts']:>+6.1f}pt  |  Avg loser:   {s['avg_loss_pts']:>6.1f}pt")
    print(f"  Sharpe-like:   {s['sharpe']:>6.2f}   |  Final equity: ${s['final_equity']:>.2f}")
    print(f"{'-' * 70}")

    # Per-regime
    print(f"\n  {'Regime':<12s} | {'Trades':>6s} | {'WR':>5s} | {'PF':>6s} | {'AvgPnL':>7s} | {'NetPts':>7s}")
    print(f"  {'-' * 12}-+-{'-' * 6}-+-{'-' * 5}-+-{'-' * 6}-+-{'-' * 7}-+-{'-' * 7}")
    for k, v in results["by_regime"].items():
        print(f"  {k:<12s} | {v['trades']:>6d} | {v['wr']:>4.0f}% | {v['pf']:>6.2f} | {v['avg_pnl']:>+6.1f}pt | {v['net_pts']:>+6.1f}pt")

    # Per-session
    print(f"\n  {'Session':<12s} | {'Trades':>6s} | {'WR':>5s} | {'PF':>6s} | {'AvgPnL':>7s} | {'NetPts':>7s}")
    print(f"  {'-' * 12}-+-{'-' * 6}-+-{'-' * 5}-+-{'-' * 6}-+-{'-' * 7}-+-{'-' * 7}")
    for k, v in results["by_session"].items():
        print(f"  {k:<12s} | {v['trades']:>6d} | {v['wr']:>4.0f}% | {v['pf']:>6.2f} | {v['avg_pnl']:>+6.1f}pt | {v['net_pts']:>+6.1f}pt")

    # Per-close type
    print(f"\n  {'CloseType':<12s} | {'Trades':>6s} | {'WR':>5s} | {'PF':>6s} | {'AvgPnL':>7s} | {'NetPts':>7s}")
    print(f"  {'-' * 12}-+-{'-' * 6}-+-{'-' * 5}-+-{'-' * 6}-+-{'-' * 7}-+-{'-' * 7}")
    for k, v in results["by_close_type"].items():
        print(f"  {k:<12s} | {v['trades']:>6d} | {v['wr']:>4.0f}% | {v['pf']:>6.2f} | {v['avg_pnl']:>+6.1f}pt | {v['net_pts']:>+6.1f}pt")

    print()


def write_equity_csv(equity_curve: list[dict], path: str):
    if not equity_curve:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time", "equity", "drawdown_pct", "pnl_pts", "close_type"])
        w.writeheader()
        w.writerows(equity_curve)
    print(f"  Equity curve saved: {path} ({len(equity_curve)} rows)")


def write_trades_csv(trades: list[SimPosition], path: str):
    if not trades:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ticket", "direction", "entry_price", "entry_time", "exit_price", "exit_time",
        "sl_original", "tp", "pnl_pts", "close_type", "lot", "zone_type", "signal_count",
        "signals", "regime", "regime_cat", "session", "pre_score", "atr",
        "be_triggered", "lock_triggered", "stale_tightened", "peak_profit",
        "rsi_val", "ema_trend", "pd_zone",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in sorted(trades, key=lambda x: x.entry_time):
            row = {
                "ticket": t.ticket,
                "direction": t.direction,
                "entry_price": round(t.entry_price, 2),
                "entry_time": t.entry_time.isoformat() if isinstance(t.entry_time, datetime) else str(t.entry_time),
                "exit_price": round(t.exit_price, 2),
                "exit_time": t.exit_time.isoformat() if isinstance(t.exit_time, datetime) else str(t.exit_time),
                "sl_original": round(t.sl_original, 2),
                "tp": round(t.tp, 2),
                "pnl_pts": round(t.pnl_pts, 1),
                "close_type": t.close_type,
                "lot": t.lot,
                "zone_type": t.zone_type,
                "signal_count": t.signal_count,
                "signals": "|".join(t.signals),
                "regime": t.regime,
                "regime_cat": t.regime_cat,
                "session": t.session,
                "pre_score": round(t.pre_score, 3),
                "atr": round(t.atr, 1),
                "be_triggered": t.be_triggered,
                "lock_triggered": t.lock_triggered,
                "stale_tightened": t.stale_tightened,
                "peak_profit": round(t.peak_profit, 1),
                "rsi_val": round(t.rsi_val, 1),
                "ema_trend": t.ema_trend,
                "pd_zone": t.pd_zone,
            }
            w.writerow(row)
    print(f"  Trades log saved: {path} ({len(trades)} trades)")


# ===============================================================================
# Compare Mode
# ===============================================================================

def print_comparison(r_adaptive: dict, r_static: dict):
    sa = r_adaptive.get("summary", {})
    ss = r_static.get("summary", {})
    if not sa or not ss:
        print("Cannot compare -- one or both runs had no trades")
        return

    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: ADAPTIVE vs STATIC")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<20s} | {'Adaptive':>12s} | {'Static':>12s} | {'Delta':>10s}")
    print(f"  {'-' * 20}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}")

    metrics = [
        ("Total trades", "total", "d", 0),
        ("Win rate %", "win_rate", ".1f", 1),
        ("Profit factor", "profit_factor", ".2f", 1),
        ("Net PnL pts", "net_pnl_pts", "+.1f", 1),
        ("Net PnL $", "net_pnl_usd", "+.2f", 1),
        ("Return %", "return_pct", "+.1f", 1),
        ("Max DD %", "max_dd_pct", ".1f", -1),  # lower is better
        ("Avg RR", "avg_rr", ".2f", 1),
        ("Sharpe", "sharpe", ".2f", 1),
    ]

    for label, key, fmt, better_sign in metrics:
        va = sa.get(key, 0)
        vs = ss.get(key, 0)
        delta = va - vs
        arrow = ""
        if better_sign > 0:
            arrow = " +" if delta > 0 else " -" if delta < 0 else "  "
        elif better_sign < 0:
            arrow = " +" if delta < 0 else " -" if delta > 0 else "  "
        va_s = f"{va:{fmt}}"
        vs_s = f"{vs:{fmt}}"
        d_str = f"{delta:{fmt}}"
        print(f"  {label:<20s} | {va_s:>12s} | {vs_s:>12s} |{arrow}{d_str:>8s}")

    print()


# ===============================================================================
# Main
# ===============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Smart Trader --6-Month Backtest (algo pre-score, no Claude)"
    )
    parser.add_argument("--months", type=int, default=6, help="Backtest period in months (default: 6)")
    parser.add_argument("--balance", type=float, default=100.0, help="Starting balance USD (default: 100)")
    parser.add_argument("--spread", type=float, default=3.0, help="Fixed spread in points (default: 3.0)")
    parser.add_argument("--pre-score-min", type=float, default=0.35, help="Min algo pre-score (default: 0.35)")
    parser.add_argument("--no-adaptive", action="store_true", help="Use static config params (no adaptive)")
    parser.add_argument("--compare", action="store_true", help="Run both adaptive and static, show comparison")
    parser.add_argument("--verbose", action="store_true", help="Print each trade as it happens")
    parser.add_argument("--output", default="logs/backtest_equity.csv", help="Equity curve CSV path")
    parser.add_argument("--symbol", default=None, help="Override symbol (default: from config)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Smart Trader --Backtest Engine")
    print(f"  Months: {args.months} | Balance: ${args.balance} | Spread: {args.spread}pt")
    print(f"  Pre-score min: {args.pre_score_min} | Adaptive: {not args.no_adaptive}")
    print("=" * 70)

    # Load config
    cfg = load_config()
    symbol = args.symbol or cfg.get("mt5", {}).get("symbol", "XAUUSD")

    # Connect to MT5
    if not connect_mt5(cfg):
        sys.exit(1)

    try:
        # Fetch data
        df_m15, df_h1, df_h4 = fetch_historical(symbol, args.months)
        if df_m15.empty or df_h1.empty:
            print("ERROR: Failed to fetch required data")
            sys.exit(1)

        period_str = f"{df_m15['time'].iloc[0].date()} -> {df_m15['time'].iloc[-1].date()}"
        print(f"\n  Period: {period_str}")
        print(f"  Bars: M15={len(df_m15)}, H1={len(df_h1)}, H4={len(df_h4)}")

        if args.compare:
            # -- Run adaptive -------------------------------------------------
            print(f"\n{'-' * 70}")
            print("  Running ADAPTIVE simulation...")
            trades_a = run_simulation(
                df_m15, df_h1, df_h4, cfg,
                use_adaptive=True, spread=args.spread,
                balance=args.balance, pre_score_min=args.pre_score_min,
                verbose=args.verbose,
            )
            results_a = compute_results(trades_a, args.balance)

            # -- Run static ---------------------------------------------------
            print(f"\n{'-' * 70}")
            print("  Running STATIC simulation...")
            trades_s = run_simulation(
                df_m15, df_h1, df_h4, cfg,
                use_adaptive=False, spread=args.spread,
                balance=args.balance, pre_score_min=args.pre_score_min,
                verbose=args.verbose,
            )
            results_s = compute_results(trades_s, args.balance)

            # -- Print both ---------------------------------------------------
            print_results(results_a, f"ADAPTIVE | {period_str}")
            print_results(results_s, f"STATIC   | {period_str}")
            print_comparison(results_a, results_s)

            # -- Save CSVs ----------------------------------------------------
            base = Path(args.output).stem
            parent = Path(args.output).parent
            write_equity_csv(results_a["equity_curve"], str(parent / f"{base}_adaptive.csv"))
            write_equity_csv(results_s["equity_curve"], str(parent / f"{base}_static.csv"))
            write_trades_csv(trades_a, str(parent / f"backtest_trades_adaptive.csv"))
            write_trades_csv(trades_s, str(parent / f"backtest_trades_static.csv"))

        else:
            # -- Single run ---------------------------------------------------
            use_adaptive = not args.no_adaptive
            label = "ADAPTIVE" if use_adaptive else "STATIC"
            print(f"\n  Running {label} simulation...")
            trades = run_simulation(
                df_m15, df_h1, df_h4, cfg,
                use_adaptive=use_adaptive, spread=args.spread,
                balance=args.balance, pre_score_min=args.pre_score_min,
                verbose=args.verbose,
            )
            results = compute_results(trades, args.balance)
            print_results(results, f"{label} | {period_str}")
            write_equity_csv(results["equity_curve"], args.output)
            write_trades_csv(trades, str(Path(args.output).parent / "backtest_trades.csv"))

    finally:
        mt5.shutdown()
        print("MT5 disconnected.")


if __name__ == "__main__":
    main()
