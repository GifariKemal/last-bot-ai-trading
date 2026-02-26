"""
Trading Bot
Main bot controller that orchestrates all components.
"""

import time
import csv
import os
from typing import Dict, Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .health_monitor import HealthMonitor
from ..core.mt5_connector import MT5Connector
from ..core.data_manager import DataManager
from ..core.timeframe_manager import TimeframeManager
from ..indicators.technical import TechnicalIndicators
from ..indicators.smc_indicators import SMCIndicators
from ..analysis import (
    MarketAnalyzer,
    VolatilityAnalyzer,
    TrendAnalyzer,
    MTFAnalyzer,
    ConfluenceScorer,
    AdaptiveConfluenceScorer,
    RegimeDetector,
    TradeAnalyzer,
)
from ..core.constants import MarketRegime
from ..strategy.smc_strategy import SMCStrategy
from ..strategy.entry_quality import EntryQualityEngine
from ..risk_management import (
    SLTPCalculator,
    StructureSLTPCalculator,
    PositionSizer,
    DrawdownMonitor,
    MicroAccountManager,
)
from ..position_management import (
    PositionTracker,
    RecoveryManager,
)
from ..sessions import SessionManager
from ..execution import OrderExecutor, EmergencyHandler
from ..notifications import TelegramNotifier
from ..analysis.trade_journal import TradeJournal
from ..bot_logger import get_logger


class TradingBot:
    """Main trading bot controller."""

    def __init__(self, config: Dict):
        """
        Initialize trading bot.

        Args:
            config: Complete bot configuration
        """
        self.logger = get_logger()
        self.config = config
        self.running = False

        # Core components
        self.logger.info("Initializing trading bot...")

        # MT5 Connection
        self.mt5 = MT5Connector(config.get("mt5", {}))

        # Data management
        self.data_manager = DataManager()
        self.timeframe_manager = TimeframeManager(self.mt5, self.data_manager)

        # Indicators
        self.technical_indicators = TechnicalIndicators(config.get("indicators", {}))
        if config.get("use_smc_v4", False):
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):  # suppress library print
                from ..indicators.smc_v4_adapter import SMCIndicatorsV4
                self.smc_indicators = SMCIndicatorsV4(config.get("smc_indicators", {}))
            self.logger.info("V4 library-based SMC detection ENABLED")
        else:
            self.smc_indicators = SMCIndicators(config.get("smc_indicators", {}))

        # Analysis
        self.market_analyzer = MarketAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.mtf_analyzer = MTFAnalyzer()
        self.confluence_scorer = ConfluenceScorer(config)

        # Strategy
        self.strategy = SMCStrategy(config)

        # Risk Management
        self.sltp_calculator = SLTPCalculator(config)
        self.position_sizer = PositionSizer(config)
        self.drawdown_monitor = DrawdownMonitor(config)

        # V3: Regime-adaptive trading
        self.use_adaptive_scorer = config.get("use_adaptive_scorer", False)
        self.regime_detector = RegimeDetector(config.get("regime_detection", {}))
        if self.use_adaptive_scorer:
            self.adaptive_scorer = AdaptiveConfluenceScorer(config)
            self.logger.info("V3 Adaptive Confluence Scorer ENABLED")
        # Dynamic Entry Gate (shared with BacktestEngine)
        self.entry_quality_engine = EntryQualityEngine()

        self.structure_sltp = StructureSLTPCalculator(config)
        # Bug #53: 93e05f7 changed config.get("risk",{}) → config, but micro_account
        # is nested under config["risk"], not at top level.  config gives empty micro_cfg
        # → defaults: max_risk_dollars=2.0, max_risk_pct=2.0% (blocks every trade).
        self.micro_account = MicroAccountManager(config.get("risk", {}))

        # Configurable exit stage thresholds (V3)
        exit_cfg = config.get("exit_stages", {})
        self.be_trigger_rr = exit_cfg.get("be_trigger_rr", 1.0)
        self.partial_close_rr = exit_cfg.get("partial_close_rr", 1.5)

        # Stale trade exit config
        stale_cfg = config.get("stale_trade", {})
        self.stale_trade_enabled = stale_cfg.get("enabled", False)
        self.stale_trade_min_hours = stale_cfg.get("min_hours", 3)
        self.stale_trade_min_peak_rr = stale_cfg.get("min_peak_rr", 0.3)

        # Position Management
        self.position_tracker = PositionTracker()
        self.recovery_manager = RecoveryManager(config, self.position_tracker)

        # Session Management
        self.session_manager = SessionManager(config.get("session", {}))

        # Execution
        self.order_executor = OrderExecutor(self.mt5, config)
        self.emergency_handler = EmergencyHandler(self.mt5, self.order_executor)

        # Bot intelligence
        self.health_monitor = HealthMonitor()

        # Trade Journal (per-ticket lifecycle tracker, 2-min snapshots)
        self.journal = TradeJournal("logs/trade_journal")

        # Telegram notifications
        tg_cfg = config.get("telegram", {})
        self.telegram = TelegramNotifier(
            token=tg_cfg.get("token", ""),
            chat_id=str(tg_cfg.get("chat_id", "")),
            enabled=bool(tg_cfg.get("enabled", False)),
        )

        # Trading symbol (single source of truth — replaces XAUUSD hardcodes)
        self.symbol = config.get("trading", {}).get("symbol", "XAUUSDm")

        # State
        self.loop_interval = config.get("bot", {}).get("loop_interval", 1)
        self._last_m15_candle_time = None  # Track actual M15 candle open time
        self._startup_warmup = True  # Skip signal processing on first M15 cycle
        self._last_market_data = None       # Cache last full analysis for fast path
        self.last_volatility_level = "medium"

        # Tick capture settings (every 2 minutes)
        self.tick_capture_interval = 120  # seconds
        self.last_tick_capture = None

        # SL-hit directional cooldown: prevent cascade re-entries
        # After BUY SL hit, block BUY for N candles (and vice versa)
        self._sl_cooldown = {}  # {"BUY": candle_time, "SELL": candle_time}
        self._sl_cooldown_candles = 2  # Block same direction for 2 M15 candles (30 min)

        # Market open/closed state tracker — sends Telegram on transitions
        self._market_allowed_prev = None  # None = not yet checked
        self._drawdown_paused_prev = False  # Telegram: only notify on transition

        # Position direction limits (prevents same-direction pile-ups)
        _pos_lim = config.get("position_limits", {})
        self._max_per_direction = _pos_lim.get("max_positions_per_direction", 2)
        self._min_spacing_pips = _pos_lim.get("min_position_distance", 20.0)

        # Timezone
        self.wib_tz = timezone(timedelta(hours=7))

        # Data directories
        self.trade_history_dir = Path("data/trade_history")
        self.trade_history_dir.mkdir(parents=True, exist_ok=True)

        # Trade history analyzer
        self.trade_analyzer = TradeAnalyzer(str(self.trade_history_dir))

        self.logger.info("Trading bot initialized successfully")

    def start(self) -> None:
        """Start the trading bot."""
        try:
            self.logger.info("=" * 60)
            self.logger.info("STARTING TRADING BOT")
            self.logger.info("=" * 60)

            # Connect to MT5
            self.logger.info("Connecting to MT5...")
            if not self.mt5.connect():
                raise Exception("MT5 connection failed")

            self.health_monitor.update_mt5_status(True)
            self.logger.info("MT5 connected successfully")

            # Initialize drawdown monitor
            account_info = self.mt5.get_account_info()
            if account_info:
                self.drawdown_monitor.initialize(account_info)

            # Run trade history analysis at startup
            trade_analysis = None
            try:
                trade_analysis = self.trade_analyzer.get_full_analysis()
                summary = self.trade_analyzer.get_summary_text(trade_analysis)
                self.logger.info("─" * 60)
                self.logger.info("TRADE HISTORY ANALYSIS")
                self.logger.info("─" * 60)
                for line in summary.splitlines():
                    self.logger.info(line if line.strip() else "")
                self.logger.info("─" * 60)
                self.telegram.send_trade_stats(trade_analysis)
            except Exception as e:
                self.logger.warning(f"Trade history analysis failed (non-fatal): {e}")

            # Start main loop
            self.running = True
            self.logger.info("Entering main loop...")

            # Notify Telegram — BOT STARTED
            try:
                _session_check = self.session_manager.is_trading_allowed()
                _ai = account_info or {}
                # Compute SL/TP range for notification
                _rw = self.config.get("regime_weights", {})
                _sl_vals = [
                    v.get("atr_sl_mult") for v in _rw.values()
                    if isinstance(v, dict) and v.get("atr_sl_mult")
                ]
                _sl_range = (min(_sl_vals), max(_sl_vals)) if _sl_vals else None
                _tp_atr = self.config.get("take_profit", {}).get("atr_multiplier")
                self.telegram.send_bot_started(
                    balance=float(_ai.get("balance", 0)),
                    equity=float(_ai.get("equity", 0)),
                    scorer_on=self.use_adaptive_scorer,
                    be_rr=self.be_trigger_rr,
                    pc_rr=self.partial_close_rr,
                    max_pos=self.config.get("position_limits", {}).get("max_open_positions", 3),
                    is_smc_v4=self.config.get("use_smc_v4", False),
                    session_check=_session_check,
                    trade_analysis=trade_analysis,
                    sl_atr_range=_sl_range,
                    tp_atr=_tp_atr,
                )
            except Exception as e:
                self.logger.warning(f"BOT STARTED notification failed: {e}")

            self._main_loop()

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
            # Don't close positions on Ctrl+C / SIGTERM - just disconnect
            self.stop(close_positions=False)
        except Exception as e:
            self.logger.critical(f"Fatal error starting bot: {e}")
            import traceback
            self.logger.critical(traceback.format_exc())
            self.health_monitor.record_error(e, "start")
            # Don't close positions on crash - just disconnect
            self.stop(close_positions=False)

    def stop(self, close_positions: bool = False) -> None:
        """
        Stop the trading bot.

        Args:
            close_positions: If True, close all positions before stopping.
                             Default False - positions stay open for next restart.
        """
        self.logger.info("=" * 60)
        self.logger.info("STOPPING TRADING BOT")
        self.logger.info("=" * 60)

        self.running = False

        # Only close positions if explicitly requested (e.g., emergency)
        if close_positions:
            self.logger.warning("Closing all positions before shutdown...")
            shutdown_result = self.emergency_handler.safe_shutdown("Bot stopped - close requested")
            if shutdown_result.get("success"):
                self.logger.info("All positions closed successfully")
            else:
                self.logger.warning("Some positions may remain open")
        else:
            # Normal shutdown - keep positions open
            open_count = len(self.mt5.get_positions()) if self.mt5.connected else 0
            if open_count > 0:
                self.logger.info(
                    f"Keeping {open_count} positions open (will be managed on restart)"
                )
            else:
                self.logger.info("No open positions")

        # Disconnect MT5
        if self.mt5.connected:
            self.mt5.disconnect()
            self.logger.info("MT5 disconnected")

        # Final statistics
        stats = self.health_monitor.get_statistics()
        self.logger.info(f"Bot ran for {stats['uptime_hours']:.1f} hours")
        self.logger.info(f"Total loops: {stats['loop_count']}")
        self.logger.info(f"Total signals: {stats['signals_generated']}")
        self.logger.info(f"Orders executed: {stats['orders_executed']}")

        self.telegram.send_bot_status(
            "BOT STOPPED",
            f"\U0001f552 Uptime: {stats['uptime_hours']:.1f}h | Loops: {stats['loop_count']}\n"
            f"\U0001f4ca Signals: {stats['signals_generated']} | Orders: {stats['orders_executed']}"
        )

        self.logger.info("Bot stopped successfully")

    def _is_new_m15_candle(self) -> bool:
        """Check if a new M15 candle has formed since last analysis."""
        if self._last_m15_candle_time is None:
            return True  # Startup → immediate analysis
        return self.timeframe_manager.is_new_bar(self.symbol, "M15", self._last_m15_candle_time)

    def _main_loop(self) -> None:
        """Main trading loop - NEVER exits unless self.running is False.

        Split into fast path (~1s) and slow path (on new M15 candle):
        - Fast: connection check, session, account/positions, position management, tick capture
        - Slow: full indicator/SMC/confluence analysis, signal generation, trade execution
        """
        consecutive_errors = 0
        consecutive_empty_data = 0  # Track failed data fetches
        max_consecutive_errors = 50  # After 50 consecutive errors, increase sleep
        _last_closed_log: Optional[datetime] = None  # Throttle "market closed" log

        while self.running:
            try:
                loop_start = time.time()

                # ── Fast path (every iteration ~1s) ──────────────────────

                # 1. Active MT5 connection check (ping, not just flag)
                if not self.mt5.ensure_connected():
                    self.logger.error("MT5 connection lost - reconnecting...")
                    self.health_monitor.update_mt5_status(False)
                    reconnect = self.emergency_handler.handle_connection_loss()
                    if not reconnect.get("success"):
                        self.logger.warning(
                            "Failed to reconnect to MT5, retrying in 30 seconds..."
                        )
                        time.sleep(30)
                        continue  # Retry instead of breaking
                    self.health_monitor.update_mt5_status(True)
                    consecutive_empty_data = 0

                # 2. Check if should trade
                session_check = self.session_manager.is_trading_allowed()
                is_allowed = session_check.get("allowed", False)
                mkt_status = session_check.get("status", "OPEN" if is_allowed else "CLOSED")
                mkt_reason = session_check.get("reason", "")
                opens_in = session_check.get("opens_in_minutes")

                # Telegram notification on market state transitions
                if is_allowed and self._market_allowed_prev is False:
                    # Transition: closed → open
                    self.telegram.send_market_status("OPEN", f"Trading resumed — {mkt_reason}")
                    self.logger.info(f"Market OPEN: {mkt_reason}")
                elif not is_allowed and self._market_allowed_prev is True:
                    # Transition: open → closed
                    self.logger.info(f"Market CLOSED: {mkt_reason}")
                    self.telegram.send_market_status(
                        mkt_status if mkt_status in ("MAINTENANCE", "WEEKEND") else "CLOSED",
                        mkt_reason,
                        opens_in_minutes=int(opens_in) if opens_in is not None else None,
                    )
                self._market_allowed_prev = is_allowed

                if not is_allowed:
                    # Log on first iteration and every 15 minutes thereafter
                    now_utc = datetime.now(timezone.utc)
                    if _last_closed_log is None or (now_utc - _last_closed_log).total_seconds() >= 900:
                        eta_str = ""
                        if opens_in is not None:
                            h, m = divmod(int(opens_in), 60)
                            eta_str = f" — opens in {h}h {m:02d}m" if h > 0 else f" — opens in {m}m"
                        self.logger.info(f"Market CLOSED ({mkt_status}){eta_str} | Bot alive, waiting...")
                        _last_closed_log = now_utc
                    time.sleep(self.loop_interval)
                    continue

                # 3. Get account and position data
                account_info = self.mt5.get_account_info()
                current_positions = self.mt5.get_positions(self.symbol)

                # Sync position tracker
                sync_report = self.position_tracker.sync_with_mt5(current_positions)

                # Bug #36 fix: record P/L for positions closed externally (SL/TP by MT5)
                for closed_pos in sync_report.get("closed_externally", []):
                    self.drawdown_monitor.record_trade_result(closed_pos)
                    profit = closed_pos.get("profit", 0)
                    ticket = closed_pos.get("ticket")
                    direction = closed_pos.get("direction", "")
                    entry_price = closed_pos.get("entry_price", 0)
                    self.logger.info(
                        f"External close recorded: #{ticket} P/L: ${profit:.2f}"
                    )
                    # Determine exit price + reason for SL/TP/trailing closes
                    sl_p  = closed_pos.get("sl") or closed_pos.get("stop_loss") or 0
                    tp_p  = closed_pos.get("tp") or closed_pos.get("take_profit") or 0
                    lot   = closed_pos.get("volume", 0.01)
                    # Infer actual exit price from profit (handles trailing SL correctly)
                    if entry_price and lot and lot > 0:
                        sign = 1 if "BUY" in str(direction).upper() else -1
                        implied_exit = entry_price + sign * profit / (lot * 100)
                    else:
                        implied_exit = None
                    if profit < 0:
                        reason = "SL Hit"
                        exit_price = sl_p if sl_p else (implied_exit or entry_price)
                    elif profit > 0:
                        # Distinguish TP hit vs trailing SL win by comparing implied exit to levels
                        if implied_exit and sl_p and tp_p:
                            if abs(implied_exit - tp_p) < abs(implied_exit - sl_p):
                                reason = "TP Hit"
                                exit_price = tp_p
                            else:
                                reason = "Trailing SL Hit"
                                exit_price = implied_exit
                        elif tp_p and implied_exit and abs(implied_exit - tp_p) < 1.0:
                            reason = "TP Hit"
                            exit_price = tp_p
                        else:
                            reason = "Trailing SL Hit"
                            exit_price = implied_exit or entry_price
                    else:
                        reason = "Breakeven"
                        exit_price = entry_price
                    self.telegram.send_exit(
                        direction=direction,
                        ticket=ticket,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        profit=profit,
                        reason=f"{reason} (MT5)",
                        lot=lot,
                        sl=sl_p,
                        entry_time=closed_pos.get("entry_time", ""),
                    )
                    self._log_trade_to_csv("CLOSE", {
                        "ticket": ticket,
                        "direction": direction,
                        "price": f"{exit_price:.2f}",
                        "profit": profit,
                        "session": closed_pos.get("entry_session", ""),
                        "smc_signals": closed_pos.get("entry_smc_signals", ""),
                        "regime": closed_pos.get("entry_regime", ""),
                        "comment": f"{reason} (MT5 external)",
                    })
                    # Trade Journal: log external close (SL/TP hit by MT5)
                    self.journal.log_exit(ticket, {
                        "price": exit_price,
                        "pnl_usd": profit,
                        "exit_reason": f"{reason} (MT5 external)",
                    })

                    # SL-hit cooldown: block same direction for N candles
                    if profit < -0.50:  # SL hit (not just spread loss)
                        dir_key = str(direction).upper()
                        if "BUY" in dir_key:
                            dir_key = "BUY"
                        elif "SELL" in dir_key:
                            dir_key = "SELL"
                        self._sl_cooldown[dir_key] = datetime.utcnow()  # wall-clock, not Polars time
                        self.logger.info(
                            f"SL COOLDOWN: {dir_key} blocked for {self._sl_cooldown_candles} candles"
                        )
                        _cd_min = int(self._sl_cooldown_candles * 15)
                        self.telegram.send_bot_status(
                            f"\u23f8 SL COOLDOWN \u2014 {dir_key}",
                            f"\u203a {dir_key} entries blocked for {self._sl_cooldown_candles} candles ({_cd_min}min)\n"
                            f"\u203a Resumes automaticaly after cooldown",
                        )

                # 4. Check risk limits (only blocks NEW entries, not exit management)
                risk_check = self.drawdown_monitor.check_trading_allowed(account_info)
                trading_paused = not risk_check.get("allowed")
                if trading_paused:
                    if not self._drawdown_paused_prev:
                        self.logger.warning(f"Risk limits exceeded: {risk_check['reason']} (exit management continues)")
                    if not self._drawdown_paused_prev:
                        # Build detail from violations or reason
                        violations = risk_check.get("violations", [])
                        if violations:
                            detail_lines = [f"\u203a {v['message']}" for v in violations]
                        else:
                            detail_lines = [f"\u203a {risk_check['reason']}"]
                        detail_lines.append("\u203a Exit management tetap aktif")
                        pause_until = risk_check.get("pause_until")
                        if pause_until:
                            remaining = (pause_until - datetime.now()).total_seconds() / 60
                            if remaining > 0:
                                detail_lines.append(f"\u203a Cooldown: {remaining:.0f} min")
                        self.telegram.send_bot_status(
                            "TRADING PAUSED \u26a0\ufe0f",
                            "\n".join(detail_lines),
                        )
                elif self._drawdown_paused_prev:
                    self.telegram.send_bot_status(
                        "TRADING RESUMED \u2705",
                        "\u203a Risk limits cleared \u2014 entry scanning aktif kembali",
                    )
                self._drawdown_paused_prev = trading_paused

                # 4.5 Emergency safety check (equity drop, margin level, large drawdown)
                emergency_active = False
                if self.emergency_handler.is_emergency_active():
                    self.logger.warning("Bot in EMERGENCY state — trading halted. Manual restart required.")
                    emergency_active = True
                else:
                    emg = self.emergency_handler.check_emergency_conditions(
                        account_info or {}, current_positions or []
                    )
                    if emg.get("emergency_needed"):
                        trigger_msg = emg["triggers"][0]["message"] if emg.get("triggers") else "Unknown"
                        self.logger.critical(f"EMERGENCY CONDITIONS MET: {trigger_msg}")
                        self.emergency_handler.emergency_stop(reason=trigger_msg)
                        emergency_active = True
                        self.telegram.send_bot_status(
                            "\U0001f6a8 EMERGENCY STOP",
                            f"\u203a {trigger_msg}\n"
                            f"\u203a ALL trading halted\n"
                            f"\u203a Manual restart required",
                        )

                # 5. Position management ALWAYS runs (BE, trailing, partial close protect profit)
                if current_positions and self._last_market_data:
                    # Update cached market_data with fresh tick price
                    tick = self.mt5.get_tick(self.symbol)
                    if tick:
                        fresh_price = tick.get("bid", 0)
                        self._last_market_data["current_price"] = fresh_price
                        self._last_market_data["market_data"]["bid"] = tick.get("bid", 0)
                        self._last_market_data["market_data"]["ask"] = tick.get("ask", 0)
                        self._last_market_data["market_data"]["spread"] = tick.get("ask", 0) - tick.get("bid", 0)
                    self._manage_positions(self._last_market_data, current_positions)

                # 6. Capture tick data (every 2 min)
                self._capture_tick()

                # Skip signal processing if paused or emergency
                if emergency_active:
                    time.sleep(10)
                    continue
                if trading_paused:
                    time.sleep(self.loop_interval)
                    continue

                # ── Slow path (only on new M15 candle) ───────────────────

                if self._is_new_m15_candle():
                    # Clear cache before fresh fetch
                    self.timeframe_manager.clear_cache()

                    # Full analysis pipeline
                    market_data = self._fetch_and_process_data()
                    if not market_data:
                        consecutive_empty_data += 1
                        if consecutive_empty_data >= 10 and consecutive_empty_data % 10 == 0:
                            self.logger.warning(
                                f"Data fetch failed {consecutive_empty_data} times - forcing MT5 reconnect"
                            )
                            self.mt5.disconnect()
                            time.sleep(5)
                            self.mt5.connect()
                        time.sleep(self.loop_interval)
                        continue
                    consecutive_empty_data = 0

                    # Update candle time from M15 dataframe
                    df_m15 = market_data.get("df_m15")
                    if df_m15 is not None and len(df_m15) > 0:
                        self._last_m15_candle_time = df_m15["time"][-1]

                    # Cache for fast-path position management
                    self._last_market_data = market_data

                    # Track volatility level for dynamic position limits
                    vol_analysis = market_data.get("volatility_analysis", {})
                    vol_level = vol_analysis.get("level")
                    if vol_level:
                        self.last_volatility_level = vol_level.value if hasattr(vol_level, "value") else str(vol_level)

                    # Full position management with fresh analysis
                    self._manage_positions(market_data, current_positions)

                    # Build position info for combined telegram message
                    position_info = self._build_position_info(current_positions)

                    # Startup warmup: first M15 cycle is analysis-only, no new entries.
                    # Prevents blind entry on restart before bot has observed market context.
                    gate_result = None
                    if self._startup_warmup:
                        self._startup_warmup = False
                        self.logger.info("Startup warmup: analysis complete, skipping signals until next M15 candle")
                        gate_result = {"passed": False, "reason": "Warmup \u2014 analisis pertama, menunggu candle berikutnya"}
                    else:
                        # Generate and evaluate new signals (dynamic max positions)
                        max_pos = self._get_dynamic_max_positions(self.last_volatility_level, account_info)
                        if len(current_positions) < max_pos:
                            gate_result = self._process_new_signals(market_data, current_positions, account_info)
                        else:
                            gate_result = {"passed": False, "reason": f"BULL: Max position reached | BEAR: Max position reached"}

                    # Send combined telegram: analysis + position + gates
                    self.telegram.send_signal_analysis(
                        {
                            "current_price": market_data["current_price"],
                            "technical_indicators": market_data["technical_indicators"],
                            "smc_analysis": market_data["smc_analysis"],
                            "confluence_scores": market_data["confluence_scores"],
                            "market_analysis": market_data["market_analysis"],
                            "trend_analysis": market_data["trend_analysis"],
                            "mtf_analysis": market_data["mtf_analysis"],
                            "volatility_analysis": market_data.get("volatility_analysis", {}),
                            "market_data": market_data["market_data"],
                            "session_name": market_data.get("session_name", "Unknown"),
                            "threshold": market_data.get("threshold", 0.55),
                            "regime": market_data.get("regime", "unknown"),
                        },
                        gate_result=gate_result,
                        position_info=position_info,
                    )

                # 7. Record successful loop
                self.health_monitor.record_loop_iteration()
                consecutive_errors = 0  # Reset on success

                # 8. Sleep until next iteration
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.loop_interval - loop_duration)
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                self.logger.info("KeyboardInterrupt in main loop")
                self.running = False
                break

            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"Error in main loop ({consecutive_errors}): {e}")
                self.health_monitor.record_error(e, "main_loop")

                # Adaptive sleep: longer delays when errors persist
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.warning(
                        f"{consecutive_errors} consecutive errors, sleeping 60s..."
                    )
                    time.sleep(60)
                elif consecutive_errors >= 10:
                    time.sleep(10)
                else:
                    time.sleep(self.loop_interval)

    def _fetch_and_process_data(self) -> Dict:
        """Fetch and process market data (matches backtest pipeline)."""
        try:
            # Fetch multi-timeframe data
            mtf_data = self.timeframe_manager.fetch_multiple_timeframes(
                self.symbol,
                ["H1", "M15", "M5", "M1"]
            )

            if not mtf_data or "M15" not in mtf_data:
                return None

            df_m15 = mtf_data["M15"]

            if len(df_m15) < 100:
                self.logger.debug("Insufficient M15 data (need 100+ bars)")
                return None

            # Note: basic features & price changes already added by TimeframeManager

            # Calculate technical indicators
            df_m15 = self.technical_indicators.calculate_all(df_m15)

            # Calculate SMC indicators on full dataset
            df_m15 = self.smc_indicators.calculate_all(df_m15)

            # Update mtf_data with processed M15 (so MTF analyzer sees indicators)
            mtf_data["M15"] = df_m15

            # V3: Detect market regime
            regime_result = self.regime_detector.detect(df_m15)
            current_regime = regime_result["regime"]

            # Also calculate indicators on M5 for MTF alignment + LTF confirmation
            if "M5" in mtf_data:
                mtf_data["M5"] = self.technical_indicators.calculate_all(mtf_data["M5"])

            # Get current price and tick data
            tick = self.mt5.get_tick(self.symbol)
            current_price = tick.get("bid", 0) if tick else 0
            current_time = datetime.utcnow()

            # Get SMC signals for current price (matches backtest)
            bullish_smc = self.smc_indicators.get_bullish_signals(df_m15, current_price)
            bearish_smc = self.smc_indicators.get_bearish_signals(df_m15, current_price)
            smc_analysis = {
                "bullish": bullish_smc,
                "bearish": bearish_smc,
            }

            # Technical indicator values (proper format for confluence scorer)
            ema_20_val = df_m15["ema_20"][-1] if "ema_20" in df_m15.columns else current_price
            ema_50_val = df_m15["ema_50"][-1] if "ema_50" in df_m15.columns else current_price
            technical_indicators = {
                "atr": df_m15["atr_14"][-1] if "atr_14" in df_m15.columns else 15.0,
                "rsi": df_m15["rsi_14"][-1] if "rsi_14" in df_m15.columns else 50.0,
                "ema_20": ema_20_val,
                "ema": {
                    20: ema_20_val,
                    50: ema_50_val,
                },
                "macd": {
                    "histogram": df_m15["macd_histogram"][-1] if "macd_histogram" in df_m15.columns else None,
                },
            }

            # Market analysis (correct signatures - single df arg)
            market_analysis = self.market_analyzer.analyze(df_m15)
            volatility_analysis = self.volatility_analyzer.analyze(df_m15)

            # Enrich market_analysis with volatility details for confluence scorer
            market_analysis["volatility"] = volatility_analysis

            # Trend analysis
            trend_analysis = self.trend_analyzer.analyze(df_m15)

            # MTF analysis
            mtf_analysis = self.mtf_analyzer.analyze(mtf_data)

            # H1 Bias: determine trend direction from H1 EMA20 vs EMA50
            h1_bias = "neutral"
            if "H1" in mtf_data:
                df_h1 = mtf_data["H1"]
                if "ema_20" not in df_h1.columns:
                    df_h1 = self.technical_indicators.calculate_all(df_h1)
                if "ema_20" in df_h1.columns and "ema_50" in df_h1.columns and len(df_h1) > 0:
                    h1_ema20 = df_h1["ema_20"][-1]
                    h1_ema50 = df_h1["ema_50"][-1]
                    h1_close = df_h1["close"][-1]
                    self.logger.debug(
                        f"H1 Bias: EMA20={h1_ema20:.2f} EMA50={h1_ema50:.2f} Close={h1_close:.2f}"
                    )
                    if h1_ema20 > h1_ema50 and h1_close > h1_ema50:
                        h1_bias = "bullish"
                    elif h1_ema20 < h1_ema50 and h1_close < h1_ema50:
                        h1_bias = "bearish"
            else:
                self.logger.warning("H1 data not available in mtf_data — h1_bias defaults to neutral")
            mtf_analysis["h1_bias"] = h1_bias

            # Build LTF (M5) confirmation data if enabled
            # M5 indicators already calculated above for MTF alignment
            ltf_data = None
            ltf_cfg = self.config.get("mtf_analysis", {}).get("ltf_confirmation", {})
            if ltf_cfg.get("enabled", False) and "M5" in mtf_data:
                try:
                    df_m5 = mtf_data["M5"]
                    lookback = ltf_cfg.get("lookback_bars", 6)
                    if len(df_m5) >= lookback:
                        m15_time = df_m15["time"][-1]
                        ltf_data = {"m5_df": df_m5, "current_m15_time": m15_time}
                except Exception as e:
                    self.logger.debug(f"LTF data prep failed (non-fatal): {e}")
                    ltf_data = None

            # Confluence scoring - calculate for both directions
            from ..core.constants import TrendDirection
            if self.use_adaptive_scorer:
                bullish_confluence = self.adaptive_scorer.calculate_score(
                    TrendDirection.BULLISH,
                    bullish_smc,
                    technical_indicators,
                    market_analysis,
                    mtf_analysis,
                    regime=current_regime,
                    ltf_data=ltf_data,
                    opposing_smc=bearish_smc,
                )
                bearish_confluence = self.adaptive_scorer.calculate_score(
                    TrendDirection.BEARISH,
                    bearish_smc,
                    technical_indicators,
                    market_analysis,
                    mtf_analysis,
                    regime=current_regime,
                    ltf_data=ltf_data,
                    opposing_smc=bullish_smc,
                )
            else:
                bullish_confluence = self.confluence_scorer.calculate_score(
                    TrendDirection.BULLISH,
                    bullish_smc,
                    technical_indicators,
                    market_analysis,
                    mtf_analysis,
                    ltf_data=ltf_data,
                )
                bearish_confluence = self.confluence_scorer.calculate_score(
                    TrendDirection.BEARISH,
                    bearish_smc,
                    technical_indicators,
                    market_analysis,
                    mtf_analysis,
                    ltf_data=ltf_data,
                )
            confluence_scores = {
                "bullish": bullish_confluence,
                "bearish": bearish_confluence,
            }

            # Feed RSI to entry signal generator for bounce detection
            rsi_val = technical_indicators.get("rsi")
            if rsi_val is not None:
                self.strategy.entry_generator.update_rsi_history(rsi_val)

            # Determine effective threshold
            if self.use_adaptive_scorer:
                effective_threshold = bullish_confluence.get("min_confluence", 0.55)
            else:
                effective_threshold = self.config.get("strategy", {}).get("entry", {}).get("min_confluence_score", 0.55)

            # Log detailed analysis (fires every call - method is only called on new candle)
            now = datetime.utcnow()
            session_info = self.session_manager.get_current_session(force_update=True)
            session_name = session_info.get("name", "Unknown") if session_info else "No Session"
            wib_now = datetime.now(timezone(timedelta(hours=7)))

            # Plain text for log file
            bull_s = bullish_confluence['score']
            bear_s = bearish_confluence['score']
            self.logger.info(
                f"M15 | {now.strftime('%H:%M')} UTC | {session_name} | "
                f"{current_price:.2f} | RSI {technical_indicators['rsi']:.1f} | "
                f"{current_regime.value} | Bull:{bull_s:.2f} Bear:{bear_s:.2f} Thr:{effective_threshold:.2f}"
            )

            # Colorized console output
            self._print_colored_analysis(
                now, wib_now, session_name, current_price,
                technical_indicators, ema_20_val, ema_50_val,
                current_regime, regime_result,
                bullish_smc, bearish_smc,
                bullish_confluence, bearish_confluence,
                effective_threshold, mtf_analysis, market_analysis,
                trend_analysis, h1_bias,
            )

            # Telegram data stored in return dict (sent by main loop after gate check)

            return {
                "current_price": current_price,
                "current_time": current_time,
                "df_m15": df_m15,
                "smc_analysis": smc_analysis,
                "technical_indicators": technical_indicators,
                "market_analysis": market_analysis,
                "volatility_analysis": volatility_analysis,
                "trend_analysis": trend_analysis,
                "mtf_analysis": mtf_analysis,
                "confluence_scores": confluence_scores,
                "regime_result": regime_result,
                "regime": current_regime.value,
                "session_name": session_name,
                "threshold": effective_threshold,
                "market_data": {
                    "bid": tick.get("bid", 0) if tick else 0,
                    "ask": tick.get("ask", 0) if tick else 0,
                    "spread": tick.get("spread", 0.02) if tick else 0.02,
                    "time": current_time,
                },
            }

        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            self.health_monitor.record_error(e, "fetch_data")
            return None

    def _is_approaching_daily_close(self) -> bool:
        """Return True during pre-maintenance hour (DST-aware).

        Exness XAUUSDm maintenance runs after NY close daily.
        Winter: maintenance 22:00-23:00, pre-close hour = 21
        Summer: maintenance 21:00-22:00, pre-close hour = 20
        """
        import pytz
        now_utc = datetime.now(pytz.UTC)
        pre_close = self.session_manager.get_pre_close_hour(now_utc)
        return now_utc.hour == pre_close

    def _get_dynamic_max_positions(self, volatility_level: str = "medium", account_info: Dict = None) -> int:
        """Calculate dynamic max positions based on volatility and drawdown."""
        # Fixed max 3 regardless of volatility — 2nd/3rd entry gated by 2 SMC signal rule
        max_pos = 3

        # Reduce if in drawdown
        if account_info:
            balance = account_info.get("balance", 0)
            equity = account_info.get("equity", balance)
            if balance > 0:
                drawdown_pct = ((balance - equity) / balance) * 100
                if drawdown_pct > 5:
                    max_pos = max(1, max_pos - 2)
                elif drawdown_pct > 3:
                    max_pos = max(1, max_pos - 1)

        return max_pos

    def _manage_positions(self, market_data: Dict, current_positions: list) -> None:
        """Manage existing positions with multi-stage exit strategy."""
        try:
            current_price = market_data["current_price"]
            atr = market_data["technical_indicators"]["atr"]

            for position in current_positions:
                ticket = position.get("ticket")

                # Bug #36b fix: skip positions no longer in tracker (closed by MT5
                # or already processed this iteration — prevents duplicate execution).
                if not self.position_tracker.get_position(ticket):
                    continue

                # Update metrics
                self.position_tracker.update_position_metrics(
                    ticket, current_price, position.get("profit")
                )

                # --- Multi-stage exit management ---
                entry_price = position.get("open_price", position.get("price_open", 0))
                current_sl = position.get("sl", 0)
                direction = position.get("type", "BUY")  # Already "BUY"/"SELL" string

                # Bug #47 fix: use ORIGINAL SL distance for RR calculation.
                # After BE/trailing modify current_sl, abs(entry - current_sl) shrinks
                # to ~2 pips, making rr_ratio 10x inflated → tightest trail factor (0.35)
                # immediately. Use entry_sl from tracker for stable RR throughout trade.
                tracked_entry_sl = (self.position_tracker.get_position(ticket) or {}).get("entry_sl", 0)
                if tracked_entry_sl > 0:
                    sl_distance = abs(entry_price - tracked_entry_sl)
                else:
                    sl_distance = abs(entry_price - current_sl) if current_sl > 0 else atr * 3.0

                # Calculate unrealized profit distance
                if direction == "BUY":
                    profit_distance = current_price - entry_price
                else:
                    profit_distance = entry_price - current_price

                rr_ratio = profit_distance / sl_distance if sl_distance > 0 else 0

                # Stage 1: Move SL to breakeven at configured RR threshold
                tracked = self.position_tracker.get_position(ticket)
                already_be = (tracked or {}).get("breakeven_set", False)

                # Trade Journal: snapshot every 2 minutes (throttled internally)
                self.journal.log_snapshot(ticket, {
                    "price": current_price,
                    "pnl_usd": position.get("profit", 0),
                    "rr": rr_ratio,
                    "sl": current_sl,
                    "stage": self._get_trade_stage(tracked, rr_ratio),
                })

                if rr_ratio >= self.be_trigger_rr and current_sl != 0 and not already_be:
                    buffer = sl_distance * 0.1
                    if direction == "BUY":
                        be_sl = entry_price + buffer
                        if current_sl < be_sl:
                            result = self.order_executor.modify_position(
                                ticket, new_sl=be_sl, reason="Breakeven (1:1 RR reached)"
                            )
                            if result.get("success"):
                                self.logger.info(f"#{ticket} SL moved to breakeven: {be_sl:.2f}")
                                self.position_tracker.update_position(ticket, {"breakeven_set": True})
                                self.telegram.send_modification(ticket, "BREAKEVEN", f"SL \u2192 {be_sl:.2f} ({self.be_trigger_rr:.1f}:1 RR)")
                    else:
                        be_sl = entry_price - buffer
                        if current_sl > be_sl:
                            result = self.order_executor.modify_position(
                                ticket, new_sl=be_sl, reason=f"Breakeven ({self.be_trigger_rr:.1f}:1 RR reached)"
                            )
                            if result.get("success"):
                                self.logger.info(f"#{ticket} SL moved to breakeven: {be_sl:.2f}")
                                self.position_tracker.update_position(ticket, {"breakeven_set": True})
                                self.telegram.send_modification(ticket, "BREAKEVEN", f"SL \u2192 {be_sl:.2f} ({self.be_trigger_rr:.1f}:1 RR)")

                # Stage 1.5: Pre-close profit lock (23:00–23:59 UTC daily)
                # Before daily maintenance window, lock 65% of current profit via SL.
                # Prevents giving back unrealized gains during overnight low-liquidity period.
                already_preclose_locked = (tracked or {}).get("pre_close_locked", False)
                if (
                    self._is_approaching_daily_close()
                    and rr_ratio >= 0.30
                    and profit_distance > 0
                    and not already_preclose_locked
                ):
                    lock_profit = profit_distance * 0.65
                    if direction == "BUY":
                        lock_sl = round(entry_price + lock_profit, 2)
                        if lock_sl > current_sl:
                            result = self.order_executor.modify_position(
                                ticket, new_sl=lock_sl, reason="Pre-close profit lock"
                            )
                            if result.get("success"):
                                self.logger.info(
                                    f"#{ticket} Pre-close lock: SL\u2192{lock_sl:.2f} (65% of {profit_distance:.1f} pts locked)"
                                )
                                self.position_tracker.update_position(ticket, {"pre_close_locked": True})
                                self.telegram.send_modification(
                                    ticket, "PRE-CLOSE LOCK",
                                    f"SL \u2192 {lock_sl:.2f} (65% profit locked, maintenance approaching)"
                                )
                    else:
                        lock_sl = round(entry_price - lock_profit, 2)
                        if lock_sl < current_sl or current_sl == 0:
                            result = self.order_executor.modify_position(
                                ticket, new_sl=lock_sl, reason="Pre-close profit lock"
                            )
                            if result.get("success"):
                                self.logger.info(
                                    f"#{ticket} Pre-close lock: SL\u2192{lock_sl:.2f} (65% of {profit_distance:.1f} pts locked)"
                                )
                                self.position_tracker.update_position(ticket, {"pre_close_locked": True})
                                self.telegram.send_modification(
                                    ticket, "PRE-CLOSE LOCK",
                                    f"SL \u2192 {lock_sl:.2f} (65% profit locked, maintenance approaching)"
                                )

                # Stage 2: Dynamic partial close / profit lock
                # Use position's actual TP/SL to compute effective_partial_rr.
                # Old value (2.73) always EXCEEDED TP_RR (1.29–2.31) → never fired.
                # Formula: max(tp_rr * 0.65, 1.0) — captures 65% of TP distance,
                #   minimum 1.0R (always above BE at 0.77R and below min TP_RR 1.29R).
                tp_price = position.get("tp", 0) or position.get("price_tp", 0)
                if tp_price and tp_price > 0 and sl_distance > 0:
                    tp_dist = abs(float(tp_price) - entry_price)
                    tp_rr   = tp_dist / sl_distance
                    effective_partial_rr = max(tp_rr * 0.65, 1.0)
                else:
                    effective_partial_rr = self.partial_close_rr  # fallback to config

                if rr_ratio >= effective_partial_rr and current_sl != 0:
                    volume = position.get("volume", 0.01)
                    already_partial = (tracked or {}).get("partial_closed", False)

                    if not already_partial and volume > 0.01:
                        close_volume = round(volume / 2, 2)
                        close_volume = max(close_volume, 0.01)  # Minimum 0.01 lot
                        self.logger.info(
                            f"#{ticket} Stage 2: Partial close {close_volume} lots at {rr_ratio:.1f}:1 RR "
                            f"(effective_partial={effective_partial_rr:.2f}R)"
                        )
                        closed = self.mt5.close_position(ticket, volume=close_volume)
                        if closed:
                            self.logger.info(f"#{ticket} Partial close successful - 50% profit locked")
                            self.position_tracker.update_position(ticket, {"partial_closed": True})
                        else:
                            self.logger.warning(f"#{ticket} Partial close failed")
                    elif not already_partial and volume <= 0.01:
                        # Can't partial close 0.01 lot — lock 50% profit via SL instead
                        lock_profit = profit_distance * 0.5
                        if direction == "BUY":
                            lock_sl = round(entry_price + lock_profit, 2)
                            if lock_sl > current_sl:
                                result = self.order_executor.modify_position(
                                    ticket, new_sl=lock_sl, reason=f"Profit Lock ({rr_ratio:.1f}:1 RR, min lot)"
                                )
                                if result.get("success"):
                                    self.logger.info(f"#{ticket} Profit locked: SL\u2192{lock_sl:.2f} (50% of {profit_distance:.1f} at {rr_ratio:.1f}R, partial_thr={effective_partial_rr:.2f}R)")
                                    self.telegram.send_modification(ticket, "PROFIT LOCK", f"SL \u2192 {lock_sl:.2f} (50% locked at {rr_ratio:.1f}:1 RR)")
                        else:
                            lock_sl = round(entry_price - lock_profit, 2)
                            if lock_sl < current_sl or current_sl == 0:
                                result = self.order_executor.modify_position(
                                    ticket, new_sl=lock_sl, reason=f"Profit Lock ({rr_ratio:.1f}:1 RR, min lot)"
                                )
                                if result.get("success"):
                                    self.logger.info(f"#{ticket} Profit locked: SL\u2192{lock_sl:.2f} (50% of {profit_distance:.1f} at {rr_ratio:.1f}R, partial_thr={effective_partial_rr:.2f}R)")
                                    self.telegram.send_modification(ticket, "PROFIT LOCK", f"SL \u2192 {lock_sl:.2f} (50% locked at {rr_ratio:.1f}:1 RR)")
                        self.position_tracker.update_position(ticket, {"partial_closed": True})
                        self.logger.info(f"#{ticket} Lot too small for partial close, profit locked via SL")

                # Stage 3: Progressive RR-based trailing (activates after BE, not after partial)
                # Bug fix: trail now fires after BE (0.77R), not after partial close (2.73R > TP).
                # Progressive factor tightens as profit grows to protect larger gains:
                #   <0.9R → 0.65 (loose, give room early)
                #   0.9–1.5R → 0.50 (normal, matching backtest)
                #   >1.5R  → 0.35 (tight, protect substantial profits)
                if already_be and profit_distance > 0:
                    # Track peak profit price
                    prev_peak = (tracked or {}).get("peak_profit_price", entry_price)
                    if direction == "BUY":
                        peak_price = max(prev_peak, current_price)
                        peak_profit_dist = peak_price - entry_price
                    else:
                        peak_price = min(prev_peak, current_price)
                        peak_profit_dist = entry_price - peak_price
                    self.position_tracker.update_position(ticket, {"peak_profit_price": peak_price})

                    # Progressive trail factor based on current RR
                    # XAUUSD routinely pulls back 30-40% of a move. f=0.35 at 1.5R
                    # was too tight — stopped out this trade at +1.36R when TP was 2.44R.
                    # Widened: only tighten to 0.35 above 3R (deep in TP territory).
                    if rr_ratio < 1.0:
                        trail_factor = 0.65  # loose: give room near BE
                    elif rr_ratio < 2.0:
                        trail_factor = 0.55  # moderate: building profit
                    elif rr_ratio < 3.0:
                        trail_factor = 0.45  # tighter: approaching TP
                    else:
                        trail_factor = 0.35  # tightest: protect windfall (above 3R)
                    trail_distance = peak_profit_dist * trail_factor

                    # Min increment guard: avoid spamming MT5 for sub-pip moves
                    MIN_TRAIL_INCREMENT = 0.5
                    if direction == "BUY":
                        new_sl = round(peak_price - trail_distance, 2)
                        if new_sl >= current_sl + MIN_TRAIL_INCREMENT:
                            result = self.order_executor.modify_position(
                                ticket, new_sl=new_sl, reason="RR Trailing Stop"
                            )
                            if result.get("success"):
                                self.logger.info(
                                    f"#{ticket} RR Trail: SL\u2192{new_sl:.2f} (peak:{peak_price:.2f}, trail:{trail_distance:.1f}, f={trail_factor})"
                                )
                                self.telegram.send_modification(ticket, "TRAILING", f"SL \u2192 {new_sl:.2f} (peak:{peak_price:.2f} | trail:{trail_distance:.1f} | f={trail_factor})")
                    else:
                        new_sl = round(peak_price + trail_distance, 2)
                        if new_sl <= current_sl - MIN_TRAIL_INCREMENT or current_sl == 0:
                            result = self.order_executor.modify_position(
                                ticket, new_sl=new_sl, reason="RR Trailing Stop"
                            )
                            if result.get("success"):
                                self.logger.info(
                                    f"#{ticket} RR Trail: SL\u2192{new_sl:.2f} (peak:{peak_price:.2f}, trail:{trail_distance:.1f}, f={trail_factor})"
                                )
                                self.telegram.send_modification(ticket, "TRAILING", f"SL \u2192 {new_sl:.2f} (peak:{peak_price:.2f} | trail:{trail_distance:.1f} | f={trail_factor})")

                # Stale trade exit: position open N hours but peak profit never reached min threshold.
                # Momentum is dead — cut loss early rather than waiting for full SL hit.
                # Skip if BE already set — MT5's SL will close at 0 loss, no point closing at a real loss.
                if self.stale_trade_enabled and profit_distance < 0 and not already_be:
                    stale_already_tried = (tracked or {}).get("stale_exit_attempted", False)
                    if not stale_already_tried:
                        entry_time = (tracked or {}).get("entry_time")
                        if not entry_time:
                            self.logger.warning(f"#{ticket} Stale check skipped: no entry_time in tracker")
                        else:
                            # Bug #48 safe: parse ISO string + ensure tz-aware
                            if isinstance(entry_time, str):
                                entry_time = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                            now_utc = datetime.now(timezone.utc)
                            if entry_time.tzinfo is None:
                                entry_time = entry_time.replace(tzinfo=timezone.utc)
                            age_hours = (now_utc - entry_time).total_seconds() / 3600

                            peak_profit_dist = (tracked or {}).get("peak_profit", 0)
                            peak_rr = peak_profit_dist / sl_distance if sl_distance > 0 else 0

                            # After restart, peak_profit resets to 0.0 — we don't know true peak.
                            # Log it so it's visible, but still fire (full SL loss is worse than false exit).
                            if (tracked or {}).get("entry_sl_from_restart") and peak_profit_dist == 0:
                                self.logger.info(
                                    f"#{ticket} Stale check post-restart: peak tracking reset, peak_rr={peak_rr:.2f} may be understated"
                                )

                            if age_hours >= self.stale_trade_min_hours and peak_rr < self.stale_trade_min_peak_rr:
                                self.logger.info(
                                    f"#{ticket} Stale trade exit: peak={peak_rr:.2f}R (min {self.stale_trade_min_peak_rr}R), age={age_hours:.1f}h"
                                )
                                result = self.order_executor.execute_exit(
                                    ticket, f"Stale trade exit (peak {peak_rr:.2f}R in {age_hours:.1f}h)"
                                )
                                if result.get("success"):
                                    self.position_tracker.update_position(
                                        ticket, {"stale_exit_attempted": True}
                                    )
                                    tracked_pos = self.position_tracker.get_position(ticket)
                                    if tracked_pos and "profit" not in result:
                                        result["profit"] = tracked_pos.get("profit", 0)
                                    self.drawdown_monitor.record_trade_result(result)
                                    self.position_tracker.remove_position(ticket, result)
                                    self.telegram.send_recovery_action(
                                        ticket, "STALE EXIT",
                                        f"P/L: ${result.get('profit', 0):.2f} | No momentum (peak {peak_rr:.2f}R in {age_hours:.1f}h)"
                                    )
                                    self._log_trade_to_csv("CLOSE", {
                                        "ticket": ticket,
                                        "direction": direction,
                                        "price": current_price,
                                        "profit": result.get("profit", ""),
                                        "session": (tracked or {}).get("entry_session", ""),
                                        "smc_signals": (tracked or {}).get("entry_smc_signals", ""),
                                        "comment": "Stale trade exit",
                                    })
                                    self.journal.log_exit(ticket, {
                                        "price": current_price,
                                        "pnl_usd": result.get("profit", 0),
                                        "exit_reason": f"Stale trade exit (peak {peak_rr:.2f}R, {age_hours:.1f}h)",
                                    })
                                    continue

                # Near-SL early exit: within 20% of full SL AND held 3+ hours → cut loss early.
                # Accepts ~80% of max loss rather than waiting for full SL hit.
                # Avoids letting a near-dead position sit until time_exit (48h).
                if rr_ratio < -0.80:
                    near_sl_already_tried = (tracked or {}).get("near_sl_exit_attempted", False)
                    if not near_sl_already_tried:
                        entry_time = (tracked or {}).get("entry_time")
                        if not entry_time:
                            self.logger.warning(f"#{ticket} Near-SL check skipped: no entry_time in tracker")
                        else:
                            # Bug #48 fix: entry_time stored as ISO string, must parse before subtraction
                            if isinstance(entry_time, str):
                                entry_time = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                            now_utc = datetime.now(timezone.utc)
                            # Ensure both are offset-aware for subtraction
                            if entry_time.tzinfo is None:
                                entry_time = entry_time.replace(tzinfo=timezone.utc)
                            age_hours = (now_utc - entry_time).total_seconds() / 3600
                            if age_hours >= 3.0:
                                self.logger.info(
                                    f"#{ticket} Near-SL early exit: RR={rr_ratio:.2f}, age={age_hours:.1f}h"
                                )
                                self.position_tracker.update_position(
                                    ticket, {"near_sl_exit_attempted": True}
                                )
                                result = self.order_executor.execute_exit(
                                    ticket, "Near-SL early exit (80% loss, 3h+)"
                                )
                                if result.get("success"):
                                    tracked_pos = self.position_tracker.get_position(ticket)
                                    if tracked_pos and "profit" not in result:
                                        result["profit"] = tracked_pos.get("profit", 0)
                                    self.drawdown_monitor.record_trade_result(result)
                                    self.position_tracker.remove_position(ticket, result)
                                    self.telegram.send_recovery_action(
                                        ticket, "EARLY EXIT",
                                        f"Loss: ${result.get('profit', 0):.2f} | Near-SL ({rr_ratio:.2f}R) after {age_hours:.1f}h"
                                    )
                                    self._log_trade_to_csv("CLOSE", {
                                        "ticket": ticket,
                                        "direction": direction,
                                        "price": current_price,
                                        "profit": result.get("profit", ""),
                                        "session": (tracked or {}).get("entry_session", ""),
                                        "smc_signals": (tracked or {}).get("entry_smc_signals", ""),
                                        "comment": "Near-SL early exit",
                                    })
                                    self.journal.log_exit(ticket, {
                                        "price": current_price,
                                        "pnl_usd": result.get("profit", 0),
                                        "exit_reason": f"Near-SL early exit ({rr_ratio:.2f}R, {age_hours:.1f}h)",
                                    })
                                    continue

                # --- Feature 4: Recovery Manager for losing positions ---
                if profit_distance < 0:
                    recovery_rec = self.recovery_manager.analyze_position_recovery(ticket)
                    if recovery_rec and recovery_rec.get("needs_recovery"):
                        recommended = recovery_rec.get("recommended_action")
                        if recommended:
                            action_type = recommended.get("type")
                            if action_type == "time_exit":
                                # Execute time-based recovery exit
                                self.logger.info(f"#{ticket} Recovery: time exit after max recovery period")
                                result = self.order_executor.execute_exit(ticket, "Recovery time exit")
                                if result.get("success"):
                                    tracked_pos = self.position_tracker.get_position(ticket)
                                    if tracked_pos and "profit" not in result:
                                        result["profit"] = tracked_pos.get("profit", 0)
                                    self.drawdown_monitor.record_trade_result(result)
                                    self.position_tracker.remove_position(ticket, result)
                                    self.telegram.send_recovery_action(
                                        ticket, "TIME EXIT",
                                        f"Loss: ${result.get('profit', 0):.2f} | Exceeded max recovery time"
                                    )
                                    self._log_trade_to_csv("CLOSE", {
                                        "ticket": ticket,
                                        "direction": direction,
                                        "price": current_price,
                                        "profit": result.get("profit", ""),
                                        "session": tracked_pos.get("entry_session", "") if tracked_pos else "",
                                        "smc_signals": tracked_pos.get("entry_smc_signals", "") if tracked_pos else "",
                                        "comment": "Recovery time exit",
                                    })
                                    self.journal.log_exit(ticket, {
                                        "price": current_price,
                                        "pnl_usd": result.get("profit", 0),
                                        "exit_reason": "Recovery time exit",
                                    })
                                    continue
                            elif action_type == "move_to_breakeven":
                                # Move SL to breakeven for recovering positions.
                                # Guard: recovery_manager uses stale tracker price — verify
                                # against LIVE current_price to avoid MT5 INVALID_STOPS error
                                # (broker rejects SL above market for BUY / below for SELL).
                                action_result = self.recovery_manager.execute_recovery_action(
                                    ticket, "move_to_breakeven"
                                )
                                if action_result.get("requires_execution"):
                                    new_sl = action_result.get("new_sl")
                                    if new_sl:
                                        # Validate SL is on the correct side of LIVE price
                                        be_valid = (
                                            (direction == "BUY" and current_price > new_sl) or
                                            (direction == "SELL" and current_price < new_sl)
                                        )
                                        if not be_valid:
                                            self.logger.warning(
                                                f"#{ticket} Recovery BE skipped: live price "
                                                f"{current_price:.2f} on wrong side of BE SL {new_sl:.2f} "
                                                f"(would cause INVALID_STOPS)"
                                            )
                                        else:
                                            mod_result = self.order_executor.modify_position(
                                                ticket, new_sl=new_sl, reason="Recovery breakeven"
                                            )
                                            if mod_result.get("success"):
                                                self.telegram.send_recovery_action(
                                                    ticket, "BREAKEVEN",
                                                    f"SL moved to {new_sl:.2f} (recovery)"
                                                )

                # Check strategy exit signals (regime-adaptive structure exit threshold)
                regime_result = market_data.get("regime_result", {})
                current_regime = regime_result.get("regime", MarketRegime.RANGE_WIDE)
                exit_signal = self.strategy.exit_generator.check_exit_conditions(
                    position,
                    current_price,
                    market_data["smc_analysis"],
                    market_data["market_analysis"],
                    market_data["technical_indicators"],
                    regime=current_regime,
                )

                if exit_signal.get("should_exit"):
                    self.logger.info(f"Exit signal for #{ticket}: {exit_signal['reason']}")
                    result = self.order_executor.execute_exit(ticket, exit_signal["reason"])
                    if result.get("success"):
                        # Bug #36a fix: execute_exit doesn't return profit.
                        # Get actual P/L from position tracker before removing.
                        tracked_pos = self.position_tracker.get_position(ticket)
                        if tracked_pos and "profit" not in result:
                            result["profit"] = tracked_pos.get("profit", 0)
                        self.drawdown_monitor.record_trade_result(result)
                        self.position_tracker.remove_position(ticket, result)

                        # Use structured exit_type field from exit signal
                        exit_type = exit_signal.get("exit_type")

                        actual_profit = result.get("profit", 0)

                        # Distinguish trailing SL win from true SL loss in all notifications
                        close_comment = exit_signal["reason"]
                        if close_comment == "Stop Loss hit" and actual_profit > 0:
                            close_comment = "Trailing Stop Hit (Profit Secured)"

                        if exit_type == "RECOVERY_EXIT":
                            self.telegram.send_recovery_exit(
                                ticket=ticket,
                                profit=actual_profit,
                                reason=exit_signal["reason"],
                            )
                        elif exit_type == "SESSION_EXIT":
                            ny_check = self.session_manager.should_exit_ny_close()
                            self.telegram.send_session_exit(
                                ticket=ticket,
                                profit=actual_profit,
                                reason=exit_signal["reason"],
                                minutes_to_close=ny_check.get("minutes_until_close", 0),
                            )
                        elif exit_type == "EARLY_PROFIT":
                            self.telegram.send_early_profit_exit(
                                ticket=ticket,
                                profit=actual_profit,
                                reason=exit_signal["reason"],
                            )
                        else:
                            # Standard exit notification
                            self.telegram.send_exit(
                                direction=direction,
                                ticket=ticket,
                                entry_price=entry_price,
                                exit_price=current_price,
                                profit=actual_profit,
                                reason=close_comment,
                                lot=position.get("volume", 0.01),
                                sl=tracked_pos.get("entry_sl", 0) if tracked_pos else 0,
                                entry_time=tracked_pos.get("entry_time", "") if tracked_pos else "",
                            )

                        # Log close to CSV (tracked_pos captured before remove)
                        self._log_trade_to_csv("CLOSE", {
                            "ticket": ticket,
                            "direction": direction,
                            "price": current_price,
                            "profit": result.get("profit", ""),
                            "session": tracked_pos.get("entry_session", "") if tracked_pos else "",
                            "smc_signals": tracked_pos.get("entry_smc_signals", "") if tracked_pos else "",
                            "regime": tracked_pos.get("entry_regime", "") if tracked_pos else "",
                            "comment": close_comment,
                        })
                        # Trade Journal: log bot-initiated exit
                        self.journal.log_exit(ticket, {
                            "price": current_price,
                            "pnl_usd": actual_profit,
                            "exit_reason": close_comment,
                        })

                        # SL cooldown for bot-initiated exit with loss
                        if actual_profit < -0.50:
                            dir_key = str(direction).upper()
                            if "BUY" in dir_key:
                                dir_key = "BUY"
                            elif "SELL" in dir_key:
                                dir_key = "SELL"
                            self._sl_cooldown[dir_key] = datetime.utcnow()  # wall-clock, not Polars time
                            self.logger.info(
                                f"SL COOLDOWN: {dir_key} blocked for {self._sl_cooldown_candles} candles"
                            )
                            _cd_min = int(self._sl_cooldown_candles * 15)
                            self.telegram.send_bot_status(
                                f"\u23f8 SL COOLDOWN \u2014 {dir_key}",
                                f"\u203a {dir_key} entries blocked for {self._sl_cooldown_candles} candles ({_cd_min}min)\n"
                                f"\u203a Resumes automatically after cooldown",
                            )

        except Exception as e:
            self.logger.error(f"Error managing positions: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            self.health_monitor.record_error(e, "manage_positions")

    def _get_trade_stage(self, tracked: dict, rr_ratio: float) -> str:
        """Derive current trade stage from position tracker flags and RR ratio."""
        if not tracked:
            return "OPEN"
        if tracked.get("partial_closed"):
            return "PROFIT_LOCKED" if tracked.get("volume", 0.02) <= 0.01 else "PARTIAL_CLOSED"
        if tracked.get("pre_close_locked"):
            return "PRE_CLOSE_LOCKED"
        if tracked.get("breakeven_set"):
            return "TRAILING" if rr_ratio >= 1.5 else "BE_REACHED"
        return "OPEN"

    def _build_position_info(self, current_positions: list):
        """Build position info dict for telegram display."""
        if not current_positions:
            return None
        pos = current_positions[0]  # Max 1 position
        ticket = pos.get("ticket", 0)
        tracker_data = self.position_tracker.positions.get(ticket, {})
        entry_sl = tracker_data.get("entry_sl", pos.get("sl", 0))
        entry_price = pos.get("open_price", pos.get("price_open", 0))
        current_price = pos.get("current_price", pos.get("price_current", 0))
        sl_dist = abs(entry_price - entry_sl) if entry_sl and entry_price else 0
        direction = str(pos.get("type", "BUY")).upper()
        if "BUY" in direction:
            profit_dist = current_price - entry_price
        else:
            profit_dist = entry_price - current_price
        rr_current = profit_dist / sl_dist if sl_dist > 0 else 0

        stage = "OPEN"
        if tracker_data.get("partial_closed"):
            stage = "PARTIAL"
        elif tracker_data.get("pre_close_locked"):
            stage = "LOCKED"
        elif tracker_data.get("breakeven_set"):
            stage = "TRAILING" if rr_current >= 1.5 else "BE"

        dir_label = "BUY" if "BUY" in direction else "SELL"
        return {
            "ticket": ticket,
            "direction": dir_label,
            "entry_price": entry_price,
            "current_price": current_price,
            "profit": pos.get("profit", 0),
            "sl": pos.get("sl", 0),
            "tp": pos.get("tp", 0),
            "rr_current": rr_current,
            "stage": stage,
        }

    def _process_new_signals(self, market_data: Dict, current_positions: list, account_info: Dict):
        """Process new trading signals. Returns gate result dict or None."""
        try:
            # Generate signal (matches backtest pipeline)
            regime_result = market_data.get("regime_result", {})
            current_regime = regime_result.get("regime", MarketRegime.RANGE_WIDE)
            decision = self.strategy.analyze_and_signal(
                market_data["current_price"],
                market_data["smc_analysis"],
                market_data["technical_indicators"],
                market_data["market_analysis"],
                market_data["mtf_analysis"],
                market_data["confluence_scores"],
                current_positions,
                account_info,
                market_data["market_data"],
                regime=current_regime,
            )

            self.health_monitor.record_signal(decision.get("has_entry", False))

            # Log decision summary (only log "no entry" at debug level to reduce noise)
            summary = self.strategy.get_decision_summary(decision)
            if decision.get("has_entry"):
                self.logger.info(f"SIGNAL FOUND: {summary}")
            else:
                self.logger.info(f"Signal check: {summary}")

            # If no entry signal, return gate rejection info (shown in combined telegram)
            if not decision.get("has_entry"):
                entry_sig = decision.get("entry_signal")
                if entry_sig:
                    gate_reasons = entry_sig.get("reasons", ["No conditions met"])
                    gate_msg = gate_reasons[0] if gate_reasons else "Unknown"
                    return {"passed": False, "reason": gate_msg}
                return {"passed": False, "reason": "No conditions met"}

            entry_signal = decision["entry_signal"]

            # ── Dynamic Entry Gate: classify tier ────────────────────────────────
            sig_dir_for_tier = str(entry_signal.get("direction", "BUY")).upper()
            tier_key = "bullish" if "BUY" in sig_dir_for_tier else "bearish"
            raw_smc = market_data["smc_analysis"].get(
                "bullish" if "BUY" in sig_dir_for_tier else "bearish", {}
            )
            score_for_tier = market_data["confluence_scores"].get(
                tier_key, {}
            )
            atr_for_tier = market_data["technical_indicators"].get("atr", 1.0)
            # recent_bar_ranges: use last 3 bars from df_m15 if available
            recent_ranges = []
            df_m15 = market_data.get("df_m15")
            if df_m15 is not None and len(df_m15) >= 3:
                try:
                    for j in range(-3, 0):
                        h = float(df_m15["high"][j])
                        l = float(df_m15["low"][j])
                        recent_ranges.append(h - l)
                except Exception:
                    pass
            entry_quality = self.entry_quality_engine.classify(
                score_result=score_for_tier,
                smc_signals=raw_smc,
                technical={"atr": atr_for_tier, "recent_bar_ranges": recent_ranges},
            )
            entry_signal["quality_tier"] = entry_quality.tier.value
            entry_signal["quality_score"] = entry_quality.score
            entry_signal["has_zone"] = entry_quality.has_zone
            entry_signal["quality_label"] = entry_quality.tier.label
            self.logger.info(
                f"Entry quality: {entry_quality.tier.label} | score={entry_quality.score:.3f} | "
                f"zone={'YES' if entry_quality.has_zone else 'NO'} | "
                f"disp={entry_quality.displacement_strength:.2f}x | "
                f"reasons: {', '.join(entry_quality.reasons)}"
            )
            # ────────────────────────────────────────────────────────────────────

            # SL-hit directional cooldown check (wall-clock based — reliable Python datetime)
            sig_dir = str(entry_signal.get("direction", "")).upper()
            if sig_dir in self._sl_cooldown:
                cooldown_start = self._sl_cooldown[sig_dir]
                if cooldown_start:
                    elapsed = (datetime.utcnow() - cooldown_start).total_seconds()
                    candles_passed = elapsed / 900  # 15-min candles
                    if candles_passed < self._sl_cooldown_candles:
                        remaining = self._sl_cooldown_candles - candles_passed
                        self.logger.info(
                            f"SL COOLDOWN ACTIVE: {sig_dir} blocked ({remaining:.1f} candles remaining)"
                        )
                        return {"passed": False, "quality_tier": entry_quality.tier.label, "reason": f"BULL: SL cooldown {sig_dir} ({remaining:.0f}c) | BEAR: SL cooldown {sig_dir} ({remaining:.0f}c)"}
                    else:
                        # Cooldown expired, remove
                        del self._sl_cooldown[sig_dir]

            # Check max positions per direction (e.g. max 2 BUY, max 2 SELL)
            same_dir_positions = [
                p for p in current_positions
                if str(p.get("type", "")).upper() == sig_dir
            ]
            if len(same_dir_positions) >= self._max_per_direction:
                self.logger.info(
                    f"DIR LIMIT: {sig_dir} blocked — {len(same_dir_positions)}/{self._max_per_direction} "
                    f"already open in this direction"
                )
                return {"passed": False, "quality_tier": entry_quality.tier.label, "reason": f"BULL: Direction limit ({sig_dir}) | BEAR: Direction limit ({sig_dir})"}

            # Check position spacing (prevent entries at same price level)
            entry_price_check = entry_signal["price"]
            for p in same_dir_positions:
                existing_price = p.get("open_price", p.get("price_open", 0))
                if existing_price > 0:
                    distance = abs(entry_price_check - existing_price)
                    if distance < self._min_spacing_pips:
                        self.logger.info(
                            f"SPACING BLOCK: {sig_dir} @ {entry_price_check:.2f} too close to "
                            f"#{p.get('ticket')} @ {existing_price:.2f} "
                            f"({distance:.1f} pips, min {self._min_spacing_pips})"
                        )
                        return {"passed": False, "quality_tier": entry_quality.tier.label, "reason": f"BULL: Spacing too close #{p.get('ticket')} ({distance:.1f}pts) | BEAR: Spacing block"}

            # Calculate SL/TP
            vol_level = market_data["volatility_analysis"].get("level")
            if self.use_adaptive_scorer:
                regime_result = market_data.get("regime_result", {})
                current_regime = regime_result.get("regime", MarketRegime.RANGE_WIDE)
                session_info_sltp = self.session_manager.get_current_session()
                session_key = session_info_sltp.get("key", "default") if session_info_sltp else "default"
                # Get swing points for structure-aware SL/TP (matches backtest)
                df_m15_sltp = market_data.get("df_m15")
                swing_pts = None
                if df_m15_sltp is not None:
                    try:
                        swing_pts = self.smc_indicators.structure.get_swing_points(df_m15_sltp, n=5)
                    except Exception:
                        pass
                sltp = self.structure_sltp.calculate_sl_tp(
                    entry_signal["price"],
                    entry_signal["direction"],
                    market_data["technical_indicators"]["atr"],
                    vol_level,
                    regime=current_regime,
                    swing_points=swing_pts,
                    session=session_key,
                )
            else:
                sltp = self.sltp_calculator.calculate_sl_tp(
                    entry_signal["price"],
                    entry_signal["direction"],
                    market_data["technical_indicators"]["atr"],
                    vol_level
                )

            # Micro account safety validation
            balance = (account_info or {}).get("balance", 0)
            if self.micro_account.is_micro_account(balance):
                spread = market_data["market_data"].get("spread", 0.25)
                consec_losses = getattr(self.drawdown_monitor, "consecutive_losses", 0)
                micro_check = self.micro_account.validate_trade(
                    balance=balance,
                    sl_distance=sltp["sl_distance_pips"],
                    spread=spread,
                    open_positions=len(current_positions),
                    consecutive_losses=consec_losses,
                )
                if not micro_check.get("approved"):
                    reasons = micro_check.get('reasons', [])
                    self.logger.info(f"Micro account rejected: {reasons}")
                    reason_str = reasons[0] if reasons else "Micro account limit"
                    return {"passed": False, "quality_tier": entry_quality.tier.label, "reason": f"BULL: Micro acct — {reason_str} | BEAR: Micro acct — {reason_str}"}

            # Calculate position size
            vol_value = vol_level.value if hasattr(vol_level, "value") else str(vol_level)
            size_info = self.position_sizer.calculate_position_size(
                account_info,
                sltp["sl_distance_pips"],
                market_data["market_analysis"],
                vol_value
            )

            # Execute entry
            self.logger.info(
                f"EXECUTING: {entry_signal['direction']} @ {entry_signal['price']:.2f} "
                f"| Confidence: {entry_signal['confidence']:.2f} "
                f"| SL: {sltp['sl']:.2f} | TP: {sltp['tp']:.2f} "
                f"| Lot: {size_info['lot_size']}"
            )

            # Build informative MT5 comment from SMC signals + confidence + tier
            smc = entry_signal.get("smc_context", {})
            smc_parts = []
            if smc.get("has_choch"):
                smc_parts.append("CH")
            elif smc.get("has_bos"):
                smc_parts.append("BS")
            if smc.get("in_fvg"):
                smc_parts.append("FG")
            if smc.get("at_order_block"):
                smc_parts.append("OB")
            if smc.get("liquidity_swept"):
                smc_parts.append("LS")
            smc_str = "|".join(smc_parts) if smc_parts else "~"
            quality_label_short = entry_signal.get("quality_tier", "MED")[0]  # H/M/L
            trade_comment = f"{smc_str}|{entry_signal['confidence']:.2f}|T{quality_label_short}"

            result = self.order_executor.execute_entry(
                entry_signal,
                size_info["lot_size"],
                sltp["sl"],
                sltp["tp"],
                comment=trade_comment
            )

            self.health_monitor.record_order(result.get("success", False))

            if result.get("success"):
                self.logger.info(f"ORDER FILLED: Ticket #{result['ticket']}")
                self.position_tracker.add_position(result)

                # Bug #54 fix: add to signal cooldown history ONLY after order fills.
                # Previously added in smc_strategy.py after validation — before
                # micro_account check — so rejected trades still triggered 75-min cooldown.
                self.strategy.validator.add_signal_to_history(entry_signal)

                # Store entry context in tracker for CSV at close time
                session_info = self.session_manager.get_current_session()
                self.position_tracker.update_position(result["ticket"], {
                    "entry_session": session_info.get("name", "") if session_info else "",
                    "entry_smc_signals": trade_comment,
                    "entry_regime": market_data.get("regime", ""),
                    "entry_sl": sltp["sl"],
                    "entry_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                })

                # Trade Journal: log full entry context
                regime_str = market_data.get("regime", "")
                account_bal = (account_info or {}).get("balance", 0)
                self.journal.log_entry(result["ticket"], {
                    "symbol": self.symbol,
                    "direction": entry_signal["direction"],
                    "volume": size_info["lot_size"],
                    "price": entry_signal["price"],
                    "sl": sltp["sl"],
                    "tp": sltp["tp"],
                    "confluence": entry_signal.get("confidence", 0),
                    "smc_signals": trade_comment,
                    "regime": regime_str,
                    "session": session_info.get("name", "") if session_info else "",
                    "balance": account_bal,
                })

                # Telegram entry notification
                self.telegram.send_entry(
                    direction=entry_signal["direction"],
                    price=entry_signal["price"],
                    sl=sltp["sl"],
                    tp=sltp["tp"],
                    lot=size_info["lot_size"],
                    ticket=result["ticket"],
                    confidence=entry_signal.get("confidence", 0),
                    smc_signals=trade_comment,
                    session=session_info.get("name", "") if session_info else "",
                    regime=market_data.get("regime", ""),
                    quality_tier=entry_signal.get("quality_label", "[B:MED]"),
                )

                # Log to trade CSV
                self._log_trade_to_csv("OPEN", {
                    "ticket": result.get("ticket"),
                    "direction": entry_signal["direction"],
                    "volume": size_info["lot_size"],
                    "price": entry_signal["price"],
                    "sl": sltp["sl"],
                    "tp": sltp["tp"],
                    "confluence": round(entry_signal.get("confidence", 0), 4),
                    "smc_signals": trade_comment,
                    "session": session_info.get("name", "") if session_info else "",
                    "regime": market_data.get("regime", ""),
                    "comment": trade_comment,
                })
                return {"passed": True, "quality_tier": entry_quality.tier.label, "reason": "All gates passed \u2014 entry executed"}
            else:
                self.logger.warning(f"ORDER FAILED: {result.get('error')}")
                self.telegram.send_bot_status(
                    "ORDER FAILED",
                    f"\u274c {entry_signal['direction']} @ {entry_signal['price']:.2f}\n"
                    f"\u2514 Error: {result.get('error', 'Unknown')}\n"
                    f"\u2514 Score: {entry_signal.get('confidence', 0):.2f} | {trade_comment}"
                )
                return {"passed": False, "quality_tier": entry_quality.tier.label, "reason": f"BULL: Order failed ({result.get('error', 'Unknown')}) | BEAR: Order failed"}

        except Exception as e:
            self.logger.error(f"Error processing signals: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            self.health_monitor.record_error(e, "process_signals")
            return {"passed": False, "reason": f"BULL: Signal error ({type(e).__name__}) | BEAR: Signal error"}

    def get_status(self) -> Dict:
        """
        Get current bot status.

        Returns:
            Status dictionary
        """
        return {
            "running": self.running,
            "health": self.health_monitor.check_health(),
            "statistics": self.health_monitor.get_statistics(),
            "positions": self.position_tracker.get_position_count(),
            "session": self.session_manager.get_current_session(),
        }

    # ─── Colored Console Analysis ─────────────────────────────────────────

    def _print_colored_analysis(
        self, now, wib_now, session_name, price,
        tech, ema20, ema50, regime, regime_result,
        bull_smc, bear_smc, bull_conf, bear_conf,
        threshold, mtf, market, trend, h1_bias,
    ):
        """Print colorized M15 analysis to console (not to log file)."""
        # ANSI
        RS = "\033[0m";  BD = "\033[1m";  DM = "\033[2m"
        RD = "\033[31m"; GR = "\033[32m"; YL = "\033[33m"; CY = "\033[36m"
        GY = "\033[90m"; BR = "\033[91m"; BG = "\033[92m"; BY = "\033[93m"
        BC = "\033[96m"; BW = "\033[97m"; BM = "\033[95m"

        def on_off(val, label):
            return f"{BG}{label}{RS}" if val else f"{GY}{label}{RS}"

        def sc(val, thr):
            return BG if val >= thr else BR

        rsi = tech["rsi"]
        atr = tech["atr"]
        macd_h = (tech.get("macd") or {}).get("histogram") or 0.0
        ema_dir = "bearish" if ema20 < ema50 else "bullish"
        ema_clr = BR if ema20 < ema50 else BG

        # RSI color
        if rsi < 30: rsi_c = BR
        elif rsi < 40: rsi_c = RD
        elif rsi > 70: rsi_c = BG
        elif rsi > 60: rsi_c = GR
        else: rsi_c = YL

        # MACD color
        macd_c = BG if macd_h > 0 else BR

        # Regime color
        if "DOWN" in regime.value or "BEARISH" in regime.value:
            reg_c = BR
        elif "UP" in regime.value or "BULLISH" in regime.value:
            reg_c = BG
        else:
            reg_c = BY

        # Bull/Bear SMC active signals
        def smc_active(smc):
            parts = []
            if smc["structure"]["choch"]: parts.append("CHoCH")
            if smc["structure"]["bos"]: parts.append("BOS")
            if smc["fvg"]["in_zone"]: parts.append("FVG")
            if smc["order_block"]["at_zone"]: parts.append("OB")
            if smc["liquidity"]["swept"]: parts.append("Liq")
            return parts

        bull_active = smc_active(bull_smc)
        bear_active = smc_active(bear_smc)
        bull_str = " ".join(f"{BG}{s}{RS}" for s in bull_active) if bull_active else f"{GY}--{RS}"
        bear_str = " ".join(f"{BR}{s}{RS}" for s in bear_active) if bear_active else f"{GY}--{RS}"

        # Confluence breakdown
        def conf_line(conf, label, clr, thr):
            s = conf["score"]
            brk = conf.get("breakdown", {})
            smc_t = brk.get("smc", {}).get("total", 0)
            tech_t = brk.get("technical", {}).get("total", 0)
            ltf_t = brk.get("bonus", {}).get("details", {}).get("ltf_confirmation", 0)
            adj = brk.get("adjustments", {}).get("details", {})
            ct = adj.get("counter_trend", 0)
            sc_c = BG if round(s, 2) >= thr else BR
            tag = f"{BG}PASS{RS}" if round(s, 2) >= thr else f"{BR}FAIL{RS}"
            parts = f"smc:{smc_t:.2f} + tech:{tech_t:.2f} + ltf:{ltf_t:.2f}"
            if ct != 0:
                parts += f" {BR}ct:{ct:.2f}{RS}"
            return f"  {clr}{BD}{label}{RS} {sc_c}{BD}{s:.2f}{RS}/{GY}{thr:.2f}{RS} {tag}  {DM}({parts}){RS}"

        # MTF / Trend / H1
        mtf_al = mtf.get("is_aligned", False)
        trend_dir = trend.get("direction", "NEUTRAL")
        if hasattr(trend_dir, "value"):
            trend_dir = trend_dir.value
        trend_c = BG if "BULL" in str(trend_dir) else BR if "BEAR" in str(trend_dir) else YL

        w = 64
        line = f"{GY}{'-' * w}{RS}"
        dline = f"{CY}{'=' * w}{RS}"

        print(f"\n{dline}")
        print(f"  {BD}{BW}M15 ANALYSIS{RS}  {CY}{now.strftime('%H:%M')} UTC / {wib_now.strftime('%H:%M')} WIB{RS}  {BY}{session_name}{RS}")
        print(line)
        print(f"  {BD}{BW}{price:.2f}{RS}  {GY}ATR{RS} {atr:.2f}  {GY}RSI{RS} {rsi_c}{rsi:.1f}{RS}  {GY}MACD{RS} {macd_c}{macd_h:+.2f}{RS}")
        print(f"  {GY}EMA20{RS} {ema20:.2f}  {GY}EMA50{RS} {ema50:.2f}  {ema_clr}[{ema_dir}]{RS}")
        print(f"  {reg_c}{BD}{regime.value}{RS} {GY}({regime_result.get('confidence', 0):.2f}){RS}  {GY}Adaptive{RS} {'ON' if self.use_adaptive_scorer else 'OFF'}")
        print(line)
        print(f"  {GR}BULL SMC{RS} {bull_str}  {GY}raw{RS} {bull_smc['confluence_score']:.2f}")
        print(f"  {RD}BEAR SMC{RS} {bear_str}  {GY}raw{RS} {bear_smc['confluence_score']:.2f}")
        print(line)
        print(conf_line(bull_conf, "BULL", GR, threshold))
        print(conf_line(bear_conf, "BEAR", RD, threshold))
        mtf_s = f"{BG}YES{RS}" if mtf_al else f"{GY}no{RS}"
        print(f"  {GY}MTF{RS} {mtf_s}  {GY}Trend{RS} {trend_c}{trend_dir}{RS}  {GY}H1{RS} {h1_bias.upper()}")
        print(dline)

    # ─── Tick Capture ────────────────────────────────────────────────────────

    def _capture_tick(self) -> None:
        """Capture tick data every 2 minutes to market log."""
        now = datetime.utcnow()
        if self.last_tick_capture and (now - self.last_tick_capture).total_seconds() < self.tick_capture_interval:
            return

        try:
            tick = self.mt5.get_tick(self.symbol)
            if not tick:
                return

            self.last_tick_capture = now
            wib_now = datetime.now(self.wib_tz)
            spread = tick.get("ask", 0) - tick.get("bid", 0)

            self.logger.bind(market=True).info(
                f"TICK | {now.strftime('%H:%M:%S')} UTC / {wib_now.strftime('%H:%M:%S')} WIB | "
                f"Bid: {tick['bid']:.2f} | Ask: {tick['ask']:.2f} | "
                f"Spread: {spread:.2f} | Vol: {tick.get('volume', 0)}"
            )
        except Exception as e:
            self.logger.debug(f"Tick capture error: {e}")

    # ─── Trade CSV Logger ────────────────────────────────────────────────────

    def _log_trade_to_csv(self, trade_type: str, trade_data: Dict) -> None:
        """
        Log trade event to CSV file with dual timestamps.

        Args:
            trade_type: 'OPEN', 'CLOSE', 'MODIFY'
            trade_data: Trade details dictionary
        """
        try:
            now_utc = datetime.now(timezone.utc)
            now_wib = datetime.now(self.wib_tz)
            date_str = now_utc.strftime("%Y-%m-%d")
            csv_path = self.trade_history_dir / f"trades_{date_str}.csv"

            # Check if file exists to write header
            write_header = not csv_path.exists()

            row = {
                "date": date_str,
                "time_utc": now_utc.strftime("%H:%M:%S"),
                "time_wib": now_wib.strftime("%H:%M:%S"),
                "type": trade_type,
                "ticket": trade_data.get("ticket", ""),
                "direction": trade_data.get("direction", ""),
                "symbol": trade_data.get("symbol", self.symbol),
                "volume": trade_data.get("volume", ""),
                "price": trade_data.get("price", ""),
                "sl": trade_data.get("sl", ""),
                "tp": trade_data.get("tp", ""),
                "profit": trade_data.get("profit", ""),
                "confluence": trade_data.get("confluence", ""),
                "smc_signals": trade_data.get("smc_signals", ""),
                "session": trade_data.get("session", ""),
                "regime": trade_data.get("regime", ""),
                "comment": trade_data.get("comment", ""),
            }

            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

            self.logger.bind(trade=True).info(
                f"TRADE CSV | {trade_type} | "
                f"{now_utc.strftime('%H:%M')} UTC / {now_wib.strftime('%H:%M')} WIB | "
                f"Ticket: {trade_data.get('ticket', 'N/A')} | "
                f"Dir: {trade_data.get('direction', '')} | "
                f"Price: {trade_data.get('price', '')} | "
                f"P/L: {trade_data.get('profit', '')}"
            )
        except Exception as e:
            self.logger.error(f"Trade CSV logging error: {e}")
