"""
Backtest Engine V3
Simulates strategy execution on historical data with regime detection,
adaptive scoring, structure SL/TP, micro account safety, and configurable exits.
"""

import polars as pl
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from ..bot_logger import get_logger
from .historical_data import HistoricalDataManager
from .performance_metrics import PerformanceMetrics
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
)
from ..strategy.smc_strategy import SMCStrategy
from ..risk_management import SLTPCalculator, PositionSizer, MicroAccountManager
from ..risk_management.structure_sl_tp import StructureSLTPCalculator
from ..sessions import SessionManager
from ..core.constants import TrendDirection, MarketRegime
from ..core.data_manager import DataManager
from ..strategy.exit_signals import STRUCTURE_EXIT_MIN_RR
from ..strategy.entry_quality import EntryQualityEngine, EntryTier


# Realistic spread model by session (UTC hours)
SESSION_SPREAD_MODEL = {
    # (start_hour, end_hour): spread in price units
    (0, 8):   0.40,   # Asian
    (8, 13):  0.15,   # London
    (13, 17): 0.10,   # Overlap
    (17, 22): 0.25,   # New York afternoon
    (22, 24): 0.40,   # Late session
}


def _get_spread_for_time(ts) -> float:
    """Get realistic spread based on time of day."""
    try:
        hour = ts.hour if hasattr(ts, "hour") else 12
    except Exception:
        return 0.25
    for (start, end), spread in SESSION_SPREAD_MODEL.items():
        if start <= hour < end:
            return spread
    return 0.25


class BacktestEngine:
    """
    Backtest trading strategy on historical data.
    V3: regime detection, adaptive scoring, structure SL/TP, micro account.
    """

    def __init__(self, config: Dict):
        self.logger = get_logger()
        self.config = config

        # Components
        self.data_manager = HistoricalDataManager()
        self.metrics_calculator = PerformanceMetrics()
        self.price_data_manager = DataManager()

        # Indicators
        self.technical = TechnicalIndicators(config.get("indicators", {}))
        if config.get("use_smc_v4", False):
            from ..indicators.smc_v4_adapter import SMCIndicatorsV4
            self.smc = SMCIndicatorsV4(config.get("smc_indicators", {}))
            self.logger.info("V4 library-based SMC detection ENABLED")
        else:
            self.smc = SMCIndicators(config.get("smc_indicators", {}))

        # Analysis
        self.market_analyzer = MarketAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.mtf_analyzer = MTFAnalyzer()

        # V3: Regime detector + adaptive scorer
        self.regime_detector = RegimeDetector(config.get("regime_detection", {}))
        self.use_adaptive_scorer = config.get("use_adaptive_scorer", True)
        if self.use_adaptive_scorer:
            self.confluence_scorer = AdaptiveConfluenceScorer(config)
        else:
            self.confluence_scorer = ConfluenceScorer(config)

        # Dynamic Entry Gate (shared with live bot)
        self.entry_quality_engine = EntryQualityEngine()

        # Strategy
        self.strategy = SMCStrategy(config)

        # Risk management
        # NOTE: risk_config.yaml has NO "risk:" wrapper — keys are top-level.
        # Pass `config` directly so each manager finds its key (stop_loss, position_sizing,
        # micro_account, etc.) at the top level, matching how the live bot reads risk_config.yaml.
        risk_cfg = config  # Full config — top-level keys match risk_config.yaml structure
        self.sltp_calculator = SLTPCalculator(risk_cfg)
        self.structure_sltp = StructureSLTPCalculator(risk_cfg)
        self.use_structure_sltp = risk_cfg.get("structure_sl_tp", {}).get("enabled", True)
        self.position_sizer = PositionSizer(risk_cfg)

        # V3: Micro account manager (reads micro_account key from config)
        self.micro_manager = MicroAccountManager(risk_cfg)

        # Position direction limits (mirrors live bot logic)
        _pos_lim = risk_cfg.get("position_limits", {})
        self._max_per_direction = _pos_lim.get("max_positions_per_direction", 2)
        self._min_spacing_pips = _pos_lim.get("min_position_distance", 20.0)

        # Session management
        self.session_manager = SessionManager(config.get("session", {}))

        # State
        self.trades = []
        self.open_positions = []
        self.balance = 0
        self.initial_balance = 0
        self.df_m5 = None
        self.consecutive_losses = 0

        # V3: Regime tracking for metrics
        self._regime_per_trade = []

        # Dynamic Entry Gate: tier tracking per trade
        self._tier_per_trade: List[str] = []  # "HIGH" / "MEDIUM" / "LOW" per trade

        # Structure exit config (None = uses STRUCTURE_EXIT_MIN_RR defaults, backward compatible)
        self._structure_exit_config = config.get("structure_exit", None)

        # --- PHASE 1 SYNC: Live bot parity ---
        # Exit stage thresholds: read directly from top-level (no "risk:" wrapper)
        _exit_cfg = config.get("exit_stages", {})
        self._be_trigger_rr    = _exit_cfg.get("be_trigger_rr",    0.77)
        self._partial_close_rr = _exit_cfg.get("partial_close_rr", 2.73)
        self._trail_activation = _exit_cfg.get("trail_activation_rr", 2.72)

        # Stale trade exit: read directly from top-level
        _stale_cfg = config.get("stale_trade", {})
        self._stale_enabled  = _stale_cfg.get("enabled", False)
        self._stale_min_hours = _stale_cfg.get("min_hours", 3)
        self._stale_min_peak = _stale_cfg.get("min_peak_rr", 0.3)

        # Account protection: consecutive loss pause
        _ap_cfg = config.get("account_protection", {})
        self._max_consecutive_losses  = _ap_cfg.get("max_consecutive_losses", 3)
        self._pause_bars_on_loss      = 4   # 60 min / 15 min per bar = 4 bars
        self._min_win_to_reset_streak = _ap_cfg.get("min_win_to_reset_streak_usd", 5.0)
        self._pause_until_bar         = 0   # bar index until which new entries are paused

        # Time-based exit (matches trading_rules.yaml time_exit_hours=48)
        self._time_exit_hours = config.get("strategy", {}).get("exit", {}).get("time_exit_hours", 48)

    def run_backtest(
        self,
        mt5,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 10000.0,
        timeframe: str = "M15",
        use_cache: bool = True
    ) -> Dict:
        """Run backtest on historical data."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING BACKTEST V3")
        self.logger.info("=" * 80)
        self.logger.info(f"Symbol: {symbol}")
        self.logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        self.logger.info(f"Timeframe: {timeframe}")
        self.logger.info(f"Initial Balance: ${initial_balance:.2f}")
        self.logger.info(f"Adaptive Scorer: {self.use_adaptive_scorer}")
        self.logger.info(f"Structure SL/TP: {self.use_structure_sltp}")
        self.logger.info("=" * 80)

        # Initialize state
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.trades = []
        self.open_positions = []
        self.consecutive_losses = 0
        self._regime_per_trade = []
        self._pause_until_bar = 0
        # Reset RSI bounce history (Phase 1 sync)
        self.strategy.entry_generator._recent_rsi_values = []

        # Fetch and prepare data
        df = self._prepare_data(mt5, symbol, timeframe, start_date, end_date, use_cache)
        if df is None:
            return {"success": False, "error": "Insufficient data"}

        # Simulate trading
        self.logger.info("\nSimulating trades...")
        self._simulate_trading(df, symbol)

        # Close remaining positions
        if self.open_positions:
            self.logger.info(f"\nClosing {len(self.open_positions)} remaining positions")
            for pos in self.open_positions[:]:
                self._close_position(pos, df["close"][-1], "Backtest end")

        # Calculate metrics with regime breakdown
        self.logger.info("\nCalculating performance metrics...")
        metrics = self.metrics_calculator.calculate_all_metrics(
            self.trades, self.initial_balance, self.balance, start_date, end_date
        )

        # Add regime breakdown
        if self._regime_per_trade:
            metrics["regime_breakdown"] = self.metrics_calculator.regime_breakdown_metrics(
                self.trades, self._regime_per_trade
            )

        report = self.metrics_calculator.generate_report(metrics)
        self.logger.info("\n" + report)

        return {
            "success": True,
            "metrics": metrics,
            "trades": self.trades,
            "report": report,
        }

    def run_backtest_fast(
        self,
        df_prepared: pl.DataFrame,
        initial_balance: float = 10000.0,
        df_m5: Optional[pl.DataFrame] = None,
    ) -> Dict:
        """
        Fast backtest using pre-calculated indicator DataFrame.
        Skips data fetching and indicator calculation (done once by optimizer).

        Args:
            df_prepared: DataFrame with ALL indicators pre-calculated.
            initial_balance: Starting balance.
            df_m5: Optional pre-calculated M5 data.

        Returns:
            Backtest results dict.
        """
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.trades = []
        self.open_positions = []
        self.consecutive_losses = 0
        self._regime_per_trade = []
        self._tier_per_trade = []
        self.df_m5 = df_m5
        self._pause_until_bar = 0
        # Reset RSI bounce history (Phase 1 sync)
        self.strategy.entry_generator._recent_rsi_values = []

        # Initialize SMC internal state (swing points, structure cache)
        # Without this, get_bullish/bearish_signals returns empty results
        self.smc.calculate_all(df_prepared)

        self._simulate_trading(df_prepared, "XAUUSDm")

        # Close remaining
        if self.open_positions:
            for pos in self.open_positions[:]:
                self._close_position(pos, df_prepared["close"][-1], "Backtest end")

        start_date = df_prepared["time"][0]
        end_date = df_prepared["time"][-1]

        metrics = self.metrics_calculator.calculate_all_metrics(
            self.trades, self.initial_balance, self.balance, start_date, end_date
        )

        if self._regime_per_trade:
            metrics["regime_breakdown"] = self.metrics_calculator.regime_breakdown_metrics(
                self.trades, self._regime_per_trade
            )

        # Dynamic Entry Gate: tier breakdown metrics
        if self.trades:
            metrics["tier_breakdown"] = self._compute_tier_breakdown(self.trades)

        return {
            "success": True,
            "metrics": metrics,
            "trades": self.trades,
        }

    def _compute_tier_breakdown(self, trades: List[Dict]) -> Dict:
        """Compute per-tier performance metrics for Dynamic Entry Gate analysis."""
        from collections import defaultdict

        buckets: Dict[str, List[Dict]] = defaultdict(list)
        for t in trades:
            tier = t.get("quality_tier", "MEDIUM")
            buckets[tier].append(t)

        result = {}
        for tier_val in ["HIGH", "MEDIUM", "LOW"]:
            group = buckets.get(tier_val, [])
            if not group:
                result[tier_val] = {"count": 0}
                continue

            profits = [t["profit"] for t in group]
            wins = [p for p in profits if p > 0]
            losses = [p for p in profits if p <= 0]
            total_win = sum(wins)
            total_loss = abs(sum(losses)) or 0.001
            pf = total_win / total_loss if total_loss > 0 else float("inf")
            wr = len(wins) / len(group) * 100

            # Average RR (relative to SL distance)
            rr_values = []
            for t in group:
                ep = t.get("entry_price", 0)
                xp = t.get("exit_price", 0)
                sl = t.get("sl", 0)
                sl_dist = abs(ep - sl)
                if sl_dist > 0:
                    direction = t.get("direction", "BUY")
                    if "BUY" in str(direction).upper():
                        rr = (xp - ep) / sl_dist
                    else:
                        rr = (ep - xp) / sl_dist
                    rr_values.append(rr)

            avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0.0
            avg_score = sum(t.get("quality_score", 0) for t in group) / len(group)
            zone_pct = sum(1 for t in group if t.get("has_zone")) / len(group) * 100

            result[tier_val] = {
                "count": len(group),
                "win_rate": round(wr, 1),
                "profit_factor": round(pf, 2),
                "avg_rr": round(avg_rr, 3),
                "avg_score": round(avg_score, 3),
                "zone_pct": round(zone_pct, 1),
                "total_profit": round(sum(profits), 2),
            }

        return result

    def _prepare_data(
        self, mt5, symbol, timeframe, start_date, end_date, use_cache
    ) -> Optional[pl.DataFrame]:
        """Fetch and prepare data with all indicators."""
        self.logger.info("Fetching historical data...")
        df = self.data_manager.prepare_backtest_data(
            mt5, symbol, timeframe, start_date, end_date, use_cache
        )

        if df is None or len(df) < 100:
            self.logger.error("Insufficient data for backtesting")
            return None

        data_info = self.data_manager.get_data_info(df)
        self.logger.info(f"Loaded {data_info['bars']} bars")

        # Add features
        df = self.price_data_manager.add_basic_features(df)
        df = self.price_data_manager.add_price_changes(df)

        # Calculate indicators
        self.logger.info("Calculating technical indicators...")
        df = self.technical.calculate_all(df)

        self.logger.info("Calculating SMC indicators...")
        df = self.smc.calculate_all(df)

        # Fetch M5 data if LTF confirmation enabled
        ltf_cfg = self.config.get("mtf_analysis", {}).get("ltf_confirmation", {})
        if ltf_cfg.get("enabled", False):
            self.logger.info("Fetching M5 data for LTF confirmation...")
            df_m5 = self.data_manager.fetch_historical_data(
                mt5, symbol, "M5", start_date, end_date, use_cache
            )
            if df_m5 is not None and len(df_m5) > 0:
                df_m5 = self.price_data_manager.add_basic_features(df_m5)
                df_m5 = self.price_data_manager.add_price_changes(df_m5)
                df_m5 = self.technical.calculate_all(df_m5)
                self.df_m5 = df_m5
                self.logger.info(f"M5 data ready: {len(df_m5)} bars")
            else:
                self.df_m5 = None

        return df

    def _simulate_trading(self, df: pl.DataFrame, symbol: str) -> None:
        """Simulate trading with regime detection and adaptive scoring."""
        total_bars = len(df)
        signal_count = 0
        entry_count = 0

        for i in range(100, total_bars):
            self._current_bar_index = i  # Phase 1 sync: track for pause logic
            current_bar = df[i]
            current_price = current_bar["close"][0]
            current_time = current_bar["time"][0]

            # Pre-compute SMC/regime for structure exit — always when positions open
            # (Phase 1 sync: structure exit runs on every bar, not just when config set)
            smc_for_exit = None
            regime_for_exit = None
            if self.open_positions:
                df_slice_exit = df[:i+1]
                regime_result_exit = self.regime_detector.detect(df_slice_exit)
                regime_for_exit = regime_result_exit["regime"]
                bullish_exit = self.smc.get_bullish_signals(df_slice_exit, current_price)
                bearish_exit = self.smc.get_bearish_signals(df_slice_exit, current_price)
                smc_for_exit = {"bullish": bullish_exit, "bearish": bearish_exit}

            # Update open positions (with structure exit data if available)
            self._update_positions(
                current_price, current_bar, bar_index=i,
                smc_analysis=smc_for_exit, regime=regime_for_exit,
            )

            # Check basic trading conditions + consecutive loss pause
            if not self._should_trade(current_time) or i < self._pause_until_bar:
                continue

            # Get data slice (reuse if already computed for exit)
            df_slice = df[:i+1]

            # V3: Detect market regime (reuse if already computed for exit)
            if regime_for_exit is not None:
                regime = regime_for_exit
            else:
                regime_result = self.regime_detector.detect(df_slice)
                regime = regime_result["regime"]

            # Get SMC signals (reuse if already computed for exit)
            if smc_for_exit is not None:
                smc_analysis = smc_for_exit
            else:
                bullish_smc = self.smc.get_bullish_signals(df_slice, current_price)
                bearish_smc = self.smc.get_bearish_signals(df_slice, current_price)
                smc_analysis = {"bullish": bullish_smc, "bearish": bearish_smc}

            # Technical indicators
            ema_20_val = df_slice["ema_20"][-1] if "ema_20" in df_slice.columns else current_price
            ema_50_val = df_slice["ema_50"][-1] if "ema_50" in df_slice.columns else current_price
            technical_indicators = {
                "atr": df_slice["atr_14"][-1] if "atr_14" in df_slice.columns else 15.0,
                "rsi": df_slice["rsi_14"][-1] if "rsi_14" in df_slice.columns else 50.0,
                "ema_20": ema_20_val,
                "ema": {20: ema_20_val, 50: ema_50_val},
                "macd": {
                    "histogram": df_slice["macd_histogram"][-1] if "macd_histogram" in df_slice.columns else None,
                },
            }

            # Market analysis
            market_analysis = self.market_analyzer.analyze(df_slice)
            volatility_analysis = self.volatility_analyzer.analyze(df_slice)
            market_analysis["volatility"] = volatility_analysis

            trend_analysis = self.trend_analyzer.analyze(df_slice)

            # MTF analysis
            mtf_data = {"M15": df_slice}
            mtf_analysis = self.mtf_analyzer.analyze(mtf_data)

            # LTF data
            ltf_data = None
            if self.df_m5 is not None:
                ltf_data = {"m5_df": self.df_m5, "current_m15_time": current_time}

            # Phase 1 sync: update RSI bounce history (persistent 5-bar window)
            rsi_val = technical_indicators.get("rsi", 50.0)
            self.strategy.entry_generator.update_rsi_history(rsi_val)

            # V3: Adaptive confluence scoring with regime
            # Phase 1 sync: pass opposing_smc for OPPOSING_CHOCH_PENALTY (-0.15)
            bullish_smc = smc_analysis["bullish"]
            bearish_smc = smc_analysis["bearish"]
            if self.use_adaptive_scorer:
                bullish_confluence = self.confluence_scorer.calculate_score(
                    TrendDirection.BULLISH, bullish_smc, technical_indicators,
                    market_analysis, mtf_analysis, regime=regime, ltf_data=ltf_data,
                    opposing_smc=bearish_smc,
                )
                bearish_confluence = self.confluence_scorer.calculate_score(
                    TrendDirection.BEARISH, bearish_smc, technical_indicators,
                    market_analysis, mtf_analysis, regime=regime, ltf_data=ltf_data,
                    opposing_smc=bullish_smc,
                )
            else:
                bullish_confluence = self.confluence_scorer.calculate_score(
                    TrendDirection.BULLISH, bullish_smc, technical_indicators,
                    market_analysis, mtf_analysis, ltf_data=ltf_data,
                )
                bearish_confluence = self.confluence_scorer.calculate_score(
                    TrendDirection.BEARISH, bearish_smc, technical_indicators,
                    market_analysis, mtf_analysis, ltf_data=ltf_data,
                )

            confluence_scores = {"bullish": bullish_confluence, "bearish": bearish_confluence}

            # ── Dynamic Entry Gate: classify quality tier ────────────────────────
            # Build technical dict with recent bar ranges for displacement strength
            recent_ranges = []
            if i >= 3:
                for j in range(max(0, i-2), i+1):
                    h = df[j]["high"][0] if hasattr(df[j]["high"], "__getitem__") else float(df["high"][j])
                    l = df[j]["low"][0] if hasattr(df[j]["low"], "__getitem__") else float(df["low"][j])
                    recent_ranges.append(h - l)
            tech_for_quality = {
                "atr": technical_indicators.get("atr", 1.0),
                "recent_bar_ranges": recent_ranges,
            }
            quality_tiers = self.entry_quality_engine.classify_both_directions(
                bull_score=bullish_confluence,
                bear_score=bearish_confluence,
                bull_smc=bullish_smc,
                bear_smc=bearish_smc,
                technical=tech_for_quality,
            )
            # ────────────────────────────────────────────────────────────────────

            # Generate signal
            mock_account = {"balance": self.balance, "equity": self.balance, "margin_free": self.balance}

            # V3: Realistic spread model
            spread = _get_spread_for_time(current_time)
            mock_market_data = {"bid": current_price, "ask": current_price + spread, "spread": spread, "time": current_time}

            # Phase 1 sync: pass regime so generate_signal uses correct min_smc_signals
            # (Bug #41 equivalent — live bot passes regime via trading_bot.py, backtest must too)
            decision = self.strategy.analyze_and_signal(
                current_price, smc_analysis, technical_indicators,
                market_analysis, mtf_analysis, confluence_scores,
                self.open_positions, mock_account, mock_market_data,
                regime=regime,
            )

            if decision.get("has_entry"):
                signal_count += 1
                entry_signal = decision["entry_signal"]

                # Attach quality tier to signal (based on direction)
                sig_dir = entry_signal.get("direction", "BUY").upper()
                tier_key = "bullish" if "BUY" in sig_dir else "bearish"
                entry_quality = quality_tiers.get(tier_key)
                entry_signal["quality_tier"] = entry_quality.tier.value if entry_quality else "MEDIUM"
                entry_signal["quality_score"] = entry_quality.score if entry_quality else 0.0
                entry_signal["has_zone"] = entry_quality.has_zone if entry_quality else False

                if self._can_open_position(entry_signal["direction"], entry_signal["price"]):
                    atr = technical_indicators["atr"]

                    # V3: Structure SL/TP with regime awareness
                    if self.use_structure_sltp:
                        # Get swing points from SMC structure
                        swing_pts = self.smc.structure.get_swing_points(df_slice, n=5)
                        session_name = self._get_session_name(current_time)

                        sltp = self.structure_sltp.calculate_sl_tp(
                            entry_signal["price"],
                            entry_signal["direction"],
                            atr,
                            volatility_analysis["level"],
                            regime=regime,
                            swing_points=swing_pts,
                            session=session_name,
                        )
                    else:
                        sltp = self.sltp_calculator.calculate_sl_tp(
                            entry_signal["price"],
                            entry_signal["direction"],
                            atr,
                            volatility_analysis["level"]
                        )

                    # V3: Micro account validation
                    if self.micro_manager.is_micro_account(self.balance):
                        validation = self.micro_manager.validate_trade(
                            self.balance,
                            sltp["sl_distance_pips"],
                            spread,
                            len(self.open_positions),
                            self.consecutive_losses,
                        )
                        if not validation["approved"]:
                            self.logger.debug(
                                f"MICRO REJECT: {', '.join(validation.get('reasons', []))}"
                            )
                            continue

                    # Position size
                    vol_level_str = volatility_analysis["level"].value if hasattr(volatility_analysis["level"], "value") else str(volatility_analysis["level"])
                    size_info = self.position_sizer.calculate_position_size(
                        mock_account, sltp["sl_distance_pips"],
                        market_analysis, vol_level_str
                    )

                    # Open position
                    self._open_position(
                        entry_signal, size_info["lot_size"],
                        sltp["sl"], sltp["tp"], current_time,
                        regime=regime, sltp_info=sltp, bar_index=i,
                    )
                    entry_count += 1

            # Progress logging
            if i % 1000 == 0:
                progress = (i / total_bars) * 100
                self.logger.info(
                    f"Progress: {progress:.1f}% | Signals: {signal_count} | "
                    f"Entries: {entry_count} | Open: {len(self.open_positions)} | "
                    f"Balance: ${self.balance:.2f} | Regime: {regime.value}"
                )

        self.logger.info(f"\nSimulation complete:")
        self.logger.info(f"  Total bars: {total_bars} | Signals: {signal_count} | Entries: {entry_count}")

    def _get_max_positions(self) -> int:
        """
        Get max open positions.
        Fixed at 3 (from config), reduced for micro accounts or drawdown.
        NOT tied to volatility or entry logic — separate concern.
        """
        max_pos = self.config.get("risk", {}).get("position_limits", {}).get(
            "max_open_positions", 3
        )

        # Micro account override
        if self.micro_manager.is_micro_account(self.balance):
            max_pos = min(max_pos, self.micro_manager.max_positions)

        # Reduce in significant drawdown
        if self.initial_balance > 0:
            dd_pct = ((self.initial_balance - self.balance) / self.initial_balance) * 100
            if dd_pct > 5:
                max_pos = max(1, max_pos - 2)
            elif dd_pct > 3:
                max_pos = max(1, max_pos - 1)

        return max_pos

    def _should_trade(self, current_time: datetime) -> bool:
        """Check if should trade at current time."""
        if len(self.open_positions) >= self._get_max_positions():
            return False
        session_check = self.session_manager.is_trading_allowed(current_time)
        if not session_check.get("allowed"):
            return False
        return True

    def _can_open_position(self, direction: str = None, entry_price: float = None) -> bool:
        """Check if can open new position (separate from entry logic)."""
        max_pos = self._get_max_positions()
        if len(self.open_positions) >= max_pos:
            return False

        if direction and entry_price:
            # Check max positions per direction
            same_dir = [p for p in self.open_positions if p["direction"] == direction]
            if len(same_dir) >= self._max_per_direction:
                return False

            # Check position spacing (prevent stacking at same price level)
            for p in same_dir:
                if abs(entry_price - p["entry_price"]) < self._min_spacing_pips:
                    return False

        return True

    def _get_session_name(self, ts) -> str:
        """Get session name from timestamp."""
        try:
            hour = ts.hour if hasattr(ts, "hour") else 12
        except Exception:
            return "default"
        if 13 <= hour < 17:
            return "overlap"
        elif 8 <= hour < 13:
            return "london"
        elif 17 <= hour < 22:
            return "new_york"
        else:
            return "asian"

    def _open_position(
        self, signal: Dict, lot_size: float, sl: float, tp: float,
        entry_time: datetime, regime: MarketRegime = MarketRegime.RANGE_WIDE,
        sltp_info: Optional[Dict] = None, bar_index: int = 0,
    ) -> None:
        """Open new position with multi-stage exit tracking."""
        entry_price = signal["price"]
        direction = signal["direction"]
        sl_distance = abs(entry_price - sl)
        quality_tier = signal.get("quality_tier", EntryTier.TIER_B.value)

        # Phase 1 sync: read exit thresholds from risk_config (not sltp_info defaults)
        # Live bot uses 0.77/dynamic/2.72 — NOT 1.0/1.5/2.0 old defaults
        be_trigger     = sltp_info.get("be_trigger_rr",      self._be_trigger_rr)    if sltp_info else self._be_trigger_rr
        trail_activation = sltp_info.get("trail_activation_rr", self._trail_activation) if sltp_info else self._trail_activation

        # Dynamic partial close: max(tp_rr * 0.65, 1.0) — matches live bot logic
        tp_rr = abs(tp - entry_price) / sl_distance if sl_distance > 0 else 2.0
        partial_close = max(tp_rr * 0.65, 1.0)

        position = {
            "ticket": len(self.trades) + len(self.open_positions) + 1,
            "direction": direction,
            "entry_price": entry_price,
            "sl": sl,
            "original_sl": sl,
            "tp": tp,
            "lot_size": lot_size,
            "original_lot_size": lot_size,
            "entry_time": entry_time,
            "confidence": signal["confidence"],
            "sl_distance": sl_distance,
            "regime": regime,
            # Structure exit tracking
            "entry_bar_index": bar_index,
            # Multi-stage exit
            "stage": 0,
            "peak_profit_price": entry_price,
            "partial_profit_taken": 0.0,
            # V3: Configurable exit thresholds
            "be_trigger_rr": be_trigger,
            "partial_close_rr": partial_close,
            "trail_activation_rr": trail_activation,
            # Dynamic Entry Gate tier
            "quality_tier": quality_tier,
            "quality_score": signal.get("quality_score", signal["confidence"]),
            "has_zone": signal.get("has_zone", False),
        }

        self.open_positions.append(position)
        self._regime_per_trade.append(regime.value)
        self._tier_per_trade.append(quality_tier)

        self.logger.debug(
            f"OPEN: {direction} @ {entry_price:.2f} | SL: {sl:.2f} | "
            f"TP: {tp:.2f} | Lot: {lot_size} | Regime: {regime.value} | "
            f"Tier: {quality_tier}"
        )

    def _update_positions(
        self, current_price: float, current_bar: Dict,
        bar_index: int = 0, smc_analysis: Optional[Dict] = None,
        regime: Optional[MarketRegime] = None,
    ) -> None:
        """Update positions with multi-stage flexible exit strategy."""
        bar_high = current_bar["high"][0]
        bar_low = current_bar["low"][0]

        for position in self.open_positions[:]:
            direction = position["direction"]
            entry_price = position["entry_price"]
            sl = position["sl"]
            tp = position["tp"]
            sl_distance = position.get("sl_distance", abs(entry_price - sl))
            stage = position.get("stage", 0)

            # V3: Configurable exit thresholds
            be_rr = position.get("be_trigger_rr", 1.0)
            partial_rr = position.get("partial_close_rr", 1.5)

            # --- CHECK SL ---
            if direction == "BUY":
                if bar_low <= sl:
                    self._close_position(position, sl, "Stop Loss")
                    continue
            else:
                if bar_high >= sl:
                    self._close_position(position, sl, "Stop Loss")
                    continue

            # Phase 1 sync: extract current_time for time/stale checks
            current_time = current_bar["time"][0]

            # --- STRUCTURE EXIT (Phase 1 sync: always on, uses STRUCTURE_EXIT_MIN_RR) ---
            # min_hold = 4 bars (60 min / 15 min per bar) — matches live bot MIN_HOLD_MINUTES=60
            MIN_HOLD_BARS = 4
            if smc_analysis:
                bars_held = bar_index - position.get("entry_bar_index", bar_index)
                if bars_held >= MIN_HOLD_BARS:
                    has_opposite_choch = False
                    if direction == "BUY":
                        has_opposite_choch = (
                            smc_analysis.get("bearish", {}).get("structure", {}).get("choch", False)
                        )
                    else:
                        has_opposite_choch = (
                            smc_analysis.get("bullish", {}).get("structure", {}).get("choch", False)
                        )

                    if has_opposite_choch:
                        if direction == "BUY":
                            pnl_fraction = (current_price - entry_price) / sl_distance if sl_distance > 0 else 0
                        else:
                            pnl_fraction = (entry_price - current_price) / sl_distance if sl_distance > 0 else 0

                        # Use STRUCTURE_EXIT_MIN_RR (Optuna-optimized) or config override
                        current_regime = regime or position.get("regime", MarketRegime.RANGE_WIDE)
                        if self._structure_exit_config:
                            min_rr = self._get_structure_exit_min_rr(current_regime)
                        else:
                            min_rr = STRUCTURE_EXIT_MIN_RR.get(current_regime, 1.1)

                        if pnl_fraction >= min_rr:
                            self._close_position(position, current_price, "Structure Exit")
                            continue

            # --- STALE TRADE EXIT (Phase 1 sync: dead-momentum close before full SL) ---
            if self._stale_enabled and not position.get("stale_exit_attempted", False):
                try:
                    entry_time = position["entry_time"]
                    if hasattr(current_time, "total_seconds"):
                        age_hours = 0.0
                    else:
                        age_hours = (current_time - entry_time).total_seconds() / 3600

                    peak_rr = (
                        abs(position.get("peak_profit_price", entry_price) - entry_price) / sl_distance
                        if sl_distance > 0 else 0
                    )
                    in_loss = (
                        (direction == "BUY" and current_price < entry_price) or
                        (direction == "SELL" and current_price > entry_price)
                    )

                    if age_hours >= self._stale_min_hours and peak_rr < self._stale_min_peak and in_loss:
                        position["stale_exit_attempted"] = True
                        self._close_position(
                            position, current_price,
                            f"Stale Exit (peak {peak_rr:.2f}R in {age_hours:.1f}h)"
                        )
                        continue
                except Exception:
                    pass  # Graceful fallback if time types don't match

            # --- TIME-BASED EXIT (Phase 1 sync: 48h, matches trading_rules.yaml) ---
            try:
                entry_time = position["entry_time"]
                age_hours_t = (current_time - entry_time).total_seconds() / 3600
                if age_hours_t > self._time_exit_hours:
                    self._close_position(position, current_price, "Time Exit (no progress)")
                    continue
            except Exception:
                pass

            # --- CURRENT PROFIT ---
            if direction == "BUY":
                current_profit_distance = current_price - entry_price
                position["peak_profit_price"] = max(position.get("peak_profit_price", entry_price), bar_high)
                peak_profit = position["peak_profit_price"] - entry_price
            else:
                current_profit_distance = entry_price - current_price
                position["peak_profit_price"] = min(position.get("peak_profit_price", entry_price), bar_low)
                peak_profit = entry_price - position["peak_profit_price"]

            rr_ratio = current_profit_distance / sl_distance if sl_distance > 0 else 0

            # --- STAGE 0 → 1: Breakeven ---
            if stage == 0 and rr_ratio >= be_rr:
                buffer = sl_distance * 0.1
                if direction == "BUY":
                    position["sl"] = entry_price + buffer
                else:
                    position["sl"] = entry_price - buffer
                position["stage"] = 1

            # --- STAGE 1 → 2: Partial close ---
            if stage == 1 and rr_ratio >= partial_rr:
                if position["lot_size"] > 0.01:
                    partial_lot = position["lot_size"] * 0.5
                    partial_profit = current_profit_distance * partial_lot * 100
                    self.balance += partial_profit
                    position["partial_profit_taken"] += partial_profit
                    position["lot_size"] -= partial_lot
                    position["stage"] = 2
                else:
                    lock_profit = current_profit_distance * 0.5
                    if direction == "BUY":
                        position["sl"] = entry_price + lock_profit
                    else:
                        position["sl"] = entry_price - lock_profit
                    position["stage"] = 2

            # --- STAGE 2+: Trailing ---
            if stage >= 2:
                trail_distance = peak_profit * 0.5
                if trail_distance > sl_distance * 0.5:
                    if direction == "BUY":
                        new_sl = position["peak_profit_price"] - trail_distance
                        if new_sl > position["sl"]:
                            position["sl"] = new_sl
                            position["stage"] = 3
                    else:
                        new_sl = position["peak_profit_price"] + trail_distance
                        if new_sl < position["sl"]:
                            position["sl"] = new_sl
                            position["stage"] = 3

            # --- CHECK TP ---
            if direction == "BUY":
                if bar_high >= tp:
                    self._close_position(position, tp, "Take Profit")
                    continue
            else:
                if bar_low <= tp:
                    self._close_position(position, tp, "Take Profit")
                    continue

    def _get_structure_exit_min_rr(self, regime: MarketRegime) -> float:
        """Get minimum RR for structure exit based on regime and config."""
        cfg = self._structure_exit_config or {}

        # Map regime to config category
        if regime in (MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN):
            return cfg.get("strong_trend_min_rr", 0.0)
        elif regime in (MarketRegime.WEAK_TREND_UP, MarketRegime.WEAK_TREND_DOWN):
            return cfg.get("weak_trend_min_rr", 0.5)
        elif regime in (MarketRegime.RANGE_TIGHT, MarketRegime.RANGE_WIDE):
            return cfg.get("range_min_rr", 1.0)
        elif regime == MarketRegime.VOLATILE_BREAKOUT:
            return cfg.get("volatile_min_rr", 0.3)
        elif regime == MarketRegime.REVERSAL:
            return cfg.get("reversal_min_rr", 0.0)
        else:
            # Conservative fallback = range
            return cfg.get("range_min_rr", 1.0)

    def _close_position(self, position: Dict, exit_price: float, reason: str) -> None:
        """Close position and record trade."""
        direction = position["direction"]
        entry_price = position["entry_price"]
        remaining_lot = position["lot_size"]

        if direction == "BUY":
            remaining_profit = (exit_price - entry_price) * remaining_lot * 100
        else:
            remaining_profit = (entry_price - exit_price) * remaining_lot * 100

        self.balance += remaining_profit

        partial_taken = position.get("partial_profit_taken", 0.0)
        total_profit = partial_taken + remaining_profit

        # Phase 1 sync: only reset streak on meaningful win (>= min_win_to_reset_streak)
        # A scratch/near-BE win does NOT reset the counter — matches drawdown_monitor.py fix
        if total_profit >= self._min_win_to_reset_streak:
            self.consecutive_losses = 0
        elif total_profit < 0:
            self.consecutive_losses += 1
            # Trigger pause if consecutive losses hit limit
            if self.consecutive_losses >= self._max_consecutive_losses:
                current_bar_idx = self.trades.__len__()  # approximation
                self._pause_until_bar = getattr(self, "_current_bar_index", 0) + self._pause_bars_on_loss

        trade = {
            "ticket": position["ticket"],
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "sl": position.get("original_sl", position["sl"]),
            "tp": position["tp"],
            "profit": total_profit,
            "remaining_profit": remaining_profit,
            "partial_profit": partial_taken,
            "reason": reason,
            "entry_time": position["entry_time"],
            "confidence": position["confidence"],
            "exit_stage": position.get("stage", 0),
            "original_lot": position.get("original_lot_size", remaining_lot),
            "regime": position.get("regime", MarketRegime.RANGE_WIDE).value
                if isinstance(position.get("regime"), MarketRegime)
                else str(position.get("regime", "")),
            # Dynamic Entry Gate
            "quality_tier": position.get("quality_tier", "MEDIUM"),
            "quality_score": position.get("quality_score", 0.0),
            "has_zone": position.get("has_zone", False),
        }

        self.trades.append(trade)
        self.open_positions.remove(position)

        self.logger.debug(
            f"CLOSE: {direction} @ {exit_price:.2f} | "
            f"Total: ${total_profit:.2f} | Reason: {reason} | "
            f"Stage: {position.get('stage', 0)}"
        )

    def save_results(self, results: Dict, output_file: str) -> None:
        """Save backtest results to file."""
        import json
        from pathlib import Path

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serializable_results = self._make_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Results saved to {output_path}")

    def _make_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, MarketRegime):
            return obj.value
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
