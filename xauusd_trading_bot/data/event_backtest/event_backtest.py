"""
Event-Aware Backtest Engine
Extends the existing BacktestEngine with fundamental event bias injection.

Architecture:
  - Imports BacktestEngine from src/ (read-only, no modifications to live bot)
  - EventScorer computes gold_bias for each bar's timestamp
  - confluence_delta is added to the SMC confluence score BEFORE signal decision
  - Tier configs control which data sources are used

Tier Definitions:
  baseline  = No event data (pure SMC, same as current live bot)
  tier_a    = Holiday + Economic calendar + DXY
  tier_b    = News RSS/GDELT only
  tier_ab   = Tier A + Tier B combined
"""

import sys
import os

# Add project root to path so we can import from src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import polars as pl
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.backtesting.backtest_engine import BacktestEngine
from src.core.constants import TrendDirection
from event_scorer import EventScorer, DEFAULT_CONFIG


# ─── Tier Configurations ─────────────────────────────────────────────────────

TIER_CONFIGS = {
    'baseline': {
        'use_events':    False,
        'use_dxy':       False,
        'use_news':      False,
        'economic_weight': 0.0,
        'dxy_weight':      0.0,
        'news_weight':     0.0,
        'boost_multiplier':   0.0,
        'penalty_multiplier': 0.0,
    },
    'tier_a': {
        'use_events':    True,
        'use_dxy':       True,
        'use_news':      False,
        'economic_weight': 0.45,
        'dxy_weight':      0.30,
        'news_weight':     0.0,
        'boost_multiplier':   0.12,
        'penalty_multiplier': 0.08,
        # These are the defaults; Optuna will optimize them
    },
    'tier_b': {
        'use_events':    False,
        'use_dxy':       False,
        'use_news':      True,
        'economic_weight': 0.0,
        'dxy_weight':      0.0,
        'news_weight':     1.0,
        'boost_multiplier':   0.10,
        'penalty_multiplier': 0.06,
    },
    'tier_ab': {
        'use_events':    True,
        'use_dxy':       True,
        'use_news':      True,
        'economic_weight': 0.40,
        'dxy_weight':      0.25,
        'news_weight':     0.20,
        'boost_multiplier':   0.15,
        'penalty_multiplier': 0.10,
    },
}


class EventBacktestEngine(BacktestEngine):
    """
    Backtest engine extended with fundamental event bias injection.

    The key modification is in _simulate_trading:
    After computing bullish_confluence and bearish_confluence,
    we add confluence_delta from EventScorer before signal generation.
    """

    def __init__(self, config: Dict, tier_label: str = 'baseline', scorer_config: dict = None):
        """
        Args:
            config: Standard bot config dict
            tier_label: One of 'baseline', 'tier_a', 'tier_b', 'tier_ab'
            scorer_config: Optional override for EventScorer parameters (used by Optuna)
        """
        super().__init__(config)

        self.tier_label = tier_label
        self.tier_config = TIER_CONFIGS.get(tier_label, TIER_CONFIGS['baseline']).copy()

        # If scorer_config override provided (from Optuna), merge with tier defaults
        if scorer_config:
            self.tier_config.update(scorer_config)

        # Create event scorer (only if not baseline)
        if self.tier_config.get('use_events') or self.tier_config.get('use_news'):
            self.event_scorer = EventScorer(self.tier_config)
        else:
            self.event_scorer = None

        # Stats for reporting
        self.event_boosts   = 0  # Times confluence was boosted
        self.event_penalties = 0  # Times confluence was penalized
        self.event_blocked   = 0  # Times trade was blocked by penalty

    def _simulate_trading(self, df: pl.DataFrame, symbol: str) -> None:
        """
        Override the base simulation to inject event bias.
        Identical to parent except we modify confluence scores after calculation.
        """
        total_bars = len(df)
        signal_count = 0
        entry_count = 0

        for i in range(100, total_bars):
            current_bar  = df[i]
            current_price = current_bar["close"][0]
            current_time  = current_bar["time"][0]

            # Ensure timezone-aware datetime for EventScorer
            if isinstance(current_time, datetime):
                if current_time.tzinfo is None:
                    current_time_tz = current_time.replace(tzinfo=timezone.utc)
                else:
                    current_time_tz = current_time
            else:
                # Handle int timestamps from Polars
                current_time_tz = datetime.fromtimestamp(int(current_time) / 1e9
                    if int(current_time) > 1e12 else int(current_time), tz=timezone.utc)

            # Update open positions
            self._update_positions(current_price, current_bar)

            if not self._should_trade(current_time):
                continue

            df_slice = df[:i+1]

            # SMC signals
            bullish_smc = self.smc.get_bullish_signals(df_slice, current_price)
            bearish_smc = self.smc.get_bearish_signals(df_slice, current_price)
            smc_analysis = {"bullish": bullish_smc, "bearish": bearish_smc}

            # Technical indicators
            ema_20_val = df_slice["ema_20"][-1] if "ema_20" in df_slice.columns else current_price
            ema_50_val = df_slice["ema_50"][-1] if "ema_50" in df_slice.columns else current_price
            technical_indicators = {
                "atr":   df_slice["atr_14"][-1] if "atr_14" in df_slice.columns else 15.0,
                "rsi":   df_slice["rsi_14"][-1] if "rsi_14" in df_slice.columns else 50.0,
                "ema_20": ema_20_val,
                "ema": {20: ema_20_val, 50: ema_50_val},
                "macd": {
                    "histogram": df_slice["macd_histogram"][-1]
                    if "macd_histogram" in df_slice.columns else None,
                },
            }

            # Market + trend analysis
            market_analysis     = self.market_analyzer.analyze(df_slice)
            volatility_analysis = self.volatility_analyzer.analyze(df_slice)
            market_analysis["volatility"] = volatility_analysis
            trend_analysis = self.trend_analyzer.analyze(df_slice)
            mtf_data       = {"M15": df_slice}
            mtf_analysis   = self.mtf_analyzer.analyze(mtf_data)

            # Base confluence scores
            bullish_confluence = self.confluence_scorer.calculate_score(
                TrendDirection.BULLISH, bullish_smc,
                technical_indicators, market_analysis, mtf_analysis
            )
            bearish_confluence = self.confluence_scorer.calculate_score(
                TrendDirection.BEARISH, bearish_smc,
                technical_indicators, market_analysis, mtf_analysis
            )

            # ── EVENT BIAS INJECTION ──────────────────────────────────────
            # confluence_scorer.calculate_score() returns a dict:
            #   {"score": float, "raw_score": float, "breakdown": dict, "passing": bool}
            # We modify the "score" field before passing to strategy
            if self.event_scorer is not None:
                event_ctx_buy  = self.event_scorer.score(current_time_tz, 'BUY')
                event_ctx_sell = self.event_scorer.score(current_time_tz, 'SELL')

                bull_score = bullish_confluence.get("score", 0.0)
                bear_score = bearish_confluence.get("score", 0.0)

                # Liquidity penalty: reduce confluence on low-liquidity days
                liquidity = event_ctx_buy['liquidity_factor']
                if liquidity < 0.8:
                    liquidity_penalty = (0.8 - liquidity) * 0.2
                    bull_score = max(0, bull_score - liquidity_penalty)
                    bear_score = max(0, bear_score - liquidity_penalty)

                # Directional confluence delta
                bull_delta = event_ctx_buy['confluence_delta']
                bear_delta = event_ctx_sell['confluence_delta']

                if bull_delta > 0:
                    self.event_boosts += 1
                elif bull_delta < 0:
                    self.event_penalties += 1

                bull_score = max(0.0, min(1.0, bull_score + bull_delta))
                bear_score = max(0.0, min(1.0, bear_score + bear_delta))

                # Write back modified scores
                bullish_confluence["score"] = bull_score
                bearish_confluence["score"] = bear_score
            # ─────────────────────────────────────────────────────────────

            confluence_scores = {
                "bullish": bullish_confluence,
                "bearish": bearish_confluence,
            }

            # Generate signal
            mock_account     = {"balance": self.balance, "equity": self.balance,
                                 "margin_free": self.balance}
            mock_market_data = {"bid": current_price, "ask": current_price,
                                 "spread": 0.02, "time": current_time}

            decision = self.strategy.analyze_and_signal(
                current_price, smc_analysis, technical_indicators,
                market_analysis, mtf_analysis, confluence_scores,
                self.open_positions, mock_account, mock_market_data
            )

            if decision.get("has_entry"):
                signal_count += 1
                entry_signal = decision["entry_signal"]

                vol_level_str = (
                    volatility_analysis["level"].value
                    if hasattr(volatility_analysis["level"], "value")
                    else str(volatility_analysis["level"])
                )

                if self._can_open_position(vol_level_str):
                    sltp = self.sltp_calculator.calculate_sl_tp(
                        entry_signal["price"],
                        entry_signal["direction"],
                        technical_indicators["atr"],
                        volatility_analysis["level"]
                    )
                    size_info = self.position_sizer.calculate_position_size(
                        mock_account, sltp["sl_distance_pips"],
                        market_analysis, volatility_analysis["level"].value
                    )
                    self._open_position(
                        entry_signal, size_info["lot_size"],
                        sltp["sl"], sltp["tp"], current_time
                    )
                    entry_count += 1

            if i % 500 == 0:
                progress = (i / total_bars) * 100
                self.logger.info(
                    f"[{self.tier_label}] {progress:.0f}% | "
                    f"Signals={signal_count} Entries={entry_count} "
                    f"Open={len(self.open_positions)} Balance=${self.balance:.2f}"
                )

        self.logger.info(
            f"[{self.tier_label}] Done: {entry_count} trades | "
            f"Boosts={self.event_boosts} Penalties={self.event_penalties}"
        )

    def run_event_backtest(
        self,
        mt5,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 10000.0,
        timeframe: str = "M15",
        use_cache: bool = True
    ) -> Dict:
        """
        Run event-aware backtest. Returns standard metrics + event stats.
        """
        result = self.run_backtest(
            mt5, symbol, start_date, end_date,
            initial_balance, timeframe, use_cache
        )

        if result.get("success"):
            result["tier_label"]   = self.tier_label
            result["tier_config"]  = self.tier_config
            result["event_boosts"]    = self.event_boosts
            result["event_penalties"] = self.event_penalties
            result["event_impact"]    = {
                "boosts":    self.event_boosts,
                "penalties": self.event_penalties,
                "total_adjustments": self.event_boosts + self.event_penalties,
            }

        return result


# ─── Convenience function for run_comparison.py ───────────────────────────────

def run_tier(
    config: dict,
    mt5,
    tier_label: str,
    start_date: datetime,
    end_date: datetime,
    scorer_override: dict = None,
    initial_balance: float = 10000.0,
) -> dict:
    """
    Run a single tier backtest. Returns result dict with metrics and event stats.

    Args:
        config: Bot configuration dict
        mt5: MT5 connector object
        tier_label: 'baseline', 'tier_a', 'tier_b', 'tier_ab'
        start_date: Backtest start
        end_date: Backtest end
        scorer_override: Optional Optuna-optimized scorer params
        initial_balance: Starting account balance

    Returns:
        Result dict from EventBacktestEngine.run_event_backtest()
    """
    engine = EventBacktestEngine(config, tier_label, scorer_override)
    return engine.run_event_backtest(
        mt5, 'XAUUSD', start_date, end_date,
        initial_balance=initial_balance
    )
