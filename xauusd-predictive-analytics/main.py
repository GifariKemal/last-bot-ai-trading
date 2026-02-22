"""
XAUUSD Predictive Analytics — Entry Point
==========================================

Pipeline
--------
1.  Connect to MetaTrader 5 terminal.
2.  Fetch 50 000 M15 OHLCV candles (batched, ~2 years of history).
3.  Build 40 regime-aware features (TA-Lib + session H/L proximity + calendar).
4.  Create ATR-threshold labels — bars with |move| < 0.25×ATR are dropped
    as market noise, leaving only bars with a "meaningful" directional move.
5.  Walk-forward CV (Dual-Head) → unbiased win-rate + coverage estimate.
6.  Train final Dual-Head model on ALL labelled data.
7.  Predict the live (latest complete) bar.
8.  Log FAKE ORDER if a specialist fires with > 60 % confidence.
"""

import sys

from loguru import logger

from config.settings import AppConfig
from core.connector import MT5Connector
from data.fetcher import DataFetcher
from models.catboost_model import DualCatBoostModel
from models.walk_forward import WalkForwardCV
from strategies.feature_engineer import FeatureEngineer

# ── Logger — coloured, structured output ──────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
    colorize=True,
    level="DEBUG",
)


def main() -> None:
    logger.info("=" * 60)
    logger.info("  XAUUSD Predictive Analytics — Dual-Head Edition")
    logger.info("=" * 60)

    config = AppConfig()

    # ── 1. Connect to MT5 ─────────────────────────────────────────────────────
    with MT5Connector(config.mt5) as _conn:

        logger.info(
            f"Symbol: {config.trading.symbol} | "
            f"Timeframe: {config.trading.timeframe} | "
            f"Target bars: {config.trading.candles:,}"
        )

        # ── 2. Fetch OHLCV (batched) ──────────────────────────────────────────
        fetcher = DataFetcher(config.trading)
        raw_df  = fetcher.fetch_ohlcv()

        # ── 3 & 4. Features + ATR-filtered labels ─────────────────────────────
        fe          = FeatureEngineer()
        featured_df = fe.build_features(raw_df)
        labelled_df = fe.create_labels(
            featured_df,
            atr_threshold=config.model.label_atr_threshold,
        )

        # ── 5. Walk-forward CV (Dual-Head) ────────────────────────────────────
        # Symmetric thresholds — stable baseline validated across 3 folds.
        # Optuna sell_threshold=0.694 overfit to search window (bear specialist
        # rarely exceeds 0.694 in unseen data).  Symmetric 0.51 consistently
        # produces BUY + SELL signals across all folds.
        BUY_THRESHOLD  = 0.516   # Optuna Trial #0 (MTF run, stable)
        SELL_THRESHOLD = 0.531   # Optuna Trial #0 (MTF run, stable)

        wf = WalkForwardCV(
            model_config=config.model,
            initial_train_bars=20_000,
            fold_size=5_000,
            max_folds=6,
            buy_threshold=BUY_THRESHOLD,
            sell_threshold=SELL_THRESHOLD,
        )
        wf_summary = wf.run(labelled_df)

        logger.info(
            f"Walk-forward result: "
            f"WinRate {wf_summary['accuracy_mean']:.4f} "
            f"± {wf_summary['accuracy_std']:.4f} | "
            f"Coverage {wf_summary['coverage_mean']:.1%} "
            f"over {wf_summary['n_folds']} folds"
        )

        # ── 6. Final Dual-Head model on ALL labelled data ─────────────────────
        logger.info("")
        logger.info("Training final Dual-Head model on full dataset …")
        predictor = DualCatBoostModel(
            config.model,
            buy_threshold=BUY_THRESHOLD,
            sell_threshold=SELL_THRESHOLD,
        )
        metrics   = predictor.train(labelled_df)

        logger.info(
            f"Final model: WinRate {metrics['accuracy']:.4f} | "
            f"Coverage {metrics['coverage']:.1%} | "
            f"Trades {int(metrics['n_trades']):,} / {int(metrics['test_rows']):,}"
        )

        # ── 7. Live prediction — latest bar ───────────────────────────────────
        live_df              = featured_df.dropna()
        buy_prob, sell_prob, signal = predictor.predict_latest(live_df)

        # ── 8. Signal output ──────────────────────────────────────────────────
        logger.info("-" * 60)
        if signal == 1:
            logger.success(
                f"*** FAKE BUY ORDER *** | "
                f"Symbol: {config.trading.symbol} | "
                f"P(BUY): {buy_prob:.2%} | "
                f"WF WinRate: {wf_summary['accuracy_mean']:.4f}"
            )
        elif signal == -1:
            logger.success(
                f"*** FAKE SELL ORDER *** | "
                f"Symbol: {config.trading.symbol} | "
                f"P(SELL): {sell_prob:.2%} | "
                f"WF WinRate: {wf_summary['accuracy_mean']:.4f}"
            )
        else:
            logger.info(
                f"No trade signal. "
                f"P(BUY) = {buy_prob:.2%} (need ≥{BUY_THRESHOLD:.0%}) | "
                f"P(SELL) = {sell_prob:.2%} (need ≥{SELL_THRESHOLD:.0%})"
            )
        logger.info("-" * 60)

    logger.info("Bot cycle complete.")


if __name__ == "__main__":
    main()
