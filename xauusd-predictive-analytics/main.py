"""
XAUUSD Predictive Analytics — Kimi K2 Edition
==============================================

Pipeline
--------
1.  Connect to MetaTrader 5 terminal.
2.  Fetch 3 000 M15 bars (fast — no training data needed).
3.  Build regime-aware features (TA-Lib + HTF).
4.  Call Kimi K2 API with market context snapshot.
5.  Log signal, confidence, and reasoning.
"""

import sys

from loguru import logger

from config.settings import AppConfig
from core.connector import MT5Connector
from data.fetcher import DataFetcher
from models.kimi_predictor import KimiPredictor
from strategies.feature_engineer import FeatureEngineer

# ── Logger ────────────────────────────────────────────────────────────────────
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
    logger.info("  XAUUSD Predictive Analytics — Kimi K2 Edition")
    logger.info("=" * 60)

    config = AppConfig()

    if not config.kimi.api_key:
        logger.error(
            "KIMI_API_KEY is not set. "
            "Add it to your .env file and restart."
        )
        return

    # ── 1. Connect to MT5 ─────────────────────────────────────────────────────
    with MT5Connector(config.mt5) as _conn:

        logger.info(
            f"Symbol: {config.trading.symbol} | "
            f"Timeframe: {config.trading.timeframe} | "
            f"Bars to fetch: {config.trading.candles:,}"
        )

        # ── 2. Fetch OHLCV ────────────────────────────────────────────────────
        fetcher = DataFetcher(config.trading)
        raw_df  = fetcher.fetch_ohlcv()

        # ── 3. Build features ─────────────────────────────────────────────────
        fe          = FeatureEngineer()
        featured_df = fe.build_features(raw_df)

        # ── 4. Call Kimi K2 ───────────────────────────────────────────────────
        predictor = KimiPredictor(config.kimi)
        result    = predictor.predict(featured_df)

        # ── 5. Log signal output ──────────────────────────────────────────────
        logger.info("-" * 60)
        signal      = result["signal"]
        signal_lbl  = result["signal_label"]
        confidence  = result["confidence"]
        reasoning   = result["reasoning"]
        key_factors = result["key_factors"]

        if signal == 1:
            logger.success(
                f"*** BUY SIGNAL *** | "
                f"Symbol: {config.trading.symbol} | "
                f"Confidence: {confidence:.1%}"
            )
        elif signal == -1:
            logger.success(
                f"*** SELL SIGNAL *** | "
                f"Symbol: {config.trading.symbol} | "
                f"Confidence: {confidence:.1%}"
            )
        else:
            logger.info(
                f"No trade signal | "
                f"Confidence: {confidence:.1%}"
            )

        logger.info(f"Reasoning  : {reasoning}")
        if key_factors:
            logger.info(f"Key factors: {' | '.join(key_factors)}")

        logger.info("-" * 60)

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
