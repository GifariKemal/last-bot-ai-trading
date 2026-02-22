"""
OHLCV data fetcher for MetaTrader 5 — batched edition.

MT5 brokers cap a single copy_rates_from_pos call (commonly 10k–50k bars
depending on the broker). This fetcher pulls data in configurable-size
chunks, walking backwards in time until either the target candle count
is reached or MT5 returns fewer bars than requested (history exhausted).

Returns a clean, deduplicated, UTC-indexed pandas DataFrame.
"""

from __future__ import annotations

import numpy as np
import MetaTrader5 as mt5
import pandas as pd
from loguru import logger

from config.settings import TradingConfig

# Timeframe string → MT5 constant
TIMEFRAME_MAP: dict[str, int] = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


class DataFetcher:
    """
    Fetches historical OHLCV bars from a connected MT5 terminal.

    For large requests the data is pulled in batches (config.batch_size bars
    per call), walking backwards from the most recent bar until
    config.candles bars have been collected or history is exhausted.

    Parameters
    ----------
    config : TradingConfig
        Symbol, timeframe, candle-count, and batch-size settings.
    """

    def __init__(self, config: TradingConfig) -> None:
        self.config = config

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fetch_ohlcv(self) -> pd.DataFrame:
        """
        Fetch up to config.candles bars for the configured symbol/timeframe.

        Automatically switches to batched mode when config.candles exceeds
        config.batch_size, so the caller does not need to think about limits.

        Returns
        -------
        pd.DataFrame
            Columns : open, high, low, close, tick_volume, spread, real_volume
            Index   : datetime (UTC, timezone-aware), sorted ascending

        Raises
        ------
        ValueError
            If MT5 returns no data at all on the first batch.
        """
        tf = TIMEFRAME_MAP.get(self.config.timeframe)
        if tf is None:
            raise ValueError(
                f"Unknown timeframe '{self.config.timeframe}'. "
                f"Valid options: {list(TIMEFRAME_MAP.keys())}"
            )

        logger.info(
            f"Fetching up to {self.config.candles:,} bars of "
            f"{self.config.symbol} {self.config.timeframe} "
            f"(batch size: {self.config.batch_size:,}) …"
        )

        if self.config.candles <= self.config.batch_size:
            df = self._fetch_single(tf, self.config.candles)
        else:
            df = self._fetch_batched(tf)

        logger.success(
            f"Dataset ready: {len(df):,} bars | "
            f"Range: {df.index[0].date()} → {df.index[-1].date()} "
            f"({(df.index[-1] - df.index[0]).days} days)"
        )
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_single(self, tf: int, count: int) -> pd.DataFrame:
        """One-shot fetch for small requests."""
        rates = mt5.copy_rates_from_pos(self.config.symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            raise ValueError(
                f"MT5 returned no data for {self.config.symbol}. "
                f"Error: {error}. Is the symbol in Market Watch?"
            )
        return self._to_dataframe(rates)

    def _fetch_batched(self, tf: int) -> pd.DataFrame:
        """
        Walk backwards through history in batch_size chunks.

        MT5's copy_rates_from_pos(symbol, tf, start_pos, count) uses
        start_pos as an offset from the *latest* bar (0 = latest).
        Consecutive calls with start_pos += len(previous_batch) walk
        cleanly backwards without gaps or overlaps.
        """
        batches: list[pd.DataFrame] = []
        offset: int = 0
        remaining: int = self.config.candles

        while remaining > 0:
            batch_count = min(remaining, self.config.batch_size)
            rates = mt5.copy_rates_from_pos(
                self.config.symbol, tf, offset, batch_count
            )

            if rates is None or len(rates) == 0:
                logger.warning(
                    f"MT5 returned no data at offset {offset:,}. "
                    "History limit reached — stopping early."
                )
                break

            batch_df = self._to_dataframe(rates)
            batches.append(batch_df)

            fetched = len(rates)
            offset    += fetched
            remaining -= fetched

            oldest = batch_df.index[0].date()
            logger.info(
                f"  Batch {len(batches):>2}: {fetched:,} bars | "
                f"oldest bar: {oldest} | "
                f"total so far: {offset:,}"
            )

            # Broker returned fewer bars than asked → history exhausted
            if fetched < batch_count:
                logger.info("History limit reached — no more bars available.")
                break

        if not batches:
            error = mt5.last_error()
            raise ValueError(
                f"MT5 returned no data for {self.config.symbol}. "
                f"Error: {error}. Is the symbol in Market Watch?"
            )

        # Merge, deduplicate (timestamps), sort ascending
        df = pd.concat(batches)
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def _to_dataframe(rates: np.ndarray) -> pd.DataFrame:
        """Convert raw MT5 structured array to a clean UTC DataFrame."""
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        return df
