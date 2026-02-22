"""
Walk-Forward Cross-Validation for time-series financial data.

Why walk-forward instead of k-fold?
------------------------------------
Standard k-fold randomly shuffles data, causing look-ahead bias — the model
trains on future bars and tests on past ones. On financial time-series this
produces falsely optimistic accuracy numbers that collapse in live trading.

Walk-forward CV strictly respects temporal order:
  Fold 1: Train [0 : 20 000]   → Test [20 000 : 25 000]
  Fold 2: Train [0 : 25 000]   → Test [25 000 : 30 000]   (expanding window)
  Fold 3: Train [0 : 30 000]   → Test [30 000 : 35 000]
  …

For the Dual-Head model, the key metrics per fold are:
  accuracy  – win rate on triggered trades only (higher bar than raw accuracy)
  coverage  – fraction of test bars where a BUY or SELL signal fires

Usage
-----
    wf = WalkForwardCV(model_config, initial_train_bars=20_000, fold_size=5_000)
    summary = wf.run(labelled_df)
    print(f"Mean win rate: {summary['accuracy_mean']:.4f}")
    print(f"Mean coverage: {summary['coverage_mean']:.2%}")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import ModelConfig
from models.catboost_model import DualCatBoostModel, FEATURE_COLUMNS


@dataclass
class FoldResult:
    """Metrics for a single walk-forward fold."""

    fold:        int
    train_start: str
    train_end:   str
    test_start:  str
    test_end:    str
    train_rows:  int
    test_rows:   int
    accuracy:    float   # win rate on triggered trades
    coverage:    float   # fraction of bars where a signal fired
    n_trades:    int
    n_buy:       int
    n_sell:      int
    bull_best_iter: int
    bear_best_iter: int


class WalkForwardCV:
    """
    Expanding-window walk-forward cross-validator for DualCatBoostModel.

    Parameters
    ----------
    model_config : ModelConfig
        Shared model hyper-parameters (random seed, etc.).
    initial_train_bars : int
        Number of bars in the first training window.
    fold_size : int
        Number of bars in each test (out-of-sample) window.
    max_folds : int
        Maximum number of folds to run (caps runtime on large datasets).
    buy_threshold : float
        BUY fires when bull specialist confidence exceeds this.
    sell_threshold : float
        SELL fires when bear specialist confidence exceeds this.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        initial_train_bars: int = 20_000,
        fold_size: int = 5_000,
        max_folds: int = 6,
        buy_threshold:  float = 0.51,
        sell_threshold: float = 0.51,
    ) -> None:
        self.model_config       = model_config
        self.initial_train_bars = initial_train_bars
        self.fold_size          = fold_size
        self.max_folds          = max_folds
        self.buy_threshold      = buy_threshold
        self.sell_threshold     = sell_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> dict:
        """
        Execute the full walk-forward CV run.

        Returns
        -------
        dict with keys:
            n_folds, accuracy_mean, accuracy_std, accuracy_min, accuracy_max,
            coverage_mean, fold_results.
        """
        min_required = self.initial_train_bars + self.fold_size
        if len(df) < min_required:
            raise ValueError(
                f"Not enough rows for walk-forward CV. "
                f"Need >= {min_required:,}, got {len(df):,}."
            )

        n_possible = (len(df) - self.initial_train_bars) // self.fold_size
        n_folds    = min(self.max_folds, n_possible)

        logger.info("")
        logger.info("=" * 65)
        logger.info("  Walk-Forward CV  (Dual-Head, Expanding Window)")
        logger.info("=" * 65)
        logger.info(f"  Total rows        : {len(df):,}")
        logger.info(f"  Initial train     : {self.initial_train_bars:,} bars")
        logger.info(f"  Fold size         : {self.fold_size:,} bars  (~{self.fold_size // 96} days on M15)")
        logger.info(f"  Folds to run      : {n_folds}")
        logger.info(f"  Features          : {len(FEATURE_COLUMNS)}")
        logger.info(f"  BUY  threshold    : {self.buy_threshold:.0%}  (aggressive — trend supports longs)")
        logger.info(f"  SELL threshold    : {self.sell_threshold:.0%}  (conservative — structural gold bullishness)")
        logger.info("=" * 65)

        fold_results: list[FoldResult] = []

        # One shared DualCatBoostModel instance provides the train_fold interface
        dual = DualCatBoostModel(
            self.model_config,
            buy_threshold=self.buy_threshold,
            sell_threshold=self.sell_threshold,
        )

        for fold_idx in range(n_folds):
            test_start = self.initial_train_bars + fold_idx * self.fold_size
            test_end   = test_start + self.fold_size

            if test_end > len(df):
                break

            train_df = df.iloc[:test_start]
            test_df  = df.iloc[test_start:test_end]

            result = self._run_fold(dual, fold_idx + 1, n_folds, train_df, test_df)
            fold_results.append(result)

        summary = self._aggregate(fold_results)
        self._log_summary(summary)
        return summary

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _run_fold(
        dual: DualCatBoostModel,
        fold_num: int,
        total_folds: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> FoldResult:
        train_start_dt = train_df.index[0].date()
        train_end_dt   = train_df.index[-1].date()
        test_start_dt  = test_df.index[0].date()
        test_end_dt    = test_df.index[-1].date()

        logger.info(
            f"\n  Fold {fold_num}/{total_folds} | "
            f"Train: {train_start_dt} → {train_end_dt} ({len(train_df):,}) | "
            f"Test : {test_start_dt} → {test_end_dt} ({len(test_df):,})"
        )

        X_train = train_df[FEATURE_COLUMNS];  y_train = train_df["target"]
        X_test  = test_df[FEATURE_COLUMNS];   y_test  = test_df["target"]

        metrics = dual.train_fold(X_train, y_train, X_test, y_test)

        acc      = metrics["accuracy"]
        coverage = metrics["coverage"]
        n_trades = metrics["n_trades"]
        n_buy    = metrics.get("n_buy",  0)
        n_sell   = metrics.get("n_sell", 0)
        b_iter   = metrics.get("bull_best_iter", 0)
        s_iter   = metrics.get("bear_best_iter", 0)

        logger.info(
            f"    WinRate: {acc:.4f} | Coverage: {coverage:.1%} | "
            f"Trades: {n_trades} (B:{n_buy} S:{n_sell}) | "
            f"Bull iter: {b_iter} | Bear iter: {s_iter}"
        )

        return FoldResult(
            fold=fold_num,
            train_start=str(train_start_dt),
            train_end=str(train_end_dt),
            test_start=str(test_start_dt),
            test_end=str(test_end_dt),
            train_rows=len(train_df),
            test_rows=len(test_df),
            accuracy=acc,
            coverage=coverage,
            n_trades=n_trades,
            n_buy=n_buy,
            n_sell=n_sell,
            bull_best_iter=b_iter,
            bear_best_iter=s_iter,
        )

    @staticmethod
    def _aggregate(fold_results: list[FoldResult]) -> dict:
        accs      = [r.accuracy  for r in fold_results]
        coverages = [r.coverage  for r in fold_results]
        return {
            "n_folds":        len(fold_results),
            "accuracy_mean":  float(np.mean(accs)),
            "accuracy_std":   float(np.std(accs)),
            "accuracy_min":   float(np.min(accs)),
            "accuracy_max":   float(np.max(accs)),
            "coverage_mean":  float(np.mean(coverages)),
            "coverage_std":   float(np.std(coverages)),
            "fold_results":   fold_results,
        }

    @staticmethod
    def _log_summary(summary: dict) -> None:
        results = summary["fold_results"]
        sep     = "  " + "-" * 71

        logger.info("")
        logger.info("=" * 73)
        logger.info("  Walk-Forward CV — Summary  (Dual-Head)")
        logger.info("=" * 73)
        logger.info(
            f"  {'Fold':<6} {'Test Period':<26} "
            f"{'WinRate':>9} {'Coverage':>9} {'Trades':>7} {'B':>5} {'S':>5}"
        )
        logger.info(sep)

        for r in results:
            period = f"{r.test_start} → {r.test_end}"
            logger.info(
                f"  {r.fold:<6} {period:<26} "
                f"{r.accuracy:>9.4f} {r.coverage:>9.1%} "
                f"{r.n_trades:>7} {r.n_buy:>5} {r.n_sell:>5}"
            )

        logger.info(sep)
        logger.info(
            f"  {'MEAN':<6} {'':26} "
            f"{summary['accuracy_mean']:>9.4f} "
            f"{summary['coverage_mean']:>9.1%}"
        )
        logger.info(
            f"  {'STD':<6} {'':26} {summary['accuracy_std']:>9.4f} "
            f"{summary['coverage_std']:>9.1%}"
        )
        logger.info(
            f"  {'RANGE':<6} {'':26} "
            f"{summary['accuracy_min']:.4f} – {summary['accuracy_max']:.4f}"
        )
        logger.info("=" * 73)

        # Stability verdict — based on win-rate CV
        cv = summary["accuracy_std"] / max(summary["accuracy_mean"], 1e-9)
        if cv < 0.02:
            logger.success(f"  Stability: STABLE   (CV = {cv:.4f} < 0.02)")
        elif cv < 0.05:
            logger.warning(f"  Stability: MODERATE (CV = {cv:.4f})")
        else:
            logger.error(
                f"  Stability: UNSTABLE (CV = {cv:.4f} > 0.05) "
                "— increase regularisation or reduce depth."
            )
        logger.info("")
