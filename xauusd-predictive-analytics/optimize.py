"""
Optuna hyperparameter search for the Dual-Head CatBoost model.

Strategy
--------
Data fetch + feature engineering run ONCE before the study starts — they
are the expensive parts (~30 s).  Each trial only reruns:
  - create_labels   (atr_threshold is a search param)
  - WalkForwardCV   (model + threshold params are search params)

Objective: maximise walk-forward accuracy_mean (win rate on triggered trades).

Search space
------------
  depth              int    4 – 8
  learning_rate      float  0.01 – 0.10  (log scale)
  l2_leaf_reg        float  1 – 20       (log scale)
  min_data_in_leaf   int    10 – 100
  buy_threshold      float  0.50 – 0.60
  sell_threshold     float  0.50 – 0.70
  label_atr_thresh   float  0.15 – 0.50

Usage
-----
    python optimize.py
    python optimize.py --trials 50
"""

from __future__ import annotations

import argparse
import sys

import optuna
from loguru import logger

from config.settings import AppConfig, ModelConfig
from core.connector import MT5Connector
from data.fetcher import DataFetcher
from models.catboost_model import (
    DualCatBoostModel,
    FEATURE_COLUMNS,
    _build_specialist,
    CAT_FEATURE_INDICES,
)
from models.walk_forward import WalkForwardCV
from strategies.feature_engineer import FeatureEngineer

# ── Patch _build_specialist so Optuna can inject custom params ────────────────
from catboost import CatBoostClassifier  # type: ignore


def _build_specialist_custom(
    random_state: int,
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
    min_data_in_leaf: int,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=1_000,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=random_state,
        auto_class_weights="Balanced",
        l2_leaf_reg=l2_leaf_reg,
        min_data_in_leaf=min_data_in_leaf,
        cat_features=CAT_FEATURE_INDICES,
        verbose=False,
        allow_writing_files=False,
    )


# ── Logger — minimal during Optuna (trial progress logged manually) ───────────
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    ),
    colorize=True,
    level="WARNING",   # suppress per-fold chatter; trials log their own summary
)

# Separate logger for trial summaries (always visible)
trial_logger = logger.bind(name="optuna")
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<cyan>TRIAL   </cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
    level="INFO",
    filter=lambda r: r["extra"].get("name") == "optuna",
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Custom DualCatBoostModel that accepts injected specialist params ──────────
class TunableDualModel(DualCatBoostModel):
    """DualCatBoostModel with overrideable specialist hyperparameters."""

    def __init__(
        self,
        config: ModelConfig,
        buy_threshold: float,
        sell_threshold: float,
        depth: int,
        learning_rate: float,
        l2_leaf_reg: float,
        min_data_in_leaf: int,
    ) -> None:
        super().__init__(config, buy_threshold, sell_threshold)
        self._depth            = depth
        self._learning_rate    = learning_rate
        self._l2_leaf_reg      = l2_leaf_reg
        self._min_data_in_leaf = min_data_in_leaf

    def _make_specialist(self) -> CatBoostClassifier:
        return _build_specialist_custom(
            random_state=self.config.random_state,
            depth=self._depth,
            learning_rate=self._learning_rate,
            l2_leaf_reg=self._l2_leaf_reg,
            min_data_in_leaf=self._min_data_in_leaf,
        )

    def train_fold(self, X_train, y_train, X_test, y_test) -> dict:
        from sklearn.metrics import accuracy_score
        import pandas as pd

        bull = self._make_specialist()
        bear = self._make_specialist()

        bull.fit(X_train, y_train,     eval_set=(X_test, y_test),     early_stopping_rounds=50)
        bear.fit(X_train, 1 - y_train, eval_set=(X_test, 1 - y_test), early_stopping_rounds=50)

        buy_probs  = bull.predict_proba(X_test)[:, 1]
        sell_probs = bear.predict_proba(X_test)[:, 1]
        signals    = pd.Series(
            [self._resolve(b, s) for b, s in zip(buy_probs, sell_probs)],
            index=X_test.index,
        )

        traded = signals != 0
        if traded.sum() == 0:
            return {
                "accuracy": 0.0, "coverage": 0.0,
                "n_trades": 0,   "n_buy": 0, "n_sell": 0,
                "bull_best_iter": int(bull.get_best_iteration() or 0),
                "bear_best_iter": int(bear.get_best_iteration() or 0),
            }

        sig_binary = signals[traded].map({1: 1, -1: 0})
        acc        = float(accuracy_score(y_test[traded], sig_binary))
        coverage   = float(traded.sum()) / len(X_test)

        return {
            "accuracy":       acc,
            "coverage":       coverage,
            "n_trades":       int(traded.sum()),
            "n_buy":          int((signals == 1).sum()),
            "n_sell":         int((signals == -1).sum()),
            "bull_best_iter": int(bull.get_best_iteration() or 0),
            "bear_best_iter": int(bear.get_best_iteration() or 0),
        }


# ── Objective ─────────────────────────────────────────────────────────────────
def make_objective(featured_df, config: AppConfig):
    fe = FeatureEngineer()

    def objective(trial: optuna.Trial) -> float:
        # ── Sample hyperparameters ─────────────────────────────────────────
        # label_atr_thresh fixed at 0.170 (best from prior search, gives
        # enough labelled rows without discarding too much signal).
        LABEL_ATR_THRESH = 0.170

        depth              = trial.suggest_int  ("depth",            4,    8)
        learning_rate      = trial.suggest_float("learning_rate",    0.01, 0.10, log=True)
        l2_leaf_reg        = trial.suggest_float("l2_leaf_reg",      1.0,  20.0, log=True)
        min_data_in_leaf   = trial.suggest_int  ("min_data_in_leaf", 10,   100)
        buy_threshold      = trial.suggest_float("buy_threshold",    0.50, 0.60)
        sell_threshold     = trial.suggest_float("sell_threshold",   0.50, 0.70)

        # ── Re-label (fixed ATR threshold) ────────────────────────────────
        label_atr_thresh = LABEL_ATR_THRESH
        labelled_df = fe.create_labels(featured_df, atr_threshold=label_atr_thresh)

        if len(labelled_df) < 25_000:
            # Too few labelled rows → unfeasible, prune
            raise optuna.TrialPruned()

        # ── Walk-forward CV ────────────────────────────────────────────────
        dual = TunableDualModel(
            config=config.model,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            min_data_in_leaf=min_data_in_leaf,
        )

        wf = WalkForwardCV(
            model_config=config.model,
            initial_train_bars=20_000,
            fold_size=5_000,
            max_folds=6,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )

        # Monkey-patch WalkForwardCV to use our TunableDualModel
        import types

        def _run_fold_patched(dual_inner, fold_num, total_folds, train_df, test_df):
            from models.walk_forward import FoldResult
            train_start_dt = train_df.index[0].date()
            train_end_dt   = train_df.index[-1].date()
            test_start_dt  = test_df.index[0].date()
            test_end_dt    = test_df.index[-1].date()

            X_train = train_df[FEATURE_COLUMNS]; y_train = train_df["target"]
            X_test  = test_df[FEATURE_COLUMNS];  y_test  = test_df["target"]

            metrics  = dual.train_fold(X_train, y_train, X_test, y_test)
            return FoldResult(
                fold=fold_num,
                train_start=str(train_start_dt), train_end=str(train_end_dt),
                test_start=str(test_start_dt),   test_end=str(test_end_dt),
                train_rows=len(train_df),         test_rows=len(test_df),
                accuracy=metrics["accuracy"],     coverage=metrics["coverage"],
                n_trades=metrics["n_trades"],     n_buy=metrics["n_buy"],
                n_sell=metrics["n_sell"],
                bull_best_iter=metrics["bull_best_iter"],
                bear_best_iter=metrics["bear_best_iter"],
            )

        wf._run_fold = _run_fold_patched

        summary  = wf.run(labelled_df)
        acc_mean = summary["accuracy_mean"]
        coverage = summary["coverage_mean"]
        n_folds  = summary["n_folds"]

        # ── Coverage floor & fold floor ────────────────────────────────────
        # Minimum 10% coverage AND minimum 2 folds.
        # Trials that fire on < 10% of bars are statistically unreliable
        # (e.g. trial #6 scored 80% WinRate on just ~5 trades — meaningless).
        MIN_COVERAGE = 0.10
        MIN_FOLDS    = 2

        if coverage < MIN_COVERAGE or n_folds < MIN_FOLDS:
            trial_logger.info(
                f"Trial {trial.number:>3} | PRUNED "
                f"(coverage={coverage:.1%} < {MIN_COVERAGE:.0%} "
                f"or folds={n_folds} < {MIN_FOLDS}) | "
                f"buy={buy_threshold:.2f} sell={sell_threshold:.2f} "
                f"atr={label_atr_thresh:.2f}"
            )
            raise optuna.TrialPruned()

        trial_logger.info(
            f"Trial {trial.number:>3} | "
            f"WinRate {acc_mean:.4f} | Coverage {coverage:.1%} | "
            f"Folds {n_folds} | "
            f"depth={depth} lr={learning_rate:.3f} l2={l2_leaf_reg:.1f} "
            f"leaf={min_data_in_leaf} "
            f"buy={buy_threshold:.2f} sell={sell_threshold:.2f} "
            f"atr={label_atr_thresh:.2f}"
        )
        return acc_mean

    return objective


# ── Main ──────────────────────────────────────────────────────────────────────
def main(n_trials: int = 10) -> None:
    print()
    print("=" * 68)
    print("  Optuna Hyperparameter Search — Dual-Head CatBoost (XAUUSD M15)")
    print(f"  Trials: {n_trials}")
    print("=" * 68)

    config = AppConfig()

    with MT5Connector(config.mt5) as _conn:
        print(f"\n  Fetching M15 data …")
        fetcher     = DataFetcher(config.trading)
        raw_df      = fetcher.fetch_ohlcv()

        print(f"  Building features …")
        fe          = FeatureEngineer()
        featured_df = fe.build_features(raw_df)

        print(f"  Starting Optuna study ({n_trials} trials) …\n")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(
            make_objective(featured_df, config),
            n_trials=n_trials,
            show_progress_bar=False,
        )

    # ── Results ───────────────────────────────────────────────────────────
    best = study.best_trial
    print()
    print("=" * 68)
    print("  Optuna Search Complete")
    print("=" * 68)
    print(f"  Best WinRate : {best.value:.4f}  (trial #{best.number})")
    print(f"  Best params  :")
    for k, v in best.params.items():
        print(f"    {k:<22} = {v}")
    print("=" * 68)

    # Top-5 trials
    print("\n  Top-5 trials by WinRate:")
    trials_sorted = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True,
    )[:5]
    print(f"  {'#':>4}  {'WinRate':>9}  params")
    print("  " + "-" * 62)
    for t in trials_sorted:
        params_str = "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in t.params.items())
        print(f"  {t.number:>4}  {t.value:>9.4f}  {params_str}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    main(n_trials=args.trials)
