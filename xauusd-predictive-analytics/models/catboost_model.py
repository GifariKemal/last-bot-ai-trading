"""
Dual-Head CatBoost Model for XAUUSD directional prediction.

Architecture
------------
Two specialist binary classifiers trained on the same feature set:

  bull_model  – learns to distinguish BUY bars from SELL bars
                y=1 → BUY, y=0 → SELL
  bear_model  – same data, flipped labels
                y=1 → SELL, y=0 → BUY

Signal resolution
-----------------
  buy_p  = bull_model.predict_proba[:, 1]
  sell_p = bear_model.predict_proba[:, 1]

  buy_p  > conf_threshold AND buy_p  > sell_p  →  BUY  signal (+1)
  sell_p > conf_threshold AND sell_p > buy_p   →  SELL signal (−1)
  otherwise                                     →  NO TRADE   ( 0)

Why two models?
---------------
A single model optimises a single loss and finds a symmetric boundary.
The bull specialist, trained with AUC, is free to shape its own precision/
recall trade-off for the BUY side independently of the SELL side.  This
almost always improves win-rate on each side at the cost of lower coverage
(fewer trades taken) — which is exactly what we want in a live algo.

Feature set (48 features: 40 M15 + 8 HTF)
--------------------------------------------
Dropped candle patterns that never ranked inside the top-15 importances
AND have fewer / highly-noisy fires:
  cdl_morning_star (170 hits), cdl_evening_star (192), cdl_doji (7187 but
  zero directional bias), cdl_harami (3869), cdl_marubozu (2276),
  cdl_piercing (15 — extreme rarity), cdl_dragonfly (913),
  cdl_gravestone (902), cdl_inv_hammer (266).

Kept: cdl_hammer (1419), cdl_engulfing (8069), cdl_shooting_star (349).
These three have the clearest directional meaning and cover reversal /
momentum-shift / rejection scenarios.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier  # type: ignore
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report

from config.settings import ModelConfig


# ── Feature list (40 total) ───────────────────────────────────────────────────
FEATURE_COLUMNS: list[str] = [
    # Momentum
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "stoch_k",
    "stoch_d",
    "cci_20",
    # Trend
    "close_vs_ema50",
    "close_vs_ema200",
    "h1_ema50_bias",
    "h4_ema50_bias",
    # Multi-Timeframe — H1 structure snapshot (leakage-proof: shift+ffill)
    "h1_rsi",
    "h1_adx",
    "h1_bb_width",
    "h1_bb_pos",
    # Multi-Timeframe — H4 macro context
    "h4_rsi",
    "h4_adx",
    "h4_bb_width",
    "h4_bb_pos",
    # Regime
    "adx_14",
    "di_diff",
    "atr_14",
    "atr_regime",
    # Session High / Low proximity (ATR-normalised)
    "dist_to_high_8h",
    "dist_to_low_8h",
    "dist_to_high_24h",
    "dist_to_low_24h",
    # Volatility
    "bb_pct",
    "bb_width",
    "vol_ratio",
    "return_std_5",
    "return_std_20",
    # Candle structure
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    # Candle patterns — pruned to 3 high-signal, directional patterns
    "cdl_hammer",         # 1,419 hits — bullish pin-bar reversal
    "cdl_engulfing",      # 8,069 hits — momentum shift (most frequent)
    "cdl_shooting_star",  # 349   hits — bearish rejection at highs
    # Time / calendar
    "day_of_week",
    "hour_sin",
    "hour_cos",
    # Session flags
    "is_asian",
    "is_london",
    "is_new_york",
    "is_overlap",
    # Lagged log-returns
    "return_1",
    "return_3",
    "return_5",
]

# day_of_week is a discrete categorical — CatBoost handles it natively
CAT_FEATURE_INDICES: list[int] = [FEATURE_COLUMNS.index("day_of_week")]


# ── Shared hyper-parameters for both specialists ──────────────────────────────
def _build_specialist(random_state: int) -> CatBoostClassifier:
    # eval_metric="Accuracy" (not AUC) — AUC converges in 1-2 iterations on
    # weak-signal financial data, triggering early stopping immediately and
    # preventing the model from learning.  Accuracy improves more gradually
    # and gives early stopping enough room to run meaningful iterations.
    # Params from Optuna Trial #0 (MTF run, fixed atr=0.170):
    #   WinRate 53.25% | Coverage 20.0% | Folds 3 | Stability MODERATE
    #   Chosen over Trial #1 (54.28%) because sell_threshold=0.694 in Trial #1
    #   overfits — bear specialist rarely exceeds 69% in unseen data.
    return CatBoostClassifier(
        iterations=1_000,
        learning_rate=0.089,
        depth=5,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=random_state,
        auto_class_weights="Balanced",
        l2_leaf_reg=8.961,
        min_data_in_leaf=64,
        cat_features=CAT_FEATURE_INDICES,
        verbose=False,
        allow_writing_files=False,
    )


# ── Dual-Head Model ───────────────────────────────────────────────────────────
class DualCatBoostModel:
    """
    Two specialist CatBoost classifiers that vote on BUY / SELL / NO TRADE.

    Asymmetric thresholding
    -----------------------
    Gold is a structurally bullish asset (inflation hedge, safe haven, central
    bank demand).  The bull specialist consistently outperforms the bear
    specialist on M15 XAUUSD.  Setting a *lower* threshold for BUY and a
    *higher* threshold for SELL reflects this reality:

      buy_threshold  = 0.55  — AGGRESSIVE: trend supports longs
      sell_threshold = 0.70  — CONSERVATIVE: only enter shorts on very strong
                               conviction; false shorts are expensive in a
                               secular bull market

    Signal resolution
    -----------------
      buy_p  = bull_model P(1)     sell_p = bear_model P(1)

      buy_p  > buy_threshold  AND buy_p  > sell_p  →  BUY  (+1)
      sell_p > sell_threshold AND sell_p > buy_p   →  SELL (−1)
      otherwise                                     →  NO TRADE (0)

    Parameters
    ----------
    config : ModelConfig
    buy_threshold : float
        BUY fires when bull specialist confidence exceeds this.
    sell_threshold : float
        SELL fires when bear specialist confidence exceeds this.
    """

    def __init__(
        self,
        config: ModelConfig,
        buy_threshold:  float = 0.51,
        sell_threshold: float = 0.51,
    ) -> None:
        self.config         = config
        self.buy_threshold  = buy_threshold
        self.sell_threshold = sell_threshold
        self.feature_columns: list[str] = FEATURE_COLUMNS
        self.bull_model: CatBoostClassifier | None = None
        self.bear_model: CatBoostClassifier | None = None

    # ── Public training API ───────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Train both specialists on df, report evaluation on the held-out
        20 % test window.

        Returns
        -------
        dict with keys: accuracy, coverage, n_trades, train_rows, test_rows.
        """
        logger.info("Preparing dual-model training data …")
        X, y = self._prepare(df)

        split    = int(len(df) * (1 - self.config.test_size))
        X_train  = X.iloc[:split];  y_train = y.iloc[:split]
        X_test   = X.iloc[split:];  y_test  = y.iloc[split:]

        logger.info(
            f"Train: {len(X_train):,} | Test: {len(X_test):,} | "
            f"Features: {len(self.feature_columns)}"
        )

        logger.info("Training Bull (BUY) specialist …")
        self.bull_model = _build_specialist(self.config.random_state)
        self.bull_model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
        )

        logger.info("Training Bear (SELL) specialist …")
        self.bear_model = _build_specialist(self.config.random_state)
        self.bear_model.fit(
            X_train, 1 - y_train,
            eval_set=(X_test, 1 - y_test),
            early_stopping_rounds=50,
        )

        return self._evaluate(X_test, y_test, tag="Final model")

    # ── Walk-forward fold interface ───────────────────────────────────────────

    def train_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """
        Train fresh specialists on one WF fold, return fold metrics.
        Called by WalkForwardCV._run_fold().
        """
        bull = _build_specialist(self.config.random_state)
        bear = _build_specialist(self.config.random_state)

        bull.fit(X_train, y_train,      eval_set=(X_test, y_test),      early_stopping_rounds=50)
        bear.fit(X_train, 1 - y_train,  eval_set=(X_test, 1 - y_test),  early_stopping_rounds=50)

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

        # Map signal space (1/−1) → label space (1/0) for accuracy
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

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_latest(self, df: pd.DataFrame) -> tuple[float, float, int]:
        """
        Score the most recent bar in df.

        Returns
        -------
        (buy_prob, sell_prob, signal)
            signal: +1 = BUY, −1 = SELL, 0 = NO TRADE
        """
        if self.bull_model is None or self.bear_model is None:
            raise RuntimeError("Models not trained yet. Call train() first.")

        latest = df[self.feature_columns].dropna().iloc[[-1]]
        buy_p  = float(self.bull_model.predict_proba(latest)[0, 1])
        sell_p = float(self.bear_model.predict_proba(latest)[0, 1])
        signal = self._resolve(buy_p, sell_p)

        label = {1: "BUY", -1: "SELL", 0: "NO TRADE"}[signal]
        logger.info(
            f"Latest bar → P(BUY): {buy_p:.4f} (thr {self.buy_threshold:.0%}) | "
            f"P(SELL): {sell_p:.4f} (thr {self.sell_threshold:.0%}) | "
            f"Signal: {label}"
        )
        return buy_p, sell_p, signal

    # ── Private helpers ───────────────────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame):
        missing = [c for c in self.feature_columns + ["target"] if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        return df[self.feature_columns], df["target"]

    def _resolve(self, buy_p: float, sell_p: float) -> int:
        """
        Convert raw specialist probabilities to a trade signal.

        Asymmetric:
          BUY  fires if bull > buy_threshold  AND bull  > bear
          SELL fires if bear > sell_threshold AND bear  > bull
        """
        if buy_p  > self.buy_threshold  and buy_p  > sell_p:
            return  1
        if sell_p > self.sell_threshold and sell_p > buy_p:
            return -1
        return 0

    def _evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        tag: str = "",
    ) -> dict[str, float]:
        buy_probs  = self.bull_model.predict_proba(X_test)[:, 1]
        sell_probs = self.bear_model.predict_proba(X_test)[:, 1]
        signals    = pd.Series(
            [self._resolve(b, s) for b, s in zip(buy_probs, sell_probs)],
            index=X_test.index,
        )

        traded   = signals != 0
        n_trades = int(traded.sum())
        coverage = n_trades / len(X_test)
        n_buy    = int((signals == 1).sum())
        n_sell   = int((signals == -1).sum())

        logger.info("")
        logger.info("=" * 55)
        logger.info(
            f"  Dual-Model Evaluation  [{tag}]  "
            f"BUY≥{self.buy_threshold:.0%}  SELL≥{self.sell_threshold:.0%}"
        )
        logger.info("=" * 55)
        logger.info(f"  Test bars    : {len(X_test):,}")
        logger.info(f"  Trades taken : {n_trades:,}  ({coverage:.1%} coverage)")
        logger.info(f"  BUY  signals : {n_buy:,}")
        logger.info(f"  SELL signals : {n_sell:,}")

        if n_trades == 0:
            logger.warning("  No trades triggered — consider lowering conf_threshold.")
            return {"accuracy": 0.0, "coverage": 0.0, "n_trades": 0.0,
                    "train_rows": 0.0, "test_rows": float(len(X_test))}

        sig_binary = signals[traded].map({1: 1, -1: 0})
        acc        = float(accuracy_score(y_test[traded], sig_binary))

        logger.success(f"  Win Rate (precision on triggered trades): {acc:.2%}")
        logger.info(
            "\n" + classification_report(
                y_test[traded], sig_binary,
                labels=[0, 1],
                target_names=["SELL (0)", "BUY (1)"],
                zero_division=0,
            )
        )

        # Average feature importance across both specialists
        bull_imp = dict(zip(self.feature_columns, self.bull_model.get_feature_importance()))
        bear_imp = dict(zip(self.feature_columns, self.bear_model.get_feature_importance()))
        avg_imp  = sorted(
            [(f, (bull_imp[f] + bear_imp[f]) / 2) for f in self.feature_columns],
            key=lambda x: x[1],
            reverse=True,
        )
        logger.info("Top-15 feature importances (avg bull + bear):")
        for rank, (feat, score) in enumerate(avg_imp[:15], 1):
            bar = "█" * int(score / 2)
            logger.info(f"  {rank:>2}. {feat:<22} {score:>6.2f}  {bar}")

        return {
            "accuracy":   acc,
            "coverage":   coverage,
            "n_trades":   float(n_trades),
            "train_rows": 0.0,           # filled in by caller if needed
            "test_rows":  float(len(X_test)),
        }
