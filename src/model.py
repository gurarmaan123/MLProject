"""Models and walk-forward evaluation.

The only correct way to test a trading model is the way you would have had to
use it: train on the past, predict the future, never the reverse. Every number
this module reports is out-of-sample under that constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS = ("logit", "hgb", "majority")


def make_model(name: str, seed: int = 0) -> Pipeline:
    """Build a pipeline. Scaling and imputation live *inside* the pipeline so
    they are refit on each training fold and never see test-set statistics."""
    if name == "logit":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, max_iter=2000, random_state=seed)),
            ]
        )
    if name == "hgb":
        return Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=3,
                        learning_rate=0.03,
                        max_iter=300,
                        min_samples_leaf=100,
                        l2_regularization=1.0,
                        early_stopping=False,
                        random_state=seed,
                    ),
                )
            ]
        )
    if name == "majority":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("clf", DummyClassifier(strategy="prior")),
            ]
        )
    raise ValueError(f"unknown model {name!r}; expected one of {MODELS}")


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "logit",
    n_splits: int = 10,
    embargo: int = 5,
    min_train: int = 1260,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[Fold], list[Pipeline]]:
    """Expanding-window walk-forward with an embargo gap.

    The embargo drops the last `embargo` training rows before each test block.
    Without it, a row labelled with a 5-day forward return sitting at the end of
    the training set overlaps the first days of the test set -- a small but real
    leak that flatters every metric downstream.

    `y` may contain NaN for rows inside the label deadband. Those rows are
    excluded from *training* but still *predicted on*: which day lands in the
    deadband is a function of the future return, so a backtest that traded only
    labelled days would be using information it could not have had.

    Returns (predictions, folds, fitted_models). `predictions` is indexed by
    date, is entirely out-of-sample, and carries NaN in `y_true` for deadband
    rows.
    """
    X, y = X.align(y, join="inner", axis=0)
    n = len(X)
    if n <= min_train + n_splits:
        raise ValueError(f"only {n} usable rows; need more than {min_train}")

    bounds = np.linspace(min_train, n, n_splits + 1).astype(int)
    frames, folds, fitted = [], [], []

    for i in range(n_splits):
        test_lo, test_hi = bounds[i], bounds[i + 1]
        train_hi = max(test_lo - embargo, 1)
        if test_hi - test_lo == 0 or train_hi < 50:
            continue

        y_tr = y.iloc[:train_hi].dropna()
        X_tr = X.loc[y_tr.index]
        X_te, y_te = X.iloc[test_lo:test_hi], y.iloc[test_lo:test_hi]
        if len(y_tr) < 50 or y_tr.nunique() < 2:
            continue

        model = make_model(model_name, seed=seed)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]

        frames.append(
            pd.DataFrame(
                {"proba": proba, "y_true": y_te.values, "fold": i}, index=X_te.index
            )
        )
        folds.append(
            Fold(
                train_start=X_tr.index[0],
                train_end=X_tr.index[-1],
                test_start=X_te.index[0],
                test_end=X_te.index[-1],
                n_train=len(X_tr),
                n_test=len(X_te),
            )
        )
        fitted.append(model)

    preds = pd.concat(frames).sort_index()
    preds["pred"] = (preds["proba"] > 0.5).astype(int)
    return preds, folds, fitted


def permutation_importance_oos(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    seed: int = 0,
) -> pd.Series:
    """Drop in AUC when each column is shuffled, measured out-of-sample."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    scores = {}
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        scores[col] = float(np.mean(drops))
    return pd.Series(scores).sort_values(ascending=False)
