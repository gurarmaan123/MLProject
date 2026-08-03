"""Sanity tests, mostly aimed at look-ahead bias.

Run with: python -m pytest tests/ -q   (or: python tests/test_pipeline.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import backtest, features, labels, model  # noqa: E402


def synthetic_ohlcv(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.integers(1e6, 1e7, n).astype(float),
        },
        index=idx,
    )


def test_features_have_no_lookahead():
    """Truncating the data must not change any feature value on earlier rows.

    If a feature peeked at the future, the values before the cut would shift
    once the future was removed.
    """
    df = synthetic_ohlcv()
    full = features.build(df)
    cut = features.build(df.iloc[:-50])
    common = cut.index
    pd.testing.assert_frame_equal(
        full.loc[common], cut, check_exact=False, rtol=1e-9
    )


def test_labels_look_forward_exactly_horizon_days():
    df = synthetic_ohlcv()
    h = 5
    lab = labels.make(df, horizon=h, threshold=0.0)
    expected = np.log(df["close"].iloc[h] / df["close"].iloc[0])
    assert abs(lab["fwd_ret"].iloc[0] - expected) < 1e-12
    # The final `h` rows cannot be known and must be NaN.
    assert lab["fwd_ret"].iloc[-h:].isna().all()


def test_deadband_drops_middle_and_balances_classes():
    df = synthetic_ohlcv()
    wide = labels.make(df, horizon=5, threshold=1.0)["target"].dropna()
    narrow = labels.make(df, horizon=5, threshold=0.0)["target"].dropna()
    assert len(wide) < len(narrow)
    assert 0.3 < wide.mean() < 0.7


def test_walk_forward_is_strictly_causal():
    """Every test block must start after its training block ends."""
    df = synthetic_ohlcv()
    X = features.build(df).iloc[250:]
    y = labels.make(df, horizon=5)["target"].reindex(X.index)
    _, folds, _ = model.walk_forward(
        X, y, model_name="logit", n_splits=4, embargo=5, min_train=600
    )
    assert folds, "no folds produced"
    for f in folds:
        assert f.train_end < f.test_start
        gap = (X.index.get_loc(f.test_start) - X.index.get_loc(f.train_end))
        assert gap >= 5, f"embargo not honoured: {gap} rows"
    # Expanding window: each fold trains on at least as much as the previous.
    sizes = [f.n_train for f in folds]
    assert sizes == sorted(sizes)


def test_shuffled_labels_produce_no_auc():
    """The leakage control. If this fails, something is leaking."""
    from sklearn.metrics import roc_auc_score

    df = synthetic_ohlcv(2500)
    X = features.build(df).iloc[250:]
    y = labels.make(df, horizon=5)["target"].reindex(X.index)
    rng = np.random.default_rng(0)
    mask = y.notna()
    y.loc[mask] = rng.permutation(y[mask].values)

    preds, _, _ = model.walk_forward(
        X, y, model_name="logit", n_splits=4, embargo=5, min_train=900
    )
    scored = preds.dropna(subset=["y_true"])
    auc = roc_auc_score(scored["y_true"], scored["proba"])
    assert 0.42 < auc < 0.58, f"shuffled labels scored AUC {auc:.3f} -- leak"


def test_costs_reduce_returns():
    df = synthetic_ohlcv()
    proba = pd.Series(np.random.default_rng(1).uniform(0, 1, len(df)), index=df.index)
    nxt = np.log(df["close"]).diff().shift(-1)
    pos = backtest.build_positions(proba, horizon=5, long_threshold=0.5)
    free = backtest.run(pos, nxt, cost_bps=0.0)
    costly = backtest.run(pos, nxt, cost_bps=10.0)
    assert costly["net_ret"].sum() < free["net_ret"].sum()
    assert costly["cost"].sum() > 0


def test_positions_never_exceed_full_exposure():
    """Overlapping tranches must average, not stack into leverage."""
    df = synthetic_ohlcv()
    proba = pd.Series(0.99, index=df.index)
    pos = backtest.build_positions(proba, horizon=5, long_threshold=0.5, mode="absolute")
    assert pos.abs().max() <= 1.0 + 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} passed")
