"""Classification diagnostics and plots.

Accuracy against 50% is the wrong yardstick -- equities drift up, so the
majority class is already above 50%. Everything here is reported against the
majority-class baseline, and the calibration table is the one to actually read:
you do not need to be right often, you need to be right when confident.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


def labelled(preds: pd.DataFrame) -> pd.DataFrame:
    """Rows with a real target -- deadband rows are predicted but not scored."""
    return preds.dropna(subset=["y_true"])


def classification_report(preds: pd.DataFrame) -> pd.Series:
    preds = labelled(preds)
    y, p = preds["y_true"], preds["proba"]
    majority = max(y.mean(), 1 - y.mean())
    acc = accuracy_score(y, preds["pred"])
    return pd.Series(
        {
            "n_oos": len(preds),
            "accuracy": acc,
            "majority_baseline": majority,
            "edge_vs_baseline": acc - majority,
            "auc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, p),
        }
    )


def calibration(preds: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Predicted probability vs realized frequency, by confidence bucket."""
    preds = labelled(preds)
    q = pd.qcut(preds["proba"], bins, duplicates="drop")
    tbl = preds.groupby(q, observed=True).agg(
        n=("y_true", "size"),
        mean_proba=("proba", "mean"),
        actual_up_rate=("y_true", "mean"),
    )
    tbl["error"] = tbl["mean_proba"] - tbl["actual_up_rate"]
    return tbl


def by_year(preds: pd.DataFrame) -> pd.DataFrame:
    """Stability check. An edge that lives in two good years is not an edge."""
    preds = labelled(preds)
    g = preds.groupby(preds.index.year)
    rows = g.apply(
        lambda d: pd.Series(
            {
                "n": len(d),
                "accuracy": accuracy_score(d["y_true"], d["pred"]),
                "up_rate": d["y_true"].mean(),
                "auc": roc_auc_score(d["y_true"], d["proba"])
                if d["y_true"].nunique() > 1
                else np.nan,
            }
        ),
        include_groups=False,
    )
    rows["edge"] = rows["accuracy"] - rows[["up_rate"]].apply(
        lambda s: np.maximum(s, 1 - s)
    )["up_rate"]
    return rows


def plot_summary(
    bt: pd.DataFrame,
    preds: pd.DataFrame,
    importance: pd.Series | None,
    ticker: str,
    out_dir: Path = REPORT_DIR,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(bt.index, bt["equity"], label="model (net)", lw=1.4)
    ax.plot(bt.index, bt["buy_hold"], label="buy & hold", lw=1.4, alpha=0.75)
    ax.set_yscale("log")
    ax.set_title(f"{ticker}: out-of-sample equity curve (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    dd = bt["equity"] / bt["equity"].cummax() - 1
    dd_bh = bt["buy_hold"] / bt["buy_hold"].cummax() - 1
    ax.fill_between(bt.index, dd, 0, alpha=0.6, label="model")
    ax.fill_between(bt.index, dd_bh, 0, alpha=0.35, label="buy & hold")
    ax.set_title("Drawdown")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    cal = calibration(preds, bins=8)
    ax.plot(cal["mean_proba"], cal["actual_up_rate"], "o-", label="model")
    lims = [cal["mean_proba"].min(), cal["mean_proba"].max()]
    ax.plot(lims, lims, "--", color="gray", label="perfect calibration")
    ax.set_xlabel("predicted P(up)")
    ax.set_ylabel("realized up rate")
    ax.set_title("Calibration (out-of-sample)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if importance is not None and len(importance):
        top = importance.head(12).iloc[::-1]
        ax.barh(top.index, top.values)
        ax.set_title("Permutation importance (AUC drop, OOS)")
        ax.grid(alpha=0.3, axis="x")
    else:
        ax.axis("off")

    fig.tight_layout()
    path = out_dir / f"{ticker}_summary.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
