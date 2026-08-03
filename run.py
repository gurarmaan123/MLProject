"""End-to-end pipeline: download -> features -> labels -> walk-forward -> backtest.

    python run.py                          # SPY, logistic baseline
    python run.py --model hgb              # gradient boosting
    python run.py --shuffle-labels         # leakage control: this MUST score ~0
    python run.py --ticker XLE XLF XLK     # is the edge consistent across ETFs?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import backtest, data, evaluate, features, labels, model  # noqa: E402

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker", nargs="+", default=["SPY"])
    p.add_argument("--benchmark", default="SPY", help="for relative-strength features")
    p.add_argument("--start", default="1993-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--model", default="logit", choices=model.MODELS)
    p.add_argument("--horizon", type=int, default=5, help="forecast horizon, trading days")
    p.add_argument("--threshold", type=float, default=0.5, help="label deadband, in sigma")
    p.add_argument(
        "--signal-mode",
        default="relative",
        choices=("relative", "absolute"),
        help="'relative' = cutoff is a rolling quantile of the model's own probabilities "
        "(drift-neutral); 'absolute' = fixed P(up) cutoff",
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=0.55,
        help="cutoff to go long: a quantile in relative mode, a probability in absolute mode",
    )
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--cost-bps", type=float, default=2.5, help="one-way cost per unit turnover")
    p.add_argument("--splits", type=int, default=12)
    p.add_argument("--min-train", type=int, default=1260, help="~5y of bars before the first test")
    p.add_argument("--shuffle-labels", action="store_true", help="leakage control")
    p.add_argument("--refresh", action="store_true", help="re-download instead of using cache")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def run_one(ticker: str, bench: pd.DataFrame | None, args: argparse.Namespace) -> dict:
    print(f"\n{'=' * 78}\n{ticker}\n{'=' * 78}")

    px = data.load(ticker, args.start, args.end, refresh=args.refresh)
    print(f"bars      : {len(px):,}  {px.index[0].date()} -> {px.index[-1].date()}")

    X = features.build(px, benchmark=bench)
    lab = labels.make(px, horizon=args.horizon, threshold=args.threshold)
    print(f"features  : {X.shape[1]}")
    print(f"labels    : {labels.summarize(lab)}")

    # Drop the warm-up window where long-lookback features are undefined.
    X = X.loc[X.notna().mean(axis=1) > 0.9].copy()
    y = lab["target"].reindex(X.index)

    if args.shuffle_labels:
        rng = np.random.default_rng(args.seed)
        mask = y.notna()
        y.loc[mask] = rng.permutation(y[mask].values)
        print("labels    : SHUFFLED (control run -- any edge here is a bug)")

    preds, folds, fitted = model.walk_forward(
        X,
        y,
        model_name=args.model,
        n_splits=args.splits,
        embargo=args.horizon,
        min_train=args.min_train,
        seed=args.seed,
    )
    print(
        f"walk-fwd  : {len(folds)} folds, first test {folds[0].test_start.date()}, "
        f"last test {folds[-1].test_end.date()}, {len(preds):,} OOS days"
    )

    report = evaluate.classification_report(preds)
    print("\n--- classification (out-of-sample) ---")
    print(report.to_string())

    print("\n--- calibration ---")
    print(evaluate.calibration(preds).to_string())

    print("\n--- by year ---")
    print(evaluate.by_year(preds).to_string())

    bt = backtest.run(
        backtest.build_positions(
            preds["proba"],
            horizon=args.horizon,
            long_threshold=args.confidence,
            short_threshold=1 - args.confidence if args.allow_short else None,
            mode=args.signal_mode,
        ),
        lab["ret_1d_next"].reindex(preds.index),
        cost_bps=args.cost_bps,
    )
    m = backtest.metrics(bt)
    print(
        f"\n--- backtest ({args.cost_bps}bp one-way, {args.confidence} "
        f"{args.signal_mode} cutoff) ---"
    )
    print(m.to_string())
    print(
        f"turnover/yr: {m.attrs['turnover_per_year']:.1f}x   "
        f"total cost drag: {m.attrs['total_cost']:.1%}"
    )

    alpha = backtest.timing_alpha(bt)
    print("\n--- timing alpha vs the asset (the number to believe) ---")
    print(alpha.to_string())
    verdict = (
        "alpha is statistically distinguishable from luck"
        if abs(alpha["alpha_tstat"]) > 2
        else "alpha is NOT distinguishable from luck -- the strategy is mostly exposure"
    )
    print(f"  -> {verdict}")

    print("\n--- confidence-cutoff sweep (robustness, not a knob to tune) ---")
    print(
        backtest.sweep_threshold(
            preds["proba"], lab["ret_1d_next"].reindex(preds.index),
            args.horizon, args.cost_bps, mode=args.signal_mode,
        ).to_string()
    )

    importance = None
    if args.model != "majority":
        scored = evaluate.labelled(preds)
        importance = model.permutation_importance_oos(
            fitted[-1], X.loc[scored.index], scored["y_true"], n_repeats=5, seed=args.seed
        )
        print("\n--- permutation importance (OOS AUC drop) ---")
        print(importance.head(10).to_string())

    if not args.no_plot:
        path = evaluate.plot_summary(bt, preds, importance, ticker)
        print(f"\nplot      : {path}")

    return {
        "ticker": ticker,
        "accuracy": report["accuracy"],
        "baseline": report["majority_baseline"],
        "edge": report["edge_vs_baseline"],
        "auc": report["auc"],
        "sharpe_net": m.loc["model (net of costs)", "sharpe"],
        "sharpe_bh": m.loc["buy & hold", "sharpe"],
        "cagr_net": m.loc["model (net of costs)", "cagr"],
        "cagr_bh": m.loc["buy & hold", "cagr"],
        "max_dd_net": m.loc["model (net of costs)", "max_drawdown"],
        "alpha_ann": alpha["alpha_annual"],
        "alpha_t": alpha["alpha_tstat"],
    }


def main(argv=None) -> int:
    args = parse_args(argv)

    bench = None
    if args.benchmark:
        bench = data.load(args.benchmark, args.start, args.end, refresh=args.refresh)

    results = [run_one(t, bench, args) for t in args.ticker]

    if len(results) > 1:
        print(f"\n{'=' * 78}\nCROSS-SECTION\n{'=' * 78}")
        print(pd.DataFrame(results).set_index("ticker").to_string())
        print(
            "\nAn edge present in one ETF and absent in the rest is noise. "
            "Look for consistency, not the best row."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
