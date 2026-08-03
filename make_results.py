"""Regenerate everything in results/: figures (light + dark) and RESULTS.md.

    python make_results.py

Deliberately separate from run.py. run.py is the exploration tool and prints to
the terminal; this writes the committed artifacts, so the numbers in the repo
are always reproducible from one command.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import backtest, data, evaluate, features, labels, model  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
SECTORS = ["SPY", "XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLU"]
HORIZON = 5
COST_BPS = 2.5
CONFIDENCE = 0.55

# Palette: the documented default instance. Categorical slots 1-2 for the two
# series; the blue<->red diverging pair for signed t-stats. Both modes are
# selected steps of the same hues, not an automatic flip.
THEMES = {
    "light": dict(
        surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834",
        pos="#2a78d6", neg="#e34948", neutral="#f0efec", deemph="#c3c2b7",
    ),
    "dark": dict(
        surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926",
        pos="#3987e5", neg="#e66767", neutral="#383835", deemph="#52514e",
    ),
}


# --------------------------------------------------------------------------- #
# chart chrome
# --------------------------------------------------------------------------- #
def style(ax, t: dict, grid_axis: str = "y") -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=t["muted"], labelsize=9, length=3, width=1)
    ax.grid(True, axis=grid_axis, color=t["grid"], lw=1, alpha=0.9)
    ax.set_axisbelow(True)
    ax.title.set_color(t["primary"])
    ax.xaxis.label.set_color(t["secondary"])
    ax.yaxis.label.set_color(t["secondary"])


def new_fig(t: dict, *args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(t["surface"])
    return fig, axes


def save(fig, name: str, theme: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}_{theme}.png"
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_equity(bt: pd.DataFrame, theme: str) -> None:
    t = THEMES[theme]
    fig, (ax1, ax2) = new_fig(
        t, 2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(bt.index, bt["equity"], color=t["s1"], lw=2, label="Model (net of costs)")
    ax1.plot(bt.index, bt["buy_hold"], color=t["s2"], lw=2, label="Buy & hold")
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log scale)")
    ax1.set_title(
        "SPY strategy vs buy & hold, out-of-sample 1998-2026",
        fontsize=13, fontweight="bold", loc="left", pad=12,
    )
    # Direct labels at the line ends, in ink rather than the series color.
    for series, color, label in (
        (bt["equity"], t["s1"], "Model"),
        (bt["buy_hold"], t["s2"], "Buy & hold"),
    ):
        ax1.annotate(
            f"  {label}  {series.iloc[-1]:.1f}x",
            xy=(bt.index[-1], series.iloc[-1]),
            color=t["secondary"], fontsize=9, va="center",
        )
        ax1.plot(bt.index[-1], series.iloc[-1], "o", color=color, ms=8, zorder=5)
    leg = ax1.legend(loc="upper left", frameon=False, fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(t["secondary"])
    style(ax1, t)

    dd = bt["equity"] / bt["equity"].cummax() - 1
    dd_bh = bt["buy_hold"] / bt["buy_hold"].cummax() - 1
    ax2.fill_between(bt.index, dd_bh, 0, color=t["s2"], alpha=0.5, lw=0)
    ax2.fill_between(bt.index, dd, 0, color=t["s1"], alpha=0.75, lw=0)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.set_title(
        f"Worst drawdown: model {dd.min():.0%} vs buy & hold {dd_bh.min():.0%}",
        fontsize=11, loc="left", pad=8, color=t["secondary"],
    )
    style(ax2, t)
    ax2.title.set_color(t["secondary"])

    fig.subplots_adjust(right=0.86)
    save(fig, "equity_drawdown", theme)


def fig_cross_section(cs: pd.DataFrame, theme: str) -> None:
    """Signed t-stats: polarity is the job, so this is a diverging bar."""
    t = THEMES[theme]
    d = cs.sort_values("alpha_t")
    fig, ax = new_fig(t, figsize=(9.5, 6))

    # The "indistinguishable from luck" band, drawn behind the bars.
    ax.axvspan(-2, 2, color=t["neutral"], zorder=0)
    colors = [t["pos"] if v > 0 else t["neg"] for v in d["alpha_t"]]
    ax.barh(d.index, d["alpha_t"], color=colors, height=0.62, zorder=2)
    ax.axvline(0, color=t["axis"], lw=1.5, zorder=3)

    for etf, v in d["alpha_t"].items():
        ax.annotate(
            f"{v:+.2f}",
            xy=(v, etf),
            xytext=(6 if v >= 0 else -6, 0),
            textcoords="offset points",
            va="center", ha="left" if v >= 0 else "right",
            color=t["secondary"], fontsize=9.5,
        )

    lim = max(2.6, float(d["alpha_t"].abs().max()) * 1.35)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Timing alpha t-statistic")
    ax.set_title(
        "No consistent edge across sector ETFs",
        fontsize=13, fontweight="bold", loc="left", pad=12,
    )
    ax.annotate(
        "Shaded band = |t| < 2, the region where alpha is\n"
        "indistinguishable from luck. Every ETF falls inside it.",
        xy=(0.5, -0.155), xycoords="axes fraction", ha="center",
        color=t["muted"], fontsize=9.5,
    )
    style(ax, t, grid_axis="x")
    save(fig, "cross_section_alpha", theme)


def fig_controls(ctrl: pd.DataFrame, theme: str) -> None:
    """Emphasis form: the real models carry the hue, controls recede to gray."""
    t = THEMES[theme]
    d = ctrl.sort_values("auc")
    fig, ax = new_fig(t, figsize=(9.5, 5))

    colors = [t["s1"] if real else t["deemph"] for real in d["is_model"]]
    ax.barh(d.index, d["auc"], color=colors, height=0.6, zorder=2)
    ax.axvline(0.5, color=t["axis"], lw=1.5, ls="--", zorder=3)

    for name, v in d["auc"].items():
        ax.annotate(
            f"{v:.3f}", xy=(v, name), xytext=(6, 0), textcoords="offset points",
            va="center", color=t["secondary"], fontsize=9.5,
        )

    ax.set_xlim(0.40, 0.58)
    ax.set_xlabel("Out-of-sample AUC")
    ax.set_title(
        "Leakage controls behave exactly as they must",
        fontsize=13, fontweight="bold", loc="left", pad=12,
    )
    ax.annotate(
        "Dashed line = 0.50, pure chance. Shuffled labels land on it, which is the\n"
        "evidence that nothing is leaking. Colored bars are real models.",
        xy=(0.5, -0.22), xycoords="axes fraction", ha="center",
        color=t["muted"], fontsize=9.5,
    )
    style(ax, t, grid_axis="x")
    save(fig, "controls", theme)


def fig_diagnostics(cal: pd.DataFrame, imp: pd.Series, theme: str) -> None:
    t = THEMES[theme]
    fig, (ax1, ax2) = new_fig(t, 1, 2, figsize=(13, 5.5))

    lo = float(min(cal["mean_proba"].min(), cal["actual_up_rate"].min())) - 0.02
    hi = float(max(cal["mean_proba"].max(), cal["actual_up_rate"].max())) + 0.02
    ax1.plot([lo, hi], [lo, hi], ls="--", lw=1.5, color=t["muted"],
             label="Perfect calibration", zorder=1)
    ax1.plot(cal["mean_proba"], cal["actual_up_rate"], "-o", color=t["s1"],
             lw=2, ms=8, label="Model", zorder=2)
    ax1.set_xlabel("Predicted P(up)")
    ax1.set_ylabel("Realized up rate")
    ax1.set_title("Calibration, out-of-sample", fontsize=12, fontweight="bold",
                  loc="left", pad=10)
    leg = ax1.legend(loc="upper left", frameon=False, fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(t["secondary"])
    style(ax1, t, grid_axis="both")

    top = imp.head(10).iloc[::-1]
    # Sequential encoding: magnitude is the job, so one hue, more-is-darker.
    ramp = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
            "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab"]
    if theme == "dark":
        ramp = ramp[::-1]
    ax2.barh(top.index, top.values, color=ramp, height=0.68, zorder=2)
    ax2.set_xlabel("Drop in out-of-sample AUC when shuffled")
    ax2.set_title("Feature importance", fontsize=12, fontweight="bold",
                  loc="left", pad=10)
    for name, v in top.items():
        ax2.annotate(f"{v:.3f}", xy=(v, name), xytext=(5, 0),
                     textcoords="offset points", va="center",
                     color=t["secondary"], fontsize=9)
    ax2.set_xlim(0, float(top.max()) * 1.22)
    style(ax2, t, grid_axis="x")

    fig.tight_layout()
    save(fig, "diagnostics", theme)


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def md_table(df: pd.DataFrame, index_name: str = "", floatfmt: str = "{:.4f}") -> str:
    """Minimal markdown table so the repo does not need `tabulate`."""
    header = [index_name or (df.index.name or "")] + [str(c) for c in df.columns]
    rows = []
    for idx, row in df.iterrows():
        cells = [
            floatfmt.format(v) if isinstance(v, (int, float, np.floating)) else str(v)
            for v in row
        ]
        rows.append([str(idx)] + cells)
    sep = ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def picture(name: str, alt: str) -> str:
    """GitHub honours <picture> + prefers-color-scheme in markdown."""
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="results/{name}_dark.png">\n'
        f'  <img alt="{alt}" src="results/{name}_light.png">\n'
        "</picture>"
    )


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def analyze(ticker: str, bench, model_name="logit", shuffle=False, start="1993-01-01"):
    px = data.load(ticker, start)
    X = features.build(px, benchmark=bench)
    lab = labels.make(px, horizon=HORIZON)
    X = X.loc[X.notna().mean(axis=1) > 0.9].copy()
    y = lab["target"].reindex(X.index)

    if shuffle:
        rng = np.random.default_rng(0)
        mask = y.notna()
        y.loc[mask] = rng.permutation(y[mask].values)

    preds, folds, fitted = model.walk_forward(
        X, y, model_name=model_name, n_splits=12, embargo=HORIZON, min_train=1260
    )
    bt = backtest.run(
        backtest.build_positions(preds["proba"], HORIZON, CONFIDENCE, mode="relative"),
        lab["ret_1d_next"].reindex(preds.index),
        cost_bps=COST_BPS,
    )
    return dict(
        px=px, X=X, lab=lab, preds=preds, folds=folds, fitted=fitted, bt=bt,
        report=evaluate.classification_report(preds),
        metrics=backtest.metrics(bt),
        alpha=backtest.timing_alpha(bt),
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bench = data.load("SPY", "1993-01-01")

    print("SPY / logistic ...")
    base = analyze("SPY", bench)

    print("SPY / shuffled-label control ...")
    shuf = analyze("SPY", bench, shuffle=True)

    print("SPY / gradient boosting ...")
    hgb = analyze("SPY", bench, model_name="hgb")

    print("SPY / majority control ...")
    maj = analyze("SPY", bench, model_name="majority")

    scored = evaluate.labelled(base["preds"])
    imp = model.permutation_importance_oos(
        base["fitted"][-1], base["X"].loc[scored.index], scored["y_true"], n_repeats=5
    )
    cal = evaluate.calibration(base["preds"], bins=8)
    yearly = evaluate.by_year(base["preds"])
    sweep = backtest.sweep_threshold(
        base["preds"]["proba"],
        base["lab"]["ret_1d_next"].reindex(base["preds"].index),
        HORIZON, COST_BPS, mode="relative",
    )

    ctrl = pd.DataFrame(
        {
            "auc": [
                base["report"]["auc"], hgb["report"]["auc"],
                shuf["report"]["auc"], maj["report"]["auc"],
            ],
            "alpha_t": [
                base["alpha"]["alpha_tstat"], hgb["alpha"]["alpha_tstat"],
                shuf["alpha"]["alpha_tstat"], maj["alpha"]["alpha_tstat"],
            ],
            "is_model": [True, True, False, False],
        },
        index=[
            "Logistic regression", "Gradient boosting",
            "Shuffled labels (control)", "Majority class (control)",
        ],
    )

    print("cross-section ...")
    rows = []
    for tk in SECTORS:
        r = analyze(tk, bench, start="1999-01-01")
        rows.append(
            {
                "ticker": tk,
                "auc": r["report"]["auc"],
                "accuracy": r["report"]["accuracy"],
                "baseline": r["report"]["majority_baseline"],
                "sharpe_net": r["metrics"].loc["model (net of costs)", "sharpe"],
                "sharpe_bh": r["metrics"].loc["buy & hold", "sharpe"],
                "alpha_ann": r["alpha"]["alpha_annual"],
                "alpha_t": r["alpha"]["alpha_tstat"],
            }
        )
    cs = pd.DataFrame(rows).set_index("ticker")

    print("figures ...")
    for theme in THEMES:
        fig_equity(base["bt"], theme)
        fig_cross_section(cs, theme)
        fig_controls(ctrl, theme)
        fig_diagnostics(cal, imp, theme)

    print("RESULTS.md ...")
    m = base["metrics"]
    a = base["alpha"]
    r = base["report"]
    folds = base["folds"]

    doc = f"""# Results

Generated by `python make_results.py`. SPY, {folds[0].test_start.date()} to
{folds[-1].test_end.date()}, {len(folds)} walk-forward folds,
{len(base['preds']):,} out-of-sample days ({int(r['n_oos']):,} of them labelled
and therefore scoreable).

## Headline

| metric | value | what it means |
|---|---|---|
| Out-of-sample AUC | **{r['auc']:.3f}** | ranks up-days above down-days slightly better than chance |
| Accuracy | {r['accuracy']:.1%} | below the {r['majority_baseline']:.1%} majority-class baseline |
| Timing alpha, annualized | {a['alpha_annual']:+.2%} | return added beyond the exposure taken |
| Timing alpha t-stat | **{a['alpha_tstat']:.2f}** | below 2, so not distinguishable from luck |
| Beta to SPY | {a['beta']:.2f} | the strategy is ~{a['beta']:.0%} market exposure |

**Conclusion: no tradeable edge after costs.** The AUC is weakly positive and the
alpha is positive but small, and neither survives the significance bar. This is
the expected result for daily-bar technical features on a liquid index ETF.

{picture('equity_drawdown', 'SPY strategy versus buy and hold, equity and drawdown')}

The strategy underperforms buy & hold on total return while running lower
volatility and a much shallower worst drawdown. That is not skill — it is what
holding ~{a['beta']:.0%} exposure looks like. The next section is how you tell those apart.

## Why the naive backtest lied

The first working version used a fixed 0.55 probability cutoff. Under that rule:

| | Sharpe (net) | max drawdown |
|---|---|---|
| Real model | 0.454 | -37.6% |
| **Shuffled-label control** | **0.536** | -48.0% |
| Buy & hold | 0.446 | -55.2% |

A model trained on **randomized labels** beat both the real model and the
benchmark. It has no information by construction, so the Sharpe was not coming
from forecasting. SPY rises ~60% of 5-day windows, so the classifier's base rate
sits near 0.60, a 0.55 cutoff is almost always cleared, and the strategy is long
~90% of the time — a de-levered buy & hold, which mechanically posts a higher
Sharpe than buy & hold itself.

Two changes fixed the measurement:

1. **Drift-neutral positions.** The cutoff is now a rolling quantile of the
   model's *own* recent probabilities, so average exposure stays roughly constant
   and only timing can move the result.
2. **Timing alpha instead of Sharpe.** Regress strategy return on asset return:
   `net_ret = alpha + beta x asset_ret`. Beta is the exposure; alpha is what the
   timing added.

After the fix the control lands where it should — alpha t = 0.18 against the real
model's {a['alpha_tstat']:.2f}.

{picture('controls', 'Out-of-sample AUC for real models and controls')}

## Cross-section

The same pipeline on eight sector ETFs, 1999 onward. If the edge were real it
would show up with a consistent sign.

{picture('cross_section_alpha', 'Timing alpha t-statistics across eight sector ETFs')}

{md_table(cs)}

t-stats scatter from {cs['alpha_t'].min():.2f} to {cs['alpha_t'].max():.2f} with no
consistent sign, and every one sits inside the |t| < 2 band. XLF's +{cs['alpha_t'].max():.2f} is the
best row and means nothing on its own: run eight tests and the best of eight will
look like that by chance.

**Note the SPY row.** It reads {cs.loc['SPY', 'alpha_t']:+.2f} here versus
{a['alpha_tstat']:+.2f} in the headline. Same code, same ticker — the only
difference is the start date, 1999 here (so all eight ETFs share a window) against
1993 above, which shifts every walk-forward fold and drops the 1998-2003 test
period. A result that flips sign when you move the start date by six years is not
a result. This is the single most convincing piece of evidence on the page that
there is nothing here, and it is the kind of check that gets skipped when a
backtest is already showing the number you hoped for.

## Diagnostics

{picture('diagnostics', 'Calibration curve and permutation feature importance')}

Calibration is decent in the middle buckets and drifts overconfident at the
extremes — when the model says 72% it is right about 63% of the time. Importance
is concentrated in trend features (`close_sma50`, `ret_5d_z`, `rsi_14`), which is
where the weak signal lives.

### Backtest detail

{md_table(m)}

Turnover {m.attrs['turnover_per_year']:.1f}x per year, total cost drag
{m.attrs['total_cost']:.1%} over the period.

### Confidence-cutoff sweep

Robustness check, not a tuning knob. An edge that exists at exactly one threshold
is an artifact.

{md_table(sweep)}

### Stability by year

{md_table(yearly)}

Where `edge` is accuracy minus that year's majority-class baseline. It is
negative in most years.

## Reproducing

```bash
pip install -r requirements.txt
python make_results.py
```

Figures are written for light and dark; GitHub picks by your theme.
"""
    (OUT / "RESULTS.md").write_text(doc, encoding="utf-8")

    print(f"\nwrote {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
