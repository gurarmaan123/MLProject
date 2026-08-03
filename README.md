# ETF Direction Forecasting

Predicting whether an ETF's price rises or falls over the next few trading days,
using price/volume-derived features and a walk-forward-validated classifier.

The point of this project is not the model. It is the **measurement**. Anyone can
get 60% accuracy on this problem by accident; the work is in building an
evaluation that will tell you when you have nothing — and this one does.

## Result

On SPY, 1993–2026, ~7,000 out-of-sample days across 12 walk-forward folds:

| metric | value | read |
|---|---|---|
| out-of-sample AUC | **0.530** | weak but positive ranking information |
| accuracy | 58.5% | *below* the 59.3% majority-class baseline |
| timing alpha (annual) | +1.4% | |
| alpha t-stat | **1.00** | not distinguishable from luck |

Across eight sector ETFs the alpha t-stats scatter around zero (−0.72 to +1.14)
with no consistent sign. **The honest conclusion is that daily-bar technical
features do not produce a tradeable edge on liquid index ETFs after costs.**
That is the expected answer, and being able to demonstrate it rigorously is worth
more than a backtest that claims otherwise.

## Quickstart

```bash
pip install -r requirements.txt
```

```bash
python run.py
```

Other runs:

```bash
python run.py --model hgb --horizon 10
```

```bash
python run.py --shuffle-labels
```

```bash
python run.py --ticker SPY XLE XLF XLK XLV XLI XLP XLU --start 1999-01-01
```

```bash
python tests/test_pipeline.py
```

## How it works

```
data.py      yfinance download, split/dividend adjusted, cached to parquet
features.py  26 stationary features from OHLCV
labels.py    forward return -> up/down with a volatility-scaled deadband
model.py     pipelines + expanding-window walk-forward with an embargo
backtest.py  positions, costs, matched-exposure benchmark, timing alpha
evaluate.py  calibration, per-year stability, permutation importance, plots
run.py       orchestration and CLI
```

### Features

All 26 are stationary by construction — ratios, z-scores, or bounded
oscillators, never a raw price or volume level. A model trained on raw levels
learns "2005 prices were low" and generalizes to nothing.

- **Momentum** — log returns over 1/5/10/21/63 days
- **Trend** — price vs SMA20/50, SMA10/50 and SMA50/200 crossover ratios
- **Volatility** — realized vol at 10d/21d, plus `vol10/vol63` as a regime signal
- **Volume** — log volume vs its own 20-day mean, and a 5d/60d volume trend
- **Oscillators** — RSI(5), RSI(14), distance from 52-week high/low, intraday range, close location within the bar
- **Calendar** — day of week, month end
- **Relative strength** — 5d and 21d return minus the benchmark's (for non-SPY tickers)

### Labels

`target = 1` if the forward `h`-day return exceeds `+0.5σ`, `0` if below `−0.5σ`,
`NaN` in between. Sigma is the trailing 21-day realized volatility scaled to the
horizon, so the threshold adapts to regime instead of being a fixed percentage.
This keeps ~61% of rows and asks a question worth asking: *did it make a move
that was large relative to current conditions*, rather than *did it tick up*.

## The five things that make the evaluation trustworthy

These are the parts worth reading; each one exists because the naive version is
wrong in a way that inflates results.

**1. Walk-forward, never a random split.** `train_test_split(shuffle=True)` on
time series trains on the future to predict the past. Here the window expands:
train on everything up to date *t*, test the block after it, repeat.

**2. An embargo gap.** A row labelled with a 5-day forward return, sitting at the
end of the training set, overlaps the first days of the test set. The last
`horizon` training rows are dropped before each test block.
(`test_walk_forward_is_strictly_causal` enforces this.)

**3. Deadband rows are predicted, not skipped.** Which day falls inside the
deadband depends on the *future* return. Training excludes those rows, but the
backtest still trades every day — otherwise the strategy would be using
information it could not have had.

**4. A matched-exposure benchmark.** This one caught a real bug. Equities drift
up, so the model's base rate is ~60%; against a fixed 0.55 cutoff it sits long
~90% of the time and posts a Sharpe above buy & hold purely by being a de-levered
version of it. The shuffled-label control scored *better* than the real model
under that rule. Two fixes: the default position rule compares each probability
to a rolling quantile of the model's **own** recent probabilities (drift-neutral,
roughly constant exposure), and the metrics table always shows a constant
position at the strategy's average exposure alongside it.

**5. Timing alpha, not Sharpe.** `net_ret = α + β·asset_ret`. Beta is the
exposure you took; alpha is what the timing added. With a t-stat below ~2, you
have nothing. This is the number to report.

Supporting these: scalers and imputers live **inside** the sklearn pipeline so
they refit per fold and never see test-set statistics; costs are charged at
2.5bp per unit of turnover one-way; and `--shuffle-labels` is a control run that
must score AUC ≈ 0.50 and alpha ≈ 0 (it scores 0.496 and t = 0.18).

## Models

`--model logit` (default) is regularized logistic regression. It is the baseline
on purpose: if a 300-tree gradient booster cannot beat it, that tells you the
signal is linear-or-nonexistent. Here `--model hgb` scores *worse*
out-of-sample (AUC 0.509 vs 0.530), which is the usual outcome on data this
noisy.

`--model majority` is the do-nothing control.

## Interpreting output

- **Accuracy vs `majority_baseline`** — the baseline is above 50% because
  equities drift. Beat the baseline, not the coin flip.
- **Calibration table** — predicted probability vs realized frequency by bucket.
  You do not need to be right often, you need to be right when confident.
- **By-year table** — an edge concentrated in two good years is not an edge.
- **Cutoff sweep** — robustness, not a knob. An edge that exists at one threshold
  is an artifact.
- **Permutation importance** — OOS AUC drop when each feature is shuffled.
  Trend features (`close_sma50`, `ret_5d_z`, `rsi_14`) carry what little signal
  there is.

## Known limitations

- Trades are assumed filled at the close of the signal day; no slippage model
  beyond the flat cost, no bid-ask, no market impact.
- Survivorship is not an issue for these ETFs but would be for a stock universe.
- Yahoo data has occasional bad prints; there is no outlier scrubbing.
- Costs are a flat 2.5bp; a retail account paying more would see less.
- No hyperparameter search — deliberately. Tuning against the walk-forward folds
  would overfit the very thing meant to be an honest holdout.

## Possible extensions

- Cross-sectional ranking across many ETFs (predict *relative* performance)
  rather than single-asset timing — this is where the literature finds real
  signal.
- Longer horizons (21–63 days), where momentum effects are better documented.
- Non-price features: VIX term structure, credit spreads, breadth.
- Purged K-fold with combinatorial splits (López de Prado) for tighter error bars.
