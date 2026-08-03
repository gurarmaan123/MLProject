"""Turn out-of-sample probabilities into a strategy and cost it honestly.

Two details that most tutorial backtests get wrong:

1. Overlapping signals. A 5-day-ahead forecast made every day gives you five
   overlapping bets. Treating each as a full-size position double-counts. Here
   the position is the *average* of the last `horizon` signals -- i.e. five
   equally sized tranches, each held for five days.

2. Costs. Trading is not free. Turnover is charged at `cost_bps` per unit of
   position change, one way. At SPY's spread ~1-2bp is realistic for a retail
   account; the default here is deliberately a little pessimistic. Most signals
   that look profitable gross are gone after this line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def build_positions(
    proba: pd.Series,
    horizon: int = 5,
    long_threshold: float = 0.55,
    short_threshold: float | None = None,
    mode: str = "relative",
    lookback: int = 252,
) -> pd.Series:
    """Map probabilities to a target exposure in [-1, 1].

    mode="absolute" compares the probability to a fixed cutoff. This sounds
    right and is usually a trap: equities drift up, so the model's base rate is
    ~60% and a 0.55 cutoff leaves you long ~90% of the time. You then measure a
    de-levered buy & hold and call it a signal.

    mode="relative" (default) compares each probability to the rolling
    distribution of the model's own recent probabilities, going long only when
    today is in the top `1 - long_threshold` of the last `lookback` days. That
    holds average exposure roughly constant, so whatever P&L survives is timing
    rather than drift. Only past probabilities enter the quantile.

    Long-only unless `short_threshold` is set.
    """
    if mode == "absolute":
        long_cut = pd.Series(long_threshold, index=proba.index)
        short_cut = (
            pd.Series(short_threshold, index=proba.index)
            if short_threshold is not None
            else None
        )
    elif mode == "relative":
        roll = proba.rolling(lookback, min_periods=60)
        long_cut = roll.quantile(long_threshold)
        short_cut = roll.quantile(short_threshold) if short_threshold is not None else None
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'absolute' or 'relative'")

    signal = (proba > long_cut).astype(float)
    if short_cut is not None:
        signal -= (proba < short_cut).astype(float)
    # Average the last `horizon` signals -> overlapping tranches, not leverage.
    return signal.rolling(horizon, min_periods=1).mean()


def run(
    positions: pd.Series,
    next_day_return: pd.Series,
    cost_bps: float = 2.5,
) -> pd.DataFrame:
    """Daily P&L. `positions[t]` is held into `next_day_return[t]` (= r[t+1])."""
    pos, ret = positions.align(next_day_return, join="inner")
    pos, ret = pos.fillna(0.0), ret.fillna(0.0)

    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * cost_bps / 10_000
    gross = pos * ret
    net = gross - cost

    return pd.DataFrame(
        {
            "position": pos,
            "asset_ret": ret,
            "gross_ret": gross,
            "cost": cost,
            "net_ret": net,
            "equity": np.exp(net.cumsum()),
            "buy_hold": np.exp(ret.cumsum()),
        }
    )


def _drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def metrics(bt: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side strategy vs buy-and-hold. Returns are logs; annualized."""
    years = len(bt) / TRADING_DAYS

    def stats(log_ret: pd.Series, equity: pd.Series, label: str) -> dict:
        vol = log_ret.std() * np.sqrt(TRADING_DAYS)
        cagr = float(np.exp(log_ret.sum() / years) - 1) if years > 0 else np.nan
        return {
            "strategy": label,
            "total_return": float(equity.iloc[-1] - 1),
            "cagr": cagr,
            "vol": float(vol),
            "sharpe": float(log_ret.mean() / log_ret.std() * np.sqrt(TRADING_DAYS))
            if log_ret.std() > 0
            else np.nan,
            "max_drawdown": _drawdown(equity),
            "time_in_market": float((bt["position"].abs() > 0).mean())
            if label != "buy & hold"
            else 1.0,
        }

    # The benchmark that matters. A model that is simply long 90% of the time
    # inherits most of the asset's return at less than its volatility, which
    # flatters Sharpe without any forecasting skill at all. Holding a constant
    # position equal to the strategy's *average* exposure isolates that effect,
    # so anything left over is timing.
    avg_exposure = float(bt["position"].mean())
    matched = bt["asset_ret"] * avg_exposure

    rows = [
        stats(bt["net_ret"], bt["equity"], "model (net of costs)"),
        stats(bt["gross_ret"], np.exp(bt["gross_ret"].cumsum()), "model (gross)"),
        stats(matched, np.exp(matched.cumsum()), f"const {avg_exposure:.0%} exposure"),
        stats(bt["asset_ret"], bt["buy_hold"], "buy & hold"),
    ]
    out = pd.DataFrame(rows).set_index("strategy")
    out.loc[f"const {avg_exposure:.0%} exposure", "time_in_market"] = avg_exposure
    out.attrs["total_cost"] = float(bt["cost"].sum())
    out.attrs["turnover_per_year"] = float(
        bt["position"].diff().abs().sum() / (len(bt) / TRADING_DAYS)
    )
    return out


def timing_alpha(bt: pd.DataFrame) -> pd.Series:
    """Regress strategy return on asset return: net_ret = alpha + beta*asset_ret.

    Beta is the exposure the strategy took; alpha is what the timing added on
    top of it. This is the number to believe. A |t| below ~2 means the alpha is
    indistinguishable from luck -- which, for a daily-bar retail signal on a
    liquid index ETF, is the expected and honest outcome.
    """
    x = bt["asset_ret"].values
    y = bt["net_ret"].values
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = len(y) - 2
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(A.T @ A)))
    return pd.Series(
        {
            "alpha_daily": coef[0],
            "alpha_annual": coef[0] * TRADING_DAYS,
            "alpha_tstat": coef[0] / se[0],
            "beta": coef[1],
            "r_squared": 1 - resid.var() / y.var(),
        }
    )


def sweep_threshold(
    proba: pd.Series,
    next_day_return: pd.Series,
    horizon: int,
    cost_bps: float = 2.5,
    mode: str = "relative",
    grid: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Sharpe and timing alpha as a function of the confidence cutoff.

    Read this as a robustness check, not a knob to tune: an edge that only
    exists at one threshold is an artifact.
    """
    if grid is None:
        grid = (
            (0.30, 0.40, 0.50, 0.60, 0.70)
            if mode == "relative"
            else (0.50, 0.52, 0.54, 0.56, 0.58, 0.60)
        )
    rows = []
    for t in grid:
        bt = run(build_positions(proba, horizon, t, mode=mode), next_day_return, cost_bps)
        m = metrics(bt).loc["model (net of costs)"]
        a = timing_alpha(bt)
        rows.append(
            {
                "threshold": t,
                "sharpe_net": m["sharpe"],
                "cagr_net": m["cagr"],
                "max_dd": m["max_drawdown"],
                "exposure": m["time_in_market"],
                "alpha_ann": a["alpha_annual"],
                "alpha_t": a["alpha_tstat"],
            }
        )
    return pd.DataFrame(rows).set_index("threshold")
