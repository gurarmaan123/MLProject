"""Forward-return labels with a volatility-scaled deadband.

Plain "did it go up over the next h days?" is roughly a coin flip, and a model
that always answers "up" scores ~53% because equities drift. So we scale the
forward return by recent volatility and drop the middle band: what is left is
"did it make a move that was large *relative to its current regime*". That is a
question worth asking, and the classes are close to balanced by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def make(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.5,
    vol_window: int = 21,
) -> pd.DataFrame:
    """Return a frame of forward returns and the classification target.

    Args:
        df: OHLCV frame indexed by date.
        horizon: holding period in trading days.
        threshold: deadband half-width in units of forward-horizon sigma.
            0.0 keeps every row (plain up/down); 0.5 is a good default.
        vol_window: lookback for the volatility used to scale the return.

    Columns:
        fwd_ret     log return from close[t] to close[t+horizon]
        fwd_ret_z   that return divided by the expected horizon sigma
        target      1 (up), 0 (down), NaN inside the deadband or unknowable
    """
    close = df["close"]
    r1 = np.log(close).diff()

    # Volatility as known at t -- no future data.
    sigma_daily = r1.rolling(vol_window).std()
    sigma_horizon = sigma_daily * np.sqrt(horizon)

    fwd_ret = np.log(close.shift(-horizon) / close)
    fwd_ret_z = fwd_ret / sigma_horizon.replace(0, np.nan)

    target = pd.Series(np.nan, index=df.index, dtype=float)
    target[fwd_ret_z > threshold] = 1.0
    target[fwd_ret_z < -threshold] = 0.0

    out = pd.DataFrame(
        {
            "fwd_ret": fwd_ret,
            "fwd_ret_z": fwd_ret_z,
            "target": target,
            "ret_1d_next": r1.shift(-1),  # next-day return, used by the backtest
        }
    )
    out.attrs["horizon"] = horizon
    out.attrs["threshold"] = threshold
    return out


def summarize(labels: pd.DataFrame) -> str:
    y = labels["target"].dropna()
    kept = len(y) / max(len(labels.dropna(subset=["fwd_ret"])), 1)
    up = y.mean() if len(y) else float("nan")
    return (
        f"horizon={labels.attrs.get('horizon')}d  "
        f"deadband=+/-{labels.attrs.get('threshold')} sigma  "
        f"kept {kept:.0%} of rows  "
        f"class balance {up:.1%} up / {1 - up:.1%} down"
    )
