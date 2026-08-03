"""Feature engineering.

Every feature is stationary by construction -- ratios, z-scores or bounded
oscillators, never a raw price or volume level. A model trained on raw levels
learns "2005 prices were low" and generalizes to nothing.

All features on row `t` use information available at the close of day `t` only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI, bounded 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def realized_vol(log_ret: pd.Series, window: int) -> pd.Series:
    """Annualized realized volatility."""
    return log_ret.rolling(window).std() * np.sqrt(TRADING_DAYS)


def build(df: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the feature matrix from OHLCV bars.

    Args:
        df: OHLCV frame indexed by date (see data.load).
        benchmark: optional OHLCV frame for a reference index (e.g. SPY).
            Used for relative-strength features; skipped if None or if it is
            the same series as `df`.
    """
    close, volume = df["close"], df["volume"]
    r1 = np.log(close).diff()
    out = pd.DataFrame(index=df.index)

    # --- momentum: log returns over several lookbacks -----------------------
    for n in (1, 5, 10, 21, 63):
        out[f"ret_{n}d"] = np.log(close / close.shift(n))

    # --- trend: price relative to its own moving averages -------------------
    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    out["close_sma20"] = close / sma20 - 1
    out["close_sma50"] = close / sma50 - 1
    out["sma10_sma50"] = sma10 / sma50 - 1
    out["sma50_sma200"] = sma50 / sma200 - 1

    # --- volatility, in levels and as a regime ratio ------------------------
    vol10 = realized_vol(r1, 10)
    vol21 = realized_vol(r1, 21)
    vol63 = realized_vol(r1, 63)
    out["vol_10d"] = vol10
    out["vol_21d"] = vol21
    out["vol_ratio"] = vol10 / vol63          # >1 = vol expanding
    out["ret_5d_z"] = out["ret_5d"] / (vol21 * np.sqrt(5 / TRADING_DAYS))

    # --- volume, normalized against its own recent average ------------------
    out["vol_rel_20d"] = np.log(volume / volume.rolling(20).mean().replace(0, np.nan))
    out["vol_trend"] = (
        volume.rolling(5).mean() / volume.rolling(60).mean().replace(0, np.nan) - 1
    )

    # --- oscillators and range position -------------------------------------
    out["rsi_14"] = rsi(close, 14)
    out["rsi_5"] = rsi(close, 5)
    hi252 = close.rolling(TRADING_DAYS).max()
    lo252 = close.rolling(TRADING_DAYS).min()
    out["dist_52w_high"] = close / hi252 - 1
    out["dist_52w_low"] = close / lo252 - 1
    out["intraday_range"] = (df["high"] - df["low"]) / close
    out["close_loc"] = (close - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)

    # --- calendar ------------------------------------------------------------
    dow = df.index.dayofweek
    for d in range(1, 5):  # Monday is the dropped baseline
        out[f"dow_{d}"] = (dow == d).astype(float)
    out["month_end"] = (df.index.day >= 25).astype(float)

    # --- relative strength vs the broad market -------------------------------
    if benchmark is not None and not benchmark["close"].equals(close):
        bench = benchmark["close"].reindex(df.index).ffill()
        for n in (5, 21):
            rel = np.log(close / close.shift(n)) - np.log(bench / bench.shift(n))
            out[f"rel_ret_{n}d"] = rel

    return out.replace([np.inf, -np.inf], np.nan)
