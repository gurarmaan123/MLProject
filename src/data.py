"""Download and cache daily OHLCV bars from Yahoo Finance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalize(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance returns either flat or (field, ticker) MultiIndex columns."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        levels = df.columns.get_level_values(-1)
        if ticker in set(levels):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}, got {list(df.columns)}")

    df = df[COLUMNS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.sort_index().dropna(subset=["close"])


def load(
    ticker: str,
    start: str = "1993-01-01",
    end: str | None = None,
    refresh: bool = False,
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """Return split/dividend-adjusted daily bars, cached to parquet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{ticker}_{start}_{end or 'latest'}.parquet"

    if path.exists() and not refresh:
        return pd.read_parquet(path)

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    df = _normalize(raw, ticker)
    df.to_parquet(path)
    return df


if __name__ == "__main__":
    spy = load("SPY")
    print(spy.tail())
    print(f"\n{len(spy)} rows, {spy.index[0].date()} to {spy.index[-1].date()}")
