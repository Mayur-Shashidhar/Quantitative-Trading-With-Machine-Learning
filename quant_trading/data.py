from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _normalise_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.to_datetime(result.index)
    result = result[~result.index.duplicated(keep="first")].sort_index()
    return result


def _extract_ticker(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalise yfinance's single- and multi-ticker column layouts."""
    if isinstance(frame.columns, pd.MultiIndex):
        levels = [set(frame.columns.get_level_values(i)) for i in range(frame.columns.nlevels)]
        if ticker in levels[0]:
            result = frame[ticker]
        elif ticker in levels[-1]:
            result = frame.xs(ticker, axis=1, level=-1)
        else:
            raise KeyError(f"Ticker {ticker!r} not found in downloaded columns")
    else:
        result = frame
    missing = [column for column in PRICE_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError(f"Missing price columns for {ticker}: {missing}")
    return _normalise_index(result[PRICE_COLUMNS].copy())


def download_prices(tickers: Iterable[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    tickers = list(tickers)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if len(tickers) == 1 and not isinstance(raw.columns, pd.MultiIndex):
        return {tickers[0]: _extract_ticker(raw, tickers[0])}
    return {ticker: _extract_ticker(raw, ticker) for ticker in tickers if ticker in raw.columns.get_level_values(0) or ticker in raw.columns.get_level_values(-1)}


def load_price_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return _normalise_index(frame[PRICE_COLUMNS])


def align_close_prices(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = {ticker: frame["Close"] for ticker, frame in price_frames.items()}
    return pd.DataFrame(closes).sort_index()


def clean_supply_prices(prices: pd.DataFrame, target_ticker: str, max_missing_fraction: float = 0.30) -> pd.DataFrame:
    """Apply the notebook's <=30% retention rule and five-day forward fill."""
    if target_ticker not in prices:
        raise ValueError(f"Target ticker {target_ticker!r} is absent from supply prices")
    retained = prices.columns[prices.isna().mean() <= max_missing_fraction]
    result = prices.loc[:, retained].ffill(limit=5)
    if result[target_ticker].isna().any():
        raise ValueError("Target price series contains missing values after cleaning")
    return result
