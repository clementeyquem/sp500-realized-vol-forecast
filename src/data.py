"""Download and cache the S&P 500 and VIX daily close series.

Data source: Yahoo Finance (yfinance), adjusted closes.
The downloaded series are cached as CSV under data/ so that results are
reproducible without depending on Yahoo Finance availability.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

START = "2000-01-01"
END = "2024-12-31"
SPX_TICKER = "^GSPC"
VIX_TICKER = "^VIX"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "spx_vix_close.csv"


def download_close_series(start: str = START, end: str = END) -> pd.DataFrame:
    """Download SPX and VIX closes from Yahoo Finance.

    Returns a DataFrame indexed by date with columns ['spx', 'vix'],
    restricted to dates where both series are available.
    """
    spx = yf.download(SPX_TICKER, start=start, end=end, auto_adjust=True, progress=False)
    vix = yf.download(VIX_TICKER, start=start, end=end, auto_adjust=True, progress=False)

    # squeeze() turns the single-column DataFrame returned by yfinance into a Series.
    df = pd.DataFrame(
        {"spx": spx["Close"].squeeze(), "vix": vix["Close"].squeeze()}
    ).dropna()
    df.index.name = "date"
    return df


def load_close_series(use_cache: bool = True, refresh: bool = False) -> pd.DataFrame:
    """Load SPX and VIX closes, from the CSV cache when available.

    Set refresh=True to re-download and overwrite the cache.
    """
    if use_cache and CACHE_FILE.exists() and not refresh:
        df = pd.read_csv(CACHE_FILE, index_col="date", parse_dates=True)
        return df

    df = download_close_series()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(CACHE_FILE)
    return df


def log_returns(spx_close: pd.Series) -> pd.Series:
    """Daily close-to-close log returns.

    The first value is NaN by construction and is dropped downstream.
    """
    return np.log(spx_close / spx_close.shift(1))
