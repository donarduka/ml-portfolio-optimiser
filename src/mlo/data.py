import os
import pandas as pd
import yfinance as yf

from .config import TICKERS, START, END

CACHE_PATH = "data/prices.csv"

def fetch_prices(force: bool = False) -> pd.DataFrame:
    """Get adjusted close prices for TICKERS, cached locally."""
    os.makedirs("data", exist_ok=True)

    # load from cache unless force refresh
    if not force and os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)

    df = yf.download(
        TICKERS,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
    )["Close"]

    df = df.dropna(how="all").ffill()
    df.to_csv(CACHE_PATH)
    return df
