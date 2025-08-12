import pandas as pd
import numpy as np

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns from price levels."""
    return np.log(prices).diff().dropna()

def rolling_vol(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Rolling stdev (default ~1 trading month)."""
    return returns.rolling(window).std().dropna()

def momentum(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """% change over a window (default ~3 months)."""
    return prices.pct_change(window).dropna()

def make_feature_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix: avg vol, avg momentum, avg abs return."""
    r = log_returns(prices)
    vol = rolling_vol(r)
    mom = momentum(prices)

    # align on the same index before concatenation
    X = pd.concat(
        {
            "vol": vol.mean(axis=1),
            "mom": mom.mean(axis=1).reindex(vol.index).ffill(),
            "abs_ret": r.abs().mean(axis=1).reindex(vol.index).ffill(),
        },
        axis=1,
    ).dropna()

    return X
