import pandas as pd
from .features import log_returns

_FREQ_MAP = {"M": "ME"}  # pandas: month-end alias

def rebalance_dates(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Rebalance dates snapped to trading days in idx."""
    freq = _FREQ_MAP.get(freq, freq)
    return pd.date_range(idx.min(), idx.max(), freq=freq).intersection(idx)

def backtest(
    prices: pd.DataFrame,
    weights_by_date: pd.DataFrame,
    tcost_bps: float = 5,
) -> pd.DataFrame:
    """Vectorised backtest with simple turnover costs."""
    if prices.empty or weights_by_date.empty:
        raise ValueError("prices/weights_by_date cannot be empty")

    # keep common assets only
    cols = prices.columns.intersection(weights_by_date.columns)
    prices, weights_by_date = prices[cols], weights_by_date[cols]

    rets = log_returns(prices)

    # hold last target weights between rebalances
    weights = weights_by_date.reindex(rets.index).ffill().fillna(0.0)

    # turnover cost in bps
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    costs = (tcost_bps / 1e4) * turnover

    # use yesterday's weights on today's returns
    port_ret = (weights.shift().fillna(0.0) * rets).sum(axis=1) - costs
    equity = (1.0 + port_ret).cumprod()

    return pd.DataFrame({"ret": port_ret, "equity": equity})

def perf_stats(port: pd.DataFrame) -> dict:
    """Annualised return/vol, Sharpe, max drawdown."""
    r = port["ret"]
    ann = 252
    mean = r.mean() * ann
    vol = r.std() * ann**0.5
    sharpe = mean / vol if vol > 0 else float("nan")
    dd = (port["equity"] / port["equity"].cummax() - 1.0).min()
    return {"ann_return": mean, "ann_vol": vol, "sharpe": sharpe, "max_drawdown": dd}
