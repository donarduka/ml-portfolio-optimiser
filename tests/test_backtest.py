import pandas as pd
from mlo.backtest import backtest

def test_backtest_runs_and_has_equity():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame(
        {"A": [100, 101, 102, 103, 104, 105],
         "B": [50, 49, 51, 52, 53, 54]},
        index=idx,
    )
    # one rebalance at t0; weights are held forward
    weights = pd.DataFrame([{"A": 0.6, "B": 0.4}], index=[idx[0]])

    out = backtest(prices, weights, tcost_bps=5)

    assert {"ret", "equity"}.issubset(out.columns)
    assert len(out) > 0
    assert out["equity"].iloc[-1] > 0
    assert out["ret"].isna().sum() == 0
