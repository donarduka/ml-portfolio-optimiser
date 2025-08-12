import pandas as pd
from mlo.features import log_returns, make_feature_matrix

def test_log_returns_shape():
    # simple price series: expect n-1 rows after diff
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"A": [100, 101, 100, 102]}, index=idx)
    r = log_returns(prices)
    assert r.shape[0] == 3
    assert "A" in r.columns

def test_make_feature_matrix_columns():
    # 80 days of dummy prices, 1 asset
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    prices = pd.DataFrame({"SPY": range(1, 81)}, index=idx)
    X = make_feature_matrix(prices)
    assert {"vol", "mom", "abs_ret"}.issubset(X.columns)
