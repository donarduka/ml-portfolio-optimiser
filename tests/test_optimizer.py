import numpy as np
import pandas as pd
from mlo.optimiser import mean_variance_weights

def test_weights_sum_to_one_and_boxed():
    # basic case: 3 assets, identity covariance
    mu = pd.Series([0.1, 0.08, 0.05], index=["A", "B", "C"])
    cov = pd.DataFrame(np.eye(3), index=mu.index, columns=mu.index)
    w = mean_variance_weights(mu, cov, lam=5.0, max_w=0.8, min_w=0.0, long_only=True)

    assert abs(w.sum() - 1) < 1e-6
    assert (w >= 0).all()
    assert (w <= 0.8).all()
