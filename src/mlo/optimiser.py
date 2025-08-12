import cvxpy as cp
import numpy as np
import pandas as pd

def mean_variance_weights(
    mu: pd.Series,
    cov: pd.DataFrame,
    lam: float, # risk aversion
    max_w: float,
    min_w: float,
    long_only: bool = True,
) -> pd.Series:
    """Mean–variance optimiser with box constraints; weights sum to 1."""
    assets = mu.index.tolist()
    n = len(assets)

    # align and make covariance symmetric
    C = cov.reindex(index=assets, columns=assets).astype(float)
    Sigma = ((C + C.T) / 2).values + 1e-6 * np.eye(n) # small ridge

    w = cp.Variable(n)
    obj = cp.Maximize(mu.values @ w - lam * cp.quad_form(w, Sigma))

    cons = [cp.sum(w) == 1, w <= max_w, w >= min_w]
    if long_only:
        cons.append(w >= 0)

    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        prob.solve(verbose=False)

    if w.value is None or not np.isfinite(w.value).all():
        # fallback: clipped equal weights
        ew = np.clip(np.ones(n) / n, min_w, max_w)
        ew = ew / ew.sum()
        return pd.Series(ew, index=assets)

    out = np.clip(w.value, min_w, max_w)
    out = out / out.sum()
    return pd.Series(out, index=assets)
