import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import (
    REBALANCE, TCOST_BPS, MAX_WEIGHT, MIN_WEIGHT, LONG_ONLY, RISK_AVERSION, N_REGIMES
)
from .data import fetch_prices
from .features import log_returns, make_feature_matrix
from .regime import fit_regimes, label_regimes
from .optimiser import mean_variance_weights
from .backtest import rebalance_dates, backtest, perf_stats
from .report import save_report


def main() -> None:
    prices = fetch_prices()
    rets = log_returns(prices)

    X = make_feature_matrix(prices)
    km = fit_regimes(X, N_REGIMES)
    regimes = label_regimes(X, km)

    r_dates = rebalance_dates(rets.index, REBALANCE)
    weights = []

    for dt in r_dates:
        hist = rets.loc[:dt]
        if len(hist) < 252:
            continue

        reg_now = regimes.loc[:dt].iloc[-1]
        mask = (regimes == reg_now).reindex(hist.index).fillna(False).astype(bool)
        r_reg = hist[mask].dropna()
        if len(r_reg) < 60:
            r_reg = hist.tail(252)

        mu = r_reg.mean()
        lw = LedoitWolf().fit(r_reg)
        cov = pd.DataFrame(lw.covariance_, index=r_reg.columns, columns=r_reg.columns)

        w = mean_variance_weights(
            mu=mu,
            cov=cov,
            lam=RISK_AVERSION,
            max_w=MAX_WEIGHT,
            min_w=MIN_WEIGHT,
            long_only=LONG_ONLY,
        )
        w.name = dt
        weights.append(w)

    if not weights:
        raise RuntimeError("Not enough data to compute any rebalance points")

    weights_df = pd.DataFrame(weights)
    port = backtest(prices, weights_df, tcost_bps=TCOST_BPS)

    os.makedirs("reports", exist_ok=True)
    port.to_csv("reports/portfolio.csv")
    weights_df.to_csv("reports/weights.csv")

    save_report(port, weights_df)

    stats = perf_stats(port)
    print("--- Summary ---")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")
    print("Outputs saved to: reports/")


if __name__ == "__main__":
    main()