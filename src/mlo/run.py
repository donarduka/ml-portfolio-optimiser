import os
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
from .backtest import rebalance_dates, backtest
from .report import save_report


def main() -> None:
    # data
    prices = fetch_prices()
    rets = log_returns(prices)

    # regimes
    X = make_feature_matrix(prices)
    km = fit_regimes(X, N_REGIMES) # k-means for market regimes
    regimes = label_regimes(X, km)

    # monthly (end) rebalancing with expanding history
    r_dates = rebalance_dates(rets.index, REBALANCE)
    weights = []

    for dt in r_dates:
        hist = rets.loc[:dt]
        if len(hist) < 252:  # need ~1y history
            continue

        reg_now = regimes.loc[:dt].iloc[-1]
        # use history from the current regime; fallback to last 252d if too short
        mask = (regimes == reg_now).reindex(hist.index).fillna(False).astype(bool)
        r_reg = hist[mask].dropna()
        if len(r_reg) < 60:
            r_reg = hist.tail(252)

        mu = r_reg.mean()
        # shrinkage covariance is more stable than sample cov
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

    # SPY benchmark (use arithmetic returns to match (1+ret).cumprod())
    spy = prices["SPY"].reindex(port.index).ffill()
    spy_ret = spy.pct_change().fillna(0)
    port["spy_equity"] = (1 + spy_ret).cumprod()

    # persist outputs
    os.makedirs("reports", exist_ok=True)
    port.to_csv("reports/portfolio.csv")
    weights_df.to_csv("reports/weights.csv")

    save_report(port, weights_df)
    print("Done. See reports/summary.txt, reports/equity.png and reports/weights.png")


if __name__ == "__main__":
    main()
# This script is the entry point for running the portfolio optimisation and backtesting.
# It fetches data, computes features, identifies market regimes,
# optimises portfolio weights, runs backtests, and saves reports.
# The main function orchestrates the entire workflow.