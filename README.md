# ML Portfolio Optimiser

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A regime-based portfolio optimisation system using K-Means clustering to detect market environments and mean-variance optimisation with Ledoit-Wolf shrinkage covariance to allocate weights within each regime.

## Strategy

- Download adjusted close prices from Yahoo Finance (SPY, QQQ, TLT, GLD)
- Build market features: rolling volatility, momentum, mean absolute return
- Cluster into market regimes using K-Means
- At each monthly rebalance, estimate mean and shrinkage covariance from regime-matched history
- Solve mean-variance optimisation with position caps using CVXPY
- Backtest with monthly rebalancing and transaction costs

## Results (2012–present)

| Metric | Value |
|--------|-------|
| Annualised Return | 10.72% |
| Annualised Volatility | 15.36% |
| Sharpe Ratio | 0.70 |
| Max Drawdown | -31.00% |

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m mlo.run
```

Outputs are saved to `reports/`.

## Repository Layout

```
ml-portfolio-optimiser/
├── src/
│   └── mlo/
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── regime.py
│       ├── optimiser.py
│       ├── backtest.py
│       ├── report.py
│       └── run.py
├── tests/
│   ├── test_optimizer.py
│   ├── test_features.py
│   └── test_backtest.py
├── reports/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TICKERS` | SPY QQQ TLT GLD | Asset universe |
| `START` | 2012-01-01 | Backtest start date |
| `N_REGIMES` | 3 | Number of K-Means clusters |
| `RISK_AVERSION` | 2.0 | Mean-variance risk aversion parameter |
| `MAX_WEIGHT` | 0.6 | Maximum weight per asset |
| `TCOST_BPS` | 5 | Transaction cost in basis points |
| `REBALANCE` | M | Rebalance frequency (M = monthly) |

## Running Tests

```bash
pytest tests/
```

## Notes

- Ledoit-Wolf shrinkage is used in place of sample covariance for stability on short regime windows
- Falls back to the last 252 trading days if regime history is insufficient
- Production extensions would include walk-forward parameter selection and out-of-sample regime validation

## License

MIT