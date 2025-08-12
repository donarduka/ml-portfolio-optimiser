# ML Portfolio Optimiser

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python project for regime-based portfolio optimisation using historical market data.

**Strategy overview**
- Download prices from Yahoo Finance (SPY, QQQ, TLT, GLD).
- Build features: volatility, momentum, absolute returns.
- Cluster market regimes with K-Means.
- Run mean–variance optimisation within each regime.
- Backtest with monthly rebalancing and transaction costs.

---

## Requirements
- Python 3.11+
- Packages in `pyproject.toml`
- Data source: Yahoo Finance

---

## Usage

```bash
# activate env
source .venv/bin/activate

# run strategy (PYTHONPATH set via .env)
python -m mlo.run

