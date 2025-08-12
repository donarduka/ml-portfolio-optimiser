# ML Portfolio Optimiser

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

