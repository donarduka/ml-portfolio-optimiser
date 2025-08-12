from datetime import date

# assets and date range
TICKERS = ["SPY", "QQQ", "TLT", "GLD"]
START = "2012-01-01"
END = date.today().isoformat()

# backtest parameters
REBALANCE = "M" # monthly
TCOST_BPS = 5 # basis points
MAX_WEIGHT = 0.6
MIN_WEIGHT = 0.0
LONG_ONLY = True
RISK_AVERSION = 5.0

# ML parameters
N_REGIMES = 2
