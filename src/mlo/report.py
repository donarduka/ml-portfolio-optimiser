import os
import pandas as pd
import matplotlib.pyplot as plt
from .backtest import perf_stats

def save_report(port: pd.DataFrame, weights_by_date: pd.DataFrame) -> None:
    """Save backtest summary stats and plots to the reports folder."""
    os.makedirs("reports", exist_ok=True)

    # write summary text
    stats = perf_stats(port)
    with open("reports/summary.txt", "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v:.4f}\n")

    # equity curve
    plt.figure()
    port["equity"].plot(title="Portfolio Equity Curve")
    plt.tight_layout()
    plt.savefig("reports/equity.png")
    plt.close()

    # weights over time
    plt.figure()
    weights_by_date.plot.area(title="Weights Over Time", figsize=(8, 4))
    plt.tight_layout()
    plt.savefig("reports/weights.png")
    plt.close()

    # equity vs SPY benchmark
    if "spy_equity" in port.columns:
        plt.figure()
        port[["equity", "spy_equity"]].plot(title="Equity vs SPY")
        plt.tight_layout()
        plt.savefig("reports/equity_vs_spy.png")
        plt.close()
