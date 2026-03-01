import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from .backtest import perf_stats

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
BG = "#FAFAFA"
GRID = "#E0E0E0"


def _style(ax):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#555555", labelsize=9)


def save_report(port: pd.DataFrame, weights_by_date: pd.DataFrame) -> None:
    os.makedirs("reports", exist_ok=True)

    stats = perf_stats(port)
    with open("reports/summary.txt", "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v:.4f}\n")

    # equity curve
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(BG)
    _style(ax)
    ax.plot(port.index, port["equity"].values, color=PALETTE[0], linewidth=1.8)
    ax.set_title("Portfolio Equity Curve", fontsize=13, fontweight="bold", pad=10, color="#222222")
    ax.set_ylabel("Portfolio Value (normalised)", fontsize=10, color="#555555")
    ax.annotate(
        f"Sharpe: {stats['sharpe']:.2f}   Ann. Return: {stats['ann_return']*100:.1f}%   Max DD: {stats['max_drawdown']*100:.1f}%",
        xy=(0.01, 0.97), xycoords="axes fraction", fontsize=9, color="#555555", va="top"
    )
    plt.tight_layout()
    plt.savefig("reports/equity.png", dpi=200, bbox_inches="tight")
    plt.close()

    # weights over time
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(BG)
    _style(ax)
    weights_by_date.plot.area(ax=ax, color=PALETTE[:len(weights_by_date.columns)], alpha=0.85)
    ax.set_title("Portfolio Weight Allocation Over Time", fontsize=13, fontweight="bold", pad=10, color="#222222")
    ax.set_ylabel("Weight", fontsize=10, color="#555555")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
    plt.tight_layout()
    plt.savefig("reports/weights.png", dpi=200, bbox_inches="tight")
    plt.close()