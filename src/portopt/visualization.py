"""Figures for the analysis: frontier, weights, equity curves, drawdowns."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_frontier(frontier: pd.DataFrame, points: dict[str, tuple[float, float]],
                  save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(frontier["volatility"], frontier["expected_return"], "b-", lw=2,
            label="Efficient frontier")
    for label, (vol, ret) in points.items():
        ax.scatter([vol], [ret], s=80, zorder=5, label=label)
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.set_title("Mean-Variance Efficient Frontier (with constraints)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_weights(weights: np.ndarray, tickers: list[str], title: str,
                 save_path: str | Path) -> None:
    order = np.argsort(weights)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([tickers[i] for i in order], weights[order])
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_equity_curves(equities: dict[str, pd.Series], save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, eq in equities.items():
        ax.plot(eq.index, eq.values, label=name, lw=1.8)
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_title("Out-of-sample backtest: growth of $1")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_drawdown(equity: pd.Series, name: str, save_path: str | Path) -> None:
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(dd.index, dd.values * 100, 0, alpha=0.6)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(f"Drawdown — {name}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
