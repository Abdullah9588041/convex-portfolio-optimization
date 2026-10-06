"""Honest out-of-sample backtest with no look-ahead bias.

Protocol
--------
* Expanding estimation window: at each month-end rebalance date ``t`` only
  returns strictly before ``t`` are used to estimate moments and optimize.
* Monthly rebalancing; the chosen weights are held until the next rebalance.
* Transaction costs: one-way turnover ``0.5 * ||w_new - w_old||_1`` is charged
  ``transaction_cost_bps`` basis points, applied as a one-time hit to equity
  on the rebalance date. Default 10 bps — stated explicitly, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .moments import TRADING_DAYS, ledoit_wolf_cov, portfolio_stats, sample_moments
from .optimize import Constraints, SolveResult, max_sharpe, min_variance

StrategyFn = Callable[[pd.DataFrame, list[str]], np.ndarray]


@dataclass
class BacktestConfig:
    min_lookback_days: int = 504  # ~2 trading years before first rebalance
    rebalance_rule: str = "ME"  # month-end
    transaction_cost_bps: float = 10.0
    risk_free: float = 0.0  # annualized; Sharpe ratios use this
    max_weight: float = 0.10
    sector_cap: float = 0.35


def _default_constraints(cfg: BacktestConfig, sector_of: dict[str, str]) -> Constraints:
    return Constraints(
        long_only=True,
        fully_invested=True,
        max_weight=cfg.max_weight,
        sector_caps={s: cfg.sector_cap for s in set(sector_of.values())},
        sector_of=sector_of,
    )


def max_sharpe_lw_strategy(
    window: pd.DataFrame, tickers: list[str], cfg: BacktestConfig,
    sector_of: dict[str, str],
) -> np.ndarray:
    mu_d, sigma_d, _ = ledoit_wolf_cov(window)
    mu_a, sigma_a = mu_d * TRADING_DAYS, sigma_d * TRADING_DAYS
    res = max_sharpe(mu_a.values, sigma_a.values, tickers,
                     risk_free=cfg.risk_free,
                     constraints=_default_constraints(cfg, sector_of))
    return res.weights


def min_var_sample_strategy(
    window: pd.DataFrame, tickers: list[str], cfg: BacktestConfig,
    sector_of: dict[str, str],
) -> np.ndarray:
    mu_d, sigma_d = sample_moments(window.dropna())
    mu_a, sigma_a = mu_d * TRADING_DAYS, sigma_d * TRADING_DAYS
    res = min_variance(mu_a.values, sigma_a.values, tickers,
                       constraints=_default_constraints(cfg, sector_of),
                       risk_free=cfg.risk_free)
    return res.weights


def equal_weight_strategy(
    window: pd.DataFrame, tickers: list[str], cfg: BacktestConfig,
    sector_of: dict[str, str],
) -> np.ndarray:
    return np.full(len(tickers), 1.0 / len(tickers))


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return float(dd.min())


def performance_metrics(daily_returns: pd.Series, risk_free: float = 0.0) -> dict[str, float]:
    """CAGR, annualized vol, Sharpe, max drawdown from a daily simple-return series."""
    r = daily_returns.dropna()
    n = len(r)
    if n == 0:
        return {"cagr": float("nan"), "volatility": float("nan"),
                "sharpe": float("nan"), "max_drawdown": float("nan"),
                "n_days": 0}
    equity = (1 + r).cumprod()
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan")
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    sharpe = float((r.mean() * TRADING_DAYS - risk_free) / vol) if vol > 0 else float("nan")
    return {
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
        "n_days": n,
        "final_equity": float(equity.iloc[-1]),
    }


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    daily_returns: pd.Series
    weights_history: pd.DataFrame
    turnover: pd.Series  # one-way turnover at each rebalance
    metrics: dict[str, float]
    total_cost_drag: float  # cumulative return lost to transaction costs


def run_backtest(
    prices: pd.DataFrame,
    name: str,
    strategy: StrategyFn,
    cfg: BacktestConfig,
    sector_of: dict[str, str],
    start: str | None = None,
    end: str | None = None,
) -> BacktestResult:
    tickers = list(prices.columns)
    prices = prices.loc[start:end] if (start or end) else prices
    returns = np.log(prices / prices.shift(1)).dropna()

    trading_days = returns.index
    first_possible = trading_days[cfg.min_lookback_days]
    rebalance_dates = pd.DatetimeIndex(
        [d for d in trading_days.to_series().resample(cfg.rebalance_rule).last().index
         if d >= first_possible and d <= trading_days[-1]]
    )
    # map resample labels back onto actual trading days (last trading day <= label)
    rebalance_dates = trading_days[
        trading_days.searchsorted(rebalance_dates, side="right") - 1
    ]
    rebalance_dates = pd.DatetimeIndex(sorted(set(rebalance_dates)))

    equity = pd.Series(index=trading_days, dtype=float)
    equity.iloc[0] = 1.0
    weights_hist: dict = {}
    turnover_hist: dict = {}
    w_old = np.zeros(len(tickers))
    total_cost = 0.0

    rb_set = set(rebalance_dates)
    for i, day in enumerate(trading_days):
        if day in rb_set:
            window = returns.loc[:day].iloc[:-1]  # strictly before t: no look-ahead
            w_new = np.asarray(strategy(window, tickers, cfg, sector_of), dtype=float)
            one_way_turnover = 0.5 * np.abs(w_new - w_old).sum()
            cost = one_way_turnover * cfg.transaction_cost_bps / 1e4
            total_cost += cost
            equity.iloc[i] = equity.iloc[i - 1] * (1 - cost) if i > 0 else 1.0 * (1 - cost)
            w_old = w_new
            weights_hist[day] = w_new
            turnover_hist[day] = one_way_turnover
        elif i > 0:
            if w_old.sum() == 0:
                equity.iloc[i] = equity.iloc[i - 1]  # not yet invested: hold cash
                continue
            # hold: portfolio grows with the weighted daily simple returns
            gross = float(np.dot(w_old, np.exp(returns.loc[day].values)))
            w_old = w_old * np.exp(returns.loc[day].values) / gross  # drift
            equity.iloc[i] = equity.iloc[i - 1] * gross
    equity = equity.ffill()
    # Start the reported track record at the first rebalance: the ~2y
    # estimation window is held as cash and is not part of the strategy.
    first_rb = rebalance_dates[0]
    equity = (equity.loc[first_rb:] / equity.loc[first_rb]).astype(float)
    daily_ret = equity.pct_change().dropna()
    metrics = performance_metrics(daily_ret, cfg.risk_free)
    return BacktestResult(
        name=name,
        equity=equity,
        daily_returns=daily_ret,
        weights_history=pd.DataFrame(weights_hist).T,
        turnover=pd.Series(turnover_hist),
        metrics=metrics,
        total_cost_drag=total_cost,
    )


def buy_and_hold(prices: pd.Series, name: str, risk_free: float = 0.0) -> BacktestResult:
    """Buy-and-hold benchmark (e.g. SPY)."""
    equity = (prices / prices.iloc[0]).astype(float)
    daily_ret = equity.pct_change().dropna()
    return BacktestResult(
        name=name, equity=equity, daily_returns=daily_ret,
        weights_history=pd.DataFrame(), turnover=pd.Series(dtype=float),
        metrics=performance_metrics(daily_ret, risk_free), total_cost_drag=0.0,
    )
