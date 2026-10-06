"""Tests for convex-portfolio-optimization.

Run with:  pytest
"""

import numpy as np
import pandas as pd
import pytest

from portopt.backtest import BacktestConfig, buy_and_hold, performance_metrics, run_backtest, equal_weight_strategy
from portopt.data import SECTORS, TICKERS
from portopt.moments import ledoit_wolf_cov, portfolio_stats, sample_moments
from portopt.optimize import Constraints, efficient_frontier, max_sharpe, min_variance


@pytest.fixture
def toy_market():
    """Small synthetic market with a fixed seed: no network needed."""
    rng = np.random.default_rng(0)
    n_days, n_assets = 400, 6
    tickers = [f"A{i}" for i in range(n_assets)]
    # block-structured covariance so optimization has something to do
    cov = np.eye(n_assets) * 0.0004
    cov[:3, :3] += 0.0003
    mu = np.array([0.0006, 0.0004, 0.0002, 0.0005, 0.0001, 0.0003])
    rets = rng.multivariate_normal(mu, cov, size=n_days)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    returns = pd.DataFrame(rets, index=dates, columns=tickers)
    prices = 100 * np.exp(returns.cumsum())
    prices = pd.DataFrame(prices, index=dates, columns=tickers)
    sigma = pd.DataFrame(cov, index=tickers, columns=tickers)
    return tickers, prices, returns, pd.Series(mu, index=tickers), sigma


@pytest.fixture
def constraints():
    return Constraints(long_only=True, fully_invested=True, max_weight=0.5)


def test_weights_sum_to_one(toy_market, constraints):
    tickers, _, _, mu, sigma = toy_market
    res = min_variance(mu.values, sigma.values, tickers, constraints=constraints)
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_long_only_and_max_weight(toy_market, constraints):
    tickers, _, _, mu, sigma = toy_market
    res = max_sharpe(mu.values, sigma.values, tickers, constraints=constraints)
    assert (res.weights >= -1e-8).all()
    assert (res.weights <= 0.5 + 1e-8).all()


def test_min_variance_beats_equal_weight_in_sample(toy_market, constraints):
    """On the *estimation* covariance, min-var must not lose to 1/N."""
    tickers, _, _, mu, sigma = toy_market
    res = min_variance(mu.values, sigma.values, tickers, constraints=constraints)
    ew = np.full(len(tickers), 1 / len(tickers))
    assert res.volatility <= float(np.sqrt(ew @ sigma.values @ ew)) + 1e-10


def test_frontier_monotone_volatility(toy_market, constraints):
    tickers, _, _, mu, sigma = toy_market
    front = efficient_frontier(mu.values, sigma.values, tickers, n_points=8,
                               constraints=constraints)
    vols = front["volatility"].values
    assert np.all(np.diff(vols) >= -1e-9), "frontier volatility must be non-decreasing"


def test_cvxpy_scipy_agree(toy_market, constraints):
    tickers, _, _, mu, sigma = toy_market
    a = min_variance(mu.values, sigma.values, tickers, constraints=constraints,
                     solver="cvxpy")
    b = min_variance(mu.values, sigma.values, tickers, constraints=constraints,
                     solver="scipy")
    assert np.linalg.norm(a.weights - b.weights) < 1e-3


def test_kkt_stationarity_min_variance(toy_market):
    """Stationarity residual of the KKT conditions for the min-var solution.

    min  w'Σw  s.t.  sum w = 1, w >= 0   =>
       2Σw - λ1 + ν = 0,  ν >= 0,  ν_i w_i = 0.
    With ν_i = max(0, λ - 2(Σw)_i) the residual must vanish at optimum.
    """
    tickers, _, _, mu, sigma = toy_market
    cons = Constraints(long_only=True, fully_invested=True, max_weight=None)
    res = min_variance(mu.values, sigma.values, tickers, constraints=cons)
    w = res.weights
    S = sigma.values
    grad = 2 * S @ w
    lam = grad @ w  # from w'(2Σw - λ1) = 0  =>  λ = 2 w'Σw
    nu = np.maximum(0.0, lam - grad)
    residual = grad - lam + nu
    assert np.linalg.norm(residual) < 1e-5
    assert np.all(nu * w < 1e-6)  # complementary slackness


def test_turnover_constraint_respected(toy_market):
    tickers, _, _, mu, sigma = toy_market
    w0 = np.full(len(tickers), 1 / len(tickers))
    cons = Constraints(long_only=True, fully_invested=True, max_weight=0.5,
                       turnover_limit=0.05, current_weights=w0)
    res = min_variance(mu.values, sigma.values, tickers, constraints=cons)
    assert 0.5 * np.abs(res.weights - w0).sum() <= 0.05 + 1e-8


def test_ledoit_wolf_psd(toy_market):
    _, _, returns, _, _ = toy_market
    _, sigma_lw, shrinkage = ledoit_wolf_cov(returns)
    eigvals = np.linalg.eigvalsh(sigma_lw.values)
    assert (eigvals > 0).all()
    assert 0.0 <= shrinkage <= 1.0


def test_portfolio_stats_sharpe():
    w = np.array([0.5, 0.5])
    mu = pd.Series([0.10, 0.06])
    sigma = pd.DataFrame(np.eye(2) * 0.04, index=[0, 1], columns=[0, 1])
    s = portfolio_stats(w, mu, sigma, risk_free=0.0)
    assert s["return"] == pytest.approx(0.08)
    assert s["volatility"] == pytest.approx(np.sqrt(0.02), rel=1e-9)
    assert s["sharpe"] == pytest.approx(0.08 / np.sqrt(0.02), rel=1e-9)


def test_backtest_accounting_identity(toy_market):
    """Equity curve must equal the product of net daily growth factors."""
    tickers, prices, _, _, _ = toy_market
    sector_of = {t: "S" for t in tickers}
    cfg = BacktestConfig(min_lookback_days=100, transaction_cost_bps=10.0)
    res = run_backtest(prices, "test", equal_weight_strategy, cfg, sector_of)
    assert res.equity.iloc[-1] == pytest.approx((1 + res.daily_returns).prod(), rel=1e-9)


def test_backtest_no_lookahead(toy_market):
    """With zero signal the backtest still runs; weights history aligns to rebalances."""
    tickers, prices, _, _, _ = toy_market
    sector_of = {t: "S" for t in tickers}
    cfg = BacktestConfig(min_lookback_days=100)
    res = run_backtest(prices, "test", equal_weight_strategy, cfg, sector_of)
    assert len(res.turnover) == len(res.weights_history) > 0
    first_rb = res.turnover.index[0]
    assert (res.weights_history.loc[first_rb].values == pytest.approx(1 / len(tickers)))


def test_buy_and_hold_metrics(toy_market):
    _, prices, _, _, _ = toy_market
    res = buy_and_hold(prices["A0"], "bench")
    assert res.equity.iloc[0] == pytest.approx(1.0)
    assert res.metrics["n_days"] == len(prices) - 1


def test_universe_documented():
    assert len(TICKERS) >= 30
    assert set(TICKERS) == set(SECTORS.keys())
