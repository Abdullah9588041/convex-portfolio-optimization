"""Return-moment estimation: sample moments and Ledoit-Wolf shrinkage.

We deliberately use scikit-learn's ``LedoitWolf`` implementation rather than a
hand-rolled one: it is the numerically careful reference implementation of
Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def sample_moments(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Sample mean vector and sample covariance of (daily) returns."""
    mu = returns.mean()
    sigma = returns.cov()
    return mu, sigma


def ledoit_wolf_cov(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, float]:
    """Ledoit-Wolf shrunk covariance; returns (mu, Sigma_shrunk, shrinkage_).

    ``shrinkage_`` is the optimal convex-combination weight on the structured
    target  mu*I  (scaled identity); 0 = pure sample covariance.
    """
    clean = returns.dropna()
    lw = LedoitWolf().fit(clean.values)
    sigma = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    mu = clean.mean()
    return mu, sigma, float(lw.shrinkage_)


def annualize(mu_daily: pd.Series, sigma_daily: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Annualize daily mean / covariance (252 trading days)."""
    return mu_daily * TRADING_DAYS, sigma_daily * TRADING_DAYS


def portfolio_stats(
    weights: np.ndarray, mu: pd.Series | np.ndarray, sigma: pd.DataFrame | np.ndarray,
    risk_free: float = 0.0,
) -> dict[str, float]:
    """Expected return, volatility and Sharpe of a weight vector.

    ``mu``/``sigma`` must already be on the same (e.g. annualized) scale as
    ``risk_free``.
    """
    w = np.asarray(weights, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    port_ret = float(w @ mu_arr)
    port_vol = float(np.sqrt(w @ sigma_arr @ w))
    sharpe = (port_ret - risk_free) / port_vol if port_vol > 0 else float("nan")
    return {"return": port_ret, "volatility": port_vol, "sharpe": sharpe}
