"""Mean-variance optimization via convex quadratic programming.

Two solver backends are provided for the same problems so the analysis can
compare them honestly:

* ``cvxpy``  — disciplined convex programming (default solver: Clarabel).
* ``scipy``  — ``scipy.optimize.minimize`` with SLSQP.

Max-Sharpe via cvxpy uses the standard convex reformulation
(Cornuejols & Tütüncü): with excess returns ``e = mu - rf``,

    min_w  w' Σ w   s.t.  e'w = 1,  w in C

and the optimal portfolio is ``x* = w* / sum(w*)``.  Linear constraints on the
*normalized* weights ``x`` (box, sector caps, turnover) become linear
constraints in ``w`` because ``x_i <= c``  <=>  ``w_i <= c * sum(w)``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class Constraints:
    long_only: bool = True
    fully_invested: bool = True
    max_weight: float | None = 0.10
    sector_caps: dict[str, float] | None = None
    sector_of: dict[str, str] | None = None
    turnover_limit: float | None = None
    current_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.turnover_limit is not None and self.current_weights is None:
            raise ValueError("turnover_limit requires current_weights")


@dataclass
class SolveResult:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe: float
    solver: str
    solve_time_s: float
    status: str


def _sector_index(tickers: list[str], sector_of: dict[str, str]) -> dict[str, list[int]]:
    idx: dict[str, list[int]] = {}
    for i, t in enumerate(tickers):
        idx.setdefault(sector_of[t], []).append(i)
    return idx


def _cvxpy_constraints(
    w: cp.Variable,
    tickers: list[str],
    constraints: Constraints,
    normalized: bool,
    scale: cp.Expression | None = None,
) -> list:
    """Linear constraints. If ``normalized`` is False, box/sector/turnover
    constraints apply to ``w / sum(w)`` via the ``scale = sum(w)`` trick."""
    cons: list = []
    n = len(tickers)
    s = 1.0 if normalized else scale  # type: ignore[assignment]

    if constraints.long_only:
        cons.append(w >= 0)
    if constraints.fully_invested and normalized:
        cons.append(cp.sum(w) == 1)
    if constraints.max_weight is not None:
        cons.append(w <= constraints.max_weight * s)
    if constraints.sector_caps and constraints.sector_of:
        sec_idx = _sector_index(tickers, constraints.sector_of)
        for sector, cap in constraints.sector_caps.items():
            members = sec_idx.get(sector, [])
            if members:
                cons.append(cp.sum(w[members]) <= cap * s)
    if constraints.turnover_limit is not None:
        w0 = np.asarray(constraints.current_weights, dtype=float)
        cons.append(cp.norm(w - w0 * s, 1) <= constraints.turnover_limit * s)
    return cons


def min_variance(
    mu: np.ndarray,
    sigma: np.ndarray,
    tickers: list[str],
    target_return: float | None = None,
    constraints: Constraints | None = None,
    solver: str = "cvxpy",
    risk_free: float = 0.0,
) -> SolveResult:
    """Minimum-variance portfolio, optionally hitting ``target_return``."""
    constraints = constraints or Constraints()
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(tickers)
    t0 = time.perf_counter()

    if solver == "cvxpy":
        w = cp.Variable(n)
        cons = _cvxpy_constraints(w, tickers, constraints, normalized=True)
        if target_return is not None:
            cons.append(mu @ w >= target_return)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, sigma)), cons)
        prob.solve()
        status = prob.status
        weights = np.asarray(w.value).ravel()
    elif solver == "scipy":
        cons_list: list = []
        if constraints.fully_invested:
            cons_list.append({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
        if target_return is not None:
            cons_list.append(
                {"type": "ineq", "fun": lambda w, mu=mu: mu @ w - target_return}
            )
        bounds = [(0.0, constraints.max_weight or 1.0)] * n if constraints.long_only else None
        # sector caps / turnover as nonlinear constraints for scipy
        if constraints.sector_caps and constraints.sector_of:
            sec_idx = _sector_index(tickers, constraints.sector_of)
            for sector, cap in constraints.sector_caps.items():
                members = sec_idx.get(sector, [])
                if members:
                    cons_list.append(
                        {"type": "ineq",
                         "fun": lambda w, m=members: cap - np.sum(w[m])}
                    )
        if constraints.turnover_limit is not None:
            w0 = np.asarray(constraints.current_weights, dtype=float)
            cons_list.append(
                {"type": "ineq",
                 "fun": lambda w: constraints.turnover_limit - np.sum(np.abs(w - w0))}
            )
        x0 = np.full(n, 1.0 / n)
        res = minimize(
            lambda w: float(w @ sigma @ w), x0, method="SLSQP",
            bounds=bounds, constraints=cons_list,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        status = "optimal" if res.success else f"scipy: {res.message}"
        weights = np.clip(res.x, 0, None)
        weights /= weights.sum()
    else:
        raise ValueError(f"unknown solver {solver!r}")

    dt = time.perf_counter() - t0
    vol = float(np.sqrt(weights @ sigma @ weights))
    ret = float(mu @ weights)
    sharpe = (ret - risk_free) / vol if vol > 0 else float("nan")
    return SolveResult(weights, ret, vol, sharpe, solver, dt, str(status))


def max_sharpe(
    mu: np.ndarray,
    sigma: np.ndarray,
    tickers: list[str],
    risk_free: float = 0.0,
    constraints: Constraints | None = None,
    solver: str = "cvxpy",
) -> SolveResult:
    """Maximum-Sharpe (tangency) portfolio."""
    constraints = constraints or Constraints()
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(tickers)
    t0 = time.perf_counter()

    if solver == "cvxpy":
        e = mu - risk_free
        w = cp.Variable(n)
        k = cp.sum(w)
        cons = _cvxpy_constraints(w, tickers, constraints, normalized=False, scale=k)
        cons.append(e @ w == 1)
        # avoid the degenerate w = 0 direction being "optimal"
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, sigma)), cons)
        prob.solve()
        status = prob.status
        w_val = np.asarray(w.value).ravel()
        weights = w_val / w_val.sum()
    elif solver == "scipy":
        def neg_sharpe(w: np.ndarray) -> float:
            vol = np.sqrt(w @ sigma @ w)
            return -((mu @ w - risk_free) / vol) if vol > 0 else 0.0

        cons_list: list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, constraints.max_weight or 1.0)] * n
        if constraints.sector_caps and constraints.sector_of:
            sec_idx = _sector_index(tickers, constraints.sector_of)
            for sector, cap in constraints.sector_caps.items():
                members = sec_idx.get(sector, [])
                if members:
                    cons_list.append(
                        {"type": "ineq", "fun": lambda w, m=members: cap - np.sum(w[m])}
                    )
        x0 = np.full(n, 1.0 / n)
        res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds,
                       constraints=cons_list,
                       options={"maxiter": 1000, "ftol": 1e-12})
        status = "optimal" if res.success else f"scipy: {res.message}"
        weights = np.clip(res.x, 0, None)
        weights /= weights.sum()
    else:
        raise ValueError(f"unknown solver {solver!r}")

    dt = time.perf_counter() - t0
    vol = float(np.sqrt(weights @ sigma @ weights))
    ret = float(mu @ weights)
    sharpe = (ret - risk_free) / vol if vol > 0 else float("nan")
    return SolveResult(weights, ret, vol, sharpe, solver, dt, str(status))


def max_feasible_return(
    mu: np.ndarray, tickers: list[str], constraints: Constraints | None = None
) -> float:
    """Maximum expected return attainable under the constraints (an LP)."""
    constraints = constraints or Constraints()
    mu = np.asarray(mu, dtype=float)
    w = cp.Variable(len(tickers))
    cons = _cvxpy_constraints(w, tickers, constraints, normalized=True)
    prob = cp.Problem(cp.Maximize(mu @ w), cons)
    prob.solve()
    return float(prob.value)


def efficient_frontier(
    mu: np.ndarray,
    sigma: np.ndarray,
    tickers: list[str],
    n_points: int = 25,
    constraints: Constraints | None = None,
    solver: str = "cvxpy",
    risk_free: float = 0.0,
) -> pd.DataFrame:
    """Trace the efficient frontier: min variance for a grid of target returns."""
    constraints = constraints or Constraints()
    mu = np.asarray(mu, dtype=float)
    gmin = min_variance(mu, sigma, tickers, constraints=constraints,
                        solver=solver, risk_free=risk_free)
    r_max = max_feasible_return(mu, tickers, constraints)
    targets = np.linspace(gmin.expected_return, r_max, n_points)
    rows = []
    for r in targets:
        res = min_variance(mu, sigma, tickers, target_return=float(r),
                           constraints=constraints, solver=solver,
                           risk_free=risk_free)
        rows.append({
            "target_return": float(r),
            "expected_return": res.expected_return,
            "volatility": res.volatility,
            "sharpe": res.sharpe,
            "status": res.status,
        })
    return pd.DataFrame(rows)
