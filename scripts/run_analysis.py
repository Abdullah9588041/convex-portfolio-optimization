"""End-to-end analysis: frontier, solver comparison, shrinkage study, backtest.

Usage:
    python scripts/run_analysis.py [--download]

Without --download the script uses data/prices.csv if present.
All figures go to results/figures/, metrics to results/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from portopt import BacktestConfig
from portopt.backtest import (
    buy_and_hold,
    equal_weight_strategy,
    max_sharpe_lw_strategy,
    min_var_sample_strategy,
    run_backtest,
)
from portopt.data import (
    BENCHMARK_TICKER,
    DATA_END,
    DATA_START,
    SECTORS,
    TICKERS,
    compute_log_returns,
    download_prices,
    load_cached_prices,
)
from portopt.moments import TRADING_DAYS, ledoit_wolf_cov, portfolio_stats, sample_moments
from portopt.optimize import (
    Constraints,
    efficient_frontier,
    max_sharpe,
    min_variance,
)
from portopt.visualization import (
    plot_drawdown,
    plot_equity_curves,
    plot_frontier,
    plot_weights,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
RISK_FREE = 0.0  # Sharpe ratios use rf = 0; stated explicitly in README

np.random.seed(42)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="re-download price data instead of using the cache")
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "prices.csv"

    if args.download or not cache.exists():
        prices, manifest = download_prices(TICKERS, DATA_START, DATA_END, cache)
        spy, _ = download_prices([BENCHMARK_TICKER], DATA_START, DATA_END,
                                 DATA_DIR / "spy.csv")
        print(f"data source: {manifest.source}, {manifest.n_rows} rows, "
              f"{len(manifest.tickers)} tickers")
    else:
        prices = load_cached_prices(cache)
        spy = load_cached_prices(DATA_DIR / "spy.csv")
        print(f"loaded cached data: {prices.shape}")

    tickers = list(prices.columns)
    sector_of = {t: SECTORS[t] for t in tickers}
    returns = compute_log_returns(prices)

    # ---- 1. Moments: sample vs Ledoit-Wolf -------------------------------
    mu_d_s, sigma_d_s = sample_moments(returns)
    mu_d_lw, sigma_d_lw, shrinkage = ledoit_wolf_cov(returns)
    mu_a, sigma_a = mu_d_lw * TRADING_DAYS, sigma_d_lw * TRADING_DAYS
    print(f"Ledoit-Wolf shrinkage intensity: {shrinkage:.4f}")

    constraints = Constraints(
        long_only=True, fully_invested=True, max_weight=0.10,
        sector_caps={s: 0.35 for s in set(sector_of.values())},
        sector_of=sector_of,
    )

    # ---- 2. Efficient frontier (in-sample illustration) -------------------
    t0 = time.perf_counter()
    frontier = efficient_frontier(mu_a.values, sigma_a.values, tickers,
                                  n_points=25, constraints=constraints,
                                  risk_free=RISK_FREE)
    frontier_time = time.perf_counter() - t0

    ms = max_sharpe(mu_a.values, sigma_a.values, tickers, risk_free=RISK_FREE,
                    constraints=constraints, solver="cvxpy")
    mv = min_variance(mu_a.values, sigma_a.values, tickers,
                      constraints=constraints, solver="cvxpy", risk_free=RISK_FREE)
    print(f"max-Sharpe (in-sample): return={ms.expected_return:.4f} "
          f"vol={ms.volatility:.4f} sharpe={ms.sharpe:.4f}")
    print(f"min-variance (in-sample): return={mv.expected_return:.4f} "
          f"vol={mv.volatility:.4f} sharpe={mv.sharpe:.4f}")

    plot_frontier(
        frontier,
        {"Max Sharpe": (ms.volatility, ms.expected_return),
         "Min variance": (mv.volatility, mv.expected_return)},
        FIGURES / "efficient_frontier.png",
    )
    plot_weights(ms.weights, tickers, "Max-Sharpe portfolio weights (Ledoit-Wolf)",
                 FIGURES / "weights_max_sharpe.png")
    plot_weights(mv.weights, tickers, "Min-variance portfolio weights (Ledoit-Wolf)",
                 FIGURES / "weights_min_variance.png")

    # ---- 3. Solver comparison: cvxpy vs scipy (min variance) -------------
    mv_scipy = min_variance(mu_a.values, sigma_a.values, tickers,
                            constraints=constraints, solver="scipy",
                            risk_free=RISK_FREE)
    mv_cvx = min_variance(mu_a.values, sigma_a.values, tickers,
                          constraints=constraints, solver="cvxpy",
                          risk_free=RISK_FREE)
    l2 = float(np.linalg.norm(mv_cvx.weights - mv_scipy.weights))
    linf = float(np.abs(mv_cvx.weights - mv_scipy.weights).max())
    print(f"solver agreement (min-var): L2={l2:.2e} Linf={linf:.2e} | "
          f"cvxpy {mv_cvx.solve_time_s:.3f}s vs scipy {mv_scipy.solve_time_s:.3f}s")

    ms_scipy = max_sharpe(mu_a.values, sigma_a.values, tickers, risk_free=RISK_FREE,
                          constraints=constraints, solver="scipy")
    l2_ms = float(np.linalg.norm(ms.weights - ms_scipy.weights))
    print(f"solver agreement (max-sharpe): L2={l2_ms:.2e} | "
          f"cvxpy {ms.solve_time_s:.3f}s vs scipy {ms_scipy.solve_time_s:.3f}s")

    # ---- 4. Shrinkage vs sample covariance: honest split ------------------
    mid = len(returns) // 2
    r_train, r_test = returns.iloc[:mid], returns.iloc[mid:]
    _, s_train = sample_moments(r_train)
    _, s_lw_train, _ = ledoit_wolf_cov(r_train)
    mu_tr = r_train.mean().values * TRADING_DAYS
    w_s = min_variance(mu_tr, (s_train * TRADING_DAYS).values, tickers,
                       constraints=constraints).weights
    w_lw = min_variance(mu_tr, (s_lw_train * TRADING_DAYS).values, tickers,
                        constraints=constraints).weights
    test_cov = r_test.cov().values * TRADING_DAYS
    realized_vol_sample = float(np.sqrt(w_s @ test_cov @ w_s))
    realized_vol_lw = float(np.sqrt(w_lw @ test_cov @ w_lw))
    print(f"realized vol (2nd half) — sample cov: {realized_vol_sample:.4f}, "
          f"Ledoit-Wolf: {realized_vol_lw:.4f}")

    # ---- 5. Out-of-sample backtest ----------------------------------------
    cfg = BacktestConfig(min_lookback_days=504, transaction_cost_bps=10.0,
                         risk_free=RISK_FREE)
    strategies = {
        "MaxSharpe_LW": max_sharpe_lw_strategy,
        "MinVar_Sample": min_var_sample_strategy,
        "EqualWeight": equal_weight_strategy,
    }
    results = {}
    for name, strat in strategies.items():
        res = run_backtest(prices, name, strat, cfg, sector_of)
        results[name] = res
        m = res.metrics
        print(f"{name}: CAGR={m['cagr']:.4f} vol={m['volatility']:.4f} "
              f"Sharpe={m['sharpe']:.4f} maxDD={m['max_drawdown']:.4f} "
              f"cost_drag={res.total_cost_drag:.4f}")
    spy_res = buy_and_hold(spy[BENCHMARK_TICKER].loc[prices.index[0]:], "SPY_BuyHold",
                           risk_free=RISK_FREE)
    results["SPY_BuyHold"] = spy_res
    m = spy_res.metrics
    print(f"SPY_BuyHold: CAGR={m['cagr']:.4f} vol={m['volatility']:.4f} "
          f"Sharpe={m['sharpe']:.4f} maxDD={m['max_drawdown']:.4f}")

    plot_equity_curves({k: v.equity for k, v in results.items()},
                       FIGURES / "backtest_equity.png")
    plot_drawdown(results["MaxSharpe_LW"].equity, "MaxSharpe_LW",
                  FIGURES / "drawdown_maxsharpe.png")

    # ---- 6. Metrics JSON ----------------------------------------------------
    metrics = {
        "data": {
            "tickers": tickers,
            "n_tickers": len(tickers),
            "start": str(prices.index[0].date()),
            "end": str(prices.index[-1].date()),
            "n_days": len(prices),
        },
        "ledoit_wolf_shrinkage": shrinkage,
        "in_sample": {
            "max_sharpe": portfolio_stats(ms.weights, mu_a, sigma_a, RISK_FREE),
            "min_variance": portfolio_stats(mv.weights, mu_a, sigma_a, RISK_FREE),
            "frontier_build_seconds": frontier_time,
        },
        "solver_comparison": {
            "min_var_weight_l2": l2,
            "min_var_weight_linf": linf,
            "min_var_cvxpy_s": mv_cvx.solve_time_s,
            "min_var_scipy_s": mv_scipy.solve_time_s,
            "max_sharpe_weight_l2": l2_ms,
            "max_sharpe_cvxpy_s": ms.solve_time_s,
            "max_sharpe_scipy_s": ms_scipy.solve_time_s,
        },
        "shrinkage_study": {
            "realized_vol_sample_cov": realized_vol_sample,
            "realized_vol_ledoit_wolf": realized_vol_lw,
            "note": "min-var weights fit on 1st half, realized vol on 2nd half",
        },
        "backtest": {
            "config": {
                "min_lookback_days": cfg.min_lookback_days,
                "rebalance": cfg.rebalance_rule,
                "transaction_cost_bps": cfg.transaction_cost_bps,
                "risk_free": cfg.risk_free,
                "max_weight": cfg.max_weight,
                "sector_cap": cfg.sector_cap,
                "protocol": "expanding window, no look-ahead; costs as one-time "
                             "equity hit on rebalance dates",
            },
            "strategies": {
                name: {**res.metrics,
                       "total_cost_drag": res.total_cost_drag,
                       "n_rebalances": len(res.turnover)}
                for name, res in results.items()
            },
        },
    }
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("wrote results/metrics.json and figures/")


if __name__ == "__main__":
    main()
