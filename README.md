# Convex Portfolio Optimization

Mean–variance portfolio optimization with real-world constraints, built as a
convex quadratic program. Two solver backends (**cvxpy** and **scipy**) solve
the same problems so their agreement — and speed — can be compared honestly.
Ledoit–Wolf shrinkage covariance, KKT-verified solutions, and a strictly
causal out-of-sample backtest against equal-weight and SPY benchmarks.

## Problem statement

Given a universe of liquid US equities, construct long-only, fully-invested
portfolios that trade off expected return against volatility — subject to the
constraints real mandates impose: maximum single-name weight, sector caps,
and turnover limits. Then test whether the optimization survives contact with
out-of-sample data, with transaction costs accounted for explicitly.

## Methodology

1. **Data** — Daily adjusted closes for 37 large-cap US stocks across 11
   sectors (fixed universe in `src/portopt/data.py`), 2019-01-02 → 2026-09-30
   (1,947 trading days), via Yahoo Finance with a Stooq fallback. Log returns.
2. **Moments** — Sample mean/covariance vs. Ledoit–Wolf shrinkage
   (`sklearn.covariance.LedoitWolf`); estimated shrinkage intensity: **0.018**.
3. **Optimization** — Convex QPs (see `docs/math_notes.md` for the Lagrangian,
   KKT conditions, and the max-Sharpe reformulation):
   - minimum variance (global and for target returns → efficient frontier),
   - maximum Sharpe via the convex Charnes–Cooper-style reformulation,
   - constraints: long-only, fully invested, max 10% per name, 35% sector cap,
     optional turnover limit — implemented in *both* cvxpy and scipy.
4. **Backtest** — Expanding estimation window, monthly rebalancing from 2021
   (69 rebalances), 10 bps one-way transaction costs charged as an explicit
   equity hit, **no look-ahead** (windows end the day before each rebalance).

## Results

All numbers below are produced by `scripts/run_analysis.py` and stored in
`results/metrics.json` — nothing is hand-tuned.

**In-sample (full period, Ledoit–Wolf moments)**

| Portfolio | Ann. return | Ann. vol | Sharpe (rf=0) |
|---|---|---|---|
| Max Sharpe | 25.6% | 18.8% | 1.36 |
| Min variance | 11.6% | 14.7% | 0.79 |

**Solver agreement** (same problem, both backends)

| Problem | Weight L2 distance | cvxpy time | scipy (SLSQP) time |
|---|---|---|---|
| Min variance | 5.2e-06 | 0.021 s | 0.280 s |
| Max Sharpe | 4.2e-07 | 0.022 s | 0.154 s |

The two solvers agree to ~1e-6; cvxpy is ~7–13× faster here.

**Shrinkage study** (min-var weights fit on 1st half of data, realized
volatility measured on 2nd half): sample covariance → 11.45% realized vol;
Ledoit–Wolf → 11.45%. No meaningful difference — with 1,947 daily
observations for 37 assets the sample covariance is already well-estimated,
so shrinkage intensity collapses to 0.018. Reported as found.

**Out-of-sample backtest** (invested from first rebalance, Jan 2021)

| Strategy | CAGR | Ann. vol | Sharpe | Max drawdown |
|---|---|---|---|---|
| MaxSharpe + Ledoit–Wolf | **18.35%** | 17.35% | 1.06 | −23.8% |
| MinVar + sample cov | 11.46% | **11.70%** | 0.99 | **−13.6%** |
| Equal weight | 16.49% | 14.40% | **1.13** | −18.1% |
| SPY buy & hold | 17.24% | 19.24% | 0.92 | −33.7% |

The optimized max-Sharpe portfolio edged SPY on return and Sharpe with a
much shallower drawdown; the min-variance portfolio delivered the lowest
volatility and drawdown as designed. Notably, plain equal-weighting had the
best Sharpe — a classic, honestly-reported result: optimization helps, but
1/N is a stubbornly strong benchmark.

![Efficient frontier](results/figures/efficient_frontier.png)
![Backtest equity](results/figures/backtest_equity.png)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python scripts/run_analysis.py --download   # fetch data, run everything
pytest -q                                    # 13 tests
```

## Project structure

```
src/portopt/
  data.py           Yahoo/Stooq download, caching, log returns
  moments.py        sample moments, Ledoit-Wolf shrinkage, annualization
  optimize.py       min-variance / max-Sharpe / frontier, cvxpy + scipy
  backtest.py       causal backtest, costs, performance metrics
  visualization.py  frontier, weights, equity, drawdown plots
scripts/run_analysis.py   full pipeline → results/
tests/test_portopt.py     13 tests (KKT stationarity, solver agreement,
                          accounting identity, no-look-ahead, constraints)
docs/math_notes.md        derivations: QP, KKT, Sharpe reformulation,
                          shrinkage intuition
```

## Reproducibility

- Fixed universe and date range; data cached to `data/` with a download script.
- `numpy` seed fixed; optimization is deterministic; dependencies pinned.
- `results/metrics.json` is the complete numeric record of the run above.

## Limitations & future work

- Expected returns are sample means — notoriously noisy; a Black–Litterman
  overlay or factor-model means would be the natural next step.
- Single fixed universe and one history path; no regime conditioning.
- Costs are a flat 10 bps; market impact and borrow costs are not modeled.
- Risk-free rate set to 0 throughout (stated, not hidden); excess-return
  results shift slightly with a realistic T-bill series.
- Future: turnover-constrained backtest leg, Black–Litterman views,
  conditional (regime-aware) covariance, weekly rebalancing study.

## References

- Markowitz (1952), *Portfolio Selection*, Journal of Finance.
- Ledoit & Wolf (2004), *Honey, I Shrunk the Sample Covariance Matrix*.
- Michaud (1989), *The Markowitz Optimization Enigma*.
- Cornuejols & Tütüncü, *Optimization Methods in Finance*, Ch. 5–6.

## Author

Abdullah Ajmal — BS Mathematics, MS Data Science (PAF-IAST).
