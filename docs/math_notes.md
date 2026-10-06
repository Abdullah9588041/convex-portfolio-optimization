# Mathematical Notes

## 1. The mean–variance problem

Given $n$ assets with expected (excess) return vector $\mu \in \mathbb{R}^n$
and covariance matrix $\Sigma \in \mathbb{S}^n_{++}$, a fully-invested
long-only portfolio $w$ solves, for a target return $\bar r$,

$$\min_w \; w^\top \Sigma w \quad \text{s.t.} \quad \mu^\top w \ge \bar r,
\quad \mathbf{1}^\top w = 1, \quad w \ge 0.$$

**Why it is convex.** The objective $w^\top \Sigma w$ is a convex quadratic
($\Sigma \succeq 0$), and every constraint is linear. It is therefore a convex
quadratic program (QP): any local optimum is global, and it is solvable in
polynomial time by interior-point methods.

## 2. KKT conditions (min-variance, long-only)

With multipliers $\lambda \ge 0$ (return constraint), $\gamma$ (budget) and
$\nu \ge 0$ (non-negativity), the Lagrangian is

$$\mathcal{L} = w^\top \Sigma w - \lambda(\mu^\top w - \bar r)
- \gamma(\mathbf{1}^\top w - 1) - \nu^\top w.$$

Stationarity, primal/dual feasibility and complementary slackness give

$$2\Sigma w - \lambda \mu - \gamma \mathbf{1} - \nu = 0, \qquad
\nu_i w_i = 0 \;\; \forall i, \qquad \nu \ge 0, \; w \ge 0.$$

The test `test_kkt_stationarity_min_variance` verifies this numerically:
with $\lambda$ recovered from $w^\top(2\Sigma w - \gamma\mathbf{1}) = 0$ and
$\nu_i = \max(0, \gamma - 2(\Sigma w)_i)$, the stationarity residual must
vanish and $\nu_i w_i \approx 0$.

## 3. Max-Sharpe reformulation

Maximizing $(\mu - r_f)^\top w / \sqrt{w^\top \Sigma w}$ is non-convex as
written (fractional program), but the Charnes–Cooper-style change of variable
$y = w / ((\mu - r_f)^\top w)$ turns it into the convex QP

$$\min_y \; y^\top \Sigma y \quad \text{s.t.} \quad (\mu - r_f)^\top y = 1,
\quad y \in \mathcal{C},$$

with $w^\star = y^\star / \mathbf{1}^\top y^\star$. A linear constraint on the
*normalized* weights, e.g. $x_i \le c$, becomes $y_i \le c\,\mathbf{1}^\top y$ —
still linear in $y$. Box, sector-cap and turnover constraints are all handled
this way in `optimize.max_sharpe`.

## 4. Ledoit–Wolf shrinkage (intuition)

The sample covariance $S$ is unbiased but noisy: with $n$ assets and $T$
observations it has $O(n^2/T)$ estimation error, and the optimizer
*maximizes* that error ("error maximization", Michaud 1989). Ledoit–Wolf
replaces $S$ with the convex combination

$$\hat\Sigma = (1 - \delta) S + \delta \, \hat\mu I,$$

where $\delta \in [0,1]$ is chosen to minimize expected squared Frobenius
error. Shrinkage pulls extreme eigenvalues toward their grand mean, which
dampens the optimizer's tendency to bet on spurious correlations. We use the
reference implementation `sklearn.covariance.LedoitWolf` rather than a
hand-rolled version.

## 5. Sharpe ratio and backtest accounting

Annualized Sharpe: $S = (\bar r_a - r_f) / \sigma_a$ with $r_f = 0$ throughout
this project (stated explicitly). The backtest charges
$\tfrac12 \|w_{\text{new}} - w_{\text{old}}\|_1 \times \text{bps}$ as a
one-time equity hit on each rebalance date — a conservative, fully disclosed
friction. Estimation windows are strictly causal (expanding, ending the day
*before* each rebalance), so there is no look-ahead bias.

## References

* Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*.
* Ledoit, O. & Wolf, M. (2004). Honey, I Shrunk the Sample Covariance Matrix.
  *Journal of Portfolio Management*.
* Michaud, R. (1989). The Markowitz Optimization Enigma. *Financial Analysts Journal*.
* Cornuejols, G. & Tütüncü, R. *Optimization Methods in Finance* (CUP) — Ch. 5–6.
