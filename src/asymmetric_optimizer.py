"""
asymmetric_optimizer.py

Asymmetric robust portfolio optimizer (signed/split-radius ellipsoidal uncertainty).

Formulation (Goldfarb–Iyengar generalization):

    max_{w, w+, w-, t_down, t_up}
        wᵀμ̄  -  κ_down · t_down  -  κ_up · t_up  -  λ/2 · wᵀΣw

    s.t.
        ‖ Ω_down^{1/2} w+ ‖₂ ≤ t_down     (downside-uncertainty SOC)
        ‖ Ω_up^{1/2}   w- ‖₂ ≤ t_up       (upside-uncertainty SOC)
        w = w+ - w-
        w+, w- ≥ 0
        (sum(w) == 1  if full_investment)

Key difference from symmetric model
-------------------------------------
Omega_down uses LOWER semi-variance of each asset's historical returns, so an asset
whose returns are predominantly upside-skewed has a small Omega_down entry → its
position is penalised less as κ_down rises → the asymmetric model preserves
allocation to high-upside assets while trimming high-downside-risk ones.

For long-only portfolios, w- = 0 (enforced), so only the t_down cone is active
and κ_up is irrelevant. The distinction vs the symmetric model is entirely in
Omega_down vs Omega = diag(Sigma)/T.
"""

import cvxpy as cp
import numpy as np

from config import LAMBDA, LONG_ONLY, FULL_INVESTMENT, SOLVER


class AsymmetricRobustOptimizer:
    """
    Compile-once, solve-many asymmetric robust portfolio optimizer.

    Parameters
    ----------
    Sigma       : ndarray (N×N) — covariance matrix (constant, compiled in)
    Omega_down  : ndarray (N×N) — diagonal lower-semivariance uncertainty matrix
    Omega_up    : ndarray (N×N) — diagonal upper-semivariance uncertainty matrix
    lambda_val  : float         — risk-aversion parameter (default: config.LAMBDA)
    long_only   : bool          — enforce w ≥ 0                (default: config.LONG_ONLY)
    full_investment : bool      — enforce sum(w) = 1           (default: config.FULL_INVESTMENT)
    solver_name : str           — CVXPY solver name            (default: config.SOLVER)
    """

    def __init__(
        self,
        Sigma,
        Omega_down,
        Omega_up,
        lambda_val=None,
        long_only=None,
        full_investment=None,
        solver_name=None,
    ):
        self.lambda_val      = LAMBDA           if lambda_val      is None else lambda_val
        self.long_only       = LONG_ONLY        if long_only       is None else long_only
        self.full_investment = FULL_INVESTMENT  if full_investment is None else full_investment
        self.solver_name     = SOLVER           if solver_name     is None else solver_name

        n = Sigma.shape[0]

        # ── CVXPY Parameters (updated each solve without recompilation) ────────
        self.mu_param    = cp.Parameter(n)
        self.kappa_down  = cp.Parameter(nonneg=True)
        self.kappa_up    = cp.Parameter(nonneg=True)

        # ── Decision Variables ─────────────────────────────────────────────────
        self.w_plus  = cp.Variable(n, nonneg=True)   # long book (w+ ≥ 0)
        self.w_minus = cp.Variable(n, nonneg=True)   # short book (w- ≥ 0)
        self.t_down  = cp.Variable(nonneg=True)
        self.t_up    = cp.Variable(nonneg=True)

        # Net portfolio weights: w = w+ - w-
        w_net = self.w_plus - self.w_minus

        # ── Omega square-roots ────────────────────────────────────────────────
        omega_down_sqrt = np.sqrt(np.maximum(np.diag(Omega_down), 0.0))
        omega_up_sqrt   = np.sqrt(np.maximum(np.diag(Omega_up),   0.0))

        # ── Objective ─────────────────────────────────────────────────────────
        expected_return  = self.mu_param @ w_net
        robust_penalty   = self.kappa_down * self.t_down + self.kappa_up * self.t_up
        risk_penalty     = 0.5 * self.lambda_val * cp.quad_form(w_net, cp.psd_wrap(Sigma))
        objective = cp.Maximize(expected_return - robust_penalty - risk_penalty)

        # ── Constraints ───────────────────────────────────────────────────────
        constraints = [
            # Downside SOC: protects long book against μ being lower than μ̄
            cp.norm(cp.multiply(omega_down_sqrt, self.w_plus),  2) <= self.t_down,
            # Upside SOC: protects short book against μ being higher than μ̄
            cp.norm(cp.multiply(omega_up_sqrt,   self.w_minus), 2) <= self.t_up,
        ]

        if self.long_only:
            # No short selling: w- = 0 (t_up cone becomes trivially satisfied)
            constraints.append(self.w_minus == 0)

        if self.full_investment:
            constraints.append(cp.sum(w_net) == 1)

        # ── Compile ───────────────────────────────────────────────────────────
        self.problem = cp.Problem(objective, constraints)

    def solve(self, mu, kappa_down, kappa_up=0.0):
        """
        Update parameters and solve the pre-compiled problem.

        Parameters
        ----------
        mu          : ndarray — expected returns vector
        kappa_down  : float   — solver-scale downside robustness parameter
        kappa_up    : float   — solver-scale upside robustness parameter (0 for long-only)

        Returns
        -------
        w_opt  : ndarray — optimal net weights
        status : str     — solver status string
        """
        self.mu_param.value   = mu
        self.kappa_down.value = float(kappa_down)
        self.kappa_up.value   = float(kappa_up)

        self.problem.solve(solver=getattr(cp, self.solver_name, self.solver_name))

        if self.w_plus.value is None:
            raise ValueError(
                f"Asymmetric optimization failed. Status: {self.problem.status}"
            )

        w_opt = self.w_plus.value - self.w_minus.value
        return w_opt, self.problem.status
