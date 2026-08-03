"""
optimizer.py

Robust portfolio optimization using CVXPY.
Formulates and solves the robust counterpart exactly matching the paper's framework:
    max_{w, t}  w^T mu - kappa_solver * t - 0.5 * lambda * w^T Sigma w
    s.t.        || Omega^{1/2} w ||_2 <= t
                w >= 0 (if long_only)
                sum(w) == 1 (if full_investment)
"""

import cvxpy as cp
import numpy as np

from config import (
    LAMBDA,
    LONG_ONLY,
    FULL_INVESTMENT,
    SOLVER,
)


class RobustOptimizer:
    """
    Class-based robust portfolio optimizer to compile the problem once
    and run fast sweeps over parameter grids (like mu and kappa).
    """

    def __init__(
        self,
        Sigma,
        Omega,
        lambda_val=None,
        long_only=None,
        full_investment=None,
        solver_name=None,
    ):
        """
        Initialize and compile the CVXPY problem structure.
        """
        # Fallback to config parameters if not explicitly provided
        self.lambda_val = LAMBDA if lambda_val is None else lambda_val
        self.long_only = LONG_ONLY if long_only is None else long_only
        self.full_investment = FULL_INVESTMENT if full_investment is None else full_investment
        self.solver_name = SOLVER if solver_name is None else solver_name

        n_assets = Sigma.shape[0]

        # 1. Define CP Parameters (allows updating values without re-compiling)
        self.mu_param = cp.Parameter(n_assets)
        self.kappa_param = cp.Parameter(nonneg=True)

        # 2. Define Decision Variables
        self.w = cp.Variable(n_assets)
        self.t = cp.Variable(nonneg=True)  # Auxiliary variable for SOC constraint

        # 3. Formulate Objective with psd_wrap to handle numerical noise
        expected_return = self.mu_param @ self.w
        robust_penalty = self.kappa_param * self.t
        risk_penalty = 0.5 * self.lambda_val * cp.quad_form(self.w, cp.psd_wrap(Sigma))
        objective = cp.Maximize(expected_return - robust_penalty - risk_penalty)

        # 4. Formulate Constraints
        omega_diag = np.diag(Omega)
        omega_sqrt = np.sqrt(np.maximum(omega_diag, 0.0))
        soc_constraint = cp.norm(cp.multiply(omega_sqrt, self.w), 2) <= self.t
        
        constraints = [soc_constraint]

        if self.long_only:
            constraints.append(self.w >= 0)

        if self.full_investment:
            constraints.append(cp.sum(self.w) == 1)

        # 5. Create Problem Instance
        self.problem = cp.Problem(objective, constraints)

    def solve(self, mu, kappa_solver):
        """
        Update the parameter values and solve the pre-compiled problem.
        """
        self.mu_param.value = mu
        self.kappa_param.value = kappa_solver

        # Solve using the specified solver, let failures raise exception directly
        self.problem.solve(solver=getattr(cp, self.solver_name, self.solver_name))

        if self.w.value is None:
            raise ValueError(f"Optimization failed. Status: {self.problem.status}")

        return self.w.value, self.problem.status


def solve_robust_portfolio(
    mu,
    Sigma,
    Omega,
    kappa_solver,
    lambda_val=None,
    long_only=None,
    full_investment=None,
    solver_name=None,
):
    """
    Convenience functional wrapper for one-shot robust portfolio solving.
    """
    optimizer = RobustOptimizer(
        Sigma=Sigma,
        Omega=Omega,
        lambda_val=lambda_val,
        long_only=long_only,
        full_investment=full_investment,
        solver_name=solver_name,
    )
    return optimizer.solve(mu, kappa_solver)
