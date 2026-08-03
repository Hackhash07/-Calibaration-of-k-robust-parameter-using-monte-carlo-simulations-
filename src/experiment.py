"""
experiment.py

Runs the robust portfolio calibration experiment by sweeping kappa values
and solving the robust optimization problem.
"""

from data_loader import load_returns
from statistics import (
    estimate_mean,
    estimate_covariance,
    estimate_omega,
    compute_paper_kappa_upper_bound,
    compute_solver_kappa_upper_bound,
    generate_alpha_grid,
    generate_kappa_grid,
)
from optimizer import RobustOptimizer
from config import N_ALPHA


def run_calibration_experiment():
    """
    Orchestrates the calibration experiment:
    1. Loads returns.
    2. Estimates mu, Sigma, Omega.
    3. Computes kappa bounds.
    4. Sweeps kappa values, solving the pre-compiled optimizer.
    5. Computes realized returns and weights for each portfolio.

    Returns
    -------
    results : dict
        A dictionary mapping alpha values to their portfolio weights, realized returns, and status.
    meta : dict
        A dictionary containing historical data, statistical estimators, grids, and boundaries.
    """
    # 1. Load returns
    R, asset_names, dates = load_returns()
    T, N = R.shape

    # 2. Estimate statistical inputs
    mu = estimate_mean(R)
    Sigma = estimate_covariance(R)
    Omega = estimate_omega(Sigma, T)

    # 3. Compute kappa bounds
    kappa_max_paper = compute_paper_kappa_upper_bound(mu, Sigma)
    kappa_max_solver = compute_solver_kappa_upper_bound(mu, Sigma, T)

    # 4. Generate grids
    alpha_grid = generate_alpha_grid(N_ALPHA)
    kappa_paper_grid = generate_kappa_grid(kappa_max_paper, alpha_grid)
    kappa_solver_grid = generate_kappa_grid(kappa_max_solver, alpha_grid)

    # Instantiate the pre-compiled optimizer
    optimizer = RobustOptimizer(Sigma=Sigma, Omega=Omega)

    results = {}

    # 5. Sweep kappa
    for i, alpha in enumerate(alpha_grid):
        k_paper = kappa_paper_grid[i]
        k_solver = kappa_solver_grid[i]

        # Solve optimizer
        w_opt, status = optimizer.solve(mu, k_solver)

        # Compute realized returns (daily portfolio returns over the period)
        realized_returns = R @ w_opt

        results[alpha] = {
            "alpha": alpha,
            "kappa_paper": k_paper,
            "kappa_solver": k_solver,
            "weights": w_opt,
            "realized_returns": realized_returns,
            "status": status,
        }

    meta = {
        "asset_names": asset_names,
        "dates": dates,
        "R": R,
        "mu": mu,
        "Sigma": Sigma,
        "Omega": Omega,
        "kappa_max_paper": kappa_max_paper,
        "kappa_max_solver": kappa_max_solver,
        "alpha_grid": alpha_grid,
        "kappa_paper_grid": kappa_paper_grid,
        "kappa_solver_grid": kappa_solver_grid,
    }

    return results, meta
