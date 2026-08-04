"""
asymmetric_experiment.py

Runs the asymmetric robust portfolio calibration sweep using the
AsymmetricRobustOptimizer with asset-specific semi-variance uncertainty matrices.

Key differences vs. symmetric experiment
-----------------------------------------
1. Omega_down (lower semi-variance) replaces Omega = diag(Sigma)/T for the
   downside-uncertainty SOCP cone.
2. kappa_up = 0 throughout (long-only book; no short positions to protect).
3. kappa_max is calibrated against Omega_down → same alpha grid spans the
   same [0, 1] range, but the actual solver kappa values differ.
"""

from data_loader import load_returns
from statistics import (
    estimate_mean,
    estimate_covariance,
    compute_paper_kappa_upper_bound,
    compute_solver_kappa_upper_bound,
    compute_semivariance_omegas,
    generate_alpha_grid,
    generate_kappa_grid,
)
from asymmetric_optimizer import AsymmetricRobustOptimizer
from config import N_ALPHA


def run_asymmetric_experiment():
    """
    Orchestrates the asymmetric calibration sweep.

    Kappa calibration (IMPORTANT)
    ------------------------------
    We use the *symmetric* kappa_max so both experiments share the same
    alpha-to-kappa scale. The asymmetric effect comes entirely from
    using Omega_down (lower semi-variance) instead of Omega = diag(Sigma)/T
    in the SOCP constraint.

    This makes the comparison meaningful:
        same alpha = same global robustness level
        different Omega = different per-asset penalty shape
        → asymmetric model retains more weight in low-sv_down (upside-skewed)
          assets at the same kappa, while shrinking high-sv_down assets harder.

    Returns
    -------
    results : dict  — maps alpha → {weights, realized_returns, kappa_paper, ...}
    meta    : dict  — all estimators, grids, and boundary values
    """
    # 1. Load data
    R, asset_names, dates = load_returns()
    T, N = R.shape

    # 2. Estimate inputs
    mu    = estimate_mean(R)
    Sigma = estimate_covariance(R)
    Omega_down, Omega_up = compute_semivariance_omegas(R, T)

    # 3. Kappa bounds — USE SYMMETRIC CALIBRATION for comparability
    kappa_max_paper  = compute_paper_kappa_upper_bound(mu, Sigma)
    kappa_max_solver = compute_solver_kappa_upper_bound(mu, Sigma, T)

    # 4. Build grids (same alpha structure as symmetric)
    alpha_grid        = generate_alpha_grid(N_ALPHA)
    kappa_paper_grid  = generate_kappa_grid(kappa_max_paper,  alpha_grid)
    kappa_solver_grid = generate_kappa_grid(kappa_max_solver, alpha_grid)

    # 5. Compile optimizer once (Omega_down in SOCP, Omega_up = 0 for long-only)
    optimizer = AsymmetricRobustOptimizer(
        Sigma=Sigma,
        Omega_down=Omega_down,
        Omega_up=Omega_up,
    )

    results = {}

    for i, alpha in enumerate(alpha_grid):
        k_paper  = kappa_paper_grid[i]
        k_solver = kappa_solver_grid[i]

        w_opt, status = optimizer.solve(
            mu=mu,
            kappa_down=k_solver,
            kappa_up=0.0,   # no upside penalty for long-only
        )

        results[alpha] = {
            "alpha":            alpha,
            "kappa_paper":      k_paper,
            "kappa_solver":     k_solver,
            "weights":          w_opt,
            "realized_returns": R @ w_opt,
            "status":           status,
        }

    meta = {
        "asset_names":      asset_names,
        "dates":            dates,
        "R":                R,
        "mu":               mu,
        "Sigma":            Sigma,
        "Omega_down":       Omega_down,
        "Omega_up":         Omega_up,
        "kappa_max_paper":  kappa_max_paper,
        "kappa_max_solver": kappa_max_solver,
        "alpha_grid":       alpha_grid,
        "kappa_paper_grid": kappa_paper_grid,
    }

    return results, meta
