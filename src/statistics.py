"""
statistics.py

Statistical estimators used in the robust portfolio optimization model.
"""

import numpy as np


def estimate_mean(R):
    """
    Sample mean vector.

    Parameters
    ----------
    R : ndarray
        T × N return matrix

    Returns
    -------
    mu : ndarray
        N-dimensional sample mean vector
    """

    return np.mean(R, axis=0)


def estimate_covariance(R):
    """
    Sample covariance matrix.

    Parameters
    ----------
    R : ndarray

    Returns
    -------
    Sigma : ndarray
        N × N covariance matrix
    """

    return np.cov(R, rowvar=False)


def estimate_omega(Sigma, T):
    """
    Estimate Omega exactly as defined in the paper.

    Omega = diag(Sigma) / T
    """

    return np.diag(np.diag(Sigma)) / T


def compute_asset_sharpe_ratios(mu, Sigma):
    """
    Compute the Sharpe ratio of each asset.

    SR_i = mu_i / sigma_i
    """

    sigma = np.sqrt(np.diag(Sigma))

    return mu / sigma


def compute_paper_kappa_upper_bound(mu, Sigma):
    """
    Paper's theoretical upper bound.

    kappa_max = ||SR||_2
    """

    sr = compute_asset_sharpe_ratios(mu, Sigma)

    return np.linalg.norm(sr)


def compute_solver_kappa_upper_bound(mu, Sigma, T):
    """
    Solver kappa used inside the optimizer when
    Omega = diag(Sigma)/T.

    Therefore the solver parameter is:
    kappa_solver = sqrt(T) * kappa_paper
    """

    return np.sqrt(T) * compute_paper_kappa_upper_bound(mu, Sigma)


def generate_alpha_grid(n_alpha):
    """
    Uniform alpha grid between 0 and 1.
    """

    return np.linspace(0.0, 1.0, n_alpha)


def generate_kappa_grid(kappa_max_raw, alpha_grid):
    """
    Convert alpha values into kappa values.
    """

    return alpha_grid * kappa_max_raw


def compute_semivariance_omegas(R, T):
    """
    Asset-specific lower and upper semi-variance matrices for the asymmetric model.

    For each asset i:
        sv_down_i = E[r_{i,t}² | r_{i,t} < 0]   (expected squared negative return)
        sv_up_i   = E[r_{i,t}² | r_{i,t} > 0]   (expected squared positive return)

    Omega matrices (matching the Omega = diag(Sigma)/T convention):
        Omega_down = diag(sv_down) / T
        Omega_up   = diag(sv_up)   / T

    Economic interpretation
    -----------------------
    Omega_down captures estimation risk specifically on the *loss* side.
    An asset with low sv_down (mostly upside volatility) is penalised less
    by the downside-protection penalty, so the asymmetric model preserves
    its allocation even as kappa rises.
    """
    n_assets = R.shape[1]
    sv_down = np.zeros(n_assets)
    sv_up   = np.zeros(n_assets)

    for i in range(n_assets):
        ri = R[:, i]
        neg_r = ri[ri < 0]
        pos_r = ri[ri > 0]
        sv_down[i] = np.mean(neg_r ** 2) if len(neg_r) > 0 else 1e-8
        sv_up[i]   = np.mean(pos_r ** 2) if len(pos_r) > 0 else 1e-8

    Omega_down = np.diag(sv_down) / T
    Omega_up   = np.diag(sv_up)   / T

    return Omega_down, Omega_up


def compute_asymmetric_kappa_bound(mu, Omega_down):
    """
    Upper bound for kappa_paper in the asymmetric model.

    Analogous to the symmetric bound ||SR||₂, but using the downside
    uncertainty matrix:

        kappa_max_asymmetric = ||Omega_down^{-1/2} mu||₂ / normalisation

    Simplified (same convention as symmetric): uses lower-semi-variance-based
    'Sharpe ratios':
        SR_down_i = mu_i / sqrt(diag(Omega_down)_i * T)

    The l2-norm of these gives the solver bound after multiplying by sqrt(T).
    """
    diag_down = np.diag(Omega_down)                    # = sv_down / T
    sr_down   = mu / np.sqrt(np.maximum(diag_down, 1e-12))
    return float(np.linalg.norm(sr_down))

