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
