"""
plotting.py

Generates and saves performance plots, weight allocations, and return distribution
visualizations for the robust portfolio calibration sweeps.

Audit fixes applied (2026-08-03):
  1. plot_weight_sweep:        ylim now dynamic (gross exposure can exceed 1.0)
  2. plot_weight_sweep:        weights clipped to >=0 before stackplot
  3. plot_return_distributions: zero-variance portfolio shown as axvline (not KDE)
  4. plot_portfolio_metrics:   all 6 panels, xlabel only on bottom row
  5. plot_cumulative_returns_sweep: log y-scale so tiny wealth lines are visible
  6. plot_individual_plots:    fix set_xticks before set_xticklabels (warning)
  7. plot_individual_plots:    guard for zero-weight (alpha=1) bar chart
  8. plot_monte_carlo_distributions: dynamic grid indices, guard last subplot KDE
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from config import FIGURE_DIR


# ---------------------------------------------------------------------------
# 1. Weight sweep stacked area chart
# ---------------------------------------------------------------------------

def plot_weight_sweep(results, alpha_grid, asset_names, save_path=None):
    """
    Stacked area plot of portfolio weights across the alpha sweep.

    Fixes
    -----
    - ylim is computed from the actual maximum gross exposure (not hard-coded to 1)
    - weights are clipped to >= 0 before stacking to avoid rendering artefacts
      from tiny negative solver residuals
    """
    n_assets = len(asset_names)
    n_alphas = len(alpha_grid)

    weights_matrix = np.zeros((n_assets, n_alphas))
    for idx, alpha in enumerate(alpha_grid):
        w = results[alpha]["weights"]
        weights_matrix[:, idx] = np.maximum(w, 0.0)   # clip tiny negatives

    # Maximum gross exposure across the sweep
    max_exposure = weights_matrix.sum(axis=0).max()
    ylim_top = max(1.05, max_exposure * 1.05)

    plt.figure(figsize=(10, 6))
    plt.stackplot(alpha_grid, weights_matrix, labels=asset_names)

    plt.title("Portfolio Weights Dynamics across Robustness Sweep",
              fontsize=14, fontweight="bold")
    plt.xlabel(r"Robustness Level $\alpha$ (0 = MVO, 1 = Max Robustness)",
               fontsize=12)
    plt.ylabel("Portfolio Weight Allocation", fontsize=12)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, ylim_top)

    if n_assets <= 10:
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
    else:
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0),
                   ncol=2, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 2. Realized return distribution comparison
# ---------------------------------------------------------------------------

def plot_return_distributions(results, selected_alphas, save_path=None):
    """
    KDE + histogram overlay for selected robustness levels.

    Fixes
    -----
    - Zero-variance return series (alpha=1.0) rendered as axvline, not KDE
    - y-axis limit driven by the highest continuous-series KDE peak only
    """
    plt.figure(figsize=(10, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    max_density = 1.0  # floor so y-axis is never collapsed

    for idx, alpha in enumerate(selected_alphas):
        if alpha not in results:
            continue

        returns = results[alpha]["realized_returns"]
        color = colors[idx % len(colors)]

        if np.std(returns) < 1e-4:
            # Zero-weight portfolio: returns are all ~0 → show as vertical line
            plt.axvline(
                x=float(np.mean(returns)),
                color=color,
                linestyle="--",
                label=f"Alpha = {alpha:.2f}  (Zero portfolio — "
                      r"$\kappa = \kappa_{\max}$)",
                linewidth=2.5,
            )
        else:
            kde = gaussian_kde(returns)
            x_grid = np.linspace(returns.min() - 0.01, returns.max() + 0.01, 200)
            kde_vals = kde(x_grid)
            max_density = max(max_density, float(np.max(kde_vals)))

            plt.plot(
                x_grid,
                kde_vals,
                color=color,
                label=(f"Alpha = {alpha:.2f}  "
                       f"(κ = {results[alpha]['kappa_paper']:.3f})"),
                linewidth=2.5,
            )
            plt.hist(returns, bins=30, density=True, alpha=0.15, color=color)

    plt.title("Realized Returns Distribution Comparison",
              fontsize=14, fontweight="bold")
    plt.xlabel("Portfolio Daily Returns", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.ylim(0.0, max_density * 1.15)   # 15 % head-room above highest KDE peak
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 3. Six-panel performance metrics sweep
# ---------------------------------------------------------------------------

def plot_portfolio_metrics(results, alpha_grid, save_path=None):
    """
    6-panel grid: Return, Volatility, Sharpe, HHI, Gross Exposure, Active Assets.

    Fixes
    -----
    - xlabel only placed on bottom row panels
    - Sharpe set to 0 when vol < 1e-8 (zero-portfolio edge case)
    """
    expected_returns = []
    volatilities     = []
    sharpe_ratios    = []
    hhi_values       = []
    gross_exposures  = []
    active_counts    = []

    for alpha in alpha_grid:
        w = results[alpha]["weights"]
        r = results[alpha]["realized_returns"]
        vol = float(np.std(r))

        expected_returns.append(float(np.mean(r)))
        volatilities.append(vol)
        sharpe_ratios.append(float(np.mean(r)) / vol if vol > 1e-8 else 0.0)
        hhi_values.append(float(np.sum(w ** 2)))
        gross_exposures.append(float(np.sum(np.abs(w))))
        active_counts.append(int(np.sum(w > 1e-4)))

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    xlabel = r"Robustness Level $\alpha$"

    specs = [
        (axs[0, 0], expected_returns, "darkblue",   "o", "Expected Daily Return",        "Return",                           False),
        (axs[0, 1], volatilities,     "darkred",    "s", "Daily Volatility (Risk)",       "Standard Deviation",               False),
        (axs[1, 0], sharpe_ratios,    "darkgreen",  "^", "Daily Sharpe Ratio",            "Sharpe Ratio",                     False),
        (axs[1, 1], hhi_values,       "purple",     "d", "Portfolio Concentration (HHI)", "HHI",                              False),
        (axs[2, 0], gross_exposures,  "darkorange", "p", r"Gross Exposure ($\Sigma|w_i|$)","Gross Exposure",                  True),
        (axs[2, 1], active_counts,    "teal",       "x", "Number of Active Assets",       "Active Asset Count",               True),
    ]

    for ax, values, color, marker, title, ylabel, show_xlabel in specs:
        ax.plot(alpha_grid, values, color=color, marker=marker, markersize=4)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        if show_xlabel:
            ax.set_xlabel(xlabel, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Portfolio Performance Metrics vs. Robustness Sweep",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 4. Cumulative wealth sweep
# ---------------------------------------------------------------------------

def plot_cumulative_returns_sweep(results, alpha_grid, dates, save_path=None):
    """
    All portfolio wealth-index trajectories on a single chart.

    Fixes
    -----
    - y-axis is log-scale so the tiny near-zero wealth lines of high-kappa
      portfolios are visible alongside the 100x MVO trajectory.
    - Colorbar added to replace the cluttered legend.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    cmap = plt.get_cmap("plasma")
    n_alphas = len(alpha_grid)
    label_indices = {0, n_alphas // 4, n_alphas // 2,
                     3 * n_alphas // 4, n_alphas - 1}

    for idx, alpha in enumerate(alpha_grid):
        r = results[alpha]["realized_returns"]
        wealth_index = np.cumprod(1.0 + r)
        color = cmap(idx / max(1, n_alphas - 1))
        label = f"α = {alpha:.2f}" if idx in label_indices else None
        ax.plot(dates, wealth_index, color=color, label=label,
                linewidth=1.5, alpha=0.85)

    ax.set_yscale("log")   # log scale — key fix
    ax.set_title("Portfolio Cumulative Wealth Dynamics (Robustness Sweep)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Wealth Index — log scale (Start = 1.0)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.5, which="both")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Individual portfolio dashboards
# ---------------------------------------------------------------------------

def plot_individual_plots(results, alpha_grid, dates, asset_names,
                          individual_dir):
    """
    One 2-panel figure per alpha: weight bar chart + cumulative wealth.

    Fixes
    -----
    - set_xticks() called before set_xticklabels() → no more UserWarning
    - Zero-weight portfolios (alpha=1) show a 'No active assets' message
      instead of a blank bar chart
    - Wealth panel uses log scale for consistency
    """
    individual_dir = Path(individual_dir)
    individual_dir.mkdir(parents=True, exist_ok=True)

    for alpha in alpha_grid:
        w           = results[alpha]["weights"]
        r           = results[alpha]["realized_returns"]
        k_paper     = results[alpha]["kappa_paper"]
        wealth_index = np.cumprod(1.0 + r)

        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        # --- Left panel: Asset weights bar chart ---
        active_indices = np.where(w > 1e-4)[0]

        if len(active_indices) == 0:
            axs[0].text(
                0.5, 0.5,
                "No active assets\n(zero portfolio at κ = κ_max)",
                ha="center", va="center", fontsize=13, color="gray",
                transform=axs[0].transAxes,
            )
            axs[0].set_title("Active Asset Weights (Total: 0)",
                             fontsize=12, fontweight="bold")
        else:
            active_weights = w[active_indices]
            active_names   = [asset_names[i] for i in active_indices]

            sort_idx       = np.argsort(active_weights)[::-1]
            active_weights = active_weights[sort_idx]
            active_names   = [active_names[i] for i in sort_idx]

            x_pos = np.arange(len(active_names))
            axs[0].bar(x_pos, active_weights,
                       color="steelblue", edgecolor="black", alpha=0.8)
            axs[0].set_title(
                f"Active Asset Weights (Total: {len(active_indices)})",
                fontsize=12, fontweight="bold",
            )
            axs[0].set_ylabel("Weight Allocation", fontsize=10)
            # Fix: set_xticks before set_xticklabels
            axs[0].set_xticks(x_pos)
            axs[0].set_xticklabels(active_names, rotation=45,
                                   ha="right", fontsize=8)
            axs[0].grid(True, axis="y", linestyle=":", alpha=0.6)

        # --- Right panel: Cumulative wealth (log scale) ---
        axs[1].plot(dates, wealth_index, color="darkgreen", linewidth=2)
        axs[1].set_yscale("log")
        axs[1].set_title("Portfolio Cumulative Wealth Index (log scale)",
                         fontsize=12, fontweight="bold")
        axs[1].set_xlabel("Date", fontsize=10)
        axs[1].set_ylabel("Wealth (Start = 1.0, log scale)", fontsize=10)
        axs[1].grid(True, linestyle=":", alpha=0.6, which="both")

        plt.suptitle(
            f"Portfolio Dashboard  |  α = {alpha:.2f}  |  "
            f"Paper κ = {k_paper:.4f}",
            fontsize=14, fontweight="bold", y=0.995,
        )
        plt.tight_layout()
        save_path = individual_dir / f"portfolio_alpha_{alpha:.2f}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# 6. Monte Carlo expected-return distributions grid
# ---------------------------------------------------------------------------

def plot_monte_carlo_distributions(results, alpha_grid, mu, Sigma, Omega,
                                   T, save_path=None):
    """
    2×4 grid of expected-return scenario distributions across κ levels.

    Fixes
    -----
    - Grid indices computed dynamically from len(alpha_grid), not hard-coded
    - Last subplot (k ≈ kappa_max, near-zero weights) uses axvline guard,
      same as for k=0 (std_val < 1e-7 or k_paper ≈ 0)
    """
    N_PANELS = 8
    fig, axs = plt.subplots(2, 4, figsize=(18, 10))
    axs = axs.ravel()

    # Dynamic selection: N_PANELS evenly-spaced indices across the full grid
    n_alphas = len(alpha_grid)
    raw_indices = np.linspace(0, n_alphas - 1, N_PANELS).round().astype(int)
    selected_alphas = [alpha_grid[i] for i in raw_indices]

    N = len(mu)
    M = 5000

    np.random.seed(42)
    V = np.random.normal(0.0, 1.0, size=(M, N))
    Z = V / np.linalg.norm(V, axis=1, keepdims=True)

    omega_sqrt = np.sqrt(np.maximum(np.diag(Omega), 0.0))

    for idx, alpha in enumerate(selected_alphas):
        w       = results[alpha]["weights"]
        k_paper = results[alpha]["kappa_paper"]
        ax      = axs[idx]

        expected_base   = float(w @ mu)
        scale_term      = w * omega_sqrt
        scenario_returns = expected_base + k_paper * (Z @ scale_term)

        mean_val = float(np.mean(scenario_returns))
        std_val  = float(np.std(scenario_returns))

        # Guard: degenerate case (k=0 or k≈kappa_max → zero weights)
        if np.isclose(k_paper, 0.0, atol=1e-9) or std_val < 1e-7:
            ax.axvline(x=expected_base, color="steelblue", linewidth=2.5)
            ax.set_title(
                f"k={k_paper:.3f}  mean={mean_val:.4f}  (no spread)",
                fontsize=9, fontweight="bold",
            )
            # Sensible x-limits when everything is degenerate
            span = max(abs(expected_base) * 0.1, 1e-4)
            ax.set_xlim(expected_base - span, expected_base + span)
        else:
            ax.hist(scenario_returns, bins=35, density=True,
                    color="lightblue", alpha=0.7, edgecolor="none")
            kde    = gaussian_kde(scenario_returns)
            x_min  = scenario_returns.min() - 1.5 * std_val
            x_max  = scenario_returns.max() + 1.5 * std_val
            x_grid = np.linspace(x_min, x_max, 250)
            ax.plot(x_grid, kde(x_grid), color="darkred", linewidth=2.0)
            ax.set_title(
                f"k={k_paper:.3f}  mean={mean_val:.4f}  std={std_val:.4f}",
                fontsize=9, fontweight="bold",
            )

        ax.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=8)

    plt.suptitle(
        "Monte Carlo Expected-Return Distributions vs. Robustness Level (κ)",
        fontsize=15, fontweight="bold", y=0.995,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
