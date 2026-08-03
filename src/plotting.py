"""
plotting.py

Generates and saves performance plots, weight allocations, and return distribution
visualizations for the robust portfolio calibration sweeps.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from config import FIGURE_DIR


def plot_weight_sweep(results, alpha_grid, asset_names, save_path=None):
    """
    Creates a stacked area plot showing how weights vary across the alpha sweep.
    """
    n_assets = len(asset_names)
    n_alphas = len(alpha_grid)
    
    # Extract weights for each alpha
    weights_matrix = np.zeros((n_assets, n_alphas))
    for idx, alpha in enumerate(alpha_grid):
        weights_matrix[:, idx] = results[alpha]["weights"]

    plt.figure(figsize=(10, 6))
    plt.stackplot(alpha_grid, weights_matrix, labels=asset_names)
    
    plt.title("Portfolio Weights Dynamics across Robustness Sweep", fontsize=14, fontweight="bold")
    plt.xlabel(r"Robustness Level $\alpha$ (0 = MVO, 1 = Max Robustness)", fontsize=12)
    plt.ylabel("Portfolio Weights", fontsize=12)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    
    # Place legend outside for readability if there are many assets
    if n_assets <= 10:
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
    else:
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0), ncol=2, fontsize=8)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_return_distributions(results, selected_alphas, save_path=None):
    """
    Plots the histograms and Kernel Density Estimations (KDE) comparing realized returns
    for key robustness levels.
    """
    plt.figure(figsize=(10, 6))
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for idx, alpha in enumerate(selected_alphas):
        if alpha not in results:
            continue
            
        returns = results[alpha]["realized_returns"]
        color = colors[idx % len(colors)]
        
        # Plot KDE
        kde = gaussian_kde(returns)
        x_grid = np.linspace(returns.min() - 0.01, returns.max() + 0.01, 200)
        plt.plot(
            x_grid, 
            kde(x_grid), 
            color=color, 
            label=f"Alpha = {alpha:.2f} (Paper kappa = {results[alpha]['kappa_paper']:.2f})", 
            linewidth=2.5
        )
        # Plot Histogram
        plt.hist(returns, bins=30, density=True, alpha=0.15, color=color)

    plt.title("Realized Returns Distribution Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Portfolio Daily Returns", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_portfolio_metrics(results, alpha_grid, save_path=None):
    """
    Plots six key portfolio metrics as a function of alpha:
    - Expected Daily Return
    - Daily Volatility
    - Daily Sharpe Ratio
    - Weight Concentration (Herfindahl-Hirschman Index, HHI)
    - Gross Exposure (Sum of |w_i|)
    - Active Asset Count (number of assets with w_i > 1e-4)
    """
    expected_returns = []
    volatilities = []
    sharpe_ratios = []
    hhi_values = []
    gross_exposures = []
    active_counts = []
    
    for alpha in alpha_grid:
        w = results[alpha]["weights"]
        r = results[alpha]["realized_returns"]
        
        expected_returns.append(np.mean(r))
        volatilities.append(np.std(r))
        sharpe_ratios.append(np.mean(r) / np.std(r) if np.std(r) > 1e-8 else 0.0)
        hhi_values.append(np.sum(w ** 2))
        gross_exposures.append(np.sum(np.abs(w)))
        active_counts.append(np.sum(w > 1e-4))

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    
    # 1. Expected Return
    axs[0, 0].plot(alpha_grid, expected_returns, color="darkblue", marker="o", markersize=4)
    axs[0, 0].set_title("Expected Daily Return", fontsize=12, fontweight="bold")
    axs[0, 0].set_ylabel("Return", fontsize=10)
    axs[0, 0].grid(True, linestyle=":", alpha=0.6)
    
    # 2. Volatility
    axs[0, 1].plot(alpha_grid, volatilities, color="darkred", marker="s", markersize=4)
    axs[0, 1].set_title("Daily Volatility (Risk)", fontsize=12, fontweight="bold")
    axs[0, 1].set_ylabel("Standard Deviation", fontsize=10)
    axs[0, 1].grid(True, linestyle=":", alpha=0.6)
    
    # 3. Sharpe Ratio
    axs[1, 0].plot(alpha_grid, sharpe_ratios, color="darkgreen", marker="^", markersize=4)
    axs[1, 0].set_title("Daily Sharpe Ratio", fontsize=12, fontweight="bold")
    axs[1, 0].set_ylabel("Sharpe Ratio", fontsize=10)
    axs[1, 0].grid(True, linestyle=":", alpha=0.6)
    
    # 4. HHI Concentration
    axs[1, 1].plot(alpha_grid, hhi_values, color="purple", marker="d", markersize=4)
    axs[1, 1].set_title("Portfolio Concentration (HHI)", fontsize=12, fontweight="bold")
    axs[1, 1].set_ylabel("HHI (1/N = equal, 1 = concentrated)", fontsize=10)
    axs[1, 1].grid(True, linestyle=":", alpha=0.6)

    # 5. Gross Exposure
    axs[2, 0].plot(alpha_grid, gross_exposures, color="darkorange", marker="p", markersize=4)
    axs[2, 0].set_title("Gross Exposure (Sum of |w_i|)", fontsize=12, fontweight="bold")
    axs[2, 0].set_xlabel(r"Robustness Level $\alpha$", fontsize=10)
    axs[2, 0].set_ylabel("Gross Exposure", fontsize=10)
    axs[2, 0].grid(True, linestyle=":", alpha=0.6)

    # 6. Active Asset Count
    axs[2, 1].plot(alpha_grid, active_counts, color="teal", marker="x", markersize=4)
    axs[2, 1].set_title("Number of Active Assets", fontsize=12, fontweight="bold")
    axs[2, 1].set_xlabel(r"Robustness Level $\alpha$", fontsize=10)
    axs[2, 1].set_ylabel("Active Asset Count", fontsize=10)
    axs[2, 1].grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle("Portfolio Performance Metrics vs. Robustness Sweep", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cumulative_returns_sweep(results, alpha_grid, dates, save_path=None):
    """
    Plots the cumulative returns (wealth index starting at 1.0) of all portfolios
    on a single chart using a color gradient from MVO to Max Robustness.
    """
    plt.figure(figsize=(10, 6))
    
    # Use colormap to color lines continuously
    cmap = plt.get_cmap("plasma")
    n_alphas = len(alpha_grid)
    
    for idx, alpha in enumerate(alpha_grid):
        r = results[alpha]["realized_returns"]
        # Wealth index: cumulative product of (1 + r)
        wealth_index = np.cumprod(1 + r)
        
        # Color based on relative position in the grid
        color = cmap(idx / max(1, n_alphas - 1))
        
        # Only label selected alpha thresholds to keep legend clean
        label = f"Alpha = {alpha:.2f}" if idx in [0, int(n_alphas/4), int(n_alphas/2), int(3*n_alphas/4), n_alphas-1] else None
        
        plt.plot(dates, wealth_index, color=color, label=label, linewidth=1.5, alpha=0.8)

    plt.title("Portfolio Cumulative Wealth Dynamics (Grid Sweep)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Wealth Index (Start = 1.0)", fontsize=12)
    plt.legend(fontsize=10, loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_individual_plots(results, alpha_grid, dates, asset_names, individual_dir):
    """
    Generates an individual 2-panel summary plot for each alpha value:
    - Left: Bar chart of the asset weights.
    - Right: Line chart of the cumulative return index.
    """
    individual_dir = Path(individual_dir)
    individual_dir.mkdir(parents=True, exist_ok=True)
    
    for alpha in alpha_grid:
        w = results[alpha]["weights"]
        r = results[alpha]["realized_returns"]
        k_paper = results[alpha]["kappa_paper"]
        wealth_index = np.cumprod(1 + r)
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Weights Bar Chart (Only plot assets with weights > 1e-4)
        active_indices = np.where(w > 1e-4)[0]
        active_weights = w[active_indices]
        active_names = [asset_names[i] for i in active_indices]
        
        # Sort by weight descending
        sort_idx = np.argsort(active_weights)[::-1]
        active_weights = active_weights[sort_idx]
        active_names = [active_names[i] for i in sort_idx]
        
        axs[0].bar(active_names, active_weights, color="steelblue", edgecolor="black", alpha=0.8)
        axs[0].set_title(f"Active Asset Weights (Total: {len(active_indices)})", fontsize=12, fontweight="bold")
        axs[0].set_ylabel("Weight Allocation", fontsize=10)
        axs[0].set_xticklabels(active_names, rotation=45, ha="right", fontsize=8)
        axs[0].grid(True, axis="y", linestyle=":", alpha=0.6)
        
        # 2. Cumulative Returns Line Chart
        axs[1].plot(dates, wealth_index, color="darkgreen", linewidth=2)
        axs[1].set_title("Portfolio Cumulative Wealth Index", fontsize=12, fontweight="bold")
        axs[1].set_xlabel("Date", fontsize=10)
        axs[1].set_ylabel("Wealth (Start = 1.0)", fontsize=10)
        axs[1].grid(True, linestyle=":", alpha=0.6)
        
        plt.suptitle(
            f"Portfolio Dashboard: Alpha = {alpha:.2f} (Paper Kappa = {k_paper:.4f})", 
            fontsize=15, 
            fontweight="bold", 
            y=0.98
        )
        plt.tight_layout()
        save_path = individual_dir / f"portfolio_alpha_{alpha:.2f}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()


def plot_monte_carlo_distributions(results, alpha_grid, mu, Sigma, Omega, T, save_path=None):
    """
    Plots the expected returns distribution under Monte Carlo simulations
    for the first 8 portfolios in the alpha sweep, formatted as a 2x4 grid.
    """
    fig, axs = plt.subplots(2, 4, figsize=(18, 10))
    axs = axs.ravel()  # Flatten into 8 subplots
    
    n_alphas = len(alpha_grid)
    # Take 8 alphas evenly spaced across the entire grid (from index 0 to index 20)
    selected_indices = [0, 2, 5, 8, 11, 14, 17, 20]
    selected_alphas = [alpha_grid[i] for i in selected_indices]
    
    # Run Monte Carlo simulation for N assets
    N = len(mu)
    M = 5000  # Number of scenario paths
    
    # Sample z uniformly on the surface of N-dimensional unit sphere
    np.random.seed(42)
    V = np.random.normal(0, 1, size=(M, N))
    # Normalize to project onto unit sphere surface (each row has L2 norm = 1.0)
    Z = V / np.linalg.norm(V, axis=1, keepdims=True)
    
    # Scale elements by the diagonal of Omega (which is diag(Sigma)/T)
    omega_diag = np.diag(Omega)
    omega_sqrt = np.sqrt(np.maximum(omega_diag, 0.0))
    
    for idx, alpha in enumerate(selected_alphas):
        w = results[alpha]["weights"]
        k_paper = results[alpha]["kappa_paper"]
        
        # Scenario expected return: w^T mu_sim = w^T mu + k_paper * (w * omega_sqrt) @ z^T
        expected_base = w @ mu
        scale_term = w * omega_sqrt
        
        # Daily expected returns for each scenario
        scenario_returns = expected_base + k_paper * (Z @ scale_term)
        
        mean_val = np.mean(scenario_returns)
        std_val = np.std(scenario_returns)
        
        ax = axs[idx]
        
        if np.isclose(k_paper, 0.0) or std_val < 1e-7:
            # Special case for k = 0 (no spread)
            ax.axvline(x=expected_base, color="steelblue", linewidth=2.5)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"k={k_paper:.3f} mean={mean_val:.4f} (no spread)", fontsize=10, fontweight="bold")
        else:
            # General case for k > 0 (normal-like distribution of expected returns)
            # Plot density histogram
            ax.hist(scenario_returns, bins=35, density=True, color="lightblue", alpha=0.7, edgecolor="none")
            
            # Plot KDE curve
            kde = gaussian_kde(scenario_returns)
            # Calculate range based on standard deviations to fit the curve nicely
            x_min, x_max = scenario_returns.min() - 1.5*std_val, scenario_returns.max() + 1.5*std_val
            x_grid = np.linspace(x_min, x_max, 250)
            ax.plot(x_grid, kde(x_grid), color="darkred", linewidth=2.0)
            
            ax.set_title(f"k={k_paper:.3f} mean={mean_val:.4f} std={std_val:.4f}", fontsize=10, fontweight="bold")
            
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
    plt.suptitle("Monte Carlo Expected Return Distributions vs. Robustness Level (k)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


