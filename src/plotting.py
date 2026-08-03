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
    Plots four key portfolio metrics as a function of alpha:
    - Expected Daily Return
    - Daily Volatility
    - Daily Sharpe Ratio
    - Weight Concentration (Herfindahl-Hirschman Index, HHI)
    """
    expected_returns = []
    volatilities = []
    sharpe_ratios = []
    hhi_values = []
    
    for alpha in alpha_grid:
        w = results[alpha]["weights"]
        r = results[alpha]["realized_returns"]
        
        expected_returns.append(np.mean(r))
        volatilities.append(np.std(r))
        sharpe_ratios.append(np.mean(r) / np.std(r) if np.std(r) > 1e-8 else 0.0)
        hhi_values.append(np.sum(w ** 2))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
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
    axs[1, 0].set_xlabel(r"Robustness Level $\alpha$", fontsize=10)
    axs[1, 0].set_ylabel("Sharpe Ratio", fontsize=10)
    axs[1, 0].grid(True, linestyle=":", alpha=0.6)
    
    # 4. HHI Concentration
    axs[1, 1].plot(alpha_grid, hhi_values, color="purple", marker="d", markersize=4)
    axs[1, 1].set_title("Portfolio Concentration (HHI)", fontsize=12, fontweight="bold")
    axs[1, 1].set_xlabel(r"Robustness Level $\alpha$", fontsize=10)
    axs[1, 1].set_ylabel("HHI (1/N = equal, 1 = concentrated)", fontsize=10)
    axs[1, 1].grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle("Portfolio Performance Metrics vs. Robustness Sweep", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
