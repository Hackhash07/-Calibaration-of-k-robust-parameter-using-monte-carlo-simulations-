"""
comparison_plotting.py

Side-by-side comparison plots for symmetric vs asymmetric robust portfolio models.

Figures produced
-----------------
1. cmp_metrics.png        — 6 metric panels, both models overlaid (solid vs dashed)
2. cmp_weights.png        — 1×2 stacked-area weight dynamics
3. cmp_distributions.png  — 5 KDE panels (one per alpha), both models per panel
4. cmp_wealth.png         — 1×2 log-scale wealth trajectories
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import gaussian_kde


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_metrics(results, alpha_grid):
    """Extract six scalar metric series from a results dict."""
    exp_ret, vols, sharpes, hhis, gross, active = [], [], [], [], [], []
    for alpha in alpha_grid:
        w = results[alpha]["weights"]
        r = results[alpha]["realized_returns"]
        vol = float(np.std(r))
        exp_ret.append(float(np.mean(r)))
        vols.append(vol)
        sharpes.append(float(np.mean(r)) / vol if vol > 1e-8 else 0.0)
        hhis.append(float(np.sum(w ** 2)))
        gross.append(float(np.sum(np.abs(w))))
        active.append(int(np.sum(w > 1e-4)))
    return exp_ret, vols, sharpes, hhis, gross, active


def _ref_std(results, alpha_grid):
    """Return std of the alpha=0 (MVO) portfolio — used for degenerate threshold."""
    return float(np.std(results[alpha_grid[0]]["realized_returns"]))


def _is_degenerate(returns, degen_thr):
    return float(np.std(returns)) < degen_thr


# ── 1. Overlaid Metrics ────────────────────────────────────────────────────────

def plot_comparison_metrics(sym_results, asym_results, alpha_grid, save_path=None):
    """
    6-panel metrics grid with symmetric (solid) and asymmetric (dashed) overlaid.
    """
    sym_m  = _build_metrics(sym_results,  alpha_grid)
    asym_m = _build_metrics(asym_results, alpha_grid)

    titles  = [
        "Expected Daily Return",
        "Daily Volatility (Risk)",
        "Daily Sharpe Ratio",
        "Portfolio Concentration (HHI)",
        r"Gross Exposure ($\Sigma|w_i|$)",
        "Number of Active Assets",
    ]
    ylabels = [
        "Return", "Std Dev", "Sharpe",
        "HHI", "Gross Exposure", "Active Assets",
    ]
    colors  = ["#1f77b4", "#d62728", "#2ca02c",
               "#9467bd", "#ff7f0e", "#8c564b"]

    fig, axs = plt.subplots(3, 2, figsize=(13, 12))
    axs = axs.ravel()

    for idx in range(6):
        ax = axs[idx]
        ax.plot(alpha_grid, sym_m[idx],  color=colors[idx], linewidth=2.0,
                marker="o", markersize=3, label="Symmetric (GI)")
        ax.plot(alpha_grid, asym_m[idx], color=colors[idx], linewidth=2.0,
                marker="s", markersize=3, linestyle="--", label="Asymmetric (semi-var)")
        ax.set_title(titles[idx],  fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabels[idx], fontsize=9)
        if idx >= 4:
            ax.set_xlabel(r"Robustness Level $\alpha$", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(
        "Symmetric vs Asymmetric — Portfolio Metrics Sweep",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ── 2. Weight Dynamics Side-by-Side ────────────────────────────────────────────

def plot_comparison_weights(sym_results, asym_results, alpha_grid,
                            asset_names, save_path=None):
    """
    Two stacked area plots side-by-side (same scale).
    """
    n_assets = len(asset_names)
    n_alphas = len(alpha_grid)

    def _weights_matrix(results):
        W = np.zeros((n_assets, n_alphas))
        for j, alpha in enumerate(alpha_grid):
            W[:, j] = np.maximum(results[alpha]["weights"], 0.0)
        return W

    W_sym  = _weights_matrix(sym_results)
    W_asym = _weights_matrix(asym_results)
    ylim_top = max(
        W_sym.sum(axis=0).max(),
        W_asym.sum(axis=0).max(),
    ) * 1.05
    ylim_top = max(ylim_top, 1.05)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(n_assets - 1, 1)) for i in range(n_assets)]

    for ax, W, title in [
        (axs[0], W_sym,  "Symmetric GI"),
        (axs[1], W_asym, "Asymmetric (Semi-Var)"),
    ]:
        ax.stackplot(alpha_grid, W, labels=asset_names, colors=colors)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(r"Robustness Level $\alpha$", fontsize=11)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, ylim_top)
        ax.grid(True, linestyle=":", alpha=0.4)

    axs[0].set_ylabel("Portfolio Weight Allocation", fontsize=11)
    axs[1].legend(
        loc="upper right", bbox_to_anchor=(1.28, 1.0),
        ncol=1, fontsize=7, framealpha=0.8,
    )

    plt.suptitle(
        "Symmetric vs Asymmetric — Weight Dynamics",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ── 3. Return Distribution Comparison ──────────────────────────────────────────

def plot_comparison_distributions(sym_results, asym_results,
                                  selected_alphas, save_path=None):
    """
    For each selected alpha: one panel showing both models' KDE overlaid.
    Blue = symmetric, Orange = asymmetric.
    """
    n = len(selected_alphas)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axs = np.array(axs).ravel()

    # Reference std for degenerate threshold
    ref_sym_std  = _ref_std(sym_results,  list(sym_results.keys()))
    ref_asym_std = _ref_std(asym_results, list(asym_results.keys()))
    degen_sym  = max(ref_sym_std  * 0.05, 1e-6)
    degen_asym = max(ref_asym_std * 0.05, 1e-6)

    max_density = 1.0

    for idx, alpha in enumerate(selected_alphas):
        ax = axs[idx]

        for (res, degen_thr, color, lw, ls, label) in [
            (sym_results,  degen_sym,  "#1f77b4", 2.5, "-",  "Symmetric GI"),
            (asym_results, degen_asym, "#d62728", 2.5, "--", "Asymmetric"),
        ]:
            if alpha not in res:
                continue
            returns = res[alpha]["realized_returns"]
            kp      = res[alpha]["kappa_paper"]

            if _is_degenerate(returns, degen_thr):
                ax.axvline(
                    x=float(np.mean(returns)),
                    color=color, linestyle=":", linewidth=2.0,
                    label=f"{label} (zero portfolio)",
                )
            else:
                kde    = gaussian_kde(returns)
                x_grid = np.linspace(returns.min() - 0.005,
                                     returns.max() + 0.005, 250)
                kv     = kde(x_grid)
                max_density = max(max_density, float(np.max(kv)))
                ax.plot(x_grid, kv, color=color, linewidth=lw,
                        linestyle=ls, label=f"{label} (κ={kp:.3f})")
                ax.fill_between(x_grid, kv, alpha=0.08, color=color)

        ax.set_title(f"α = {alpha:.2f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Daily Returns", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Hide empty panels
    for j in range(idx + 1, len(axs)):
        axs[j].set_visible(False)

    # Uniform y-axis across all panels
    for ax in axs[:n]:
        ax.set_ylim(0.0, max_density * 1.15)

    plt.suptitle(
        "Symmetric vs Asymmetric — Realized Return Distributions",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ── 4. Wealth Index Side-by-Side ───────────────────────────────────────────────

def plot_comparison_wealth(sym_results, asym_results, alpha_grid,
                           dates, save_path=None):
    """
    Two log-scale wealth charts side-by-side (same y-axis limits).
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    cmap = plt.get_cmap("plasma")
    n    = len(alpha_grid)
    label_set = {0, n // 4, n // 2, 3 * n // 4, n - 1}

    all_wealth = []
    for alpha in alpha_grid:
        for res in [sym_results, asym_results]:
            r = res[alpha]["realized_returns"]
            all_wealth.extend(np.cumprod(1.0 + r).tolist())

    y_min = max(min(all_wealth) * 0.9, 1e-4)
    y_max = max(all_wealth) * 1.1

    for ax, results, title in [
        (axs[0], sym_results,  "Symmetric GI"),
        (axs[1], asym_results, "Asymmetric (Semi-Var)"),
    ]:
        for idx, alpha in enumerate(alpha_grid):
            r      = results[alpha]["realized_returns"]
            wealth = np.cumprod(1.0 + r)
            color  = cmap(idx / max(1, n - 1))
            label  = f"α={alpha:.2f}" if idx in label_set else None
            ax.plot(dates, wealth, color=color, linewidth=1.5,
                    alpha=0.85, label=label)

        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Date", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5, which="both")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)

    axs[0].set_ylabel("Wealth Index (log scale, start = 1.0)", fontsize=11)

    plt.suptitle(
        "Symmetric vs Asymmetric — Cumulative Wealth Dynamics",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
