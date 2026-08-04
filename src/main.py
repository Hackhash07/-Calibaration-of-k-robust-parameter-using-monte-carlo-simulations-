"""
main.py

Main entry point for the robust portfolio calibration pipeline.
Runs BOTH the symmetric Goldfarb–Iyengar model and the asymmetric
semi-variance model, then generates individual + comparison plots.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config import FIGURE_DIR, RETURN_DIR, WEIGHT_DIR
from experiment import run_calibration_experiment
from asymmetric_experiment import run_asymmetric_experiment
import plotting
import comparison_plotting

# Comparison figures go in their own sub-directory
COMPARE_DIR = FIGURE_DIR / "comparison"
COMPARE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    sep = "=" * 70

    # ── 1. Symmetric experiment ───────────────────────────────────────────────
    print(sep)
    print("STEP 1 — Symmetric Robust Calibration (Goldfarb–Iyengar)")
    print(sep)
    sym_results, sym_meta = run_calibration_experiment()
    print("   ✓ Symmetric sweep complete\n")

    alpha_grid  = sym_meta["alpha_grid"]
    asset_names = sym_meta["asset_names"]
    dates       = sym_meta["dates"]

    # ── 2. Asymmetric experiment ──────────────────────────────────────────────
    print(sep)
    print("STEP 2 — Asymmetric Robust Calibration (Semi-Variance Omega)")
    print(sep)
    asym_results, asym_meta = run_asymmetric_experiment()
    print("   ✓ Asymmetric sweep complete\n")

    # ── 3. Save CSVs (symmetric as primary, asymmetric alongside) ────────────
    print(sep)
    print("STEP 3 — Saving weights and returns CSVs")
    print(sep)

    for tag, results in [("symmetric", sym_results), ("asymmetric", asym_results)]:
        df_w = pd.DataFrame(
            [results[a]["weights"] for a in alpha_grid],
            index=alpha_grid, columns=asset_names,
        )
        df_w.index.name = "Alpha"
        p = WEIGHT_DIR / f"portfolio_weights_{tag}.csv"
        df_w.to_csv(p)
        print(f"   Weights ({tag}) → {p}")

        df_r = pd.DataFrame(
            {f"alpha_{a:.2f}": results[a]["realized_returns"] for a in alpha_grid},
            index=dates,
        )
        p = RETURN_DIR / f"portfolio_returns_{tag}.csv"
        df_r.to_csv(p)
        print(f"   Returns ({tag}) → {p}")

    print()

    # ── 4. Individual symmetric plots (existing) ──────────────────────────────
    print(sep)
    print("STEP 4 — Symmetric individual plots")
    print(sep)

    closest_rot  = alpha_grid[np.argmin(np.abs(alpha_grid - 0.23))]
    closest_75   = alpha_grid[np.argmin(np.abs(alpha_grid - 0.75))]
    sel_alphas   = [0.0, closest_rot, 0.5, closest_75, 1.0]

    plotting.plot_weight_sweep(
        sym_results, alpha_grid, asset_names,
        save_path=FIGURE_DIR / "weight_sweep.png",
    )
    plotting.plot_return_distributions(
        sym_results, sel_alphas,
        save_path=FIGURE_DIR / "return_distributions.png",
    )
    plotting.plot_portfolio_metrics(
        sym_results, alpha_grid,
        save_path=FIGURE_DIR / "portfolio_metrics.png",
    )
    plotting.plot_cumulative_returns_sweep(
        sym_results, alpha_grid, dates,
        save_path=FIGURE_DIR / "cumulative_returns.png",
    )
    plotting.plot_individual_plots(
        sym_results, alpha_grid, dates, asset_names,
        individual_dir=FIGURE_DIR / "individual",
    )
    plotting.plot_monte_carlo_distributions(
        sym_results, alpha_grid,
        sym_meta["mu"], sym_meta["Sigma"], sym_meta["Omega"],
        T=sym_meta["R"].shape[0],
        save_path=FIGURE_DIR / "monte_carlo_distributions.png",
    )
    print(f"   ✓ Symmetric plots → {FIGURE_DIR}\n")

    # ── 4b. Asymmetric individual plots (mirror of symmetric) ─────────────────
    print(sep)
    print("STEP 4b — Asymmetric individual plots")
    print(sep)

    ASYM_DIR = FIGURE_DIR / "asymmetric"
    ASYM_DIR.mkdir(parents=True, exist_ok=True)

    plotting.plot_weight_sweep(
        asym_results, alpha_grid, asset_names,
        save_path=ASYM_DIR / "weight_sweep.png",
    )
    plotting.plot_return_distributions(
        asym_results, sel_alphas,
        save_path=ASYM_DIR / "return_distributions.png",
    )
    plotting.plot_portfolio_metrics(
        asym_results, alpha_grid,
        save_path=ASYM_DIR / "portfolio_metrics.png",
    )
    plotting.plot_cumulative_returns_sweep(
        asym_results, alpha_grid, dates,
        save_path=ASYM_DIR / "cumulative_returns.png",
    )
    plotting.plot_individual_plots(
        asym_results, alpha_grid, dates, asset_names,
        individual_dir=ASYM_DIR / "individual",
    )
    plotting.plot_monte_carlo_distributions(
        asym_results, alpha_grid,
        asym_meta["mu"], asym_meta["Sigma"], asym_meta["Omega_down"],
        T=asym_meta["R"].shape[0],
        save_path=ASYM_DIR / "monte_carlo_distributions.png",
    )
    print(f"   ✓ Asymmetric plots → {ASYM_DIR}\n")

    # ── 5. Comparison plots ───────────────────────────────────────────────────
    print(sep)
    print("STEP 5 — Symmetric vs Asymmetric comparison plots")
    print(sep)

    comparison_plotting.plot_comparison_metrics(
        sym_results, asym_results, alpha_grid,
        save_path=COMPARE_DIR / "cmp_metrics.png",
    )
    comparison_plotting.plot_comparison_weights(
        sym_results, asym_results, alpha_grid, asset_names,
        save_path=COMPARE_DIR / "cmp_weights.png",
    )
    comparison_plotting.plot_comparison_distributions(
        sym_results, asym_results, sel_alphas,
        save_path=COMPARE_DIR / "cmp_distributions.png",
    )
    comparison_plotting.plot_comparison_wealth(
        sym_results, asym_results, alpha_grid, dates,
        save_path=COMPARE_DIR / "cmp_wealth.png",
    )
    print(f"   ✓ Comparison plots → {COMPARE_DIR}\n")

    # ── 6. Summary table ──────────────────────────────────────────────────────
    print(sep)
    print("PERFORMANCE SUMMARY  (annualised, in-sample)")
    print(sep)
    print(f"{'Model':<22} {'Alpha':<8} {'Ann.Ret':>9} {'Ann.Vol':>9} "
          f"{'Sharpe':>8} {'HHI':>7} {'Active':>7}")
    print("-" * 70)

    for alpha in [0.0, closest_rot, 0.50]:
        ga = alpha_grid[np.argmin(np.abs(alpha_grid - alpha))]
        for tag, res in [("Symmetric", sym_results), ("Asymmetric", asym_results)]:
            r   = res[ga]["realized_returns"]
            w   = res[ga]["weights"]
            ar  = np.mean(r) * 252
            av  = np.std(r) * np.sqrt(252)
            sh  = ar / av if av > 1e-8 else 0.0
            hhi = np.sum(w ** 2)
            act = int(np.sum(w > 1e-4))
            lbl = f"α={ga:.2f}"
            print(f"  {tag:<18} {lbl:<8} {ar:>9.2%} {av:>9.2%} "
                  f"{sh:>8.2f} {hhi:>7.4f} {act:>7}")
        print()

    print(sep)
    print("Pipeline finished successfully!")
    print(sep)


if __name__ == "__main__":
    main()
