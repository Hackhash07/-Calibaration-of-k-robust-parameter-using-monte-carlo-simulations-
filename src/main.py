"""
main.py

Main entry point.  Runs symmetric + asymmetric sweeps, saves CSVs,
then produces FOUR consolidated figures — no per-alpha individual files.

Output layout
─────────────
output/figures/
  symmetric_summary.png          ← 6-panel symmetric model overview
  asymmetric_summary.png         ← 6-panel asymmetric model overview
  comparison/
    cmp_metrics.png              ← 6 metric panels, both models overlaid
    cmp_distributions.png        ← KDE per selected alpha, both models
    cmp_wealth.png               ← wealth trajectories side-by-side
    cmp_weights.png              ← stacked area side-by-side
    cmp_asymmetry_analysis.png   ← WHY they differ: skew ratios + weight delta
"""

import numpy as np
import pandas as pd

from config import FIGURE_DIR, RETURN_DIR, WEIGHT_DIR
from experiment import run_calibration_experiment
from asymmetric_experiment import run_asymmetric_experiment
import plotting
import comparison_plotting

COMPARE_DIR = FIGURE_DIR / "comparison"
COMPARE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    sep = "=" * 70

    # ── 1. Symmetric ──────────────────────────────────────────────────────────
    print(sep)
    print("STEP 1 — Symmetric Robust Calibration (Goldfarb–Iyengar)")
    print(sep)
    sym_results, sym_meta = run_calibration_experiment()
    print("   ✓ Symmetric sweep complete\n")

    alpha_grid  = sym_meta["alpha_grid"]
    asset_names = sym_meta["asset_names"]
    dates       = sym_meta["dates"]

    # ── 2. Asymmetric ─────────────────────────────────────────────────────────
    print(sep)
    print("STEP 2 — Asymmetric Robust Calibration (Semi-Variance Omega)")
    print(sep)
    asym_results, asym_meta = run_asymmetric_experiment()
    print("   ✓ Asymmetric sweep complete\n")

    # ── 3. Save CSVs ──────────────────────────────────────────────────────────
    print(sep)
    print("STEP 3 — Saving CSVs")
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

    # ── Selected alpha checkpoints ────────────────────────────────────────────
    closest_rot = alpha_grid[np.argmin(np.abs(alpha_grid - 0.23))]
    closest_75  = alpha_grid[np.argmin(np.abs(alpha_grid - 0.75))]
    sel_alphas  = [0.0, closest_rot, 0.5, closest_75, 1.0]

    # ── 4. Symmetric summary (all key plots in ONE figure) ────────────────────
    print(sep)
    print("STEP 4 — Symmetric consolidated summary")
    print(sep)

    plotting.plot_model_summary(
        results=sym_results,
        alpha_grid=alpha_grid,
        asset_names=asset_names,
        dates=dates,
        sel_alphas=sel_alphas,
        mu=sym_meta["mu"],
        Sigma=sym_meta["Sigma"],
        Omega=sym_meta["Omega"],
        T=sym_meta["R"].shape[0],
        title="Symmetric Robust Portfolio (Goldfarb–Iyengar)",
        save_path=FIGURE_DIR / "symmetric_summary.png",
    )
    print(f"   ✓ {FIGURE_DIR / 'symmetric_summary.png'}")

    plotting.plot_all_kappa_distributions(
        sym_results, alpha_grid,
        save_path=FIGURE_DIR / "symmetric_kappa_densities.png",
    )
    print(f"   ✓ {FIGURE_DIR / 'symmetric_kappa_densities.png'}\n")

    # ── 5. Asymmetric summary ─────────────────────────────────────────────────
    print(sep)
    print("STEP 5 — Asymmetric consolidated summary")
    print(sep)

    plotting.plot_model_summary(
        results=asym_results,
        alpha_grid=alpha_grid,
        asset_names=asset_names,
        dates=dates,
        sel_alphas=sel_alphas,
        mu=asym_meta["mu"],
        Sigma=asym_meta["Sigma"],
        Omega=asym_meta["Omega_down"],
        T=asym_meta["R"].shape[0],
        title="Asymmetric Robust Portfolio (Semi-Variance Omega)",
        save_path=FIGURE_DIR / "asymmetric_summary.png",
    )
    print(f"   ✓ {FIGURE_DIR / 'asymmetric_summary.png'}")

    plotting.plot_all_kappa_distributions(
        asym_results, alpha_grid,
        save_path=FIGURE_DIR / "asymmetric_kappa_densities.png",
    )
    print(f"   ✓ {FIGURE_DIR / 'asymmetric_kappa_densities.png'}\n")

    # ── 6. Comparison plots ───────────────────────────────────────────────────
    print(sep)
    print("STEP 6 — Symmetric vs Asymmetric comparison plots")
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
    # KEY: asymmetry diagnostic — WHY the models differ
    comparison_plotting.plot_asymmetry_analysis(
        sym_results=sym_results,
        asym_results=asym_results,
        alpha_grid=alpha_grid,
        asset_names=asset_names,
        Sigma=sym_meta["Sigma"],
        Omega_down=asym_meta["Omega_down"],
        T=sym_meta["R"].shape[0],
        save_path=COMPARE_DIR / "cmp_asymmetry_analysis.png",
    )
    print(f"   ✓ Comparison plots → {COMPARE_DIR}\n")

    # ── 7. Summary table ──────────────────────────────────────────────────────
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
            print(f"  {tag:<18} α={ga:.2f}  {ar:>9.2%} {av:>9.2%} "
                  f"{sh:>8.2f} {hhi:>7.4f} {act:>7}")
        print()
    print(sep)
    print("Pipeline finished successfully!")
    print(sep)


if __name__ == "__main__":
    main()
