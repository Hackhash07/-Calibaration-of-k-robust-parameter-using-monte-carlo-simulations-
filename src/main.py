"""
main.py

Main entry point for running the robust portfolio calibration pipeline.
Orchestrates data loading, statistics estimation, optimization sweep,
results saving, plotting, and prints a final analysis report.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    FIGURE_DIR,
    RETURN_DIR,
    WEIGHT_DIR,
)
from experiment import run_calibration_experiment
import plotting


def main():
    print("======================================================================")
    print("Starting Robust Portfolio Calibration Pipeline")
    print("======================================================================\n")

    # 1. Run the calibration sweep
    print("1. Running experiment sweep over alpha grid...")
    results, meta = run_calibration_experiment()
    print("   Experiment sweep completed successfully!\n")

    alpha_grid = meta["alpha_grid"]
    asset_names = meta["asset_names"]
    dates = meta["dates"]

    # 2. Format and save weights to CSV
    print("2. Formatting and saving optimal weights...")
    weights_records = []
    for alpha in alpha_grid:
        weights_records.append(results[alpha]["weights"])
    
    df_weights = pd.DataFrame(weights_records, index=alpha_grid, columns=asset_names)
    df_weights.index.name = "Alpha"
    weights_csv_path = WEIGHT_DIR / "portfolio_weights.csv"
    df_weights.to_csv(weights_csv_path)
    print(f"   Weights saved to: {weights_csv_path}\n")

    # 3. Format and save realized returns to CSV
    print("3. Formatting and saving realized returns...")
    returns_records = {}
    for alpha in alpha_grid:
        returns_records[f"alpha_{alpha:.2f}"] = results[alpha]["realized_returns"]
        
    df_returns = pd.DataFrame(returns_records, index=dates)
    returns_csv_path = RETURN_DIR / "portfolio_returns.csv"
    df_returns.to_csv(returns_csv_path)
    print(f"   Returns saved to: {returns_csv_path}\n")

    # 4. Generate and save plots
    print("4. Generating visualization plots...")
    
    # Chart 1: Stacked weight dynamics sweep
    plotting.plot_weight_sweep(
        results, 
        alpha_grid, 
        asset_names, 
        save_path=FIGURE_DIR / "weight_sweep.png"
    )
    
    # Chart 2: Return distributions comparing MVO, Rule of Thumb, and Robust
    # Finding alpha closest to 0.23 (half of average Sharpe ratio of 0.46)
    closest_rot_alpha = alpha_grid[np.argmin(np.abs(alpha_grid - 0.23))]
    selected_alphas = [0.0, closest_rot_alpha, 0.5]
    plotting.plot_return_distributions(
        results, 
        selected_alphas, 
        save_path=FIGURE_DIR / "return_distributions.png"
    )
    
    # Chart 3: Performance metrics grid
    plotting.plot_portfolio_metrics(
        results, 
        alpha_grid, 
        save_path=FIGURE_DIR / "portfolio_metrics.png"
    )
    print(f"   Plots saved to: {FIGURE_DIR}\n")

    # 5. Print a final summary report to the console
    print("======================================================================")
    print("PORTFOLIO PERFORMANCE SUMMARY ANALYSIS")
    print("======================================================================")
    print(f"{'Robustness (Alpha)':<20} | {'Ann. Return':<12} | {'Ann. Vol':<10} | {'Ann. Sharpe':<11} | {'HHI':<6} | {'Active Assets'}")
    print("-" * 80)
    
    # Print statistics for MVO, Rule of Thumb (approx 0.23), and Robust (0.50)
    report_alphas = [0.0, closest_rot_alpha, 0.50]
    for alpha in report_alphas:
        # Find closest alpha in actual grid
        grid_alpha = alpha_grid[np.argmin(np.abs(alpha_grid - alpha))]
        
        w = results[grid_alpha]["weights"]
        r = results[grid_alpha]["realized_returns"]
        
        # Annualized metrics (assuming daily returns)
        ann_return = np.mean(r) * 252
        ann_vol = np.std(r) * np.sqrt(252)
        ann_sharpe = ann_return / ann_vol if ann_vol > 1e-8 else 0.0
        
        # Portfolio concentration (Herfindahl-Hirschman Index)
        hhi = np.sum(w ** 2)
        active_assets = sum(w > 1e-4)
        
        label = f"{grid_alpha:.2f}"
        if grid_alpha == 0.0:
            label += " (MVO)"
        elif np.isclose(grid_alpha, closest_rot_alpha):
            label += " (Rule of Thumb)"
            
        print(f"{label:<20} | {ann_return:>11.2%} | {ann_vol:>9.2%} | {ann_sharpe:>11.2f} | {hhi:>5.3f} | {active_assets:>13}")
        
    print("======================================================================\n")
    print("Pipeline execution finished successfully!")


if __name__ == "__main__":
    main()
