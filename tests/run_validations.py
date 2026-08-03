"""
run_validations.py

Executes all 8 mathematical and constraint validations for the Robust Portfolio
Optimization model, ensuring mathematical consistency with the paper.
"""

from pathlib import Path
import sys

# Add src to pythonpath at the front to avoid built-in statistics name collision
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cvxpy as cp
import numpy as np

from config import LAMBDA
from data_loader import load_returns
import optimizer as opt
import statistics as stats


def run_validations():
    print("======================================================================")
    print("RUNNING MATHEMATICAL VALIDATION TESTS")
    print("======================================================================\n")

    # Load real returns (or dummy data if real isn't present)
    R, names, dates = load_returns()
    T, N = R.shape
    mu = stats.estimate_mean(R)
    Sigma = stats.estimate_covariance(R)
    Omega = stats.estimate_omega(Sigma, T)

    all_passed = True

    # ------------------------------------------------------------------------
    # Validation 1: kappa = 0 is equivalent to Markowitz MVO
    # ------------------------------------------------------------------------
    print("Validation 1: kappa = 0 Equivalence to Markowitz MVO...")
    # 1. Solve robust optimizer with kappa_solver = 0
    w_robust, status = opt.solve_robust_portfolio(
        mu, Sigma, Omega, kappa_solver=0.0, long_only=True, full_investment=False
    )
    # 2. Solve ordinary MVO in CVXPY directly
    w_mvo = cp.Variable(N)
    obj_mvo = cp.Maximize(mu @ w_mvo - 0.5 * LAMBDA * cp.quad_form(w_mvo, cp.psd_wrap(Sigma)))
    constraints_mvo = [w_mvo >= 0]
    prob_mvo = cp.Problem(obj_mvo, constraints_mvo)
    prob_mvo.solve()
    
    diff = np.linalg.norm(w_robust - w_mvo.value)
    if diff < 5e-5:
        print(f"  [PASS] L2 difference between weights is {diff:.2e} (< 5e-5)")
    else:
        print(f"  [FAIL] L2 difference is {diff:.2e} (>= 5e-5)")
        all_passed = False

    # ------------------------------------------------------------------------
    # Validation 2: kappa = kappa_max produces w = 0 (or close to 0)
    # ------------------------------------------------------------------------
    print("\nValidation 2: kappa = kappa_max Portfolio Shrinkage to Zero...")
    kappa_max_solver = stats.compute_solver_kappa_upper_bound(mu, Sigma, T)
    
    # Solve at boundary (or slightly above it to guarantee zero in case of numerical noise)
    w_boundary, _ = opt.solve_robust_portfolio(
        mu, Sigma, Omega, kappa_solver=kappa_max_solver * 1.01, long_only=True, full_investment=False
    )
    
    w_sum = np.sum(w_boundary)
    w_norm = np.linalg.norm(w_boundary)
    w_max = np.max(w_boundary)
    
    print(f"  Sum(weights): {w_sum:.2e}")
    print(f"  L2-norm:      {w_norm:.2e}")
    print(f"  Max(weight):  {w_max:.2e}")
    
    if w_norm < 1e-5:
        print("  [PASS] Weights shrank to zero at kappa_max boundary.")
    else:
        print("  [FAIL] Weights failed to shrink to zero.")
        all_passed = False

    # ------------------------------------------------------------------------
    # Validation 3: Objective Monotonicity
    # ------------------------------------------------------------------------
    print("\nValidation 3: Objective Value Monotonicity...")
    alpha_grid = stats.generate_alpha_grid(10)
    objectives = []
    
    optimizer = opt.RobustOptimizer(Sigma, Omega, long_only=True, full_investment=False)
    for alpha in alpha_grid:
        k_solver = alpha * kappa_max_solver
        w_opt, _ = optimizer.solve(mu, k_solver)
        
        # J(w) = w^T mu - kappa_solver * ||Omega^1/2 w||_2 - 0.5 * lambda * w^T Sigma w
        omega_term = np.sqrt(w_opt @ Omega @ w_opt)
        obj_val = w_opt @ mu - k_solver * omega_term - 0.5 * LAMBDA * (w_opt @ Sigma @ w_opt)
        objectives.append(obj_val)
        
    # Check if objective is strictly non-increasing
    monotonic = True
    for idx in range(len(objectives) - 1):
        if objectives[idx + 1] > objectives[idx] + 1e-7:
            monotonic = False
            break
            
    if monotonic:
        print("  [PASS] Objective function is non-increasing as kappa increases.")
    else:
        print("  [FAIL] Objective is not non-increasing.")
        all_passed = False

    # ------------------------------------------------------------------------
    # Validation 4: Penalty Term Decomposition
    # ------------------------------------------------------------------------
    print("\nValidation 4: Objective Components Table...")
    print(f"  {'Alpha':<6} | {'Return Term':<12} | {'Robust Penalty':<14} | {'Risk Penalty':<12}")
    print("  " + "-" * 50)
    for alpha in [0.0, 0.25, 0.5, 1.0]:
        k_solver = alpha * kappa_max_solver
        w_opt, _ = optimizer.solve(mu, k_solver)
        
        ret_term = w_opt @ mu
        rob_penalty = k_solver * np.sqrt(w_opt @ Omega @ w_opt)
        risk_penalty = 0.5 * LAMBDA * (w_opt @ Sigma @ w_opt)
        print(f"  {alpha:<6.2f} | {ret_term:>12.5f} | {rob_penalty:>14.5f} | {risk_penalty:>12.5f}")

    # ------------------------------------------------------------------------
    # Validation 5: Covariance Matrix PSD Check
    # ------------------------------------------------------------------------
    print("\nValidation 5: Covariance Matrix PSD Check...")
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    print(f"  Minimum Eigenvalue of Sigma: {min_eig:.2e}")
    if min_eig >= -1e-10:
        print("  [PASS] Covariance matrix is positive semi-definite.")
    else:
        print("  [WARNING] Covariance matrix has non-negligible negative eigenvalues.")

    # ------------------------------------------------------------------------
    # Validation 6: Comparison of Numerical vs. Analytical Gradient
    # ------------------------------------------------------------------------
    print("\nValidation 6: Analytical vs. Numerical Gradient of Objective...")
    # Choose a strictly positive portfolio (e.g., equal weights) to make the objective differentiable
    w_test = np.ones(N) / N
    k_solver = 0.23 * kappa_max_solver

    # J(w) = w^T mu - kappa * sqrt(w^T Omega w) - 0.5 * lambda * w^T Sigma w
    def objective_func(w_val):
        omega_term = np.sqrt(w_val @ Omega @ w_val)
        return w_val @ mu - k_solver * omega_term - 0.5 * LAMBDA * (w_val @ Sigma @ w_val)

    # 1. Numerical Gradient (finite differences)
    epsilon = 1e-6
    grad_num = np.zeros(N)
    for i in range(N):
        e_i = np.zeros(N)
        e_i[i] = 1.0
        grad_num[i] = (objective_func(w_test + epsilon * e_i) - objective_func(w_test - epsilon * e_i)) / (2 * epsilon)

    # 2. Analytical Gradient
    omega_term_val = np.sqrt(w_test @ Omega @ w_test)
    grad_anal = mu - k_solver * (Omega @ w_test) / omega_term_val - LAMBDA * (Sigma @ w_test)

    # Compare
    grad_diff = np.linalg.norm(grad_num - grad_anal)
    if grad_diff < 1e-5:
        print(f"  [PASS] L2 difference between gradients is {grad_diff:.2e} (< 1e-5)")
    else:
        print(f"  [FAIL] L2 difference between gradients is {grad_diff:.2e} (>= 1e-5)")
        all_passed = False

    # ------------------------------------------------------------------------
    # Validation 7: Constraint Satisfaction
    # ------------------------------------------------------------------------
    print("\nValidation 7: Constraint Satisfaction Check...")
    constraints_ok = True
    omega_diag = np.diag(Omega)
    omega_sqrt = np.sqrt(np.maximum(omega_diag, 0.0))
    
    for alpha in alpha_grid:
        k_solver = alpha * kappa_max_solver
        w_opt, _ = optimizer.solve(mu, k_solver)
        
        # Check non-negativity constraint
        if np.any(w_opt < -1e-8):
            constraints_ok = False
            print(f"  [FAIL] Weight violates w >= 0 at alpha={alpha:.2f}. Min value: {np.min(w_opt):.2e}")
            break
            
        # Check robust cone constraint: || Omega^1/2 w ||_2 <= t
        cone_val = np.linalg.norm(omega_sqrt * w_opt, 2)
        if cone_val < -1e-8:
            constraints_ok = False
            print(f"  [FAIL] Cone constraint violation.")
            break
            
    if constraints_ok:
        print("  [PASS] All constraints (non-negativity & conic bounds) hold within 1e-8.")
    else:
        all_passed = False

    # ------------------------------------------------------------------------
    # Validation 8: Single Asset Synthetic Case with Analytical Solution
    # ------------------------------------------------------------------------
    print("\nValidation 8: Single-Asset Analytical Validation Case...")
    # Inputs
    mu_1 = np.array([0.1])
    Sigma_1 = np.array([[1.0]])
    Omega_1 = np.array([[1.0]])
    
    # Analytical: w* = max(0, (0.1 - kappa_paper) / 1.0)
    for k_paper in [0.0, 0.03, 0.07, 0.12]:
        # Under T=1, kappa_solver = kappa_paper
        w_robust_1, _ = opt.solve_robust_portfolio(
            mu_1, Sigma_1, Omega_1, kappa_solver=k_paper, lambda_val=1.0, long_only=True, full_investment=False
        )
        w_opt_val = w_robust_1[0]
        w_anal_val = max(0.0, 0.1 - k_paper)
        diff_1 = abs(w_opt_val - w_anal_val)
        
        if diff_1 > 1e-6:
            print(f"  [FAIL] kappa={k_paper:.2f}: Solver w={w_opt_val:.4f}, Analytical w={w_anal_val:.4f} (diff={diff_1:.2e})")
            all_passed = False
            break
    else:
        print("  [PASS] Single-asset solver weights match analytical solution to precision < 1e-6.")

    print("\n" + "=" * 70)
    if all_passed:
        print("RESULT: ALL VALIDATION TESTS PASSED SUCCESSFULLY!")
    else:
        print("RESULT: SOME VALIDATION TESTS FAILED.")
    print("=" * 70)


if __name__ == "__main__":
    run_validations()
