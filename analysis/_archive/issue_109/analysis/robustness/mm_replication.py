"""
Purpose: Diagnostic script comparing our Bellman solver against Magnani & Munro
         (2020) Appendix D sigma values at 5 specific grid points. Tests
         alternative solver approaches to identify the source of 2-5% gaps.
Author: Claude
Date: 2026-04-07
"""

import sys
from pathlib import Path

import numpy as np

# Archived: ensure project root and local archive dir are on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.analysis.equilibrium_model import (  # noqa: E402
    MU_B,
    MU_G,
    solve_equilibrium,
)
from mm_alternative_solvers import (  # noqa: E402
    solve_discount_variant,
    solve_howard,
    solve_policy_iteration,
    solve_w_at_indifference,
)

# M&M APPENDIX D TARGET VALUES
# Each tuple: (alpha, n, d, mm_sigma)
# d = net bad signal count; pi = P(Good) = 1/(1 + (mu_B/mu_G)^d)
MM_TARGETS = [
    (0.0, 4, 2, 0.775),
    (0.0, 3, 2, 0.0155),
    (0.5, 4, 1, 0.5155),
    (0.5, 3, 2, 0.8868),
    (0.5, 2, 3, 0.86),
]

# SOLVER GRID DEPTH
T_MAX = 40


# =====
# Main function
# =====
def main():
    """Run all solver variants and compare against M&M targets."""
    targets = _build_target_table()
    _print_header(targets)
    methods = _get_solver_methods()
    results = {}
    for name, solver_fn in methods:
        print(f"\nRunning: {name}...")
        sigmas = _run_solver_at_targets(solver_fn, targets)
        results[name] = sigmas
        _print_method_row(name, sigmas, targets)
    _print_comparison_table(results, targets)
    _print_diagnosis(results, targets)


# =====
# Target table construction
# =====
def _net_bad_to_belief(d):
    """Convert net bad signal count to P(Good) belief."""
    return 1.0 / (1.0 + (MU_B / MU_G) ** d)


def _build_target_table():
    """Build target table with computed pi values."""
    targets = []
    for alpha, n, d, mm_sigma in MM_TARGETS:
        pi = _net_bad_to_belief(d)
        targets.append({
            "alpha": alpha,
            "n": n,
            "d": d,
            "pi": pi,
            "p_bad": 1 - pi,
            "mm_sigma": mm_sigma,
        })
    return targets


# =====
# Solver methods
# =====
def _get_solver_methods():
    """Return list of (name, solver_function) pairs."""
    return [
        ("Value iteration (baseline)", _solve_baseline),
        ("Value iteration (high init)", _solve_high_init),
        ("Policy iteration", solve_policy_iteration),
        ("Howard (10 sweeps)", lambda a, t: solve_howard(a, t, 10)),
        ("Howard (50 sweeps)", lambda a, t: solve_howard(a, t, 50)),
        ("Discount variant", solve_discount_variant),
        ("W=U_sell at indifference", solve_w_at_indifference),
    ]


def _solve_baseline(alpha, treatment):
    """Baseline solver with default low initialization."""
    return solve_equilibrium(alpha, treatment, t_max=T_MAX)


def _solve_high_init(alpha, treatment):
    """Baseline solver with high V initialization."""
    return solve_equilibrium(alpha, treatment, t_max=T_MAX, v_init="high")


# =====
# Solver execution
# =====
def _run_solver_at_targets(solver_fn, targets):
    """Run solver for each unique (alpha, treatment) and extract sigmas."""
    cache = {}
    sigmas = []
    for t in targets:
        key = (t["alpha"], "random")
        if key not in cache:
            cache[key] = solver_fn(t["alpha"], "random")
        result = cache[key]
        sigma = _lookup_sigma(result, t["n"], t["pi"])
        sigmas.append(sigma)
    return sigmas


def _lookup_sigma(result, n, pi):
    """Interpolate sigma from result at given (n, pi)."""
    grid = result["belief_grid"]
    sigma_vec = result["sigma"][n]
    return float(np.interp(pi, grid, sigma_vec))


# =====
# Output formatting
# =====
def _print_header(targets):
    """Print the M&M target reference table."""
    print("=" * 70)
    print("M&M Appendix D Target Values")
    print("=" * 70)
    print(f"{'alpha':>6} {'n':>3} {'d':>3} {'P(Bad)':>8} {'M&M sigma':>10}")
    print("-" * 35)
    for t in targets:
        print(
            f"{t['alpha']:>6.1f} {t['n']:>3d} {t['d']:>3d} "
            f"{t['p_bad']:>8.4f} {t['mm_sigma']:>10.4f}"
        )


def _print_method_row(name, sigmas, targets):
    """Print sigma values for one method."""
    gaps = [s - t["mm_sigma"] for s, t in zip(sigmas, targets)]
    max_gap = max(abs(g) for g in gaps)
    print(f"  Max |gap|: {max_gap:.4f}")


def _print_comparison_table(results, targets):
    """Print full comparison table across all methods."""
    print("\n" + "=" * 90)
    print("COMPARISON TABLE: sigma at each target point")
    print("=" * 90)
    _print_table_header(targets)
    _print_table_reference(targets)
    _print_table_methods(results)
    _print_table_gaps(results, targets)


def _print_table_header(targets):
    """Print column headers for the comparison table."""
    header = f"{'Method':<30}"
    for t in targets:
        header += f" a={t['alpha']},n={t['n']},d={t['d']:>5}"
    print(header)
    print("-" * 90)


def _print_table_reference(targets):
    """Print M&M reference row."""
    row = f"{'M&M Appendix D':<30}"
    for t in targets:
        row += f" {t['mm_sigma']:>13.4f}"
    print(row)
    print("-" * 90)


def _print_table_methods(results):
    """Print sigma values for each solver method."""
    for name, sigmas in results.items():
        row = f"{name:<30}"
        for s in sigmas:
            row += f" {s:>13.4f}"
        print(row)


def _print_table_gaps(results, targets):
    """Print gap rows (method - M&M) for each solver."""
    print("-" * 90)
    print("GAPS (method - M&M):")
    for name, sigmas in results.items():
        row = f"{name:<30}"
        for s, t in zip(sigmas, targets):
            gap = s - t["mm_sigma"]
            row += f" {gap:>+13.4f}"
        print(row)


def _print_diagnosis(results, targets):
    """Print diagnostic summary identifying which variant is closest."""
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    best_name, best_rmse = _rank_methods(results, targets)
    print(f"\n  Closest to M&M: {best_name} (RMSE={best_rmse:.5f})")
    _check_systematic_bias(results, targets)


def _rank_methods(results, targets):
    """Rank methods by RMSE and print each. Returns (best_name, best_rmse)."""
    best_name, best_rmse = None, float("inf")
    for name, sigmas in results.items():
        gaps = [s - t["mm_sigma"] for s, t in zip(sigmas, targets)]
        rmse = np.sqrt(np.mean(np.array(gaps) ** 2))
        max_gap = max(abs(g) for g in gaps)
        print(f"  {name:<30} RMSE={rmse:.5f}  Max|gap|={max_gap:.4f}")
        if rmse < best_rmse:
            best_rmse, best_name = rmse, name
    return best_name, best_rmse


def _check_systematic_bias(results, targets):
    """Check whether baseline gaps are all positive (systematic bias)."""
    baseline_sigmas = results["Value iteration (baseline)"]
    gaps = [s - t["mm_sigma"] for s, t in zip(baseline_sigmas, targets)]
    all_positive = all(g > 0 for g in gaps)
    print(f"  All baseline gaps positive: {all_positive}")
    if all_positive:
        print("  -> Systematic upward bias suggests a convention difference")


# %%
if __name__ == "__main__":
    main()
