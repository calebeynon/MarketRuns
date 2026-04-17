"""
Purpose: Robustness checks for the equilibrium solver and simulation.
         Tests sensitivity to grid resolution (t_max) and random seed.
Author: Claude
Date: 2026-04-08
"""

import numpy as np

from analysis.analysis.equilibrium_model import (
    MU_B,
    MU_G,
    N_INVESTORS,
    LAMBDA,
    PI_0,
    find_continuous_threshold,
    solve_equilibrium,
)
from analysis.analysis.simulate_equilibrium import (
    _simulate_one_game,
)

# PARAMETERS
ALPHA_CASES = [0.0, 0.5]
TREATMENT = "random"
T_MAX_VALUES = [10, 20, 40, 80, 160]
SEEDS = list(range(1, 51))
N_SIMULATIONS = 10_000

# M&M TARGETS: (alpha, n, d)
MM_TARGETS = [
    (0.0, 4, 2),
    (0.0, 3, 2),
    (0.5, 4, 1),
    (0.5, 3, 2),
    (0.5, 2, 3),
]


# =====
# Main function
# =====
def main():
    """Run grid resolution and seed robustness checks."""
    print("=" * 70)
    print("ROBUSTNESS CHECK 1: Grid resolution (t_max)")
    print("=" * 70)
    run_grid_sensitivity()

    print("\n" + "=" * 70)
    print("ROBUSTNESS CHECK 2: Random seed sensitivity")
    print("=" * 70)
    run_seed_sensitivity()


# =====
# Grid resolution sensitivity
# =====
def run_grid_sensitivity():
    """Test sigma and threshold stability across grid resolutions."""
    _print_sigma_sensitivity()
    _print_threshold_sensitivity()


def _net_bad_to_belief(d):
    """Convert net bad signal count to P(Good)."""
    return 1.0 / (1.0 + (MU_B / MU_G) ** d)


def _build_solver_cache():
    """Run solver for all (alpha, t_max) combos and cache results."""
    cache = {}
    for alpha in ALPHA_CASES:
        for t_max in T_MAX_VALUES:
            cache[(alpha, t_max)] = solve_equilibrium(
                alpha=alpha, treatment=TREATMENT, t_max=t_max
            )
    return cache


def _sigma_row(alpha, n, d, cache):
    """Format one sigma sensitivity row for a given M&M target."""
    pi = _net_bad_to_belief(d)
    sigmas = [
        float(np.interp(pi, cache[(alpha, t)]["belief_grid"], cache[(alpha, t)]["sigma"][n]))
        for t in T_MAX_VALUES
    ]
    max_delta = max(abs(s - sigmas[-1]) for s in sigmas[:-1])
    row = f"{'a=' + str(alpha) + ', n=' + str(n) + ', d=' + str(d):<22}"
    return row + "".join(f"{s:>10.4f}" for s in sigmas) + f"  max|Δ|={max_delta:.5f}"


def _print_sigma_sensitivity():
    """Print sigma at M&M target points across t_max values."""
    print("\nSigma at M&M target points:")
    header = f"{'Target':<22}" + "".join(f"{'t=' + str(t):>10}" for t in T_MAX_VALUES)
    print(header)
    print("-" * (22 + 10 * len(T_MAX_VALUES)))
    cache = _build_solver_cache()
    for alpha, n, d in MM_TARGETS:
        print(_sigma_row(alpha, n, d, cache))


def _threshold_row(alpha, n):
    """Format one threshold sensitivity row for a given (alpha, n) pair."""
    thresholds = []
    for t_max in T_MAX_VALUES:
        result = solve_equilibrium(alpha=alpha, treatment=TREATMENT, t_max=t_max)
        thresholds.append(
            find_continuous_threshold(n, alpha, TREATMENT, result["v_table"], result["belief_grid"])
        )
    max_delta = max(abs(t - thresholds[-1]) for t in thresholds[:-1])
    row = f"{'a=' + str(alpha) + ', n=' + str(n):<22}"
    return row + "".join(f"{t:>10.5f}" for t in thresholds) + f"  max|Δ|={max_delta:.6f}"


def _print_threshold_sensitivity():
    """Print continuous thresholds across t_max values."""
    print("\nContinuous thresholds pi*(n):")
    header = f"{'(alpha, n)':<22}" + "".join(f"{'t=' + str(t):>10}" for t in T_MAX_VALUES)
    print(header)
    print("-" * (22 + 10 * len(T_MAX_VALUES)))
    for alpha in ALPHA_CASES:
        for n in range(2, N_INVESTORS + 1):
            print(_threshold_row(alpha, n))


# =====
# Random seed sensitivity
# =====
def run_seed_sensitivity():
    """Test simulation stability across random seeds."""
    for alpha in ALPHA_CASES:
        _run_seed_check(alpha)


def _run_seed_check(alpha):
    """Run simulations with multiple seeds and report statistics."""
    print(f"\nalpha={alpha}, treatment={TREATMENT}, {N_SIMULATIONS} games/seed, "
          f"{len(SEEDS)} seeds")
    result = solve_equilibrium(alpha=alpha, treatment=TREATMENT, t_max=40)
    grid = result["belief_grid"]
    sigma_table = result["sigma"]

    # Collect avg P(Bad) at each sale position across seeds
    all_pbads = {k: [] for k in range(1, N_INVESTORS)}
    for seed in SEEDS:
        pbads = _simulate_with_seed(seed, grid, sigma_table)
        for k, pb in pbads.items():
            all_pbads[k].append(pb)

    _print_seed_results(all_pbads)


def _simulate_with_seed(seed, grid, sigma_table):
    """Run N_SIMULATIONS games with a given seed, return avg P(Bad) per position."""
    rng = np.random.default_rng(seed)
    sale_pis = {k: [] for k in range(1, N_INVESTORS)}
    for _ in range(N_SIMULATIONS):
        sales = _simulate_one_game(rng, grid, sigma_table)
        for seller_pos, pi in sales:
            sale_pis[seller_pos].append(pi)
    pbads = {}
    for k in range(1, N_INVESTORS):
        if sale_pis[k]:
            pbads[k] = 1 - np.mean(sale_pis[k])
        else:
            pbads[k] = np.nan
    return pbads


def _print_seed_results(all_pbads):
    """Print summary statistics across seeds."""
    print(f"{'Seller':<10} {'Mean P(Bad)':>12} {'SD':>10} {'Min':>10} "
          f"{'Max':>10} {'Range':>10}")
    print("-" * 62)
    for k in sorted(all_pbads.keys()):
        vals = np.array(all_pbads[k])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        mean = np.mean(vals)
        sd = np.std(vals, ddof=1)
        lo, hi = np.min(vals), np.max(vals)
        print(f"  {k}st sale" if k == 1 else f"  {k}nd sale" if k == 2
              else f"  {k}rd sale",
              f"  {mean:>10.4f} {sd:>10.4f} {lo:>10.4f} {hi:>10.4f} "
              f"{hi - lo:>10.4f}")


# %%
if __name__ == "__main__":
    main()
