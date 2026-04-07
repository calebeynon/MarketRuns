"""
Purpose: Simulate equilibrium outcomes using the Magnani & Munro (2020) model.
         Produces threshold and simulation CSV for downstream tables/plots.
Author: Claude
Date: 2026-04-05
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.analysis.equilibrium_model import (
    LAMBDA,
    MU_B,
    MU_G,
    N_INVESTORS,
    PI_0,
    find_continuous_threshold,
    solve_equilibrium,
)

# FILE PATHS
OUTPUT_CSV = Path("datastore/derived/equilibrium_thresholds.csv")

# SIMULATION PARAMETERS
ALPHA_VALUES = [round(a * 0.1, 1) for a in range(10)]  # 0.0 to 0.9
TREATMENTS = ["random", "average"]
N_SIMULATIONS = 10_000
T_MAX = 20  # belief grid depth (606 points, sufficient resolution)
SEED = 42


# =====
# Main function
# =====
def main():
    """Solve equilibrium and simulate for all (alpha, treatment) pairs."""
    rows = []
    for treatment in TREATMENTS:
        for alpha in ALPHA_VALUES:
            print(f"Solving alpha={alpha}, treatment={treatment}...", flush=True)
            result = solve_equilibrium(alpha=alpha, treatment=treatment, t_max=T_MAX)
            threshold_rows = _extract_thresholds(alpha, treatment, result)
            sim_averages = _run_simulations(alpha, treatment, result)
            rows.extend(_merge_threshold_sim(threshold_rows, sim_averages))
    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")
    _print_validation(df)


# =====
# Threshold extraction
# =====
def _extract_thresholds(alpha, treatment, result):
    """Extract continuous pi* via root-finding for each n."""
    grid = result["belief_grid"]
    v_table = result["v_table"]
    rows = []
    for n in range(1, N_INVESTORS + 1):
        threshold = find_continuous_threshold(
            n, alpha, treatment, v_table, grid)
        rows.append({"alpha": alpha, "treatment": treatment, "n": n,
                      "threshold_pi": threshold})
    return rows


# =====
# Simulation engine
# =====
def _run_simulations(alpha, treatment, result):
    """Run N_SIMULATIONS games, record pi at each sale position."""
    rng = np.random.default_rng(SEED)
    grid = result["belief_grid"]
    sigma_table = result["sigma"]
    # Collect beliefs by seller position (1st, 2nd, 3rd)
    sale_pis = {k: [] for k in range(1, N_INVESTORS)}
    for _ in range(N_SIMULATIONS):
        game_sales = _simulate_one_game(rng, grid, sigma_table)
        for seller_pos, pi in game_sales:
            sale_pis[seller_pos].append(pi)
    averages = {}
    for k in range(1, N_INVESTORS):
        avg = np.mean(sale_pis[k]) if sale_pis[k] else np.nan
        averages[k] = {"avg_pi": avg, "n_obs": len(sale_pis[k])}
    return averages


def _simulate_one_game(rng, grid, sigma_table):
    """Simulate one game, return list of (seller_position, pi) tuples."""
    n = N_INVESTORS
    pi = PI_0
    true_state = rng.choice([0, 1])  # 0=Bad, 1=Good
    seller_count = 0
    sales = []
    for _ in range(200):  # safety cap on periods
        pi = _draw_signal_and_update(rng, pi, true_state)
        sig = _lookup_sigma(pi, n, grid, sigma_table)
        n_sellers = _count_sellers(rng, n, sig)
        if n_sellers > 0:
            for _ in range(n_sellers):
                seller_count += 1
                if seller_count < N_INVESTORS:  # 4th holder never sells
                    sales.append((seller_count, pi))
            n -= n_sellers
        if n <= 1:
            break
        if rng.random() < LAMBDA:
            break
    return sales


def _draw_signal_and_update(rng, pi, true_state):
    """Draw a signal from the true state and Bayesian-update pi."""
    if true_state == 0:
        signal_bad = rng.random() < MU_B
    else:
        signal_bad = rng.random() < MU_G
    if signal_bad:
        num = pi * (1 - MU_B)
        return num / (num + (1 - pi) * MU_B)
    num = pi * (1 - MU_G)
    return num / (num + (1 - pi) * MU_G)


def _lookup_sigma(pi, n, grid, sigma_table):
    """Interpolate sigma for given belief and number of holders."""
    return float(np.interp(pi, grid, sigma_table[n]))


def _count_sellers(rng, n, sigma):
    """Count how many of n holders sell given probability sigma."""
    if sigma <= 0:
        return 0
    if sigma >= 1:
        return n
    return int(rng.binomial(n, sigma))


# =====
# Output merging and validation
# =====
def _merge_threshold_sim(threshold_rows, sim_averages):
    """Merge threshold rows with per-position simulation averages."""
    merged = []
    for row in threshold_rows:
        n = row["n"]
        # Seller position k corresponds to selling from n = N - k + 1 holders
        # k=1 sells from n=4, k=2 from n=3, k=3 from n=2; n=1 never sells
        seller_pos = N_INVESTORS - n + 1
        if seller_pos in sim_averages:
            sim = sim_averages[seller_pos]
            merged.append({**row, "avg_pi_at_sale": sim["avg_pi"],
                           "n_obs": sim["n_obs"],
                           "n_simulations": N_SIMULATIONS})
        else:
            merged.append({**row, "avg_pi_at_sale": np.nan,
                           "n_obs": 0, "n_simulations": N_SIMULATIONS})
    return merged


def _print_validation(df):
    """Print validation checks against known results."""
    rn = df[(df["alpha"] == 0.0) & (df["treatment"] == "random")]
    av = df[(df["alpha"] == 0.0) & (df["treatment"] == "average")]
    print("\nValidation: alpha=0 thresholds should match across treatments")
    for n in range(1, N_INVESTORS + 1):
        t_r = rn[rn["n"] == n]["threshold_pi"].values[0]
        t_a = av[av["n"] == n]["threshold_pi"].values[0]
        match = "OK" if abs(t_r - t_a) < 0.01 else "MISMATCH"
        print(f"  n={n}: random={t_r:.4f}, average={t_a:.4f} [{match}]")
    # Validate against M&M Table 2 (HIGH NoCB)
    print("\nValidation: avg P(Bad) at 1st sale vs M&M Table 2")
    for alpha, mm_val in [(0.0, 0.800), (0.5, 0.678)]:
        row = df[(df["alpha"] == alpha) & (df["treatment"] == "random")
                 & (df["n"] == 4)]
        if len(row) > 0:
            our_pbad = 1 - row["avg_pi_at_sale"].values[0]
            match = "OK" if abs(our_pbad - mm_val) < 0.02 else "CHECK"
            print(f"  alpha={alpha}: ours={our_pbad:.3f}, M&M={mm_val:.3f} [{match}]")


# %%
if __name__ == "__main__":
    main()
