"""
Purpose: Export full equilibrium selling probability sigma(n, belief) across
         all (alpha, treatment) cells to a single long-format CSV.
Author: Claude
Date: 2026-04-16
"""

from pathlib import Path

import pandas as pd

from analysis.analysis.equilibrium_model import N_INVESTORS, solve_equilibrium

# FILE PATHS
OUTPUT_CSV = Path(__file__).resolve().parents[1] / "output" / "full_sigmas.csv"

# SIMULATION PARAMETERS
ALPHA_VALUES = [round(a * 0.1, 1) for a in range(10)]  # 0.0 to 0.9
TREATMENTS = ["random", "average"]
T_MAX = 20  # belief grid depth (must match simulate_equilibrium.py)


# =====
# Main function
# =====
def main():
    """Solve equilibrium for every (alpha, treatment) and export sigma table."""
    rows = []
    for treatment in TREATMENTS:
        for alpha in ALPHA_VALUES:
            print(f"Solving alpha={alpha}, treatment={treatment}...",
                  flush=True)
            result = solve_equilibrium(
                alpha=alpha, treatment=treatment, t_max=T_MAX)
            rows.extend(_flatten_sigma(alpha, treatment, result))
    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")


# =====
# Extraction helpers
# =====
def _flatten_sigma(alpha, treatment, result):
    """Flatten sigma[n][i] to long-format rows over belief grid."""
    grid = result["belief_grid"]
    sigma_table = result["sigma"]
    rows = []
    for n in range(1, N_INVESTORS + 1):
        rows.extend(_rows_for_n(alpha, treatment, n, grid, sigma_table[n]))
    return rows


def _rows_for_n(alpha, treatment, n, grid, sigma_row):
    """Build one row per belief grid point for a given n."""
    return [
        {
            "alpha": alpha,
            "treatment": treatment,
            "n": n,
            "belief_p_good": float(belief),
            "sigma": float(sigma),
        }
        for belief, sigma in zip(grid, sigma_row)
    ]


# %%
if __name__ == "__main__":
    main()
