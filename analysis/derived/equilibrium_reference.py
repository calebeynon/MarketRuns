"""
Purpose: Lookup helpers against the M&M (2020) Appendix D equilibrium
threshold table at `datastore/derived/equilibrium_thresholds.csv`. Callers
pre-round alpha (via compute_alpha); we only guard float noise (1e-6).
Author: Caleb Eynon (Claude Code, impl-equilibrium)
Date: 2026-04-25
"""

import numpy as np
import pandas as pd
from pathlib import Path

# =====
# File paths and constants
# =====
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EQUILIBRIUM_CSV = PROJECT_ROOT / "datastore" / "derived" / "equilibrium_thresholds.csv"

ALPHA_TOLERANCE = 1e-6
VALID_TREATMENTS = ("random", "average")
REQUIRED_COLUMNS = ("alpha", "treatment", "n", "threshold_pi", "avg_pi_at_sale")


# =====
# Main function
# =====
def main():
    """Smoke-test the lookup against a known row in the table."""
    eq_df = load_equilibrium_table()
    print(f"Loaded {len(eq_df):,} rows from {EQUILIBRIUM_CSV.name}")
    print("alpha=0.0, random, n=4:",
          lookup_equilibrium_reference(eq_df, 0.0, "random", 4))


# =====
# Loaders
# =====
def load_equilibrium_table(csv_path: Path = EQUILIBRIUM_CSV) -> pd.DataFrame:
    """Load equilibrium_thresholds.csv and validate its schema."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Equilibrium table missing: {csv_path}. "
            "Run analysis/analysis/simulate_equilibrium.py first."
        )
    eq_df = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS) - set(eq_df.columns)
    if missing:
        raise ValueError(
            f"Equilibrium table {csv_path} missing columns: {sorted(missing)}."
        )
    return eq_df


# =====
# Lookup
# =====
def lookup_equilibrium_reference(
    eq_df: pd.DataFrame, alpha: float, treatment: str, n_at_sale: int,
) -> tuple[float, float]:
    """Return (threshold_pi, avg_pi_at_sale) for the matching row.

    n=1 rows have NaN avg_pi_at_sale because n=1 never sells in equilibrium.
    """
    _validate_lookup_args(treatment, n_at_sale)
    row = _match_row(eq_df, alpha, treatment, n_at_sale)
    return float(row["threshold_pi"]), float(row["avg_pi_at_sale"])


def _validate_lookup_args(treatment: str, n_at_sale: int) -> None:
    """Reject malformed lookup keys before hitting the DataFrame."""
    if treatment not in VALID_TREATMENTS:
        raise ValueError(
            f"treatment must be one of {VALID_TREATMENTS}; got {treatment!r}"
        )
    if not 1 <= int(n_at_sale) <= 4:
        raise ValueError(f"n_at_sale must be in [1, 4]; got {n_at_sale!r}")


def _match_row(
    eq_df: pd.DataFrame, alpha: float, treatment: str, n_at_sale: int,
) -> pd.Series:
    """Find the unique row for (alpha, treatment, n_at_sale)."""
    mask = (
        np.isclose(eq_df["alpha"], float(alpha), atol=ALPHA_TOLERANCE)
        & (eq_df["treatment"] == treatment)
        & (eq_df["n"] == n_at_sale)
    )
    matches = eq_df[mask]
    if len(matches) != 1:
        raise KeyError(
            f"no equilibrium row for alpha={alpha}, treatment={treatment}, "
            f"n={n_at_sale} (matched {len(matches)} rows)."
        )
    return matches.iloc[0]


# %%
if __name__ == "__main__":
    main()
