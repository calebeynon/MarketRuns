"""
Purpose: Build welfare timing deviation dataset for issue #116.

For each voluntary seller-round, compute pi at sale, merge the Magnani & Munro
(2020) equilibrium sell-timing thresholds at two alpha values (0.0 and 0.5),
merge the group-round welfare outcome, and emit a long-format dataset where
each seller contributes one row per alpha.

The resulting dataset powers an R regression of group welfare on the seller's
deviation from the equilibrium threshold (pi_at_sale - threshold_pi).

Author: Claude Code
Date: 2026-04-19
"""

from pathlib import Path
import pandas as pd

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
PERIOD_DATA_PATH = DATASTORE / "derived" / "individual_period_dataset_extended.csv"
EQUILIBRIUM_PATH = DATASTORE / "derived" / "equilibrium_thresholds.csv"
WELFARE_PATH = DATASTORE / "derived" / "group_round_welfare.csv"
OUTPUT_PATH = DATASTORE / "derived" / "welfare_timing_deviation.csv"

# tr1 uses the random-draw liquidation rule, tr2 uses the average rule
TREATMENT_TO_EQ = {"tr1": "random", "tr2": "average"}
ALPHAS = [0.0, 0.5]
VOLUNTARY_POSITIONS = {1, 2, 3}
KEY_COLS = [
    "session_id", "segment", "round", "player", "alpha",
    "n", "pi_at_sale", "threshold_pi", "pi_deviation", "welfare", "state",
]


# =====
# Main function
# =====
def main():
    """Build the welfare-timing-deviation dataset."""
    periods = load_periods()
    eq = load_equilibrium()
    welfare = load_welfare()

    sales = extract_sale_rows(periods)
    voluntary = filter_voluntary(sales)
    long_df = expand_by_alpha(voluntary, eq)
    long_df = merge_group_welfare(long_df, welfare)
    long_df = add_deviation_columns(long_df)

    validate(long_df)
    save(long_df)
    return long_df


# =====
# Data loading
# =====
def load_periods() -> pd.DataFrame:
    """Load the extended period-level dataset."""
    print(f"Loading periods from: {PERIOD_DATA_PATH}")
    df = pd.read_csv(PERIOD_DATA_PATH)
    print(f"  loaded {len(df):,} period-rows")
    return df


def load_equilibrium() -> pd.DataFrame:
    """Load equilibrium thresholds for the two alphas of interest."""
    print(f"Loading equilibrium thresholds from: {EQUILIBRIUM_PATH}")
    eq = pd.read_csv(EQUILIBRIUM_PATH)
    eq = eq[eq["alpha"].isin(ALPHAS)].copy()
    eq = eq[["alpha", "treatment", "n", "threshold_pi"]]
    print(f"  {len(eq)} threshold rows for alphas {ALPHAS}")
    return eq


def load_welfare() -> pd.DataFrame:
    """Load group-round welfare and rename keys to match seller dataset."""
    print(f"Loading group-round welfare from: {WELFARE_PATH}")
    welfare = pd.read_csv(WELFARE_PATH)
    welfare = welfare.rename(columns={
        "session": "session_id", "segment_num": "segment", "round_num": "round",
    })
    print(f"  {len(welfare):,} group-round welfare rows")
    return welfare


def merge_group_welfare(long_df: pd.DataFrame, welfare: pd.DataFrame) -> pd.DataFrame:
    """Merge group-round welfare onto each seller-row."""
    merged = long_df.merge(
        welfare, on=["session_id", "segment", "round", "group_id"], how="left",
    )
    if merged["welfare"].isna().any():
        missing = merged[merged["welfare"].isna()][
            ["session_id", "segment", "round", "group_id"]
        ].drop_duplicates()
        raise ValueError(f"Missing welfare for group-rounds:\n{missing}")
    return merged


# =====
# Sale-row extraction
# =====
def extract_sale_rows(periods: pd.DataFrame) -> pd.DataFrame:
    """Pick the period row where the player actually sold (state==1 rounds)."""
    mask = (periods["state"] == 1) & (periods["sold_in_round"] >= 1) & (periods["sold"] == 1)
    sales = periods.loc[mask].copy()

    # A player should have exactly one sold==1 row per (session, segment, round)
    dup_keys = ["session_id", "segment", "round", "player"]
    if sales.duplicated(dup_keys).any():
        raise ValueError(
            "Multiple sold==1 rows for a single player-round in period data; "
            "check upstream builder for duplicates."
        )

    sales["n"] = sales["prior_group_sales"].astype(int) + 1
    sales = sales.rename(columns={"signal": "pi_at_sale"})
    print(f"  extracted {len(sales):,} sale-rows (state==1, sold==1)")
    return sales


def filter_voluntary(sales: pd.DataFrame) -> pd.DataFrame:
    """Keep only voluntary sellers (sale position n in {1, 2, 3})."""
    before = len(sales)
    voluntary = sales[sales["n"].isin(VOLUNTARY_POSITIONS)].copy()
    dropped = before - len(voluntary)
    print(f"  voluntary filter: {before:,} -> {len(voluntary):,} ({dropped:,} n=4 dropped)")
    return voluntary


# =====
# Equilibrium merge
# =====
def expand_by_alpha(voluntary: pd.DataFrame, eq: pd.DataFrame) -> pd.DataFrame:
    """Cross each sale with both alpha values and merge in threshold_pi."""
    voluntary = voluntary.copy()
    voluntary["eq_treatment"] = voluntary["treatment"].map(TREATMENT_TO_EQ)
    if voluntary["eq_treatment"].isna().any():
        raise ValueError("Unmapped treatment value encountered; expected tr1 or tr2.")

    alpha_df = pd.DataFrame({"alpha": ALPHAS, "_key": 1})
    voluntary["_key"] = 1
    long_df = voluntary.merge(alpha_df, on="_key").drop(columns="_key")

    merged = merge_thresholds(long_df, eq)
    merged = merged.drop(columns=["treatment_eq", "eq_treatment"])
    print(f"  expanded to long format: {len(merged):,} rows (2x voluntary sellers)")
    return merged


def merge_thresholds(long_df: pd.DataFrame, eq: pd.DataFrame) -> pd.DataFrame:
    """Merge equilibrium threshold_pi onto (alpha, eq_treatment, n)."""
    merged = long_df.merge(
        eq,
        left_on=["alpha", "eq_treatment", "n"],
        right_on=["alpha", "treatment", "n"],
        how="left",
        suffixes=("", "_eq"),
    )
    if merged["threshold_pi"].isna().any():
        missing = merged[merged["threshold_pi"].isna()][
            ["alpha", "eq_treatment", "n"]
        ].drop_duplicates()
        raise ValueError(f"Missing equilibrium thresholds for:\n{missing}")
    return merged


# =====
# Deviation columns
# =====
def add_deviation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add pi_deviation, pi_dev_neg, pi_dev_pos, global_group_id."""
    df = df.copy()
    df["pi_deviation"] = df["pi_at_sale"] - df["threshold_pi"]
    df["pi_dev_neg"] = df["pi_deviation"].clip(upper=0.0)
    df["pi_dev_pos"] = df["pi_deviation"].clip(lower=0.0)
    df["global_group_id"] = (
        df["session_id"] + "_" + df["segment"].astype(str) + "_" + df["group_id"].astype(str)
    )
    output_cols = [
        "session_id", "segment", "round", "player", "group_id", "global_group_id",
        "treatment", "alpha", "n", "pi_at_sale", "threshold_pi",
        "pi_deviation", "pi_dev_neg", "pi_dev_pos", "welfare", "state",
    ]
    return df[output_cols]


# =====
# Validation
# =====
def validate(df: pd.DataFrame):
    """Assert no missing values in key columns and print summary stats."""
    for col in KEY_COLS + ["group_id", "treatment", "pi_dev_neg", "pi_dev_pos"]:
        if df[col].isna().any():
            raise ValueError(f"Missing values in required column '{col}'.")

    print("\n" + "=" * 60)
    print("WELFARE TIMING DEVIATION SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(df):,}")
    print(f"Unique seller-rounds: {len(df) // len(ALPHAS):,}")
    print("\nRows per alpha:")
    print(df["alpha"].value_counts().sort_index().to_string())
    print("\npi_deviation describe():")
    print(df["pi_deviation"].describe().to_string())
    print("\nHead(10):")
    print(df.head(10).to_string())


# =====
# Output
# =====
def save(df: pd.DataFrame):
    """Save the dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows to: {OUTPUT_PATH}")


# %%
if __name__ == "__main__":
    main()
