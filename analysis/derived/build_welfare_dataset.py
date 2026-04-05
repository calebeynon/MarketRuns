"""
Purpose: Compute group-round welfare, cross-validate, and merge into emotions dataset
Author: Claude
Date: 2026-04-05

Reads group_round_timing.csv (with welfare column), cross-validates welfare
against actual player payoffs in individual_period_dataset_extended.csv,
then merges welfare into the emotions_traits_selling_dataset.csv.
"""

from pathlib import Path
import pandas as pd

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DERIVED = PROJECT_ROOT / "datastore" / "derived"
INPUT_GRT = DERIVED / "group_round_timing.csv"
INPUT_EXTENDED = DERIVED / "individual_period_dataset_extended.csv"
INPUT_EMOTIONS = DERIVED / "emotions_traits_selling_dataset.csv"
OUTPUT_WELFARE = DERIVED / "group_round_welfare.csv"
OUTPUT_EMOTIONS = DERIVED / "emotions_traits_selling_dataset.csv"

# CONSTANTS
MERGE_KEYS_GRT = ["session", "segment_num", "round_num", "group_id"]
MERGE_KEYS_EMOTIONS = ["session_id", "segment", "round", "group_id"]
MAX_GROUP_EARNINGS = 80


# =====
# Main function
# =====
# %%
def main():
    """Build welfare dataset and merge into emotions dataset."""
    grt_df = load_and_validate_grt()
    cross_validate_welfare(grt_df, pd.read_csv(INPUT_EXTENDED))

    welfare_df = grt_df[MERGE_KEYS_GRT + ["welfare"]].copy()
    save_welfare_dataset(welfare_df)

    emotions_df = pd.read_csv(INPUT_EMOTIONS)
    merged = merge_welfare_into_emotions(emotions_df, welfare_df)
    save_updated_emotions(merged)
    print("\nDone.")


# =====
# Validation
# =====
def load_and_validate_grt() -> pd.DataFrame:
    """Load group_round_timing.csv and verify welfare column exists."""
    print("Loading group_round_timing.csv...")
    grt_df = pd.read_csv(INPUT_GRT)
    if "welfare" not in grt_df.columns:
        raise ValueError(
            "welfare column missing from group_round_timing.csv. "
            "Run build_group_round_timing_dataset.py first."
        )
    print(f"  {len(grt_df)} rows, welfare column present")
    return grt_df


def cross_validate_welfare(grt_df: pd.DataFrame, extended_df: pd.DataFrame):
    """Validate welfare values against actual player payoffs.

    For state=1: actual_total_earnings must equal welfare * MAX_GROUP_EARNINGS.
    For state=0: welfare must equal 1.0 (no trade is optimal).
    """
    actual = compute_actual_group_earnings(extended_df)
    merged = merge_for_validation(grt_df, actual)
    validate_state_1(merged)
    validate_state_0(merged)
    print("  Cross-validation PASSED")


def compute_actual_group_earnings(extended_df: pd.DataFrame) -> pd.DataFrame:
    """Sum round_payoff by group-round from extended dataset."""
    dedup = extended_df.sort_values("period").drop_duplicates(
        subset=["session_id", "segment", "round", "group_id", "player"],
        keep="last",
    )
    actual = (
        dedup.groupby(MERGE_KEYS_EMOTIONS)["round_payoff"]
        .sum()
        .reset_index()
    )
    actual.rename(columns={"round_payoff": "actual_total"}, inplace=True)
    return actual


def merge_for_validation(
    grt_df: pd.DataFrame, actual_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge welfare with actual earnings for comparison."""
    welfare = grt_df[MERGE_KEYS_GRT + ["welfare", "state"]].copy()
    welfare.rename(
        columns={"session": "session_id", "segment_num": "segment",
                 "round_num": "round"},
        inplace=True,
    )
    merged = welfare.merge(actual_df, on=MERGE_KEYS_EMOTIONS, how="inner")
    if len(merged) != len(grt_df):
        raise AssertionError(
            f"Merge row count mismatch: {len(merged)} vs {len(grt_df)}. "
            "Check key alignment between datasets."
        )
    return merged


def validate_state_1(merged: pd.DataFrame):
    """For state=1, actual earnings must match welfare * MAX_GROUP_EARNINGS."""
    s1 = merged[merged["state"] == 1]
    expected = s1["welfare"] * MAX_GROUP_EARNINGS
    mismatches = s1[s1["actual_total"] != expected]
    if len(mismatches) > 0:
        raise AssertionError(
            f"State=1 welfare mismatch in {len(mismatches)} rows. "
            f"First mismatch: actual={mismatches.iloc[0]['actual_total']}, "
            f"expected={mismatches.iloc[0]['welfare'] * MAX_GROUP_EARNINGS}"
        )
    print(f"  State=1: {len(s1)} rows validated")


def validate_state_0(merged: pd.DataFrame):
    """For state=0, welfare must be 1.0 (no trade is optimal)."""
    s0 = merged[merged["state"] == 0]
    non_one = s0[s0["welfare"] != 1.0]
    if len(non_one) > 0:
        raise AssertionError(
            f"State=0 welfare != 1.0 in {len(non_one)} rows. "
            f"Values found: {sorted(non_one['welfare'].unique())}"
        )
    print(f"  State=0: {len(s0)} rows validated (all 1.0)")


# =====
# Merge welfare into emotions dataset
# =====
def merge_welfare_into_emotions(
    emotions_df: pd.DataFrame, welfare_df: pd.DataFrame
) -> pd.DataFrame:
    """Left-merge welfare onto emotions dataset."""
    initial_rows = len(emotions_df)
    emotions_df = drop_existing_welfare(emotions_df)
    cols_before_merge = len(emotions_df.columns)
    welfare_renamed = rename_welfare_keys(welfare_df)

    merged = emotions_df.merge(
        welfare_renamed, on=MERGE_KEYS_EMOTIONS, how="left",
    )
    validate_merge_result(merged, initial_rows, cols_before_merge)
    return merged


def drop_existing_welfare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop welfare column if present (idempotent re-runs)."""
    if "welfare" in df.columns:
        df = df.drop(columns=["welfare"])
        print("  Dropped existing welfare column (idempotent re-run)")
    return df


def rename_welfare_keys(welfare_df: pd.DataFrame) -> pd.DataFrame:
    """Rename GRT column names to match emotions dataset keys."""
    return welfare_df.rename(
        columns={"session": "session_id", "segment_num": "segment",
                 "round_num": "round"},
    )


def validate_merge_result(
    merged: pd.DataFrame, expected_rows: int, original_cols: int
):
    """Assert merge did not change row count and welfare has no NaNs."""
    if len(merged) != expected_rows:
        raise AssertionError(
            f"Row count changed: {len(merged)} vs {expected_rows}. "
            "Likely duplicate keys in welfare dataset."
        )
    nan_count = merged["welfare"].isna().sum()
    if nan_count > 0:
        raise AssertionError(
            f"welfare has {nan_count} NaN values after merge."
        )
    if len(merged.columns) != original_cols + 1:
        raise AssertionError(
            f"Expected {original_cols + 1} columns, got {len(merged.columns)}."
        )
    print(f"  Merge OK: {len(merged)} rows, {len(merged.columns)} columns")


# =====
# Save functions
# =====
def save_welfare_dataset(df: pd.DataFrame):
    """Save standalone welfare dataset."""
    OUTPUT_WELFARE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_WELFARE, index=False)
    print(f"  Saved {len(df)} rows to {OUTPUT_WELFARE}")


def save_updated_emotions(df: pd.DataFrame):
    """Save updated emotions dataset with welfare column."""
    df.to_csv(OUTPUT_EMOTIONS, index=False)
    print(f"  Saved {len(df)} rows, {len(df.columns)} cols to {OUTPUT_EMOTIONS}")


# %%
if __name__ == "__main__":
    main()
