"""
Purpose: Build analysis-ready dataset for emotion spikes regression
Author: Claude Code
Date: 2026-02-02

Merges three data sources and computes sale_prev_period variable:
  1. individual_period_dataset.csv — selling decisions (player-period level)
  2. imotions_period_emotions_extended.csv — facial emotions with max/p95
  3. survey_traits.csv — personality traits (player level)

KEY COMPUTED VARIABLES:
  - sale_prev_period: Binary, 1 if ANY group member sold in period t-1
  - n_sales_earlier: Count of sales in periods 1 to t-2
  - player_id: Unique player identifier (session_id_player)
  - global_group_id: Unique group identifier (session_id_segment_group_id)
  - group_round_id: Group-round identifier for period-level grouping

FILTERS:
  - Excludes already_sold == 1 (players who already sold in this round)
  - Excludes rows with missing emotion data (documented in output)
"""

import pandas as pd
from pathlib import Path

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
DERIVED = DATASTORE / "derived"

INPUT_PERIOD = DERIVED / "individual_period_dataset.csv"
INPUT_EMOTIONS = DERIVED / "imotions_period_emotions_extended.csv"
INPUT_TRAITS = DERIVED / "survey_traits.csv"
OUTPUT_PATH = DERIVED / "emotion_spikes_analysis_dataset.csv"


# =====
# Main function
# =====
def main():
    """Build the emotion spikes analysis dataset."""
    print("Loading datasets...")
    period_df = load_period_data()
    emotions_df = load_emotions_data()
    traits_df = load_traits_data()

    print("\nMerging datasets...")
    merged = merge_datasets(period_df, emotions_df, traits_df)

    print("\nComputing sale timing variables...")
    merged = compute_sale_timing_vars(merged)

    print("\nAdding identifier variables...")
    merged = add_identifiers(merged)

    print("\nFiltering data...")
    filtered = filter_dataset(merged)

    print_summary(filtered, merged)
    save_dataset(filtered)

    return filtered


# =====
# Data loading
# =====
def load_period_data() -> pd.DataFrame:
    """Load individual period dataset."""
    df = pd.read_csv(INPUT_PERIOD)
    print(f"  Period data: {len(df)} rows")
    return df


def load_emotions_data() -> pd.DataFrame:
    """Load extended emotions dataset."""
    df = pd.read_csv(INPUT_EMOTIONS)
    print(f"  Emotions data: {len(df)} rows")
    return df


def load_traits_data() -> pd.DataFrame:
    """Load survey traits dataset."""
    df = pd.read_csv(INPUT_TRAITS)
    print(f"  Traits data: {len(df)} rows")
    return df


# =====
# Merge logic
# =====
def merge_datasets(
    period_df: pd.DataFrame,
    emotions_df: pd.DataFrame,
    traits_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join emotions and traits onto period dataset."""
    # Define columns to merge
    emotion_cols = [
        "session_id", "segment", "round", "period", "player",
        "fear_mean", "fear_max", "fear_p95",
        "anger_mean", "anger_max", "anger_p95",
        "n_frames",
    ]

    trait_cols = [
        "session_id", "player",
        "neuroticism", "impulsivity", "state_anxiety",
    ]

    # Merge emotions (period-level)
    merged = period_df.merge(
        emotions_df[emotion_cols],
        on=["session_id", "segment", "round", "period", "player"],
        how="left",
    )
    print(f"  After emotion merge: {len(merged)} rows")

    # Merge traits (player-level)
    merged = merged.merge(
        traits_df[trait_cols],
        on=["session_id", "player"],
        how="left",
    )
    print(f"  After trait merge: {len(merged)} rows")

    return merged


# =====
# Sale timing variable computation
# =====
def compute_sale_timing_vars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sale_prev_period and n_sales_earlier.

    Logic mirrors selling_period_regression.R lines 62-87:
      1. Compute total group sales per period within each group-round
      2. Shift to get previous period's sales count
      3. sale_prev_period = 1 if previous period had any sales
      4. n_sales_earlier = sales in periods 1 to t-2
    """
    df = df.copy()

    # Create group-round identifier
    df["group_round_id"] = (
        df["session_id"].astype(str) + "_" +
        df["segment"].astype(str) + "_" +
        df["group_id"].astype(str) + "_" +
        df["round"].astype(str)
    )

    # Compute period-level sales for each group-round
    period_sales = (
        df.groupby(["group_round_id", "period"])["sold"]
        .sum()
        .reset_index()
        .rename(columns={"sold": "n_sales_this_period"})
    )

    # Sort and shift to get previous period's sales
    period_sales = period_sales.sort_values(["group_round_id", "period"])
    period_sales["prev_period_n_sales"] = (
        period_sales.groupby("group_round_id")["n_sales_this_period"]
        .shift(1)
    )

    # Merge back to main dataframe
    df = df.merge(
        period_sales[["group_round_id", "period", "prev_period_n_sales"]],
        on=["group_round_id", "period"],
        how="left",
    )

    # sale_prev_period: 1 if any sale in period t-1
    df["sale_prev_period"] = (
        df["prev_period_n_sales"].notna() &
        (df["prev_period_n_sales"] > 0)
    ).astype(int)

    # n_sales_earlier: sales in periods 1 to t-2 (excludes t-1)
    df["n_sales_earlier"] = (
        df["prior_group_sales"] -
        df["prev_period_n_sales"].fillna(0).astype(int)
    )

    # Clean up temp columns
    df = df.drop(columns=["prev_period_n_sales"])

    return df


# =====
# Identifier variables
# =====
def add_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Add player_id and global_group_id for regression."""
    df = df.copy()

    df["player_id"] = df["session_id"].astype(str) + "_" + df["player"].astype(str)

    df["global_group_id"] = (
        df["session_id"].astype(str) + "_" +
        df["segment"].astype(str) + "_" +
        df["group_id"].astype(str)
    )

    return df


# =====
# Filtering
# =====
def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to analysis sample: already_sold == 0, valid emotions."""
    # Exclude players who already sold in this round
    filtered = df[df["already_sold"] == 0].copy()
    n_already_sold = len(df) - len(filtered)
    print(f"  Excluded {n_already_sold} rows where already_sold == 1")

    # Document missing emotions before excluding
    n_missing_emotions = filtered["fear_max"].isna().sum()
    print(f"  Found {n_missing_emotions} rows with missing emotion data")

    # Exclude missing emotions
    filtered = filtered[filtered["fear_max"].notna()]
    print(f"  Final sample: {len(filtered)} rows")

    return filtered


# =====
# Output
# =====
def print_summary(filtered: pd.DataFrame, original: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("EMOTION SPIKES DATASET SUMMARY")
    print("=" * 60)

    print(f"\nSample size: {len(filtered)} observations")
    print(f"  (from {len(original)} before filtering)")

    # sale_prev_period distribution
    sale_prev = filtered["sale_prev_period"].value_counts()
    print(f"\nsale_prev_period distribution:")
    print(f"  0 (no prior sale): {sale_prev.get(0, 0)}")
    print(f"  1 (prior sale): {sale_prev.get(1, 0)}")

    # Emotion coverage
    print("\nEmotion statistics (fear, anger):")
    for col in ["fear_max", "anger_max"]:
        non_null = filtered[col].dropna()
        print(f"  {col}: mean={non_null.mean():.3f}, max={non_null.max():.3f}")

    # Trait coverage
    print("\nTrait coverage:")
    for col in ["neuroticism", "impulsivity", "state_anxiety"]:
        n_valid = filtered[col].notna().sum()
        print(f"  {col}: {n_valid}/{len(filtered)} ({n_valid/len(filtered)*100:.1f}%)")

    # Sessions
    print(f"\nSessions: {sorted(filtered['session_id'].unique())}")
    print(f"Unique players: {filtered['player_id'].nunique()}")
    print(f"Unique groups: {filtered['global_group_id'].nunique()}")


def save_dataset(df: pd.DataFrame):
    """Save dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


# %%
if __name__ == "__main__":
    main()
