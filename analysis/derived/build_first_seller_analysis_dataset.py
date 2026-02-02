"""
Purpose: Build analysis dataset for first seller prediction regression
Author: Claude Code
Date: 2026-02-01

Merges first seller round data with survey personality traits for regression
analysis of traits predicting first seller behavior.

OUTPUT VARIABLES:
    session_id: Session identifier (e.g., "1_11-7-tr1")
    treatment: Treatment condition (tr1 or tr2)
    segment: Market segment (1-4)
    group_id: Trading group identifier (1-4)
    round: Round number within segment (1-14)
    player: Participant label (A-R, excluding I and O)
    public_signal: Public signal value (0.5 or missing for round 1)
    state: True state (0 or 1)
    is_first_seller: Whether player was first to sell (1) or not (0)
    first_sale_period: Period when first sale occurred in round
    extraversion: BFI-10 extraversion score (1-7)
    agreeableness: BFI-10 agreeableness score (1-7)
    conscientiousness: BFI-10 conscientiousness score (1-7)
    neuroticism: BFI-10 neuroticism score (1-7)
    openness: BFI-10 openness score (1-7)
    impulsivity: Impulsivity score (1-7)
    state_anxiety: State anxiety score (1-4)
    age: Participant age
    gender_female: Binary indicator (1=Female, 0=otherwise)
"""

import pandas as pd
from pathlib import Path

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
FIRST_SELLER_PATH = DATASTORE / "derived" / "first_seller_round_data.csv"
SURVEY_TRAITS_PATH = DATASTORE / "derived" / "survey_traits.csv"
OUTPUT_PATH = DATASTORE / "derived" / "first_seller_analysis_data.csv"


# =====
# Main function
# =====
def main():
    """Build first seller analysis dataset by merging round data with traits."""
    first_seller_df = load_first_seller_data()
    traits_df = load_and_prepare_traits()

    merged_df = merge_datasets(first_seller_df, traits_df)
    final_df = finalize_columns(merged_df)

    validate_dataset(final_df)
    print_summary(final_df)
    save_dataset(final_df)

    return final_df


# =====
# Data loading
# =====
def load_first_seller_data() -> pd.DataFrame:
    """Load the first seller round-level dataset."""
    print(f"Loading first seller data from: {FIRST_SELLER_PATH}")
    df = pd.read_csv(FIRST_SELLER_PATH)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    return df


def load_and_prepare_traits() -> pd.DataFrame:
    """Load survey traits and create gender_female binary variable."""
    print(f"Loading survey traits from: {SURVEY_TRAITS_PATH}")
    df = pd.read_csv(SURVEY_TRAITS_PATH)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

    # Create binary gender variable
    df["gender_female"] = (df["gender"] == "Female").astype(int)

    # Drop original gender column - only keep binary version
    df = df.drop(columns=["gender"])

    return df


# =====
# Data merging
# =====
def merge_datasets(first_seller_df: pd.DataFrame, traits_df: pd.DataFrame) -> pd.DataFrame:
    """Merge first seller data with traits on session_id and player."""
    print("\nMerging datasets...")
    rows_before = len(first_seller_df)

    # Inner join drops rows without matching traits
    merged = first_seller_df.merge(
        traits_df,
        on=["session_id", "player"],
        how="inner"
    )

    rows_dropped = rows_before - len(merged)
    print(f"  Rows before merge: {rows_before}")
    print(f"  Rows after merge: {len(merged)}")
    print(f"  Rows dropped (missing traits): {rows_dropped}")

    return merged


def finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order final columns for output."""
    output_columns = [
        "session_id", "treatment", "segment", "group_id", "round", "player",
        "public_signal", "state", "is_first_seller", "first_sale_period",
        "extraversion", "agreeableness", "conscientiousness", "neuroticism",
        "openness", "impulsivity", "state_anxiety", "age", "gender_female"
    ]
    return df[output_columns]


# =====
# Validation
# =====
def validate_dataset(df: pd.DataFrame):
    """Validate no missing values in trait columns after merge."""
    trait_cols = [
        "extraversion", "agreeableness", "conscientiousness",
        "neuroticism", "openness", "impulsivity", "state_anxiety",
        "age", "gender_female"
    ]

    print("\nValidation:")
    missing_traits = df[trait_cols].isna().sum()
    has_missing = missing_traits.sum() > 0

    if has_missing:
        print("  WARNING: Missing values in trait columns:")
        for col, count in missing_traits.items():
            if count > 0:
                print(f"    {col}: {count}")
    else:
        print("  OK: No missing values in trait columns")


# =====
# Summary and output
# =====
def print_summary(df: pd.DataFrame):
    """Print summary statistics for the analysis dataset."""
    print("\n" + "=" * 50)
    print("FIRST SELLER ANALYSIS DATA SUMMARY")
    print("=" * 50)

    print(f"\nDataset dimensions: {len(df)} rows x {len(df.columns)} columns")
    print(f"Unique participants: {df.groupby(['session_id', 'player']).ngroups}")
    print(f"Sessions: {df['session_id'].nunique()}")

    print_first_seller_counts(df)
    print_trait_stats(df)


def print_first_seller_counts(df: pd.DataFrame):
    """Print first seller counts by treatment and segment."""
    print("\nFirst seller counts by treatment and segment:")
    counts = df.groupby(["treatment", "segment"])["is_first_seller"].sum()
    totals = df.groupby(["treatment", "segment"])["is_first_seller"].count()

    for (treatment, segment), count in counts.items():
        total = totals[(treatment, segment)]
        pct = 100 * count / total
        print(f"  {treatment}, segment {segment}: {int(count)}/{total} ({pct:.1f}%)")


def print_trait_stats(df: pd.DataFrame):
    """Print trait variable summary statistics."""
    trait_cols = [
        "extraversion", "agreeableness", "conscientiousness",
        "neuroticism", "openness", "impulsivity", "state_anxiety"
    ]

    print("\nTrait summary statistics:")
    for col in trait_cols:
        print(f"  {col}: mean={df[col].mean():.2f}, sd={df[col].std():.2f}")

    print(f"\nDemographics:")
    print(f"  Age: mean={df['age'].mean():.1f}, range={df['age'].min()}-{df['age'].max()}")
    print(f"  Female: {df['gender_female'].sum()} ({100*df['gender_female'].mean():.1f}%)")


def save_dataset(df: pd.DataFrame):
    """Save the analysis dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


# %%
if __name__ == "__main__":
    main()
