"""
Purpose: Merge individual period dataset with survey traits and iMotions emotions
Author: Claude Code
Date: 2026-01-28

Combines three data sources into a single analysis-ready dataset:
  1. individual_period_dataset.csv — selling decisions (player-period level)
  2. survey_traits.csv — personality traits (player level)
  3. imotions_period_emotions.csv — facial emotions (player-period level)

OUTPUT VARIABLES:
    All columns from individual_period_dataset, plus:
    - Trait columns: extraversion, agreeableness, conscientiousness,
      neuroticism, openness, impulsivity, state_anxiety, age, gender
    - Emotion columns: anger_mean, contempt_mean, disgust_mean, fear_mean,
      joy_mean, sadness_mean, surprise_mean, engagement_mean, valence_mean, n_frames
    - global_group_id: Unique group identifier across sessions ({session_id}_{segment}_{group_id})
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
INPUT_TRAITS = DERIVED / "survey_traits.csv"
INPUT_EMOTIONS = DERIVED / "imotions_period_emotions.csv"
OUTPUT_PATH = DERIVED / "emotions_traits_selling_dataset.csv"


# =====
# Main function
# =====
def main():
    """Merge all three datasets and save."""
    print("Loading datasets...")
    period_df = pd.read_csv(INPUT_PERIOD)
    traits_df = pd.read_csv(INPUT_TRAITS)
    emotions_df = pd.read_csv(INPUT_EMOTIONS)

    print(f"  Period data: {len(period_df)} rows")
    print(f"  Traits data: {len(traits_df)} rows")
    print(f"  Emotions data: {len(emotions_df)} rows")

    print("\nMerging...")
    merged = merge_datasets(period_df, traits_df, emotions_df)
    merged = add_global_group_id(merged)

    print_merge_report(merged, period_df)
    save_dataset(merged)

    return merged


# =====
# Merge logic
# =====
def merge_datasets(
    period_df: pd.DataFrame,
    traits_df: pd.DataFrame,
    emotions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join traits and emotions onto the period dataset."""
    # Merge traits (player-level, constant across all periods)
    trait_cols = [
        "session_id", "player", "extraversion", "agreeableness",
        "conscientiousness", "neuroticism", "openness",
        "impulsivity", "state_anxiety", "age", "gender",
    ]
    merged = period_df.merge(
        traits_df[trait_cols],
        on=["session_id", "player"],
        how="left",
    )

    # Merge emotions (period-level)
    emotion_cols = [
        "session_id", "segment", "round", "period", "player",
        "anger_mean", "contempt_mean", "disgust_mean", "fear_mean",
        "joy_mean", "sadness_mean", "surprise_mean",
        "engagement_mean", "valence_mean", "n_frames",
    ]
    merged = merged.merge(
        emotions_df[emotion_cols],
        on=["session_id", "segment", "round", "period", "player"],
        how="left",
    )

    return merged


def add_global_group_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add unique group identifier across sessions and segments."""
    df["global_group_id"] = (
        df["session_id"].astype(str) + "_"
        + df["segment"].astype(str) + "_"
        + df["group_id"].astype(str)
    )
    return df


# =====
# Reporting
# =====
def print_merge_report(merged: pd.DataFrame, original: pd.DataFrame):
    """Report merge quality metrics."""
    n_total = len(merged)

    print("\n" + "=" * 50)
    print("MERGE REPORT")
    print("=" * 50)
    print(f"Total rows: {n_total} (original: {len(original)})")

    # Traits coverage
    trait_matched = merged["extraversion"].notna().sum()
    print(f"\nTraits coverage: {trait_matched}/{n_total} "
          f"({trait_matched / n_total * 100:.1f}%)")

    # Emotions coverage
    emotion_matched = merged["anger_mean"].notna().sum()
    print(f"Emotions coverage: {emotion_matched}/{n_total} "
          f"({emotion_matched / n_total * 100:.1f}%)")

    # NaN counts for key columns
    print("\nNaN counts:")
    check_cols = [
        "extraversion", "impulsivity", "state_anxiety",
        "anger_mean", "joy_mean", "n_frames",
    ]
    for col in check_cols:
        n_nan = merged[col].isna().sum()
        print(f"  {col}: {n_nan}")

    # Sessions represented
    print(f"\nSessions: {sorted(merged['session_id'].unique())}")
    print(f"Unique global_group_ids: {merged['global_group_id'].nunique()}")


def save_dataset(df: pd.DataFrame):
    """Save dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


# %%
if __name__ == "__main__":
    main()
