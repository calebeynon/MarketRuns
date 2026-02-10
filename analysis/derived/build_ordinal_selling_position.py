"""
Purpose: Build ordinal selling position dataset for ordinal logit regression
Author: Claude Code
Date: 2026-02-09

Creates a player-round level dataset with selling rank (1st-4th) within each
group-round, merged with sell-period p95 emotions and survey personality traits.

OUTPUT VARIABLES:
    session_id: Session identifier (e.g., "1_11-7-tr1")
    treatment: Treatment condition (tr1 or tr2)
    segment: Market segment (1-4)
    group_id: Trading group identifier (1-4)
    round: Round number within segment (1-14)
    player: Participant label (A-R, excluding I and O)
    player_id: Unique player identifier ("{session_id}_{player}")
    sell_period: Period when player sold (NaN if never sold)
    did_sell: Binary (0/1) - did the player sell?
    sell_rank: Selling position within group-round (1-4, min-rank for ties)
    anger_p95 ... valence_p95: 9 p95 emotion columns at sell/last period
    extraversion ... state_anxiety: BFI-10 and anxiety traits
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
ROUND_PANEL_PATH = DATASTORE / "derived" / "individual_round_panel.csv"
PERIOD_DATASET_PATH = DATASTORE / "derived" / "individual_period_dataset.csv"
EMOTIONS_PATH = DATASTORE / "derived" / "imotions_period_emotions_extended.csv"
SURVEY_TRAITS_PATH = DATASTORE / "derived" / "survey_traits.csv"
OUTPUT_PATH = DATASTORE / "derived" / "ordinal_selling_position.csv"

GROUP_ROUND_KEYS = ["session_id", "segment", "group_id", "round"]

P95_COLS = [
    "anger_p95", "contempt_p95", "disgust_p95", "fear_p95", "joy_p95",
    "sadness_p95", "surprise_p95", "engagement_p95", "valence_p95",
]


# =====
# Main function
# =====
def main():
    """Build the ordinal selling position dataset."""
    round_panel = pd.read_csv(ROUND_PANEL_PATH)
    print(f"Loaded round panel: {len(round_panel)} rows")

    round_panel = compute_selling_ranks(round_panel)
    max_periods = compute_max_periods()
    round_panel = determine_emotion_periods(round_panel, max_periods)
    round_panel = merge_emotions(round_panel)
    round_panel = merge_traits(round_panel)
    round_panel = create_derived_variables(round_panel)

    final_df = select_output_columns(round_panel)
    validate_dataset(final_df)
    print_summary(final_df)
    save_dataset(final_df)

    return final_df


# =====
# Rank computation
# =====
def compute_selling_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Compute min-rank selling position within each group-round."""
    df = df.copy()
    df["sell_rank"] = 4  # Default: non-sellers get rank 4

    sellers_mask = df["did_sell"] == 1
    sellers = df.loc[sellers_mask].copy()

    ranked = sellers.groupby(GROUP_ROUND_KEYS)["sell_period"].rank(
        method="min"
    )
    df.loc[sellers_mask, "sell_rank"] = ranked.astype(int)

    return df


# =====
# Max period computation
# =====
def compute_max_periods() -> pd.DataFrame:
    """Compute max period per group-round from period dataset."""
    period_df = pd.read_csv(PERIOD_DATASET_PATH)
    max_periods = (
        period_df.groupby(GROUP_ROUND_KEYS)["period"]
        .max()
        .reset_index()
        .rename(columns={"period": "max_period"})
    )
    print(f"Computed max periods for {len(max_periods)} group-rounds")
    return max_periods


# =====
# Emotion period determination
# =====
def determine_emotion_periods(
    df: pd.DataFrame, max_periods: pd.DataFrame
) -> pd.DataFrame:
    """Set emotion_period: sell_period for sellers, max_period for non-sellers."""
    df = df.merge(max_periods, on=GROUP_ROUND_KEYS, how="left")

    sellers_mask = df["did_sell"] == 1
    df.loc[sellers_mask, "emotion_period"] = df.loc[sellers_mask, "sell_period"]
    df.loc[~sellers_mask, "emotion_period"] = df.loc[~sellers_mask, "max_period"]
    df["emotion_period"] = df["emotion_period"].astype(int)

    df = df.drop(columns=["max_period"])
    return df


# =====
# Emotion merging
# =====
def merge_emotions(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join p95 emotions at the emotion_period for each player."""
    emotions = pd.read_csv(EMOTIONS_PATH)
    emotions = emotions[["session_id", "segment", "round", "period", "player"] + P95_COLS]

    merged = df.merge(
        emotions,
        left_on=["session_id", "segment", "round", "emotion_period", "player"],
        right_on=["session_id", "segment", "round", "period", "player"],
        how="left",
    )
    merged = merged.drop(columns=["period"])

    missing_count = merged[P95_COLS[0]].isna().sum()
    print(f"Emotion merge: {missing_count} rows missing iMotions data")
    return merged


# =====
# Trait merging
# =====
def merge_traits(df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join survey traits; drops players missing survey data."""
    traits = pd.read_csv(SURVEY_TRAITS_PATH)
    rows_before = len(df)

    merged = df.merge(traits, on=["session_id", "player"], how="inner")

    rows_dropped = rows_before - len(merged)
    print(f"Trait merge: dropped {rows_dropped} rows (missing survey data)")
    return merged


# =====
# Derived variables
# =====
def create_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create gender_female binary and player_id for random effects."""
    df = df.copy()
    df["gender_female"] = (df["gender"] == "Female").astype(int)
    df["player_id"] = df["session_id"] + "_" + df["player"]
    return df


# =====
# Column selection
# =====
def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order final output columns."""
    output_cols = [
        "session_id", "treatment", "segment", "group_id", "round",
        "player", "player_id",
        "sell_period", "did_sell", "sell_rank",
    ] + P95_COLS + [
        "extraversion", "agreeableness", "conscientiousness",
        "neuroticism", "openness", "impulsivity", "state_anxiety",
        "age", "gender_female",
    ]
    return df[output_cols]


# =====
# Validation
# =====
def validate_dataset(df: pd.DataFrame):
    """Validate sell_rank values and trait completeness."""
    print("\nValidation:")
    rank_values = set(df["sell_rank"].unique())
    print(f"  sell_rank values: {sorted(rank_values)}")
    assert rank_values <= {1, 2, 3, 4}, f"Unexpected ranks: {rank_values}"

    non_sellers = df[df["did_sell"] == 0]
    bad_ranks = non_sellers[non_sellers["sell_rank"] != 4]
    assert len(bad_ranks) == 0, "Non-sellers with rank != 4 found"
    print("  OK: All non-sellers have sell_rank == 4")

    trait_cols = [
        "extraversion", "agreeableness", "conscientiousness",
        "neuroticism", "openness", "impulsivity", "state_anxiety",
        "age", "gender_female",
    ]
    missing_traits = df[trait_cols].isna().sum().sum()
    assert missing_traits == 0, f"Missing trait values: {missing_traits}"
    print("  OK: No missing trait values")


# =====
# Summary and output
# =====
def print_summary(df: pd.DataFrame):
    """Print dataset summary: row count, rank distribution, missing emotions."""
    print("\n" + "=" * 50)
    print("ORDINAL SELLING POSITION DATASET SUMMARY")
    print("=" * 50)
    print(f"Total rows: {len(df)}")
    print(f"Unique players: {df['player_id'].nunique()}")

    print("\nSell rank distribution:")
    rank_counts = df["sell_rank"].value_counts().sort_index()
    for rank, count in rank_counts.items():
        print(f"  Rank {rank}: {count} ({100 * count / len(df):.1f}%)")

    missing = df[P95_COLS[0]].isna().sum()
    print(f"\nRows with missing emotion data: {missing}")


def save_dataset(df: pd.DataFrame):
    """Save the dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


# %%
if __name__ == "__main__":
    main()
