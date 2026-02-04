"""
Purpose: Merge datasets for chat mitigation regression analysis
Author: Claude Code
Date: 2026-02-03

Creates an analysis-ready player-period level dataset by merging:
- Individual period trading data (base)
- iMotions period-level emotions (extended, with mean/max)
- Personality traits from survey

OUTPUT VARIABLES:
    # Identifiers
    session_id: Session identifier (e.g., "1_11-7-tr1")
    segment: Segment number (1-4)
    round: Round number within segment (1-14)
    period: Period within round
    player: Participant label
    group_id: Group identifier (1-4)
    player_id: Unique player identifier (session_id + player)
    global_group_id: Unique group identifier (session_id + segment + group_id)
    treatment: Treatment condition ("tr1" or "tr2")

    # Outcome
    sold: Binary (0/1) - sold THIS period
    already_sold: Binary (0/1) - sold in prior period of round

    # Treatment indicator
    chat_segment: Binary (0 for segments 1-2, 1 for segments 3-4)

    # Time-varying emotions
    fear_mean, fear_max, anger_mean, anger_max, sadness_mean, sadness_max,
    joy_mean, valence_mean, engagement_mean, n_frames

    # Standardized emotions
    fear_z, anger_z, sadness_z

    # Time-invariant traits
    state_anxiety, extraversion, agreeableness, conscientiousness,
    neuroticism, openness, impulsivity

    # Standardized traits
    neuroticism_z, impulsivity_z

    # Controls
    signal, state, price, prior_group_sales

DATA FILTERING:
    - Rounds 1-10 only (iMotions coverage limitation)
    - Both already_sold == 0 and already_sold == 1 kept
"""

import numpy as np
import pandas as pd
from pathlib import Path

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
DERIVED_DIR = DATASTORE / "derived"

INPUT_INDIVIDUAL_PERIOD = DERIVED_DIR / "individual_period_dataset.csv"
INPUT_EMOTIONS = DERIVED_DIR / "imotions_period_emotions_extended.csv"  # Has mean/max/p95
INPUT_TRAITS = DERIVED_DIR / "survey_traits.csv"

OUTPUT_PATH = DERIVED_DIR / "chat_mitigation_dataset.csv"

# Output column order (defined as constant to keep reorder_columns short)
COLUMN_ORDER = [
    'session_id', 'segment', 'round', 'period', 'player', 'group_id',
    'player_id', 'global_group_id', 'treatment', 'sold', 'already_sold',
    'chat_segment', 'fear_mean', 'fear_max', 'anger_mean', 'anger_max',
    'sadness_mean', 'sadness_max', 'joy_mean', 'valence_mean',
    'engagement_mean', 'n_frames', 'fear_z', 'anger_z', 'sadness_z',
    'state_anxiety', 'extraversion', 'agreeableness', 'conscientiousness',
    'neuroticism', 'openness', 'impulsivity', 'neuroticism_z', 'impulsivity_z',
    'signal', 'state', 'price', 'prior_group_sales'
]


# =====
# Main function
# =====
def main():
    """Build the chat mitigation analysis dataset."""
    base_df, emotions_df, traits_df = load_all_datasets()
    merged_df = merge_all_datasets(base_df, emotions_df, traits_df)
    final_df = create_analysis_variables(merged_df)
    final_df = filter_to_imotions_coverage(final_df)
    final_df = final_df[COLUMN_ORDER]
    print_summary_statistics(final_df)
    validate_dataset(final_df)
    save_dataset(final_df)
    return final_df


def load_all_datasets() -> tuple:
    """Load all input datasets."""
    print("Loading input datasets...")
    base_df = load_base_data()
    emotions_df = load_emotions_data()
    traits_df = load_traits_data()
    return base_df, emotions_df, traits_df


# =====
# Data loading functions
# =====
def load_base_data() -> pd.DataFrame:
    """Load base individual period dataset."""
    df = pd.read_csv(INPUT_INDIVIDUAL_PERIOD)
    print(f"  Base data: {len(df)} rows")
    return df


def load_emotions_data() -> pd.DataFrame:
    """Load extended emotions data (already has string session_id)."""
    df = pd.read_csv(INPUT_EMOTIONS)
    print(f"  Emotions data: {len(df)} rows")
    return df


def load_traits_data() -> pd.DataFrame:
    """Load personality traits data."""
    df = pd.read_csv(INPUT_TRAITS)
    print(f"  Traits data: {len(df)} rows")
    return df


# =====
# Merge functions
# =====
def merge_all_datasets(
    base_df: pd.DataFrame,
    emotions_df: pd.DataFrame,
    traits_df: pd.DataFrame
) -> pd.DataFrame:
    """Perform all dataset merges."""
    merged = merge_emotions(base_df, emotions_df)
    merged = merge_traits(merged, traits_df)
    return merged


def merge_emotions(base_df: pd.DataFrame, emotions_df: pd.DataFrame) -> pd.DataFrame:
    """Left join emotions on period-level keys."""
    emotion_cols = [
        'session_id', 'segment', 'round', 'period', 'player',
        'fear_mean', 'fear_max', 'anger_mean', 'anger_max',
        'sadness_mean', 'sadness_max', 'joy_mean', 'valence_mean',
        'engagement_mean', 'n_frames'
    ]
    emotions_subset = emotions_df[emotion_cols]

    merge_keys = ['session_id', 'segment', 'round', 'period', 'player']
    merged = base_df.merge(emotions_subset, on=merge_keys, how='left')

    matched = merged['fear_mean'].notna().sum()
    unmatched = merged['fear_mean'].isna().sum()
    print(f"  Emotions merge: {matched} matched, {unmatched} unmatched")

    return merged


def merge_traits(merged_df: pd.DataFrame, traits_df: pd.DataFrame) -> pd.DataFrame:
    """Left join traits on player-level keys (time-invariant)."""
    trait_cols = [
        'session_id', 'player',
        'state_anxiety', 'extraversion', 'agreeableness',
        'conscientiousness', 'neuroticism', 'openness', 'impulsivity'
    ]
    traits_subset = traits_df[trait_cols]

    merge_keys = ['session_id', 'player']
    merged = merged_df.merge(traits_subset, on=merge_keys, how='left')

    matched = merged['neuroticism'].notna().sum()
    unmatched = merged['neuroticism'].isna().sum()
    print(f"  Traits merge: {matched} matched, {unmatched} unmatched")

    check_missing_traits(merged)

    return merged


def check_missing_traits(df: pd.DataFrame):
    """Warn if any session/player combination is missing trait data."""
    missing = df[df['neuroticism'].isna()][['session_id', 'player']].drop_duplicates()
    if len(missing) > 0:
        print(f"  WARNING: {len(missing)} players missing trait data:")
        for _, row in missing.iterrows():
            print(f"    - {row['session_id']} / {row['player']}")


# =====
# Variable creation functions
# =====
def create_analysis_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived analysis variables."""
    df = df.copy()
    df = create_id_variables(df)
    df = create_chat_segment_indicator(df)
    df = create_standardized_emotions(df)
    df = create_standardized_traits(df)
    return df


def create_id_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create unique player and group identifiers."""
    df['player_id'] = df['session_id'] + '_' + df['player']
    df['global_group_id'] = (
        df['session_id'] + '_' +
        df['segment'].astype(str) + '_' +
        df['group_id'].astype(str)
    )
    return df


def create_chat_segment_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary indicator for chat-enabled segments."""
    df['chat_segment'] = (df['segment'] >= 3).astype(int)
    return df


def create_standardized_emotions(df: pd.DataFrame) -> pd.DataFrame:
    """Create z-scored emotion variables."""
    df['fear_z'] = zscore_column(df, 'fear_mean')
    df['anger_z'] = zscore_column(df, 'anger_mean')
    df['sadness_z'] = zscore_column(df, 'sadness_mean')
    return df


def create_standardized_traits(df: pd.DataFrame) -> pd.DataFrame:
    """Create z-scored trait variables."""
    df['neuroticism_z'] = zscore_column(df, 'neuroticism')
    df['impulsivity_z'] = zscore_column(df, 'impulsivity')
    return df


def zscore_column(df: pd.DataFrame, col: str) -> pd.Series:
    """Compute z-score for a column, handling NaN values."""
    values = df[col]
    mean_val = values.mean()
    std_val = values.std()
    if std_val == 0 or pd.isna(std_val):
        return pd.Series([np.nan] * len(df), index=df.index)
    return (values - mean_val) / std_val


# =====
# Filtering functions
# =====
def filter_to_imotions_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rounds 1-10 where iMotions data is available."""
    n_before = len(df)
    df_filtered = df[df['round'] <= 10].copy()
    n_after = len(df_filtered)
    print(f"  Filtered from {n_before} to {n_after} rows")
    return df_filtered


# =====
# Output functions
# =====
def print_summary_statistics(df: pd.DataFrame):
    """Print comprehensive summary statistics."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print_basic_counts(df)
    print_sample_by_segment(df)
    print_missing_data_summary(df)


def print_basic_counts(df: pd.DataFrame):
    """Print basic observation counts."""
    print(f"\nTotal observations: {len(df)}")
    print(f"Unique sessions: {df['session_id'].nunique()}")
    print(f"Unique players (player_id): {df['player_id'].nunique()}")
    print(f"Unique groups (global_group_id): {df['global_group_id'].nunique()}")
    print(f"\nSold this period: {df['sold'].sum()}")
    print(f"Already sold: {df['already_sold'].sum()}")
    print(f"At-risk observations: {(df['already_sold'] == 0).sum()}")


def print_sample_by_segment(df: pd.DataFrame):
    """Print sample sizes by segment and chat_segment."""
    print("\nObservations by segment:")
    print(df.groupby('segment').size())
    print("\nObservations by chat_segment:")
    print(df.groupby('chat_segment').size())
    print(f"\nChat segment = 0: {(df['chat_segment'] == 0).sum()}")
    print(f"Chat segment = 1: {(df['chat_segment'] == 1).sum()}")


def print_missing_data_summary(df: pd.DataFrame):
    """Print missing data counts by variable."""
    print("\nMissing data by variable:")
    missing_vars = [
        'fear_mean', 'anger_mean', 'sadness_mean',
        'neuroticism', 'impulsivity'
    ]
    for var in missing_vars:
        n_missing = df[var].isna().sum()
        pct = 100 * n_missing / len(df)
        print(f"  {var}: {n_missing} ({pct:.1f}%)")


def validate_dataset(df: pd.DataFrame):
    """Validate dataset integrity."""
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)
    validate_key_uniqueness(df)
    validate_chat_segment_logic(df)
    validate_round_filter(df)
    validate_z_scores(df)
    validate_trait_coverage(df)


def validate_key_uniqueness(df: pd.DataFrame):
    """Check that merge keys are unique."""
    keys = ['session_id', 'segment', 'round', 'period', 'player']
    n_unique = df.groupby(keys).size().max()
    status = "PASS" if n_unique == 1 else "FAIL"
    print(f"[{status}] Key uniqueness: max {n_unique} rows per key (expected 1)")


def validate_chat_segment_logic(df: pd.DataFrame):
    """Verify chat_segment correctly maps to segments."""
    seg12_chat = df[df['segment'].isin([1, 2])]['chat_segment'].unique()
    seg34_chat = df[df['segment'].isin([3, 4])]['chat_segment'].unique()
    correct_12 = set(seg12_chat) == {0}
    correct_34 = set(seg34_chat) == {1}
    status = "PASS" if correct_12 and correct_34 else "FAIL"
    print(f"[{status}] chat_segment logic: segments 1-2={list(seg12_chat)}, "
          f"segments 3-4={list(seg34_chat)}")


def validate_round_filter(df: pd.DataFrame):
    """Verify rounds are filtered to 1-10."""
    max_round = df['round'].max()
    status = "PASS" if max_round <= 10 else "FAIL"
    print(f"[{status}] Round filter: max round = {max_round} (expected <= 10)")


def validate_z_scores(df: pd.DataFrame):
    """Verify z-scores have approximately mean 0 and std 1."""
    for col in ['fear_z', 'anger_z', 'sadness_z', 'neuroticism_z', 'impulsivity_z']:
        mean_val = df[col].mean()
        std_val = df[col].std()
        # Mean should be ~0, std should be ~1 (accounting for NaN)
        mean_ok = abs(mean_val) < 0.01 if not pd.isna(mean_val) else True
        std_ok = abs(std_val - 1) < 0.01 if not pd.isna(std_val) else True
        status = "PASS" if mean_ok and std_ok else "WARN"
        if not pd.isna(mean_val):
            print(f"[{status}] {col}: mean={mean_val:.4f}, std={std_val:.4f}")


def validate_trait_coverage(df: pd.DataFrame):
    """Check that 95+ participants have trait data."""
    n_with_traits = df[df['neuroticism'].notna()]['player_id'].nunique()
    status = "PASS" if n_with_traits >= 95 else "WARN"
    print(f"[{status}] Players with trait data: {n_with_traits} (expected >= 95)")


def save_dataset(df: pd.DataFrame):
    """Save dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


# %%
if __name__ == "__main__":
    main()
