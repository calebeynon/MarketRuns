"""
Purpose: Build player-period level iMotions emotions dataset
Author: Claude Code
Date: 2025-02-02

Aggregates iMotions facial expression data at the player-period level for
analyzing emotional responses during market trading periods.

OUTPUT VARIABLES:
    session_id: Session identifier (1-6)
    segment: Segment number within session (1-4)
    round: Round number within segment (1-10, iMotions data covers rounds 1-10)
    period: Period within round (1-indexed, oTree period)
    player: Participant label (letter A-R)

    fear_mean, fear_max, fear_std: Fear emotion aggregates
    anger_mean, anger_max, anger_std: Anger emotion aggregates
    sadness_mean, sadness_max, sadness_std: Sadness emotion aggregates
    joy_mean, joy_max: Joy emotion aggregates
    valence_mean, valence_max: Valence aggregates (-100 to 100 scale)
    engagement_mean: Engagement aggregate
    n_samples: Count of iMotions rows for data quality assessment

NOTES:
    - CRITICAL OFFSET: iMotions m{N} maps to oTree period N-1 (e.g., m2 -> period 1)
    - This is because the annotation generator pre-increments before the first MarketPeriod
    - Only MarketPeriod phase is included (not MarketPeriodWait)
    - iMotions data only covers rounds 1-10
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
IMOTIONS_DIR = DATASTORE / "imotions"
OUTPUT_PATH = DATASTORE / "derived" / "period_emotions_dataset.csv"

ANNOTATION_COL = 'Respondent Annotations active'


# =====
# Main function
# =====
def main():
    """Build the period emotions dataset from iMotions data."""
    all_records = collect_all_session_records()

    if not all_records:
        print("No data found. Check datastore/imotions directory.")
        return None

    df = pd.DataFrame(all_records)
    print_summary_statistics(df)
    save_dataset(df)
    return df


def collect_all_session_records() -> list[dict]:
    """Collect records from all sessions."""
    all_records = []
    print("Processing iMotions sessions...")

    for session_num in range(1, 7):
        session_dir = IMOTIONS_DIR / str(session_num)
        if not session_dir.exists():
            print(f"  Session {session_num}: directory not found, skipping")
            continue

        print(f"  Session {session_num}")
        records = process_session(session_dir, session_num)
        all_records.extend(records)
        print(f"    -> {len(records)} player-period observations")

    return all_records


# =====
# Session processing
# =====
def process_session(session_dir: Path, session_num: int) -> list[dict]:
    """Process all participant files in a session directory."""
    records = []
    csv_files = get_participant_files(session_dir)

    for csv_file in csv_files:
        player_letter = extract_player_letter(csv_file.name)
        if player_letter is None:
            continue
        records.extend(process_participant_safely(csv_file, session_num, player_letter))

    return records


def get_participant_files(session_dir: Path) -> list[Path]:
    """Get participant CSV files, excluding ExportMerge files."""
    csv_files = list(session_dir.glob("*.csv"))
    return [f for f in csv_files if "ExportMerge" not in f.name]


def process_participant_safely(
    file_path: Path, session_num: int, player_letter: str
) -> list[dict]:
    """Process participant file with error handling."""
    try:
        return process_participant_file(file_path, session_num, player_letter)
    except Exception as e:
        print(f"    Warning: Error processing {file_path.name}: {e}")
        return []


def process_participant_file(
    file_path: Path, session_num: int, player_letter: str
) -> list[dict]:
    """Process a single participant's iMotions file."""
    df = pd.read_csv(file_path, skiprows=24, encoding='utf-8-sig')
    if ANNOTATION_COL not in df.columns:
        return []

    df_market = filter_market_period_rows(df)
    if df_market.empty:
        return []

    return aggregate_by_period(df_market, session_num, player_letter)


def filter_market_period_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to MarketPeriod rows only (exclude MarketPeriodWait)."""
    df_market = df[df[ANNOTATION_COL].str.contains('MarketPeriod', na=False)]
    return df_market[~df_market[ANNOTATION_COL].str.contains('MarketPeriodWait', na=False)]


# =====
# Annotation parsing
# =====
def extract_player_letter(filename: str) -> str | None:
    """Extract player letter from iMotions filename."""
    match = re.match(r'\d+_([A-Z])\d+\.csv', filename)
    return match.group(1) if match else None


def parse_annotation(annotation: str) -> tuple[int, int, int] | None:
    """
    Parse iMotions annotation to extract segment, round, period.

    Returns (segment, round, oTree_period) or None if parsing fails.

    CRITICAL: iMotions m{N} corresponds to oTree period N-1.
    Example: m2 -> period 1, m3 -> period 2, etc.
    This offset exists because generate_annotations_unfiltered_v2.py pre-increments
    the market_period_counter before recording the first MarketPeriod.
    """
    match = re.match(r's(\d+)r(\d+)m(\d+)MarketPeriod', annotation)
    if not match:
        return None

    segment = int(match.group(1))
    round_num = int(match.group(2))
    imotions_period = int(match.group(3))
    otree_period = imotions_period - 1  # Apply offset: m2 -> 1, m3 -> 2, etc.
    return segment, round_num, otree_period


# =====
# Emotion aggregation
# =====
def aggregate_by_period(
    df: pd.DataFrame, session_num: int, player_letter: str
) -> list[dict]:
    """Aggregate emotions for each unique period in the dataframe."""
    records = []

    for annotation, group_df in df.groupby(ANNOTATION_COL):
        record = process_annotation_group(
            str(annotation), group_df, session_num, player_letter
        )
        if record is not None:
            records.append(record)

    return records


def process_annotation_group(
    annotation: str, group_df: pd.DataFrame, session_num: int, player_letter: str
) -> dict | None:
    """Process a single annotation group and return record or None."""
    parsed = parse_annotation(annotation)
    if parsed is None:
        return None

    segment, round_num, period = parsed

    # Skip period 0 (m1 annotations - shouldn't exist but filter if they do)
    if period < 1:
        return None

    return build_period_record(group_df, session_num, segment, round_num, period, player_letter)


def build_period_record(
    df: pd.DataFrame, session_num: int, segment: int,
    round_num: int, period: int, player: str
) -> dict:
    """Build a single player-period record with aggregated emotions."""
    record = {
        'session_id': session_num, 'segment': segment, 'round': round_num,
        'period': period, 'player': player, 'n_samples': len(df),
    }
    record.update(aggregate_all_emotions(df))
    return record


def aggregate_all_emotions(df: pd.DataFrame) -> dict:
    """Aggregate all emotion statistics into a single dict."""
    result = {}
    result.update(_aggregate_emotion(df, 'Fear', ['mean', 'max', 'std']))
    result.update(_aggregate_emotion(df, 'Anger', ['mean', 'max', 'std']))
    result.update(_aggregate_emotion(df, 'Sadness', ['mean', 'max', 'std']))
    result.update(_aggregate_emotion(df, 'Joy', ['mean', 'max']))
    result.update(_aggregate_emotion(df, 'Valence', ['mean', 'max']))
    result.update(_aggregate_emotion(df, 'Engagement', ['mean']))
    return result


def _aggregate_emotion(df: pd.DataFrame, col: str, stats: list[str]) -> dict:
    """Generic emotion aggregation helper."""
    col_lower = col.lower()
    if col not in df.columns:
        return {f'{col_lower}_{stat}': np.nan for stat in stats}

    values = pd.to_numeric(df[col], errors='coerce')
    stat_funcs = {'mean': values.mean, 'max': values.max, 'std': values.std}
    return {f'{col_lower}_{stat}': stat_funcs[stat]() for stat in stats}


# =====
# Output functions
# =====
def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics for the dataset."""
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print_basic_counts(df)
    print_quality_warnings(df)
    print_missing_data(df)
    print("\nFirst 10 rows:")
    print(df.head(10).to_string())


def print_basic_counts(df: pd.DataFrame):
    """Print basic observation counts."""
    print(f"Total observations: {len(df)}")
    print(f"Unique sessions: {df['session_id'].nunique()}")
    print(f"Unique players: {df['player'].nunique()}")
    print("\nObservations by session:")
    print(df.groupby('session_id').size())
    print("\nObservations by segment:")
    print(df.groupby('segment').size())


def print_quality_warnings(df: pd.DataFrame):
    """Print data quality warnings."""
    low_sample = (df['n_samples'] < 10).sum()
    if low_sample > 0:
        print(f"\nWARNING: {low_sample} periods have < 10 samples")
    print("\nSample size distribution:")
    print(df['n_samples'].describe())


def print_missing_data(df: pd.DataFrame):
    """Print missing data counts."""
    print("\nMissing data counts:")
    for col in ['fear_mean', 'anger_mean', 'sadness_mean', 'joy_mean']:
        print(f"  {col}: {df[col].isna().sum()}")


def save_dataset(df: pd.DataFrame):
    """Save dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


# %%
if __name__ == "__main__":
    df = main()
