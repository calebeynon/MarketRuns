"""
Purpose: Build period-level emotion averages from iMotions facial expression data
Author: Claude Code
Date: 2026-01-28

Reads raw iMotions CSV exports (one per participant per session), filters to
MarketPeriod annotations, and aggregates emotion columns to period-level means.

IMOTIONS PERIOD OFFSET:
    iMotions annotations use `m{N}` which maps to oTree period `N-1`. This offset
    exists because `generate_annotations_unfiltered_v2.py` pre-increments the
    period counter BEFORE recording each MarketPeriod annotation. As a result:
        - m2 -> period 1
        - m3 -> period 2
        - etc.
    The offset is applied in `parse_market_period_annotation()` so that downstream
    datasets use oTree-aligned period values.

OUTPUT VARIABLES:
    session_id: Session identifier (e.g., "1_11-7-tr1")
    segment: Segment number (1-4)
    round: Round number within segment (1-14)
    period: Period within round (1+, varies) — aligned to oTree periods
    player: Participant label (A-R, excluding I and O)
    anger_mean, contempt_mean, disgust_mean, fear_mean, joy_mean,
    sadness_mean, surprise_mean, engagement_mean, valence_mean: Period-level
        mean emotion scores from Affectiva AFFDEX
    n_frames: Number of valid frames in the aggregation window
"""

import re
import pandas as pd
from pathlib import Path

# =====
# File paths and constants
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
IMOTIONS_DIR = DATASTORE / "imotions"
OUTPUT_PATH = DATASTORE / "derived" / "imotions_period_emotions.csv"

# iMotions session folder -> oTree session ID
IMOTIONS_SESSION_MAP = {
    "1": "1_11-7-tr1",
    "2": "2_11-10-tr2",
    "3": "3_11-11-tr2",
    "4": "4_11-12-tr1",
    "5": "5_11-14-tr2",
    "6": "6_11-18-tr1",
}

EMOTION_COLS = [
    "Anger", "Contempt", "Disgust", "Fear", "Joy",
    "Sadness", "Surprise", "Engagement", "Valence",
]

ANNOTATION_COL = "Respondent Annotations active"
MARKET_PERIOD_REGEX = re.compile(r"^s(\d+)r(\d+)m(\d+)MarketPeriod$")

# Number of metadata rows to skip in iMotions CSV files
IMOTIONS_SKIP_ROWS = 24


# =====
# Main function
# =====
def main():
    """Build the period-level emotions dataset."""
    all_records = []

    print("Processing iMotions data...")
    for imotions_session, session_id in IMOTIONS_SESSION_MAP.items():
        print(f"  Session {imotions_session} ({session_id})")
        session_dir = IMOTIONS_DIR / imotions_session
        records = process_session(session_dir, session_id)
        all_records.extend(records)
        print(f"    -> {len(records)} period-level observations")

    df = pd.DataFrame(all_records)
    print_summary(df)
    save_dataset(df)

    return df


# =====
# Session and file processing
# =====
def process_session(session_dir: Path, session_id: str) -> list[dict]:
    """Process all participant files in an iMotions session directory."""
    records = []
    csv_files = sorted(session_dir.glob("*.csv"))

    for csv_file in csv_files:
        if csv_file.name == "ExportMerge.csv":
            continue

        player_label = extract_player_label(csv_file.name)
        if player_label is None:
            print(f"    Warning: Could not extract label from {csv_file.name}")
            continue

        file_records = process_participant_file(csv_file, session_id, player_label)
        records.extend(file_records)

    return records


def extract_player_label(filename: str) -> str | None:
    """Extract participant letter from filename pattern {order}_{letter}{suffix}.csv."""
    match = re.match(r"\d+_([A-Z])\d+\.csv", filename)
    if match:
        return match.group(1)
    return None


def process_participant_file(
    filepath: Path, session_id: str, player_label: str
) -> list[dict]:
    """Load one iMotions CSV and aggregate emotions by period."""
    df = pd.read_csv(
        filepath,
        skiprows=IMOTIONS_SKIP_ROWS,
        encoding="utf-8-sig",
        low_memory=False,
    )

    # Parse annotation column to identify MarketPeriod rows
    parsed = df[ANNOTATION_COL].apply(parse_market_period_annotation)
    mask = parsed.notna()
    market_df = df[mask].copy()

    if market_df.empty:
        return []

    # Attach parsed segment, round, period columns
    parsed_data = parsed[mask].apply(pd.Series)
    parsed_data.columns = ["segment", "round", "period"]
    market_df = pd.concat(
        [market_df.reset_index(drop=True), parsed_data.reset_index(drop=True)],
        axis=1,
    )

    return aggregate_emotions(market_df, session_id, player_label)


# =====
# Annotation parsing
# =====
def parse_market_period_annotation(annotation) -> tuple | None:
    """
    Parse a MarketPeriod annotation string into (segment, round, period).

    Returns None for non-MarketPeriod annotations.
    """
    if pd.isna(annotation):
        return None

    match = MARKET_PERIOD_REGEX.match(str(annotation))
    if not match:
        return None

    segment = int(match.group(1))
    round_num = int(match.group(2))
    m_value = int(match.group(3))

    # OFFSET EXPLANATION:
    # The iMotions annotation generator (`generate_annotations_unfiltered_v2.py`)
    # pre-increments the period counter before recording MarketPeriod annotations.
    # This means annotation m{N} corresponds to oTree period N-1:
    #   - m2 -> period 1 (first period in round)
    #   - m3 -> period 2
    #   - m4 -> period 3
    # We apply this offset here so all downstream datasets (imotions_period_emotions.csv,
    # emotions_traits_selling_dataset.csv) align with oTree period numbering.
    period = m_value - 1

    return (segment, round_num, period)


# =====
# Emotion aggregation
# =====
def aggregate_emotions(
    market_df: pd.DataFrame, session_id: str, player_label: str
) -> list[dict]:
    """Aggregate emotion columns to period-level means."""
    records = []
    grouped = market_df.groupby(["segment", "round", "period"])

    for (segment, round_num, period), group in grouped:
        record = {
            "session_id": session_id,
            "segment": segment,
            "round": round_num,
            "period": period,
            "player": player_label,
        }

        # Compute mean of each emotion column (ignoring NaN)
        for col in EMOTION_COLS:
            numeric_vals = pd.to_numeric(group[col], errors="coerce")
            record[f"{col.lower()}_mean"] = numeric_vals.mean()

        record["n_frames"] = len(group)
        records.append(record)

    return records


# =====
# Output
# =====
def print_summary(df: pd.DataFrame):
    """Print summary statistics for the emotions dataset."""
    print("\n" + "=" * 50)
    print("IMOTIONS PERIOD EMOTIONS SUMMARY")
    print("=" * 50)
    print(f"Total period-level observations: {len(df)}")
    print(f"Sessions: {df['session_id'].nunique()}")
    print(f"Unique players: {df['player'].nunique()}")
    print(f"Segments: {sorted(df['segment'].unique())}")
    print(f"Rounds: {df['round'].min()}-{df['round'].max()}")
    print(f"Periods: {df['period'].min()}-{df['period'].max()}")

    emotion_cols = [f"{c.lower()}_mean" for c in EMOTION_COLS]
    print("\nEmotion score ranges:")
    for col in emotion_cols:
        non_null = df[col].dropna()
        print(f"  {col}: {non_null.min():.3f} - {non_null.max():.3f} "
              f"(mean={non_null.mean():.3f}, NaN={df[col].isna().sum()})")

    print(f"\nn_frames: {df['n_frames'].min()} - {df['n_frames'].max()} "
          f"(mean={df['n_frames'].mean():.1f})")


def save_dataset(df: pd.DataFrame):
    """Save dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


# %%
if __name__ == "__main__":
    main()
