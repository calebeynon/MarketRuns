"""
Purpose: Build period-level emotion aggregations (mean, max, p95) from iMotions data
Author: Claude Code
Date: 2026-02-02

Extends the base iMotions processing to compute three aggregation types per emotion:
  - mean: Average emotion score during MarketPeriod
  - max: Maximum emotion score (captures peak intensity)
  - p95: 95th percentile (robust alternative to max)

OUTPUT VARIABLES:
    session_id, segment, round, period, player: Identifiers
    {emotion}_mean, {emotion}_max, {emotion}_p95: Aggregated emotion scores
    n_frames: Number of valid frames in aggregation window
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

# =====
# File paths and constants
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
IMOTIONS_DIR = DATASTORE / "imotions"
OUTPUT_PATH = DATASTORE / "derived" / "imotions_period_emotions_extended.csv"

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
IMOTIONS_SKIP_ROWS = 24


# =====
# Main function
# =====
def main():
    """Build the period-level emotions dataset with extended aggregations."""
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

    parsed = df[ANNOTATION_COL].apply(parse_market_period_annotation)
    mask = parsed.notna()
    market_df = df[mask].copy()

    if market_df.empty:
        return []

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
    """Parse MarketPeriod annotation into (segment, round, period)."""
    if pd.isna(annotation):
        return None

    match = MARKET_PERIOD_REGEX.match(str(annotation))
    if not match:
        return None

    segment = int(match.group(1))
    round_num = int(match.group(2))
    m_value = int(match.group(3))
    period = m_value - 1  # Offset: annotation m{N} -> oTree period N-1

    return (segment, round_num, period)


# =====
# Emotion aggregation
# =====
def aggregate_emotions(
    market_df: pd.DataFrame, session_id: str, player_label: str
) -> list[dict]:
    """Aggregate emotion columns to period-level mean, max, and p95."""
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

        for col in EMOTION_COLS:
            numeric_vals = pd.to_numeric(group[col], errors="coerce").dropna()

            if len(numeric_vals) > 0:
                record[f"{col.lower()}_mean"] = numeric_vals.mean()
                record[f"{col.lower()}_max"] = numeric_vals.max()
                record[f"{col.lower()}_p95"] = np.percentile(numeric_vals, 95)
            else:
                record[f"{col.lower()}_mean"] = np.nan
                record[f"{col.lower()}_max"] = np.nan
                record[f"{col.lower()}_p95"] = np.nan

        record["n_frames"] = len(group)
        records.append(record)

    return records


# =====
# Output
# =====
def print_summary(df: pd.DataFrame):
    """Print summary statistics for the emotions dataset."""
    print("\n" + "=" * 60)
    print("IMOTIONS PERIOD EMOTIONS EXTENDED SUMMARY")
    print("=" * 60)
    print(f"Total period-level observations: {len(df)}")
    print(f"Sessions: {df['session_id'].nunique()}")
    print(f"Unique players: {df['player'].nunique()}")

    print("\nFear and Anger statistics (focus emotions):")
    for emotion in ["fear", "anger"]:
        for agg in ["mean", "max", "p95"]:
            col = f"{emotion}_{agg}"
            non_null = df[col].dropna()
            print(f"  {col}: min={non_null.min():.3f}, max={non_null.max():.3f}, "
                  f"mean={non_null.mean():.3f}, NaN={df[col].isna().sum()}")

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
