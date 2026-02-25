"""
Purpose: Build group-segment level chat sentiment dataset using VADER
Author: Claude Code
Date: 2026-02-25

Scores each chat message with VADER sentiment, maps channels to groups,
and aggregates to group x segment x session level (48 rows).
"""

import re

import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =====
# File paths and constants
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
OUTPUT_PATH = DATASTORE / "derived" / "chat_sentiment_dataset.csv"

SESSIONS = {
    "1_11-7-tr1": "tr1",
    "2_11-10-tr2": "tr2",
    "3_11-11-tr2": "tr2",
    "4_11-12-tr1": "tr1",
    "5_11-14-tr2": "tr2",
    "6_11-18-tr1": "tr1",
}

CHAT_SEGMENTS = {
    "chat_noavg3": 3,
    "chat_noavg4": 4,
}

CHANNEL_PATTERN = re.compile(r"^\d+-([^-]+)-(\d+)$")


# =====
# Main function
# =====
def main():
    """Build chat sentiment dataset aggregated to group-segment level."""
    analyzer = SentimentIntensityAnalyzer()
    all_records = []

    print("Processing sessions...")
    for session_id, treatment in SESSIONS.items():
        print(f"  {session_id} (treatment {treatment})")
        records = process_session(session_id, treatment, analyzer)
        all_records.extend(records)
        print(f"    -> {len(records)} group-segment observations")

    df = pd.DataFrame(all_records)
    print_summary(df)
    save_dataset(df)

    return df


# =====
# Session processing
# =====
def process_session(
    session_id: str, treatment: str, analyzer: SentimentIntensityAnalyzer
) -> list[dict]:
    """Process all chat segments for a session."""
    session_folder = DATASTORE / session_id
    chat_df = load_chat_messages(session_folder)
    records = []

    for segment_name, segment_num in CHAT_SEGMENTS.items():
        label_to_group = build_label_group_map(session_folder, segment_name)
        segment_records = process_segment(
            chat_df, segment_name, segment_num,
            label_to_group, session_id, treatment, analyzer,
        )
        records.extend(segment_records)

    return records


def process_segment(
    chat_df: pd.DataFrame,
    segment_name: str,
    segment_num: int,
    label_to_group: dict,
    session_id: str,
    treatment: str,
    analyzer: SentimentIntensityAnalyzer,
) -> list[dict]:
    """Score messages and aggregate to group level for one segment."""
    segment_msgs = filter_segment_messages(chat_df, segment_name)
    segment_msgs = score_messages(segment_msgs, analyzer)
    channel_to_group = map_channels_to_groups(segment_msgs, label_to_group)
    segment_msgs["group_id"] = segment_msgs["channel"].map(channel_to_group)

    return aggregate_by_group(
        segment_msgs, session_id, segment_num, treatment, label_to_group,
    )


# =====
# Message filtering and scoring
# =====
def filter_segment_messages(chat_df: pd.DataFrame, segment_name: str) -> pd.DataFrame:
    """Filter chat messages to a specific segment using channel pattern."""
    mask = chat_df["channel"].apply(
        lambda ch: extract_segment_from_channel(ch) == segment_name
    )
    return chat_df[mask].copy()


def extract_segment_from_channel(channel: str) -> str | None:
    """Extract segment name from channel string (e.g., '1-chat_noavg3-133')."""
    match = CHANNEL_PATTERN.match(channel)
    return match.group(1) if match else None


def score_messages(
    df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer
) -> pd.DataFrame:
    """Add VADER compound score to each message."""
    df["compound"] = df["body"].astype(str).apply(
        lambda text: analyzer.polarity_scores(text)["compound"]
    )
    return df


# =====
# Channel-to-group mapping
# =====
def build_label_group_map(session_folder: Path, segment_name: str) -> dict:
    """Build participant label -> group_id mapping from segment CSV."""
    segment_df = load_segment_csv(session_folder, segment_name)
    pairs = segment_df[["participant.label", "group.id_in_subsession"]].drop_duplicates()
    return dict(zip(pairs["participant.label"], pairs["group.id_in_subsession"]))


def map_channels_to_groups(
    segment_msgs: pd.DataFrame, label_to_group: dict
) -> dict:
    """Map each channel to its group_id using nickname lookup."""
    channel_to_group = {}
    for channel in segment_msgs["channel"].unique():
        first_nick = segment_msgs.loc[
            segment_msgs["channel"] == channel, "nickname"
        ].iloc[0]
        channel_to_group[channel] = int(label_to_group[first_nick])
    return channel_to_group


# =====
# Aggregation
# =====
def aggregate_by_group(
    segment_msgs: pd.DataFrame,
    session_id: str,
    segment_num: int,
    treatment: str,
    label_to_group: dict,
) -> list[dict]:
    """Aggregate sentiment scores to group level, including zero-message groups."""
    all_group_ids = sorted(set(label_to_group.values()))
    records = []

    for group_id in all_group_ids:
        group_msgs = segment_msgs[segment_msgs["group_id"] == group_id]
        record = build_group_record(
            group_msgs, session_id, segment_num, treatment, int(group_id),
        )
        records.append(record)

    return records


def build_group_record(
    group_msgs: pd.DataFrame,
    session_id: str,
    segment_num: int,
    treatment: str,
    group_id: int,
) -> dict:
    """Build a single group-segment record with sentiment aggregates."""
    n = len(group_msgs)
    return {
        "session_id": session_id,
        "segment": segment_num,
        "group_id": group_id,
        "treatment": treatment,
        "vader_compound_mean": group_msgs["compound"].mean() if n > 0 else 0.0,
        "message_count": n,
        "frac_positive": (group_msgs["compound"] > 0.05).mean() if n > 0 else 0.0,
        "frac_negative": (group_msgs["compound"] < -0.05).mean() if n > 0 else 0.0,
    }


# =====
# Data loading
# =====
def load_chat_messages(session_folder: Path) -> pd.DataFrame:
    """Load ChatMessages CSV from session folder."""
    csv_files = sorted(session_folder.glob("ChatMessages-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No ChatMessages CSV in {session_folder}")
    return pd.read_csv(csv_files[0])


def load_segment_csv(session_folder: Path, segment_name: str) -> pd.DataFrame:
    """Load segment CSV (e.g., chat_noavg3_*.csv) from session folder."""
    csv_files = sorted(session_folder.glob(f"{segment_name}_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV for {segment_name} in {session_folder}")
    return pd.read_csv(csv_files[0])


# =====
# Output and summary
# =====
def print_summary(df: pd.DataFrame):
    """Print dataset summary statistics."""
    print("\n" + "=" * 50)
    print("CHAT SENTIMENT DATASET SUMMARY")
    print("=" * 50)
    print(f"Total rows: {len(df)}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"\nMean compound score: {df['vader_compound_mean'].mean():.3f}")
    print(f"Mean message count: {df['message_count'].mean():.1f}")
    print(f"\nBy treatment:")
    print(df.groupby("treatment")[["vader_compound_mean", "message_count"]].mean())
    print(f"\nBy segment:")
    print(df.groupby("segment")[["vader_compound_mean", "message_count"]].mean())


def save_dataset(df: pd.DataFrame):
    """Save dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


# %%
if __name__ == "__main__":
    df = main()
