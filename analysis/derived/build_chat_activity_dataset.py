"""
Purpose: Build chat activity dataset at player-period level for regression analysis
Author: Claude Code
Date: 2026-02-02

Creates a player-period level dataset capturing chat engagement for use in
analyzing whether chat communication mitigates emotional effects on selling.

Key insight: Chat occurs ONCE before each segment (not before each round).
- Segments 1-2: No chat (all chat variables = 0)
- Segments 3-4: Chat period occurs before trading begins

OUTPUT VARIABLES:
    session_id: Session identifier (e.g., "1_11-7-tr1")
    segment: Segment number within session (1-4)
    round: Round number within segment (1-14)
    period: Period within round (1-4, varies by round)
    player: Participant label
    group_id: Group identifier (1-4, fixed throughout session)
    messages_sent_segment: Total messages sent by player in this segment's chat
    messages_received_segment: Messages received by player from group members
    total_group_messages: Total messages in player's group for this segment
"""

import pandas as pd
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from market_data import parse_experiment

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
OUTPUT_PATH = DATASTORE / "derived" / "chat_activity_dataset.csv"

# Session folders with treatment indicators
SESSIONS = {
    "1_11-7-tr1": "tr1",
    "2_11-10-tr2": "tr2",
    "3_11-11-tr2": "tr2",
    "4_11-12-tr1": "tr1",
    "5_11-14-tr2": "tr2",
    "6_11-18-tr1": "tr1",
}

# Mapping segment names to numbers
SEGMENT_MAP = {
    "chat_noavg": 1,
    "chat_noavg2": 2,
    "chat_noavg3": 3,
    "chat_noavg4": 4,
}

# Segments with chat enabled
CHAT_SEGMENTS = [3, 4]


# =====
# Main function
# =====
def main():
    """Build the chat activity dataset."""
    all_records = []

    print("Processing sessions...")
    for session_name, treatment in SESSIONS.items():
        print(f"  {session_name} (treatment {treatment})")
        records = process_session(session_name)
        all_records.extend(records)
        print(f"    -> {len(records)} player-period observations")

    df = pd.DataFrame(all_records)
    print_summary_statistics(df)
    validate_dataset(df)
    save_dataset(df)

    return df


# =====
# Session processing
# =====
def process_session(session_name: str) -> list[dict]:
    """Process a session and return player-period chat activity records."""
    session = load_session_data(session_name)
    if session is None:
        return []

    records = []
    for segment_name, segment in session.segments.items():
        segment_num = SEGMENT_MAP.get(segment_name, 0)
        chat_counts = compute_segment_chat_counts(segment, segment_num)
        records.extend(build_segment_records(
            segment, segment_num, session_name, chat_counts
        ))
    return records


def load_session_data(session_name: str):
    """Load session data using market_data module."""
    session_folder = DATASTORE / session_name
    data_csv = session_folder / f"{session_name}_data.csv"
    chat_csv = session_folder / f"{session_name}_chat.csv"

    experiment = parse_experiment(str(data_csv), str(chat_csv))
    if not experiment.sessions:
        print(f"    Warning: No session data found for {session_name}")
        return None
    return experiment.sessions[0]


# =====
# Chat counting functions
# =====
def compute_segment_chat_counts(segment, segment_num: int) -> dict:
    """Compute chat message counts for the entire segment."""
    if segment_num not in CHAT_SEGMENTS:
        return init_zero_chat_counts(segment)
    return count_chat_messages(segment)


def init_zero_chat_counts(segment) -> dict:
    """Initialize zero chat counts for segments without chat."""
    chat_counts = {}
    for group_id, group in segment.groups.items():
        chat_counts[group_id] = {
            'messages_by_player': {label: 0 for label in group.player_labels},
            'total_messages': 0
        }
    return chat_counts


def count_chat_messages(segment) -> dict:
    """Count chat messages for each group in a chat-enabled segment."""
    chat_counts = {}
    for group_id, group in segment.groups.items():
        messages_by_player = {label: 0 for label in group.player_labels}
        total = sum_group_chat(segment, messages_by_player)
        chat_counts[group_id] = {
            'messages_by_player': messages_by_player,
            'total_messages': total
        }
    return chat_counts


def sum_group_chat(segment, messages_by_player: dict) -> int:
    """Sum chat messages across all rounds, updating messages_by_player."""
    total = 0
    for round_obj in segment.rounds.values():
        for msg in round_obj.chat_messages:
            if msg.nickname in messages_by_player:
                messages_by_player[msg.nickname] += 1
                total += 1
    return total


# =====
# Record building functions
# =====
def build_segment_records(
    segment, segment_num: int, session_name: str, chat_counts: dict
) -> list[dict]:
    """Build player-period records for a segment with chat activity."""
    records = []
    for round_num, round_obj in segment.rounds.items():
        for period_num, period in round_obj.periods.items():
            records.extend(build_period_records(
                period, segment, segment_num, session_name,
                round_num, period_num, chat_counts
            ))
    return records


def build_period_records(
    period, segment, segment_num: int, session_name: str,
    round_num: int, period_num: int, chat_counts: dict
) -> list[dict]:
    """Build records for all players in a single period."""
    records = []
    for label, player in period.players.items():
        record = build_player_record(
            label, segment, segment_num, session_name,
            round_num, period_num, chat_counts
        )
        if record:
            records.append(record)
    return records


def build_player_record(
    label: str, segment, segment_num: int, session_name: str,
    round_num: int, period_num: int, chat_counts: dict
) -> dict | None:
    """Build a single player-period record."""
    group = segment.get_group_by_player(label)
    if group is None:
        return None

    sent, received, total = get_player_chat_counts(label, group.group_id, chat_counts)
    return create_record_dict(
        session_name, segment_num, round_num, period_num,
        label, group.group_id, sent, received, total
    )


def get_player_chat_counts(label: str, group_id: int, chat_counts: dict) -> tuple:
    """Extract chat counts for a player from chat_counts structure."""
    group_chat = chat_counts.get(group_id, {})
    messages_by_player = group_chat.get('messages_by_player', {})
    total_group = group_chat.get('total_messages', 0)
    sent = messages_by_player.get(label, 0)
    return sent, total_group - sent, total_group


def create_record_dict(
    session_name: str, segment_num: int, round_num: int, period_num: int,
    label: str, group_id: int, sent: int, received: int, total: int
) -> dict:
    """Create the output record dictionary."""
    return {
        'session_id': session_name,
        'segment': segment_num,
        'round': round_num,
        'period': period_num,
        'player': label,
        'group_id': group_id,
        'messages_sent_segment': sent,
        'messages_received_segment': received,
        'total_group_messages': total,
    }


# =====
# Output and validation functions
# =====
def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics for the dataset."""
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print_basic_counts(df)
    print_messages_by_session(df)
    print_messages_by_segment(df)
    print_player_distribution(df)


def print_basic_counts(df: pd.DataFrame):
    """Print basic dataset counts."""
    print(f"Total observations: {len(df)}")
    print(f"Unique sessions: {df['session_id'].nunique()}")
    print(f"Unique players: {df['player'].nunique()}")


def print_messages_by_session(df: pd.DataFrame):
    """Print message totals by session."""
    print("\nTotal messages by session:")
    by_session = df.groupby('session_id').agg({
        'messages_sent_segment': 'sum',
        'total_group_messages': 'first'
    })
    print(by_session)


def print_messages_by_segment(df: pd.DataFrame):
    """Print message totals by segment."""
    print("\nTotal messages by segment:")
    by_segment = df.groupby('segment').agg({
        'messages_sent_segment': 'sum',
        'total_group_messages': 'mean'
    })
    print(by_segment)


def print_player_distribution(df: pd.DataFrame):
    """Print distribution of messages sent per player."""
    print("\nMessages sent distribution per player:")
    player_totals = df.drop_duplicates(
        subset=['session_id', 'segment', 'player']
    )['messages_sent_segment']
    print(player_totals.describe())


def validate_dataset(df: pd.DataFrame):
    """Validate dataset integrity."""
    print("\n" + "=" * 50)
    print("VALIDATION CHECKS")
    print("=" * 50)
    validate_no_chat_segments(df)
    validate_segments_present(df)
    validate_received_calculation(df)
    validate_non_negative(df)


def validate_no_chat_segments(df: pd.DataFrame):
    """Verify segments 1-2 have zero chat activity."""
    seg_12 = df[df['segment'].isin([1, 2])]
    has_chat = (seg_12['total_group_messages'] > 0).any()
    print(f"Segments 1-2 have chat activity: {has_chat} (should be False)")


def validate_segments_present(df: pd.DataFrame):
    """Verify all segments are present."""
    segments = sorted(df['segment'].unique())
    print(f"Segments present: {segments} (should be [1, 2, 3, 4])")


def validate_received_calculation(df: pd.DataFrame):
    """Verify messages_received equals total - sent."""
    computed = df['total_group_messages'] - df['messages_sent_segment']
    mismatch = (computed != df['messages_received_segment']).sum()
    print(f"Received calculation mismatches: {mismatch} (should be 0)")


def validate_non_negative(df: pd.DataFrame):
    """Verify no negative message counts."""
    neg_sent = (df['messages_sent_segment'] < 0).sum()
    neg_received = (df['messages_received_segment'] < 0).sum()
    neg_total = (df['total_group_messages'] < 0).sum()
    print(f"Negative message counts: sent={neg_sent}, "
          f"received={neg_received}, total={neg_total} (all should be 0)")


def save_dataset(df: pd.DataFrame):
    """Save dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


# %%
if __name__ == "__main__":
    main()
