"""
Purpose: Unit and integration tests for build_chat_sentiment_dataset.py
Author: Claude Code
Date: 2026-02-25

Validates VADER scoring, channel-to-group mapping, aggregation logic,
and the final derived CSV dimensions and values.
"""

from pathlib import Path

import pandas as pd
import pytest

from analysis.derived.build_chat_sentiment_dataset import (
    CHANNEL_PATTERN,
    CHAT_SEGMENTS,
    DATASTORE,
    OUTPUT_PATH,
    SESSIONS,
    aggregate_by_group,
    build_group_record,
    build_label_group_map,
    extract_segment_from_channel,
    filter_segment_messages,
    load_chat_messages,
    load_segment_csv,
    map_channels_to_groups,
    score_messages,
)


# =====
# Constants
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DERIVED_CSV = PROJECT_ROOT / "datastore" / "derived" / "chat_sentiment_dataset.csv"
EXPECTED_COLUMNS = [
    "session_id", "segment", "group_id", "treatment",
    "vader_compound_mean", "message_count", "frac_positive", "frac_negative",
]

DATASTORE_AVAILABLE = DATASTORE.exists() and DERIVED_CSV.exists()
skip_no_datastore = pytest.mark.skipif(
    not DATASTORE_AVAILABLE, reason="Datastore not accessible"
)


# =====
# Unit tests: channel parsing
# =====
def test_extract_segment_valid():
    """Valid channel strings return correct segment name."""
    assert extract_segment_from_channel("1-chat_noavg3-133") == "chat_noavg3"
    assert extract_segment_from_channel("2-chat_noavg4-200") == "chat_noavg4"
    assert extract_segment_from_channel("5-chat_noavg3-55") == "chat_noavg3"


def test_extract_segment_invalid():
    """Invalid channel strings return None."""
    assert extract_segment_from_channel("bad-format") is None
    assert extract_segment_from_channel("") is None
    assert extract_segment_from_channel("1-chat_noavg3") is None


def test_channel_pattern_groups():
    """Regex captures segment name and round number separately."""
    match = CHANNEL_PATTERN.match("3-chat_noavg4-150")
    assert match is not None
    assert match.group(1) == "chat_noavg4"
    assert match.group(2) == "150"


# =====
# Unit tests: message filtering
# =====
def test_filter_segment_messages():
    """Filters chat dataframe to only matching segment."""
    chat_df = pd.DataFrame({
        "channel": [
            "1-chat_noavg3-100", "1-chat_noavg3-100",
            "1-chat_noavg4-200", "1-chat_noavg-50",
        ],
        "body": ["hello", "world", "foo", "bar"],
    })
    result = filter_segment_messages(chat_df, "chat_noavg3")
    assert len(result) == 2
    assert all(result["channel"] == "1-chat_noavg3-100")


def test_filter_segment_empty():
    """Returns empty dataframe when no messages match."""
    chat_df = pd.DataFrame({
        "channel": ["1-chat_noavg4-200"],
        "body": ["test"],
    })
    result = filter_segment_messages(chat_df, "chat_noavg3")
    assert len(result) == 0


# =====
# Unit tests: VADER scoring
# =====
def test_score_messages_positive():
    """Positive text gets positive compound score."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    df = pd.DataFrame({"body": ["This is wonderful and great!"]})
    result = score_messages(df, analyzer)
    assert "compound" in result.columns
    assert result["compound"].iloc[0] > 0.05


def test_score_messages_negative():
    """Negative text gets negative compound score."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    df = pd.DataFrame({"body": ["This is terrible and awful."]})
    result = score_messages(df, analyzer)
    assert result["compound"].iloc[0] < -0.05


def test_score_messages_neutral():
    """Neutral text gets near-zero compound score."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    df = pd.DataFrame({"body": ["The meeting is at 3pm."]})
    result = score_messages(df, analyzer)
    assert abs(result["compound"].iloc[0]) <= 0.05


# =====
# Unit tests: group record building
# =====
def test_build_group_record_with_messages():
    """Group record with messages computes correct aggregates."""
    df = pd.DataFrame({"compound": [0.5, -0.2, 0.0]})
    record = build_group_record(df, "sess1", 3, "tr1", 1)
    assert record["session_id"] == "sess1"
    assert record["segment"] == 3
    assert record["group_id"] == 1
    assert record["treatment"] == "tr1"
    assert record["message_count"] == 3
    assert abs(record["vader_compound_mean"] - 0.1) < 1e-10
    # 0.5 > 0.05 (positive), -0.2 < -0.05 (negative), 0.0 neither
    assert abs(record["frac_positive"] - 1 / 3) < 1e-10
    assert abs(record["frac_negative"] - 1 / 3) < 1e-10


def test_build_group_record_empty():
    """Empty group gets zero sentiment and zero message count."""
    df = pd.DataFrame({"compound": pd.Series([], dtype=float)})
    record = build_group_record(df, "sess1", 3, "tr1", 2)
    assert record["message_count"] == 0
    assert record["vader_compound_mean"] == 0.0
    assert record["frac_positive"] == 0.0
    assert record["frac_negative"] == 0.0


# =====
# Unit tests: aggregation
# =====
def test_aggregate_includes_zero_message_groups():
    """Groups with no messages still appear in aggregation."""
    segment_msgs = pd.DataFrame({
        "group_id": [1, 1, 1],
        "compound": [0.3, 0.1, -0.1],
    })
    label_to_group = {"A": 1, "B": 2}  # group 2 has no messages
    records = aggregate_by_group(
        segment_msgs, "sess1", 3, "tr1", label_to_group,
    )
    assert len(records) == 2
    assert records[0]["group_id"] == 1
    assert records[0]["message_count"] == 3
    assert records[1]["group_id"] == 2
    assert records[1]["message_count"] == 0


# =====
# Integration tests: derived CSV
# =====
@skip_no_datastore
def test_derived_csv_exists():
    """Output CSV file exists and is non-empty."""
    assert DERIVED_CSV.exists()
    assert DERIVED_CSV.stat().st_size > 0


@skip_no_datastore
def test_derived_csv_dimensions():
    """Derived dataset has 48 rows and 8 columns."""
    df = pd.read_csv(DERIVED_CSV)
    assert df.shape == (48, 8), f"Expected (48, 8), got {df.shape}"


@skip_no_datastore
def test_derived_csv_columns():
    """All expected columns are present."""
    df = pd.read_csv(DERIVED_CSV)
    assert list(df.columns) == EXPECTED_COLUMNS


@skip_no_datastore
def test_derived_csv_no_missing():
    """No missing values in derived dataset."""
    df = pd.read_csv(DERIVED_CSV)
    assert df.isna().sum().sum() == 0


@skip_no_datastore
def test_derived_csv_sessions():
    """All 6 sessions are represented."""
    df = pd.read_csv(DERIVED_CSV)
    assert set(df["session_id"]) == set(SESSIONS.keys())


@skip_no_datastore
def test_derived_csv_segments():
    """Only segments 3 and 4 are present."""
    df = pd.read_csv(DERIVED_CSV)
    assert set(df["segment"]) == {3, 4}


@skip_no_datastore
def test_derived_csv_groups():
    """Groups are 1-4 in each session-segment."""
    df = pd.read_csv(DERIVED_CSV)
    for _, sub in df.groupby(["session_id", "segment"]):
        assert sorted(sub["group_id"].tolist()) == [1, 2, 3, 4]


@skip_no_datastore
def test_derived_csv_treatments():
    """Treatments match session assignments."""
    df = pd.read_csv(DERIVED_CSV)
    for session_id, treatment in SESSIONS.items():
        session_rows = df[df["session_id"] == session_id]
        assert (session_rows["treatment"] == treatment).all(), (
            f"Treatment mismatch for {session_id}"
        )


@skip_no_datastore
def test_derived_csv_compound_range():
    """VADER compound scores are in [-1, 1] range."""
    df = pd.read_csv(DERIVED_CSV)
    assert df["vader_compound_mean"].between(-1, 1).all()


@skip_no_datastore
def test_derived_csv_fraction_range():
    """Fraction positive and negative are in [0, 1]."""
    df = pd.read_csv(DERIVED_CSV)
    assert df["frac_positive"].between(0, 1).all()
    assert df["frac_negative"].between(0, 1).all()


@skip_no_datastore
def test_derived_csv_message_counts_positive():
    """All groups have at least some messages (non-negative counts)."""
    df = pd.read_csv(DERIVED_CSV)
    assert (df["message_count"] >= 0).all()


@skip_no_datastore
def test_derived_csv_balanced_design():
    """Each treatment x segment cell has exactly 12 observations."""
    df = pd.read_csv(DERIVED_CSV)
    counts = df.groupby(["treatment", "segment"]).size()
    assert (counts == 12).all(), f"Unbalanced: {counts.to_dict()}"


# =====
# Integration tests: group mapping against raw data
# =====
@skip_no_datastore
@pytest.mark.parametrize("session_id", list(SESSIONS.keys()))
@pytest.mark.parametrize("segment_name", list(CHAT_SEGMENTS.keys()))
def test_all_channel_nicknames_belong_to_same_group(session_id, segment_name):
    """Every nickname in a chat channel maps to the same group in the segment CSV."""
    session_folder = DATASTORE / session_id
    chat_df = load_chat_messages(session_folder)
    seg_chat = chat_df[chat_df["channel"].str.contains(segment_name)]
    label_to_group = build_label_group_map(session_folder, segment_name)

    for channel in seg_chat["channel"].unique():
        nicks = seg_chat[seg_chat["channel"] == channel]["nickname"].unique()
        groups = {int(label_to_group[n]) for n in nicks}
        assert len(groups) == 1, (
            f"{session_id}/{segment_name} channel {channel}: "
            f"nicknames {sorted(nicks)} map to multiple groups {groups}"
        )


@skip_no_datastore
@pytest.mark.parametrize("session_id", list(SESSIONS.keys()))
@pytest.mark.parametrize("segment_name", list(CHAT_SEGMENTS.keys()))
def test_channel_group_mapping_covers_all_four_groups(session_id, segment_name):
    """Each session-segment has exactly 4 channels mapping to groups 1-4."""
    session_folder = DATASTORE / session_id
    chat_df = load_chat_messages(session_folder)
    seg_chat = chat_df[chat_df["channel"].str.contains(segment_name)]
    label_to_group = build_label_group_map(session_folder, segment_name)
    channel_to_group = map_channels_to_groups(seg_chat, label_to_group)

    mapped_groups = sorted(channel_to_group.values())
    assert mapped_groups == [1, 2, 3, 4], (
        f"{session_id}/{segment_name}: channels map to {mapped_groups}, "
        f"expected [1, 2, 3, 4]"
    )


@skip_no_datastore
@pytest.mark.parametrize("session_id", list(SESSIONS.keys()))
@pytest.mark.parametrize("segment_name", list(CHAT_SEGMENTS.keys()))
def test_derived_sentiment_matches_hand_computed(session_id, segment_name):
    """Derived dataset sentiment matches independent VADER scoring of raw messages."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    session_folder = DATASTORE / session_id
    segment_num = CHAT_SEGMENTS[segment_name]

    chat_df = load_chat_messages(session_folder)
    seg_chat = chat_df[chat_df["channel"].str.contains(segment_name)]
    label_to_group = build_label_group_map(session_folder, segment_name)
    channel_to_group = map_channels_to_groups(seg_chat, label_to_group)

    derived = pd.read_csv(DERIVED_CSV)

    for channel, group_id in channel_to_group.items():
        msgs = seg_chat[seg_chat["channel"] == channel]["body"].astype(str)
        scores = [analyzer.polarity_scores(m)["compound"] for m in msgs]
        hand_mean = sum(scores) / len(scores) if scores else 0.0
        hand_count = len(scores)

        row = derived[
            (derived["session_id"] == session_id)
            & (derived["segment"] == segment_num)
            & (derived["group_id"] == group_id)
        ]
        assert len(row) == 1, (
            f"Expected 1 row for {session_id}/seg{segment_num}/g{group_id}, "
            f"got {len(row)}"
        )
        assert row["message_count"].values[0] == hand_count, (
            f"{session_id}/seg{segment_num}/g{group_id}: "
            f"count {row['message_count'].values[0]} != hand {hand_count}"
        )
        assert abs(row["vader_compound_mean"].values[0] - hand_mean) < 1e-10, (
            f"{session_id}/seg{segment_num}/g{group_id}: "
            f"mean {row['vader_compound_mean'].values[0]} != hand {hand_mean}"
        )
