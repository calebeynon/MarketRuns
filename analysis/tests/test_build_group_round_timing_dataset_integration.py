"""
Purpose: Integration tests for build_group_round_timing_dataset.py (requires datastore)
Author: Claude Code
Date: 2025-01-18
"""

from pathlib import Path

import pandas as pd
import pytest

from analysis.derived.build_group_round_timing_dataset import (
    DATASTORE,
    SEGMENTS,
    SESSIONS,
    load_segment_data,
)

# =====
# File paths and test configuration
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DERIVED_CSV = PROJECT_ROOT / "datastore" / "derived" / "group_round_timing.csv"

# Rounds per segment (from oTree PERIODS_PER_ROUND lists)
ROUNDS_PER_SEGMENT = {1: 10, 2: 5, 3: 6, 4: 9}

DATASTORE_AVAILABLE = DATASTORE.exists() and DERIVED_CSV.exists()
skip_no_datastore = pytest.mark.skipif(
    not DATASTORE_AVAILABLE, reason="Datastore not accessible"
)


# =====
# Test helpers
# =====
def _load_derived():
    """Load the derived group_round_timing dataset."""
    return pd.read_csv(DERIVED_CSV)


def _count_sellers_in_raw(session, segment, group_id, round_num):
    """Count unique players with sold==1 in a raw oTree group-round."""
    session_folder = DATASTORE / session
    raw = load_segment_data(session_folder, segment)
    grp = raw[
        (raw["group.id_in_subsession"] == group_id)
        & (raw["player.round_number_in_segment"] == round_num)
    ]
    return grp.groupby("player.id_in_group")["player.sold"].max().sum()


def _get_state_from_raw(session, segment, group_id, round_num):
    """Get the state value from a raw oTree group-round."""
    session_folder = DATASTORE / session
    raw = load_segment_data(session_folder, segment)
    mask = (
        (raw["group.id_in_subsession"] == group_id)
        & (raw["player.round_number_in_segment"] == round_num)
    )
    return int(raw.loc[mask, "player.state"].iloc[0])


def _get_derived_row(df, session, seg_num, group_id, round_num):
    """Look up a single row from the derived dataset."""
    mask = (
        (df["session"] == session)
        & (df["segment_num"] == seg_num)
        & (df["group_id"] == group_id)
        & (df["round_num"] == round_num)
    )
    return df.loc[mask].iloc[0]


# =====
# Row count
# =====
@skip_no_datastore
def test_row_count_is_720():
    """6 sessions x 4 segments x variable rounds x 4 groups = 720 rows."""
    df = _load_derived()
    assert len(df) == 720


# =====
# n_sellers accuracy against raw data
# =====
@skip_no_datastore
def test_n_sellers_matches_raw_seg1():
    """n_sellers matches count of sold players in raw data for segment 1."""
    df = _load_derived()
    row = df[
        (df["session"] == "1_11-7-tr1")
        & (df["segment_num"] == 1)
        & (df["group_id"] == 1)
        & (df["round_num"] == 2)
    ].iloc[0]
    raw_count = _count_sellers_in_raw("1_11-7-tr1", "chat_noavg", 1, 2)
    assert row["n_sellers"] == raw_count


@skip_no_datastore
def test_n_sellers_matches_raw_seg2():
    """n_sellers matches raw data for segment 2."""
    df = _load_derived()
    row = df[
        (df["session"] == "2_11-10-tr2")
        & (df["segment_num"] == 2)
        & (df["group_id"] == 2)
        & (df["round_num"] == 1)
    ].iloc[0]
    raw_count = _count_sellers_in_raw("2_11-10-tr2", "chat_noavg2", 2, 1)
    assert row["n_sellers"] == raw_count


@skip_no_datastore
def test_n_sellers_matches_raw_seg3():
    """n_sellers matches raw data for segment 3."""
    df = _load_derived()
    row = df[
        (df["session"] == "3_11-11-tr2")
        & (df["segment_num"] == 3)
        & (df["group_id"] == 3)
        & (df["round_num"] == 4)
    ].iloc[0]
    raw_count = _count_sellers_in_raw("3_11-11-tr2", "chat_noavg3", 3, 4)
    assert row["n_sellers"] == raw_count


@skip_no_datastore
def test_n_sellers_matches_raw_seg4():
    """n_sellers matches raw data for segment 4."""
    df = _load_derived()
    row = df[
        (df["session"] == "4_11-12-tr1")
        & (df["segment_num"] == 4)
        & (df["group_id"] == 4)
        & (df["round_num"] == 7)
    ].iloc[0]
    raw_count = _count_sellers_in_raw("4_11-12-tr1", "chat_noavg4", 4, 7)
    assert row["n_sellers"] == raw_count


# =====
# State accuracy against raw data
# =====
@skip_no_datastore
@pytest.mark.parametrize("session,segment,seg_num,group_id,round_num", [
    ("1_11-7-tr1", "chat_noavg", 1, 1, 3),
    ("2_11-10-tr2", "chat_noavg2", 2, 2, 3),
    ("5_11-14-tr2", "chat_noavg3", 3, 1, 2),
    ("6_11-18-tr1", "chat_noavg4", 4, 3, 5),
])
def test_state_matches_raw_data(session, segment, seg_num, group_id, round_num):
    """State in derived CSV matches player.state from raw oTree exports."""
    df = _load_derived()
    row = _get_derived_row(df, session, seg_num, group_id, round_num)
    raw_state = _get_state_from_raw(session, segment, group_id, round_num)
    assert row["state"] == raw_state


# =====
# Treatment accuracy
# =====
@skip_no_datastore
def test_treatment_matches_session_mapping():
    """Treatment values match the SESSIONS dict from the builder script."""
    df = _load_derived()
    for session_name, expected_treatment in SESSIONS.items():
        session_rows = df[df["session"] == session_name]
        assert len(session_rows) > 0, f"No rows for session {session_name}"
        actual = session_rows["treatment"].unique()
        assert list(actual) == [expected_treatment], (
            f"Treatment mismatch for {session_name}: "
            f"expected {expected_treatment}, got {actual}"
        )


# =====
# Segment and round ranges
# =====
@skip_no_datastore
def test_segment_num_values():
    """segment_num contains only {1, 2, 3, 4}."""
    df = _load_derived()
    assert set(df["segment_num"].unique()) == {1, 2, 3, 4}


@skip_no_datastore
def test_round_num_ranges_per_segment():
    """Each segment has the correct number of rounds."""
    df = _load_derived()
    for seg_num, expected_max_round in ROUNDS_PER_SEGMENT.items():
        seg_df = df[df["segment_num"] == seg_num]
        assert seg_df["round_num"].min() == 1, (
            f"Segment {seg_num} rounds should start at 1"
        )
        assert seg_df["round_num"].max() == expected_max_round, (
            f"Segment {seg_num} should have max round {expected_max_round}, "
            f"got {seg_df['round_num'].max()}"
        )
        assert seg_df["round_num"].nunique() == expected_max_round, (
            f"Segment {seg_num} should have {expected_max_round} unique rounds"
        )


# =====
# Global group ID uniqueness
# =====
@skip_no_datastore
def test_global_group_id_has_96_unique_values():
    """96 unique global_group_ids (6 sessions x 4 segments x 4 groups)."""
    df = _load_derived()
    assert df["global_group_id"].nunique() == 96


@skip_no_datastore
def test_no_duplicate_group_round_pairs():
    """Each (global_group_id, round_num) pair appears exactly once."""
    df = _load_derived()
    duplicates = df.duplicated(subset=["global_group_id", "round_num"])
    assert not duplicates.any(), (
        f"Found {duplicates.sum()} duplicate (global_group_id, round_num) pairs"
    )


# =====
# n_sellers bounds
# =====
@skip_no_datastore
def test_n_sellers_bounded_0_to_4():
    """All n_sellers values are in [0, 4]."""
    df = _load_derived()
    assert df["n_sellers"].min() >= 0
    assert df["n_sellers"].max() <= 4


@skip_no_datastore
def test_n_sellers_no_nulls():
    """n_sellers has no missing values."""
    df = _load_derived()
    assert df["n_sellers"].notna().all()


# =====
# No missing values in key columns
# =====
@skip_no_datastore
def test_no_missing_values_in_tobit_columns():
    """The 6 columns used by the tobit regression have no NAs."""
    df = _load_derived()
    tobit_cols = [
        "n_sellers", "state", "treatment",
        "segment_num", "round_num", "global_group_id",
    ]
    for col in tobit_cols:
        assert col in df.columns, f"Column {col} not found"
        assert df[col].notna().all(), f"Column {col} has missing values"


# =====
# Welfare column in group_round_timing.csv
# =====
@skip_no_datastore
def test_welfare_column_present():
    """welfare column exists in the derived CSV."""
    df = _load_derived()
    assert "welfare" in df.columns


@skip_no_datastore
def test_welfare_values_in_range():
    """All welfare values are between 0.25 and 1.0."""
    df = _load_derived()
    assert df["welfare"].min() >= 0.25
    assert df["welfare"].max() <= 1.0


@skip_no_datastore
def test_welfare_no_nans():
    """welfare column has no missing values."""
    df = _load_derived()
    assert df["welfare"].notna().all()


@skip_no_datastore
def test_welfare_state_0_all_one():
    """All state=0 rows have welfare=1.0."""
    df = _load_derived()
    s0 = df[df["state"] == 0]
    assert (s0["welfare"] == 1.0).all()


@skip_no_datastore
def test_welfare_matches_n_sellers():
    """For state=1, welfare matches the expected lookup for each n_sellers."""
    df = _load_derived()
    expected = {0: 1.0, 1: 0.85, 2: 0.675, 3: 0.475, 4: 0.25}
    s1 = df[df["state"] == 1]
    for n, w in expected.items():
        subset = s1[s1["n_sellers"] == n]
        if len(subset) > 0:
            assert subset["welfare"].unique() == pytest.approx([w]), (
                f"n_sellers={n}: expected welfare={w}, "
                f"got {subset['welfare'].unique()}"
            )


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
