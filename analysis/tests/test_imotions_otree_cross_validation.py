"""
Purpose: Cross-validation tests between iMotions and oTree period data
Author: Claude Code
Date: 2026-02-04

Validates that iMotions annotations (m{N} -> period N-1) align correctly
with oTree period data. Uses ACTUAL CSV files from the datastore.

Key insight: iMotions uses m{N} which maps to oTree period N-1:
    - m2 -> period 1 (first trading period)
    - m3 -> period 2
    - etc.
"""

import pytest
import pandas as pd
from pathlib import Path

# =====
# File path constants
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
DERIVED_DIR = DATASTORE / "derived"

IMOTIONS_PERIOD_EMOTIONS_CSV = DERIVED_DIR / "imotions_period_emotions.csv"
INDIVIDUAL_PERIOD_CSV = DERIVED_DIR / "individual_period_dataset.csv"

MERGE_KEYS = ["session_id", "segment", "round", "period", "player"]


# =====
# Main function
# =====
def main():
    """Run cross-validation tests with verbose output."""
    pytest.main([__file__, "-v"])


# =====
# Data availability check
# =====
def datasets_available() -> bool:
    """Check if both required datasets exist."""
    return (
        IMOTIONS_PERIOD_EMOTIONS_CSV.exists()
        and INDIVIDUAL_PERIOD_CSV.exists()
    )


# =====
# Data loading helpers
# =====
def load_imotions_data() -> pd.DataFrame:
    """Load iMotions period emotions dataset."""
    return pd.read_csv(IMOTIONS_PERIOD_EMOTIONS_CSV)


def load_otree_data() -> pd.DataFrame:
    """Load oTree individual period dataset."""
    return pd.read_csv(INDIVIDUAL_PERIOD_CSV)


# =====
# Period count comparison helpers
# =====
def get_period_count_by_round(df: pd.DataFrame) -> pd.DataFrame:
    """Get unique period count per session/segment/round."""
    return (
        df.groupby(["session_id", "segment", "round"])["period"]
        .nunique()
        .reset_index()
        .rename(columns={"period": "period_count"})
    )


def filter_to_common_rounds(
    imotions_counts: pd.DataFrame,
    otree_counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter both datasets to rounds present in both."""
    keys = ["session_id", "segment", "round"]
    merged = imotions_counts.merge(
        otree_counts,
        on=keys,
        suffixes=("_imotions", "_otree"),
        how="inner",
    )
    return merged


# =====
# Assertion helpers
# =====
def assert_period_counts_match(merged_df: pd.DataFrame):
    """Assert iMotions period count <= oTree period count for each round."""
    mismatches = merged_df[
        merged_df["period_count_imotions"] > merged_df["period_count_otree"]
    ]
    assert len(mismatches) == 0, (
        f"Found {len(mismatches)} rounds where iMotions has more periods than "
        f"oTree:\n{mismatches.to_string()}"
    )


def assert_periods_start_at_one(df: pd.DataFrame, source: str):
    """Assert minimum period value is 1."""
    min_period = df["period"].min()
    assert min_period == 1, (
        f"{source}: Expected periods to start at 1, got {min_period}"
    )


def get_max_period_by_round(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Get max period per session/segment/round."""
    return (
        df.groupby(["session_id", "segment", "round"])["period"]
        .max()
        .reset_index()
        .rename(columns={"period": col_name})
    )


def assert_max_periods_within_otree(
    imotions_df: pd.DataFrame,
    otree_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assert max period in iMotions <= max period in oTree per round."""
    imotions_max = get_max_period_by_round(imotions_df, "max_period_imotions")
    otree_max = get_max_period_by_round(otree_df, "max_period_otree")
    merged = imotions_max.merge(otree_max, on=["session_id", "segment", "round"])
    violations = merged[merged["max_period_imotions"] > merged["max_period_otree"]]
    assert len(violations) == 0, (
        f"iMotions max period exceeds oTree in {len(violations)} rounds:\n"
        f"{violations.to_string()}"
    )
    return merged


def filter_participant(df: pd.DataFrame, session: str, player: str) -> pd.DataFrame:
    """Filter dataframe to a specific session and player."""
    return df[(df["session_id"] == session) & (df["player"] == player)]


def get_periods_for_round(
    df: pd.DataFrame, session: str, player: str, segment: int, round_num: int
) -> set:
    """Get unique periods for a specific participant and round."""
    mask = (
        (df["session_id"] == session)
        & (df["player"] == player)
        & (df["segment"] == segment)
        & (df["round"] == round_num)
    )
    return set(df[mask]["period"])


def assert_participant_exists_in_both(
    imotions_df: pd.DataFrame,
    otree_df: pd.DataFrame,
    session: str,
    player: str,
):
    """Assert a specific participant has data in both datasets."""
    imotions_count = len(filter_participant(imotions_df, session, player))
    otree_count = len(filter_participant(otree_df, session, player))
    assert imotions_count > 0, f"Player {player} in {session} not found in iMotions"
    assert otree_count > 0, f"Player {player} in {session} not found in oTree"


# =====
# Test: Period count matches between datasets
# =====
@pytest.mark.skipif(not datasets_available(), reason="Datasets not accessible")
def test_period_count_matches_otree():
    """For each session/segment/round, iMotions period count <= oTree count.

    iMotions may have fewer periods if some frames were missing or excluded,
    but should never have MORE periods than oTree.
    """
    imotions_df = load_imotions_data()
    otree_df = load_otree_data()

    imotions_counts = get_period_count_by_round(imotions_df)
    otree_counts = get_period_count_by_round(otree_df)

    merged = filter_to_common_rounds(imotions_counts, otree_counts)

    # Must have some rounds in common
    assert len(merged) > 0, "No common rounds found between datasets"

    assert_period_counts_match(merged)


# =====
# Test: Periods start at 1 in both datasets
# =====
@pytest.mark.skipif(not datasets_available(), reason="Datasets not accessible")
def test_period_range_starts_at_one():
    """Both iMotions and oTree periods should start at 1, not 0 or 2.

    This confirms the m{N} -> period N-1 offset is correctly applied.
    """
    imotions_df = load_imotions_data()
    otree_df = load_otree_data()

    assert_periods_start_at_one(imotions_df, "iMotions")
    assert_periods_start_at_one(otree_df, "oTree")


# =====
# Test: Maximum periods match within tolerance
# =====
@pytest.mark.skipif(not datasets_available(), reason="Datasets not accessible")
def test_max_period_matches():
    """Maximum period in iMotions should not exceed oTree for same round.

    Rounds in oTree define the ground truth for how many periods existed.
    """
    imotions_df = load_imotions_data()
    otree_df = load_otree_data()

    merged = assert_max_periods_within_otree(imotions_df, otree_df)

    # Verify we tested a reasonable number of rounds
    assert len(merged) > 50, f"Only {len(merged)} rounds compared"


# =====
# Test: Specific participant period alignment
# =====
@pytest.mark.skipif(not datasets_available(), reason="Datasets not accessible")
def test_specific_participant_period_alignment():
    """Cross-check specific participant data between datasets."""
    imotions_df = load_imotions_data()
    otree_df = load_otree_data()

    # Test with participant "R" from session 1 (seen in raw data sample)
    test_session, test_player = "1_11-7-tr1", "R"
    assert_participant_exists_in_both(imotions_df, otree_df, test_session, test_player)

    # Get periods for segment 1, round 1
    imotions_periods = get_periods_for_round(imotions_df, test_session, test_player, 1, 1)
    otree_periods = get_periods_for_round(otree_df, test_session, test_player, 1, 1)

    # iMotions periods should be subset of oTree periods
    extra_periods = imotions_periods - otree_periods
    assert len(extra_periods) == 0, f"iMotions has periods not in oTree: {extra_periods}"


# =====
# Test: Merge keys produce valid join
# =====
def merge_datasets_on_keys(imotions_df: pd.DataFrame, otree_df: pd.DataFrame) -> pd.DataFrame:
    """Merge iMotions and oTree data on MERGE_KEYS."""
    return imotions_df.merge(
        otree_df[MERGE_KEYS + ["signal", "sold"]],
        on=MERGE_KEYS,
        how="inner",
    )


@pytest.mark.skipif(not datasets_available(), reason="Datasets not accessible")
def test_merge_keys_produce_valid_join():
    """Verify merge on MERGE_KEYS produces non-empty result."""
    imotions_df = load_imotions_data()
    otree_df = load_otree_data()

    merged = merge_datasets_on_keys(imotions_df, otree_df)

    # Should have substantial overlap
    assert len(merged) > 1000, f"Expected >1000 merged rows, got {len(merged)}"
    assert merged["period"].min() >= 1, "Merged data has invalid period < 1"


# %%
if __name__ == "__main__":
    main()
