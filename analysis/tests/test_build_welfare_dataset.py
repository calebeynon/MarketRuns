"""
Purpose: Tests for build_welfare_dataset.py (cross-validation and merge logic)
Author: Claude Code
Date: 2026-04-05
"""

from pathlib import Path

import pandas as pd
import pytest

from analysis.derived.build_welfare_dataset import (
    DERIVED,
    MERGE_KEYS_EMOTIONS,
    MERGE_KEYS_GRT,
    merge_welfare_into_emotions,
)

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WELFARE_CSV = PROJECT_ROOT / "datastore" / "derived" / "group_round_welfare.csv"
EMOTIONS_CSV = PROJECT_ROOT / "datastore" / "derived" / "emotions_traits_selling_dataset.csv"
GRT_CSV = PROJECT_ROOT / "datastore" / "derived" / "group_round_timing.csv"

DATASTORE_AVAILABLE = DERIVED.exists() and WELFARE_CSV.exists()
skip_no_datastore = pytest.mark.skipif(
    not DATASTORE_AVAILABLE, reason="Datastore not accessible"
)


# =====
# Unit tests: merge logic with synthetic data
# =====
def _make_synthetic_emotions():
    """Create a small synthetic emotions DataFrame."""
    return pd.DataFrame({
        "session_id": ["s1"] * 4,
        "segment": [1, 1, 2, 2],
        "round": [1, 1, 1, 1],
        "group_id": [1, 1, 1, 1],
        "player": [1, 2, 1, 2],
        "sold": [0, 1, 1, 0],
    })


def _make_synthetic_welfare():
    """Create matching synthetic welfare DataFrame."""
    return pd.DataFrame({
        "session": ["s1", "s1"],
        "segment_num": [1, 2],
        "round_num": [1, 1],
        "group_id": [1, 1],
        "welfare": [1.0, 0.85],
    })


def test_merge_welfare_adds_column():
    """Welfare column is added to the emotions DataFrame."""
    emotions = _make_synthetic_emotions()
    welfare = _make_synthetic_welfare()
    merged = merge_welfare_into_emotions(emotions, welfare)
    assert "welfare" in merged.columns


def test_merge_welfare_preserves_row_count():
    """Row count is unchanged after merge."""
    emotions = _make_synthetic_emotions()
    welfare = _make_synthetic_welfare()
    merged = merge_welfare_into_emotions(emotions, welfare)
    assert len(merged) == len(emotions)


def test_merge_welfare_no_nans():
    """No NaN welfare values after merge."""
    emotions = _make_synthetic_emotions()
    welfare = _make_synthetic_welfare()
    merged = merge_welfare_into_emotions(emotions, welfare)
    assert merged["welfare"].notna().all()


def test_merge_welfare_values_correct():
    """Merged welfare values match the source data."""
    emotions = _make_synthetic_emotions()
    welfare = _make_synthetic_welfare()
    merged = merge_welfare_into_emotions(emotions, welfare)
    seg1 = merged[merged["segment"] == 1]["welfare"].unique()
    seg2 = merged[merged["segment"] == 2]["welfare"].unique()
    assert seg1 == [1.0]
    assert seg2 == [0.85]


def test_merge_idempotent():
    """Re-merging drops existing welfare before adding new one."""
    emotions = _make_synthetic_emotions()
    emotions["welfare"] = 0.5
    welfare = _make_synthetic_welfare()
    merged = merge_welfare_into_emotions(emotions, welfare)
    assert len(merged.columns) == len(_make_synthetic_emotions().columns) + 1


# =====
# Integration tests: group_round_welfare.csv
# =====
@skip_no_datastore
def test_welfare_csv_has_720_rows():
    """group_round_welfare.csv has 720 rows (6 sessions x 4 segments x 30 rounds)."""
    df = pd.read_csv(WELFARE_CSV)
    assert len(df) == 720


@skip_no_datastore
def test_welfare_csv_columns():
    """Welfare CSV has the 4 merge keys plus welfare."""
    df = pd.read_csv(WELFARE_CSV)
    expected = set(MERGE_KEYS_GRT + ["welfare"])
    assert set(df.columns) == expected


@skip_no_datastore
def test_welfare_csv_no_nans():
    """No missing values in welfare CSV."""
    df = pd.read_csv(WELFARE_CSV)
    assert df.notna().all().all()


# =====
# Integration tests: emotions_traits_selling_dataset.csv
# =====
@skip_no_datastore
def test_emotions_dataset_has_35_columns():
    """Updated emotions dataset has 35 columns (34 original + welfare)."""
    df = pd.read_csv(EMOTIONS_CSV)
    assert len(df.columns) == 35


@skip_no_datastore
def test_emotions_dataset_has_16128_rows():
    """Emotions dataset row count unchanged at 16128."""
    df = pd.read_csv(EMOTIONS_CSV)
    assert len(df) == 16128


@skip_no_datastore
def test_welfare_all_state_0_are_one():
    """In the emotions dataset, all state=0 rows have welfare=1.0."""
    df = pd.read_csv(EMOTIONS_CSV)
    s0 = df[df["state"] == 0]
    assert (s0["welfare"] == 1.0).all()


@skip_no_datastore
def test_welfare_no_nans_in_emotions():
    """No NaN welfare values in the emotions dataset."""
    df = pd.read_csv(EMOTIONS_CSV)
    assert df["welfare"].notna().all()


@skip_no_datastore
def test_welfare_cross_validation():
    """Replicate cross-validation: state=1 actual earnings == welfare * 80."""
    extended = pd.read_csv(DERIVED / "individual_period_dataset_extended.csv")
    grt = pd.read_csv(GRT_CSV)

    dedup = extended.sort_values("period").drop_duplicates(
        subset=["session_id", "segment", "round", "group_id", "player"],
        keep="last",
    )
    actual = (
        dedup.groupby(MERGE_KEYS_EMOTIONS)["round_payoff"]
        .sum()
        .reset_index()
        .rename(columns={"round_payoff": "actual_total"})
    )

    welfare = grt[["session", "segment_num", "round_num", "group_id",
                    "welfare", "state"]].copy()
    welfare.rename(
        columns={"session": "session_id", "segment_num": "segment",
                 "round_num": "round"},
        inplace=True,
    )
    merged = welfare.merge(actual, on=MERGE_KEYS_EMOTIONS, how="inner")

    s1 = merged[merged["state"] == 1]
    expected = s1["welfare"] * 80
    mismatches = s1[s1["actual_total"] != expected]
    assert len(mismatches) == 0, (
        f"{len(mismatches)} state=1 rows with welfare mismatch"
    )

    s0 = merged[merged["state"] == 0]
    assert (s0["welfare"] == 1.0).all()


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
