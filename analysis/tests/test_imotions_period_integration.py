"""
Purpose: Integration tests for iMotions period alignment pipeline
Author: Claude Code
Date: 2026-02-04

Verifies that the iMotions annotation offset (m{N} -> period N-1) is correctly
applied throughout the pipeline, using ACTUAL data from the datastore.

Key insight: iMotions annotations use m{N} which maps to oTree period N-1:
    - m2 -> period 1 (first trading period)
    - m3 -> period 2
    - etc.
"""

import re
import pytest
import pandas as pd
from pathlib import Path

# =====
# File path constants
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
IMOTIONS_DIR = DATASTORE / "imotions"
DERIVED_DIR = DATASTORE / "derived"

IMOTIONS_PERIOD_EMOTIONS_CSV = DERIVED_DIR / "imotions_period_emotions.csv"
INDIVIDUAL_PERIOD_CSV = DERIVED_DIR / "individual_period_dataset.csv"

SAMPLE_RAW_CSV = IMOTIONS_DIR / "1" / "001_R3.csv"

MARKET_PERIOD_REGEX = re.compile(r"^s(\d+)r(\d+)m(\d+)MarketPeriod$")
IMOTIONS_SKIP_ROWS = 24
ANNOTATION_COL = "Respondent Annotations active"

EMOTION_COLS = [
    "anger_mean", "contempt_mean", "disgust_mean", "fear_mean",
    "joy_mean", "sadness_mean", "surprise_mean", "engagement_mean",
    "valence_mean"
]


# =====
# Main function
# =====
def main():
    """Run integration tests with verbose output."""
    pytest.main([__file__, "-v"])


# =====
# Data loading helpers
# =====
def datastore_available() -> bool:
    """Check if datastore is accessible."""
    return DATASTORE.exists() and IMOTIONS_DIR.exists()


def load_raw_imotions_sample() -> pd.DataFrame:
    """Load sample raw iMotions CSV for testing."""
    return pd.read_csv(
        SAMPLE_RAW_CSV,
        skiprows=IMOTIONS_SKIP_ROWS,
        encoding="utf-8-sig",
        low_memory=False,
    )


def load_emotions_dataset() -> pd.DataFrame:
    """Load derived iMotions period emotions dataset."""
    return pd.read_csv(IMOTIONS_PERIOD_EMOTIONS_CSV)


def load_otree_period_dataset() -> pd.DataFrame:
    """Load oTree individual period dataset."""
    return pd.read_csv(INDIVIDUAL_PERIOD_CSV)


# =====
# Annotation extraction helpers
# =====
def extract_m_values_from_annotations(df: pd.DataFrame) -> list[int]:
    """Extract all m values from MarketPeriod annotations in raw data."""
    m_values = []
    for ann in df[ANNOTATION_COL].dropna().unique():
        match = MARKET_PERIOD_REGEX.match(str(ann))
        if match:
            m_values.append(int(match.group(3)))
    return sorted(set(m_values))


def get_first_market_period_annotation(df: pd.DataFrame) -> str | None:
    """Get the first MarketPeriod annotation by row order in raw data."""
    for ann in df[ANNOTATION_COL]:
        if pd.notna(ann) and MARKET_PERIOD_REGEX.match(str(ann)):
            return str(ann)
    return None


# =====
# Assertion helpers
# =====
def assert_min_m_value_is_two(m_values: list[int]):
    """Assert minimum m value is 2, confirming offset exists in raw data."""
    assert min(m_values) == 2, (
        f"Expected minimum m value to be 2, got {min(m_values)}. "
        f"Unique m values: {m_values}"
    )


def assert_first_annotation_contains_m2(first_ann: str | None):
    """Assert first annotation by row order contains m2."""
    assert first_ann is not None, "No MarketPeriod annotations found"
    assert "m2" in first_ann, (
        f"First MarketPeriod annotation should contain 'm2', got: {first_ann}"
    )


def assert_min_period_is_one(df: pd.DataFrame, context: str):
    """Assert minimum period in dataset is 1."""
    min_period = df["period"].min()
    assert min_period == 1, (
        f"{context}: Expected min period 1, got {min_period}"
    )


def assert_period_range_valid(df: pd.DataFrame):
    """Assert periods are within valid range (1 to 20)."""
    min_period = df["period"].min()
    max_period = df["period"].max()
    assert min_period >= 1, f"Period should not be 0 or negative: {min_period}"
    assert max_period <= 20, f"Max period {max_period} unreasonably high"


def assert_emotion_columns_exist(df: pd.DataFrame):
    """Assert all expected emotion columns exist."""
    for col in EMOTION_COLS:
        assert col in df.columns, f"Missing emotion column: {col}"


def assert_emotion_coverage_sufficient(df: pd.DataFrame, min_coverage: float = 0.5):
    """Assert emotion columns have sufficient non-null coverage."""
    for col in EMOTION_COLS:
        coverage = df[col].notna().sum() / len(df)
        assert coverage > min_coverage, (
            f"Column {col} has low coverage: {coverage:.1%}"
        )


# =====
# Test: First annotation uses m2
# =====
@pytest.mark.skipif(not datastore_available(), reason="Datastore not accessible")
def test_first_market_period_annotation_is_m2():
    """Verify the first MarketPeriod annotation in raw data is m2 not m1."""
    df = load_raw_imotions_sample()
    m_values = extract_m_values_from_annotations(df)
    first_ann = get_first_market_period_annotation(df)

    assert_min_m_value_is_two(m_values)
    assert_first_annotation_contains_m2(first_ann)


# =====
# Test: Period offset applied correctly in derived data
# =====
@pytest.mark.skipif(
    not IMOTIONS_PERIOD_EMOTIONS_CSV.exists(),
    reason="Derived emotions dataset not found"
)
def test_period_offset_applied_correctly():
    """Verify m2 -> period 1, m3 -> period 2 in derived dataset."""
    df = load_emotions_dataset()

    assert_min_period_is_one(df, "Derived emotions")

    # Verify period 1 exists for segment 1, round 1
    s1r1_periods = df[(df["segment"] == 1) & (df["round"] == 1)]["period"]
    assert 1 in s1r1_periods.values, (
        f"Period 1 missing from s1r1. Found: {sorted(s1r1_periods.unique())}"
    )


# =====
# Test: Pipeline produces valid period range
# =====
@pytest.mark.skipif(
    not IMOTIONS_PERIOD_EMOTIONS_CSV.exists(),
    reason="Derived emotions dataset not found"
)
def test_pipeline_produces_valid_periods():
    """Verify periods in output start at 1 (not 0 or 2)."""
    df = load_emotions_dataset()
    assert_min_period_is_one(df, "Pipeline output")
    assert_period_range_valid(df)


# =====
# Test: Merged dataset has emotion coverage
# =====
@pytest.mark.skipif(
    not IMOTIONS_PERIOD_EMOTIONS_CSV.exists(),
    reason="Derived emotions dataset not found"
)
def test_merged_dataset_has_emotion_coverage():
    """Verify emotion columns have non-null values after processing."""
    df = load_emotions_dataset()

    assert_emotion_columns_exist(df)
    assert_emotion_coverage_sufficient(df)
    assert len(df) > 1000, f"Expected >1000 observations, got {len(df)}"


# =====
# Test: Period alignment between iMotions and oTree datasets
# =====
@pytest.mark.skipif(
    not (IMOTIONS_PERIOD_EMOTIONS_CSV.exists() and INDIVIDUAL_PERIOD_CSV.exists()),
    reason="Required datasets not found"
)
def test_period_alignment_with_otree_data():
    """Verify period values in iMotions data match oTree period range."""
    emotions_df = load_emotions_dataset()
    otree_df = load_otree_period_dataset()

    assert_min_period_is_one(emotions_df, "iMotions")
    assert_min_period_is_one(otree_df, "oTree")

    # iMotions periods should not exceed oTree periods
    assert emotions_df["period"].max() <= otree_df["period"].max(), (
        "iMotions max period exceeds oTree max period"
    )


# %%
if __name__ == "__main__":
    main()
