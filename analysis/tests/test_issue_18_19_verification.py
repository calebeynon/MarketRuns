"""
Purpose: Verification tests for Issues #18 and #19 data pipelines
Author: Claude Code
Date: 2026-02-04

Verifies that the datasets used by Issues #18 and #19 have correct period
alignment following the iMotions annotation offset fix (m{N} -> period N-1).

Issue #18: emotions_traits_selling_dataset.csv (emotion + trait + selling analysis)
Issue #19: first_seller_analysis_data.csv (first seller analysis with traits)
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

ISSUE_18_DATASET = DERIVED_DIR / "emotions_traits_selling_dataset.csv"
ISSUE_19_DATASET = DERIVED_DIR / "first_seller_analysis_data.csv"

EMOTION_COLS = [
    "anger_mean", "contempt_mean", "disgust_mean", "fear_mean",
    "joy_mean", "sadness_mean", "surprise_mean", "engagement_mean",
    "valence_mean"
]

TRAIT_COLS = [
    "extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness", "impulsivity", "state_anxiety"
]

MERGE_KEY_COLS = ["session_id", "segment", "round", "period", "player"]


# =====
# Main function
# =====
def main():
    """Run verification tests with verbose output."""
    pytest.main([__file__, "-v"])


# =====
# Data availability helpers
# =====
def issue_18_data_available() -> bool:
    """Check if Issue #18 dataset exists."""
    return ISSUE_18_DATASET.exists()


def issue_19_data_available() -> bool:
    """Check if Issue #19 dataset exists."""
    return ISSUE_19_DATASET.exists()


def load_issue_18_dataset() -> pd.DataFrame:
    """Load the Issue #18 emotions traits selling dataset."""
    return pd.read_csv(ISSUE_18_DATASET)


def load_issue_19_dataset() -> pd.DataFrame:
    """Load the Issue #19 first seller analysis dataset."""
    return pd.read_csv(ISSUE_19_DATASET)


# =====
# Issue #18 Tests: Emotion columns exist
# =====
@pytest.mark.skipif(not issue_18_data_available(), reason="Issue #18 dataset not found")
def test_issue_18_emotion_columns_exist():
    """Verify all emotion columns exist in Issue #18 dataset."""
    df = load_issue_18_dataset()

    missing_cols = [col for col in EMOTION_COLS if col not in df.columns]
    assert not missing_cols, f"Missing emotion columns: {missing_cols}"


# =====
# Issue #18 Tests: Emotion data coverage
# =====
@pytest.mark.skipif(not issue_18_data_available(), reason="Issue #18 dataset not found")
def test_issue_18_emotion_data_coverage():
    """Verify emotion columns have significant non-null coverage (>50%)."""
    df = load_issue_18_dataset()
    min_coverage = 0.5

    for col in EMOTION_COLS:
        if col not in df.columns:
            pytest.skip(f"Column {col} not in dataset")
        coverage = df[col].notna().sum() / len(df)
        assert coverage > min_coverage, (
            f"Column '{col}' has insufficient coverage: {coverage:.1%} < 50%"
        )


# =====
# Issue #18 Tests: Period in valid range
# =====
@pytest.mark.skipif(not issue_18_data_available(), reason="Issue #18 dataset not found")
def test_issue_18_period_in_valid_range():
    """Verify period values start at 1, not 0 or 2."""
    df = load_issue_18_dataset()

    min_period = df["period"].min()
    assert min_period == 1, (
        f"Issue #18: Min period should be 1, got {min_period}. "
        f"Period < 1 suggests offset error."
    )

    # Verify no zero or negative periods exist
    invalid_periods = df[df["period"] < 1]
    assert len(invalid_periods) == 0, (
        f"Found {len(invalid_periods)} rows with period < 1"
    )


# =====
# Issue #18 Tests: Merge keys present
# =====
@pytest.mark.skipif(not issue_18_data_available(), reason="Issue #18 dataset not found")
def test_issue_18_merge_keys_present():
    """Verify session_id, segment, round, period, player columns exist."""
    df = load_issue_18_dataset()

    missing_keys = [col for col in MERGE_KEY_COLS if col not in df.columns]
    assert not missing_keys, (
        f"Issue #18: Missing merge key columns: {missing_keys}"
    )


# =====
# Issue #19 Tests: First sale period range
# =====
@pytest.mark.skipif(not issue_19_data_available(), reason="Issue #19 dataset not found")
def test_issue_19_first_sale_period_range():
    """Verify first_sale_period starts at 1, not 2."""
    df = load_issue_19_dataset()

    # Filter to rows where first_sale_period is not null
    valid_sales = df[df["first_sale_period"].notna()]
    if len(valid_sales) == 0:
        pytest.skip("No valid first_sale_period values found")

    min_first_sale = valid_sales["first_sale_period"].min()
    assert min_first_sale >= 1, (
        f"Issue #19: Min first_sale_period should be >= 1, got {min_first_sale}"
    )

    # Verify period 1 sales exist (confirms offset was applied)
    period_1_sales = valid_sales[valid_sales["first_sale_period"] == 1]
    assert len(period_1_sales) > 0, (
        "Issue #19: No first_sale_period=1 found. "
        "This suggests offset may not be applied correctly."
    )


# =====
# Issue #19 Tests: Trait columns populated
# =====
@pytest.mark.skipif(not issue_19_data_available(), reason="Issue #19 dataset not found")
def test_issue_19_trait_columns_populated():
    """Verify trait columns have non-null values."""
    df = load_issue_19_dataset()

    for col in TRAIT_COLS:
        if col not in df.columns:
            pytest.fail(f"Issue #19: Missing trait column '{col}'")
        non_null_count = df[col].notna().sum()
        assert non_null_count > 0, (
            f"Issue #19: Trait column '{col}' has all null values"
        )


# =====
# Issue #19 Tests: Period values valid
# =====
@pytest.mark.skipif(not issue_19_data_available(), reason="Issue #19 dataset not found")
def test_issue_19_period_values_valid():
    """Verify no period values < 1 in the dataset."""
    df = load_issue_19_dataset()

    # Check round column if it exists (main grouping)
    if "round" in df.columns:
        min_round = df["round"].min()
        assert min_round >= 1, (
            f"Issue #19: Min round should be >= 1, got {min_round}"
        )

    # Check first_sale_period for invalid values
    valid_sales = df[df["first_sale_period"].notna()]
    if len(valid_sales) > 0:
        invalid_periods = valid_sales[valid_sales["first_sale_period"] < 1]
        assert len(invalid_periods) == 0, (
            f"Issue #19: Found {len(invalid_periods)} rows with "
            f"first_sale_period < 1"
        )


# %%
if __name__ == "__main__":
    main()
