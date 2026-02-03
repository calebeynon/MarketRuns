"""
Purpose: Unit tests for build_holdout_anger_dataset.py
Author: Claude
Date: 2026-02-02

Tests verify the holdout anger analysis dataset builder that:
- Filters to sessions with iMotions data
- Adds chat_available indicator
- Extracts anger during Results phase from iMotions data
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from analysis.derived.build_holdout_anger_dataset import (
    filter_to_imotions_sessions,
    add_chat_available,
    extract_results_anger,
    filter_to_results_phase,
    compute_mean_anger,
    classify_anger_result,
    get_anger_for_row,
    SESSION_TO_IMOTIONS,
)


# =====
# Test fixtures
# =====
@pytest.fixture
def sample_holdout_df():
    """Create a sample holdout DataFrame for testing."""
    return pd.DataFrame({
        "session_id": [
            "1_11-7-tr1", "1_11-7-tr1", "4_11-12-tr1",
            "6_11-18-tr1", "2_11-9-tr2", "3_11-11-tr2"
        ],
        "segment": [1, 3, 2, 4, 1, 2],
        "round": [5, 7, 3, 10, 5, 6],
        "player": ["A", "B", "C", "D", "E", "F"],
        "group_id": [1, 2, 1, 3, 1, 2],
        "round_payoff": [2.0, 4.0, 0.0, 6.0, 2.0, 4.0],
    })


@pytest.fixture
def sample_imotions_df():
    """Create a sample iMotions DataFrame for testing."""
    return pd.DataFrame({
        "Respondent Annotations active": [
            "s1r5m1Trade", "s1r5m2Trade", "s1r5m3Results",
            "s1r5m3ResultsWait", "s1r5m4Results", "s1r6m1Trade"
        ],
        "Anger": [0.5, 0.8, 1.2, 2.0, 1.8, 0.3],
    })


# =====
# Tests for filter_to_imotions_sessions
# =====
class TestFilterToImotionsSessions:
    """Tests for filter_to_imotions_sessions function."""

    def test_filters_to_valid_sessions(self, sample_holdout_df):
        """Only sessions with iMotions data should remain."""
        result = filter_to_imotions_sessions(sample_holdout_df)

        valid_sessions = list(SESSION_TO_IMOTIONS.keys())
        assert all(s in valid_sessions for s in result["session_id"])

    def test_excludes_treatment_2_sessions(self, sample_holdout_df):
        """Treatment 2 sessions should be excluded."""
        result = filter_to_imotions_sessions(sample_holdout_df)

        assert "2_11-9-tr2" not in result["session_id"].values
        assert "3_11-11-tr2" not in result["session_id"].values

    def test_preserves_columns(self, sample_holdout_df):
        """All original columns should be preserved."""
        result = filter_to_imotions_sessions(sample_holdout_df)

        assert list(result.columns) == list(sample_holdout_df.columns)

    def test_correct_row_count(self, sample_holdout_df):
        """Should have 4 rows (sessions 1, 4, 6)."""
        result = filter_to_imotions_sessions(sample_holdout_df)

        assert len(result) == 4


# =====
# Tests for add_chat_available
# =====
class TestAddChatAvailable:
    """Tests for add_chat_available function."""

    def test_segments_1_2_no_chat(self):
        """Segments 1 and 2 should have chat_available=0."""
        df = pd.DataFrame({"segment": [1, 2]})
        result = add_chat_available(df)

        assert (result["chat_available"] == 0).all()

    def test_segments_3_4_have_chat(self):
        """Segments 3 and 4 should have chat_available=1."""
        df = pd.DataFrame({"segment": [3, 4]})
        result = add_chat_available(df)

        assert (result["chat_available"] == 1).all()

    def test_mixed_segments(self, sample_holdout_df):
        """Mixed segments should have correct values."""
        result = add_chat_available(sample_holdout_df)

        for _, row in result.iterrows():
            expected = 1 if row["segment"] >= 3 else 0
            assert row["chat_available"] == expected

    def test_does_not_modify_original(self, sample_holdout_df):
        """Original DataFrame should not be modified."""
        original_cols = list(sample_holdout_df.columns)
        add_chat_available(sample_holdout_df)

        assert list(sample_holdout_df.columns) == original_cols


# =====
# Tests for filter_to_results_phase
# =====
class TestFilterToResultsPhase:
    """Tests for filter_to_results_phase function."""

    def test_matches_results_not_wait(self, sample_imotions_df):
        """Should match 'Results' but not 'ResultsWait'."""
        result = filter_to_results_phase(sample_imotions_df, "s1r5m")

        # Should match s1r5m3Results and s1r5m4Results
        assert len(result) == 2
        annotations = result["Respondent Annotations active"].tolist()
        assert "s1r5m3Results" in annotations
        assert "s1r5m4Results" in annotations
        assert "s1r5m3ResultsWait" not in annotations

    def test_no_match_wrong_segment(self, sample_imotions_df):
        """Should not match different segment."""
        result = filter_to_results_phase(sample_imotions_df, "s2r5m")

        assert result.empty

    def test_no_match_wrong_round(self, sample_imotions_df):
        """Should not match different round."""
        result = filter_to_results_phase(sample_imotions_df, "s1r6m")

        assert result.empty

    def test_handles_nan_annotations(self):
        """Should handle NaN values in annotation column."""
        df = pd.DataFrame({
            "Respondent Annotations active": [None, np.nan, "s1r1m1Results"],
            "Anger": [1.0, 2.0, 3.0],
        })
        result = filter_to_results_phase(df, "s1r1m")

        assert len(result) == 1


# =====
# Tests for compute_mean_anger
# =====
class TestComputeMeanAnger:
    """Tests for compute_mean_anger function."""

    def test_computes_mean(self):
        """Should compute mean of Anger values."""
        df = pd.DataFrame({"Anger": [1.0, 2.0, 3.0]})
        result = compute_mean_anger(df)

        assert result == 2.0

    def test_handles_nan_values(self):
        """Should ignore NaN values in mean computation."""
        df = pd.DataFrame({"Anger": [1.0, np.nan, 3.0]})
        result = compute_mean_anger(df)

        assert result == 2.0

    def test_returns_nan_for_all_nan(self):
        """Should return NaN when all values are NaN."""
        df = pd.DataFrame({"Anger": [np.nan, np.nan]})
        result = compute_mean_anger(df)

        assert np.isnan(result)

    def test_returns_nan_for_empty(self):
        """Should return NaN for empty DataFrame."""
        df = pd.DataFrame({"Anger": []})
        result = compute_mean_anger(df)

        assert np.isnan(result)


# =====
# Tests for extract_results_anger
# =====
class TestExtractResultsAnger:
    """Tests for extract_results_anger function."""

    def test_extracts_anger_for_round(self, sample_imotions_df):
        """Should extract mean anger for the specified round."""
        result = extract_results_anger(sample_imotions_df, segment=1, round_num=5)

        # s1r5m3Results has Anger=1.2, s1r5m4Results has Anger=1.8
        expected = (1.2 + 1.8) / 2
        assert result == expected

    def test_returns_none_for_missing_round(self, sample_imotions_df):
        """Should return None when no Results data for round."""
        result = extract_results_anger(sample_imotions_df, segment=2, round_num=1)

        assert result is None

    def test_returns_nan_for_no_anger_values(self):
        """Should return NaN when Results exists but no Anger values."""
        df = pd.DataFrame({
            "Respondent Annotations active": ["s1r1m1Results", "s1r1m2Results"],
            "Anger": [np.nan, np.nan],
        })
        result = extract_results_anger(df, segment=1, round_num=1)

        assert np.isnan(result)


# =====
# Tests for classify_anger_result
# =====
class TestClassifyAngerResult:
    """Tests for classify_anger_result function."""

    def test_increments_missing_results_for_none(self):
        """Should increment missing_results counter for None."""
        counters = {"missing_files": 0, "missing_results": 0, "empty_anger": 0}
        result = classify_anger_result(None, counters)

        assert result is None
        assert counters["missing_results"] == 1

    def test_increments_empty_anger_for_nan(self):
        """Should increment empty_anger counter for NaN."""
        counters = {"missing_files": 0, "missing_results": 0, "empty_anger": 0}
        result = classify_anger_result(float("nan"), counters)

        assert np.isnan(result)
        assert counters["empty_anger"] == 1

    def test_returns_value_unchanged(self):
        """Should return valid value unchanged."""
        counters = {"missing_files": 0, "missing_results": 0, "empty_anger": 0}
        result = classify_anger_result(2.5, counters)

        assert result == 2.5
        assert counters["missing_results"] == 0
        assert counters["empty_anger"] == 0


# =====
# Tests for get_anger_for_row
# =====
class TestGetAngerForRow:
    """Tests for get_anger_for_row function."""

    @patch("analysis.derived.build_holdout_anger_dataset.load_imotions_file")
    def test_increments_missing_files_when_no_file(self, mock_load):
        """Should increment missing_files when iMotions file not found."""
        mock_load.return_value = None
        row = pd.Series({
            "session_id": "1_11-7-tr1",
            "player": "A",
            "segment": 1,
            "round": 5,
        })
        counters = {"missing_files": 0, "missing_results": 0, "empty_anger": 0}

        result = get_anger_for_row(row, counters)

        assert result is None
        assert counters["missing_files"] == 1

    @patch("analysis.derived.build_holdout_anger_dataset.load_imotions_file")
    def test_extracts_anger_when_file_exists(self, mock_load, sample_imotions_df):
        """Should extract anger when iMotions file exists."""
        mock_load.return_value = sample_imotions_df
        row = pd.Series({
            "session_id": "1_11-7-tr1",
            "player": "A",
            "segment": 1,
            "round": 5,
        })
        counters = {"missing_files": 0, "missing_results": 0, "empty_anger": 0}

        result = get_anger_for_row(row, counters)

        expected = (1.2 + 1.8) / 2
        assert result == expected


# =====
# Integration tests
# =====
class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_chat_available_distribution(self, sample_holdout_df):
        """Verify chat_available is correctly distributed across segments."""
        df = filter_to_imotions_sessions(sample_holdout_df)
        df = add_chat_available(df)

        seg_1_2 = df[df["segment"].isin([1, 2])]
        seg_3_4 = df[df["segment"].isin([3, 4])]

        assert (seg_1_2["chat_available"] == 0).all()
        assert (seg_3_4["chat_available"] == 1).all()

    def test_session_mapping_keys(self):
        """Verify SESSION_TO_IMOTIONS has expected sessions."""
        expected = {"1_11-7-tr1", "4_11-12-tr1", "6_11-18-tr1"}
        assert set(SESSION_TO_IMOTIONS.keys()) == expected

    def test_session_mapping_values(self):
        """Verify SESSION_TO_IMOTIONS maps to correct iMotions session numbers."""
        assert SESSION_TO_IMOTIONS["1_11-7-tr1"] == 1
        assert SESSION_TO_IMOTIONS["4_11-12-tr1"] == 4
        assert SESSION_TO_IMOTIONS["6_11-18-tr1"] == 6


# =====
# Edge cases
# =====
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_holdout_df(self):
        """Handle empty input DataFrame."""
        df = pd.DataFrame(columns=[
            "session_id", "segment", "round", "player", "group_id", "round_payoff"
        ])
        result = filter_to_imotions_sessions(df)

        assert len(result) == 0

    def test_no_valid_sessions(self):
        """Handle case where no sessions have iMotions data."""
        df = pd.DataFrame({
            "session_id": ["2_11-9-tr2", "3_11-11-tr2"],
            "segment": [1, 2],
            "round": [1, 2],
            "player": ["A", "B"],
            "group_id": [1, 2],
            "round_payoff": [2.0, 4.0],
        })
        result = filter_to_imotions_sessions(df)

        assert len(result) == 0

    def test_round_number_with_two_digits(self):
        """Handle round numbers >= 10."""
        df = pd.DataFrame({
            "Respondent Annotations active": [
                "s1r10m1Results", "s1r10m2Results", "s1r10m3ResultsWait"
            ],
            "Anger": [1.0, 2.0, 3.0],
        })
        result = extract_results_anger(df, segment=1, round_num=10)

        assert result == 1.5

    def test_segment_4_round_14(self):
        """Handle maximum segment and round values."""
        df = pd.DataFrame({
            "Respondent Annotations active": ["s4r14m5Results"],
            "Anger": [5.0],
        })
        result = extract_results_anger(df, segment=4, round_num=14)

        assert result == 5.0


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
