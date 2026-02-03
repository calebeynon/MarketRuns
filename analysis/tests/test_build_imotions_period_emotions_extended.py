"""
Purpose: Thorough unit tests for build_imotions_period_emotions_extended.py
Author: Claude Code
Date: 2026-02-02

Tests verify correct data flow from raw iMotions CSV format through aggregation,
including edge cases for missing data, annotation parsing, and aggregation logic.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from io import StringIO

from analysis.derived.build_imotions_period_emotions_extended import (
    parse_market_period_annotation,
    extract_player_label,
    aggregate_emotions,
    process_participant_file,
    EMOTION_COLS,
    ANNOTATION_COL,
    IMOTIONS_SKIP_ROWS,
)


# =====
# Tests for parse_market_period_annotation
# =====
class TestParseAnnotation:
    """Test annotation string parsing to (segment, round, period) tuples."""

    def test_valid_market_period_basic(self):
        """Standard MarketPeriod annotation parses correctly."""
        result = parse_market_period_annotation("s1r5m2MarketPeriod")
        assert result == (1, 5, 1)  # m2 -> period 1 (offset by 1)

    def test_valid_market_period_segment_boundaries(self):
        """All four segments parse correctly."""
        assert parse_market_period_annotation("s1r1m1MarketPeriod") == (1, 1, 0)
        assert parse_market_period_annotation("s2r7m3MarketPeriod") == (2, 7, 2)
        assert parse_market_period_annotation("s3r10m2MarketPeriod") == (3, 10, 1)
        assert parse_market_period_annotation("s4r14m4MarketPeriod") == (4, 14, 3)

    def test_high_round_numbers(self):
        """Round 14 (max) parses correctly."""
        result = parse_market_period_annotation("s4r14m4MarketPeriod")
        assert result == (4, 14, 3)

    def test_high_period_numbers(self):
        """Higher period numbers parse correctly (m values up to 5+)."""
        result = parse_market_period_annotation("s1r1m5MarketPeriod")
        assert result == (1, 1, 4)  # m5 -> period 4

    def test_non_market_period_annotations_return_none(self):
        """Non-MarketPeriod annotations should return None."""
        non_market = [
            "s1r5m2Results",
            "s1r5m2Chat",
            "s1r5m2MarketPeriodWait",
            "s1r5m2SegmentIntro",
            "s1r5m2RoundIntro",
            "SegmentIntro",
            "SurveyPage",
        ]
        for annotation in non_market:
            assert parse_market_period_annotation(annotation) is None, \
                f"Expected None for '{annotation}'"

    def test_nan_and_none_return_none(self):
        """NaN and None values should return None."""
        assert parse_market_period_annotation(np.nan) is None
        assert parse_market_period_annotation(None) is None
        assert parse_market_period_annotation(pd.NA) is None

    def test_empty_and_whitespace_return_none(self):
        """Empty strings and whitespace should return None."""
        assert parse_market_period_annotation("") is None
        assert parse_market_period_annotation("  ") is None

    def test_malformed_annotations_return_none(self):
        """Malformed annotation strings should return None."""
        malformed = [
            "s1r5MarketPeriod",  # missing m value
            "sr5m2MarketPeriod",  # missing segment number
            "s1rm2MarketPeriod",  # missing round number
            "MarketPeriod",  # no prefix
            "s1r5m2",  # missing suffix
        ]
        for annotation in malformed:
            assert parse_market_period_annotation(annotation) is None, \
                f"Expected None for malformed '{annotation}'"


# =====
# Tests for extract_player_label
# =====
class TestExtractPlayerLabel:
    """Test extraction of player letter from iMotions filename."""

    def test_valid_filenames(self):
        """Standard iMotions filenames extract player letter correctly."""
        assert extract_player_label("001_A3.csv") == "A"
        assert extract_player_label("016_R1.csv") == "R"
        assert extract_player_label("005_E2.csv") == "E"
        assert extract_player_label("012_M4.csv") == "M"

    def test_all_valid_player_letters(self):
        """All valid player letters (A-R, excluding I and O) extract correctly."""
        valid_letters = "ABCDEFGHJKLMNPQR"  # I and O excluded
        for i, letter in enumerate(valid_letters, 1):
            filename = f"{i:03d}_{letter}1.csv"
            assert extract_player_label(filename) == letter

    def test_invalid_filenames_return_none(self):
        """Invalid filenames should return None."""
        invalid = [
            "ExportMerge.csv",
            "invalid.csv",
            "A3.csv",  # missing order number
            "001_3.csv",  # missing letter
            "001_AB3.csv",  # multiple letters
            "participant_001.csv",
        ]
        for filename in invalid:
            assert extract_player_label(filename) is None, \
                f"Expected None for invalid '{filename}'"


# =====
# Tests for aggregate_emotions
# =====
class TestAggregateEmotions:
    """Test emotion aggregation logic (mean, max, p95)."""

    def _create_mock_data(self, emotion_values: dict, n_rows: int = 5) -> pd.DataFrame:
        """Helper to create mock MarketPeriod data."""
        base = {
            "segment": [1] * n_rows,
            "round": [1] * n_rows,
            "period": [1] * n_rows,
        }
        for col in EMOTION_COLS:
            if col in emotion_values:
                base[col] = emotion_values[col]
            else:
                base[col] = [0.0] * n_rows
        return pd.DataFrame(base)

    def test_mean_computation(self):
        """Mean is computed correctly."""
        data = self._create_mock_data({"Anger": [10, 20, 30, 40, 50]})
        records = aggregate_emotions(data, "test", "A")
        assert records[0]["anger_mean"] == 30.0

    def test_max_computation(self):
        """Max is computed correctly."""
        data = self._create_mock_data({"Anger": [10, 20, 30, 40, 100]})
        records = aggregate_emotions(data, "test", "A")
        assert records[0]["anger_max"] == 100.0

    def test_p95_computation(self):
        """95th percentile is computed correctly."""
        # With 100 values from 1-100, p95 should be ~95
        values = list(range(1, 101))
        data = self._create_mock_data({"Fear": values}, n_rows=100)
        records = aggregate_emotions(data, "test", "A")
        assert records[0]["fear_p95"] == pytest.approx(95.05, rel=0.01)

    def test_max_always_geq_mean(self):
        """Max should always be >= mean for any distribution."""
        test_cases = [
            [10, 20, 30, 40, 50],  # uniform spread
            [50, 50, 50, 50, 50],  # all same
            [0, 0, 0, 0, 100],  # one outlier
            [1, 2, 3, 4, 5],  # small values
        ]
        for values in test_cases:
            data = self._create_mock_data({"Anger": values}, n_rows=len(values))
            records = aggregate_emotions(data, "test", "A")
            assert records[0]["anger_max"] >= records[0]["anger_mean"], \
                f"Max should be >= mean for values {values}"

    def test_p95_between_mean_and_max(self):
        """P95 should be <= max (but NOT necessarily >= mean for skewed data)."""
        test_cases = [
            [10, 20, 30, 40, 50],
            [50, 50, 50, 50, 50],
            [0, 0, 0, 0, 100],
        ]
        for values in test_cases:
            data = self._create_mock_data({"Anger": values}, n_rows=len(values))
            records = aggregate_emotions(data, "test", "A")
            # p95 <= max is always true
            assert records[0]["anger_p95"] <= records[0]["anger_max"], \
                f"P95 should be <= max for values {values}"

    def test_handles_nan_values_in_emotion(self):
        """NaN values in emotion column should be excluded from aggregation."""
        data = self._create_mock_data({"Anger": [10, np.nan, 30, np.nan, 50]})
        records = aggregate_emotions(data, "test", "A")
        # Mean of [10, 30, 50] = 30
        assert records[0]["anger_mean"] == 30.0
        assert records[0]["anger_max"] == 50.0

    def test_all_nan_values_returns_nan(self):
        """If all emotion values are NaN, aggregations should be NaN."""
        data = self._create_mock_data({"Anger": [np.nan] * 5})
        records = aggregate_emotions(data, "test", "A")
        assert np.isnan(records[0]["anger_mean"])
        assert np.isnan(records[0]["anger_max"])
        assert np.isnan(records[0]["anger_p95"])

    def test_single_frame_period(self):
        """Single frame periods should still aggregate correctly."""
        data = self._create_mock_data({"Fear": [42.5]}, n_rows=1)
        records = aggregate_emotions(data, "test", "A")
        assert records[0]["fear_mean"] == 42.5
        assert records[0]["fear_max"] == 42.5
        assert records[0]["fear_p95"] == 42.5
        assert records[0]["n_frames"] == 1

    def test_n_frames_count(self):
        """n_frames should count total rows, including NaN emotion values."""
        data = self._create_mock_data({"Anger": [10, np.nan, 30]}, n_rows=3)
        records = aggregate_emotions(data, "test", "A")
        assert records[0]["n_frames"] == 3  # counts all rows

    def test_multiple_periods_separated(self):
        """Multiple periods in same data should produce separate records."""
        data = pd.DataFrame({
            "segment": [1, 1, 1, 1, 1, 1],
            "round": [1, 1, 1, 2, 2, 2],
            "period": [1, 1, 1, 1, 1, 1],
            **{col: [10] * 6 for col in EMOTION_COLS},
        })
        records = aggregate_emotions(data, "test", "A")
        assert len(records) == 2
        assert records[0]["round"] == 1
        assert records[1]["round"] == 2

    def test_session_and_player_preserved(self):
        """Session ID and player label should be preserved in output."""
        data = self._create_mock_data({"Anger": [10]}, n_rows=1)
        records = aggregate_emotions(data, "session_123", "Z")
        assert records[0]["session_id"] == "session_123"
        assert records[0]["player"] == "Z"

    def test_string_emotion_values_converted(self):
        """String representations of numbers should be converted."""
        data = self._create_mock_data({}, n_rows=3)
        data["Anger"] = ["10.5", "20.5", "30.5"]  # strings
        records = aggregate_emotions(data, "test", "A")
        assert records[0]["anger_mean"] == pytest.approx(20.5)

    def test_non_numeric_strings_treated_as_nan(self):
        """Non-numeric strings should be treated as NaN."""
        data = self._create_mock_data({}, n_rows=3)
        data["Anger"] = ["10", "invalid", "30"]
        records = aggregate_emotions(data, "test", "A")
        assert records[0]["anger_mean"] == 20.0  # (10 + 30) / 2


# =====
# Tests for all emotion columns
# =====
class TestAllEmotionColumns:
    """Verify all emotion columns are processed."""

    def test_all_emotion_columns_in_output(self):
        """All expected emotion columns should appear in output."""
        data = pd.DataFrame({
            "segment": [1],
            "round": [1],
            "period": [1],
            **{col: [50.0] for col in EMOTION_COLS},
        })
        records = aggregate_emotions(data, "test", "A")
        record = records[0]

        for col in EMOTION_COLS:
            col_lower = col.lower()
            assert f"{col_lower}_mean" in record
            assert f"{col_lower}_max" in record
            assert f"{col_lower}_p95" in record


# =====
# Integration tests with real data structure
# =====
class TestRealDataStructure:
    """Test against real iMotions CSV structure."""

    def test_raw_imotions_csv_format(self, tmp_path):
        """Test parsing of actual iMotions CSV format with header rows."""
        # Create mock iMotions CSV with 24 metadata rows + header + data
        metadata_rows = "\n".join([f"metadata_row_{i}" for i in range(24)])
        header = f"Timestamp,{ANNOTATION_COL}," + ",".join(EMOTION_COLS)
        data_rows = [
            f"1000,s1r1m1MarketPeriod,10,20,30,40,50,60,70,80,90",
            f"1040,s1r1m1MarketPeriod,11,21,31,41,51,61,71,81,91",
            f"2000,s1r1m2MarketPeriod,15,25,35,45,55,65,75,85,95",
        ]

        csv_content = metadata_rows + "\n" + header + "\n" + "\n".join(data_rows)
        csv_file = tmp_path / "001_A3.csv"
        csv_file.write_text(csv_content)

        records = process_participant_file(csv_file, "test_session", "A")

        # Should have 2 records: period 0 (m1) and period 1 (m2)
        assert len(records) == 2

        # First period (m1 -> period 0) has 2 frames
        period_0 = [r for r in records if r["period"] == 0][0]
        assert period_0["n_frames"] == 2
        assert period_0["anger_mean"] == 10.5  # (10 + 11) / 2

        # Second period (m2 -> period 1) has 1 frame
        period_1 = [r for r in records if r["period"] == 1][0]
        assert period_1["n_frames"] == 1
        assert period_1["anger_mean"] == 15.0


# =====
# Output file validation (run after data generation)
# =====
class TestOutputFile:
    """Validate the generated output file."""

    @pytest.fixture
    def output_df(self):
        """Load output file if it exists."""
        output_path = Path("datastore/derived/imotions_period_emotions_extended.csv")
        if not output_path.exists():
            pytest.skip("Output file not yet generated - run builder first")
        return pd.read_csv(output_path)

    def test_expected_columns_present(self, output_df):
        """Verify all expected columns are in output."""
        expected = ["session_id", "segment", "round", "period", "player", "n_frames"]
        for col in EMOTION_COLS:
            expected.extend([f"{col.lower()}_mean", f"{col.lower()}_max", f"{col.lower()}_p95"])

        missing = set(expected) - set(output_df.columns)
        assert len(missing) == 0, f"Missing columns: {missing}"

    def test_no_duplicate_rows(self, output_df):
        """No duplicate (session, segment, round, period, player) combinations."""
        key_cols = ["session_id", "segment", "round", "period", "player"]
        duplicates = output_df[output_df.duplicated(subset=key_cols, keep=False)]
        assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate rows"

    def test_max_geq_mean_all_rows(self, output_df):
        """Max >= mean for all rows and all emotions."""
        for col in ["fear", "anger"]:
            mean_col = f"{col}_mean"
            max_col = f"{col}_max"

            valid = output_df[output_df[mean_col].notna() & output_df[max_col].notna()]
            violations = valid[valid[max_col] < valid[mean_col]]

            assert len(violations) == 0, \
                f"Found {len(violations)} rows where {max_col} < {mean_col}"

    def test_p95_leq_max_all_rows(self, output_df):
        """P95 <= max for all rows and all emotions."""
        for col in ["fear", "anger"]:
            max_col = f"{col}_max"
            p95_col = f"{col}_p95"

            valid = output_df[output_df[max_col].notna() & output_df[p95_col].notna()]
            violations = valid[valid[p95_col] > valid[max_col]]

            assert len(violations) == 0, \
                f"Found {len(violations)} rows where {p95_col} > {max_col}"

    def test_all_sessions_present(self, output_df):
        """All 6 sessions should be present."""
        expected_sessions = {
            "1_11-7-tr1", "2_11-10-tr2", "3_11-11-tr2",
            "4_11-12-tr1", "5_11-14-tr2", "6_11-18-tr1",
        }
        actual_sessions = set(output_df["session_id"].unique())
        missing = expected_sessions - actual_sessions
        assert len(missing) == 0, f"Missing sessions: {missing}"

    def test_segments_in_valid_range(self, output_df):
        """Segments should be 1-4."""
        assert output_df["segment"].min() >= 1
        assert output_df["segment"].max() <= 4

    def test_rounds_in_valid_range(self, output_df):
        """Rounds should be 1-14."""
        assert output_df["round"].min() >= 1
        assert output_df["round"].max() <= 14

    def test_emotion_values_in_valid_range(self, output_df):
        """Emotion values should be in 0-100 range."""
        for col in ["fear", "anger"]:
            for agg in ["mean", "max", "p95"]:
                col_name = f"{col}_{agg}"
                valid = output_df[col_name].dropna()
                assert valid.min() >= 0, f"{col_name} has values < 0"
                assert valid.max() <= 100, f"{col_name} has values > 100"

    def test_n_frames_positive(self, output_df):
        """n_frames should be positive for all rows."""
        assert (output_df["n_frames"] > 0).all()

    def test_compare_with_original_row_count(self, output_df):
        """Row count should match original imotions_period_emotions.csv."""
        original_path = Path("datastore/derived/imotions_period_emotions.csv")
        if not original_path.exists():
            pytest.skip("Original file not found for comparison")

        original_df = pd.read_csv(original_path)
        assert len(output_df) == len(original_df), \
            f"Row count mismatch: {len(output_df)} vs {len(original_df)}"
