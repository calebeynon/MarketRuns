"""
Purpose: Unit tests for build_period_emotions_dataset.py
Author: Claude Code
Date: 2025-02-02

Tests that raw iMotions data flows correctly through parsing and aggregation.
Uses temporary files with realistic iMotions CSV format to validate end-to-end.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.derived.build_period_emotions_dataset import (
    _aggregate_emotion,
    aggregate_by_period,
    extract_player_letter,
    filter_market_period_rows,
    parse_annotation,
    process_participant_file,
    ANNOTATION_COL,
)


# =====
# Helper to create mock iMotions CSV files
# =====
def create_mock_imotions_csv(
    file_path: Path,
    annotations: list[str],
    emotion_data: dict[str, list],
    n_metadata_rows: int = 24
):
    """
    Create a mock iMotions CSV file with 24-row metadata header.

    Args:
        file_path: Path to write the CSV
        annotations: List of annotation values for each data row
        emotion_data: Dict mapping column name to list of values
        n_metadata_rows: Number of metadata rows (default 24)
    """
    # Create metadata rows (24 rows)
    metadata_lines = []
    for i in range(n_metadata_rows):
        if i == 1:
            metadata_lines.append("Study,CPMR_1")
        elif i == 2:
            metadata_lines.append("Respondent Name,R3_uuid123")
        else:
            metadata_lines.append(f"Metadata row {i + 1},value")

    # Build data DataFrame
    data = {ANNOTATION_COL: annotations}
    data.update(emotion_data)
    df = pd.DataFrame(data)

    # Write file
    with open(file_path, 'w', encoding='utf-8-sig') as f:
        for line in metadata_lines:
            f.write(line + "\n")
        df.to_csv(f, index=False)


# =====
# Tests for raw data reading
# =====
class TestRawDataReading:
    """Tests that verify raw iMotions data is correctly read."""

    def test_skips_24_row_metadata_header(self):
        """Verify first 24 rows of metadata are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "001_R3.csv"

            # Create file with 24 metadata rows + data
            create_mock_imotions_csv(
                file_path,
                annotations=['s1r1m2MarketPeriod', 's1r1m2MarketPeriod'],
                emotion_data={
                    'Fear': [25.5, 30.0],
                    'Anger': [5.0, 10.0],
                    'Sadness': [0.0, 5.0],
                    'Joy': [50.0, 55.0],
                    'Valence': [20.0, 25.0],
                    'Engagement': [80.0, 85.0],
                }
            )

            records = process_participant_file(file_path, session_num=1, player_letter='R')

            # Should have 1 record (2 rows aggregated to 1 period)
            assert len(records) == 1
            assert records[0]['player'] == 'R'
            assert records[0]['n_samples'] == 2

    def test_reads_emotion_values_correctly(self):
        """Verify emotion values are read and aggregated correctly from raw data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "005_Q4.csv"

            # Known values for verification
            fear_values = [10.0, 20.0, 30.0]  # mean=20, max=30
            anger_values = [5.0, 15.0, 25.0]  # mean=15, max=25

            create_mock_imotions_csv(
                file_path,
                annotations=['s1r2m3MarketPeriod'] * 3,
                emotion_data={
                    'Fear': fear_values,
                    'Anger': anger_values,
                    'Sadness': [0.0, 0.0, 0.0],
                    'Joy': [50.0, 50.0, 50.0],
                    'Valence': [10.0, 10.0, 10.0],
                    'Engagement': [80.0, 80.0, 80.0],
                }
            )

            records = process_participant_file(file_path, session_num=2, player_letter='Q')

            assert len(records) == 1
            record = records[0]

            # Verify aggregation matches expected values
            assert record['fear_mean'] == pytest.approx(20.0)
            assert record['fear_max'] == pytest.approx(30.0)
            assert record['anger_mean'] == pytest.approx(15.0)
            assert record['anger_max'] == pytest.approx(25.0)

    def test_reads_valence_negative_values(self):
        """Verify Valence scale (-100 to 100) is read correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "003_M5.csv"

            # Valence includes negative values
            valence_values = [-50.0, 0.0, 50.0]  # mean=0, max=50

            create_mock_imotions_csv(
                file_path,
                annotations=['s2r1m2MarketPeriod'] * 3,
                emotion_data={
                    'Fear': [0.0] * 3,
                    'Anger': [0.0] * 3,
                    'Sadness': [0.0] * 3,
                    'Joy': [0.0] * 3,
                    'Valence': valence_values,
                    'Engagement': [50.0] * 3,
                }
            )

            records = process_participant_file(file_path, session_num=3, player_letter='M')
            record = records[0]

            assert record['valence_mean'] == pytest.approx(0.0)
            assert record['valence_max'] == pytest.approx(50.0)


# =====
# Tests for annotation parsing from raw data
# =====
class TestAnnotationParsing:
    """Tests that annotations are correctly parsed from raw data."""

    def test_extracts_segment_round_period(self):
        """Verify segment, round, period extracted from annotation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "001_A3.csv"

            create_mock_imotions_csv(
                file_path,
                annotations=['s3r7m4MarketPeriod'] * 5,
                emotion_data={
                    'Fear': [10.0] * 5,
                    'Anger': [10.0] * 5,
                    'Sadness': [10.0] * 5,
                    'Joy': [10.0] * 5,
                    'Valence': [10.0] * 5,
                    'Engagement': [10.0] * 5,
                }
            )

            records = process_participant_file(file_path, session_num=1, player_letter='A')
            record = records[0]

            assert record['segment'] == 3
            assert record['round'] == 7
            assert record['period'] == 3  # m4 - 1 = 3 (offset correction)

    def test_period_offset_correction_m2_to_1(self):
        """Verify m2 becomes oTree period 1."""
        result = parse_annotation('s1r1m2MarketPeriod')
        assert result == (1, 1, 1)

    def test_period_offset_correction_m5_to_4(self):
        """Verify m5 becomes oTree period 4."""
        result = parse_annotation('s2r3m5MarketPeriod')
        assert result == (2, 3, 4)

    def test_multiple_periods_in_same_round(self):
        """Verify multiple periods within same round are separated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "001_B3.csv"

            # Mix of 3 different periods in round 1
            annotations = (
                ['s1r1m2MarketPeriod'] * 10 +
                ['s1r1m3MarketPeriod'] * 10 +
                ['s1r1m4MarketPeriod'] * 10
            )

            create_mock_imotions_csv(
                file_path,
                annotations=annotations,
                emotion_data={
                    'Fear': [10.0] * 30,
                    'Anger': [10.0] * 30,
                    'Sadness': [10.0] * 30,
                    'Joy': [10.0] * 30,
                    'Valence': [10.0] * 30,
                    'Engagement': [10.0] * 30,
                }
            )

            records = process_participant_file(file_path, session_num=1, player_letter='B')

            # Should have 3 records (one per period)
            assert len(records) == 3
            periods = sorted([r['period'] for r in records])
            assert periods == [1, 2, 3]


# =====
# Tests for filtering MarketPeriod rows
# =====
class TestMarketPeriodFiltering:
    """Tests that only MarketPeriod rows are included."""

    def test_excludes_market_period_wait(self):
        """Verify MarketPeriodWait is excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "001_C3.csv"

            annotations = [
                's1r1m2MarketPeriod',      # Include
                's1r1m2MarketPeriodWait',  # Exclude
                's1r1m2MarketPeriod',      # Include
            ]

            create_mock_imotions_csv(
                file_path,
                annotations=annotations,
                emotion_data={
                    'Fear': [10.0, 99.0, 20.0],  # 99.0 should be excluded
                    'Anger': [5.0, 99.0, 15.0],
                    'Sadness': [0.0, 99.0, 10.0],
                    'Joy': [50.0, 99.0, 60.0],
                    'Valence': [25.0, 99.0, 35.0],
                    'Engagement': [80.0, 99.0, 90.0],
                }
            )

            records = process_participant_file(file_path, session_num=1, player_letter='C')
            record = records[0]

            # Mean should be (10 + 20) / 2 = 15, not include the 99
            assert record['fear_mean'] == pytest.approx(15.0)
            assert record['n_samples'] == 2  # Only 2 rows, not 3

    def test_excludes_chat_and_results_phases(self):
        """Verify Chat and Results phases are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "001_D3.csv"

            annotations = [
                's1r1m2MarketPeriod',
                's1r1m2Chat',
                's1r1m2Results',
                's1r1m2MarketPeriod',
            ]

            create_mock_imotions_csv(
                file_path,
                annotations=annotations,
                emotion_data={
                    'Fear': [10.0, 99.0, 99.0, 20.0],
                    'Anger': [5.0] * 4,
                    'Sadness': [0.0] * 4,
                    'Joy': [50.0] * 4,
                    'Valence': [25.0] * 4,
                    'Engagement': [80.0] * 4,
                }
            )

            records = process_participant_file(file_path, session_num=1, player_letter='D')
            record = records[0]

            # Only MarketPeriod rows should be included
            assert record['n_samples'] == 2
            assert record['fear_mean'] == pytest.approx(15.0)


# =====
# Tests for edge cases
# =====
class TestEdgeCases:
    """Tests for edge cases and data quality issues."""

    def test_handles_missing_emotion_columns(self):
        """Handle case where some emotion columns are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "001_E3.csv"

            # Only include some emotion columns
            create_mock_imotions_csv(
                file_path,
                annotations=['s1r1m2MarketPeriod'] * 3,
                emotion_data={
                    'Fear': [10.0, 20.0, 30.0],
                    # Missing: Anger, Sadness, Joy, Valence, Engagement
                }
            )

            records = process_participant_file(file_path, session_num=1, player_letter='E')
            record = records[0]

            # Fear should be aggregated
            assert record['fear_mean'] == pytest.approx(20.0)
            # Missing columns should be NaN
            assert np.isnan(record['anger_mean'])
            assert np.isnan(record['joy_max'])

    def test_handles_nan_in_emotion_values(self):
        """Handle NaN values in emotion data (face detection failures)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "001_F3.csv"

            create_mock_imotions_csv(
                file_path,
                annotations=['s1r1m2MarketPeriod'] * 4,
                emotion_data={
                    'Fear': [10.0, np.nan, 30.0, np.nan],  # mean of valid = 20
                    'Anger': [5.0, 15.0, 25.0, 35.0],
                    'Sadness': [0.0] * 4,
                    'Joy': [50.0] * 4,
                    'Valence': [25.0] * 4,
                    'Engagement': [80.0] * 4,
                }
            )

            records = process_participant_file(file_path, session_num=1, player_letter='F')
            record = records[0]

            # NaN should be excluded from mean calculation
            assert record['fear_mean'] == pytest.approx(20.0)
            assert record['n_samples'] == 4  # Still 4 rows, just some with NaN

    def test_skips_period_m1(self):
        """Period m1 becomes oTree period 0 and should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "001_G3.csv"

            annotations = [
                's1r1m1MarketPeriod',  # Should be skipped (period=0)
                's1r1m2MarketPeriod',  # Should be included (period=1)
            ]

            create_mock_imotions_csv(
                file_path,
                annotations=annotations,
                emotion_data={
                    'Fear': [99.0, 15.0],
                    'Anger': [0.0] * 2,
                    'Sadness': [0.0] * 2,
                    'Joy': [0.0] * 2,
                    'Valence': [0.0] * 2,
                    'Engagement': [0.0] * 2,
                }
            )

            records = process_participant_file(file_path, session_num=1, player_letter='G')

            # Only one record (m1 skipped)
            assert len(records) == 1
            assert records[0]['period'] == 1
            assert records[0]['fear_mean'] == pytest.approx(15.0)

    def test_returns_empty_for_no_market_period(self):
        """Return empty list if file has no MarketPeriod annotations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "001_H3.csv"

            create_mock_imotions_csv(
                file_path,
                annotations=['s1r1m2Chat', 's1r1m2Results'],
                emotion_data={
                    'Fear': [10.0, 20.0],
                    'Anger': [5.0, 15.0],
                    'Sadness': [0.0, 10.0],
                    'Joy': [50.0, 60.0],
                    'Valence': [25.0, 35.0],
                    'Engagement': [80.0, 90.0],
                }
            )

            records = process_participant_file(file_path, session_num=1, player_letter='H')
            assert len(records) == 0


# =====
# Tests for player letter extraction
# =====
class TestPlayerLetterExtraction:
    """Tests for extracting player letter from filename."""

    def test_extracts_letter_session_1(self):
        """Extract letter from session 1 filename (suffix 3)."""
        assert extract_player_letter("001_R3.csv") == "R"

    def test_extracts_letter_session_6(self):
        """Extract letter from session 6 filename (suffix 8)."""
        assert extract_player_letter("016_A8.csv") == "A"

    def test_rejects_export_merge(self):
        """Reject ExportMerge.csv files."""
        assert extract_player_letter("ExportMerge.csv") is None


# =====
# Integration test with realistic data volume
# =====
class TestRealisticDataVolume:
    """Tests with realistic data volumes (~100 rows per period)."""

    def test_aggregates_large_sample_correctly(self):
        """Aggregate ~100 rows per period (realistic for ~4 second period at 25Hz)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "001_R3.csv"

            n_samples = 100
            # Simulate realistic emotion values
            np.random.seed(42)
            fear_values = np.random.uniform(0, 30, n_samples).tolist()

            create_mock_imotions_csv(
                file_path,
                annotations=['s1r1m2MarketPeriod'] * n_samples,
                emotion_data={
                    'Fear': fear_values,
                    'Anger': [5.0] * n_samples,
                    'Sadness': [2.0] * n_samples,
                    'Joy': [60.0] * n_samples,
                    'Valence': [40.0] * n_samples,
                    'Engagement': [75.0] * n_samples,
                }
            )

            records = process_participant_file(file_path, session_num=1, player_letter='R')
            record = records[0]

            # Verify sample count
            assert record['n_samples'] == 100

            # Verify aggregation matches numpy calculations
            assert record['fear_mean'] == pytest.approx(np.mean(fear_values), rel=1e-5)
            assert record['fear_max'] == pytest.approx(np.max(fear_values), rel=1e-5)
            assert record['fear_std'] == pytest.approx(np.std(fear_values, ddof=1), rel=1e-5)


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
