"""
Purpose: Unit tests for build_presell_emotions.py
Author: Claude Code
Date: 2026-04-05
"""

import pytest
import pandas as pd
import numpy as np
from analysis.derived.build_presell_emotions import (
    extract_window_emotions,
    extract_all_windows,
    suffix_emotions,
    extract_player_label,
    pid_to_label,
    excel_to_epoch,
    add_global_group_id,
    EMOTION_COLS,
    PRESELL_WINDOWS_MS,
)


# =====
# Fixtures
# =====
@pytest.fixture
def sample_imotions_df():
    """Create synthetic iMotions data with known timestamps and emotions."""
    n = 100
    data = {"Timestamp": list(range(0, n * 33, 33))}  # ~30fps, 0-3267ms
    for col in EMOTION_COLS:
        data[col] = [float(i) for i in range(n)]
    return pd.DataFrame(data)


# =====
# extract_window_emotions tests
# =====
def test_extract_window_basic(sample_imotions_df):
    """Frames within window are averaged correctly."""
    result = extract_window_emotions(sample_imotions_df, 0, 99)
    # Timestamps 0, 33, 66, 99 -> indices 0, 1, 2, 3
    assert result["n_frames"] == 4
    # Mean of [0, 1, 2, 3] = 1.5 for each emotion
    assert result["anger_mean"] == pytest.approx(1.5)


def test_extract_window_empty():
    """Empty window returns NaN emotions and 0 frames."""
    df = pd.DataFrame({
        "Timestamp": [100, 200, 300],
        **{col: [1.0, 2.0, 3.0] for col in EMOTION_COLS},
    })
    result = extract_window_emotions(df, 400, 500)
    assert result["n_frames"] == 0
    for col in EMOTION_COLS:
        assert np.isnan(result[f"{col.lower()}_mean"])


def test_extract_window_boundary_inclusive(sample_imotions_df):
    """Frames exactly at window edges are included."""
    # Window [33, 66] should include timestamps 33 and 66
    result = extract_window_emotions(sample_imotions_df, 33, 66)
    assert result["n_frames"] == 2
    # Indices 1 and 2 -> mean = 1.5
    assert result["anger_mean"] == pytest.approx(1.5)


def test_extract_window_single_frame():
    """Single frame in window produces that frame's values."""
    df = pd.DataFrame({
        "Timestamp": [100, 200, 300],
        **{col: [10.0, 20.0, 30.0] for col in EMOTION_COLS},
    })
    result = extract_window_emotions(df, 195, 205)
    assert result["n_frames"] == 1
    assert result["joy_mean"] == pytest.approx(20.0)


def test_extract_window_with_nan_emotions():
    """NaN emotion values are ignored in the mean."""
    df = pd.DataFrame({
        "Timestamp": [100, 200, 300, 400],
        "Anger": [10.0, float("nan"), 30.0, float("nan")],
        **{col: [1.0] * 4 for col in EMOTION_COLS if col != "Anger"},
    })
    result = extract_window_emotions(df, 50, 450)
    assert result["n_frames"] == 4
    # Mean of [10, 30] = 20 (NaN ignored)
    assert result["anger_mean"] == pytest.approx(20.0)


def test_extract_window_all_nan_emotions():
    """All-NaN emotions produce NaN mean."""
    df = pd.DataFrame({
        "Timestamp": [100, 200],
        **{col: [float("nan"), float("nan")] for col in EMOTION_COLS},
    })
    result = extract_window_emotions(df, 50, 250)
    assert result["n_frames"] == 2
    for col in EMOTION_COLS:
        assert np.isnan(result[f"{col.lower()}_mean"])


def test_extract_window_all_emotions_present():
    """All 9 emotion columns are present in output."""
    df = pd.DataFrame({
        "Timestamp": [100],
        **{col: [5.0] for col in EMOTION_COLS},
    })
    result = extract_window_emotions(df, 50, 150)
    expected_keys = {"n_frames"} | {
        f"{c.lower()}_mean" for c in EMOTION_COLS
    }
    assert set(result.keys()) == expected_keys


# =====
# suffix_emotions and extract_all_windows tests
# =====
def test_suffix_emotions():
    """Keys are suffixed with window size."""
    emotions = {"n_frames": 5, "anger_mean": 1.0}
    result = suffix_emotions(emotions, 2000)
    assert result == {"n_frames_2000ms": 5, "anger_mean_2000ms": 1.0}


def test_extract_all_windows(sample_imotions_df):
    """All 5 window sizes produce suffixed keys."""
    click_ms = 2000.0
    result = extract_all_windows(sample_imotions_df, click_ms)
    for w in PRESELL_WINDOWS_MS:
        assert f"n_frames_{w}ms" in result
        assert f"anger_mean_{w}ms" in result


def test_extract_all_windows_nested_counts(sample_imotions_df):
    """Smaller windows have fewer or equal frames than larger ones."""
    click_ms = 2000.0
    result = extract_all_windows(sample_imotions_df, click_ms)
    counts = [result[f"n_frames_{w}ms"] for w in PRESELL_WINDOWS_MS]
    # Sorted by window size descending, so counts should be non-increasing
    for i in range(len(counts) - 1):
        assert counts[i] >= counts[i + 1]


# =====
# extract_player_label tests
# =====
def test_extract_label_standard():
    """Standard iMotions filename extracts letter."""
    assert extract_player_label("001_R3.csv") == "R"
    assert extract_player_label("016_A3.csv") == "A"


def test_extract_label_different_sessions():
    """Different session suffixes still extract the letter."""
    assert extract_player_label("005_M4.csv") == "M"
    assert extract_player_label("010_G8.csv") == "G"


def test_extract_label_export_merge():
    """ExportMerge.csv returns None."""
    assert extract_player_label("ExportMerge.csv") is None


def test_extract_label_invalid():
    """Non-matching filename returns None."""
    assert extract_player_label("random.csv") is None


# =====
# pid_to_label tests
# =====
def test_pid_to_label_first():
    """Participant ID 1 maps to A."""
    assert pid_to_label(1) == "A"


def test_pid_to_label_last():
    """Participant ID 16 maps to R."""
    assert pid_to_label(16) == "R"


def test_pid_to_label_skips_i_and_o():
    """Letters I and O are skipped in the mapping."""
    # ID 9 should be J (not I)
    assert pid_to_label(9) == "J"
    # ID 14 should be P (not O)
    assert pid_to_label(14) == "P"


def test_pid_to_label_all():
    """All 16 IDs map to the expected labels."""
    expected = "ABCDEFGHJKLMNPQR"
    for i, letter in enumerate(expected, 1):
        assert pid_to_label(i) == letter


# =====
# excel_to_epoch tests
# =====
def test_excel_to_epoch_known_value():
    """Excel serial from edited_data maps to expected epoch."""
    # From e1.csv: participant 1 RECORDING = 45968.53794
    # Expected epoch: approx 1762541678 (Nov 7, 2025 ~12:54 CST)
    epoch = excel_to_epoch(45968.53794)
    assert abs(epoch - 1762541678.0) < 1.0


def test_excel_to_epoch_roundtrip():
    """Epoch -> Excel -> epoch roundtrip preserves value."""
    original_epoch = 1762541678.0
    # epoch = (serial - 25569 + 6/24) * 86400
    # serial = epoch / 86400 + 25569 - 6/24
    serial = original_epoch / 86400 + 25569.0 - 6 / 24
    result = excel_to_epoch(serial)
    assert result == pytest.approx(original_epoch)


# =====
# add_global_group_id tests
# =====
def test_add_global_group_id():
    """Global group ID format is {session_id}_{segment}_{group_id}."""
    df = pd.DataFrame({
        "session_id": ["1_11-7-tr1"],
        "segment": [2],
        "group_id": [3],
    })
    result = add_global_group_id(df)
    assert result["global_group_id"].iloc[0] == "1_11-7-tr1_2_3"


def test_add_global_group_id_multiple():
    """Multiple rows get unique global group IDs."""
    df = pd.DataFrame({
        "session_id": ["1_11-7-tr1", "2_11-10-tr2"],
        "segment": [1, 3],
        "group_id": [1, 4],
    })
    result = add_global_group_id(df)
    assert result["global_group_id"].iloc[0] == "1_11-7-tr1_1_1"
    assert result["global_group_id"].iloc[1] == "2_11-10-tr2_3_4"


# =====
# Timestamp conversion math tests
# =====
def test_timestamp_conversion_math():
    """Verify (sell_click_time - recording_start) * 1000 gives iMotions ms."""
    rec_start = 1762541678.0
    sell_click = 1762543103.523
    expected_ms = (sell_click - rec_start) * 1000
    assert expected_ms == pytest.approx(1425523.0)


def test_timestamp_window_sizes():
    """All pre-sell window sizes are present in the constant."""
    assert PRESELL_WINDOWS_MS == [2000, 1000, 500, 100, 50]


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
