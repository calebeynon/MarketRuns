"""
Purpose: Unit tests for build_presell_emotions.py
Author: Claude Code
Date: 2026-04-05
"""

from pathlib import Path

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
    IMOTIONS_SKIP_ROWS,
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


# =====
# Integration tests (real data)
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
DERIVED = DATASTORE / "derived"
OUTPUT_CSV = DERIVED / "presell_emotions_traits_dataset.csv"
OTREE_SESSION1_SEG1 = DATASTORE / "1_11-7-tr1" / "chat_noavg_2025-11-07.csv"
EDITED_DATA_E1 = DATASTORE / "annotations" / "edited_data" / "e1.csv"
IMOTIONS_PLAYER_A = DATASTORE / "imotions" / "1" / "016_A3.csv"
TRAITS_CSV = DERIVED / "survey_traits.csv"
PERIOD_CSV = DERIVED / "individual_period_dataset.csv"

INTEGRATION_SESSIONS = [
    "1_11-7-tr1", "2_11-10-tr2", "3_11-11-tr2",
    "4_11-12-tr1", "5_11-14-tr2", "6_11-18-tr1",
]


@pytest.fixture(scope="module")
def output_df():
    """Load the output dataset once for all integration tests."""
    return pd.read_csv(OUTPUT_CSV)


# =====
# Test 1: Spot-check a specific sell event end-to-end
# =====
@pytest.mark.integration
def test_spotcheck_otree_sell_event():
    """Verify player A has sold=1 and correct sell_click_time."""
    df = pd.read_csv(OTREE_SESSION1_SEG1)
    mask = (
        (df["participant.label"] == "A")
        & (df["player.round_number_in_segment"] == 2)
        & (df["player.period_in_round"] == 1)
    )
    row = df[mask]
    assert len(row) == 1, "Expected exactly one row"
    assert row["player.sold"].values[0] == 1
    assert row["player.sell_click_time"].values[0] == pytest.approx(
        1762543103.523
    )


@pytest.mark.integration
def test_spotcheck_recording_start_epoch():
    """Verify participant 1 RECORDING serial converts correctly."""
    ed = pd.read_csv(EDITED_DATA_E1)
    serial = ed.loc[
        ed["participant_id_in_session"] == 1, "RECORDING"
    ].iloc[0]
    assert serial == pytest.approx(45968.537939814814)
    epoch = excel_to_epoch(serial)
    assert epoch == pytest.approx(1762541678.0, abs=1.0)


@pytest.mark.integration
def test_spotcheck_click_ms_computation():
    """Verify click_ms = (sell_click - recording_start) * 1000."""
    click_ms = (1762543103.523 - 1762541678.0) * 1000
    assert click_ms == pytest.approx(1425523.0, abs=1.0)


@pytest.mark.integration
def test_spotcheck_imotions_2000ms_frames():
    """Verify 60 frames in the 2000ms window for player A sell event."""
    df = pd.read_csv(
        IMOTIONS_PLAYER_A,
        skiprows=IMOTIONS_SKIP_ROWS,
        encoding="utf-8-sig",
        low_memory=False,
        usecols=["Timestamp"] + EMOTION_COLS,
    )
    click_ms = 1425523.0
    mask = (df["Timestamp"] >= click_ms - 2000) & (
        df["Timestamp"] <= click_ms
    )
    assert mask.sum() == 60


@pytest.mark.integration
def test_spotcheck_imotions_50ms_frames():
    """Verify 1 frame in the 50ms window for player A sell event."""
    df = pd.read_csv(
        IMOTIONS_PLAYER_A,
        skiprows=IMOTIONS_SKIP_ROWS,
        encoding="utf-8-sig",
        low_memory=False,
        usecols=["Timestamp"] + EMOTION_COLS,
    )
    click_ms = 1425523.0
    mask = (df["Timestamp"] >= click_ms - 50) & (
        df["Timestamp"] <= click_ms
    )
    assert mask.sum() == 1


@pytest.mark.integration
def test_spotcheck_output_values(output_df):
    """Verify output CSV has correct n_frames and anger_mean."""
    mask = (
        (output_df["session_id"] == "1_11-7-tr1")
        & (output_df["segment"] == 1)
        & (output_df["round"] == 2)
        & (output_df["period"] == 1)
        & (output_df["player"] == "A")
    )
    row = output_df[mask]
    assert len(row) == 1
    assert row["n_frames_2000ms"].values[0] == 60
    assert row["n_frames_50ms"].values[0] == 1
    assert row["anger_mean_2000ms"].values[0] == pytest.approx(
        0.1380434749, abs=1e-6
    )


# =====
# Test 2: Output row count matches raw sell events
# =====
@pytest.mark.integration
def test_output_row_count_matches_sell_events(output_df):
    """Total rows should equal total sell events across all sessions."""
    segments = {
        1: "chat_noavg", 2: "chat_noavg2",
        3: "chat_noavg3", 4: "chat_noavg4",
    }
    total_sells = 0
    for session_id in INTEGRATION_SESSIONS:
        for seg_name in segments.values():
            csv_files = list(
                (DATASTORE / session_id).glob(f"{seg_name}_*.csv")
            )
            if not csv_files:
                continue
            df = pd.read_csv(csv_files[0])
            mask = (df["player.sold"] == 1) & (
                df["player.sell_click_time"].notna()
            )
            total_sells += mask.sum()

    assert len(output_df) == total_sells
    assert len(output_df) == 676


# =====
# Test 3: Traits merge correctness
# =====
@pytest.mark.integration
def test_traits_merge_correctness(output_df):
    """Extraversion from survey_traits matches output for session 2 player A."""
    traits = pd.read_csv(TRAITS_CSV)
    t = traits[
        (traits["session_id"] == "2_11-10-tr2") & (traits["player"] == "A")
    ]
    expected_extraversion = t["extraversion"].values[0]

    o = output_df[
        (output_df["session_id"] == "2_11-10-tr2")
        & (output_df["player"] == "A")
    ]
    assert len(o) > 0, "No rows for session 2 player A"
    assert (o["extraversion"] == expected_extraversion).all()
    assert expected_extraversion == pytest.approx(3.0)


# =====
# Test 4: Period data merge correctness
# =====
@pytest.mark.integration
def test_period_data_merge_correctness(output_df):
    """Period data columns match for the spot-check sell event."""
    period = pd.read_csv(PERIOD_CSV)
    keys = {
        "session_id": "1_11-7-tr1", "segment": 1,
        "round": 2, "period": 1, "player": "A",
    }
    p_mask = pd.Series(True, index=period.index)
    o_mask = pd.Series(True, index=output_df.index)
    for col, val in keys.items():
        p_mask &= period[col] == val
        o_mask &= output_df[col] == val

    p_row = period[p_mask]
    o_row = output_df[o_mask]
    assert len(p_row) == 1 and len(o_row) == 1

    for col in ["group_id", "treatment", "signal", "state", "price"]:
        assert o_row[col].values[0] == p_row[col].values[0], (
            f"{col} mismatch: output={o_row[col].values[0]}, "
            f"period={p_row[col].values[0]}"
        )


# =====
# Test 5: Nested window consistency
# =====
@pytest.mark.integration
def test_nested_window_consistency(output_df):
    """n_frames_2000ms >= n_frames_1000ms >= ... >= n_frames_50ms."""
    windows = PRESELL_WINDOWS_MS
    for i in range(len(windows) - 1):
        larger = f"n_frames_{windows[i]}ms"
        smaller = f"n_frames_{windows[i + 1]}ms"
        violations = (output_df[larger] < output_df[smaller]).sum()
        assert violations == 0, (
            f"{larger} < {smaller} in {violations} rows"
        )


# =====
# Test 6: Zero-frame events have NaN emotions
# =====
@pytest.mark.integration
def test_zero_frame_events_have_nan_emotions(output_df):
    """Rows with n_frames_2000ms==0 must have NaN for all emotion cols."""
    zero = output_df[output_df["n_frames_2000ms"] == 0]
    assert len(zero) == 2, f"Expected 2 zero-frame rows, got {len(zero)}"

    emotion_names = [c.lower() for c in EMOTION_COLS]
    for w in PRESELL_WINDOWS_MS:
        for e in emotion_names:
            col = f"{e}_mean_{w}ms"
            assert zero[col].isna().all(), (
                f"{col} is not NaN for zero-frame rows"
            )


@pytest.mark.integration
def test_zero_frame_events_are_player_l_session1_seg4(output_df):
    """Zero-frame rows belong to player L, session 1, segment 4."""
    zero = output_df[output_df["n_frames_2000ms"] == 0]
    assert (zero["player"] == "L").all()
    assert (zero["session_id"] == "1_11-7-tr1").all()
    assert (zero["segment"] == 4).all()


# =====
# Test 7: All rows have sold=1
# =====
@pytest.mark.integration
def test_all_rows_sold(output_df):
    """Every row in the output must have sold=1."""
    assert (output_df["sold"] == 1).all()


# =====
# Test 8: No duplicate rows
# =====
@pytest.mark.integration
def test_no_duplicate_rows(output_df):
    """No duplicates on merge keys (session_id, segment, round, period, player)."""
    keys = ["session_id", "segment", "round", "period", "player"]
    dupes = output_df.duplicated(subset=keys).sum()
    assert dupes == 0, f"Found {dupes} duplicate rows on merge keys"


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
