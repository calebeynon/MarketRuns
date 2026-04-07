"""
Purpose: Integration tests for build_presell_emotions.py (require datastore)
Author: Claude Code
Date: 2026-04-05
"""

from pathlib import Path

import pytest
import pandas as pd
from analysis.derived.build_presell_emotions import (
    extract_window_emotions,
    excel_to_epoch,
    EMOTION_COLS,
    IMOTIONS_SKIP_ROWS,
    PRESELL_WINDOWS_MS,
)


# =====
# Constants
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

SEGMENTS = {
    1: "chat_noavg", 2: "chat_noavg2",
    3: "chat_noavg3", 4: "chat_noavg4",
}


@pytest.fixture(scope="module")
def output_df():
    """Load the output dataset once for all integration tests."""
    return pd.read_csv(OUTPUT_CSV)


# =====
# Helpers
# =====
def _count_sells_across_sessions():
    """Count total sell events across all sessions and segments."""
    total = 0
    for session_id in INTEGRATION_SESSIONS:
        for seg_name in SEGMENTS.values():
            csv_files = list(
                (DATASTORE / session_id).glob(f"{seg_name}_*.csv")
            )
            if not csv_files:
                continue
            df = pd.read_csv(csv_files[0])
            mask = (df["player.sold"] == 1) & (
                df["player.sell_click_time"].notna()
            )
            total += mask.sum()
    return total


def _filter_by_keys(df, keys):
    """Filter DataFrame to rows matching all key-value pairs."""
    mask = pd.Series(True, index=df.index)
    for col, val in keys.items():
        mask &= df[col] == val
    return df[mask]


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
    total_sells = _count_sells_across_sessions()
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
    p_row = _filter_by_keys(period, keys)
    o_row = _filter_by_keys(output_df, keys)
    assert len(p_row) == 1 and len(o_row) == 1

    for col in ["group_id", "treatment", "signal", "state", "price"]:
        assert o_row[col].values[0] == p_row[col].values[0], (
            f"{col} mismatch"
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
