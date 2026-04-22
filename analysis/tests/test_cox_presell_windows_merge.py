"""
Purpose: Tests for pre-sell window merge logic (issue #118).
Verifies merge semantics against synthetic fixtures and real datasets.
Author: Claude Code
Date: 2026-04-20
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from presell_merge_helpers import (
    BASE_EMOTION_COLS,
    MERGE_KEYS,
    WINDOWS,
    drop_missing_window_rows,
    merge_presell_window,
    window_emotion_cols,
    window_n_frames_col,
)

# FILE PATHS
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CSV = REPO_ROOT / "datastore" / "derived" / "emotions_traits_selling_dataset.csv"
PRESELL_CSV = REPO_ROOT / "datastore" / "derived" / "presell_emotions_traits_dataset.csv"

# SENTINELS
BASE_SENTINEL = 0.1
PRESELL_SENTINEL_50 = 0.9
PRESELL_SENTINEL_2000 = 0.5


# =====
# Main function (FIRST)
# =====
def main():
    """Run all tests with verbose output."""
    pytest.main([__file__, "-v"])


# =====
# Synthetic fixture builders
# =====
def _make_base_row(session, seg, rnd, period, group, player, sold):
    """Build a single row dict for the synthetic base frame."""
    row = {
        "session_id": session, "segment": seg, "round": rnd,
        "period": period, "group_id": group, "player": player,
        "sold": sold, "already_sold": 0, "signal": 1,
    }
    for col in BASE_EMOTION_COLS:
        row[col] = BASE_SENTINEL
    return row


def _make_presell_row(session, seg, rnd, period, group, player, val, n_frames):
    """Build a presell row with all window columns set to val, n_frames as given."""
    row = {
        "session_id": session, "segment": seg, "round": rnd,
        "period": period, "group_id": group, "player": player,
    }
    for window in WINDOWS:
        row[window_n_frames_col(window)] = n_frames
        for col in window_emotion_cols(window):
            row[col] = val if n_frames > 0 else np.nan
    return row


# =====
# Fixtures
# =====
@pytest.fixture
def synthetic_base():
    """Small base frame with mix of sold==0 / sold==1 rows, all sentinel values."""
    rows = [
        _make_base_row("s1", 1, 1, 1, 1, 1, sold=0),
        _make_base_row("s1", 1, 1, 2, 1, 2, sold=1),
        _make_base_row("s1", 1, 1, 3, 1, 3, sold=1),
        _make_base_row("s1", 1, 1, 4, 1, 4, sold=1),
        _make_base_row("s1", 1, 2, 1, 1, 1, sold=0),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_presell():
    """Presell frame for a subset of sold==1 rows + one zero-frame row."""
    rows = [
        _make_presell_row("s1", 1, 1, 2, 1, 2, PRESELL_SENTINEL_50, n_frames=10),
        _make_presell_row("s1", 1, 1, 3, 1, 3, PRESELL_SENTINEL_2000, n_frames=0),
    ]
    presell = pd.DataFrame(rows)
    _overwrite_50ms_sentinel(presell, 0, PRESELL_SENTINEL_50)
    _overwrite_2000ms_sentinel(presell, 0, PRESELL_SENTINEL_2000)
    return presell


def _overwrite_50ms_sentinel(df, idx, val):
    """Mark the 50ms window columns with a distinct sentinel on one row."""
    for col in window_emotion_cols(50):
        df.loc[idx, col] = val


def _overwrite_2000ms_sentinel(df, idx, val):
    """Mark the 2000ms window columns with a distinct sentinel on one row."""
    for col in window_emotion_cols(2000):
        df.loc[idx, col] = val


@pytest.fixture(scope="module")
def real_base():
    """Load the real base CSV."""
    if not BASE_CSV.exists():
        pytest.skip(f"Base dataset missing: {BASE_CSV}")
    return pd.read_csv(BASE_CSV)


@pytest.fixture(scope="module")
def real_presell():
    """Load the real presell CSV."""
    if not PRESELL_CSV.exists():
        pytest.skip(f"Presell dataset missing: {PRESELL_CSV}")
    return pd.read_csv(PRESELL_CSV)


# =====
# TestMergeBasic
# =====
class TestMergeBasic:
    """Shape invariants: no duplication, keys preserved, base cols intact."""

    def test_no_row_duplication(self, synthetic_base, synthetic_presell):
        """Merge must preserve row count for every window."""
        for window in WINDOWS:
            merged = merge_presell_window(synthetic_base, synthetic_presell, window)
            assert len(merged) == len(synthetic_base), f"window {window} duplicated"

    def test_merge_keys_preserved(self, synthetic_base, synthetic_presell):
        """MERGE_KEYS still uniquely identify rows after merge."""
        merged = merge_presell_window(synthetic_base, synthetic_presell, 500)
        assert not merged.duplicated(subset=MERGE_KEYS).any()

    def test_base_columns_preserved(self, synthetic_base, synthetic_presell):
        """Non-emotion columns (already_sold, signal, sold) unchanged."""
        merged = merge_presell_window(synthetic_base, synthetic_presell, 500)
        for col in ["already_sold", "signal", "sold"]:
            assert (merged[col].values == synthetic_base[col].values).all()


# =====
# TestMergeSemantics
# =====
class TestMergeSemantics:
    """Overwrite rules: sold==0 keeps base, sold==1 takes window when matched."""

    def test_sold_0_rows_keep_base_values(self, synthetic_base, synthetic_presell):
        """For sold==0 rows, BASE_EMOTION_COLS remain at BASE_SENTINEL."""
        merged = merge_presell_window(synthetic_base, synthetic_presell, 500)
        zeros = merged[merged["sold"] == 0]
        for col in BASE_EMOTION_COLS:
            assert (zeros[col] == BASE_SENTINEL).all(), f"{col} changed on sold==0"

    def test_sold_1_rows_get_window_values(self, synthetic_base, synthetic_presell):
        """sold==1 matched rows take window sentinel values."""
        merged = merge_presell_window(synthetic_base, synthetic_presell, 50)
        matched = merged[(merged["sold"] == 1) & (merged["player"] == 2)]
        for col in BASE_EMOTION_COLS:
            assert matched[col].iloc[0] == pytest.approx(PRESELL_SENTINEL_50)

    def test_sold_1_no_match_keeps_base(self, synthetic_base, synthetic_presell):
        """sold==1 without presell match retains base sentinel."""
        merged = merge_presell_window(synthetic_base, synthetic_presell, 500)
        unmatched = merged[(merged["sold"] == 1) & (merged["player"] == 4)]
        for col in BASE_EMOTION_COLS:
            assert unmatched[col].iloc[0] == pytest.approx(BASE_SENTINEL)

    def test_different_windows_give_different_values(
        self, synthetic_base, synthetic_presell
    ):
        """50ms and 2000ms yield different overwritten values on matched sold==1."""
        merged_50 = merge_presell_window(synthetic_base, synthetic_presell, 50)
        merged_2000 = merge_presell_window(synthetic_base, synthetic_presell, 2000)
        matched_50 = merged_50[(merged_50["sold"] == 1) & (merged_50["player"] == 2)]
        matched_2000 = merged_2000[
            (merged_2000["sold"] == 1) & (merged_2000["player"] == 2)
        ]
        assert matched_50["anger_mean"].iloc[0] != matched_2000["anger_mean"].iloc[0]


# =====
# TestDropMissingWindow
# =====
class TestDropMissingWindow:
    """drop_missing_window_rows: removes only sold==1 rows with NaN window cols."""

    def test_drop_missing_removes_zero_frame_rows(
        self, synthetic_base, synthetic_presell
    ):
        """sold==1 rows whose window had n_frames==0 (NaN values) are dropped."""
        merged = merge_presell_window(synthetic_base, synthetic_presell, 500)
        filtered, n_dropped = drop_missing_window_rows(merged, 500)
        assert n_dropped == 2, f"expected 2 drops (player 3 zero-frame + player 4 unmatched)"
        assert len(filtered) == len(synthetic_base) - 2

    def test_drop_missing_preserves_sold_0(self, synthetic_base, synthetic_presell):
        """sold==0 rows are never dropped even with missing window data."""
        merged = merge_presell_window(synthetic_base, synthetic_presell, 500)
        filtered, _ = drop_missing_window_rows(merged, 500)
        zero_in = (synthetic_base["sold"] == 0).sum()
        zero_out = (filtered["sold"] == 0).sum()
        assert zero_in == zero_out

    def test_drop_missing_preserves_sold_1_with_data(
        self, synthetic_base, synthetic_presell
    ):
        """sold==1 rows with valid window values are kept."""
        merged = merge_presell_window(synthetic_base, synthetic_presell, 50)
        filtered, _ = drop_missing_window_rows(merged, 50)
        kept = filtered[(filtered["sold"] == 1) & (filtered["player"] == 2)]
        assert len(kept) == 1


# =====
# TestRealData
# =====
class TestRealData:
    """Regression check against the actual presell + base datasets."""

    def test_all_windows_produce_valid_merge_real_data(
        self, real_base, real_presell
    ):
        """Every window merges without duplication and yields a non-empty result."""
        base_rows = len(real_base)
        for window in WINDOWS:
            merged = merge_presell_window(real_base, real_presell, window)
            _assert_merge_shape(merged, base_rows)
            filtered, n_dropped = drop_missing_window_rows(merged, window)
            _assert_filter_shape(filtered, n_dropped, base_rows)


# =====
# Real-data shape assertions
# =====
def _assert_merge_shape(merged, base_rows):
    """Merged frame has exactly base_rows rows and unique merge keys."""
    assert len(merged) == base_rows
    assert not merged.duplicated(subset=MERGE_KEYS).any()


def _assert_filter_shape(filtered, n_dropped, base_rows):
    """Filtered frame has a non-negative drop count, valid bounds, surviving sales."""
    assert n_dropped >= 0
    assert 0 < len(filtered) <= base_rows
    assert (filtered["sold"] == 1).sum() > 0


# %%
if __name__ == "__main__":
    main()
