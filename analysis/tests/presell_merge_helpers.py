"""
Purpose: Mirror of R merge logic for pre-sell window Cox regressions (issue #118).
Used by test_cox_presell_windows_merge.py to validate merge semantics in isolation.
Author: Claude Code
Date: 2026-04-20
"""

from pathlib import Path
import pandas as pd
import numpy as np

# FILE PATHS
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = REPO_ROOT / "datastore" / "derived" / "emotions_traits_selling_dataset.csv"
PRESELL_PATH = REPO_ROOT / "datastore" / "derived" / "presell_emotions_traits_dataset.csv"

# CONSTANTS
WINDOWS = [50, 100, 500, 1000, 2000]
BASE_EMOTION_COLS = [
    "fear_mean", "anger_mean", "contempt_mean", "disgust_mean",
    "joy_mean", "sadness_mean", "surprise_mean",
    "engagement_mean", "valence_mean",
]
MERGE_KEYS = ["session_id", "segment", "round", "period", "group_id", "player"]


# =====
# Main (smoke test)
# =====
def main():
    """Load both datasets and run one merge to confirm helpers work."""
    base = pd.read_csv(BASE_PATH)
    presell = pd.read_csv(PRESELL_PATH)
    merged = merge_presell_window(base, presell, 500)
    filtered, n_dropped = drop_missing_window_rows(merged, 500)
    print(
        f"Base rows: {len(base)}; Presell rows: {len(presell)}; "
        f"Merged: {len(merged)}; After drop: {len(filtered)}; Dropped: {n_dropped}"
    )


# =====
# Window column naming
# =====
def window_emotion_cols(window):
    """Return list of presell emotion column names for a given window."""
    return [f"{col}_{window}ms" for col in BASE_EMOTION_COLS]


def window_n_frames_col(window):
    """Return the n_frames column name for a given window."""
    return f"n_frames_{window}ms"


# =====
# Merge logic
# =====
def merge_presell_window(base_df, presell_df, window):
    """Left-join presell window columns onto base and overwrite BASE_EMOTION_COLS.

    For sold==1 rows with a non-NaN presell match on the window emotion columns,
    the base emotion columns are overwritten with the window values. For all
    other rows (sold==0, no match, or NaN window value), base values are kept.
    Raises if the merge duplicates rows.
    """
    window_cols = window_emotion_cols(window)
    n_frames_col = window_n_frames_col(window)
    presell_subset = presell_df[MERGE_KEYS + window_cols + [n_frames_col]].copy()
    merged = base_df.merge(presell_subset, on=MERGE_KEYS, how="left")
    if len(merged) != len(base_df):
        raise ValueError(
            f"Merge duplicated rows: base={len(base_df)}, merged={len(merged)}"
        )
    _overwrite_base_with_window(merged, window)
    return merged


def _overwrite_base_with_window(merged, window):
    """In-place overwrite of BASE_EMOTION_COLS with window values where valid."""
    window_cols = window_emotion_cols(window)
    mask = _window_apply_mask(merged, window)
    for base_col, win_col in zip(BASE_EMOTION_COLS, window_cols):
        merged.loc[mask, base_col] = merged.loc[mask, win_col]


def _window_apply_mask(merged, window):
    """Rows where window values should overwrite base emotions.

    Applies when sold==1 AND all window emotion columns are non-NaN. Rows that
    did not match on the merge have NaN across all window columns, so they are
    excluded automatically.
    """
    window_cols = window_emotion_cols(window)
    sold_mask = merged["sold"] == 1
    non_nan_mask = merged[window_cols].notna().all(axis=1)
    return sold_mask & non_nan_mask


# =====
# Filtering sold==1 rows with missing window data
# =====
def drop_missing_window_rows(merged_df, window):
    """Drop sold==1 rows whose window emotion values are missing.

    Returns a (filtered_df, n_dropped) tuple. A row is dropped when sold==1 AND
    any of the window emotion columns are NaN (either no presell match or the
    window had zero valid frames).
    """
    drop_mask = _missing_window_mask(merged_df, window)
    filtered = merged_df.loc[~drop_mask].copy()
    return filtered, int(drop_mask.sum())


def _missing_window_mask(merged_df, window):
    """Boolean mask identifying sold==1 rows with NaN in any window column."""
    window_cols = window_emotion_cols(window)
    sold_mask = merged_df["sold"] == 1
    any_nan_mask = merged_df[window_cols].isna().any(axis=1)
    return sold_mask & any_nan_mask


if __name__ == "__main__":
    main()
