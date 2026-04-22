"""
Purpose: Python mirror of R `add_reactive_flag` logic (issue #118).
A reactive sale occurs when a player sells in period T and at least one
group-mate sold in period T-1 (same session/segment/round/group).
Used by test_cox_presell_reactive_flag.py to validate the R implementation.
Author: Claude Code
Date: 2026-04-21
"""

from pathlib import Path
import pandas as pd

# FILE PATHS
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = REPO_ROOT / "datastore" / "derived" / "emotions_traits_selling_dataset.csv"

# CONSTANTS
GROUP_KEYS = ["session_id", "segment", "round", "group_id"]
REQUIRED_COLS = GROUP_KEYS + ["period", "sold"]


# =====
# Public API
# =====
def add_reactive_flag(df):
    """Return a copy of df with `group_sold_prev_period` and `reactive_sale` columns.

    reactive_sale = 1 iff sold==1 AND any group-mate (same session/segment/
    round/group_id) sold==1 in period T-1. First-period sellers and sellers
    after a gap (nobody sold in T-1) get reactive_sale=0.
    """
    _validate_columns(df)
    out = df.copy()
    group_period_sold = _group_period_sold_sums(out)
    prev = _shift_to_next_period(group_period_sold)
    out = out.merge(prev, on=GROUP_KEYS + ["period"], how="left")
    out["group_sold_prev_period"] = (out["group_sold_prev_period"].fillna(0) > 0).astype(int)
    out["reactive_sale"] = (out["sold"].astype(int) * out["group_sold_prev_period"]).astype(int)
    return out


# =====
# Internals
# =====
def _validate_columns(df):
    """Raise if any required column is missing."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"add_reactive_flag missing columns: {missing}")


def _group_period_sold_sums(df):
    """Sum `sold` by (session, segment, round, group, period)."""
    return (
        df.groupby(GROUP_KEYS + ["period"], as_index=False)["sold"]
        .sum()
        .rename(columns={"sold": "group_sold_prev_period"})
    )


def _shift_to_next_period(group_period_sold):
    """Shift period forward by 1 so a row at period T carries T-1's group sold sum."""
    shifted = group_period_sold.copy()
    shifted["period"] = shifted["period"] + 1
    return shifted
