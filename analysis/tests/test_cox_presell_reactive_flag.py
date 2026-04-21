"""
Purpose: Tests for reactive_sale flag mirror (issue #118).
Validates semantics of `add_reactive_flag` against synthetic panels and
real-data invariants from the emotions_traits_selling_dataset.
Author: Claude Code
Date: 2026-04-21
"""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from reactive_flag_helpers import add_reactive_flag

# FILE PATHS
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CSV = REPO_ROOT / "datastore" / "derived" / "emotions_traits_selling_dataset.csv"


# =====
# Main
# =====
def main():
    """Run all tests with verbose output."""
    pytest.main([__file__, "-v"])


# =====
# Synthetic panel builders
# =====
def _row(session, seg, rnd, group, player, period, sold):
    """Build a single synthetic panel row."""
    return {
        "session_id": session, "segment": seg, "round": rnd,
        "group_id": group, "player": player, "period": period, "sold": sold,
    }


def _panel(rows):
    """Wrap list of row dicts into a DataFrame."""
    return pd.DataFrame(rows)


# =====
# Core semantics tests
# =====
def test_first_seller_not_reactive():
    """Player selling in period 1 has no T-1 — reactive_sale must be 0."""
    df = _panel([
        _row("s1", 1, 1, 1, 1, 1, 1),
        _row("s1", 1, 1, 1, 2, 1, 0),
    ])
    out = add_reactive_flag(df)
    seller = out[(out["player"] == 1) & (out["period"] == 1)]
    assert seller["reactive_sale"].iloc[0] == 0


def test_seller_after_T_minus_1_sale_is_reactive():
    """Group-mate sold in period 3, player sells in period 4 — reactive_sale=1."""
    df = _panel([
        _row("s1", 1, 1, 1, 1, 3, 1),
        _row("s1", 1, 1, 1, 2, 3, 0),
        _row("s1", 1, 1, 1, 1, 4, 0),
        _row("s1", 1, 1, 1, 2, 4, 1),
    ])
    out = add_reactive_flag(df)
    seller = out[(out["player"] == 2) & (out["period"] == 4)]
    assert seller["reactive_sale"].iloc[0] == 1


def test_seller_after_T_minus_2_gap_not_reactive():
    """Group-mate sold in period 2, nothing in period 3, player sells period 4 — reactive_sale=0."""
    df = _panel([
        _row("s1", 1, 1, 1, 1, 2, 1),
        _row("s1", 1, 1, 1, 2, 2, 0),
        _row("s1", 1, 1, 1, 1, 3, 0),
        _row("s1", 1, 1, 1, 2, 3, 0),
        _row("s1", 1, 1, 1, 1, 4, 0),
        _row("s1", 1, 1, 1, 2, 4, 1),
    ])
    out = add_reactive_flag(df)
    seller = out[(out["player"] == 2) & (out["period"] == 4)]
    assert seller["reactive_sale"].iloc[0] == 0


def test_non_seller_rows_always_zero():
    """sold==0 rows must have reactive_sale=0 even when group-mate sold in T-1."""
    df = _panel([
        _row("s1", 1, 1, 1, 1, 3, 1),
        _row("s1", 1, 1, 1, 2, 3, 0),
        _row("s1", 1, 1, 1, 1, 4, 0),
        _row("s1", 1, 1, 1, 2, 4, 0),
        _row("s1", 1, 1, 1, 3, 4, 0),
    ])
    out = add_reactive_flag(df)
    non_sellers = out[out["sold"] == 0]
    assert (non_sellers["reactive_sale"] == 0).all()


def test_sales_in_both_T_minus_1_and_T_minus_2_still_reactive():
    """Focal seller at T=4: player A sold at T-2 (period 2), player B sold at T-1 (period 3).
    The T-1 sale triggers reactive_sale=1; the additional T-2 sale does not block or suppress it.
    """
    df = _panel([
        _row("s1", 1, 1, 1, 1, 2, 1),
        _row("s1", 1, 1, 1, 2, 2, 0),
        _row("s1", 1, 1, 1, 3, 2, 0),
        _row("s1", 1, 1, 1, 1, 3, 0),
        _row("s1", 1, 1, 1, 2, 3, 1),
        _row("s1", 1, 1, 1, 3, 3, 0),
        _row("s1", 1, 1, 1, 1, 4, 0),
        _row("s1", 1, 1, 1, 2, 4, 0),
        _row("s1", 1, 1, 1, 3, 4, 1),
    ])
    out = add_reactive_flag(df)
    seller = out[(out["player"] == 3) & (out["period"] == 4)]
    assert seller["reactive_sale"].iloc[0] == 1


def test_different_groups_independent():
    """Group 1's period-2 sale must not affect group 2's period-3 seller."""
    df = _panel([
        _row("s1", 1, 1, 1, 1, 2, 1),
        _row("s1", 1, 1, 1, 2, 2, 0),
        _row("s1", 1, 1, 2, 3, 2, 0),
        _row("s1", 1, 1, 2, 4, 2, 0),
        _row("s1", 1, 1, 1, 1, 3, 0),
        _row("s1", 1, 1, 1, 2, 3, 0),
        _row("s1", 1, 1, 2, 3, 3, 1),
        _row("s1", 1, 1, 2, 4, 3, 0),
    ])
    out = add_reactive_flag(df)
    g2_seller = out[(out["group_id"] == 2) & (out["player"] == 3) & (out["period"] == 3)]
    assert g2_seller["reactive_sale"].iloc[0] == 0


def test_different_rounds_independent():
    """Round 1 period 14 sale must NOT carry into round 2 period 1."""
    df = _panel([
        _row("s1", 1, 1, 1, 1, 14, 1),
        _row("s1", 1, 1, 1, 2, 14, 0),
        _row("s1", 1, 2, 1, 1, 1, 0),
        _row("s1", 1, 2, 1, 2, 1, 1),
    ])
    out = add_reactive_flag(df)
    r2_seller = out[(out["round"] == 2) & (out["player"] == 2) & (out["period"] == 1)]
    assert r2_seller["reactive_sale"].iloc[0] == 0


def test_different_segments_independent():
    """Segment 1 period 14 sale must NOT carry into segment 2 period 1."""
    df = _panel([
        _row("s1", 1, 1, 1, 1, 14, 1),
        _row("s1", 1, 1, 1, 2, 14, 0),
        _row("s1", 2, 1, 1, 1, 1, 0),
        _row("s1", 2, 1, 1, 2, 1, 1),
    ])
    out = add_reactive_flag(df)
    seg2_seller = out[(out["segment"] == 2) & (out["player"] == 2) & (out["period"] == 1)]
    assert seg2_seller["reactive_sale"].iloc[0] == 0


def test_output_has_required_columns():
    """Both group_sold_prev_period and reactive_sale must be added."""
    df = _panel([_row("s1", 1, 1, 1, 1, 1, 0)])
    out = add_reactive_flag(df)
    assert "group_sold_prev_period" in out.columns
    assert "reactive_sale" in out.columns


def test_input_not_mutated():
    """add_reactive_flag must not mutate the input dataframe."""
    df = _panel([_row("s1", 1, 1, 1, 1, 1, 1)])
    _ = add_reactive_flag(df)
    assert "reactive_sale" not in df.columns


def test_missing_columns_raises():
    """Missing required columns must raise ValueError."""
    df = pd.DataFrame({"session_id": ["s1"], "period": [1], "sold": [1]})
    with pytest.raises(ValueError, match="missing columns"):
        add_reactive_flag(df)


# =====
# Real-data invariants
# =====
@pytest.fixture(scope="module")
def real_base():
    """Load base panel and apply the already_sold==0 filter Cox uses."""
    if not BASE_CSV.exists():
        pytest.skip(f"Base dataset missing: {BASE_CSV}")
    df = pd.read_csv(BASE_CSV)
    return df[df["already_sold"] == 0].copy()


def test_real_data_invariants(real_base):
    """On the real panel: reactive_sale is bounded, all reactive rows are sold==1, no NaN."""
    out = add_reactive_flag(real_base)
    total_sold = int(out["sold"].sum())
    total_reactive = int(out["reactive_sale"].sum())
    assert 0 <= total_reactive <= total_sold
    reactive_rows = out[out["reactive_sale"] == 1]
    assert (reactive_rows["sold"] == 1).all()
    assert not out["reactive_sale"].isna().any()
    assert not out["group_sold_prev_period"].isna().any()


def test_real_data_summary(real_base, capsys):
    """Print real-data totals for visibility (total_sold, total_reactive, ratio)."""
    out = add_reactive_flag(real_base)
    total_sold = int(out["sold"].sum())
    total_reactive = int(out["reactive_sale"].sum())
    ratio = total_reactive / total_sold if total_sold else float("nan")
    with capsys.disabled():
        print(f"\n[reactive-flag] total_sold={total_sold}, total_reactive={total_reactive}, ratio={ratio:.4f}")


# %%
if __name__ == "__main__":
    main()
