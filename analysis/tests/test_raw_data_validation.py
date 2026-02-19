"""
Purpose: Validate cumulative/previous-period dummies and interaction terms
         against raw data by tracing specific group-rounds and checking
         full-dataset consistency.
Author: Claude Code
Date: 2026-02-18
"""

import pandas as pd
import pytest
from pathlib import Path

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EMOTIONS_DATASET = (
    PROJECT_ROOT / "datastore" / "derived"
    / "emotions_traits_selling_dataset.csv"
)

# COLUMN LISTS
CUM_DUMMIES = ["dummy_1_cum", "dummy_2_cum", "dummy_3_cum"]
PREV_DUMMIES = ["dummy_1_prev", "dummy_2_prev", "dummy_3_prev"]
INTERACTION_TERMS = [
    "int_1_1", "int_2_1", "int_2_2",
    "int_3_1", "int_3_2", "int_3_3",
]
ALL_DUMMY_COLS = CUM_DUMMIES + PREV_DUMMIES + INTERACTION_TERMS
EXPECTED_FILTERED_ROWS = 13728


# =====
# Main function
# =====
def main():
    pytest.main([__file__, "-v"])


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def raw_df():
    """Load raw CSV without any dummy construction."""
    if not EMOTIONS_DATASET.exists():
        pytest.skip(f"Dataset not found: {EMOTIONS_DATASET}")
    return pd.read_csv(EMOTIONS_DATASET)


@pytest.fixture(scope="module")
def filtered_df(raw_df):
    """Filter to already_sold == 0 and add constructed columns."""
    df = raw_df[raw_df["already_sold"] == 0].copy()
    df["group_round_id"] = _build_group_round_id(df)
    df = _add_cum_dummies(df)
    df = _add_prev_period_dummies(df)
    df = _add_interaction_terms(df)
    return df


# =====
# Variable construction helpers (replicate R logic in Python)
# =====
def _build_group_round_id(df):
    return (
        df["session_id"] + "_" + df["segment"].astype(str)
        + "_" + df["group_id"].astype(str)
        + "_" + df["round"].astype(str)
    )


def _add_cum_dummies(df):
    for n in [1, 2, 3]:
        df[f"dummy_{n}_cum"] = (df["prior_group_sales"] == n).astype(int)
    return df


def _compute_period_sales(df):
    """Sum sold per group-round-period, then lag by one period."""
    ps = (
        df.groupby(["group_round_id", "period"])["sold"]
        .sum().reset_index().rename(columns={"sold": "n_sales"})
    )
    ps = ps.sort_values(["group_round_id", "period"])
    ps["prev_n_sales"] = (
        ps.groupby("group_round_id")["n_sales"].shift(1)
    )
    ps["prev_n_sales"] = ps["prev_n_sales"].fillna(0).astype(int)
    return ps[["group_round_id", "period", "prev_n_sales"]]


def _add_prev_period_dummies(df):
    lagged = _compute_period_sales(df)
    df = df.merge(lagged, on=["group_round_id", "period"], how="left")
    for n in [1, 2, 3]:
        df[f"dummy_{n}_prev"] = (df["prev_n_sales"] == n).astype(int)
    return df


def _add_interaction_terms(df):
    for c, p in [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)]:
        df[f"int_{c}_{p}"] = df[f"dummy_{c}_cum"] * df[f"dummy_{p}_prev"]
    return df


def _get_group_round(df, session_id, segment, group_id, round_num):
    """Extract a single group-round from the filtered dataset."""
    mask = (
        (df["session_id"] == session_id)
        & (df["segment"] == segment)
        & (df["group_id"] == group_id)
        & (df["round"] == round_num)
    )
    return df[mask].sort_values(["period", "player"]).copy()


def _assert_period_cum(gr, period, expected_pgs, expected_dummies):
    """Assert prior_group_sales and cum dummy values for a period."""
    rows = gr[gr["period"] == period] if isinstance(period, int) else gr[gr["period"] >= period]
    assert len(rows) > 0, f"No rows for period {period}"
    assert (rows["prior_group_sales"] == expected_pgs).all()
    for i, val in enumerate(expected_dummies, start=1):
        assert (rows[f"dummy_{i}_cum"] == val).all()


def _assert_prev_n_sales(gr, period, expected):
    """Assert all rows in a given period have the expected prev_n_sales."""
    rows = gr[gr["period"] == period]
    if len(rows) == 0:
        return
    actual = rows["prev_n_sales"].unique()
    assert len(actual) == 1 and actual[0] == expected, (
        f"Period {period}: expected prev_n_sales={expected}, got {actual}"
    )


def _manual_prev_sales(df, row):
    """Compute prev_n_sales by looking up actual sales in prior period."""
    if row["period"] == 1:
        return 0
    prev = df[
        (df["group_round_id"] == row["group_round_id"])
        & (df["period"] == row["period"] - 1)
    ]
    return int(prev["sold"].sum())


# =====
# A. Validate cumulative dummies against raw prior_group_sales
# =====
class TestCumulativeDummiesVsRawData:

    def test_simultaneous_sales_period_1(self, filtered_df, raw_df):
        """1_11-7-tr1 seg=1 gid=1 round=2: period 1 has 0 prior sales."""
        gr = _get_group_round(filtered_df, "1_11-7-tr1", 1, 1, 2)
        _assert_period_cum(gr, 1, expected_pgs=0, expected_dummies=[0, 0, 0])

    def test_simultaneous_sales_period_2(self, filtered_df, raw_df):
        """Period 2: A sold in p1, so prior_group_sales=1 => dummy_1_cum=1."""
        gr = _get_group_round(filtered_df, "1_11-7-tr1", 1, 1, 2)
        _assert_period_cum(gr, 2, expected_pgs=1, expected_dummies=[1, 0, 0])

    def test_simultaneous_sales_period_3_plus(self, filtered_df, raw_df):
        """Period 3+: 3 prior sales (A in p1, J+N in p2) => dummy_3_cum=1."""
        gr = _get_group_round(filtered_df, "1_11-7-tr1", 1, 1, 2)
        p3_plus = gr[gr["period"] >= 3]
        assert len(p3_plus) > 0
        assert (p3_plus["prior_group_sales"] == 3).all()
        assert (p3_plus["dummy_3_cum"] == 1).all()

    def test_sequential_sales(self, filtered_df, raw_df):
        """3_11-11-tr2 seg=1 gid=4 round=2: sales progress 0->1->2."""
        gr = _get_group_round(filtered_df, "3_11-11-tr2", 1, 4, 2)
        _assert_period_cum(gr, 1, expected_pgs=0, expected_dummies=[0, 0, 0])
        _assert_period_cum(gr, 2, expected_pgs=1, expected_dummies=[1, 0, 0])
        _assert_period_cum(gr, 3, expected_pgs=2, expected_dummies=[0, 1, 0])

    def test_prior_group_sales_matches_running_count(self, filtered_df, raw_df):
        """Recompute prior_group_sales from sold column for 50 group-rounds."""
        all_df = raw_df.copy()
        all_df["group_round_id"] = _build_group_round_id(all_df)
        for gri in filtered_df["group_round_id"].unique()[:50]:
            _verify_prior_sales(all_df, filtered_df, gri)

    def test_zero_sales_round(self, filtered_df):
        """1_11-7-tr1 seg=1 gid=1 round=1: no sales, all dummies 0."""
        gr = _get_group_round(filtered_df, "1_11-7-tr1", 1, 1, 1)
        assert len(gr) > 0
        assert (gr["prior_group_sales"] == 0).all()
        for col in CUM_DUMMIES:
            assert (gr[col] == 0).all()


def _verify_prior_sales(all_df, filtered_df, gri):
    """Recompute prior_group_sales from raw sold column for one group-round."""
    raw_gr = all_df[all_df["group_round_id"] == gri]
    filt_gr = filtered_df[filtered_df["group_round_id"] == gri]
    if len(filt_gr) == 0:
        return
    sales = raw_gr.groupby("period")["sold"].sum().sort_index()
    cum = sales.cumsum().shift(1).fillna(0).astype(int)
    for _, row in filt_gr.iterrows():
        expected = cum.get(row["period"], 0)
        assert row["prior_group_sales"] == expected, (
            f"{gri} p{row['period']}: pgs={row['prior_group_sales']} != {expected}"
        )


# =====
# B. Validate previous-period dummies against raw data
# =====
class TestPrevPeriodDummiesVsRawData:

    def test_double_sale_creates_prev_2(self, filtered_df):
        """1_11-7-tr1 seg=1 gid=1 round=2:
        p1=1 sale, p2=2 sales => prev_n_sales: p1=0, p2=1, p3=2, p4+=0."""
        gr = _get_group_round(filtered_df, "1_11-7-tr1", 1, 1, 2)
        _assert_prev_n_sales(gr, period=1, expected=0)
        _assert_prev_n_sales(gr, period=2, expected=1)
        _assert_prev_n_sales(gr, period=3, expected=2)
        for p in range(4, 11):
            _assert_prev_n_sales(gr, period=p, expected=0)

    def test_sequential_single_sales(self, filtered_df):
        """3_11-11-tr2 seg=1 gid=4 round=2:
        p1=1 sale, p2=1 sale, p3=2 sales => prev: p1=0, p2=1, p3=1."""
        gr = _get_group_round(filtered_df, "3_11-11-tr2", 1, 4, 2)
        _assert_prev_n_sales(gr, period=1, expected=0)
        _assert_prev_n_sales(gr, period=2, expected=1)
        _assert_prev_n_sales(gr, period=3, expected=1)

    def test_prev_dummies_match_prev_n_sales(self, filtered_df):
        """Verify dummy_X_prev == (prev_n_sales == X) for all rows."""
        for n in [1, 2, 3]:
            col = f"dummy_{n}_prev"
            expected = (filtered_df["prev_n_sales"] == n).astype(int)
            assert (filtered_df[col] == expected).all(), (
                f"{col} mismatches with prev_n_sales=={n}"
            )

    def test_manual_lag_sample(self, filtered_df):
        """For 100 sampled rows, look up sales in prior period directly."""
        sample = filtered_df.sample(100, random_state=42)
        for _, row in sample.iterrows():
            expected = _manual_prev_sales(filtered_df, row)
            assert row["prev_n_sales"] == expected, (
                f"{row['group_round_id']} p{row['period']}: "
                f"prev_n_sales={row['prev_n_sales']} != {expected}"
            )


# =====
# C. Validate interaction terms against both components
# =====
class TestInteractionTermsVsComponents:

    def test_all_interactions_equal_product(self, filtered_df):
        for c, p in [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)]:
            expected = filtered_df[f"dummy_{c}_cum"] * filtered_df[f"dummy_{p}_prev"]
            assert (filtered_df[f"int_{c}_{p}"] == expected).all()

    def test_nonzero_implies_both_dummies(self, filtered_df):
        """If int_X_Y == 1, both component dummies must be 1."""
        for c, p in [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)]:
            ones = filtered_df[filtered_df[f"int_{c}_{p}"] == 1]
            if len(ones) == 0:
                continue
            assert (ones[f"dummy_{c}_cum"] == 1).all()
            assert (ones[f"dummy_{p}_prev"] == 1).all()

    def test_specific_interaction_from_traced_round(self, filtered_df):
        """1_11-7-tr1 seg=1 gid=1 round=2 period 2:
        pgs=1, prev_n_sales=1 => int_1_1=1, all others 0."""
        gr = _get_group_round(filtered_df, "1_11-7-tr1", 1, 1, 2)
        p2 = gr[gr["period"] == 2]
        assert len(p2) > 0
        assert (p2["int_1_1"] == 1).all()
        for col in ["int_2_1", "int_2_2", "int_3_1", "int_3_2", "int_3_3"]:
            assert (p2[col] == 0).all(), f"{col} should be 0"


# =====
# D. Validate edge cases from raw data
# =====
class TestEdgeCases:

    def test_first_period_all_prev_and_interactions_zero(self, filtered_df):
        """Period 1 has no previous period."""
        p1 = filtered_df[filtered_df["period"] == 1]
        assert len(p1) > 0
        for col in PREV_DUMMIES + INTERACTION_TERMS:
            assert (p1[col] == 0).all(), f"{col} nonzero in period 1"

    def test_zero_prior_sales_implies_zero_cum_and_interactions(self, filtered_df):
        zero_pgs = filtered_df[filtered_df["prior_group_sales"] == 0]
        assert len(zero_pgs) > 0
        for col in CUM_DUMMIES + INTERACTION_TERMS:
            assert (zero_pgs[col] == 0).all(), f"{col} nonzero at pgs=0"

    def test_prior_group_sales_range(self, filtered_df):
        """Max is 3 (4 players minus self), min is 0."""
        assert filtered_df["prior_group_sales"].max() <= 3
        assert filtered_df["prior_group_sales"].min() >= 0

    def test_prev_n_sales_range(self, filtered_df):
        """At most 4 players can sell in one period."""
        assert filtered_df["prev_n_sales"].max() <= 4
        assert filtered_df["prev_n_sales"].min() >= 0


# =====
# E. Validate consistency across the full dataset
# =====
class TestFullDatasetConsistency:

    def test_filtered_row_count(self, filtered_df):
        assert len(filtered_df) == EXPECTED_FILTERED_ROWS

    def test_no_missing_values_in_dummies(self, filtered_df):
        for col in ALL_DUMMY_COLS:
            assert filtered_df[col].isna().sum() == 0, f"{col} has NAs"

    def test_all_dummies_are_binary(self, filtered_df):
        for col in ALL_DUMMY_COLS:
            vals = set(filtered_df[col].unique())
            assert vals.issubset({0, 1}), f"{col} non-binary: {vals}"

    def test_cum_dummies_mutually_exclusive(self, filtered_df):
        total = sum(filtered_df[col] for col in CUM_DUMMIES)
        assert (total <= 1).all()

    def test_prev_dummies_mutually_exclusive(self, filtered_df):
        total = sum(filtered_df[col] for col in PREV_DUMMIES)
        assert (total <= 1).all()

    def test_cum_dummy_counts_match_prior_group_sales(self, filtered_df):
        pgs_counts = filtered_df["prior_group_sales"].value_counts()
        for n in [1, 2, 3]:
            assert filtered_df[f"dummy_{n}_cum"].sum() == pgs_counts.get(n, 0)

    def test_no_missing_prior_group_sales(self, filtered_df):
        assert filtered_df["prior_group_sales"].isna().sum() == 0

    def test_max_four_players_per_period(self, filtered_df):
        counts = filtered_df.groupby(["group_round_id", "period"]).size()
        assert (counts <= 4).all()


# %%
if __name__ == "__main__":
    main()
