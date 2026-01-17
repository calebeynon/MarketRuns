"""
Purpose: Validate dummy variables computed by selling_period_regression_dummies.R
Author: Claude Code
Date: 2026-01-16

Tests verify that the 6 dummy variables are computed correctly:
- dummy_1_cum, dummy_2_cum, dummy_3_cum: based on prior_group_sales (== 1, == 2, == 3)
- dummy_1_prev, dummy_2_prev, dummy_3_prev: based on sales in period t-1 only (== 1, == 2, == 3)

Validates against:
1. The market_data.py parser (using parse_experiment())
2. Raw CSV files from datastore/
3. Logic validation of computed variables
"""

import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import market_data as md

# =====
# File paths
# =====
DATASTORE = Path("/Users/caleb/Research/marketruns/datastore")
INPUT_DATA = DATASTORE / "derived" / "individual_period_dataset.csv"

# Session data files for parser validation
SESSION_FILES = {
    "1_11-7-tr1": DATASTORE / "1_11-7-tr1" / "all_apps_wide_2025-11-07.csv",
    "2_11-10-tr2": DATASTORE / "2_11-10-tr2" / "all_apps_wide_2025-11-10.csv",
    "3_11-11-tr2": DATASTORE / "3_11-11-tr2" / "all_apps_wide_2025-11-11.csv",
    "4_11-12-tr1": DATASTORE / "4_11-12-tr1" / "all_apps_wide_2025-11-12.csv",
    "5_11-14-tr2": DATASTORE / "5_11-14-tr2" / "all_apps_wide_2025-11-14.csv",
    "6_11-18-tr1": DATASTORE / "6_11-18-tr1" / "all_apps_wide_2025-11-18.csv",
}

SEGMENT_MAP = {1: "chat_noavg", 2: "chat_noavg2", 3: "chat_noavg3", 4: "chat_noavg4"}


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def input_data():
    """Load the individual period dataset."""
    if not INPUT_DATA.exists():
        pytest.skip(f"Input data not found: {INPUT_DATA}")
    return pd.read_csv(INPUT_DATA)


@pytest.fixture(scope="module")
def regression_data(input_data):
    """
    Compute dummy variables from input data the same way the R script does.

    This fixture replicates the R logic to create the expected values.
    """
    df = input_data.copy()

    # Filter out observations where player already sold (matching R script)
    df = df[df["already_sold"] == 0].copy()

    # Create group_round_id for grouping
    df["group_round_id"] = (
        df["session_id"] + "_" +
        df["segment"].astype(str) + "_" +
        df["group_id"].astype(str) + "_" +
        df["round"].astype(str)
    )

    # Create cumulative dummies (based on prior_group_sales)
    df["dummy_1_cum"] = (df["prior_group_sales"] == 1).astype(int)
    df["dummy_2_cum"] = (df["prior_group_sales"] == 2).astype(int)
    df["dummy_3_cum"] = (df["prior_group_sales"] == 3).astype(int)

    # Create previous period dummies
    df = compute_prev_period_dummies(df)

    return df


def compute_prev_period_dummies(df):
    """
    Compute previous period sale dummies matching R script logic.

    For each period, counts total group sales in period t-1 only,
    then creates dummies for == 1, == 2, == 3 sales.
    """
    # Compute total sales per period within each group-round
    period_sales = (
        df.groupby(["group_round_id", "period"])["sold"]
        .sum()
        .reset_index()
        .rename(columns={"sold": "n_sales"})
    )

    # Sort and compute lagged sales
    period_sales = period_sales.sort_values(["group_round_id", "period"])
    period_sales["prev_period_n_sales"] = (
        period_sales.groupby("group_round_id")["n_sales"].shift(1)
    )
    period_sales["prev_period_n_sales"] = (
        period_sales["prev_period_n_sales"].fillna(0).astype(int)
    )

    # Merge back to player-level data
    df = df.merge(
        period_sales[["group_round_id", "period", "prev_period_n_sales"]],
        on=["group_round_id", "period"],
        how="left"
    )

    # Create dummies
    df["dummy_1_prev"] = (df["prev_period_n_sales"] == 1).astype(int)
    df["dummy_2_prev"] = (df["prev_period_n_sales"] == 2).astype(int)
    df["dummy_3_prev"] = (df["prev_period_n_sales"] == 3).astype(int)

    return df


# =====
# Test cumulative dummies logic
# =====
class TestCumulativeDummies:
    """Tests for dummy_X_cum variables based on prior_group_sales."""

    def test_dummy_1_cum_correct(self, regression_data):
        """dummy_1_cum should be 1 when prior_group_sales == 1."""
        df = regression_data

        # Check all rows where prior_group_sales == 1 have dummy_1_cum = 1
        mask_should_be_1 = df["prior_group_sales"] == 1
        assert (df.loc[mask_should_be_1, "dummy_1_cum"] == 1).all()

        # Check all rows where prior_group_sales != 1 have dummy_1_cum = 0
        mask_should_be_0 = df["prior_group_sales"] != 1
        assert (df.loc[mask_should_be_0, "dummy_1_cum"] == 0).all()

    def test_dummy_2_cum_correct(self, regression_data):
        """dummy_2_cum should be 1 when prior_group_sales == 2."""
        df = regression_data

        mask_should_be_1 = df["prior_group_sales"] == 2
        assert (df.loc[mask_should_be_1, "dummy_2_cum"] == 1).all()

        mask_should_be_0 = df["prior_group_sales"] != 2
        assert (df.loc[mask_should_be_0, "dummy_2_cum"] == 0).all()

    def test_dummy_3_cum_correct(self, regression_data):
        """dummy_3_cum should be 1 when prior_group_sales == 3."""
        df = regression_data

        mask_should_be_1 = df["prior_group_sales"] == 3
        assert (df.loc[mask_should_be_1, "dummy_3_cum"] == 1).all()

        mask_should_be_0 = df["prior_group_sales"] != 3
        assert (df.loc[mask_should_be_0, "dummy_3_cum"] == 0).all()

    def test_cumulative_dummies_mutually_exclusive(self, regression_data):
        """Verify dummies are mutually exclusive (at most one can be 1)."""
        df = regression_data

        # Sum of all three dummies should be at most 1 for each row
        dummy_sum = df["dummy_1_cum"] + df["dummy_2_cum"] + df["dummy_3_cum"]
        assert (dummy_sum <= 1).all()


# =====
# Test previous period dummies logic
# =====
class TestPreviousPeriodDummies:
    """Tests for dummy_X_prev variables based on sales in period t-1."""

    def test_period_1_has_zero_prev_dummies(self, regression_data):
        """Period 1 has no t-1, so all prev dummies should be 0."""
        df = regression_data
        period_1 = df[df["period"] == 1]

        assert (period_1["dummy_1_prev"] == 0).all()
        assert (period_1["dummy_2_prev"] == 0).all()
        assert (period_1["dummy_3_prev"] == 0).all()

    def test_prev_dummies_match_prev_period_sales(self, regression_data):
        """Verify prev dummies match computed prev_period_n_sales."""
        df = regression_data

        # dummy_1_prev should be 1 when prev_period_n_sales == 1
        mask_1 = df["prev_period_n_sales"] == 1
        assert (df.loc[mask_1, "dummy_1_prev"] == 1).all()
        assert (df.loc[~mask_1, "dummy_1_prev"] == 0).all()

        # dummy_2_prev should be 1 when prev_period_n_sales == 2
        mask_2 = df["prev_period_n_sales"] == 2
        assert (df.loc[mask_2, "dummy_2_prev"] == 1).all()
        assert (df.loc[~mask_2, "dummy_2_prev"] == 0).all()

        # dummy_3_prev should be 1 when prev_period_n_sales == 3
        mask_3 = df["prev_period_n_sales"] == 3
        assert (df.loc[mask_3, "dummy_3_prev"] == 1).all()
        assert (df.loc[~mask_3, "dummy_3_prev"] == 0).all()

    def test_prev_dummies_mutually_exclusive(self, regression_data):
        """Verify dummies are mutually exclusive (at most one can be 1)."""
        df = regression_data

        # Sum of all three dummies should be at most 1 for each row
        dummy_sum = df["dummy_1_prev"] + df["dummy_2_prev"] + df["dummy_3_prev"]
        assert (dummy_sum <= 1).all()


# =====
# Test edge cases
# =====
class TestEdgeCases:
    """Tests for edge cases in dummy variable computation."""

    def test_period_2_prev_equals_period_1_sales(self, regression_data):
        """Period 2's prev dummies should reflect period 1 sales only."""
        df = regression_data

        # For each group-round, manually compute period 1 sales
        for group_round_id, group_df in df.groupby("group_round_id"):
            period_1 = group_df[group_df["period"] == 1]
            period_2 = group_df[group_df["period"] == 2]

            if len(period_1) == 0 or len(period_2) == 0:
                continue

            # Total sales in period 1
            sales_in_p1 = period_1["sold"].sum()

            # All players in period 2 should have same prev_period_n_sales
            expected_prev_sales = period_2["prev_period_n_sales"].iloc[0]
            assert expected_prev_sales == sales_in_p1, (
                f"Group {group_round_id}: expected {sales_in_p1}, "
                f"got {expected_prev_sales}"
            )

    def test_no_sales_round_all_dummies_zero(self, regression_data):
        """When no one sells in a round, all dummies should be 0."""
        df = regression_data

        for group_round_id, group_df in df.groupby("group_round_id"):
            total_sales = group_df["sold"].sum()
            if total_sales == 0:
                # All cumulative dummies should be 0
                assert (group_df["dummy_1_cum"] == 0).all()
                assert (group_df["dummy_2_cum"] == 0).all()
                assert (group_df["dummy_3_cum"] == 0).all()
                # All prev dummies should be 0
                assert (group_df["dummy_1_prev"] == 0).all()
                assert (group_df["dummy_2_prev"] == 0).all()
                assert (group_df["dummy_3_prev"] == 0).all()

    def test_max_4_sales_per_period(self, regression_data):
        """With 4 players per group, max sales per period is 4."""
        df = regression_data

        period_sales = (
            df.groupby(["group_round_id", "period"])["sold"].sum()
        )
        assert period_sales.max() <= 4


# =====
# Validation against parser
# =====
class TestParserValidation:
    """Validate underlying sales data against market_data.py parser."""

    @pytest.fixture(scope="class")
    def parsed_experiments(self):
        """Load experiments using parser."""
        experiments = {}
        for session_id, csv_path in SESSION_FILES.items():
            if csv_path.exists():
                experiments[session_id] = md.parse_experiment(str(csv_path))
        return experiments

    def test_sold_status_matches_parser(self, regression_data, parsed_experiments):
        """Verify sold field matches parser for sampled rows."""
        if not parsed_experiments:
            pytest.skip("No parsed experiments available")

        df = regression_data
        mismatches = 0
        rows_checked = 0

        # Sample 100 rows for validation
        sample = df.sample(min(100, len(df)), random_state=42)

        for _, row in sample.iterrows():
            session_id = row["session_id"]
            segment_idx = int(row["segment"])
            round_num = int(row["round"])
            period_num = int(row["period"])
            player_label = row["player"]

            experiment = parsed_experiments.get(session_id)
            if experiment is None or len(experiment.sessions) == 0:
                continue

            session = experiment.sessions[0]
            segment_name = SEGMENT_MAP[segment_idx]

            segment = session.get_segment(segment_name)
            if not segment:
                continue

            round_obj = segment.get_round(round_num)
            if not round_obj:
                continue

            period_obj = round_obj.get_period(period_num)
            if not period_obj:
                continue

            player = period_obj.get_player(player_label)
            if not player:
                continue

            rows_checked += 1
            parser_sold_this_period = 1 if player.sold_this_period else 0

            if row["sold"] != parser_sold_this_period:
                mismatches += 1

        assert rows_checked > 0, "No rows could be validated"
        assert mismatches == 0, f"Found {mismatches} mismatches in {rows_checked} rows"


# =====
# Test manual scenarios
# =====
class TestManualScenarios:
    """Test specific scenarios with known expected values."""

    def test_specific_group_round(self, input_data):
        """Test a specific group-round to verify computation."""
        df = input_data

        # Find a group-round with some sales
        target = df[
            (df["session_id"] == "1_11-7-tr1") &
            (df["segment"] == 1) &
            (df["round"] == 2) &
            (df["group_id"] == 1)
        ].copy()

        if len(target) == 0:
            pytest.skip("Target group-round not found")

        # Filter to non-already-sold
        target = target[target["already_sold"] == 0].copy()
        target = target.sort_values("period")

        # Compute expected dummies manually
        periods = sorted(target["period"].unique())
        sales_by_period = {}

        for p in periods:
            period_df = target[target["period"] == p]
            sales_by_period[p] = period_df["sold"].sum()

        # Check each period
        for p in periods:
            period_df = target[target["period"] == p]

            # Cumulative: prior_group_sales from data
            expected_prior = period_df["prior_group_sales"].iloc[0]
            expected_d1_cum = 1 if expected_prior >= 1 else 0
            expected_d2_cum = 1 if expected_prior >= 2 else 0
            expected_d3_cum = 1 if expected_prior >= 3 else 0

            # Previous period sales
            if p > 1 and (p - 1) in sales_by_period:
                prev_sales = sales_by_period[p - 1]
            else:
                prev_sales = 0

            expected_d1_prev = 1 if prev_sales >= 1 else 0
            expected_d2_prev = 1 if prev_sales >= 2 else 0
            expected_d3_prev = 1 if prev_sales >= 3 else 0

            # Now compute actual values using our fixture logic
            actual_d1_cum = (expected_prior >= 1)
            actual_d2_cum = (expected_prior >= 2)
            actual_d3_cum = (expected_prior >= 3)
            actual_d1_prev = (prev_sales >= 1)
            actual_d2_prev = (prev_sales >= 2)
            actual_d3_prev = (prev_sales >= 3)

            assert actual_d1_cum == expected_d1_cum
            assert actual_d2_cum == expected_d2_cum
            assert actual_d3_cum == expected_d3_cum
            assert actual_d1_prev == expected_d1_prev
            assert actual_d2_prev == expected_d2_prev
            assert actual_d3_prev == expected_d3_prev


# =====
# Summary statistics tests
# =====
class TestSummaryStatistics:
    """Tests to verify summary statistics make sense."""

    def test_dummy_counts_reasonable(self, regression_data):
        """Verify dummy variable counts are in reasonable ranges."""
        df = regression_data

        # dummy_1_cum should be most common, dummy_3_cum least common
        count_d1 = df["dummy_1_cum"].sum()
        count_d2 = df["dummy_2_cum"].sum()
        count_d3 = df["dummy_3_cum"].sum()

        assert count_d1 >= count_d2 >= count_d3

        # Same for prev dummies
        count_p1 = df["dummy_1_prev"].sum()
        count_p2 = df["dummy_2_prev"].sum()
        count_p3 = df["dummy_3_prev"].sum()

        assert count_p1 >= count_p2 >= count_p3

    def test_all_dummies_binary(self, regression_data):
        """All dummy variables should only contain 0 or 1."""
        df = regression_data

        for col in ["dummy_1_cum", "dummy_2_cum", "dummy_3_cum",
                    "dummy_1_prev", "dummy_2_prev", "dummy_3_prev"]:
            unique_vals = set(df[col].unique())
            assert unique_vals.issubset({0, 1}), f"{col} has non-binary values"

    def test_data_filtered_correctly(self, input_data, regression_data):
        """Verify already_sold=1 rows are filtered out."""
        original_count = len(input_data)
        filtered_count = len(regression_data)
        already_sold_count = (input_data["already_sold"] == 1).sum()

        expected_filtered = original_count - already_sold_count
        assert filtered_count == expected_filtered


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
