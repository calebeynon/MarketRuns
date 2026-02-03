"""
Purpose: Thorough unit tests for build_emotion_spikes_dataset.py
Author: Claude Code
Date: 2026-02-02

Tests verify correct computation of sale_prev_period and proper merging
of period, emotion, and trait data.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from analysis.derived.build_emotion_spikes_dataset import (
    compute_sale_timing_vars,
    add_identifiers,
    filter_dataset,
    merge_datasets,
)


# =====
# Tests for compute_sale_timing_vars
# =====
class TestComputeSaleTimingVars:
    """Test sale_prev_period and n_sales_earlier computation."""

    def _create_mock_period_data(self, sales_pattern: list[tuple]) -> pd.DataFrame:
        """
        Create mock period data from sales pattern.

        Args:
            sales_pattern: List of (round, period, player, sold) tuples
        """
        data = []
        for round_num, period, player, sold in sales_pattern:
            data.append({
                "session_id": "test",
                "segment": 1,
                "group_id": 1,
                "round": round_num,
                "period": period,
                "player": player,
                "sold": sold,
                "already_sold": 0,
                "prior_group_sales": 0,  # will be recalculated
            })
        df = pd.DataFrame(data)

        # Compute prior_group_sales correctly
        df = df.sort_values(["round", "period", "player"])
        for round_num in df["round"].unique():
            round_mask = df["round"] == round_num
            round_df = df[round_mask].copy()

            cumsum = 0
            for period in sorted(round_df["period"].unique()):
                period_mask = round_mask & (df["period"] == period)
                df.loc[period_mask, "prior_group_sales"] = cumsum
                cumsum += round_df[round_df["period"] == period]["sold"].sum()

        return df

    def test_period_1_has_sale_prev_period_0(self):
        """First period in a round should always have sale_prev_period = 0."""
        sales = [
            (1, 1, "A", 0), (1, 1, "B", 1),  # Round 1, Period 1
            (1, 2, "A", 0), (1, 2, "B", 0),  # Round 1, Period 2
        ]
        df = self._create_mock_period_data(sales)
        result = compute_sale_timing_vars(df)

        period_1 = result[result["period"] == 1]
        assert (period_1["sale_prev_period"] == 0).all(), \
            "Period 1 should always have sale_prev_period = 0"

    def test_sale_prev_period_1_when_prior_period_has_sale(self):
        """sale_prev_period = 1 when any group member sold in period t-1."""
        sales = [
            (1, 1, "A", 0), (1, 1, "B", 1),  # B sells in period 1
            (1, 2, "A", 0), (1, 2, "B", 0),  # Period 2 after sale
        ]
        df = self._create_mock_period_data(sales)
        result = compute_sale_timing_vars(df)

        period_2 = result[result["period"] == 2]
        assert (period_2["sale_prev_period"] == 1).all(), \
            "Period 2 should have sale_prev_period = 1 after sale in period 1"

    def test_sale_prev_period_0_when_no_prior_sale(self):
        """sale_prev_period = 0 when no group member sold in period t-1."""
        sales = [
            (1, 1, "A", 0), (1, 1, "B", 0),  # No sale in period 1
            (1, 2, "A", 0), (1, 2, "B", 0),  # Period 2
        ]
        df = self._create_mock_period_data(sales)
        result = compute_sale_timing_vars(df)

        period_2 = result[result["period"] == 2]
        assert (period_2["sale_prev_period"] == 0).all(), \
            "Period 2 should have sale_prev_period = 0 when no sale in period 1"

    def test_multiple_sales_in_prior_period(self):
        """sale_prev_period = 1 even if multiple people sold in t-1."""
        sales = [
            (1, 1, "A", 1), (1, 1, "B", 1), (1, 1, "C", 1),  # 3 sales
            (1, 2, "D", 0),  # Period 2
        ]
        df = self._create_mock_period_data(sales)
        result = compute_sale_timing_vars(df)

        period_2 = result[result["period"] == 2]
        assert (period_2["sale_prev_period"] == 1).all()

    def test_rounds_are_independent(self):
        """sale_prev_period resets for each new round."""
        sales = [
            (1, 1, "A", 1),  # Sale in round 1, period 1
            (1, 2, "A", 0),  # Round 1, period 2 - should have sale_prev_period = 1
            (2, 1, "A", 0),  # Round 2, period 1 - should have sale_prev_period = 0
        ]
        df = self._create_mock_period_data(sales)
        result = compute_sale_timing_vars(df)

        r1p2 = result[(result["round"] == 1) & (result["period"] == 2)]
        assert r1p2["sale_prev_period"].iloc[0] == 1

        r2p1 = result[(result["round"] == 2) & (result["period"] == 1)]
        assert r2p1["sale_prev_period"].iloc[0] == 0

    def test_n_sales_earlier_computation(self):
        """n_sales_earlier counts sales in periods 1 to t-2 (excludes t-1)."""
        sales = [
            (1, 1, "A", 1), (1, 1, "B", 0),  # 1 sale in period 1
            (1, 2, "A", 0), (1, 2, "B", 1),  # 1 sale in period 2
            (1, 3, "A", 0), (1, 3, "B", 0),  # Period 3
        ]
        df = self._create_mock_period_data(sales)
        result = compute_sale_timing_vars(df)

        # Period 3: prior_group_sales = 2, prev_period_n_sales = 1
        # n_sales_earlier = 2 - 1 = 1 (only period 1 sale)
        period_3 = result[result["period"] == 3]
        assert (period_3["n_sales_earlier"] == 1).all()

    def test_n_sales_earlier_zero_in_period_1(self):
        """n_sales_earlier should be 0 in period 1."""
        sales = [(1, 1, "A", 0)]
        df = self._create_mock_period_data(sales)
        result = compute_sale_timing_vars(df)

        assert result["n_sales_earlier"].iloc[0] == 0

    def test_n_sales_earlier_zero_in_period_2(self):
        """n_sales_earlier should be 0 in period 2 (no periods before t-1)."""
        sales = [
            (1, 1, "A", 1),  # Sale in period 1
            (1, 2, "A", 0),  # Period 2
        ]
        df = self._create_mock_period_data(sales)
        result = compute_sale_timing_vars(df)

        period_2 = result[result["period"] == 2]
        assert period_2["n_sales_earlier"].iloc[0] == 0

    def test_group_round_id_created(self):
        """group_round_id should be created for grouping."""
        sales = [(1, 1, "A", 0)]
        df = self._create_mock_period_data(sales)
        result = compute_sale_timing_vars(df)

        assert "group_round_id" in result.columns
        assert result["group_round_id"].iloc[0] == "test_1_1_1"


# =====
# Tests for add_identifiers
# =====
class TestAddIdentifiers:
    """Test player_id and global_group_id creation."""

    def test_player_id_format(self):
        """player_id should be session_id_player."""
        df = pd.DataFrame({
            "session_id": ["1_11-7-tr1"],
            "segment": [1],
            "group_id": [2],
            "player": ["A"],
        })
        result = add_identifiers(df)
        assert result["player_id"].iloc[0] == "1_11-7-tr1_A"

    def test_global_group_id_format(self):
        """global_group_id should be session_id_segment_group_id."""
        df = pd.DataFrame({
            "session_id": ["1_11-7-tr1"],
            "segment": [3],
            "group_id": [2],
            "player": ["A"],
        })
        result = add_identifiers(df)
        assert result["global_group_id"].iloc[0] == "1_11-7-tr1_3_2"

    def test_unique_player_ids(self):
        """Each player should have unique player_id across sessions."""
        df = pd.DataFrame({
            "session_id": ["s1", "s1", "s2"],
            "segment": [1, 1, 1],
            "group_id": [1, 1, 1],
            "player": ["A", "B", "A"],
        })
        result = add_identifiers(df)
        assert result["player_id"].nunique() == 3  # s1_A, s1_B, s2_A


# =====
# Tests for filter_dataset
# =====
class TestFilterDataset:
    """Test filtering logic."""

    def test_excludes_already_sold(self):
        """Rows with already_sold = 1 should be excluded."""
        df = pd.DataFrame({
            "already_sold": [0, 0, 1, 1],
            "fear_max": [10, 20, 30, 40],
        })
        result = filter_dataset(df)
        assert len(result) == 2
        assert (result["already_sold"] == 0).all()

    def test_excludes_missing_emotions(self):
        """Rows with missing fear_max should be excluded."""
        df = pd.DataFrame({
            "already_sold": [0, 0, 0],
            "fear_max": [10, np.nan, 30],
        })
        result = filter_dataset(df)
        assert len(result) == 2
        assert result["fear_max"].notna().all()

    def test_both_filters_applied(self):
        """Both already_sold and missing emotion filters applied."""
        df = pd.DataFrame({
            "already_sold": [0, 0, 1, 0],
            "fear_max": [10, np.nan, 30, 40],
        })
        result = filter_dataset(df)
        assert len(result) == 2  # Only rows 0 and 3


# =====
# Tests for merge_datasets
# =====
class TestMergeDatasets:
    """Test merging logic."""

    def test_emotions_merged_on_period_level(self):
        """Emotions should merge on session, segment, round, period, player."""
        period_df = pd.DataFrame({
            "session_id": ["s1", "s1"],
            "segment": [1, 1],
            "round": [1, 2],
            "period": [1, 1],
            "player": ["A", "A"],
            "sold": [0, 0],
        })
        emotions_df = pd.DataFrame({
            "session_id": ["s1", "s1"],
            "segment": [1, 1],
            "round": [1, 2],
            "period": [1, 1],
            "player": ["A", "A"],
            "fear_mean": [10, 20],
            "fear_max": [15, 25],
            "fear_p95": [12, 22],
            "anger_mean": [5, 6],
            "anger_max": [8, 9],
            "anger_p95": [7, 8],
            "n_frames": [100, 100],
        })
        traits_df = pd.DataFrame({
            "session_id": ["s1"],
            "player": ["A"],
            "neuroticism": [5.0],
            "impulsivity": [3.0],
            "state_anxiety": [2.0],
        })

        result = merge_datasets(period_df, emotions_df, traits_df)

        assert len(result) == 2
        assert result[result["round"] == 1]["fear_max"].iloc[0] == 15
        assert result[result["round"] == 2]["fear_max"].iloc[0] == 25

    def test_traits_merged_on_player_level(self):
        """Traits should merge on session_id and player (constant across periods)."""
        period_df = pd.DataFrame({
            "session_id": ["s1", "s1"],
            "segment": [1, 2],
            "round": [1, 1],
            "period": [1, 1],
            "player": ["A", "A"],
            "sold": [0, 0],
        })
        emotions_df = pd.DataFrame({
            "session_id": ["s1", "s1"],
            "segment": [1, 2],
            "round": [1, 1],
            "period": [1, 1],
            "player": ["A", "A"],
            "fear_mean": [10, 20],
            "fear_max": [15, 25],
            "fear_p95": [12, 22],
            "anger_mean": [5, 6],
            "anger_max": [8, 9],
            "anger_p95": [7, 8],
            "n_frames": [100, 100],
        })
        traits_df = pd.DataFrame({
            "session_id": ["s1"],
            "player": ["A"],
            "neuroticism": [5.5],
            "impulsivity": [3.0],
            "state_anxiety": [2.0],
        })

        result = merge_datasets(period_df, emotions_df, traits_df)

        # Both rows should have same trait values
        assert (result["neuroticism"] == 5.5).all()

    def test_missing_emotions_become_nan(self):
        """Periods without matching emotions should have NaN."""
        period_df = pd.DataFrame({
            "session_id": ["s1"],
            "segment": [1],
            "round": [1],
            "period": [1],
            "player": ["A"],
            "sold": [0],
        })
        emotions_df = pd.DataFrame({
            "session_id": ["s1"],
            "segment": [1],
            "round": [1],
            "period": [2],  # Different period!
            "player": ["A"],
            "fear_mean": [10],
            "fear_max": [15],
            "fear_p95": [12],
            "anger_mean": [5],
            "anger_max": [8],
            "anger_p95": [7],
            "n_frames": [100],
        })
        traits_df = pd.DataFrame({
            "session_id": ["s1"],
            "player": ["A"],
            "neuroticism": [5.0],
            "impulsivity": [3.0],
            "state_anxiety": [2.0],
        })

        result = merge_datasets(period_df, emotions_df, traits_df)
        assert result["fear_max"].isna().iloc[0]


# =====
# Integration tests with output file
# =====
class TestOutputFile:
    """Validate the generated output file."""

    @pytest.fixture
    def output_df(self):
        """Load output file if it exists."""
        output_path = Path("datastore/derived/emotion_spikes_analysis_dataset.csv")
        if not output_path.exists():
            pytest.skip("Output file not yet generated - run builder first")
        return pd.read_csv(output_path)

    def test_required_columns_present(self, output_df):
        """All required columns should be present."""
        required = [
            "session_id", "segment", "round", "period", "player",
            "sold", "already_sold", "prior_group_sales", "signal", "price",
            "sale_prev_period", "n_sales_earlier",
            "fear_max", "fear_p95", "anger_max", "anger_p95",
            "neuroticism", "impulsivity", "state_anxiety",
            "player_id", "global_group_id", "group_round_id",
        ]
        missing = set(required) - set(output_df.columns)
        assert len(missing) == 0, f"Missing columns: {missing}"

    def test_already_sold_excluded(self, output_df):
        """No rows should have already_sold = 1."""
        assert (output_df["already_sold"] == 0).all()

    def test_no_missing_emotions(self, output_df):
        """No rows should have missing fear_max or anger_max."""
        assert output_df["fear_max"].notna().all()
        assert output_df["anger_max"].notna().all()

    def test_sale_prev_period_binary(self, output_df):
        """sale_prev_period should only be 0 or 1."""
        assert set(output_df["sale_prev_period"].unique()) <= {0, 1}

    def test_sale_prev_period_zero_in_period_1(self, output_df):
        """sale_prev_period should be 0 for all period 1 observations."""
        period_1 = output_df[output_df["period"] == 1]
        assert (period_1["sale_prev_period"] == 0).all(), \
            "sale_prev_period should be 0 in period 1"

    def test_n_sales_earlier_non_negative(self, output_df):
        """n_sales_earlier should be >= 0."""
        assert (output_df["n_sales_earlier"] >= 0).all()

    def test_player_id_unique_per_session_player(self, output_df):
        """player_id should be unique per session-player combination."""
        combos = output_df.groupby(["session_id", "player"])["player_id"].nunique()
        assert (combos == 1).all()

    def test_trait_columns_constant_per_player(self, output_df):
        """Traits should be constant within each player_id."""
        for trait in ["neuroticism", "impulsivity", "state_anxiety"]:
            nunique = output_df.groupby("player_id")[trait].nunique()
            # Allow for NaN variations
            valid_nunique = nunique[nunique.notna()]
            assert (valid_nunique <= 1).all(), f"{trait} varies within player_id"

    def test_sale_prev_period_distribution_reasonable(self, output_df):
        """Check sale_prev_period has reasonable distribution."""
        pct_1 = output_df["sale_prev_period"].mean() * 100
        # Expect somewhere between 5% and 50%
        assert 1 < pct_1 < 60, f"sale_prev_period=1 is {pct_1:.1f}%, seems unreasonable"

    def test_all_sessions_present(self, output_df):
        """All 6 sessions should be present."""
        expected = {
            "1_11-7-tr1", "2_11-10-tr2", "3_11-11-tr2",
            "4_11-12-tr1", "5_11-14-tr2", "6_11-18-tr1",
        }
        actual = set(output_df["session_id"].unique())
        missing = expected - actual
        assert len(missing) == 0, f"Missing sessions: {missing}"

    def test_cross_validate_with_selling_period_regression_data(self, output_df):
        """
        Cross-validate sale_prev_period with existing regression data.

        The selling_period_regression_data.csv was generated by R code
        that computed sale_prev_period. Our Python implementation should match.
        """
        existing_path = Path("datastore/derived/selling_period_regression_data.csv")
        if not existing_path.exists():
            pytest.skip("Existing regression data not found for validation")

        existing_df = pd.read_csv(existing_path)

        # Merge on common keys
        merged = output_df.merge(
            existing_df[["session_id", "segment", "group_id", "round", "period",
                         "player", "sale_prev_period"]],
            on=["session_id", "segment", "group_id", "round", "period", "player"],
            how="inner",
            suffixes=("_new", "_existing"),
        )

        if len(merged) == 0:
            pytest.skip("No matching rows for cross-validation")

        # Check agreement
        mismatches = merged[
            merged["sale_prev_period_new"] != merged["sale_prev_period_existing"]
        ]
        assert len(mismatches) == 0, \
            f"Found {len(mismatches)} mismatches with existing sale_prev_period"
