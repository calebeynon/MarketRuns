"""
Purpose: Validate interaction terms and counting process survival structure
Author: Claude Code
Date: 2026-02-23

Tests verify interaction term correctness and Surv(period_start, period, sold)
counting process format for Cox survival regression.
"""

import pytest
from cox_test_helpers import INTERACTION_VARS


# =====
# Main function
# =====
def main():
    """Run all tests with verbose output."""
    pytest.main([__file__, "-v"])


# =====
# Interaction term correctness
# =====
class TestInteractionTerms:
    """Verify interaction terms = cumulative dummy * prev-period dummy."""

    def test_int_1_1_equals_product(self, emotion_filtered):
        """int_1_1 == dummy_1_cum * dummy_1_prev."""
        expected = (
            emotion_filtered["dummy_1_cum"]
            * emotion_filtered["dummy_1_prev"]
        )
        assert (emotion_filtered["int_1_1"] == expected).all()

    def test_int_2_1_equals_product(self, emotion_filtered):
        """int_2_1 == dummy_2_cum * dummy_1_prev."""
        expected = (
            emotion_filtered["dummy_2_cum"]
            * emotion_filtered["dummy_1_prev"]
        )
        assert (emotion_filtered["int_2_1"] == expected).all()

    def test_int_2_2_equals_product(self, emotion_filtered):
        """int_2_2 == dummy_2_cum * dummy_2_prev."""
        expected = (
            emotion_filtered["dummy_2_cum"]
            * emotion_filtered["dummy_2_prev"]
        )
        assert (emotion_filtered["int_2_2"] == expected).all()

    def test_int_3_3_equals_product(self, emotion_filtered):
        """int_3_3 == dummy_3_cum * dummy_3_prev."""
        expected = (
            emotion_filtered["dummy_3_cum"]
            * emotion_filtered["dummy_3_prev"]
        )
        assert (emotion_filtered["int_3_3"] == expected).all()

    def test_interactions_zero_when_no_prior_sales(self, emotion_filtered):
        """When prior_group_sales == 0, all interactions must be 0."""
        zero = emotion_filtered[
            emotion_filtered["prior_group_sales"] == 0
        ]
        for col in INTERACTION_VARS:
            assert (zero[col] == 0).all(), f"{col} non-zero"


# =====
# Counting process survival structure
# =====
class TestCountingProcess:
    """Verify Surv(period_start, period, sold) structure is correct."""

    def test_period_start_equals_period_minus_one(self, base_data):
        """period_start must equal period - 1 for all rows."""
        assert (
            base_data["period_start"] == base_data["period"] - 1
        ).all()

    def test_period_start_nonnegative(self, base_data):
        """No negative start times."""
        assert (base_data["period_start"] >= 0).all()

    def test_interval_width_is_one(self, base_data):
        """Each row covers exactly one unit: (period-1, period]."""
        width = base_data["period"] - base_data["period_start"]
        assert (width == 1).all()

    def test_intervals_contiguous_all_sellers(self, emotion_filtered):
        """Intervals are contiguous within each player-group-round."""
        _verify_contiguous(emotion_filtered)

    def test_intervals_contiguous_first_sellers(self, first_seller_data):
        """Same contiguity check for first-seller subsample."""
        _verify_contiguous(first_seller_data)

    def test_no_overlapping_intervals(self, emotion_filtered):
        """No duplicate periods within a player-group-round."""
        for pgr, grp in emotion_filtered.groupby(
            "player_group_round_id"
        ):
            periods = grp["period"].values
            assert len(periods) == len(set(periods)), f"Dup in {pgr}"

    def test_event_only_in_last_interval(self, emotion_filtered):
        """sold == 1 only in the last interval."""
        for pgr, grp in emotion_filtered.groupby(
            "player_group_round_id"
        ):
            sales = grp[grp["sold"] == 1]
            if len(sales) == 0:
                continue
            assert sales["period"].iloc[0] == grp["period"].max()

    def test_period_start_exists(self, base_data):
        """period_start column exists."""
        assert "period_start" in base_data.columns

    def test_first_period_start_nonnegative(self, emotion_filtered):
        """Earliest period_start within any player-group-round >= 0."""
        mins = emotion_filtered.groupby(
            "player_group_round_id"
        )["period_start"].min()
        assert (mins >= 0).all()


# =====
# Contiguity helper
# =====
def _verify_contiguous(df):
    """Check intervals are contiguous within each player-group-round."""
    for pgr, grp in df.groupby("player_group_round_id"):
        grp_s = grp.sort_values("period")
        periods = grp_s["period"].values
        starts = grp_s["period_start"].values
        for i in range(1, len(periods)):
            assert starts[i] == periods[i - 1], (
                f"{pgr}: gap at {i}"
            )


# %%
if __name__ == "__main__":
    main()
