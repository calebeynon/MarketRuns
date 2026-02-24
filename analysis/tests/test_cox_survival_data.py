"""
Purpose: Validate Cox survival regression data structure and covariates
Author: Claude Code
Date: 2026-02-23

Tests verify observation counts, subsample construction, covariates,
interaction terms, and counting process survival format.
"""

import pytest
from cox_test_helpers import (
    NONVALENCE_EMOTIONS, COX_CONTROLS,
    CASCADE_DUMMIES, INTERACTION_VARS,
)


# =====
# Main function
# =====
def main():
    """Run all tests with verbose output."""
    pytest.main([__file__, "-v"])


# =====
# Observation counts (pinned to regression output)
# =====
class TestObservationCounts:
    """Pin sample sizes to match R regression output."""

    def test_emotion_filtered_size(self, emotion_filtered):
        """Emotion-complete sample: 13,713 rows (from R output)."""
        assert len(emotion_filtered) == 13713

    def test_all_sellers_events(self, emotion_filtered):
        """All Sellers events (sold == 1): 674 pre-coxme."""
        assert emotion_filtered["sold"].sum() == 674

    def test_all_sellers_participants(self, emotion_filtered):
        """All Sellers unique participants."""
        assert emotion_filtered["player_id"].nunique() == 96

    def test_first_seller_subsample_size(self, first_seller_data):
        """First-seller subsample: 1,218 rows (from R output)."""
        assert len(first_seller_data) == 1218

    def test_first_seller_events(self, first_seller_data):
        """First-seller events (sold == 1)."""
        assert first_seller_data["sold"].sum() == 481

    def test_first_seller_participants(self, first_seller_data):
        """First-seller unique participants."""
        assert first_seller_data["player_id"].nunique() == 84

    def test_first_seller_group_rounds(self, first_seller_data):
        """First-seller unique player-group-rounds: 481."""
        n = first_seller_data["player_group_round_id"].nunique()
        assert n == 481

    def test_all_sessions_in_all_sellers(self, emotion_filtered):
        """All 6 sessions appear in All Sellers sample."""
        assert emotion_filtered["session_id"].nunique() == 6

    def test_all_segments_in_all_sellers(self, emotion_filtered):
        """All 4 segments appear."""
        assert set(emotion_filtered["segment"].unique()) == {1, 2, 3, 4}


# =====
# First-seller subsample construction
# =====
class TestFirstSellerSubsample:
    """Verify first-seller identification and subsample properties."""

    def test_all_prior_sales_zero(self, first_seller_data):
        """Every row has prior_group_sales == 0."""
        assert (first_seller_data["prior_group_sales"] == 0).all()

    def test_cascade_dummies_all_zero(self, first_seller_data):
        """Cascade dummies must all be 0."""
        for col in CASCADE_DUMMIES:
            assert (first_seller_data[col] == 0).all(), (
                f"{col} has non-zero values"
            )

    def test_interaction_terms_all_zero(self, first_seller_data):
        """Interaction terms must all be 0."""
        for col in INTERACTION_VARS:
            assert (first_seller_data[col] == 0).all(), (
                f"{col} has non-zero values"
            )

    def test_includes_presale_periods(self, first_seller_data):
        """Survival data: includes at-risk periods before the sale."""
        n_presale = (first_seller_data["sold"] == 0).sum()
        n_sold = (first_seller_data["sold"] == 1).sum()
        assert n_presale > 0 and n_sold > 0
        assert n_presale > n_sold

    def test_each_player_round_has_one_sale(self, first_seller_data):
        """Each player-group-round has exactly one sold == 1."""
        sold_counts = (
            first_seller_data[first_seller_data["sold"] == 1]
            .groupby("player_group_round_id").size()
        )
        assert (sold_counts == 1).all()

    def test_periods_consecutive(self, first_seller_data):
        """Periods are consecutive within each player-group-round."""
        for pgr, grp in first_seller_data.groupby(
            "player_group_round_id"
        ):
            periods = sorted(grp["period"].values)
            expected = list(range(periods[0], periods[-1] + 1))
            assert periods == expected, f"Gap in {pgr}: {periods}"

    def test_sale_always_in_last_period(self, first_seller_data):
        """sold == 1 is always the last period for that player-round."""
        for pgr, grp in first_seller_data.groupby(
            "player_group_round_id"
        ):
            sale_row = grp[grp["sold"] == 1]
            if len(sale_row) == 0:
                continue
            assert sale_row["period"].iloc[0] == grp["period"].max()

    def test_is_subset_of_all_sellers(
        self, emotion_filtered, first_seller_data
    ):
        """Every first-seller row exists in the all-sellers sample."""
        assert set(first_seller_data.index).issubset(
            set(emotion_filtered.index)
        )

    def test_matches_panel_c_subsample(
        self, emotion_filtered, first_seller_data
    ):
        """First-seller IDs match Panel C logic."""
        panel_c_ids = _get_first_seller_ids(emotion_filtered)
        cox_ids = set(first_seller_data["player_group_round_id"].unique())
        assert panel_c_ids == cox_ids


def _get_first_seller_ids(df):
    """Return set of player-group-round IDs matching Panel C logic."""
    return set(
        df.loc[
            (df["prior_group_sales"] == 0) & (df["sold"] == 1),
            "player_group_round_id",
        ].unique()
    )


# =====
# Covariate sets per model
# =====
class TestCovariates:
    """Verify the right covariates are available for each model."""

    def test_nonvalence_excludes_joy(self):
        """Joy is intentionally excluded."""
        assert "joy_mean" not in NONVALENCE_EMOTIONS

    def test_nonvalence_excludes_valence(self):
        """Valence is not in non-valence set."""
        assert "valence_mean" not in NONVALENCE_EMOTIONS

    def test_nonvalence_has_seven_emotions(self):
        """Exactly 7 non-valence emotions."""
        assert len(NONVALENCE_EMOTIONS) == 7

    def test_nonvalence_nonnull_all_sellers(self, emotion_filtered):
        """All 7 non-valence emotions have no NaNs."""
        for col in NONVALENCE_EMOTIONS:
            assert emotion_filtered[col].notna().all(), f"{col} has NaNs"

    def test_valence_nonnull_all_sellers(self, emotion_filtered):
        """Valence has no NaNs in All Sellers data."""
        assert emotion_filtered["valence_mean"].notna().all()

    def test_nonvalence_nonnull_first_sellers(self, first_seller_data):
        """All 7 non-valence emotions have no NaNs in first-seller."""
        for col in NONVALENCE_EMOTIONS:
            assert first_seller_data[col].notna().all(), f"{col} has NaNs"

    def test_valence_nonnull_first_sellers(self, first_seller_data):
        """Valence has no NaNs in first-seller data."""
        assert first_seller_data["valence_mean"].notna().all()

    def test_cascade_dummies_have_variation(self, emotion_filtered):
        """Cascade dummies exist and have variation in All Sellers."""
        for col in CASCADE_DUMMIES:
            assert emotion_filtered[col].nunique() == 2

    def test_interaction_terms_present(self, emotion_filtered):
        """Interaction terms exist in All Sellers data."""
        for col in INTERACTION_VARS:
            assert col in emotion_filtered.columns

    def test_controls_present_all_sellers(self, emotion_filtered):
        """All control variables present in All Sellers data."""
        for col in COX_CONTROLS:
            assert col in emotion_filtered.columns, f"Missing: {col}"

    def test_controls_present_first_sellers(self, first_seller_data):
        """All control variables present in first-seller data."""
        for col in COX_CONTROLS:
            assert col in first_seller_data.columns, f"Missing: {col}"

    def test_signal_continuous_0_1(self, emotion_filtered):
        """Signal is a Bayesian posterior in [0, 1]."""
        valid = emotion_filtered["signal"].dropna()
        assert valid.min() >= 0 and valid.max() <= 1

    def test_gender_female_binary(self, emotion_filtered):
        """gender_female is 0 or 1."""
        assert set(emotion_filtered["gender_female"].unique()) <= {0, 1}


# =====
# Interaction term correctness
# =====
class TestInteractionTerms:
    """Verify interaction terms = cumulative dummy * prev-period dummy."""

    def test_int_1_1_equals_product(self, emotion_filtered):
        """int_1_1 == dummy_1_cum * dummy_1_prev."""
        expected = emotion_filtered["dummy_1_cum"] * emotion_filtered["dummy_1_prev"]
        assert (emotion_filtered["int_1_1"] == expected).all()

    def test_int_2_1_equals_product(self, emotion_filtered):
        """int_2_1 == dummy_2_cum * dummy_1_prev."""
        expected = emotion_filtered["dummy_2_cum"] * emotion_filtered["dummy_1_prev"]
        assert (emotion_filtered["int_2_1"] == expected).all()

    def test_int_2_2_equals_product(self, emotion_filtered):
        """int_2_2 == dummy_2_cum * dummy_2_prev."""
        expected = emotion_filtered["dummy_2_cum"] * emotion_filtered["dummy_2_prev"]
        assert (emotion_filtered["int_2_2"] == expected).all()

    def test_int_3_3_equals_product(self, emotion_filtered):
        """int_3_3 == dummy_3_cum * dummy_3_prev."""
        expected = emotion_filtered["dummy_3_cum"] * emotion_filtered["dummy_3_prev"]
        assert (emotion_filtered["int_3_3"] == expected).all()

    def test_interactions_zero_when_no_prior_sales(self, emotion_filtered):
        """When prior_group_sales == 0, all interactions must be 0."""
        zero = emotion_filtered[emotion_filtered["prior_group_sales"] == 0]
        for col in INTERACTION_VARS:
            assert (zero[col] == 0).all(), f"{col} non-zero"


# =====
# Counting process survival structure
# =====
class TestCountingProcess:
    """Verify Surv(period_start, period, sold) structure is correct."""

    def test_period_start_equals_period_minus_one(self, base_data):
        """period_start must equal period - 1 for all rows."""
        assert (base_data["period_start"] == base_data["period"] - 1).all()

    def test_period_start_nonnegative(self, base_data):
        """No negative start times."""
        assert (base_data["period_start"] >= 0).all()

    def test_interval_width_is_one(self, base_data):
        """Each row covers exactly one unit: (period-1, period]."""
        assert ((base_data["period"] - base_data["period_start"]) == 1).all()

    def test_intervals_contiguous_all_sellers(self, emotion_filtered):
        """Intervals are contiguous within each player-group-round."""
        _verify_contiguous(emotion_filtered)

    def test_intervals_contiguous_first_sellers(self, first_seller_data):
        """Same contiguity check for first-seller subsample."""
        _verify_contiguous(first_seller_data)

    def test_no_overlapping_intervals(self, emotion_filtered):
        """No duplicate periods within a player-group-round."""
        for pgr, grp in emotion_filtered.groupby("player_group_round_id"):
            periods = grp["period"].values
            assert len(periods) == len(set(periods)), f"Dup in {pgr}"

    def test_event_only_in_last_interval(self, emotion_filtered):
        """sold == 1 only in the last interval."""
        for pgr, grp in emotion_filtered.groupby("player_group_round_id"):
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
