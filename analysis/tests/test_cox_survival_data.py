"""
Purpose: Validate Cox survival regression data structure and covariates
Author: Claude Code
Date: 2026-02-23

Tests verify observation counts, subsample construction, and covariates.
"""

import pytest
from cox_test_helpers import (
    DISCRETE_EMOTIONS, COX_CONTROLS,
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
        """Emotion-complete sample before demographic NA drops."""
        assert len(emotion_filtered) == 13713

    def test_all_sellers_events(self, emotion_filtered):
        """All Sellers events (sold == 1) in emotion-complete sample."""
        assert emotion_filtered["sold"].sum() == 674

    def test_all_sellers_participants(self, emotion_filtered):
        """All Sellers unique participants."""
        assert emotion_filtered["player_id"].nunique() == 96

    def test_first_seller_subsample_size(self, first_seller_data):
        """First-seller subsample before demographic NA drops."""
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

    def test_discrete_includes_joy(self):
        """Joy is included in discrete emotions."""
        assert "joy_mean" in DISCRETE_EMOTIONS

    def test_discrete_excludes_valence(self):
        """Valence is not in discrete emotions set."""
        assert "valence_mean" not in DISCRETE_EMOTIONS

    def test_discrete_has_eight_emotions(self):
        """Exactly 8 discrete emotions."""
        assert len(DISCRETE_EMOTIONS) == 8

    def test_discrete_nonnull_all_sellers(self, emotion_filtered):
        """All 8 discrete emotions have no NaNs."""
        for col in DISCRETE_EMOTIONS:
            assert emotion_filtered[col].notna().all(), f"{col} has NaNs"

    def test_valence_nonnull_all_sellers(self, emotion_filtered):
        """Valence has no NaNs in All Sellers data."""
        assert emotion_filtered["valence_mean"].notna().all()

    def test_discrete_nonnull_first_sellers(self, first_seller_data):
        """All 8 discrete emotions have no NaNs in first-seller."""
        for col in DISCRETE_EMOTIONS:
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


# %%
if __name__ == "__main__":
    main()
