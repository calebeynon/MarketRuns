"""
Purpose: Validate data fed into cox_survival_regression.R against raw data
Author: Claude Code
Date: 2026-02-23

Tests verify that the Cox survival regression receives correct data:
- Base sample: already_sold == 0, emotion-complete cases
- First-seller subsample: prior_group_sales == 0 & sold == 1 identifiers
- Covariate sets: non-valence (7 emotions) vs valence-only split
- Observation counts pinned to regression output
- Cross-validation against market_data.py parser for sold, signal, and
  first-seller identity
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import market_data as md

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
DERIVED_DIR = DATASTORE / "derived"
EMOTIONS_DATASET = DERIVED_DIR / "emotions_traits_selling_dataset.csv"

SESSION_FILES = {
    "1_11-7-tr1": DATASTORE / "1_11-7-tr1" / "all_apps_wide_2025-11-07.csv",
    "2_11-10-tr2": DATASTORE / "2_11-10-tr2" / "all_apps_wide_2025-11-10.csv",
    "3_11-11-tr2": DATASTORE / "3_11-11-tr2" / "all_apps_wide_2025-11-11.csv",
    "4_11-12-tr1": DATASTORE / "4_11-12-tr1" / "all_apps_wide_2025-11-12.csv",
    "5_11-14-tr2": DATASTORE / "5_11-14-tr2" / "all_apps_wide_2025-11-14.csv",
    "6_11-18-tr1": DATASTORE / "6_11-18-tr1" / "all_apps_wide_2025-11-18.csv",
}
SEGMENT_MAP = {
    1: "chat_noavg", 2: "chat_noavg2",
    3: "chat_noavg3", 4: "chat_noavg4",
}

ALL_EMOTIONS = [
    "fear_mean", "anger_mean", "contempt_mean", "disgust_mean",
    "joy_mean", "sadness_mean", "surprise_mean",
    "engagement_mean", "valence_mean",
]
NONVALENCE_EMOTIONS = [
    "fear_mean", "anger_mean", "contempt_mean", "disgust_mean",
    "sadness_mean", "surprise_mean", "engagement_mean",
]
COX_CONTROLS = [
    "signal", "round", "segment", "treatment", "age", "gender_female",
]
CASCADE_DUMMIES = ["dummy_1_cum", "dummy_2_cum", "dummy_3_cum"]
INTERACTION_VARS = [
    "int_1_1", "int_2_1", "int_2_2",
    "int_3_1", "int_3_2", "int_3_3",
]


# =====
# Main function
# =====
def main():
    """Run all tests with verbose output."""
    pytest.main([__file__, "-v"])


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def raw_data():
    """Load raw emotions_traits_selling_dataset.csv."""
    if not EMOTIONS_DATASET.exists():
        pytest.skip(f"Dataset not found: {EMOTIONS_DATASET}")
    return pd.read_csv(EMOTIONS_DATASET)


@pytest.fixture(scope="module")
def base_data(raw_data):
    """Replicate R prepare_base_data(): filter already_sold == 0."""
    df = raw_data[raw_data["already_sold"] == 0].copy()
    df["player_id"] = df["session_id"] + "_" + df["player"].astype(str)
    df["group_round_id"] = (
        df["session_id"] + "_" + df["segment"].astype(str) + "_"
        + df["group_id"].astype(str) + "_" + df["round"].astype(str)
    )
    df["player_group_round_id"] = (
        df["player_id"] + "_" + df["segment"].astype(str) + "_"
        + df["group_id"].astype(str) + "_" + df["round"].astype(str)
    )
    df["dummy_1_cum"] = (df["prior_group_sales"] == 1).astype(int)
    df["dummy_2_cum"] = (df["prior_group_sales"] == 2).astype(int)
    df["dummy_3_cum"] = (df["prior_group_sales"] == 3).astype(int)
    df["gender_female"] = (df["gender"] == "Female").astype(int)
    df = add_prev_period_dummies(df)
    df = add_interaction_terms(df)
    return df


@pytest.fixture(scope="module")
def emotion_filtered(base_data):
    """Emotion-complete cases — the sample passed to both panels."""
    return base_data.dropna(subset=ALL_EMOTIONS).copy()


@pytest.fixture(scope="module")
def first_seller_data(emotion_filtered):
    """First-seller subsample: player-group-rounds where someone sold
    with prior_group_sales == 0, keeping all at-risk periods."""
    first_ids = emotion_filtered.loc[
        (emotion_filtered["prior_group_sales"] == 0)
        & (emotion_filtered["sold"] == 1),
        "player_group_round_id"
    ].unique()
    return emotion_filtered[
        emotion_filtered["player_group_round_id"].isin(first_ids)
    ].copy()


@pytest.fixture(scope="module")
def parsed_experiments():
    """Load raw session data via market_data parser."""
    experiments = {}
    for session_id, csv_path in SESSION_FILES.items():
        if csv_path.exists():
            experiments[session_id] = md.parse_experiment(str(csv_path))
    if not experiments:
        pytest.skip("No raw session files found")
    return experiments


# =====
# Helpers: replicate R dummy/interaction logic in Python
# =====
def add_prev_period_dummies(df):
    """Replicate create_prev_period_dummies() from R helpers."""
    period_sales = (
        df.groupby(["group_round_id", "period"])["sold"]
        .sum().reset_index()
    )
    period_sales = period_sales.rename(columns={"sold": "n_sales"})
    period_sales = period_sales.sort_values(
        ["group_round_id", "period"]
    )
    period_sales["prev_n_sales"] = (
        period_sales.groupby("group_round_id")["n_sales"].shift(1)
    )
    period_sales["prev_n_sales"] = (
        period_sales["prev_n_sales"].fillna(0).astype(int)
    )
    df = df.merge(
        period_sales[["group_round_id", "period", "prev_n_sales"]],
        on=["group_round_id", "period"], how="left",
    )
    df["prev_n_sales"] = df["prev_n_sales"].fillna(0).astype(int)
    df["dummy_1_prev"] = (df["prev_n_sales"] == 1).astype(int)
    df["dummy_2_prev"] = (df["prev_n_sales"] == 2).astype(int)
    df["dummy_3_prev"] = (df["prev_n_sales"] == 3).astype(int)
    df = df.drop(columns=["prev_n_sales"])
    return df


def add_interaction_terms(df):
    """Replicate create_interaction_terms() from R helpers."""
    df["int_1_1"] = df["dummy_1_cum"] * df["dummy_1_prev"]
    df["int_2_1"] = df["dummy_2_cum"] * df["dummy_1_prev"]
    df["int_2_2"] = df["dummy_2_cum"] * df["dummy_2_prev"]
    df["int_3_1"] = df["dummy_3_cum"] * df["dummy_1_prev"]
    df["int_3_2"] = df["dummy_3_cum"] * df["dummy_2_prev"]
    df["int_3_3"] = df["dummy_3_cum"] * df["dummy_3_prev"]
    return df


def get_player_group(session, segment_name, player_label):
    """Look up which group a player belongs to in a segment."""
    segment = session.get_segment(segment_name)
    if not segment:
        return None
    for group in segment.groups.values():
        if player_label in group.player_labels:
            return group.group_id
    return None


# =====
# Observation counts (pinned to regression output)
# =====
class TestObservationCounts:
    """Pin sample sizes to match R regression output."""

    def test_emotion_filtered_size(self, emotion_filtered):
        """Emotion-complete sample: 13,713 rows (from R output)."""
        assert len(emotion_filtered) == 13713

    def test_all_sellers_events(self, emotion_filtered):
        """All Sellers events (sold == 1): 674 pre-coxme.
        coxme reports 659 after dropping rows with NA in non-emotion
        covariates (age, gender, etc.)."""
        assert emotion_filtered["sold"].sum() == 674

    def test_all_sellers_participants(self, emotion_filtered):
        """All Sellers unique participants."""
        n = emotion_filtered["player_id"].nunique()
        assert n == 96

    def test_first_seller_subsample_size(self, first_seller_data):
        """First-seller subsample: 1,218 rows (from R output)."""
        assert len(first_seller_data) == 1218

    def test_first_seller_events(self, first_seller_data):
        """First-seller events (sold == 1)."""
        n_events = first_seller_data["sold"].sum()
        assert n_events == 481

    def test_first_seller_participants(self, first_seller_data):
        """First-seller unique participants."""
        n = first_seller_data["player_id"].nunique()
        assert n == 84

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
        """Every row in first-seller data has prior_group_sales == 0."""
        assert (first_seller_data["prior_group_sales"] == 0).all()

    def test_cascade_dummies_all_zero(self, first_seller_data):
        """Cascade dummies must all be 0 (prior_group_sales == 0)."""
        for col in CASCADE_DUMMIES:
            assert (first_seller_data[col] == 0).all(), (
                f"{col} has non-zero values in first-seller data"
            )

    def test_interaction_terms_all_zero(self, first_seller_data):
        """Interaction terms must all be 0 (cascade dummies are 0)."""
        for col in INTERACTION_VARS:
            assert (first_seller_data[col] == 0).all(), (
                f"{col} has non-zero values in first-seller data"
            )

    def test_includes_presale_periods(self, first_seller_data):
        """Survival data: includes at-risk periods before the sale."""
        n_presale = (first_seller_data["sold"] == 0).sum()
        n_sold = (first_seller_data["sold"] == 1).sum()
        assert n_presale > 0
        assert n_sold > 0
        assert n_presale > n_sold

    def test_each_player_round_has_one_sale(self, first_seller_data):
        """Each player-group-round has exactly one sold == 1."""
        sold_counts = (
            first_seller_data[first_seller_data["sold"] == 1]
            .groupby("player_group_round_id").size()
        )
        assert (sold_counts == 1).all()

    def test_periods_monotonic_within_player_round(
        self, first_seller_data
    ):
        """Periods are consecutive within each player-group-round."""
        for pgr, grp in first_seller_data.groupby(
            "player_group_round_id"
        ):
            periods = sorted(grp["period"].values)
            assert periods == list(range(periods[0], periods[-1] + 1)), (
                f"Non-consecutive periods for {pgr}: {periods}"
            )

    def test_sale_always_in_last_period(self, first_seller_data):
        """The sold == 1 row is always the last period for that player-round
        (can't be at risk after selling)."""
        for pgr, grp in first_seller_data.groupby(
            "player_group_round_id"
        ):
            sale_row = grp[grp["sold"] == 1]
            if len(sale_row) == 0:
                continue
            max_period = grp["period"].max()
            assert sale_row["period"].iloc[0] == max_period, (
                f"{pgr}: sale at period {sale_row['period'].iloc[0]} "
                f"but max period is {max_period}"
            )

    def test_first_seller_is_subset_of_all_sellers(
        self, emotion_filtered, first_seller_data
    ):
        """Every first-seller row exists in the all-sellers sample."""
        all_idx = set(emotion_filtered.index)
        first_idx = set(first_seller_data.index)
        assert first_idx.issubset(all_idx)

    def test_matches_panel_c_subsample(
        self, emotion_filtered, first_seller_data
    ):
        """First-seller IDs match the Panel C logic from
        unified_selling_regression_panel_c.R."""
        panel_c_ids = set(
            emotion_filtered.loc[
                (emotion_filtered["prior_group_sales"] == 0)
                & (emotion_filtered["sold"] == 1),
                "player_group_round_id"
            ].unique()
        )
        cox_first_ids = set(
            first_seller_data["player_group_round_id"].unique()
        )
        assert panel_c_ids == cox_first_ids, (
            f"Mismatch: {len(panel_c_ids)} Panel C vs "
            f"{len(cox_first_ids)} Cox first-seller IDs. "
            f"Only in Panel C: {panel_c_ids - cox_first_ids}, "
            f"Only in Cox: {cox_first_ids - panel_c_ids}"
        )


# =====
# Covariate sets per model
# =====
class TestCovariates:
    """Verify the right covariates are available for each model."""

    def test_nonvalence_excludes_joy(self):
        """Joy is intentionally excluded from non-valence emotions."""
        assert "joy_mean" not in NONVALENCE_EMOTIONS

    def test_nonvalence_excludes_valence(self):
        """Valence is not in non-valence set."""
        assert "valence_mean" not in NONVALENCE_EMOTIONS

    def test_nonvalence_has_seven_emotions(self):
        """Exactly 7 non-valence emotions."""
        assert len(NONVALENCE_EMOTIONS) == 7

    def test_nonvalence_emotions_nonnull_all_sellers(
        self, emotion_filtered
    ):
        """All 7 non-valence emotions have no NaNs in All Sellers data."""
        for col in NONVALENCE_EMOTIONS:
            assert emotion_filtered[col].notna().all(), (
                f"{col} has NaNs in emotion-filtered data"
            )

    def test_valence_nonnull_all_sellers(self, emotion_filtered):
        """Valence has no NaNs in All Sellers data."""
        assert emotion_filtered["valence_mean"].notna().all()

    def test_nonvalence_emotions_nonnull_first_sellers(
        self, first_seller_data
    ):
        """All 7 non-valence emotions have no NaNs in first-seller data."""
        for col in NONVALENCE_EMOTIONS:
            assert first_seller_data[col].notna().all(), (
                f"{col} has NaNs in first-seller data"
            )

    def test_valence_nonnull_first_sellers(self, first_seller_data):
        """Valence has no NaNs in first-seller data."""
        assert first_seller_data["valence_mean"].notna().all()

    def test_cascade_dummies_present_all_sellers(self, emotion_filtered):
        """Cascade dummies exist and have variation in All Sellers."""
        for col in CASCADE_DUMMIES:
            assert col in emotion_filtered.columns
            assert emotion_filtered[col].nunique() == 2  # 0 and 1

    def test_interaction_terms_present_all_sellers(
        self, emotion_filtered
    ):
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
        """Signal is a Bayesian posterior probability in [0, 1]."""
        valid = emotion_filtered["signal"].dropna()
        assert valid.min() >= 0, "Signal below 0"
        assert valid.max() <= 1, "Signal above 1"

    def test_gender_female_binary(self, emotion_filtered):
        """gender_female is 0 or 1."""
        assert set(
            emotion_filtered["gender_female"].unique()
        ).issubset({0, 1})


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

    def test_interactions_zero_when_no_prior_sales(
        self, emotion_filtered
    ):
        """When prior_group_sales == 0, all interactions must be 0."""
        zero_prior = emotion_filtered[
            emotion_filtered["prior_group_sales"] == 0
        ]
        for col in INTERACTION_VARS:
            assert (zero_prior[col] == 0).all(), (
                f"{col} non-zero with 0 prior sales"
            )


# =====
# Cross-validation against raw parser: sold field
# =====
class TestRawParserSold:
    """Cross-validate sold against market_data.py parser."""

    def test_sold_matches_parser_all_sellers(
        self, emotion_filtered, parsed_experiments
    ):
        """Validate sold field for 200 random All Sellers rows."""
        sample = emotion_filtered.sample(200, random_state=42)
        mismatches, checked = validate_sold_against_parser(
            sample, parsed_experiments
        )
        assert checked > 100, f"Only checked {checked} rows"
        assert mismatches == 0, (
            f"{mismatches}/{checked} sold mismatches vs parser"
        )

    def test_sold_matches_parser_first_sellers(
        self, first_seller_data, parsed_experiments
    ):
        """Validate sold field for 200 random first-seller rows."""
        sample = first_seller_data.sample(
            min(200, len(first_seller_data)), random_state=43
        )
        mismatches, checked = validate_sold_against_parser(
            sample, parsed_experiments
        )
        assert checked > 50, f"Only checked {checked} rows"
        assert mismatches == 0, (
            f"{mismatches}/{checked} sold mismatches in first sellers"
        )


def validate_sold_against_parser(sample, parsed_experiments):
    """Check sold field against parser for a sample of rows."""
    mismatches = checked = 0
    for _, row in sample.iterrows():
        exp = parsed_experiments.get(row["session_id"])
        if exp is None or not exp.sessions:
            continue
        session = exp.sessions[0]
        seg = session.get_segment(SEGMENT_MAP[int(row["segment"])])
        if not seg:
            continue
        rnd = seg.get_round(int(row["round"]))
        if not rnd:
            continue
        per = rnd.get_period(int(row["period"]))
        if not per:
            continue
        player = per.get_player(row["player"])
        if not player:
            continue
        checked += 1
        parser_sold = 1 if player.sold_this_period else 0
        if row["sold"] != parser_sold:
            mismatches += 1
    return mismatches, checked


# =====
# Cross-validation against raw parser: signal
# =====
class TestRawParserSignal:
    """Cross-validate signal against market_data.py parser."""

    def test_signal_matches_parser(
        self, emotion_filtered, parsed_experiments
    ):
        """Validate signal field for 200 random rows."""
        sample = emotion_filtered.sample(200, random_state=99)
        mismatches = checked = 0
        for _, row in sample.iterrows():
            exp = parsed_experiments.get(row["session_id"])
            if exp is None or not exp.sessions:
                continue
            session = exp.sessions[0]
            seg = session.get_segment(SEGMENT_MAP[int(row["segment"])])
            if not seg:
                continue
            rnd = seg.get_round(int(row["round"]))
            if not rnd:
                continue
            per = rnd.get_period(int(row["period"]))
            if not per:
                continue
            player = per.get_player(row["player"])
            if not player or player.signal is None:
                continue
            checked += 1
            if abs(row["signal"] - player.signal) > 0.001:
                mismatches += 1
        assert checked > 100, f"Only checked {checked}"
        assert mismatches == 0, (
            f"{mismatches}/{checked} signal mismatches"
        )


# =====
# Cross-validation against raw parser: first-seller identity
# =====
class TestRawParserFirstSeller:
    """Verify first-seller identity against raw session data."""

    def test_first_seller_identity_matches_parser(
        self, first_seller_data, parsed_experiments
    ):
        """For each group-round in the first-seller sample, verify the
        player who sold is actually the first seller in raw data."""
        sold_rows = first_seller_data[first_seller_data["sold"] == 1]
        mismatches = checked = 0

        for _, row in sold_rows.iterrows():
            exp = parsed_experiments.get(row["session_id"])
            if exp is None or not exp.sessions:
                continue
            session = exp.sessions[0]
            seg_name = SEGMENT_MAP[int(row["segment"])]
            seg = session.get_segment(seg_name)
            if not seg:
                continue
            rnd = seg.get_round(int(row["round"]))
            if not rnd:
                continue

            group_id = int(row["group_id"])
            # Find the first period with a sale in this group
            first_sale_period = find_first_sale_period(
                rnd, seg, session, group_id, seg_name
            )
            if first_sale_period is None:
                continue

            checked += 1
            # The dataset row's period should equal the first sale period
            if int(row["period"]) != first_sale_period:
                mismatches += 1

        assert checked > 200, f"Only checked {checked} group-rounds"
        assert mismatches == 0, (
            f"{mismatches}/{checked} first-seller period mismatches. "
            "Some 'first sellers' sold after the actual first sale."
        )

    def test_no_prior_sales_in_raw_data(
        self, first_seller_data, parsed_experiments
    ):
        """For 100 first-seller sale rows, verify no one in their group
        sold in any earlier period of the same round."""
        sold_rows = first_seller_data[
            first_seller_data["sold"] == 1
        ].sample(min(100, len(first_seller_data)), random_state=77)
        mismatches = checked = 0

        for _, row in sold_rows.iterrows():
            exp = parsed_experiments.get(row["session_id"])
            if exp is None or not exp.sessions:
                continue
            session = exp.sessions[0]
            seg_name = SEGMENT_MAP[int(row["segment"])]
            seg = session.get_segment(seg_name)
            if not seg:
                continue
            rnd = seg.get_round(int(row["round"]))
            if not rnd:
                continue

            group_id = int(row["group_id"])
            sale_period = int(row["period"])

            # Count group sales before this period
            sales_before = count_group_sales_before(
                rnd, seg, session, group_id, seg_name, sale_period
            )
            checked += 1
            if sales_before != 0:
                mismatches += 1

        assert checked > 50, f"Only checked {checked}"
        assert mismatches == 0, (
            f"{mismatches}/{checked} rows had prior group sales != 0 "
            "according to raw parser"
        )


def find_first_sale_period(rnd, seg, session, group_id, seg_name):
    """Find the first period with any sale in a group-round."""
    for period_num in sorted(rnd.periods.keys()):
        period = rnd.get_period(period_num)
        if not period:
            continue
        for label, pdata in period.players.items():
            if pdata.sold_this_period:
                pg = get_player_group(session, seg_name, label)
                if pg == group_id:
                    return period_num
    return None


def count_group_sales_before(
    rnd, seg, session, group_id, seg_name, before_period
):
    """Count how many sales happened in this group before a given period."""
    total = 0
    for period_num in sorted(rnd.periods.keys()):
        if period_num >= before_period:
            break
        period = rnd.get_period(period_num)
        if not period:
            continue
        for label, pdata in period.players.items():
            if pdata.sold_this_period:
                pg = get_player_group(session, seg_name, label)
                if pg == group_id:
                    total += 1
    return total


# %%
if __name__ == "__main__":
    main()
