"""
Purpose: Validate data fed into unified_selling_regression.R against raw data
Author: Claude Code
Date: 2026-02-06

Tests verify that the data preparation for the unified regression table
(issue #31) matches raw session data parsed by market_data.py, and that
all filtering, subsetting, and variable construction follows the logic
established in issues #4, #18, and #19.
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
INDIVIDUAL_PERIOD_DATASET = DERIVED_DIR / "individual_period_dataset.csv"

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

EMOTION_COLS = [
    "fear_mean", "anger_mean", "contempt_mean", "disgust_mean",
    "joy_mean", "sadness_mean", "surprise_mean",
    "engagement_mean", "valence_mean",
]
TRAIT_COLS = [
    "extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness", "impulsivity", "state_anxiety",
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
    """Replicate R main script's prepare_base_data() logic in Python."""
    df = raw_data.copy()
    df = df[df["already_sold"] == 0].copy()
    df["player_id"] = df["session_id"] + "_" + df["player"].astype(str)
    df["global_group_id"] = (
        df["session_id"] + "_"
        + df["segment"].astype(str) + "_"
        + df["group_id"].astype(str)
    )
    df["group_round_id"] = (
        df["session_id"] + "_"
        + df["segment"].astype(str) + "_"
        + df["group_id"].astype(str) + "_"
        + df["round"].astype(str)
    )
    df["player_group_round_id"] = (
        df["player_id"] + "_"
        + df["segment"].astype(str) + "_"
        + df["group_id"].astype(str) + "_"
        + df["round"].astype(str)
    )
    df["time_id"] = (
        df["segment"].astype(str) + "_"
        + df["round"].astype(str) + "_"
        + df["period"].astype(str)
    )
    df["dummy_1_cum"] = (df["prior_group_sales"] == 1).astype(int)
    df["dummy_2_cum"] = (df["prior_group_sales"] == 2).astype(int)
    df["dummy_3_cum"] = (df["prior_group_sales"] == 3).astype(int)
    df["gender_female"] = (df["gender"] == "Female").astype(int)
    return df


@pytest.fixture(scope="module")
def emotion_filtered(base_data):
    """Filter to complete emotion cases (matching R's complete.cases)."""
    return base_data.dropna(subset=EMOTION_COLS).copy()


@pytest.fixture(scope="module")
def panel_a_data(emotion_filtered):
    """All participants sample for Panel A."""
    return emotion_filtered


@pytest.fixture(scope="module")
def panel_b_data(emotion_filtered, base_data):
    """Second sellers sample for Panel B with dummy_prev_period."""
    df = emotion_filtered
    second_ids = df.loc[
        (df["prior_group_sales"] == 1) & (df["sold"] == 1),
        "player_group_round_id"
    ].unique()
    df_second = df[df["player_group_round_id"].isin(second_ids)].copy()
    first_sales = base_data.loc[
        (base_data["prior_group_sales"] == 0) & (base_data["sold"] == 1),
        ["group_round_id", "period"]
    ].groupby("group_round_id")["period"].min().reset_index()
    first_sales.columns = ["group_round_id", "first_sale_period"]
    df_second = df_second.merge(
        first_sales, on="group_round_id", how="left"
    )
    df_second["dummy_prev_period"] = (
        df_second["first_sale_period"] == (df_second["period"] - 1)
    ).astype(int)
    return df_second


@pytest.fixture(scope="module")
def panel_c_data(emotion_filtered):
    """First sellers sample for Panel C."""
    df = emotion_filtered
    first_ids = df.loc[
        (df["prior_group_sales"] == 0) & (df["sold"] == 1),
        "player_group_round_id"
    ].unique()
    return df[df["player_group_round_id"].isin(first_ids)].copy()


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
# Base data preparation validation
# =====
class TestBaseDataPreparation:
    """Verify data prep matches raw CSV and issue #4 logic."""

    def test_already_sold_filtered(self, raw_data, base_data):
        """Only already_sold == 0 rows survive."""
        expected = (raw_data["already_sold"] == 0).sum()
        assert len(base_data) == expected

    def test_no_already_sold_rows(self, base_data):
        """No already_sold == 1 rows in filtered data."""
        assert (base_data["already_sold"] == 0).all()

    def test_cumulative_dummies_match_prior_sales(self, base_data):
        """dummy_N_cum == 1 iff prior_group_sales == N."""
        for n in [1, 2, 3]:
            col = f"dummy_{n}_cum"
            mask = base_data["prior_group_sales"] == n
            assert (base_data.loc[mask, col] == 1).all()
            assert (base_data.loc[~mask, col] == 0).all()

    def test_cumulative_dummies_mutually_exclusive(self, base_data):
        """At most one cumulative dummy is 1 per row."""
        total = (
            base_data["dummy_1_cum"]
            + base_data["dummy_2_cum"]
            + base_data["dummy_3_cum"]
        )
        assert (total <= 1).all()

    def test_gender_female_binary(self, base_data):
        """gender_female is 0 or 1."""
        assert set(base_data["gender_female"].unique()).issubset({0, 1})

    def test_gender_female_matches_gender_column(self, base_data):
        """gender_female == 1 iff gender == 'Female'."""
        female_mask = base_data["gender"] == "Female"
        assert (base_data.loc[female_mask, "gender_female"] == 1).all()
        assert (base_data.loc[~female_mask, "gender_female"] == 0).all()

    def test_player_id_format(self, base_data):
        """player_id = session_id + '_' + player."""
        sample = base_data.head(100)
        for _, row in sample.iterrows():
            expected = f"{row['session_id']}_{row['player']}"
            assert row["player_id"] == expected

    def test_time_id_unique_per_player(self, base_data):
        """Each player_id × time_id should appear at most once."""
        dupes = base_data.duplicated(
            subset=["player_id", "time_id"], keep=False
        )
        assert not dupes.any(), (
            f"Found {dupes.sum()} duplicate player_id × time_id pairs"
        )


# =====
# Panel A: All participants validation
# =====
class TestPanelA:
    """Verify Panel A sample matches issue #4 + #18 logic."""

    def test_emotion_filter_removes_na_rows(self, base_data, panel_a_data):
        """Emotion filter only removes rows with NA emotions."""
        has_na = base_data[EMOTION_COLS].isna().any(axis=1)
        expected = len(base_data) - has_na.sum()
        assert len(panel_a_data) == expected

    def test_no_emotion_nans(self, panel_a_data):
        """No NaN values in emotion columns after filtering."""
        for col in EMOTION_COLS:
            assert panel_a_data[col].notna().all(), f"NaN found in {col}"

    def test_traits_present(self, panel_a_data):
        """Trait columns exist and have values."""
        for col in TRAIT_COLS:
            assert col in panel_a_data.columns
            assert panel_a_data[col].notna().any()

    def test_sample_size_reasonable(self, panel_a_data):
        """Panel A should have ~13,700 obs."""
        assert 13000 < len(panel_a_data) < 15000

    def test_sold_matches_parser(self, panel_a_data, parsed_experiments):
        """Validate sold field against market_data parser for 200 rows."""
        sample = panel_a_data.sample(
            min(200, len(panel_a_data)), random_state=42
        )
        mismatches = 0
        checked = 0
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
        assert checked > 0, "No rows validated against parser"
        assert mismatches == 0, (
            f"{mismatches}/{checked} sold mismatches vs parser"
        )

    def test_signal_matches_parser(self, panel_a_data, parsed_experiments):
        """Validate signal field against market_data parser for 200 rows."""
        sample = panel_a_data.sample(
            min(200, len(panel_a_data)), random_state=99
        )
        mismatches = 0
        checked = 0
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
            if player.signal is None:
                continue
            checked += 1
            if abs(row["signal"] - player.signal) > 0.001:
                mismatches += 1
        assert checked > 0, "No signal rows validated"
        assert mismatches == 0, (
            f"{mismatches}/{checked} signal mismatches vs parser"
        )


# =====
# Panel B: Second sellers validation
# =====
class TestPanelB:
    """Verify Panel B follows issue #4 second-seller logic."""

    def test_all_second_sellers_have_one_prior_sale(self, panel_b_data):
        """Every sale in Panel B happened when prior_group_sales == 1."""
        sold_rows = panel_b_data[panel_b_data["sold"] == 1]
        assert (sold_rows["prior_group_sales"] == 1).all()

    def test_no_overlap_with_first_sellers(self, panel_b_data, panel_c_data):
        """No player-group-round appears in both Panel B and Panel C."""
        b_ids = set(panel_b_data["player_group_round_id"].unique())
        c_ids = set(panel_c_data["player_group_round_id"].unique())
        overlap = b_ids & c_ids
        assert len(overlap) == 0, f"Overlap: {overlap}"

    def test_dummy_prev_period_binary(self, panel_b_data):
        """dummy_prev_period is 0 or 1."""
        vals = set(panel_b_data["dummy_prev_period"].unique())
        assert vals.issubset({0, 1})

    def test_dummy_prev_period_logic(self, panel_b_data, base_data):
        """dummy_prev_period == 1 iff first sale was in period - 1."""
        first_sales = base_data.loc[
            (base_data["prior_group_sales"] == 0) & (base_data["sold"] == 1),
            ["group_round_id", "period"]
        ].groupby("group_round_id")["period"].min()

        for _, row in panel_b_data.sample(
            min(200, len(panel_b_data)), random_state=42
        ).iterrows():
            gr_id = row["group_round_id"]
            if gr_id not in first_sales.index:
                continue
            first_period = first_sales[gr_id]
            expected = 1 if first_period == (row["period"] - 1) else 0
            assert row["dummy_prev_period"] == expected, (
                f"gr={gr_id} period={row['period']} "
                f"first_sale={first_period} expected={expected}"
            )

    def test_second_seller_sample_size(self, panel_b_data):
        """Panel B should have ~600 obs."""
        assert 400 < len(panel_b_data) < 1000

    def test_sale_always_after_first_sale(self, panel_b_data, base_data):
        """Second sale period >= first sale period in same group-round."""
        first_sales = base_data.loc[
            (base_data["prior_group_sales"] == 0) & (base_data["sold"] == 1),
            ["group_round_id", "period"]
        ].groupby("group_round_id")["period"].min()

        sold_rows = panel_b_data[panel_b_data["sold"] == 1]
        for _, row in sold_rows.iterrows():
            gr_id = row["group_round_id"]
            if gr_id not in first_sales.index:
                continue
            assert row["period"] >= first_sales[gr_id], (
                f"Second sale at period {row['period']} before "
                f"first sale at period {first_sales[gr_id]}"
            )


# =====
# Panel C: First sellers validation
# =====
class TestPanelC:
    """Verify Panel C follows issue #4 first-seller logic."""

    def test_all_first_sellers_have_zero_prior_sales(self, panel_c_data):
        """Every sale in Panel C happened when prior_group_sales == 0."""
        sold_rows = panel_c_data[panel_c_data["sold"] == 1]
        assert (sold_rows["prior_group_sales"] == 0).all()

    def test_no_cascade_variables_needed(self, panel_c_data):
        """All prior_group_sales should be 0 for first sellers."""
        assert (panel_c_data["prior_group_sales"] == 0).all()

    def test_first_seller_sample_size(self, panel_c_data):
        """Panel C should have ~1,200 obs."""
        assert 800 < len(panel_c_data) < 2000

    def test_includes_presale_observations(self, panel_c_data):
        """Sample includes periods before the sale (sold == 0)."""
        n_presale = (panel_c_data["sold"] == 0).sum()
        n_sold = (panel_c_data["sold"] == 1).sum()
        assert n_presale > 0, "No presale observations"
        assert n_sold > 0, "No sale observations"
        assert n_presale > n_sold, "Should have more presale than sale obs"

    def test_each_player_round_has_one_sale(self, panel_c_data):
        """Each player-group-round has exactly one sold == 1."""
        sold_counts = (
            panel_c_data[panel_c_data["sold"] == 1]
            .groupby("player_group_round_id")
            .size()
        )
        assert (sold_counts == 1).all()


# =====
# Cross-panel consistency
# =====
class TestCrossPanelConsistency:
    """Verify consistency across all three panels."""

    def test_panels_use_same_base_data(
        self, panel_a_data, panel_b_data, panel_c_data
    ):
        """Panel B + C player-group-rounds should all exist in Panel A."""
        a_pgr = set(panel_a_data["player_group_round_id"].unique())
        b_pgr = set(panel_b_data["player_group_round_id"].unique())
        c_pgr = set(panel_c_data["player_group_round_id"].unique())
        assert b_pgr.issubset(a_pgr)
        assert c_pgr.issubset(a_pgr)

    def test_panel_b_c_no_overlap(self, panel_b_data, panel_c_data):
        """Panel B and Panel C player-group-rounds don't overlap."""
        b_pgr = set(panel_b_data["player_group_round_id"].unique())
        c_pgr = set(panel_c_data["player_group_round_id"].unique())
        assert len(b_pgr & c_pgr) == 0

    def test_all_sessions_represented(self, panel_a_data):
        """All 6 sessions appear in Panel A."""
        sessions = set(panel_a_data["session_id"].unique())
        assert len(sessions) == 6

    def test_all_segments_represented(self, panel_a_data):
        """All 4 segments appear in Panel A."""
        segments = set(panel_a_data["segment"].unique())
        assert segments == {1, 2, 3, 4}

    def test_treatment_consistent_within_session(self, panel_a_data):
        """Each session has exactly one treatment value."""
        treatment_per_session = (
            panel_a_data.groupby("session_id")["treatment"]
            .nunique()
        )
        assert (treatment_per_session == 1).all()


# =====
# Emotion and trait data validation (issue #18, #19)
# =====
class TestEmotionTraitData:
    """Verify emotion and trait columns match issue #18 and #19."""

    def test_emotions_are_time_varying(self, panel_a_data):
        """Emotions should vary within a player across periods."""
        varying_count = 0
        for player_id in panel_a_data["player_id"].unique()[:20]:
            player_df = panel_a_data[panel_a_data["player_id"] == player_id]
            if len(player_df) < 2:
                continue
            if player_df["fear_mean"].nunique() > 1:
                varying_count += 1
        assert varying_count > 0, "Emotions don't vary within players"

    def test_traits_are_time_invariant(self, panel_a_data):
        """Traits should be constant within a player."""
        for player_id in panel_a_data["player_id"].unique()[:20]:
            player_df = panel_a_data[panel_a_data["player_id"] == player_id]
            if len(player_df) < 2:
                continue
            for trait in TRAIT_COLS:
                vals = player_df[trait].dropna().unique()
                assert len(vals) <= 1, (
                    f"Trait {trait} varies for {player_id}: {vals}"
                )

    def test_demographics_time_invariant(self, panel_a_data):
        """Age and gender_female constant within player."""
        for player_id in panel_a_data["player_id"].unique()[:20]:
            player_df = panel_a_data[panel_a_data["player_id"] == player_id]
            if len(player_df) < 2:
                continue
            assert player_df["age"].nunique() <= 1
            assert player_df["gender_female"].nunique() <= 1

    def test_emotion_values_reasonable(self, panel_a_data):
        """Emotion means should be in [0, 100] range."""
        for col in EMOTION_COLS:
            valid = panel_a_data[col].dropna()
            if col == "valence_mean":
                assert valid.min() >= -100
            else:
                assert valid.min() >= 0, f"{col} has negative values"
            assert valid.max() <= 100, f"{col} exceeds 100"

    def test_trait_values_reasonable(self, panel_a_data):
        """Trait scores should be positive."""
        for col in TRAIT_COLS:
            valid = panel_a_data[col].dropna()
            assert valid.min() >= 0, f"{col} has negative values"


# =====
# Prior group sales validation against parser (issue #4)
# =====
class TestPriorGroupSalesParser:
    """Cross-validate prior_group_sales against market_data parser."""

    def test_prior_sales_matches_parser(
        self, panel_a_data, parsed_experiments
    ):
        """Validate prior_group_sales by counting sales from parser."""
        sample = panel_a_data.sample(
            min(100, len(panel_a_data)), random_state=7
        )
        mismatches = 0
        checked = 0

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

            # Count sales in periods before this one
            sales_before = 0
            group_id = int(row["group_id"])
            for p_num in range(1, int(row["period"])):
                per = rnd.get_period(p_num)
                if not per:
                    continue
                for label, pdata in per.players.items():
                    if pdata.sold_this_period:
                        player_group = get_player_group(
                            session, SEGMENT_MAP[int(row["segment"])],
                            label
                        )
                        if player_group == group_id:
                            sales_before += 1
            checked += 1
            if row["prior_group_sales"] != sales_before:
                mismatches += 1

        assert checked > 0, "No rows validated"
        assert mismatches == 0, (
            f"{mismatches}/{checked} prior_group_sales mismatches"
        )


def get_player_group(session, segment_name, player_label):
    """Look up which group a player belongs to in a segment."""
    segment = session.get_segment(segment_name)
    if not segment:
        return None
    for group in segment.groups.values():
        if player_label in group.player_labels:
            return group.group_id
    return None


# %%
if __name__ == "__main__":
    main()
