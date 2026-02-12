"""
Purpose: Unit tests for build_ordinal_selling_position.py
Author: Claude Code
Date: 2026-02-09
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
from analysis.derived.build_ordinal_selling_position import (
    compute_selling_ranks,
    create_derived_variables,
    determine_emotion_periods,
    merge_emotions,
    P95_COLS,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
OUTPUT_CSV = DATASTORE / "derived" / "ordinal_selling_position.csv"


# =====
# Helper to create mock DataFrames
# =====
def create_group_round_df(sell_periods, did_sells):
    """Create a mock group-round DataFrame for 4 players.

    Args:
        sell_periods: list of 4 sell_period values (NaN for non-sellers)
        did_sells: list of 4 did_sell values (0 or 1)

    Returns:
        DataFrame with columns matching compute_selling_ranks input
    """
    return pd.DataFrame({
        "session_id": ["s1"] * 4,
        "segment": [1] * 4,
        "group_id": [1] * 4,
        "round": [1] * 4,
        "player": ["A", "B", "C", "D"],
        "sell_period": sell_periods,
        "did_sell": did_sells,
    })


# =====
# Rank computation tests
# =====
def test_all_sell_different_periods():
    """4 players sell in periods 1,2,3,4 -> ranks 1,2,3,4."""
    df = create_group_round_df(
        sell_periods=[1, 2, 3, 4],
        did_sells=[1, 1, 1, 1],
    )
    result = compute_selling_ranks(df)

    expected = {"A": 1, "B": 2, "C": 3, "D": 4}
    for player, rank in expected.items():
        actual = result.loc[result["player"] == player, "sell_rank"].iloc[0]
        assert actual == rank, f"Player {player}: expected {rank}, got {actual}"


def test_two_players_tie_period_1():
    """2 sell in period 1, 1 in period 3 -> ranks 1,1,3; non-seller=4."""
    df = create_group_round_df(
        sell_periods=[1, 1, 3, float("nan")],
        did_sells=[1, 1, 1, 0],
    )
    result = compute_selling_ranks(df)

    expected = {"A": 1, "B": 1, "C": 3, "D": 4}
    for player, rank in expected.items():
        actual = result.loc[result["player"] == player, "sell_rank"].iloc[0]
        assert actual == rank, f"Player {player}: expected {rank}, got {actual}"


def test_three_players_tie():
    """3 sell in same period -> all rank 1, non-seller rank 4."""
    df = create_group_round_df(
        sell_periods=[2, 2, 2, float("nan")],
        did_sells=[1, 1, 1, 0],
    )
    result = compute_selling_ranks(df)

    expected = {"A": 1, "B": 1, "C": 1, "D": 4}
    for player, rank in expected.items():
        actual = result.loc[result["player"] == player, "sell_rank"].iloc[0]
        assert actual == rank, f"Player {player}: expected {rank}, got {actual}"


def test_all_four_tie():
    """4 sell in same period -> all rank 1."""
    df = create_group_round_df(
        sell_periods=[3, 3, 3, 3],
        did_sells=[1, 1, 1, 1],
    )
    result = compute_selling_ranks(df)

    for player in ["A", "B", "C", "D"]:
        actual = result.loc[result["player"] == player, "sell_rank"].iloc[0]
        assert actual == 1, f"Player {player}: expected 1, got {actual}"


def test_no_sellers():
    """Nobody sells -> all rank 4."""
    df = create_group_round_df(
        sell_periods=[float("nan")] * 4,
        did_sells=[0, 0, 0, 0],
    )
    result = compute_selling_ranks(df)

    for player in ["A", "B", "C", "D"]:
        actual = result.loc[result["player"] == player, "sell_rank"].iloc[0]
        assert actual == 4, f"Player {player}: expected 4, got {actual}"


def test_one_seller():
    """1 sells, 3 hold -> seller rank 1, holders rank 4."""
    df = create_group_round_df(
        sell_periods=[5, float("nan"), float("nan"), float("nan")],
        did_sells=[1, 0, 0, 0],
    )
    result = compute_selling_ranks(df)

    expected = {"A": 1, "B": 4, "C": 4, "D": 4}
    for player, rank in expected.items():
        actual = result.loc[result["player"] == player, "sell_rank"].iloc[0]
        assert actual == rank, f"Player {player}: expected {rank}, got {actual}"


def test_two_sellers_two_holders():
    """2 sell in periods 1 and 3 -> ranks 1, 2, 4, 4."""
    df = create_group_round_df(
        sell_periods=[1, float("nan"), 3, float("nan")],
        did_sells=[1, 0, 1, 0],
    )
    result = compute_selling_ranks(df)

    expected = {"A": 1, "B": 4, "C": 2, "D": 4}
    for player, rank in expected.items():
        actual = result.loc[result["player"] == player, "sell_rank"].iloc[0]
        assert actual == rank, f"Player {player}: expected {rank}, got {actual}"


def test_non_sellers_always_rank_4():
    """Non-sellers always get rank 4 regardless of scenario."""
    scenarios = [
        # (sell_periods, did_sells, non_seller_players)
        ([1, float("nan"), float("nan"), float("nan")], [1, 0, 0, 0], ["B", "C", "D"]),
        ([1, 2, float("nan"), float("nan")], [1, 1, 0, 0], ["C", "D"]),
        ([1, 2, 3, float("nan")], [1, 1, 1, 0], ["D"]),
        ([float("nan")] * 4, [0, 0, 0, 0], ["A", "B", "C", "D"]),
    ]

    for sell_periods, did_sells, non_sellers in scenarios:
        df = create_group_round_df(sell_periods, did_sells)
        result = compute_selling_ranks(df)

        for player in non_sellers:
            actual = result.loc[result["player"] == player, "sell_rank"].iloc[0]
            assert actual == 4, (
                f"Non-seller {player} expected rank 4, got {actual}"
            )


# =====
# Emotion period tests
# =====
def _make_emotion_test_data(did_sell, sell_period, max_period=5):
    """Create a single-row DataFrame and max_periods for emotion tests."""
    df = pd.DataFrame({
        "session_id": ["s1"], "segment": [1], "group_id": [1],
        "round": [1], "player": ["A"],
        "did_sell": [did_sell], "sell_period": [sell_period],
    })
    max_periods = pd.DataFrame({
        "session_id": ["s1"], "segment": [1], "group_id": [1],
        "round": [1], "max_period": [max_period],
    })
    return df, max_periods


def test_seller_emotion_period_is_sell_period():
    """Seller who sold in period 3 gets emotion_period=3."""
    df, max_periods = _make_emotion_test_data(did_sell=1, sell_period=3)
    result = determine_emotion_periods(df, max_periods)
    assert result.iloc[0]["emotion_period"] == 3


def test_non_seller_emotion_period_is_max_period():
    """Non-seller gets emotion_period equal to max period of group-round."""
    df, max_periods = _make_emotion_test_data(
        did_sell=0, sell_period=float("nan"), max_period=5,
    )
    result = determine_emotion_periods(df, max_periods)
    assert result.iloc[0]["emotion_period"] == 5


# =====
# Dataset-level tests (use output CSV if it exists)
# =====
@pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent.parent
         / "datastore" / "derived" / "ordinal_selling_position.csv").exists(),
    reason="Output CSV not yet built",
)
def test_output_columns_present():
    """Verify all expected output columns exist."""
    df = pd.read_csv(OUTPUT_CSV)

    expected_columns = [
        "session_id", "treatment", "segment", "group_id", "round",
        "player", "player_id", "sell_period", "did_sell", "sell_rank",
        "anger_p95", "contempt_p95", "disgust_p95", "fear_p95", "joy_p95",
        "sadness_p95", "surprise_p95", "engagement_p95", "valence_p95",
        "extraversion", "agreeableness", "conscientiousness", "neuroticism",
        "openness", "impulsivity", "state_anxiety", "age", "gender_female",
    ]
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"


@pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent.parent
         / "datastore" / "derived" / "ordinal_selling_position.csv").exists(),
    reason="Output CSV not yet built",
)
def test_sell_rank_range():
    """All sell_rank values are in {1, 2, 3, 4}."""
    df = pd.read_csv(OUTPUT_CSV)
    valid_ranks = {1, 2, 3, 4}
    actual_ranks = set(df["sell_rank"].unique())
    assert actual_ranks.issubset(valid_ranks), (
        f"Unexpected ranks: {actual_ranks - valid_ranks}"
    )


@pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent.parent
         / "datastore" / "derived" / "ordinal_selling_position.csv").exists(),
    reason="Output CSV not yet built",
)
def test_non_seller_rank_always_4():
    """Rows with did_sell=0 always have sell_rank=4."""
    df = pd.read_csv(OUTPUT_CSV)
    non_sellers = df[df["did_sell"] == 0]
    assert (non_sellers["sell_rank"] == 4).all(), (
        "Found non-sellers with sell_rank != 4"
    )


# =====
# Ground-truth tests against actual raw data
# =====
def _get_derived_ranks(df, session_id, segment, group_id, rnd):
    """Extract sell_rank dict {player: rank} for a specific group-round."""
    sub = df[
        (df["session_id"] == session_id)
        & (df["segment"] == segment)
        & (df["group_id"] == group_id)
        & (df["round"] == rnd)
    ]
    return dict(zip(sub["player"], sub["sell_rank"]))


@pytest.mark.skipif(not OUTPUT_CSV.exists(), reason="Output CSV not yet built")
def test_raw_no_sellers_s1_seg1_g1_r1():
    """Session 1, seg 1, group 1, round 1: nobody sold -> all rank 4."""
    df = pd.read_csv(OUTPUT_CSV)
    ranks = _get_derived_ranks(df, "1_11-7-tr1", 1, 1, 1)
    assert ranks == {"A": 4, "E": 4, "J": 4, "N": 4}


@pytest.mark.skipif(not OUTPUT_CSV.exists(), reason="Output CSV not yet built")
def test_raw_one_seller_s1_seg1_g1_r5():
    """Session 1, seg 1, group 1, round 5: N sells period 1."""
    df = pd.read_csv(OUTPUT_CSV)
    ranks = _get_derived_ranks(df, "1_11-7-tr1", 1, 1, 5)
    assert ranks == {"A": 4, "E": 4, "J": 4, "N": 1}


@pytest.mark.skipif(not OUTPUT_CSV.exists(), reason="Output CSV not yet built")
def test_raw_three_sellers_s1_seg1_g1_r3():
    """Session 1, seg 1, group 1, round 3: A=p1, N=p2, J=p8, E=hold."""
    df = pd.read_csv(OUTPUT_CSV)
    ranks = _get_derived_ranks(df, "1_11-7-tr1", 1, 1, 3)
    assert ranks == {"A": 1, "N": 2, "J": 3, "E": 4}


@pytest.mark.skipif(not OUTPUT_CSV.exists(), reason="Output CSV not yet built")
def test_raw_four_sellers_no_ties_s2_seg1_g3_r7():
    """Session 2, seg 1, group 3, round 7: L=p1, G=p4, Q=p7, C=p9."""
    df = pd.read_csv(OUTPUT_CSV)
    ranks = _get_derived_ranks(df, "2_11-10-tr2", 1, 3, 7)
    assert ranks == {"L": 1, "G": 2, "Q": 3, "C": 4}


@pytest.mark.skipif(not OUTPUT_CSV.exists(), reason="Output CSV not yet built")
def test_raw_tie_at_period_2_s1_seg1_g1_r2():
    """Session 1, seg 1, group 1, round 2: A=p1, J=p2, N=p2, E=hold."""
    df = pd.read_csv(OUTPUT_CSV)
    ranks = _get_derived_ranks(df, "1_11-7-tr1", 1, 1, 2)
    assert ranks == {"A": 1, "J": 2, "N": 2, "E": 4}


@pytest.mark.skipif(not OUTPUT_CSV.exists(), reason="Output CSV not yet built")
def test_raw_tie_at_period_1_s1_seg1_g1_r6():
    """Session 1, seg 1, group 1, round 6: A=p1, N=p1, E=hold, J=hold."""
    df = pd.read_csv(OUTPUT_CSV)
    ranks = _get_derived_ranks(df, "1_11-7-tr1", 1, 1, 6)
    assert ranks == {"A": 1, "N": 1, "E": 4, "J": 4}


# =====
# create_derived_variables tests
# =====
def test_gender_female_encodes_female_as_1():
    """Female gender is encoded as 1."""
    df = pd.DataFrame({
        "session_id": ["s1", "s1"],
        "player": ["A", "B"],
        "gender": ["Female", "Female"],
    })
    result = create_derived_variables(df)
    assert (result["gender_female"] == 1).all()


def test_gender_female_encodes_male_as_0():
    """Male gender is encoded as 0."""
    df = pd.DataFrame({
        "session_id": ["s1"],
        "player": ["E"],
        "gender": ["Male"],
    })
    result = create_derived_variables(df)
    assert result.iloc[0]["gender_female"] == 0


def test_gender_female_encodes_prefer_not_to_say_as_0():
    """'Prefer not to say' is encoded as 0 (non-Female)."""
    df = pd.DataFrame({
        "session_id": ["s1"],
        "player": ["X"],
        "gender": ["Prefer not to say"],
    })
    result = create_derived_variables(df)
    assert result.iloc[0]["gender_female"] == 0


def test_gender_female_mixed_values():
    """Verified against production data: Female=1, Male=0."""
    df = pd.DataFrame({
        "session_id": ["1_11-7-tr1"] * 3,
        "player": ["A", "E", "B"],
        "gender": ["Female", "Male", "Female"],
    })
    result = create_derived_variables(df)
    expected = {"A": 1, "E": 0, "B": 1}
    for player, val in expected.items():
        actual = result.loc[result["player"] == player, "gender_female"].iloc[0]
        assert actual == val, f"Player {player}: expected {val}, got {actual}"


def test_player_id_construction():
    """player_id is '{session_id}_{player}'."""
    df = pd.DataFrame({
        "session_id": ["1_11-7-tr1", "2_11-10-tr2"],
        "player": ["A", "G"],
        "gender": ["Female", "Male"],
    })
    result = create_derived_variables(df)
    assert result.iloc[0]["player_id"] == "1_11-7-tr1_A"
    assert result.iloc[1]["player_id"] == "2_11-10-tr2_G"


def test_player_id_matches_production_format():
    """Regression: player_id format verified against ordinal_selling_position.csv."""
    df = pd.DataFrame({
        "session_id": ["1_11-7-tr1"],
        "player": ["N"],
        "gender": ["Female"],
    })
    result = create_derived_variables(df)
    assert result.iloc[0]["player_id"] == "1_11-7-tr1_N"


def test_create_derived_variables_does_not_mutate_input():
    """Input DataFrame should not be modified."""
    df = pd.DataFrame({
        "session_id": ["s1"],
        "player": ["A"],
        "gender": ["Female"],
    })
    original_cols = list(df.columns)
    create_derived_variables(df)
    assert list(df.columns) == original_cols


# =====
# merge_emotions tests
# =====
def _make_merge_emotions_inputs(df_rows, emotions_rows):
    """Build a player DataFrame and emotions DataFrame for merge tests.

    Args:
        df_rows: list of dicts with player-level fields + emotion_period
        emotions_rows: list of dicts with emotion-level fields + p95 values
    """
    df = pd.DataFrame(df_rows)
    emotions = pd.DataFrame(emotions_rows)
    return df, emotions


@patch("analysis.derived.build_ordinal_selling_position.pd.read_csv")
def test_merge_emotions_with_nan_values(mock_read_csv):
    """Emotions with NaN p95 values are preserved after merge."""
    df = pd.DataFrame({
        "session_id": ["s1"], "segment": [1], "round": [1],
        "emotion_period": [2], "player": ["A"],
    })
    emotions = pd.DataFrame({
        "session_id": ["s1"], "segment": [1], "round": [1],
        "period": [2], "player": ["A"],
        **{col: [np.nan] for col in P95_COLS},
    })
    mock_read_csv.return_value = emotions
    result = merge_emotions(df)

    assert len(result) == 1, "Should retain the row even with NaN emotions"
    for col in P95_COLS:
        assert pd.isna(result.iloc[0][col]), f"{col} should be NaN"


@patch("analysis.derived.build_ordinal_selling_position.pd.read_csv")
def test_merge_emotions_no_matching_rows_gives_nan(mock_read_csv):
    """Player with no matching emotion rows gets NaN for all p95 columns."""
    df = pd.DataFrame({
        "session_id": ["s1"], "segment": [1], "round": [1],
        "emotion_period": [3], "player": ["A"],
    })
    # Emotions exist for a different player only
    emotions = pd.DataFrame({
        "session_id": ["s1"], "segment": [1], "round": [1],
        "period": [3], "player": ["B"],
        **{col: [0.5] for col in P95_COLS},
    })
    mock_read_csv.return_value = emotions
    result = merge_emotions(df)

    # Left join: player A is kept, but all p95 cols are NaN
    assert len(result) == 1
    assert result.iloc[0]["player"] == "A"
    for col in P95_COLS:
        assert pd.isna(result.iloc[0][col]), (
            f"{col} should be NaN when no emotion match"
        )


@patch("analysis.derived.build_ordinal_selling_position.pd.read_csv")
def test_merge_emotions_partial_coverage(mock_read_csv):
    """One player has emotions, another does not -> mixed NaN."""
    df = pd.DataFrame({
        "session_id": ["s1", "s1"], "segment": [1, 1],
        "round": [1, 1], "emotion_period": [2, 2],
        "player": ["A", "B"],
    })
    emotions = pd.DataFrame({
        "session_id": ["s1"], "segment": [1], "round": [1],
        "period": [2], "player": ["A"],
        **{col: [0.42] for col in P95_COLS},
    })
    mock_read_csv.return_value = emotions
    result = merge_emotions(df)

    assert len(result) == 2
    # Player A has emotions
    row_a = result.loc[result["player"] == "A"].iloc[0]
    assert row_a["anger_p95"] == pytest.approx(0.42)
    # Player B has NaN emotions
    row_b = result.loc[result["player"] == "B"].iloc[0]
    for col in P95_COLS:
        assert pd.isna(row_b[col]), f"Player B {col} should be NaN"


# =====
# Multi-group-round rank independence tests
# =====
def test_ranks_independent_across_groups_same_round():
    """Ranks are computed independently per group within the same round."""
    df = pd.DataFrame({
        "session_id": ["s1"] * 8,
        "segment": [1] * 8,
        "group_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "round": [1] * 8,
        "player": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "sell_period": [3, 1, float("nan"), 2, 1, 1, 3, float("nan")],
        "did_sell": [1, 1, 0, 1, 1, 1, 1, 0],
    })
    result = compute_selling_ranks(df)

    # Group 1: B(p1)->1, D(p2)->2, A(p3)->3, C(hold)->4
    g1 = result[result["group_id"] == 1]
    g1_ranks = dict(zip(g1["player"], g1["sell_rank"]))
    assert g1_ranks == {"A": 3, "B": 1, "C": 4, "D": 2}

    # Group 2: E,F(p1)->tied 1, G(p3)->3, H(hold)->4
    g2 = result[result["group_id"] == 2]
    g2_ranks = dict(zip(g2["player"], g2["sell_rank"]))
    assert g2_ranks == {"E": 1, "F": 1, "G": 3, "H": 4}


def test_ranks_independent_across_rounds_same_group():
    """Ranks are computed independently per round within the same group."""
    df = pd.DataFrame({
        "session_id": ["s1"] * 8,
        "segment": [1] * 8,
        "group_id": [1] * 8,
        "round": [1, 1, 1, 1, 2, 2, 2, 2],
        "player": ["A", "B", "C", "D", "A", "B", "C", "D"],
        "sell_period": [1, 2, 3, float("nan"), 5, float("nan"), 1, 1],
        "did_sell": [1, 1, 1, 0, 1, 0, 1, 1],
    })
    result = compute_selling_ranks(df)

    # Round 1: A(p1)->1, B(p2)->2, C(p3)->3, D(hold)->4
    r1 = result[result["round"] == 1]
    r1_ranks = dict(zip(r1["player"], r1["sell_rank"]))
    assert r1_ranks == {"A": 1, "B": 2, "C": 3, "D": 4}

    # Round 2: C,D(p1)->tied 1, A(p5)->3, B(hold)->4
    r2 = result[result["round"] == 2]
    r2_ranks = dict(zip(r2["player"], r2["sell_rank"]))
    assert r2_ranks == {"A": 3, "B": 4, "C": 1, "D": 1}


def test_ranks_independent_across_groups_and_rounds():
    """Ranks computed independently across 2 groups x 2 rounds."""
    df = pd.DataFrame({
        "session_id": ["s1"] * 8,
        "segment": [1] * 8,
        "group_id": [1, 1, 2, 2, 1, 1, 2, 2],
        "round": [1, 1, 1, 1, 2, 2, 2, 2],
        "player": ["A", "B", "C", "D", "A", "B", "C", "D"],
        "sell_period": [
            1, float("nan"),   # g1r1: A sells p1, B holds
            3, 3,              # g2r1: C,D tie at p3
            float("nan"), 2,   # g1r2: A holds, B sells p2
            1, 5,              # g2r2: C p1, D p5
        ],
        "did_sell": [1, 0, 1, 1, 0, 1, 1, 1],
    })
    result = compute_selling_ranks(df)

    def _get_ranks(gid, rnd):
        sub = result[(result["group_id"] == gid) & (result["round"] == rnd)]
        return dict(zip(sub["player"], sub["sell_rank"]))

    assert _get_ranks(1, 1) == {"A": 1, "B": 4}
    assert _get_ranks(2, 1) == {"C": 1, "D": 1}
    assert _get_ranks(1, 2) == {"A": 4, "B": 1}
    assert _get_ranks(2, 2) == {"C": 1, "D": 2}


@pytest.mark.skipif(not OUTPUT_CSV.exists(), reason="Output CSV not yet built")
def test_real_data_ranks_independent_across_groups_s1_seg1_r3():
    """Verified: session 1, seg 1, round 3 ranks differ across groups."""
    df = pd.read_csv(OUTPUT_CSV)
    g1 = _get_derived_ranks(df, "1_11-7-tr1", 1, 1, 3)
    g2 = _get_derived_ranks(df, "1_11-7-tr1", 1, 2, 3)

    # Group 1: A=1, N=2, J=3, E=hold (verified above)
    assert g1 == {"A": 1, "N": 2, "J": 3, "E": 4}
    # Group 2: F,K tie at p2, B+P hold (from real data inspection)
    assert g2 == {"B": 4, "F": 1, "K": 1, "P": 4}


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
