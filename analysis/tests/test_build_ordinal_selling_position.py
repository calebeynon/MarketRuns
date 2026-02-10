"""
Purpose: Unit tests for build_ordinal_selling_position.py
Author: Claude Code
Date: 2026-02-09
"""

import pandas as pd
import pytest
from pathlib import Path
from analysis.derived.build_ordinal_selling_position import (
    compute_selling_ranks,
    determine_emotion_periods,
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


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
