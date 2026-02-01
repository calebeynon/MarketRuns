"""
Purpose: Unit tests for build_first_seller_round_dataset.py
Author: Claude Code
Date: 2025-02-01

Tests the first seller identification logic:
- A first seller is a player who sells BEFORE their group mates
- All players who sell in the earliest period are first sellers
- If no one sold, no one is marked as first seller
"""

import pandas as pd
import pytest
from analysis.derived.build_first_seller_round_dataset import (
    find_first_sale_info,
    find_sellers_by_period,
    get_player_sell_period,
    process_group_round,
)


# =====
# Helper to create mock DataFrames
# =====
def create_player_df(
    player_label: str = "A",
    n_periods: int = 4,
    sell_in_period: int = None,
    signal: float = 0.5,
    state: int = 1
) -> pd.DataFrame:
    """
    Create a mock player DataFrame for a single round.

    Args:
        player_label: Player label (A, B, C, D)
        n_periods: Number of periods in the round
        sell_in_period: Period when player sells (None = never sell)
        signal: Player's signal value
        state: Asset state

    Returns:
        DataFrame with columns matching raw oTree export structure
    """
    rows = []
    sold_status = 0

    for period in range(1, n_periods + 1):
        if sell_in_period is not None and period >= sell_in_period:
            sold_status = 1

        rows.append({
            "participant.label": player_label,
            "player.period_in_round": period,
            "player.sold": sold_status,
            "player.signal": signal,
            "player.state": state,
            "player.price": 8 - 2 * (period - 1),
            "group.id_in_subsession": 1,
            "player.round_number_in_segment": 1,
        })

    return pd.DataFrame(rows)


def create_group_round_df(
    n_players: int = 4,
    n_periods: int = 4,
    sales_by_player: dict = None,
    signal: float = 0.5,
    state: int = 1
) -> pd.DataFrame:
    """
    Create a mock group-round DataFrame.

    Args:
        n_players: Number of players in group
        n_periods: Number of periods in round
        sales_by_player: Dict mapping player label -> period sold (or None)
        signal: Default signal value
        state: Asset state

    Returns:
        DataFrame with columns matching raw oTree export structure
    """
    if sales_by_player is None:
        sales_by_player = {}

    player_labels = ["A", "B", "C", "D"][:n_players]
    all_dfs = []

    for label in player_labels:
        sell_period = sales_by_player.get(label)
        player_df = create_player_df(
            player_label=label,
            n_periods=n_periods,
            sell_in_period=sell_period,
            signal=signal,
            state=state
        )
        all_dfs.append(player_df)

    return pd.concat(all_dfs, ignore_index=True)


# =====
# Test cases for get_player_sell_period
# =====
def test_player_never_sold():
    """Player holds all periods - returns None."""
    player_df = create_player_df(n_periods=4, sell_in_period=None)
    result = get_player_sell_period(player_df)
    assert result is None


def test_player_sold_period_1():
    """Player sold in period 1."""
    player_df = create_player_df(n_periods=4, sell_in_period=1)
    result = get_player_sell_period(player_df)
    assert result == 1


def test_player_sold_period_3():
    """Player sold in period 3."""
    player_df = create_player_df(n_periods=4, sell_in_period=3)
    result = get_player_sell_period(player_df)
    assert result == 3


def test_player_sold_last_period():
    """Player sold in the last period."""
    player_df = create_player_df(n_periods=4, sell_in_period=4)
    result = get_player_sell_period(player_df)
    assert result == 4


# =====
# Test cases for find_sellers_by_period
# =====
def test_no_sellers():
    """No one sold - returns empty dict."""
    df = create_group_round_df(n_players=4, n_periods=4, sales_by_player={})
    result = find_sellers_by_period(df)
    assert result == {}


def test_one_seller():
    """One player sold."""
    df = create_group_round_df(n_players=4, sales_by_player={"A": 2})
    result = find_sellers_by_period(df)
    assert result == {"A": 2}


def test_multiple_sellers_different_periods():
    """Multiple players sold in different periods."""
    sales = {"A": 1, "B": 2, "C": 3}
    df = create_group_round_df(n_players=4, sales_by_player=sales)
    result = find_sellers_by_period(df)
    assert result == {"A": 1, "B": 2, "C": 3}


def test_all_sellers_same_period():
    """All players sold in the same period."""
    sales = {"A": 2, "B": 2, "C": 2, "D": 2}
    df = create_group_round_df(n_players=4, sales_by_player=sales)
    result = find_sellers_by_period(df)
    assert result == sales


# =====
# Test cases for find_first_sale_info
# =====
def test_first_sale_info_no_sales():
    """No sales - first_sale_period is None, first_sellers is empty."""
    df = create_group_round_df(n_players=4, sales_by_player={})
    result = find_first_sale_info(df)

    assert result["first_sale_period"] is None
    assert result["public_signal"] is None
    assert result["first_sellers"] == set()


def test_first_sale_info_single_first_seller():
    """One player sells first, others sell later."""
    sales = {"A": 1, "B": 3, "C": 4}  # A sells first
    df = create_group_round_df(n_players=4, sales_by_player=sales, signal=0.67)
    result = find_first_sale_info(df)

    assert result["first_sale_period"] == 1
    assert result["public_signal"] == 0.67
    assert result["first_sellers"] == {"A"}


def test_first_sale_info_multiple_first_sellers():
    """Two players sell in the same earliest period."""
    sales = {"A": 2, "B": 2, "C": 4}  # A and B are both first sellers
    df = create_group_round_df(n_players=4, sales_by_player=sales, signal=0.75)
    result = find_first_sale_info(df)

    assert result["first_sale_period"] == 2
    assert result["public_signal"] == 0.75
    assert result["first_sellers"] == {"A", "B"}


def test_first_sale_info_all_sell_same_period():
    """All four players sell in the same period - all are first sellers."""
    sales = {"A": 3, "B": 3, "C": 3, "D": 3}
    df = create_group_round_df(n_players=4, sales_by_player=sales, signal=0.8)
    result = find_first_sale_info(df)

    assert result["first_sale_period"] == 3
    assert result["public_signal"] == 0.8
    assert result["first_sellers"] == {"A", "B", "C", "D"}


# =====
# Test cases for process_group_round
# =====
def test_output_contains_required_columns():
    """Verify output records contain all required columns."""
    df = create_group_round_df(n_players=4, n_periods=4)
    result = process_group_round(
        df, session_name="test_session", segment_idx=1,
        group_id=1, round_num=1, treatment="tr1"
    )

    required_columns = [
        "session_id", "treatment", "segment", "group_id", "round",
        "player", "public_signal", "state", "is_first_seller", "first_sale_period"
    ]

    assert len(result) == 4  # 4 players
    for col in required_columns:
        assert col in result[0], f"Missing column: {col}"


def test_metadata_preserved():
    """Verify session metadata is preserved in output."""
    df = create_group_round_df(n_players=4, n_periods=4)
    result = process_group_round(
        df, session_name="1_11-7-tr1", segment_idx=3,
        group_id=2, round_num=7, treatment="tr1"
    )
    result_df = pd.DataFrame(result)

    assert all(result_df["session_id"] == "1_11-7-tr1")
    assert all(result_df["segment"] == 3)
    assert all(result_df["group_id"] == 2)
    assert all(result_df["round"] == 7)
    assert all(result_df["treatment"] == "tr1")


def test_no_sales_no_first_sellers():
    """No one sold - no one should be marked as first seller."""
    df = create_group_round_df(n_players=4, n_periods=4, sales_by_player={})
    result = process_group_round(
        df, session_name="test", segment_idx=1,
        group_id=1, round_num=1, treatment="tr1"
    )
    result_df = pd.DataFrame(result)

    assert all(result_df["is_first_seller"] == 0)
    assert all(result_df["first_sale_period"].isna())
    assert all(result_df["public_signal"].isna())


def test_single_first_seller_identified():
    """Only the player who sold first should be marked as first seller."""
    sales = {"A": 1, "B": 3, "C": 4}  # A sells first
    df = create_group_round_df(n_players=4, sales_by_player=sales)
    result = process_group_round(
        df, session_name="test", segment_idx=1,
        group_id=1, round_num=1, treatment="tr1"
    )
    result_df = pd.DataFrame(result)

    # Only A should be first seller
    a_row = result_df[result_df["player"] == "A"].iloc[0]
    assert a_row["is_first_seller"] == 1

    # B, C, D should not be first sellers
    for label in ["B", "C", "D"]:
        row = result_df[result_df["player"] == label].iloc[0]
        assert row["is_first_seller"] == 0

    # All should have same first_sale_period
    assert all(result_df["first_sale_period"] == 1)


def test_multiple_first_sellers_same_period():
    """All players who sold in the earliest period are first sellers."""
    sales = {"A": 2, "B": 2, "C": 4}  # A and B both sell in period 2
    df = create_group_round_df(n_players=4, sales_by_player=sales)
    result = process_group_round(
        df, session_name="test", segment_idx=1,
        group_id=1, round_num=1, treatment="tr1"
    )
    result_df = pd.DataFrame(result)

    # A and B should be first sellers
    a_row = result_df[result_df["player"] == "A"].iloc[0]
    b_row = result_df[result_df["player"] == "B"].iloc[0]
    assert a_row["is_first_seller"] == 1
    assert b_row["is_first_seller"] == 1

    # C and D should not be first sellers
    c_row = result_df[result_df["player"] == "C"].iloc[0]
    d_row = result_df[result_df["player"] == "D"].iloc[0]
    assert c_row["is_first_seller"] == 0
    assert d_row["is_first_seller"] == 0

    # All should have same first_sale_period
    assert all(result_df["first_sale_period"] == 2)


def test_all_players_first_sellers():
    """When all players sell in the same period, all are first sellers."""
    sales = {"A": 1, "B": 1, "C": 1, "D": 1}
    df = create_group_round_df(n_players=4, sales_by_player=sales)
    result = process_group_round(
        df, session_name="test", segment_idx=1,
        group_id=1, round_num=1, treatment="tr1"
    )
    result_df = pd.DataFrame(result)

    assert all(result_df["is_first_seller"] == 1)
    assert result_df["is_first_seller"].sum() == 4


def test_public_signal_captured():
    """Public signal should match the signal at first sale period."""
    sales = {"A": 2}
    df = create_group_round_df(n_players=4, sales_by_player=sales, signal=0.675)
    result = process_group_round(
        df, session_name="test", segment_idx=1,
        group_id=1, round_num=1, treatment="tr1"
    )
    result_df = pd.DataFrame(result)

    assert all(result_df["public_signal"] == 0.675)


def test_state_captured():
    """State should be captured for all players."""
    df = create_group_round_df(n_players=4, state=0)
    result = process_group_round(
        df, session_name="test", segment_idx=1,
        group_id=1, round_num=1, treatment="tr1"
    )
    result_df = pd.DataFrame(result)

    assert all(result_df["state"] == 0)


def test_one_row_per_player():
    """Verify each player has exactly one row in output."""
    sales = {"A": 1, "B": 2}
    df = create_group_round_df(n_players=4, n_periods=4, sales_by_player=sales)
    result = process_group_round(
        df, session_name="test", segment_idx=1,
        group_id=1, round_num=1, treatment="tr1"
    )

    # Should have exactly 4 records (one per player)
    assert len(result) == 4

    # Each player should appear exactly once
    players = [r["player"] for r in result]
    assert len(set(players)) == 4


def test_later_seller_not_first_seller():
    """Player who sold after the first seller should not be first seller."""
    sales = {"A": 1, "B": 2, "C": 3, "D": 4}
    df = create_group_round_df(n_players=4, sales_by_player=sales)
    result = process_group_round(
        df, session_name="test", segment_idx=1,
        group_id=1, round_num=1, treatment="tr1"
    )
    result_df = pd.DataFrame(result)

    # Only A should be first seller (sold in period 1)
    assert result_df["is_first_seller"].sum() == 1
    a_row = result_df[result_df["player"] == "A"].iloc[0]
    assert a_row["is_first_seller"] == 1


def test_holder_not_first_seller():
    """Player who never sold should not be first seller."""
    sales = {"A": 1}  # Only A sells, B, C, D hold
    df = create_group_round_df(n_players=4, sales_by_player=sales)
    result = process_group_round(
        df, session_name="test", segment_idx=1,
        group_id=1, round_num=1, treatment="tr1"
    )
    result_df = pd.DataFrame(result)

    # B, C, D should not be first sellers
    for label in ["B", "C", "D"]:
        row = result_df[result_df["player"] == label].iloc[0]
        assert row["is_first_seller"] == 0


# =====
# Edge case tests
# =====
def test_three_first_sellers():
    """Three players sell in the same earliest period."""
    sales = {"A": 1, "B": 1, "C": 1, "D": 4}
    df = create_group_round_df(n_players=4, sales_by_player=sales)
    result = process_group_round(
        df, session_name="test", segment_idx=1,
        group_id=1, round_num=1, treatment="tr1"
    )
    result_df = pd.DataFrame(result)

    # A, B, C should be first sellers
    assert result_df["is_first_seller"].sum() == 3
    for label in ["A", "B", "C"]:
        row = result_df[result_df["player"] == label].iloc[0]
        assert row["is_first_seller"] == 1

    # D should not be first seller
    d_row = result_df[result_df["player"] == "D"].iloc[0]
    assert d_row["is_first_seller"] == 0


def test_first_sale_in_last_period():
    """First sale occurs in the last period."""
    sales = {"A": 4}  # Only A sells, and in the last period
    df = create_group_round_df(n_players=4, n_periods=4, sales_by_player=sales)
    result = process_group_round(
        df, session_name="test", segment_idx=1,
        group_id=1, round_num=1, treatment="tr1"
    )
    result_df = pd.DataFrame(result)

    assert result_df["is_first_seller"].sum() == 1
    assert all(result_df["first_sale_period"] == 4)

    a_row = result_df[result_df["player"] == "A"].iloc[0]
    assert a_row["is_first_seller"] == 1


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
