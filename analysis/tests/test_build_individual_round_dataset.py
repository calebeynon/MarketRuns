"""
Purpose: Unit tests for build_individual_round_dataset.py
Author: Claude Code
Date: 2025-01-18
"""

import pandas as pd
import pytest
from analysis.derived.build_individual_round_dataset import (
    get_player_sell_info,
    process_group_round,
)


# =====
# Helper to create mock DataFrames
# =====
def create_player_df(
    n_periods: int = 3,
    sell_in_period: int = None,
    signal: float = 0.5,
    state: int = 1
):
    """
    Create a mock player DataFrame for a single round.

    Args:
        n_periods: Number of periods in the round
        sell_in_period: Period when player sells (None = never sell)
        signal: Player's private signal
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
            'participant.label': 'A',
            'player.period_in_round': period,
            'player.sold': sold_status,
            'player.signal': signal,
            'player.state': state,
            'player.price': 8 - 2 * (period - 1),
            'group.id_in_subsession': 1,
            'player.round_number_in_segment': 1,
        })

    return pd.DataFrame(rows)


def create_group_round_df(
    n_players: int = 4,
    n_periods: int = 3,
    sales_by_player: dict = None,
    signal: float = 0.5,
    state: int = 1
):
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

    player_labels = ['A', 'B', 'C', 'D'][:n_players]
    rows = []

    for label in player_labels:
        sell_period = sales_by_player.get(label)
        sold_status = 0

        for period in range(1, n_periods + 1):
            if sell_period is not None and period >= sell_period:
                sold_status = 1

            rows.append({
                'participant.label': label,
                'player.period_in_round': period,
                'player.sold': sold_status,
                'player.signal': signal,
                'player.state': state,
                'player.price': 8 - 2 * (period - 1),
                'group.id_in_subsession': 1,
                'player.round_number_in_segment': 1,
            })

    return pd.DataFrame(rows)


# =====
# Test cases for get_player_sell_info
# =====
def test_player_never_sold():
    """Player holds all periods - all values should be None."""
    player_df = create_player_df(n_periods=4, sell_in_period=None)
    sell_period, sell_price, signal = get_player_sell_info(player_df)

    assert sell_period is None
    assert sell_price is None
    assert signal is None


def test_player_sold_period_1():
    """Player sold in period 1."""
    player_df = create_player_df(n_periods=4, sell_in_period=1, signal=0.6)
    sell_period, sell_price, signal = get_player_sell_info(player_df)

    assert sell_period == 1
    assert sell_price == 8  # Price in period 1
    assert signal == 0.6


def test_player_sold_period_3():
    """Player sold in period 3."""
    player_df = create_player_df(n_periods=4, sell_in_period=3, signal=0.7)
    sell_period, sell_price, signal = get_player_sell_info(player_df)

    assert sell_period == 3
    assert sell_price == 4  # Price in period 3 (8 - 2*2)
    assert signal == 0.7


def test_player_sold_last_period():
    """Player sold in the last period."""
    player_df = create_player_df(n_periods=4, sell_in_period=4, signal=0.8)
    sell_period, sell_price, signal = get_player_sell_info(player_df)

    assert sell_period == 4
    assert sell_price == 2  # Price in period 4 (8 - 2*3)
    assert signal == 0.8


# =====
# Test cases for process_group_round
# =====
def test_output_contains_required_columns():
    """Verify output records contain all required columns."""
    df = create_group_round_df(n_players=4, n_periods=3)
    result = process_group_round(
        df, session_name='test_session', segment_idx=1,
        group_id=1, round_num=1, treatment='tr1'
    )

    required_columns = [
        'session_id', 'treatment', 'segment', 'group_id', 'round',
        'player', 'signal', 'state', 'sell_period', 'did_sell', 'sell_price'
    ]

    assert len(result) == 4  # 4 players
    for col in required_columns:
        assert col in result[0], f"Missing column: {col}"


def test_metadata_preserved():
    """Verify session metadata is preserved in output."""
    df = create_group_round_df(n_players=4, n_periods=2)
    result = process_group_round(
        df, session_name='1_11-7-tr1', segment_idx=3,
        group_id=2, round_num=7, treatment='tr1'
    )
    result_df = pd.DataFrame(result)

    assert all(result_df['session_id'] == '1_11-7-tr1')
    assert all(result_df['segment'] == 3)
    assert all(result_df['group_id'] == 2)
    assert all(result_df['round'] == 7)
    assert all(result_df['treatment'] == 'tr1')


def test_signal_and_state_captured():
    """Verify state is captured; signal is NA for non-sellers."""
    # Non-sellers should have NA signal
    df = create_group_round_df(n_players=2, n_periods=2, signal=0.75, state=0)
    result = process_group_round(
        df, session_name='test', segment_idx=1,
        group_id=1, round_num=1, treatment='tr1'
    )
    result_df = pd.DataFrame(result)

    assert all(result_df['signal'].isna())  # No sales, so no signal
    assert all(result_df['state'] == 0)


def test_no_sales_in_round():
    """No one sold - all did_sell should be 0, all signals NA."""
    df = create_group_round_df(n_players=4, n_periods=3, sales_by_player={})
    result = process_group_round(
        df, session_name='test', segment_idx=1,
        group_id=1, round_num=1, treatment='tr1'
    )
    result_df = pd.DataFrame(result)

    assert all(result_df['did_sell'] == 0)
    assert all(result_df['sell_period'].isna())
    assert all(result_df['sell_price'].isna())
    assert all(result_df['signal'].isna())


def test_all_players_sell():
    """All players sell in different periods."""
    sales = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    df = create_group_round_df(n_players=4, n_periods=4, sales_by_player=sales)
    result = process_group_round(
        df, session_name='test', segment_idx=1,
        group_id=1, round_num=1, treatment='tr1'
    )
    result_df = pd.DataFrame(result)

    assert all(result_df['did_sell'] == 1)

    # Check each player's sell_period
    for label, expected_period in sales.items():
        player_row = result_df[result_df['player'] == label].iloc[0]
        assert player_row['sell_period'] == expected_period


def test_mixed_sellers_and_holders():
    """Some players sell, others hold."""
    sales = {'A': 1, 'C': 3}  # B and D never sell
    df = create_group_round_df(n_players=4, n_periods=4, sales_by_player=sales)
    result = process_group_round(
        df, session_name='test', segment_idx=1,
        group_id=1, round_num=1, treatment='tr1'
    )
    result_df = pd.DataFrame(result)

    # Check sellers
    a_row = result_df[result_df['player'] == 'A'].iloc[0]
    assert a_row['did_sell'] == 1
    assert a_row['sell_period'] == 1

    c_row = result_df[result_df['player'] == 'C'].iloc[0]
    assert c_row['did_sell'] == 1
    assert c_row['sell_period'] == 3

    # Check holders
    b_row = result_df[result_df['player'] == 'B'].iloc[0]
    assert b_row['did_sell'] == 0
    assert pd.isna(b_row['sell_period'])

    d_row = result_df[result_df['player'] == 'D'].iloc[0]
    assert d_row['did_sell'] == 0
    assert pd.isna(d_row['sell_price'])


def test_sell_price_matches_period():
    """Verify sell_price matches the price at the sell period."""
    sales = {'A': 1, 'B': 2, 'C': 3}
    df = create_group_round_df(n_players=3, n_periods=3, sales_by_player=sales)
    result = process_group_round(
        df, session_name='test', segment_idx=1,
        group_id=1, round_num=1, treatment='tr1'
    )
    result_df = pd.DataFrame(result)

    # Price formula: 8 - 2*(period-1)
    expected_prices = {'A': 8, 'B': 6, 'C': 4}

    for label, expected_price in expected_prices.items():
        player_row = result_df[result_df['player'] == label].iloc[0]
        assert player_row['sell_price'] == expected_price


def test_one_row_per_player():
    """Verify each player has exactly one row in output."""
    sales = {'A': 1, 'B': 2}
    df = create_group_round_df(n_players=4, n_periods=4, sales_by_player=sales)
    result = process_group_round(
        df, session_name='test', segment_idx=1,
        group_id=1, round_num=1, treatment='tr1'
    )

    # Should have exactly 4 records (one per player)
    assert len(result) == 4

    # Each player should appear exactly once
    players = [r['player'] for r in result]
    assert len(set(players)) == 4


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
