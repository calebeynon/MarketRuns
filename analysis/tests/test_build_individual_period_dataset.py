"""
Purpose: Unit tests for build_individual_period_dataset.py
Author: Claude Code
Date: 2025-01-14
"""

import pandas as pd
import pytest
from analysis.derived.build_individual_period_dataset import process_group_round


# =====
# Helper to create mock DataFrames
# =====
def create_group_round_df(
    n_players: int = 4,
    n_periods: int = 3,
    sales_by_period: dict = None
):
    """
    Create a mock group-round DataFrame.

    Args:
        n_players: Number of players in group (default 4)
        n_periods: Number of periods in round (default 3)
        sales_by_period: Dict mapping period -> list of player labels who sell
                        e.g., {1: ['A'], 2: ['B', 'C']} means A sells in p1,
                        B and C sell in p2

    Returns:
        DataFrame with columns matching raw oTree export structure
    """
    if sales_by_period is None:
        sales_by_period = {}

    player_labels = ['A', 'B', 'C', 'D'][:n_players]
    rows = []

    # Track cumulative sold status for each player
    cumulative_sold = {label: 0 for label in player_labels}

    for period in range(1, n_periods + 1):
        # Update cumulative sold based on who sells this period
        sellers_this_period = sales_by_period.get(period, [])
        for seller in sellers_this_period:
            cumulative_sold[seller] = 1

        # Create row for each player in this period
        for label in player_labels:
            rows.append({
                'participant.label': label,
                'player.period_in_round': period,
                'player.sold': cumulative_sold[label],
                'player.signal': 0.5,
                'player.state': 1,
                'player.price': 8 - 2 * (period - 1),
                'group.id_in_subsession': 1,
            })

    return pd.DataFrame(rows)


# =====
# Test cases for process_group_round
# =====
def test_no_sales():
    """No one sold - all computed fields should be 0."""
    df = create_group_round_df(n_players=4, n_periods=3, sales_by_period={})
    result = process_group_round(
        df, session_name='test', segment_idx=1,
        group_id=1, round_num=1, treatment=1
    )

    assert len(result) == 12  # 4 players * 3 periods
    assert all(r['sold'] == 0 for r in result)
    assert all(r['already_sold'] == 0 for r in result)
    assert all(r['prior_group_sales'] == 0 for r in result)


def test_single_sale_period_1():
    """Player A sells in period 1."""
    df = create_group_round_df(
        n_players=4, n_periods=3, sales_by_period={1: ['A']}
    )
    result = process_group_round(
        df, session_name='test', segment_idx=1,
        group_id=1, round_num=1, treatment=1
    )

    # Convert to DataFrame for easier querying
    result_df = pd.DataFrame(result)

    # Player A in period 1: sold=1, prior_group_sales=0 (no prior periods)
    a_p1 = result_df[
        (result_df['player'] == 'A') & (result_df['period'] == 1)
    ].iloc[0]
    assert a_p1['sold'] == 1
    assert a_p1['prior_group_sales'] == 0

    # Other players in period 1: sold=0, prior_group_sales=0
    others_p1 = result_df[
        (result_df['player'] != 'A') & (result_df['period'] == 1)
    ]
    assert all(others_p1['sold'] == 0)
    assert all(others_p1['prior_group_sales'] == 0)


def test_single_sale_period_2():
    """Player A sells in period 2 (sold=0 in p1, sold=1 in p2)."""
    df = create_group_round_df(
        n_players=4, n_periods=3, sales_by_period={2: ['A']}
    )
    result = process_group_round(
        df, session_name='test', segment_idx=1,
        group_id=1, round_num=1, treatment=1
    )
    result_df = pd.DataFrame(result)

    # Period 1: no one sold yet
    p1 = result_df[result_df['period'] == 1]
    assert all(p1['sold'] == 0)
    assert all(p1['prior_group_sales'] == 0)

    # Period 2: A sells, prior_group_sales=0 (no sales in p1)
    a_p2 = result_df[
        (result_df['player'] == 'A') & (result_df['period'] == 2)
    ].iloc[0]
    assert a_p2['sold'] == 1
    assert a_p2['prior_group_sales'] == 0


def test_multiple_sales_same_period():
    """Players A and B both sell in period 1."""
    df = create_group_round_df(
        n_players=4, n_periods=3, sales_by_period={1: ['A', 'B']}
    )
    result = process_group_round(
        df, session_name='test', segment_idx=1,
        group_id=1, round_num=1, treatment=1
    )
    result_df = pd.DataFrame(result)

    # Both A and B have sold=1 in period 1
    ab_p1 = result_df[
        (result_df['player'].isin(['A', 'B'])) &
        (result_df['period'] == 1)
    ]
    assert all(ab_p1['sold'] == 1)

    # prior_group_sales=0 for all in period 1 (same period doesn't count)
    p1 = result_df[result_df['period'] == 1]
    assert all(p1['prior_group_sales'] == 0)


def test_sequential_sales():
    """Player A sells period 1, Player B sells period 2."""
    df = create_group_round_df(
        n_players=4, n_periods=3, sales_by_period={1: ['A'], 2: ['B']}
    )
    result = process_group_round(
        df, session_name='test', segment_idx=1,
        group_id=1, round_num=1, treatment=1
    )
    result_df = pd.DataFrame(result)

    # Period 1: A sold, B not yet
    a_p1 = result_df[
        (result_df['player'] == 'A') & (result_df['period'] == 1)
    ].iloc[0]
    assert a_p1['sold'] == 1

    b_p1 = result_df[
        (result_df['player'] == 'B') & (result_df['period'] == 1)
    ].iloc[0]
    assert b_p1['sold'] == 0
    assert b_p1['prior_group_sales'] == 0

    # Period 2: B sells, A already sold
    b_p2 = result_df[
        (result_df['player'] == 'B') & (result_df['period'] == 2)
    ].iloc[0]
    assert b_p2['sold'] == 1
    assert b_p2['prior_group_sales'] == 1  # A sold in period 1

    a_p2 = result_df[
        (result_df['player'] == 'A') & (result_df['period'] == 2)
    ].iloc[0]
    assert a_p2['already_sold'] == 1


def test_already_sold_tracking():
    """Player A sells in period 1, verify already_sold in subsequent periods."""
    df = create_group_round_df(
        n_players=4, n_periods=4, sales_by_period={1: ['A']}
    )
    result = process_group_round(
        df, session_name='test', segment_idx=1,
        group_id=1, round_num=1, treatment=1
    )
    result_df = pd.DataFrame(result)

    # A in period 1: sold=1, already_sold=0 (selling now, not already)
    a_p1 = result_df[
        (result_df['player'] == 'A') & (result_df['period'] == 1)
    ].iloc[0]
    assert a_p1['sold'] == 1
    assert a_p1['already_sold'] == 0

    # A in periods 2, 3, 4: already_sold=1
    for period in [2, 3, 4]:
        a_row = result_df[
            (result_df['player'] == 'A') &
            (result_df['period'] == period)
        ].iloc[0]
        assert a_row['already_sold'] == 1
        assert a_row['sold'] == 0  # Didn't sell THIS period


def test_prior_group_sales_excludes_self():
    """Verify prior_group_sales doesn't count the player's own prior sale."""
    df = create_group_round_df(
        n_players=4, n_periods=3, sales_by_period={1: ['A'], 2: ['B']}
    )
    result = process_group_round(
        df, session_name='test', segment_idx=1,
        group_id=1, round_num=1, treatment=1
    )
    result_df = pd.DataFrame(result)

    # In period 2, B should see prior_group_sales=1 (A sold in p1)
    b_p2 = result_df[
        (result_df['player'] == 'B') & (result_df['period'] == 2)
    ].iloc[0]
    assert b_p2['prior_group_sales'] == 1

    # In period 2, A (already sold) should see prior_group_sales=0
    # (doesn't count own sale)
    a_p2 = result_df[
        (result_df['player'] == 'A') & (result_df['period'] == 2)
    ].iloc[0]
    assert a_p2['prior_group_sales'] == 0


def test_all_players_sell_different_periods():
    """4 players sell in periods 1,2,3,4 - verify cumulative prior_group_sales."""
    df = create_group_round_df(
        n_players=4, n_periods=4,
        sales_by_period={1: ['A'], 2: ['B'], 3: ['C'], 4: ['D']}
    )
    result = process_group_round(
        df, session_name='test', segment_idx=1,
        group_id=1, round_num=1, treatment=1
    )
    result_df = pd.DataFrame(result)

    # Period 1: A sells, prior_group_sales=0 for all
    p1 = result_df[result_df['period'] == 1]
    assert all(p1['prior_group_sales'] == 0)

    # Period 2: B sells, prior_group_sales should be 1 for B,C,D (A sold in p1)
    # A already sold so doesn't count others for their decision
    b_p2 = result_df[
        (result_df['player'] == 'B') & (result_df['period'] == 2)
    ].iloc[0]
    assert b_p2['prior_group_sales'] == 1

    # Period 3: C sells, prior_group_sales=2 for C,D (A sold in p1, B in p2)
    c_p3 = result_df[
        (result_df['player'] == 'C') & (result_df['period'] == 3)
    ].iloc[0]
    assert c_p3['prior_group_sales'] == 2

    # Period 4: D sells, prior_group_sales=3 (A,B,C all sold before)
    d_p4 = result_df[
        (result_df['player'] == 'D') & (result_df['period'] == 4)
    ].iloc[0]
    assert d_p4['prior_group_sales'] == 3


def test_output_contains_required_columns():
    """Verify output records contain all required columns."""
    df = create_group_round_df(n_players=4, n_periods=2, sales_by_period={})
    result = process_group_round(
        df, session_name='test_session', segment_idx=2,
        group_id=3, round_num=5, treatment=1
    )

    required_columns = [
        'session_id', 'treatment', 'segment', 'group_id', 'round',
        'player', 'period', 'sold', 'already_sold',
        'prior_group_sales', 'signal', 'state', 'price'
    ]

    assert len(result) > 0
    for col in required_columns:
        assert col in result[0], f"Missing column: {col}"


def test_metadata_preserved():
    """Verify session metadata is preserved in output."""
    df = create_group_round_df(n_players=4, n_periods=2, sales_by_period={})
    result = process_group_round(
        df, session_name='1_11-7-tr1', segment_idx=3,
        group_id=2, round_num=7, treatment=1
    )
    result_df = pd.DataFrame(result)

    assert all(result_df['session_id'] == '1_11-7-tr1')
    assert all(result_df['segment'] == 3)
    assert all(result_df['group_id'] == 2)
    assert all(result_df['round'] == 7)
    assert all(result_df['treatment'] == 1)


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
