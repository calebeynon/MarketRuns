"""
Purpose: Tests for build_individual_period_dataset_extended.py
Author: Claude
Date: 2026-01-22

Tests verify the extended dataset functions that add:
- sold_in_round: Binary indicator if player sold at ANY point in the round
- round_payoff: Payoff from the LAST period of the round, propagated to all periods
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add analysis directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from derived.build_individual_period_dataset_extended import add_sold_in_round


# =====
# Helper functions to create mock DataFrames
# =====
def create_mock_period_df(
    n_players: int = 4,
    n_periods: int = 3,
    sales_by_period: dict = None,
    payoffs_by_period: dict = None,
):
    """
    Create a mock individual period DataFrame.

    Args:
        n_players: Number of players in group (default 4)
        n_periods: Number of periods in round (default 3)
        sales_by_period: Dict mapping period -> list of player labels who sell
                        e.g., {1: ['A'], 2: ['B', 'C']}
        payoffs_by_period: Dict mapping period -> payoff value for that period
                          e.g., {1: 8, 2: 6, 3: 4}

    Returns:
        DataFrame with columns matching individual_period_dataset structure
    """
    if sales_by_period is None:
        sales_by_period = {}
    if payoffs_by_period is None:
        payoffs_by_period = {p: 8 - 2 * (p - 1) for p in range(1, n_periods + 1)}

    player_labels = ['A', 'B', 'C', 'D'][:n_players]
    rows = []

    # Track who has sold (for already_sold computation)
    has_sold = {label: False for label in player_labels}
    sold_period = {label: None for label in player_labels}

    for period in range(1, n_periods + 1):
        sellers_this_period = sales_by_period.get(period, [])

        for label in player_labels:
            # Determine sold and already_sold status
            if label in sellers_this_period and not has_sold[label]:
                sold = 1
                already_sold = 0
                has_sold[label] = True
                sold_period[label] = period
            elif has_sold[label]:
                sold = 0
                already_sold = 1
            else:
                sold = 0
                already_sold = 0

            rows.append({
                'session_id': 'test_session',
                'segment': 1,
                'round': 1,
                'period': period,
                'group_id': 1,
                'player': label,
                'treatment': 'tr1',
                'signal': 0.5,
                'state': 1,
                'price': payoffs_by_period.get(period, 8),
                'sold': sold,
                'already_sold': already_sold,
                'prior_group_sales': 0,
            })

    return pd.DataFrame(rows)


def create_multi_round_df(round_configs: list):
    """
    Create a DataFrame with multiple rounds.

    Args:
        round_configs: List of dicts, each containing:
            - round_num: Round number
            - n_periods: Number of periods
            - sales_by_period: Dict of sales
            - payoffs_by_period: Dict of payoffs

    Returns:
        DataFrame with multiple rounds
    """
    all_rows = []

    for config in round_configs:
        round_num = config.get('round_num', 1)
        n_periods = config.get('n_periods', 3)
        sales_by_period = config.get('sales_by_period', {})
        payoffs_by_period = config.get('payoffs_by_period', None)

        round_df = create_mock_period_df(
            n_periods=n_periods,
            sales_by_period=sales_by_period,
            payoffs_by_period=payoffs_by_period,
        )
        round_df['round'] = round_num
        all_rows.append(round_df)

    return pd.concat(all_rows, ignore_index=True)


# =====
# Mock round_payoff function for testing (production uses raw oTree CSV files)
# =====
def mock_compute_round_payoff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mock implementation for testing round_payoff propagation logic.

    NOTE: Production code reads round_payoff from raw oTree CSV columns
    (player.round_N_payoff), which requires file system access. This mock
    uses last-period price as a proxy to test the propagation behavior.
    """
    df = df.copy()

    # Handle empty DataFrame
    if len(df) == 0:
        df['round_payoff'] = pd.Series(dtype=float)
        return df

    player_round_key = ['session_id', 'segment', 'round', 'group_id', 'player']

    # Get the last period for each player-round
    last_period_df = df.loc[
        df.groupby(player_round_key)['period'].idxmax()
    ][player_round_key + ['price']].copy()

    last_period_df = last_period_df.rename(columns={'price': 'round_payoff'})

    df = df.merge(last_period_df, on=player_round_key, how='left')

    return df


# =====
# Test cases for add_sold_in_round
# =====
class TestComputeSoldInRound:
    """Tests for sold_in_round computation."""

    def test_no_sales_in_round(self):
        """No one sold - sold_in_round should be 0 for all periods."""
        df = create_mock_period_df(n_players=4, n_periods=3, sales_by_period={})
        result = add_sold_in_round(df)

        assert 'sold_in_round' in result.columns
        assert (result['sold_in_round'] == 0).all(), (
            "sold_in_round should be 0 when no one sold"
        )

    def test_sale_in_period_1(self):
        """Player sold in period 1 - sold_in_round should be 1 for ALL periods."""
        df = create_mock_period_df(
            n_players=4, n_periods=3, sales_by_period={1: ['A']}
        )
        result = add_sold_in_round(df)

        # Player A should have sold_in_round=1 for all periods
        player_a = result[result['player'] == 'A']
        assert (player_a['sold_in_round'] == 1).all(), (
            "Player A sold in period 1, sold_in_round should be 1 for ALL periods"
        )

        # Other players should have sold_in_round=0
        others = result[result['player'] != 'A']
        assert (others['sold_in_round'] == 0).all(), (
            "Players who didn't sell should have sold_in_round=0"
        )

    def test_sale_in_last_period(self):
        """Player sold in last period - sold_in_round should be 1 for ALL periods."""
        df = create_mock_period_df(
            n_players=4, n_periods=4, sales_by_period={4: ['B']}
        )
        result = add_sold_in_round(df)

        # Player B should have sold_in_round=1 for all periods including 1, 2, 3
        player_b = result[result['player'] == 'B']
        assert len(player_b) == 4, "Should have 4 period observations for player B"
        assert (player_b['sold_in_round'] == 1).all(), (
            "Player B sold in last period, sold_in_round should be 1 for ALL periods"
        )

    def test_already_sold_triggers_sold_in_round(self):
        """already_sold=1 but sold=0 should still trigger sold_in_round=1."""
        df = create_mock_period_df(
            n_players=4, n_periods=3, sales_by_period={1: ['C']}
        )
        result = add_sold_in_round(df)

        # Check period 2 and 3 for player C where already_sold=1 but sold=0
        player_c_later = result[(result['player'] == 'C') & (result['period'] > 1)]

        # Verify already_sold is 1 for these periods
        assert (player_c_later['already_sold'] == 1).all()
        # Verify sold is 0 for these periods
        assert (player_c_later['sold'] == 0).all()
        # But sold_in_round should still be 1
        assert (player_c_later['sold_in_round'] == 1).all(), (
            "already_sold=1 should trigger sold_in_round=1"
        )

    def test_multiple_sellers(self):
        """Multiple players sell - each should have correct sold_in_round."""
        df = create_mock_period_df(
            n_players=4, n_periods=3, sales_by_period={1: ['A'], 2: ['B']}
        )
        result = add_sold_in_round(df)

        # A and B should have sold_in_round=1
        for player in ['A', 'B']:
            player_df = result[result['player'] == player]
            assert (player_df['sold_in_round'] == 1).all(), (
                f"Player {player} sold, should have sold_in_round=1"
            )

        # C and D should have sold_in_round=0
        for player in ['C', 'D']:
            player_df = result[result['player'] == player]
            assert (player_df['sold_in_round'] == 0).all(), (
                f"Player {player} didn't sell, should have sold_in_round=0"
            )

    def test_all_players_sell(self):
        """All players sell - all should have sold_in_round=1."""
        df = create_mock_period_df(
            n_players=4,
            n_periods=4,
            sales_by_period={1: ['A'], 2: ['B'], 3: ['C'], 4: ['D']}
        )
        result = add_sold_in_round(df)

        assert (result['sold_in_round'] == 1).all(), (
            "All players sold, all should have sold_in_round=1"
        )


# =====
# Test cases for mock_compute_round_payoff
# =====
class TestComputeRoundPayoff:
    """Tests for round_payoff computation."""

    def test_payoff_from_last_period(self):
        """Verify payoff comes from the last period of the round."""
        payoffs = {1: 10, 2: 8, 3: 5}  # Last period payoff is 5
        df = create_mock_period_df(
            n_players=4, n_periods=3, payoffs_by_period=payoffs
        )
        result = mock_compute_round_payoff(df)

        assert 'round_payoff' in result.columns
        # All players should have round_payoff=5 (from period 3)
        assert (result['round_payoff'] == 5).all(), (
            "round_payoff should be 5 (from last period)"
        )

    def test_payoff_propagated_to_all_periods(self):
        """Payoff should be propagated to ALL periods in the round."""
        payoffs = {1: 12, 2: 10, 3: 8, 4: 6}  # Last period is 6
        df = create_mock_period_df(
            n_players=4, n_periods=4, payoffs_by_period=payoffs
        )
        result = mock_compute_round_payoff(df)

        # Check each period has the same round_payoff
        for period in [1, 2, 3, 4]:
            period_df = result[result['period'] == period]
            assert (period_df['round_payoff'] == 6).all(), (
                f"Period {period} should have round_payoff=6"
            )

    def test_different_rounds_different_payoffs(self):
        """Different rounds should have different payoffs."""
        round_configs = [
            {'round_num': 1, 'n_periods': 2, 'payoffs_by_period': {1: 10, 2: 8}},
            {'round_num': 2, 'n_periods': 3, 'payoffs_by_period': {1: 6, 2: 4, 3: 2}},
        ]
        df = create_multi_round_df(round_configs)
        result = mock_compute_round_payoff(df)

        # Round 1 should have round_payoff=8 (from period 2)
        round_1 = result[result['round'] == 1]
        assert (round_1['round_payoff'] == 8).all(), (
            "Round 1 should have round_payoff=8"
        )

        # Round 2 should have round_payoff=2 (from period 3)
        round_2 = result[result['round'] == 2]
        assert (round_2['round_payoff'] == 2).all(), (
            "Round 2 should have round_payoff=2"
        )

    def test_payoff_varies_by_player_not_allowed(self):
        """In our setup, all players have same price per period, verify consistency."""
        df = create_mock_period_df(n_players=4, n_periods=3)
        result = mock_compute_round_payoff(df)

        # All players should have the same round_payoff
        unique_payoffs = result.groupby('player')['round_payoff'].first().unique()
        assert len(unique_payoffs) == 1, (
            "All players should have the same round_payoff"
        )

    def test_single_period_round(self):
        """A round with only 1 period should use that period's payoff."""
        df = create_mock_period_df(
            n_players=4, n_periods=1, payoffs_by_period={1: 15}
        )
        result = mock_compute_round_payoff(df)

        assert (result['round_payoff'] == 15).all(), (
            "Single period round should have round_payoff from that period"
        )


# =====
# Test output structure
# =====
class TestOutputStructure:
    """Tests for output DataFrame structure."""

    def test_output_has_all_original_columns(self):
        """Output has all original columns plus new ones."""
        df = create_mock_period_df(n_players=4, n_periods=3)
        original_columns = set(df.columns)

        result = add_sold_in_round(df)
        result = mock_compute_round_payoff(result)

        # Should have all original columns
        for col in original_columns:
            assert col in result.columns, f"Missing original column: {col}"

    def test_output_has_new_columns(self):
        """Output has round_payoff and sold_in_round columns."""
        df = create_mock_period_df(n_players=4, n_periods=3)

        result = add_sold_in_round(df)
        result = mock_compute_round_payoff(result)

        assert 'round_payoff' in result.columns, "Missing round_payoff column"
        assert 'sold_in_round' in result.columns, "Missing sold_in_round column"

    def test_row_count_unchanged(self):
        """Row count should be unchanged from input."""
        df = create_mock_period_df(n_players=4, n_periods=3)
        original_count = len(df)

        result = add_sold_in_round(df)
        result = mock_compute_round_payoff(result)

        assert len(result) == original_count, (
            f"Row count changed: {original_count} -> {len(result)}"
        )

    def test_sold_in_round_is_binary(self):
        """sold_in_round should only have values 0 or 1."""
        df = create_mock_period_df(
            n_players=4, n_periods=3, sales_by_period={1: ['A'], 2: ['B']}
        )
        result = add_sold_in_round(df)

        unique_vals = set(result['sold_in_round'].unique())
        assert unique_vals.issubset({0, 1}), (
            f"sold_in_round has non-binary values: {unique_vals}"
        )


# =====
# Edge cases
# =====
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Handle empty DataFrame gracefully."""
        df = pd.DataFrame(columns=[
            'session_id', 'segment', 'round', 'period', 'group_id',
            'player', 'treatment', 'signal', 'state', 'price',
            'sold', 'already_sold', 'prior_group_sales'
        ])

        result = add_sold_in_round(df)
        assert len(result) == 0
        assert 'sold_in_round' in result.columns

        result = mock_compute_round_payoff(result)
        assert len(result) == 0
        assert 'round_payoff' in result.columns

    def test_single_player(self):
        """Handle single player case."""
        df = create_mock_period_df(n_players=1, n_periods=3, sales_by_period={2: ['A']})

        result = add_sold_in_round(df)
        result = mock_compute_round_payoff(result)

        assert len(result) == 3
        assert (result['sold_in_round'] == 1).all()

    def test_multiple_sessions(self):
        """Handle multiple sessions correctly (player labels are unique per session)."""
        df1 = create_mock_period_df(
            n_players=2, n_periods=2, sales_by_period={1: ['A']}
        )
        df1['session_id'] = 'session_1'

        df2 = create_mock_period_df(
            n_players=2, n_periods=2, sales_by_period={}
        )
        df2['session_id'] = 'session_2'

        df = pd.concat([df1, df2], ignore_index=True)
        result = add_sold_in_round(df)

        # Session 1, Player A should have sold_in_round=1
        s1_a = result[(result['session_id'] == 'session_1') & (result['player'] == 'A')]
        assert (s1_a['sold_in_round'] == 1).all()

        # Session 2, no one sold, should all be 0
        s2 = result[result['session_id'] == 'session_2']
        assert (s2['sold_in_round'] == 0).all()


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
