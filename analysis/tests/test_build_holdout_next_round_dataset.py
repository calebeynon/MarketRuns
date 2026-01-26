"""
Purpose: Unit tests for build_holdout_next_round_dataset.py
Author: Claude
Date: 2026-01-22

Tests verify the holdout next-round analysis dataset builder that:
- Filters to treatment 1, state 0, holdouts only
- Links each holdout to their next round behavior
- Computes prior sales count
"""

import pandas as pd
import pytest
from analysis.derived.build_holdout_next_round_dataset import (
    filter_to_holdouts,
    aggregate_to_round_level,
    add_identifiers,
    link_next_round_behavior,
    build_next_round_lookup,
    compute_prior_sales,
    build_sales_history,
)


# =====
# Helper functions to create mock DataFrames
# =====
def create_mock_period_df(
    session_id: str = "test_session",
    segment: int = 1,
    round_num: int = 1,
    n_periods: int = 3,
    n_players: int = 4,
    treatment: str = "tr1",
    state: int = 0,
    sold_in_round_by_player: dict = None,
    round_payoff: float = 0.0,
) -> pd.DataFrame:
    """
    Create a mock individual period DataFrame for one round.

    Args:
        session_id: Session identifier
        segment: Segment number (1-4)
        round_num: Round number within segment (1-14)
        n_periods: Number of periods in round
        n_players: Number of players
        treatment: Treatment identifier ('tr1' or 'tr2')
        state: Asset state (0 or 1)
        sold_in_round_by_player: Dict mapping player -> 1 if sold, 0 if not
        round_payoff: Payoff value for the round

    Returns:
        DataFrame matching individual_period_dataset_extended structure
    """
    if sold_in_round_by_player is None:
        sold_in_round_by_player = {}

    player_labels = ['A', 'B', 'C', 'D'][:n_players]
    rows = []

    for period in range(1, n_periods + 1):
        for i, player in enumerate(player_labels):
            sold_in_round = sold_in_round_by_player.get(player, 0)
            rows.append({
                'session_id': session_id,
                'segment': segment,
                'round': round_num,
                'period': period,
                'group_id': 1,
                'player': player,
                'treatment': treatment,
                'signal': 0.5,
                'state': state,
                'price': 8 - 2 * (period - 1),
                'sold': 0,
                'already_sold': 0,
                'sold_in_round': sold_in_round,
                'round_payoff': round_payoff,
            })

    return pd.DataFrame(rows)


def create_multi_round_df(round_configs: list) -> pd.DataFrame:
    """
    Create a DataFrame with multiple rounds.

    Args:
        round_configs: List of dicts, each containing parameters for
                       create_mock_period_df

    Returns:
        DataFrame with multiple rounds
    """
    all_dfs = []
    for config in round_configs:
        round_df = create_mock_period_df(**config)
        all_dfs.append(round_df)

    return pd.concat(all_dfs, ignore_index=True)


def create_holdout_round_df(
    session_id: str = "test_session",
    segment: int = 1,
    round_num: int = 1,
    player: str = "A",
    group_id: int = 1,
    round_payoff: float = 0.0,
) -> pd.DataFrame:
    """Create a minimal holdout round-level DataFrame."""
    return pd.DataFrame([{
        'session_id': session_id,
        'segment': segment,
        'round': round_num,
        'player': player,
        'group_id': group_id,
        'round_payoff': round_payoff,
    }])


# =====
# Test cases for filter_to_holdouts
# =====
class TestFilterToHoldouts:
    """Tests for filter_to_holdouts function."""

    def test_filters_to_treatment_1_only(self):
        """Verify tr2 sessions are excluded."""
        round_configs = [
            {'treatment': 'tr1', 'state': 0, 'sold_in_round_by_player': {}},
            {'treatment': 'tr2', 'state': 0, 'sold_in_round_by_player': {}},
        ]
        # Set different session IDs
        round_configs[0]['session_id'] = 'tr1_session'
        round_configs[1]['session_id'] = 'tr2_session'

        df = create_multi_round_df(round_configs)
        result = filter_to_holdouts(df)

        assert len(result) > 0, "Should have some holdouts"
        assert (result['treatment'] == 'tr1').all(), (
            "All results should be treatment 1"
        )
        assert 'tr2_session' not in result['session_id'].values, (
            "tr2 session should be excluded"
        )

    def test_filters_to_state_0_only(self):
        """Verify state=1 rounds are excluded."""
        round_configs = [
            {'state': 0, 'sold_in_round_by_player': {}},
            {'state': 1, 'sold_in_round_by_player': {}, 'round_num': 2},
        ]
        df = create_multi_round_df(round_configs)
        result = filter_to_holdouts(df)

        assert len(result) > 0, "Should have some holdouts"
        assert (result['state'] == 0).all(), (
            "All results should have state=0"
        )

    def test_filters_to_holdouts_only(self):
        """Verify sold_in_round=1 observations are excluded."""
        # Create round with some sellers and some holdouts
        sold_in_round_by_player = {'A': 1, 'B': 1, 'C': 0, 'D': 0}
        df = create_mock_period_df(
            state=0,
            sold_in_round_by_player=sold_in_round_by_player
        )
        result = filter_to_holdouts(df)

        assert len(result) > 0, "Should have some holdouts"
        assert (result['sold_in_round'] == 0).all(), (
            "All results should be holdouts (sold_in_round=0)"
        )
        # Only players C and D should remain
        assert set(result['player'].unique()) == {'C', 'D'}, (
            "Only holdout players should remain"
        )

    def test_excludes_all_sellers(self):
        """When all players sold, no holdouts should remain."""
        sold_in_round_by_player = {'A': 1, 'B': 1, 'C': 1, 'D': 1}
        df = create_mock_period_df(
            state=0,
            sold_in_round_by_player=sold_in_round_by_player
        )
        result = filter_to_holdouts(df)

        assert len(result) == 0, "No holdouts when everyone sold"


# =====
# Test cases for aggregate_to_round_level
# =====
class TestAggregateToRoundLevel:
    """Tests for aggregate_to_round_level function."""

    def test_aggregates_periods_to_single_row(self):
        """Multiple periods per round should aggregate to one row."""
        df = create_mock_period_df(
            n_periods=5,
            n_players=2,
            round_payoff=4.0
        )
        result = aggregate_to_round_level(df)

        # Should have 1 row per player (2 players)
        assert len(result) == 2, (
            f"Expected 2 rows (one per player), got {len(result)}"
        )

    def test_preserves_round_payoff(self):
        """round_payoff should be preserved after aggregation."""
        df = create_mock_period_df(n_periods=3, round_payoff=6.0)
        result = aggregate_to_round_level(df)

        assert (result['round_payoff'] == 6.0).all(), (
            "round_payoff should be preserved"
        )

    def test_preserves_identifiers(self):
        """Session, segment, round, player should be preserved."""
        df = create_mock_period_df(
            session_id='my_session',
            segment=3,
            round_num=7,
        )
        result = aggregate_to_round_level(df)

        assert (result['session_id'] == 'my_session').all()
        assert (result['segment'] == 3).all()
        assert (result['round'] == 7).all()


# =====
# Test cases for add_identifiers
# =====
class TestAddIdentifiers:
    """Tests for add_identifiers function."""

    def test_global_round_calculation(self):
        """Verify global_round = (segment-1)*14 + round."""
        test_cases = [
            {'segment': 1, 'round': 1, 'expected': 1},
            {'segment': 1, 'round': 14, 'expected': 14},
            {'segment': 2, 'round': 1, 'expected': 15},
            {'segment': 2, 'round': 14, 'expected': 28},
            {'segment': 3, 'round': 5, 'expected': 33},
            {'segment': 4, 'round': 10, 'expected': 52},
        ]

        for tc in test_cases:
            df = create_holdout_round_df(segment=tc['segment'], round_num=tc['round'])
            result = add_identifiers(df)

            assert result['global_round'].iloc[0] == tc['expected'], (
                f"segment={tc['segment']}, round={tc['round']}: "
                f"expected global_round={tc['expected']}, "
                f"got {result['global_round'].iloc[0]}"
            )

    def test_global_group_id_format(self):
        """Verify format is '{session}_{segment}_{group}'."""
        df = create_holdout_round_df(
            session_id='1_11-7-tr1',
            segment=2,
            group_id=3
        )
        result = add_identifiers(df)

        expected = '1_11-7-tr1_2_3'
        assert result['global_group_id'].iloc[0] == expected, (
            f"Expected '{expected}', got '{result['global_group_id'].iloc[0]}'"
        )


# =====
# Test cases for link_next_round_behavior
# =====
class TestLinkNextRoundBehavior:
    """Tests for link_next_round_behavior function."""

    def test_links_to_correct_next_round(self):
        """Verify round N links to round N+1 in same segment."""
        # Round 5 holdout, Round 6 behavior
        round_configs = [
            {
                'round_num': 5,
                'state': 0,
                'sold_in_round_by_player': {'A': 0},
                'n_players': 1
            },
            {
                'round_num': 6,
                'state': 0,
                'sold_in_round_by_player': {'A': 1},
                'n_players': 1
            },
        ]
        df_full = create_multi_round_df(round_configs)

        # Create holdout from round 5
        df_holdouts = create_holdout_round_df(round_num=5, player='A')

        result = link_next_round_behavior(df_holdouts, df_full)

        assert len(result) == 1, "Should have one linked observation"
        assert result['sold_next_round'].iloc[0] == 1, (
            "Player A sold in round 6, so sold_next_round should be 1"
        )

    def test_excludes_round_14(self):
        """Verify last round of segment is excluded (no next round)."""
        # Round 14 has no next round within segment
        round_configs = [
            {
                'round_num': 14,
                'state': 0,
                'sold_in_round_by_player': {'A': 0},
                'n_players': 1
            },
        ]
        df_full = create_multi_round_df(round_configs)

        df_holdouts = create_holdout_round_df(round_num=14, player='A')

        result = link_next_round_behavior(df_holdouts, df_full)

        assert len(result) == 0, (
            "Round 14 should be excluded (no valid next round)"
        )

    def test_does_not_link_across_segments(self):
        """Verify segment boundary not crossed."""
        # Segment 1, Round 14 should NOT link to Segment 2, Round 1
        round_configs = [
            {
                'segment': 1,
                'round_num': 14,
                'state': 0,
                'sold_in_round_by_player': {'A': 0},
                'n_players': 1
            },
            {
                'segment': 2,
                'round_num': 1,
                'state': 0,
                'sold_in_round_by_player': {'A': 1},
                'n_players': 1
            },
        ]
        df_full = create_multi_round_df(round_configs)

        df_holdouts = create_holdout_round_df(
            segment=1, round_num=14, player='A'
        )

        result = link_next_round_behavior(df_holdouts, df_full)

        # Should NOT link across segments
        assert len(result) == 0, (
            "Should not link segment 1 round 14 to segment 2 round 1"
        )

    def test_sold_next_round_values(self):
        """Verify sold_next_round correctly reflects next round outcome."""
        # Test both sold and not sold cases
        round_configs = [
            # Player A: holdout in R1, sells in R2
            {'round_num': 1, 'sold_in_round_by_player': {'A': 0, 'B': 0}},
            {'round_num': 2, 'sold_in_round_by_player': {'A': 1, 'B': 0}},
            # Player B: holdout in R1, holds in R2
        ]
        df_full = create_multi_round_df(round_configs)

        df_holdouts = pd.concat([
            create_holdout_round_df(round_num=1, player='A'),
            create_holdout_round_df(round_num=1, player='B'),
        ], ignore_index=True)

        result = link_next_round_behavior(df_holdouts, df_full)

        result_a = result[result['player'] == 'A']
        result_b = result[result['player'] == 'B']

        assert result_a['sold_next_round'].iloc[0] == 1, (
            "Player A sold in next round"
        )
        assert result_b['sold_next_round'].iloc[0] == 0, (
            "Player B did not sell in next round"
        )


# =====
# Test cases for compute_prior_sales
# =====
class TestComputePriorSales:
    """Tests for compute_prior_sales function."""

    def test_prior_sales_starts_at_zero(self):
        """First eligible round has prior_sales=0."""
        round_configs = [
            {'round_num': 1, 'sold_in_round_by_player': {'A': 0}},
        ]
        df_full = create_multi_round_df(round_configs)
        df_holdouts = create_holdout_round_df(round_num=1, player='A')

        result = compute_prior_sales(df_holdouts, df_full)

        assert result['prior_sales'].iloc[0] == 0, (
            "First round should have prior_sales=0"
        )

    def test_prior_sales_increments_correctly(self):
        """Count increases with each sale."""
        # Player A: sells in R1, holds in R2, sells in R3, holds in R4
        round_configs = [
            {'round_num': 1, 'sold_in_round_by_player': {'A': 1}, 'n_players': 1},
            {'round_num': 2, 'sold_in_round_by_player': {'A': 0}, 'n_players': 1},
            {'round_num': 3, 'sold_in_round_by_player': {'A': 1}, 'n_players': 1},
            {'round_num': 4, 'sold_in_round_by_player': {'A': 0}, 'n_players': 1},
            {'round_num': 5, 'sold_in_round_by_player': {'A': 0}, 'n_players': 1},
        ]
        df_full = create_multi_round_df(round_configs)

        # Test holdout at rounds 2, 4, 5
        df_holdouts = pd.concat([
            create_holdout_round_df(round_num=2, player='A'),
            create_holdout_round_df(round_num=4, player='A'),
            create_holdout_round_df(round_num=5, player='A'),
        ], ignore_index=True)

        result = compute_prior_sales(df_holdouts, df_full)
        result = result.sort_values('round').reset_index(drop=True)

        # Round 2: sold in R1 -> prior_sales=1
        assert result[result['round'] == 2]['prior_sales'].iloc[0] == 1, (
            "Round 2 should have prior_sales=1 (sold in R1)"
        )
        # Round 4: sold in R1, R3 -> prior_sales=2
        assert result[result['round'] == 4]['prior_sales'].iloc[0] == 2, (
            "Round 4 should have prior_sales=2 (sold in R1, R3)"
        )
        # Round 5: sold in R1, R3 -> prior_sales=2
        assert result[result['round'] == 5]['prior_sales'].iloc[0] == 2, (
            "Round 5 should have prior_sales=2 (sold in R1, R3)"
        )

    def test_prior_sales_counts_across_segments(self):
        """Sales in earlier segments count."""
        # Player A: sells in segment 1 round 14, holdout in segment 2 round 1
        round_configs = [
            {
                'segment': 1,
                'round_num': 14,
                'sold_in_round_by_player': {'A': 1},
                'n_players': 1
            },
            {
                'segment': 2,
                'round_num': 1,
                'sold_in_round_by_player': {'A': 0},
                'n_players': 1
            },
        ]
        df_full = create_multi_round_df(round_configs)

        df_holdouts = create_holdout_round_df(
            segment=2, round_num=1, player='A'
        )

        result = compute_prior_sales(df_holdouts, df_full)

        assert result['prior_sales'].iloc[0] == 1, (
            "Segment 2 round 1 should count sale from segment 1"
        )

    def test_prior_sales_does_not_count_current_round(self):
        """Prior sales should not include current round."""
        # Player sells in R5, we check R5's prior_sales
        round_configs = [
            {'round_num': r, 'sold_in_round_by_player': {'A': 0}, 'n_players': 1}
            for r in range(1, 5)
        ]
        round_configs.append({
            'round_num': 5,
            'sold_in_round_by_player': {'A': 1},
            'n_players': 1
        })
        df_full = create_multi_round_df(round_configs)

        # Even though player sells in R5, prior_sales at R5 should be 0
        df_holdouts = create_holdout_round_df(round_num=5, player='A')

        result = compute_prior_sales(df_holdouts, df_full)

        # This is a special case - player sold in R5 but we're asking about
        # prior_sales AT R5. Should be 0 (no sales BEFORE R5).
        assert result['prior_sales'].iloc[0] == 0, (
            "Prior sales should not include current round"
        )


# =====
# Test cases for build_sales_history
# =====
class TestBuildSalesHistory:
    """Tests for build_sales_history helper function."""

    def test_returns_correct_columns(self):
        """Verify output has expected columns."""
        df = create_mock_period_df()
        result = build_sales_history(df)

        expected_cols = ['session_id', 'segment', 'round', 'player', 'prior_sales']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_excludes_treatment_2(self):
        """Sales history should only use treatment 1 data."""
        round_configs = [
            {'treatment': 'tr1', 'sold_in_round_by_player': {'A': 1}},
            {
                'treatment': 'tr2',
                'sold_in_round_by_player': {'A': 1},
                'session_id': 'tr2_session',
                'round_num': 2
            },
        ]
        df = create_multi_round_df(round_configs)
        result = build_sales_history(df)

        assert 'tr2_session' not in result['session_id'].values, (
            "tr2 sessions should be excluded from sales history"
        )


# =====
# Test cases for build_next_round_lookup
# =====
class TestBuildNextRoundLookup:
    """Tests for build_next_round_lookup helper function."""

    def test_returns_correct_columns(self):
        """Verify output has expected columns."""
        df = create_mock_period_df()
        result = build_next_round_lookup(df)

        expected_cols = [
            'session_id', 'segment', 'round', 'player',
            'sold_next_round', 'signal_next_round'
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_excludes_treatment_2(self):
        """Lookup should only include treatment 1 data."""
        round_configs = [
            {'treatment': 'tr1'},
            {'treatment': 'tr2', 'session_id': 'tr2_session', 'round_num': 2},
        ]
        df = create_multi_round_df(round_configs)
        result = build_next_round_lookup(df)

        assert 'tr2_session' not in result['session_id'].values, (
            "tr2 sessions should be excluded from lookup"
        )


# =====
# Integration tests
# =====
class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_basic(self):
        """Test basic pipeline flow with simple data."""
        # Create dataset with:
        # - Round 1: A sells, B holds (state=0)
        # - Round 2: B sells (state=0)
        round_configs = [
            {
                'round_num': 1,
                'state': 0,
                'sold_in_round_by_player': {'A': 1, 'B': 0},
                'n_players': 2,
                'round_payoff': 2.0
            },
            {
                'round_num': 2,
                'state': 0,
                'sold_in_round_by_player': {'A': 0, 'B': 1},
                'n_players': 2,
                'round_payoff': 4.0
            },
        ]
        df = create_multi_round_df(round_configs)

        # Run pipeline steps
        df_holdouts = filter_to_holdouts(df)
        df_round = aggregate_to_round_level(df_holdouts)
        df_round = add_identifiers(df_round)
        df_round = link_next_round_behavior(df_round, df)
        df_round = compute_prior_sales(df_round, df)

        # Should have 1 observation: B holdout in R1 -> sold in R2
        assert len(df_round) == 1, f"Expected 1 observation, got {len(df_round)}"
        assert df_round['player'].iloc[0] == 'B'
        assert df_round['sold_next_round'].iloc[0] == 1

    def test_mixed_treatments_and_states(self):
        """Verify filtering works with mixed data."""
        round_configs = [
            # tr1, state=0, holdout -> should be included
            {
                'session_id': 'sess1',
                'treatment': 'tr1',
                'state': 0,
                'round_num': 1,
                'sold_in_round_by_player': {'A': 0},
                'n_players': 1
            },
            # Next round for linking
            {
                'session_id': 'sess1',
                'treatment': 'tr1',
                'state': 0,
                'round_num': 2,
                'sold_in_round_by_player': {'A': 0},
                'n_players': 1
            },
            # tr2 -> excluded
            {
                'session_id': 'sess2',
                'treatment': 'tr2',
                'state': 0,
                'round_num': 1,
                'sold_in_round_by_player': {'A': 0},
                'n_players': 1
            },
            # state=1 -> excluded
            {
                'session_id': 'sess3',
                'treatment': 'tr1',
                'state': 1,
                'round_num': 1,
                'sold_in_round_by_player': {'A': 0},
                'n_players': 1
            },
        ]
        df = create_multi_round_df(round_configs)

        df_holdouts = filter_to_holdouts(df)
        df_round = aggregate_to_round_level(df_holdouts)
        df_round = add_identifiers(df_round)
        df_round = link_next_round_behavior(df_round, df)

        # Only sess1 round 1 should remain
        assert len(df_round) == 1, f"Expected 1 observation, got {len(df_round)}"
        assert df_round['session_id'].iloc[0] == 'sess1'


# =====
# Edge cases
# =====
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_after_filtering(self):
        """Handle case where no holdouts exist."""
        # All players sold
        df = create_mock_period_df(
            state=0,
            sold_in_round_by_player={'A': 1, 'B': 1, 'C': 1, 'D': 1}
        )
        result = filter_to_holdouts(df)

        assert len(result) == 0, "Should be empty when no holdouts"

    def test_multiple_sessions(self):
        """Handle multiple sessions correctly."""
        round_configs = [
            {
                'session_id': 'session_1',
                'round_num': 1,
                'sold_in_round_by_player': {'A': 0},
                'n_players': 1
            },
            {
                'session_id': 'session_1',
                'round_num': 2,
                'sold_in_round_by_player': {'A': 1},
                'n_players': 1
            },
            {
                'session_id': 'session_2',
                'round_num': 1,
                'sold_in_round_by_player': {'A': 0},
                'n_players': 1
            },
            {
                'session_id': 'session_2',
                'round_num': 2,
                'sold_in_round_by_player': {'A': 0},
                'n_players': 1
            },
        ]
        df = create_multi_round_df(round_configs)

        df_holdouts = filter_to_holdouts(df)
        df_round = aggregate_to_round_level(df_holdouts)
        df_round = add_identifiers(df_round)
        df_round = link_next_round_behavior(df_round, df)

        # Both session 1 and session 2 should have round 1 linked
        assert len(df_round) == 2, "Should have 2 observations"

        sess1 = df_round[df_round['session_id'] == 'session_1']
        sess2 = df_round[df_round['session_id'] == 'session_2']

        assert sess1['sold_next_round'].iloc[0] == 1, (
            "Session 1 player sold in next round"
        )
        assert sess2['sold_next_round'].iloc[0] == 0, (
            "Session 2 player did not sell in next round"
        )

    def test_player_never_sold(self):
        """Player who never sold should have prior_sales=0."""
        round_configs = [
            {'round_num': r, 'sold_in_round_by_player': {'A': 0}, 'n_players': 1}
            for r in range(1, 6)
        ]
        df = create_multi_round_df(round_configs)

        df_holdouts = pd.concat([
            create_holdout_round_df(round_num=r, player='A')
            for r in range(1, 5)  # Exclude round 5 (no next round data)
        ], ignore_index=True)

        result = compute_prior_sales(df_holdouts, df)

        assert (result['prior_sales'] == 0).all(), (
            "Player who never sold should have prior_sales=0"
        )


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
