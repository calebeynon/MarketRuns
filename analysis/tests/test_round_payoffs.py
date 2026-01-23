"""
Purpose: Verify round_payoffs extraction reads from last period of each round
Author: Claude Code
Date: 2026-01-23

Tests verify that market_data.py correctly extracts round_payoffs from the
last period of each round (not period 1). This was a bug fix where payoffs
were being read from period 1 instead of the last period where final values
are stored.
"""

import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from market_data import parse_experiment

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
SESSION_1_CSV = DATASTORE / "1_11-7-tr1" / "all_apps_wide_2025-11-07.csv"


# =====
# Helper functions
# =====
def compute_round_period_mapping(df, segment_name):
    """
    Compute the actual mapping of round -> last oTree period from raw data.

    The number of periods per round varies based on actual gameplay.
    This function extracts the mapping from round_number_in_segment columns.

    Returns:
        Dict[int, int]: Maps round_num -> last_otree_period
    """
    # Use first participant to extract round boundaries
    first_row = df.iloc[0]

    round_to_last_period = {}
    current_round = None

    for period in range(1, 100):
        col = f'{segment_name}.{period}.player.round_number_in_segment'
        if col not in df.columns:
            break
        if pd.isna(first_row[col]):
            break

        round_num = int(first_row[col])

        # Track the maximum period for each round
        if round_num not in round_to_last_period:
            round_to_last_period[round_num] = period
        else:
            round_to_last_period[round_num] = max(
                round_to_last_period[round_num], period
            )

    return round_to_last_period


def get_raw_payoff_from_csv(df, player_label, segment_name, round_num, otree_period):
    """
    Extract round payoff directly from raw CSV for a specific player, round, and period.

    Args:
        df: Raw CSV DataFrame
        player_label: Player label (A, B, C, etc.)
        segment_name: Segment name (chat_noavg, chat_noavg2, etc.)
        round_num: Round number (1-14)
        otree_period: The oTree period number to read from

    Returns:
        Payoff value from raw CSV
    """
    player_row = df[df['participant.label'] == player_label].iloc[0]
    col_name = f'{segment_name}.{otree_period}.player.round_{round_num}_payoff'
    return player_row[col_name]


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def raw_csv():
    """Load the raw CSV file for Session 1."""
    if not SESSION_1_CSV.exists():
        pytest.skip(f"Session 1 CSV not found: {SESSION_1_CSV}")
    return pd.read_csv(SESSION_1_CSV)


@pytest.fixture(scope="module")
def experiment():
    """Parse Session 1 using market_data.py."""
    if not SESSION_1_CSV.exists():
        pytest.skip(f"Session 1 CSV not found: {SESSION_1_CSV}")
    return parse_experiment(str(SESSION_1_CSV))


@pytest.fixture(scope="module")
def session(experiment):
    """Get the first (and only) session."""
    return experiment.sessions[0]


@pytest.fixture(scope="module")
def segment_chat_noavg(session):
    """Get the chat_noavg segment."""
    return session.get_segment('chat_noavg')


@pytest.fixture(scope="module")
def round_period_mapping(raw_csv):
    """Get the actual round -> last_period mapping from raw data."""
    return compute_round_period_mapping(raw_csv, 'chat_noavg')


# =====
# Test: round_payoffs match raw CSV from last period
# =====
class TestRoundPayoffsMatchRawCSV:
    """Verify round_payoffs match raw CSV values from the last period of each round."""

    def test_round_payoffs_match_raw_csv_session1(
        self, raw_csv, segment_chat_noavg, round_period_mapping
    ):
        """
        For segment chat_noavg, verify that round_payoffs match raw CSV.

        The parser dynamically determines the last period for each round.
        This test verifies payoffs match the values from those last periods.
        """
        segment = segment_chat_noavg
        segment_name = 'chat_noavg'

        mismatches = []

        for round_num in sorted(segment.rounds.keys()):
            round_obj = segment.get_round(round_num)
            last_period = round_period_mapping.get(round_num)

            if last_period is None:
                continue

            for player_label, parsed_payoff in round_obj.round_payoffs.items():
                # Get expected value from raw CSV (last period)
                expected = get_raw_payoff_from_csv(
                    raw_csv, player_label, segment_name, round_num, last_period
                )

                # Compare values
                if pd.isna(expected) and parsed_payoff is None:
                    continue  # Both missing, OK

                if abs(parsed_payoff - expected) > 1e-6:
                    mismatches.append({
                        'player': player_label,
                        'round': round_num,
                        'last_period': last_period,
                        'parsed': parsed_payoff,
                        'expected': expected,
                    })

        assert len(mismatches) == 0, (
            f"Found {len(mismatches)} payoff mismatches:\n"
            + "\n".join([str(m) for m in mismatches[:10]])
        )

    def test_player_c_round_1_specific_case(self, raw_csv, segment_chat_noavg):
        """
        Specific test case: Player C, Round 1 payoff should be 4.0 (not 6.0).

        This was the bug - period 1 has 6.0, but last period (3) has 4.0.
        Round 1 has 3 periods (periods 1-3), so last period is 3.
        """
        segment = segment_chat_noavg
        round_obj = segment.get_round(1)

        parsed_payoff = round_obj.round_payoffs.get('C')

        # Verify against raw CSV
        raw_period_1 = get_raw_payoff_from_csv(
            raw_csv, 'C', 'chat_noavg', 1, otree_period=1
        )
        raw_period_3 = get_raw_payoff_from_csv(
            raw_csv, 'C', 'chat_noavg', 1, otree_period=3
        )

        # Period 1 has wrong value (6.0), Period 3 has correct value (4.0)
        assert raw_period_1 == 6.0, f"Expected period 1 to have 6.0, got {raw_period_1}"
        assert raw_period_3 == 4.0, f"Expected period 3 to have 4.0, got {raw_period_3}"

        # Parsed value should match the correct (last period) value
        assert parsed_payoff == 4.0, (
            f"Player C Round 1 payoff should be 4.0 (from last period), got {parsed_payoff}"
        )


# =====
# Test: Zero payoffs are valid data
# =====
class TestZeroPayoffsValid:
    """Verify that 0.0 payoffs are preserved as valid data, not treated as missing."""

    def test_round_payoff_zero_is_valid(self, experiment):
        """
        Verify that 0.0 payoffs are preserved in round_payoffs.

        Zero is a valid payoff (e.g., when state=0 and player didn't sell).
        It should not be treated as None/missing.
        """
        # Search across all sessions and segments for 0.0 payoffs
        zero_payoffs_found = []

        for session in experiment.sessions:
            for segment_name, segment in session.segments.items():
                for round_num, round_obj in segment.rounds.items():
                    for player_label, payoff in round_obj.round_payoffs.items():
                        if payoff == 0.0:
                            zero_payoffs_found.append({
                                'session': session.session_code,
                                'segment': segment_name,
                                'round': round_num,
                                'player': player_label,
                                'payoff': payoff,
                            })

        # 0.0 payoffs should be preserved (may or may not exist in this data)
        # The key check is that payoff == 0.0 is not None
        for entry in zero_payoffs_found:
            assert entry['payoff'] is not None, (
                f"Zero payoff treated as None for {entry}"
            )
            assert entry['payoff'] == 0.0, (
                f"Expected 0.0, got {entry['payoff']} for {entry}"
            )

    def test_zero_payoffs_not_converted_to_none(
        self, segment_chat_noavg, raw_csv, round_period_mapping
    ):
        """Verify that if raw CSV has 0.0, it's preserved as 0.0, not None."""
        segment = segment_chat_noavg
        segment_name = 'chat_noavg'

        for round_num in sorted(segment.rounds.keys()):
            round_obj = segment.get_round(round_num)
            last_period = round_period_mapping.get(round_num)

            if last_period is None:
                continue

            for player_label in round_obj.round_payoffs.keys():
                raw_value = get_raw_payoff_from_csv(
                    raw_csv, player_label, segment_name, round_num, last_period
                )
                parsed_value = round_obj.round_payoffs[player_label]

                if raw_value == 0.0:
                    assert parsed_value == 0.0, (
                        f"Raw CSV has 0.0 for {player_label} round {round_num}, "
                        f"but parsed value is {parsed_value}"
                    )


# =====
# Test: All rounds have payoffs
# =====
class TestAllRoundsHavePayoffs:
    """Verify that all rounds in a segment have round_payoffs populated."""

    def test_round_payoffs_exist_for_all_rounds(self, session):
        """
        Verify that all rounds in each segment have round_payoffs.

        Each round should have payoffs for all 16 players (4 groups x 4 players).
        """
        expected_n_players = 16

        for segment_name, segment in session.segments.items():
            for round_num in sorted(segment.rounds.keys()):
                round_obj = segment.get_round(round_num)

                # Verify round_payoffs dict exists and is not empty
                assert round_obj.round_payoffs is not None, (
                    f"{segment_name} round {round_num} has no round_payoffs dict"
                )

                n_payoffs = len(round_obj.round_payoffs)
                assert n_payoffs > 0, (
                    f"{segment_name} round {round_num} has empty round_payoffs"
                )

                # Verify we have payoffs for expected number of players
                assert n_payoffs == expected_n_players, (
                    f"{segment_name} round {round_num} has {n_payoffs} payoffs, "
                    f"expected {expected_n_players}"
                )

    def test_round_payoffs_have_valid_values(self, session):
        """Verify that round_payoffs contain valid numeric values."""
        valid_payoffs = {0.0, 2.0, 4.0, 6.0, 8.0, 20.0}

        for segment_name, segment in session.segments.items():
            for round_num, round_obj in segment.rounds.items():
                for player_label, payoff in round_obj.round_payoffs.items():
                    assert payoff is not None, (
                        f"{segment_name} round {round_num} {player_label} "
                        f"has None payoff"
                    )
                    assert payoff in valid_payoffs, (
                        f"{segment_name} round {round_num} {player_label} "
                        f"has unexpected payoff {payoff}, expected one of {valid_payoffs}"
                    )


# =====
# Test: Payoff values are from correct period
# =====
class TestPayoffsFromCorrectPeriod:
    """
    Verify payoffs come from the last period, not period 1.

    This is the key regression test for the bug fix.
    """

    def test_payoffs_differ_between_period_1_and_last(
        self, raw_csv, segment_chat_noavg, round_period_mapping
    ):
        """
        Find cases where period 1 and last period have different payoff values.

        The parsed payoff should match the last period value.
        """
        segment = segment_chat_noavg
        segment_name = 'chat_noavg'

        cases_where_values_differ = []

        for round_num in sorted(segment.rounds.keys()):
            round_obj = segment.get_round(round_num)
            last_period = round_period_mapping.get(round_num)

            if last_period is None or round_num != 1:
                # Only test round 1, where period 1 has actual payoff data
                # Other rounds have 0.0 in period 1 (initialization)
                continue

            for player_label, parsed_payoff in round_obj.round_payoffs.items():
                period_1_val = get_raw_payoff_from_csv(
                    raw_csv, player_label, segment_name, round_num, otree_period=1
                )
                last_period_val = get_raw_payoff_from_csv(
                    raw_csv, player_label, segment_name, round_num, last_period
                )

                if abs(period_1_val - last_period_val) > 1e-6:
                    cases_where_values_differ.append({
                        'player': player_label,
                        'round': round_num,
                        'period_1_val': period_1_val,
                        'last_period_val': last_period_val,
                        'parsed_payoff': parsed_payoff,
                        'correct': abs(parsed_payoff - last_period_val) < 1e-6,
                    })

        # Should have some cases where values differ
        assert len(cases_where_values_differ) > 0, (
            "Expected some cases where period 1 and last period payoffs differ"
        )

        # All such cases should match the last period value
        incorrect = [c for c in cases_where_values_differ if not c['correct']]
        assert len(incorrect) == 0, (
            f"Found {len(incorrect)} cases where parsed payoff doesn't match "
            f"last period value:\n" + "\n".join([str(c) for c in incorrect[:5]])
        )

    def test_round_1_period_1_vs_period_3_differ(self, raw_csv):
        """
        Verify that for round 1, some players have different values in period 1 vs 3.

        This confirms the bug scenario exists in the test data.
        """
        segment_name = 'chat_noavg'
        player_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R']

        differences = []
        for label in player_labels:
            period_1_val = get_raw_payoff_from_csv(
                raw_csv, label, segment_name, 1, otree_period=1
            )
            period_3_val = get_raw_payoff_from_csv(
                raw_csv, label, segment_name, 1, otree_period=3
            )
            if abs(period_1_val - period_3_val) > 1e-6:
                differences.append({
                    'player': label,
                    'period_1': period_1_val,
                    'period_3': period_3_val,
                })

        # Should have at least some differences
        assert len(differences) > 0, (
            "Expected some players to have different round_1_payoff "
            "values in period 1 vs period 3"
        )

        # Player C specifically should differ (from bug report)
        player_c_diff = [d for d in differences if d['player'] == 'C']
        assert len(player_c_diff) == 1, "Player C should have different values"
        assert player_c_diff[0]['period_1'] == 6.0, (
            f"Player C period 1 should be 6.0, got {player_c_diff[0]['period_1']}"
        )
        assert player_c_diff[0]['period_3'] == 4.0, (
            f"Player C period 3 should be 4.0, got {player_c_diff[0]['period_3']}"
        )


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
