"""
Purpose: Unit tests for build_chat_mitigation_dataset.py
Author: Claude Code
Date: 2026-02-02

Tests verify merge logic and variable transformations against realistic
input data structures matching the expected CSV schemas from upstream scripts.
"""

import numpy as np
import pandas as pd
import pytest

from analysis.derived.build_chat_mitigation_dataset import (
    create_id_variables,
    create_chat_segment_indicator,
    create_standardized_emotions,
    create_standardized_traits,
    zscore_column,
    filter_to_imotions_coverage,
    merge_emotions,
    merge_traits,
    merge_chat,
    merge_all_datasets,
    create_analysis_variables,
    SESSION_ID_MAP,
    COLUMN_ORDER,
)


# =====
# Factory functions to create realistic mock DataFrames
# =====
def create_base_df(
    sessions=None,
    n_segments=4,
    n_rounds=14,
    n_periods=3,
    n_players=4
):
    """
    Create mock individual_period_dataset matching raw schema.

    Schema: session_id, segment, round, period, group_id, player, treatment,
            signal, state, price, sold, already_sold, prior_group_sales
    """
    if sessions is None:
        sessions = ['1_11-7-tr1']
    records = []
    treatments = {'1_11-7-tr1': 'tr1', '2_11-10-tr2': 'tr2'}
    players = ['A', 'B', 'C', 'D'][:n_players]

    for session in sessions:
        treatment = treatments.get(session, 'tr1')
        for seg in range(1, n_segments + 1):
            for rnd in range(1, n_rounds + 1):
                for per in range(1, n_periods + 1):
                    for i, player in enumerate(players):
                        records.append({
                            'session_id': session,
                            'segment': seg,
                            'round': rnd,
                            'period': per,
                            'group_id': 1,
                            'player': player,
                            'treatment': treatment,
                            'signal': 0.5,
                            'state': 1,
                            'price': 8 - 2 * (per - 1),
                            'sold': 0,
                            'already_sold': 0,
                            'prior_group_sales': 0,
                        })
    return pd.DataFrame(records)


def create_emotions_df(
    sessions=None,
    n_segments=4,
    n_rounds=10,
    n_periods=3,
    n_players=4
):
    """
    Create mock period_emotions_dataset with session_id already converted to string.

    Note: In the actual script, session_id conversion happens in load_emotions_data().
    For testing merge_emotions directly, we need the session_id already as string.
    """
    if sessions is None:
        sessions = ['1_11-7-tr1']
    records = []
    players = ['A', 'B', 'C', 'D'][:n_players]

    for session in sessions:
        for seg in range(1, n_segments + 1):
            for rnd in range(1, n_rounds + 1):
                for per in range(1, n_periods + 1):
                    for player in players:
                        records.append({
                            'session_id': session,
                            'segment': seg,
                            'round': rnd,
                            'period': per,
                            'player': player,
                            'fear_mean': np.random.uniform(0, 0.3),
                            'fear_max': np.random.uniform(0.3, 0.6),
                            'anger_mean': np.random.uniform(0, 0.2),
                            'anger_max': np.random.uniform(0.2, 0.4),
                            'sadness_mean': np.random.uniform(0, 0.1),
                            'sadness_max': np.random.uniform(0.1, 0.2),
                            'joy_mean': np.random.uniform(0.4, 0.7),
                            'valence_mean': np.random.uniform(-20, 40),
                            'engagement_mean': np.random.uniform(0.6, 0.9),
                            'n_samples': np.random.randint(50, 150),
                        })
    return pd.DataFrame(records)


def create_traits_df(sessions=None, n_players=4):
    """
    Create mock personality_traits_dataset matching raw schema.

    Schema: session_id, player, state_anxiety, extraversion, agreeableness,
            conscientiousness, neuroticism, openness, impulsivity
    """
    if sessions is None:
        sessions = ['1_11-7-tr1']
    records = []
    players = ['A', 'B', 'C', 'D'][:n_players]

    for session in sessions:
        for player in players:
            records.append({
                'session_id': session,
                'player': player,
                'state_anxiety': np.random.uniform(1, 4),
                'extraversion': np.random.uniform(1, 7),
                'agreeableness': np.random.uniform(1, 7),
                'conscientiousness': np.random.uniform(1, 7),
                'neuroticism': np.random.uniform(1, 7),
                'openness': np.random.uniform(1, 7),
                'impulsivity': np.random.uniform(1, 7),
            })
    return pd.DataFrame(records)


def create_chat_df(
    sessions=None,
    n_segments=4,
    n_rounds=14,
    n_periods=3,
    n_players=4
):
    """
    Create mock chat_activity_dataset matching raw schema.

    Schema: session_id, segment, round, period, player, group_id,
            messages_sent_segment, messages_received_segment, total_group_messages
    """
    if sessions is None:
        sessions = ['1_11-7-tr1']
    records = []
    players = ['A', 'B', 'C', 'D'][:n_players]

    for session in sessions:
        for seg in range(1, n_segments + 1):
            is_chat_seg = seg >= 3
            for rnd in range(1, n_rounds + 1):
                for per in range(1, n_periods + 1):
                    for player in players:
                        records.append({
                            'session_id': session,
                            'segment': seg,
                            'round': rnd,
                            'period': per,
                            'player': player,
                            'group_id': 1,
                            'messages_sent_segment': 5 if is_chat_seg else 0,
                            'messages_received_segment': 15 if is_chat_seg else 0,
                            'total_group_messages': 20 if is_chat_seg else 0,
                        })
    return pd.DataFrame(records)


# =====
# Tests for SESSION_ID_MAP
# =====
class TestSessionIdMapping:
    """Tests for session ID integer to string mapping."""

    def test_maps_all_six_sessions(self):
        """All 6 sessions should be in the mapping."""
        assert len(SESSION_ID_MAP) == 6
        for i in range(1, 7):
            assert i in SESSION_ID_MAP

    def test_mapping_format_correct(self):
        """Session ID strings should match expected format."""
        expected = {
            1: "1_11-7-tr1",
            2: "2_11-10-tr2",
            3: "3_11-11-tr2",
            4: "4_11-12-tr1",
            5: "5_11-14-tr2",
            6: "6_11-18-tr1",
        }
        assert SESSION_ID_MAP == expected


# =====
# Tests for ID variable creation
# =====
class TestCreateIdVariables:
    """Tests for player_id and global_group_id creation."""

    def test_player_id_format(self):
        """player_id should be session_id_player."""
        df = pd.DataFrame({
            'session_id': ['1_11-7-tr1', '2_11-10-tr2'],
            'player': ['A', 'B'],
            'segment': [1, 2],
            'group_id': [1, 2],
        })
        result = create_id_variables(df)
        assert result.loc[0, 'player_id'] == '1_11-7-tr1_A'
        assert result.loc[1, 'player_id'] == '2_11-10-tr2_B'

    def test_global_group_id_format(self):
        """global_group_id should be session_id_segment_group_id."""
        df = pd.DataFrame({
            'session_id': ['1_11-7-tr1'],
            'player': ['A'],
            'segment': [3],
            'group_id': [2],
        })
        result = create_id_variables(df)
        assert result.loc[0, 'global_group_id'] == '1_11-7-tr1_3_2'

    def test_unique_across_sessions(self):
        """IDs should be unique across different sessions."""
        df = pd.DataFrame({
            'session_id': ['1_11-7-tr1', '2_11-10-tr2'],
            'player': ['A', 'A'],  # Same player label
            'segment': [1, 1],
            'group_id': [1, 1],
        })
        result = create_id_variables(df)
        assert result['player_id'].nunique() == 2


# =====
# Tests for chat_segment indicator
# =====
class TestCreateChatSegmentIndicator:
    """Tests for chat_segment binary indicator creation."""

    def test_segments_1_2_are_zero(self):
        """Segments 1 and 2 should have chat_segment = 0."""
        df = pd.DataFrame({'segment': [1, 2, 1, 2]})
        result = create_chat_segment_indicator(df)
        assert all(result.loc[result['segment'].isin([1, 2]), 'chat_segment'] == 0)

    def test_segments_3_4_are_one(self):
        """Segments 3 and 4 should have chat_segment = 1."""
        df = pd.DataFrame({'segment': [3, 4, 3, 4]})
        result = create_chat_segment_indicator(df)
        assert all(result.loc[result['segment'].isin([3, 4]), 'chat_segment'] == 1)

    def test_chat_segment_is_integer(self):
        """chat_segment should be integer type, not boolean."""
        df = pd.DataFrame({'segment': [1, 3]})
        result = create_chat_segment_indicator(df)
        assert result['chat_segment'].dtype in [np.int64, np.int32, int]


# =====
# Tests for z-score standardization
# =====
class TestZscoreColumn:
    """Tests for z-score standardization function."""

    def test_zscore_basic(self):
        """Z-scored data should have mean ~0 and std ~1."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = zscore_column(df, 'x')
        assert abs(result.mean()) < 0.001
        assert abs(result.std() - 1.0) < 0.001

    def test_zscore_with_nan(self):
        """NaN values should remain NaN after z-scoring."""
        df = pd.DataFrame({'x': [1.0, 2.0, np.nan, 4.0, 5.0]})
        result = zscore_column(df, 'x')
        assert pd.isna(result.iloc[2])
        assert abs(result.dropna().mean()) < 0.01

    def test_zscore_zero_variance(self):
        """Zero variance should return all NaN."""
        df = pd.DataFrame({'x': [5.0, 5.0, 5.0]})
        result = zscore_column(df, 'x')
        assert all(pd.isna(result))


class TestCreateStandardizedEmotions:
    """Tests for standardized emotion variable creation."""

    def test_creates_all_z_variables(self):
        """Should create fear_z, anger_z, sadness_z."""
        df = pd.DataFrame({
            'fear_mean': [0.1, 0.2, 0.3, 0.4, 0.5],
            'anger_mean': [0.05, 0.1, 0.15, 0.2, 0.25],
            'sadness_mean': [0.02, 0.04, 0.06, 0.08, 0.1],
        })
        result = create_standardized_emotions(df)
        assert 'fear_z' in result.columns
        assert 'anger_z' in result.columns
        assert 'sadness_z' in result.columns

    def test_z_scores_have_correct_properties(self):
        """Z-scored columns should have mean ~0, std ~1."""
        df = pd.DataFrame({
            'fear_mean': np.random.uniform(0, 0.5, 100),
            'anger_mean': np.random.uniform(0, 0.3, 100),
            'sadness_mean': np.random.uniform(0, 0.2, 100),
        })
        result = create_standardized_emotions(df)
        for col in ['fear_z', 'anger_z', 'sadness_z']:
            assert abs(result[col].mean()) < 0.01
            assert abs(result[col].std() - 1.0) < 0.01


class TestCreateStandardizedTraits:
    """Tests for standardized trait variable creation."""

    def test_creates_all_z_variables(self):
        """Should create neuroticism_z and impulsivity_z."""
        df = pd.DataFrame({
            'neuroticism': [3.0, 4.0, 5.0, 3.5, 4.5],
            'impulsivity': [3.5, 4.5, 5.5, 4.0, 5.0],
        })
        result = create_standardized_traits(df)
        assert 'neuroticism_z' in result.columns
        assert 'impulsivity_z' in result.columns


# =====
# Tests for round filtering
# =====
class TestFilterToImotionsCoverage:
    """Tests for filtering to rounds 1-10."""

    def test_excludes_rounds_above_10(self):
        """Rounds 11-14 should be excluded."""
        df = pd.DataFrame({'round': [1, 5, 10, 11, 14]})
        result = filter_to_imotions_coverage(df)
        assert len(result) == 3
        assert result['round'].max() == 10

    def test_includes_all_rounds_1_10(self):
        """All rounds 1-10 should be preserved."""
        df = pd.DataFrame({'round': list(range(1, 15))})
        result = filter_to_imotions_coverage(df)
        for r in range(1, 11):
            assert r in result['round'].values


# =====
# Tests for merge functions
# =====
class TestMergeEmotions:
    """Tests for emotions merge logic."""

    def test_merge_on_correct_keys(self):
        """Should merge on session_id, segment, round, period, player."""
        base_df = create_base_df(n_segments=2, n_rounds=5, n_periods=2, n_players=2)
        emotions_df = create_emotions_df(n_segments=2, n_rounds=5, n_periods=2, n_players=2)
        result = merge_emotions(base_df, emotions_df)
        assert 'fear_mean' in result.columns
        assert len(result) == len(base_df)

    def test_unmatched_rows_have_nan(self):
        """Rows without matching emotions should have NaN."""
        base_df = pd.DataFrame({
            'session_id': ['1_11-7-tr1', '1_11-7-tr1'],
            'segment': [1, 1],
            'round': [1, 11],  # Round 11 won't have emotions
            'period': [1, 1],
            'player': ['A', 'A'],
        })
        emotions_df = pd.DataFrame({
            'session_id': ['1_11-7-tr1'],
            'segment': [1],
            'round': [1],
            'period': [1],
            'player': ['A'],
            'fear_mean': [0.1],
            'fear_max': [0.2],
            'anger_mean': [0.05],
            'anger_max': [0.1],
            'sadness_mean': [0.02],
            'sadness_max': [0.04],
            'joy_mean': [0.5],
            'valence_mean': [10.0],
            'engagement_mean': [0.8],
            'n_samples': [100],
        })
        result = merge_emotions(base_df, emotions_df)
        assert result.loc[0, 'fear_mean'] == pytest.approx(0.1)
        assert pd.isna(result.loc[1, 'fear_mean'])


class TestMergeTraits:
    """Tests for personality traits merge logic."""

    def test_merge_on_player_level_keys(self):
        """Should merge on session_id and player only (time-invariant)."""
        base_df = create_base_df(n_segments=2, n_rounds=2, n_periods=2, n_players=2)
        traits_df = create_traits_df(n_players=2)
        result = merge_traits(base_df, traits_df)
        assert 'neuroticism' in result.columns
        assert len(result) == len(base_df)

    def test_traits_replicated_across_periods(self):
        """Same trait values should appear for all periods of same player."""
        base_df = pd.DataFrame({
            'session_id': ['1_11-7-tr1'] * 4,
            'segment': [1, 1, 2, 2],
            'round': [1, 2, 1, 2],
            'period': [1, 1, 1, 1],
            'player': ['A', 'A', 'A', 'A'],
        })
        traits_df = pd.DataFrame({
            'session_id': ['1_11-7-tr1'],
            'player': ['A'],
            'state_anxiety': [2.5],
            'extraversion': [4.0],
            'agreeableness': [5.0],
            'conscientiousness': [4.5],
            'neuroticism': [3.5],
            'openness': [5.5],
            'impulsivity': [4.0],
        })
        result = merge_traits(base_df, traits_df)
        assert all(result['neuroticism'] == 3.5)

    def test_missing_player_traits_are_nan(self):
        """Players not in traits dataset should have NaN."""
        base_df = pd.DataFrame({
            'session_id': ['1_11-7-tr1'],
            'player': ['Z'],
        })
        traits_df = pd.DataFrame({
            'session_id': ['1_11-7-tr1'],
            'player': ['A'],
            'state_anxiety': [2.5],
            'extraversion': [4.0],
            'agreeableness': [5.0],
            'conscientiousness': [4.5],
            'neuroticism': [3.5],
            'openness': [5.5],
            'impulsivity': [4.0],
        })
        result = merge_traits(base_df, traits_df)
        assert pd.isna(result['neuroticism'].iloc[0])


class TestMergeChat:
    """Tests for chat activity merge logic."""

    def test_merge_on_correct_keys(self):
        """Should merge on session_id, segment, round, period, player, group_id."""
        base_df = create_base_df(n_segments=2, n_rounds=5, n_periods=2, n_players=2)
        chat_df = create_chat_df(n_segments=2, n_rounds=5, n_periods=2, n_players=2)
        result = merge_chat(base_df, chat_df)
        assert 'messages_sent_segment' in result.columns
        assert len(result) == len(base_df)


# =====
# Tests for full merge pipeline
# =====
class TestMergeAllDatasets:
    """Tests for complete merge pipeline."""

    def test_all_columns_present_after_merge(self):
        """Final merged dataset should have all expected columns."""
        np.random.seed(42)
        base_df = create_base_df(n_segments=2, n_rounds=5, n_periods=2, n_players=2)
        emotions_df = create_emotions_df(n_segments=2, n_rounds=5, n_periods=2, n_players=2)
        traits_df = create_traits_df(n_players=2)
        chat_df = create_chat_df(n_segments=2, n_rounds=5, n_periods=2, n_players=2)

        result = merge_all_datasets(base_df, emotions_df, traits_df, chat_df)

        expected_cols = [
            'session_id', 'segment', 'round', 'period', 'player', 'group_id',
            'treatment', 'signal', 'state', 'price', 'sold', 'already_sold',
            'prior_group_sales', 'fear_mean', 'neuroticism', 'messages_sent_segment'
        ]
        for col in expected_cols:
            assert col in result.columns


class TestCreateAnalysisVariables:
    """Tests for analysis variable creation pipeline."""

    def test_creates_all_derived_variables(self):
        """Should create player_id, global_group_id, chat_segment, z-scores."""
        df = pd.DataFrame({
            'session_id': ['1_11-7-tr1'] * 4,
            'player': ['A', 'A', 'B', 'B'],
            'segment': [1, 3, 1, 3],
            'group_id': [1, 1, 1, 1],
            'fear_mean': [0.1, 0.2, 0.3, 0.4],
            'anger_mean': [0.1, 0.2, 0.3, 0.4],
            'sadness_mean': [0.1, 0.2, 0.3, 0.4],
            'neuroticism': [3.0, 3.0, 4.0, 4.0],
            'impulsivity': [3.5, 3.5, 4.5, 4.5],
        })
        result = create_analysis_variables(df)

        assert 'player_id' in result.columns
        assert 'global_group_id' in result.columns
        assert 'chat_segment' in result.columns
        assert 'fear_z' in result.columns
        assert 'neuroticism_z' in result.columns


# =====
# Tests for output column order
# =====
class TestColumnOrder:
    """Tests for COLUMN_ORDER specification."""

    def test_contains_all_required_columns(self):
        """COLUMN_ORDER should include all documented output columns."""
        required = [
            # Identifiers
            'session_id', 'segment', 'round', 'period', 'player', 'group_id',
            'player_id', 'global_group_id', 'treatment',
            # Outcome
            'sold', 'already_sold',
            # Treatment
            'chat_segment',
            # Emotions
            'fear_mean', 'fear_max', 'anger_mean', 'anger_max',
            'sadness_mean', 'sadness_max', 'joy_mean', 'valence_mean',
            'engagement_mean', 'n_samples', 'fear_z', 'anger_z', 'sadness_z',
            # Chat
            'messages_sent_segment', 'messages_received_segment', 'total_group_messages',
            # Traits
            'state_anxiety', 'extraversion', 'agreeableness', 'conscientiousness',
            'neuroticism', 'openness', 'impulsivity', 'neuroticism_z', 'impulsivity_z',
            # Controls
            'signal', 'state', 'price', 'prior_group_sales'
        ]
        for col in required:
            assert col in COLUMN_ORDER, f"Missing column: {col}"

    def test_no_duplicate_columns(self):
        """COLUMN_ORDER should not have duplicates."""
        assert len(COLUMN_ORDER) == len(set(COLUMN_ORDER))


# =====
# Integration tests
# =====
class TestEndToEndIntegration:
    """Integration tests for complete pipeline."""

    def test_full_pipeline_produces_valid_output(self):
        """Full pipeline should produce dataset with correct structure."""
        np.random.seed(42)
        base_df = create_base_df(n_segments=4, n_rounds=10, n_periods=2, n_players=4)
        emotions_df = create_emotions_df(n_segments=4, n_rounds=10, n_periods=2, n_players=4)
        traits_df = create_traits_df(n_players=4)
        chat_df = create_chat_df(n_segments=4, n_rounds=10, n_periods=2, n_players=4)

        merged = merge_all_datasets(base_df, emotions_df, traits_df, chat_df)
        final = create_analysis_variables(merged)
        final = filter_to_imotions_coverage(final)

        # Check we have expected number of observations
        # 1 session * 4 segments * 10 rounds * 2 periods * 4 players = 320
        assert len(final) == 320

    def test_chat_segment_correctly_assigned(self):
        """chat_segment should match segment 1-2 vs 3-4 split."""
        np.random.seed(42)
        base_df = create_base_df(n_segments=4, n_rounds=2, n_periods=1, n_players=1)
        emotions_df = create_emotions_df(n_segments=4, n_rounds=2, n_periods=1, n_players=1)
        traits_df = create_traits_df(n_players=1)
        chat_df = create_chat_df(n_segments=4, n_rounds=2, n_periods=1, n_players=1)

        merged = merge_all_datasets(base_df, emotions_df, traits_df, chat_df)
        final = create_analysis_variables(merged)

        seg_12 = final[final['segment'].isin([1, 2])]
        seg_34 = final[final['segment'].isin([3, 4])]

        assert all(seg_12['chat_segment'] == 0)
        assert all(seg_34['chat_segment'] == 1)

    def test_z_scores_computed_correctly(self):
        """Z-scores should have mean ~0 and std ~1 in final dataset."""
        np.random.seed(42)
        base_df = create_base_df(n_segments=4, n_rounds=10, n_periods=2, n_players=4)
        emotions_df = create_emotions_df(n_segments=4, n_rounds=10, n_periods=2, n_players=4)
        traits_df = create_traits_df(n_players=4)
        chat_df = create_chat_df(n_segments=4, n_rounds=10, n_periods=2, n_players=4)

        merged = merge_all_datasets(base_df, emotions_df, traits_df, chat_df)
        final = create_analysis_variables(merged)

        for col in ['fear_z', 'anger_z', 'sadness_z', 'neuroticism_z', 'impulsivity_z']:
            assert abs(final[col].mean()) < 0.01


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
