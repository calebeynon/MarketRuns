"""
Purpose: Unit tests for build_chat_activity_dataset.py
Author: Claude Code
Date: 2026-02-02
"""

import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from analysis.derived.build_chat_activity_dataset import (
    compute_segment_chat_counts,
    init_zero_chat_counts,
    sum_group_chat,
    build_segment_records,
    build_player_record,
    get_player_chat_counts,
    create_record_dict,
)


# =====
# Mock classes to simulate market_data structures
# =====
@dataclass
class MockChatMessage:
    """Mock ChatMessage for testing."""
    nickname: str
    body: str = "test message"


@dataclass
class MockPlayerPeriodData:
    """Mock PlayerPeriodData for testing."""
    label: str


@dataclass
class MockPeriod:
    """Mock Period for testing."""
    period_in_round: int
    players: Dict[str, MockPlayerPeriodData] = field(default_factory=dict)


@dataclass
class MockRound:
    """Mock Round for testing."""
    round_number_in_segment: int
    periods: Dict[int, MockPeriod] = field(default_factory=dict)
    chat_messages: List[MockChatMessage] = field(default_factory=list)


@dataclass
class MockGroup:
    """Mock Group for testing."""
    group_id: int
    player_labels: List[str] = field(default_factory=list)


@dataclass
class MockSegment:
    """Mock Segment for testing."""
    name: str
    rounds: Dict[int, MockRound] = field(default_factory=dict)
    groups: Dict[int, MockGroup] = field(default_factory=dict)

    def get_group_by_player(self, label: str) -> Optional[MockGroup]:
        """Find which group a player belongs to."""
        for group in self.groups.values():
            if label in group.player_labels:
                return group
        return None


# =====
# Mock segment factory functions
# =====
def create_mock_segment(n_players=4, n_rounds=2, n_periods=2, chat_msgs=None):
    """Create a mock segment with configurable structure."""
    labels = ['A', 'B', 'C', 'D'][:n_players]
    groups = {1: MockGroup(group_id=1, player_labels=labels)}
    rounds = build_mock_rounds(labels, n_rounds, n_periods, chat_msgs)
    return MockSegment(name="test_segment", rounds=rounds, groups=groups)


def build_mock_rounds(labels, n_rounds, n_periods, chat_msgs):
    """Build mock rounds with periods and optional chat."""
    rounds = {}
    for r in range(1, n_rounds + 1):
        periods = build_mock_periods(labels, n_periods)
        msgs = build_chat_messages(chat_msgs) if r == 1 else []
        rounds[r] = MockRound(round_number_in_segment=r, periods=periods, chat_messages=msgs)
    return rounds


def build_mock_periods(labels, n_periods):
    """Build mock periods with players."""
    periods = {}
    for p in range(1, n_periods + 1):
        players = {lbl: MockPlayerPeriodData(label=lbl) for lbl in labels}
        periods[p] = MockPeriod(period_in_round=p, players=players)
    return periods


def build_chat_messages(chat_msgs):
    """Build chat message list from dict of {player: count}."""
    if not chat_msgs:
        return []
    msgs = []
    for label, count in chat_msgs.items():
        msgs.extend([MockChatMessage(nickname=label) for _ in range(count)])
    return msgs


# =====
# Tests for chat counting functions
# =====
class TestComputeSegmentChatCounts:
    """Tests for compute_segment_chat_counts function."""

    def test_no_chat_segment_returns_zeros(self):
        """Segments 1-2 should return zero chat counts."""
        segment = create_mock_segment(n_players=4)
        for seg_num in [1, 2]:
            result = compute_segment_chat_counts(segment, seg_num)
            assert result[1]['total_messages'] == 0
            assert all(result[1]['messages_by_player'][l] == 0 for l in 'ABCD')

    def test_chat_segment_counts_messages(self):
        """Segments 3-4 should count chat messages."""
        segment = create_mock_segment(n_players=4, chat_msgs={'A': 3, 'B': 2})
        for seg_num in [3, 4]:
            result = compute_segment_chat_counts(segment, seg_num)
            assert result[1]['total_messages'] == 5
            assert result[1]['messages_by_player']['A'] == 3
            assert result[1]['messages_by_player']['B'] == 2


class TestInitZeroChatCounts:
    """Tests for init_zero_chat_counts function."""

    def test_creates_zero_counts_for_all_players(self):
        """Should create zero counts for each player in each group."""
        segment = create_mock_segment(n_players=4)
        result = init_zero_chat_counts(segment)
        assert result[1]['total_messages'] == 0
        assert len(result[1]['messages_by_player']) == 4


class TestSumGroupChat:
    """Tests for sum_group_chat function."""

    def test_counts_messages_for_group_players(self):
        """Should count messages only from players in the group."""
        segment = create_mock_segment(n_players=4, chat_msgs={'A': 2, 'B': 1, 'C': 3})
        msgs_by_player = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        total = sum_group_chat(segment, msgs_by_player)
        assert total == 6
        assert msgs_by_player == {'A': 2, 'B': 1, 'C': 3, 'D': 0}

    def test_ignores_messages_from_unknown_players(self):
        """Messages from players not in dict should be ignored."""
        segment = create_mock_segment(n_players=4, chat_msgs={'A': 2, 'E': 5})
        msgs_by_player = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        total = sum_group_chat(segment, msgs_by_player)
        assert total == 2


# =====
# Tests for record building functions
# =====
class TestBuildSegmentRecords:
    """Tests for build_segment_records function."""

    def test_creates_correct_number_of_records(self):
        """Should create one record per player per period per round."""
        segment = create_mock_segment(n_players=4, n_rounds=2, n_periods=3)
        chat_counts = {1: {'messages_by_player': {l: 0 for l in 'ABCD'}, 'total_messages': 0}}
        result = build_segment_records(segment, 1, 'test_session', chat_counts)
        assert len(result) == 24  # 4 players * 2 rounds * 3 periods


class TestBuildPlayerRecord:
    """Tests for build_player_record function."""

    def test_returns_none_for_unknown_player(self):
        """Should return None if player not in any group."""
        segment = create_mock_segment(n_players=4)
        chat_counts = {1: {'messages_by_player': {'A': 0}, 'total_messages': 0}}
        result = build_player_record('Z', segment, 1, 'test_session', 1, 1, chat_counts)
        assert result is None

    def test_creates_correct_record_structure(self):
        """Should create record with all required fields."""
        segment = create_mock_segment(n_players=4)
        chat_counts = {1: {'messages_by_player': {'A': 3, 'B': 2, 'C': 0, 'D': 0}, 'total_messages': 5}}
        result = build_player_record('A', segment, 3, 'test_session', 2, 1, chat_counts)
        assert result['messages_sent_segment'] == 3
        assert result['messages_received_segment'] == 2
        assert result['total_group_messages'] == 5


class TestGetPlayerChatCounts:
    """Tests for get_player_chat_counts function."""

    def test_extracts_correct_counts(self):
        """Should return (sent, received, total) tuple."""
        chat_counts = {1: {'messages_by_player': {'A': 5}, 'total_messages': 10}}
        sent, received, total = get_player_chat_counts('A', 1, chat_counts)
        assert (sent, received, total) == (5, 5, 10)

    def test_handles_missing_group(self):
        """Should return zeros for unknown group."""
        sent, received, total = get_player_chat_counts('A', 99, {})
        assert (sent, received, total) == (0, 0, 0)


class TestCreateRecordDict:
    """Tests for create_record_dict function."""

    def test_creates_correct_dict_structure(self):
        """Should create dict with all fields correctly populated."""
        result = create_record_dict('session_1', 3, 5, 2, 'B', 2, 4, 6, 10)
        assert result['session_id'] == 'session_1'
        assert result['messages_sent_segment'] == 4


# =====
# Integration tests
# =====
class TestChatActivityIntegration:
    """Integration tests verifying end-to-end behavior."""

    def test_no_chat_segments_have_zero_activity(self):
        """Segments 1-2 should have all zeros for chat activity."""
        segment = create_mock_segment(n_players=4, n_rounds=1, n_periods=2, chat_msgs={'A': 5})
        chat_counts = compute_segment_chat_counts(segment, 1)
        records = build_segment_records(segment, 1, 'test', chat_counts)
        assert all(r['total_group_messages'] == 0 for r in records)

    def test_chat_segments_have_activity(self):
        """Segments 3-4 should have chat activity from messages."""
        segment = create_mock_segment(n_players=4, n_rounds=1, n_periods=2, chat_msgs={'A': 5, 'B': 3})
        chat_counts = compute_segment_chat_counts(segment, 3)
        records = build_segment_records(segment, 3, 'test', chat_counts)
        a_record = next(r for r in records if r['player'] == 'A')
        assert a_record['messages_sent_segment'] == 5
        assert a_record['messages_received_segment'] == 3

    def test_received_equals_total_minus_sent(self):
        """Received messages should equal total group minus sent."""
        segment = create_mock_segment(n_players=4, n_rounds=1, n_periods=1, chat_msgs={'A': 2, 'B': 3})
        chat_counts = compute_segment_chat_counts(segment, 3)
        records = build_segment_records(segment, 3, 'test', chat_counts)
        for r in records:
            assert r['messages_received_segment'] == r['total_group_messages'] - r['messages_sent_segment']

    def test_same_chat_counts_across_periods_and_rounds(self):
        """All periods/rounds in a segment should have same chat counts."""
        segment = create_mock_segment(n_players=4, n_rounds=3, n_periods=2, chat_msgs={'A': 5})
        chat_counts = compute_segment_chat_counts(segment, 3)
        records = build_segment_records(segment, 3, 'test', chat_counts)
        a_records = [r for r in records if r['player'] == 'A']
        assert len(a_records) == 6
        assert all(r['messages_sent_segment'] == 5 for r in a_records)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_segment(self):
        """Should handle segment with no rounds."""
        segment = MockSegment(name="empty", rounds={}, groups={1: MockGroup(group_id=1, player_labels=['A'])})
        result = init_zero_chat_counts(segment)
        assert result[1]['total_messages'] == 0

    def test_player_with_no_messages(self):
        """Player who sent no messages should have sent=0, received=all."""
        segment = create_mock_segment(n_players=4, n_rounds=1, n_periods=1, chat_msgs={'A': 5, 'B': 5})
        chat_counts = compute_segment_chat_counts(segment, 3)
        records = build_segment_records(segment, 3, 'test', chat_counts)
        c_record = next(r for r in records if r['player'] == 'C')
        assert c_record['messages_sent_segment'] == 0
        assert c_record['messages_received_segment'] == 10


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
