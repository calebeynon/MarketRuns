"""
Purpose: Unit tests for build_group_round_timing_dataset.py
Author: Claude Code
Date: 2025-01-18
"""

import pandas as pd
import pytest
from analysis.derived.build_group_round_timing_dataset import get_sellers_with_timing


# =====
# Test get_sellers_with_timing
# =====
def test_no_sales():
    """No one sold - should return empty list."""
    df = pd.DataFrame({
        "player.sold": [0, 0, 0, 0],
        "player.period_in_round": [1, 1, 1, 1],
        "player.signal": [0.5, 0.5, 0.5, 0.5],
        "player.id_in_group": [1, 2, 3, 4],
        "participant.label": ["A", "B", "C", "D"],
    })
    result = get_sellers_with_timing(df)

    assert result == []


def test_single_sale_period_1():
    """One person sells in period 1."""
    df = pd.DataFrame({
        "player.sold": [1, 0, 0, 0],
        "player.period_in_round": [1, 1, 1, 1],
        "player.signal": [0.5, 0.5, 0.5, 0.5],
        "player.id_in_group": [1, 2, 3, 4],
        "participant.label": ["A", "B", "C", "D"],
    })
    result = get_sellers_with_timing(df)

    assert len(result) == 1
    assert result[0]["period"] == 1
    assert result[0]["label"] == "A"
    assert result[0]["signal"] == 0.5


def test_multiple_sales_same_period():
    """Two sellers in the same period - should be ordered by label."""
    df = pd.DataFrame({
        "player.sold": [1, 1, 0, 0],
        "player.period_in_round": [1, 1, 1, 1],
        "player.signal": [0.5, 0.6, 0.5, 0.5],
        "player.id_in_group": [1, 2, 3, 4],
        "participant.label": ["B", "A", "C", "D"],
    })
    result = get_sellers_with_timing(df)

    assert len(result) == 2
    # First by period (both 1), then by label (A before B)
    assert result[0]["label"] == "A"
    assert result[1]["label"] == "B"


def test_sales_different_periods():
    """Sales across periods 1, 2, 3 - should be ordered by period."""
    df = pd.DataFrame({
        "player.sold": [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        "player.period_in_round": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
        "player.signal": [0.5] * 4 + [0.325] * 4 + [0.188] * 4 + [0.1] * 4,
        "player.id_in_group": [1, 2, 3, 4] * 4,
        "participant.label": ["A", "B", "C", "D"] * 4,
    })
    result = get_sellers_with_timing(df)

    assert len(result) == 3
    assert result[0]["period"] == 2
    assert result[0]["label"] == "A"
    assert result[1]["period"] == 3
    assert result[1]["label"] == "B"
    assert result[2]["period"] == 4
    assert result[2]["label"] == "C"


def test_all_four_sell():
    """All 4 players sell across different periods."""
    df = pd.DataFrame({
        "player.sold": [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        "player.period_in_round": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
        "player.signal": [0.5] * 4 + [0.325] * 4 + [0.188] * 4 + [0.1] * 4,
        "player.id_in_group": [1, 2, 3, 4] * 4,
        "participant.label": ["A", "B", "C", "D"] * 4,
    })
    result = get_sellers_with_timing(df)

    assert len(result) == 4
    # A sells first in period 1, B in period 2, C in period 3, D in period 4
    assert result[0]["period"] == 1
    assert result[0]["label"] == "A"
    assert result[1]["period"] == 2
    assert result[1]["label"] == "B"
    assert result[2]["period"] == 3
    assert result[2]["label"] == "C"
    assert result[3]["period"] == 4
    assert result[3]["label"] == "D"


def test_seller_ordering_by_period():
    """Verify sellers ordered by period first, then by label."""
    df = pd.DataFrame({
        # D sells in period 1, A and C sell in period 2
        "player.sold": [0, 0, 0, 1, 1, 0, 1, 1],
        "player.period_in_round": [1, 1, 1, 1, 2, 2, 2, 2],
        "player.signal": [0.5, 0.5, 0.5, 0.5, 0.325, 0.325, 0.325, 0.325],
        "player.id_in_group": [1, 2, 3, 4, 1, 2, 3, 4],
        "participant.label": ["A", "B", "C", "D", "A", "B", "C", "D"],
    })
    result = get_sellers_with_timing(df)

    assert len(result) == 3
    # D first (period 1), then A and C (period 2, alphabetical)
    assert result[0]["period"] == 1
    assert result[0]["label"] == "D"
    assert result[1]["period"] == 2
    assert result[1]["label"] == "A"
    assert result[2]["period"] == 2
    assert result[2]["label"] == "C"


def test_signal_values_preserved():
    """Signal value at sale time is correctly captured for each seller."""
    df = pd.DataFrame({
        "player.sold": [1, 0, 0, 0, 1, 1, 0, 0],
        "player.period_in_round": [1, 1, 1, 1, 2, 2, 2, 2],
        "player.signal": [0.5, 0.5, 0.5, 0.5, 0.675, 0.675, 0.675, 0.675],
        "player.id_in_group": [1, 2, 3, 4, 1, 2, 3, 4],
        "participant.label": ["A", "B", "C", "D", "A", "B", "C", "D"],
    })
    result = get_sellers_with_timing(df)

    assert len(result) == 2
    # A sold in period 1 with signal 0.5
    assert result[0]["period"] == 1
    assert result[0]["label"] == "A"
    assert result[0]["signal"] == 0.5
    # B sold in period 2 with signal 0.675
    assert result[1]["period"] == 2
    assert result[1]["label"] == "B"
    assert result[1]["signal"] == 0.675


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
