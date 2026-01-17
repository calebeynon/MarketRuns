"""
Purpose: Unit tests for build_first_sale_dataset.py
Author: Claude Code
Date: 2025-01-11
"""

import pandas as pd
import pytest
from analysis.derived.build_first_sale_dataset import get_first_sale_for_group_round


# =====
# Test get_first_sale_for_group_round
# =====
def test_no_sales():
    """No one sold - should return None for signal."""
    df = pd.DataFrame({
        "player.sold": [0, 0, 0, 0],
        "player.period_in_round": [1, 1, 1, 1],
        "player.signal": [0.5, 0.5, 0.5, 0.5],
    })
    result = get_first_sale_for_group_round(df)

    assert result["first_sale_period"] is None
    assert result["signal_at_first_sale"] is None
    assert result["n_sellers_first_period"] == 0


def test_single_sale_period_1():
    """One person sells in period 1."""
    df = pd.DataFrame({
        "player.sold": [1, 0, 0, 0],
        "player.period_in_round": [1, 1, 1, 1],
        "player.signal": [0.5, 0.5, 0.5, 0.5],
    })
    result = get_first_sale_for_group_round(df)

    assert result["first_sale_period"] == 1
    assert result["signal_at_first_sale"] == 0.5
    assert result["n_sellers_first_period"] == 1


def test_multiple_sales_same_period():
    """Multiple people sell in the same period."""
    df = pd.DataFrame({
        "player.sold": [1, 1, 0, 0],
        "player.period_in_round": [1, 1, 1, 1],
        "player.signal": [0.5, 0.5, 0.5, 0.5],
    })
    result = get_first_sale_for_group_round(df)

    assert result["first_sale_period"] == 1
    assert result["signal_at_first_sale"] == 0.5
    assert result["n_sellers_first_period"] == 2


def test_sale_in_later_period():
    """First sale happens in period 3."""
    df = pd.DataFrame({
        "player.sold": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "player.period_in_round": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        "player.signal": [0.5] * 4 + [0.325] * 4 + [0.188] * 4,
    })
    result = get_first_sale_for_group_round(df)

    assert result["first_sale_period"] == 3
    assert result["signal_at_first_sale"] == 0.188
    assert result["n_sellers_first_period"] == 1


def test_sales_across_multiple_periods():
    """Sales in periods 2 and 3 - should return period 2."""
    df = pd.DataFrame({
        "player.sold": [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        "player.period_in_round": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        "player.signal": [0.5] * 4 + [0.325] * 4 + [0.188] * 4,
    })
    result = get_first_sale_for_group_round(df)

    assert result["first_sale_period"] == 2
    assert result["signal_at_first_sale"] == 0.325
    assert result["n_sellers_first_period"] == 1


def test_signal_value_preserved():
    """Signal value at first sale is correctly captured."""
    df = pd.DataFrame({
        "player.sold": [0, 0, 0, 0, 1, 1, 0, 0],
        "player.period_in_round": [1, 1, 1, 1, 2, 2, 2, 2],
        "player.signal": [0.5, 0.5, 0.5, 0.5, 0.675, 0.675, 0.675, 0.675],
    })
    result = get_first_sale_for_group_round(df)

    assert result["first_sale_period"] == 2
    assert result["signal_at_first_sale"] == 0.675
    assert result["n_sellers_first_period"] == 2


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
