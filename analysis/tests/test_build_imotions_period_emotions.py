"""
Purpose: Unit tests for build_imotions_period_emotions.py
Author: Claude Code
Date: 2026-01-28
"""

import pytest
import pandas as pd
import numpy as np
from analysis.derived.build_imotions_period_emotions import (
    parse_market_period_annotation,
    extract_player_label,
    aggregate_emotions,
    EMOTION_COLS,
    ANNOTATION_COL,
)


# =====
# Annotation parsing tests
# =====
def test_parse_valid_market_period():
    """Valid MarketPeriod annotation parses correctly with m-1 offset."""
    result = parse_market_period_annotation("s1r1m2MarketPeriod")
    assert result == (1, 1, 1)  # seg=1, round=1, period=m2-1=1


def test_parse_period_offset():
    """Annotation m{N} maps to period N-1."""
    result = parse_market_period_annotation("s2r5m4MarketPeriod")
    assert result == (2, 5, 3)  # seg=2, round=5, period=m4-1=3


def test_parse_high_m_value():
    """High m values (long rounds) map correctly."""
    result = parse_market_period_annotation("s1r2m10MarketPeriod")
    assert result == (1, 2, 9)  # period=m10-1=9


def test_parse_all_segments():
    """All 4 segment numbers parse correctly."""
    for seg in range(1, 5):
        result = parse_market_period_annotation(f"s{seg}r1m2MarketPeriod")
        assert result[0] == seg


def test_parse_wait_annotation_returns_none():
    """MarketPeriodWait is NOT a MarketPeriod — should return None."""
    assert parse_market_period_annotation("s1r1m2MarketPeriodWait") is None


def test_parse_payoff_wait_returns_none():
    """MarketPeriodPayoffWait should return None."""
    assert parse_market_period_annotation("s1r1m2MarketPeriodPayoffWait") is None


def test_parse_results_returns_none():
    """Results annotation should return None."""
    assert parse_market_period_annotation("s1r1m3Results") is None


def test_parse_results_wait_returns_none():
    """ResultsWait annotation should return None."""
    assert parse_market_period_annotation("s1r1m3ResultsWait") is None


def test_parse_segment_intro_returns_none():
    """SegmentIntro annotation should return None."""
    assert parse_market_period_annotation("s1r1m1SegmentIntro") is None


def test_parse_chat_returns_none():
    """Chat annotation should return None."""
    assert parse_market_period_annotation("s3r1m2Chat") is None


def test_parse_survey_returns_none():
    """Survey annotation should return None."""
    assert parse_market_period_annotation("Survey") is None


def test_parse_nan_returns_none():
    """NaN annotation returns None."""
    assert parse_market_period_annotation(float("nan")) is None


def test_parse_empty_string_returns_none():
    """Empty string returns None."""
    assert parse_market_period_annotation("") is None


def test_parse_label_returns_none():
    """Label annotation (calibration) returns None."""
    assert parse_market_period_annotation("Label") is None


# =====
# Player label extraction tests
# =====
def test_extract_label_standard():
    """Standard filename pattern extracts letter correctly."""
    assert extract_player_label("001_R3.csv") == "R"
    assert extract_player_label("016_A3.csv") == "A"


def test_extract_label_different_sessions():
    """Different session suffixes still extract the letter."""
    assert extract_player_label("005_M4.csv") == "M"
    assert extract_player_label("010_G8.csv") == "G"


def test_extract_label_export_merge():
    """ExportMerge.csv should not match (returns None)."""
    assert extract_player_label("ExportMerge.csv") is None


def test_extract_label_invalid_format():
    """Non-matching filename returns None."""
    assert extract_player_label("random_file.csv") is None


# =====
# Period offset regression tests
# =====
def test_m1_is_segment_intro_not_market_period():
    """m1 annotations are SegmentIntro, never MarketPeriod.

    Even if someone mistakenly created an m1MarketPeriod annotation, it would
    produce period 0, which is invalid. This test documents expected behavior.
    """
    # Real m1 annotations are SegmentIntro, which returns None
    assert parse_market_period_annotation("s1r1m1SegmentIntro") is None
    assert parse_market_period_annotation("s2r1m1SegmentIntro") is None

    # Hypothetical m1MarketPeriod would produce period 0 (invalid edge case)
    # This documents the formula behavior, not that such annotations exist
    result = parse_market_period_annotation("s1r1m1MarketPeriod")
    assert result == (1, 1, 0)  # period = m1 - 1 = 0


def test_offset_formula_regression():
    """Explicit verification that period = m_value - 1.

    This formula corrects for the pre-increment in generate_annotations_unfiltered_v2.py.
    Multiple examples ensure the formula doesn't regress.
    """
    # m2 -> period 1 (first actual MarketPeriod in a round)
    assert parse_market_period_annotation("s1r1m2MarketPeriod") == (1, 1, 1)

    # m3 -> period 2
    assert parse_market_period_annotation("s1r1m3MarketPeriod") == (1, 1, 2)

    # m4 -> period 3
    assert parse_market_period_annotation("s1r1m4MarketPeriod") == (1, 1, 3)

    # m5 -> period 4
    assert parse_market_period_annotation("s2r3m5MarketPeriod") == (2, 3, 4)

    # m6 -> period 5
    assert parse_market_period_annotation("s3r10m6MarketPeriod") == (3, 10, 5)


def test_high_period_values():
    """High m values (m10, m11, m12+) parse correctly for long rounds."""
    # m10 -> period 9
    assert parse_market_period_annotation("s1r2m10MarketPeriod") == (1, 2, 9)

    # m11 -> period 10
    assert parse_market_period_annotation("s2r5m11MarketPeriod") == (2, 5, 10)

    # m12 -> period 11
    assert parse_market_period_annotation("s3r8m12MarketPeriod") == (3, 8, 11)

    # m15 -> period 14 (very long round)
    assert parse_market_period_annotation("s4r14m15MarketPeriod") == (4, 14, 14)


def test_boundary_m2_produces_period_1():
    """m2 is the first MarketPeriod annotation, mapping to period 1.

    This is the critical boundary case: m1 is SegmentIntro, m2 is the first
    actual MarketPeriod. Ensure m2 produces period 1 (not 0 or 2).
    """
    # Test across all segments to ensure consistent behavior
    assert parse_market_period_annotation("s1r1m2MarketPeriod")[2] == 1
    assert parse_market_period_annotation("s2r1m2MarketPeriod")[2] == 1
    assert parse_market_period_annotation("s3r1m2MarketPeriod")[2] == 1
    assert parse_market_period_annotation("s4r1m2MarketPeriod")[2] == 1

    # Also test across different rounds
    assert parse_market_period_annotation("s1r5m2MarketPeriod")[2] == 1
    assert parse_market_period_annotation("s1r14m2MarketPeriod")[2] == 1


def test_annotation_non_market_phases_return_none():
    """Non-MarketPeriod phases return None. Only exact 'MarketPeriod' parses."""
    # MarketOutcome (end of period, not the trading phase)
    assert parse_market_period_annotation("s1r1m2MarketOutcome") is None
    assert parse_market_period_annotation("s2r5m4MarketOutcome") is None

    # SegmentIntro (beginning of segment)
    assert parse_market_period_annotation("s1r1m1SegmentIntro") is None
    assert parse_market_period_annotation("s3r1m1SegmentIntro") is None

    # Chat phase (only in segments 3-4)
    assert parse_market_period_annotation("s3r1m2Chat") is None
    assert parse_market_period_annotation("s4r5m3Chat") is None


def test_annotation_results_and_round_phases_return_none():
    """Results, ResultsWait, RoundIntro, and partial matches return None."""
    assert parse_market_period_annotation("s1r1m3Results") is None
    assert parse_market_period_annotation("s2r2m4ResultsWait") is None
    assert parse_market_period_annotation("s1r2m1RoundIntro") is None

    # Partial matches should fail (regex requires exact MarketPeriod suffix)
    assert parse_market_period_annotation("s1r1m2MarketPeriodPartial") is None
    assert parse_market_period_annotation("s1r1m2market_period") is None


# =====
# Aggregation tests
# =====
def make_market_df(
    n_frames: int = 5, segment: int = 1, round_num: int = 1,
    period: int = 1, emotion_values: dict = None,
):
    """Create a mock market DataFrame for aggregation testing."""
    data = {"segment": [segment] * n_frames, "round": [round_num] * n_frames,
            "period": [period] * n_frames}
    emotion_values = emotion_values or {}
    for col in EMOTION_COLS:
        data[col] = emotion_values.get(col, [0.5] * n_frames)
    return pd.DataFrame(data)


def test_aggregate_single_period():
    """Single period with constant values aggregates to those values."""
    market_df = make_market_df(n_frames=10, emotion_values={
        "Anger": [0.1] * 10,
        "Joy": [0.8] * 10,
    })
    records = aggregate_emotions(market_df, "test_session", "A")
    assert len(records) == 1
    assert records[0]["anger_mean"] == pytest.approx(0.1)
    assert records[0]["joy_mean"] == pytest.approx(0.8)
    assert records[0]["n_frames"] == 10


def test_aggregate_varying_values():
    """Aggregation computes correct mean with varying values."""
    market_df = make_market_df(n_frames=4, emotion_values={
        "Fear": [0.0, 0.2, 0.4, 0.6],
    })
    records = aggregate_emotions(market_df, "test_session", "B")
    assert records[0]["fear_mean"] == pytest.approx(0.3)


def test_aggregate_with_nan():
    """NaN frames are ignored in mean calculation."""
    market_df = make_market_df(n_frames=4, emotion_values={
        "Sadness": [0.2, float("nan"), 0.4, float("nan")],
    })
    records = aggregate_emotions(market_df, "test_session", "C")
    # Mean of [0.2, 0.4] = 0.3
    assert records[0]["sadness_mean"] == pytest.approx(0.3)
    # n_frames counts all rows, including NaN
    assert records[0]["n_frames"] == 4


def test_aggregate_multiple_periods():
    """Multiple periods produce separate records."""
    df1 = make_market_df(n_frames=3, segment=1, round_num=1, period=1)
    df2 = make_market_df(n_frames=5, segment=1, round_num=1, period=2)
    market_df = pd.concat([df1, df2], ignore_index=True)

    records = aggregate_emotions(market_df, "test_session", "D")
    assert len(records) == 2

    # Check that both periods are present
    periods = {r["period"] for r in records}
    assert periods == {1, 2}


def test_aggregate_metadata():
    """Session and player metadata is preserved in output."""
    market_df = make_market_df(n_frames=2, segment=3, round_num=7, period=2)
    records = aggregate_emotions(market_df, "4_11-12-tr1", "K")
    assert records[0]["session_id"] == "4_11-12-tr1"
    assert records[0]["player"] == "K"
    assert records[0]["segment"] == 3
    assert records[0]["round"] == 7
    assert records[0]["period"] == 2


def test_aggregate_all_nan_emotion():
    """All-NaN emotion column produces NaN mean."""
    market_df = make_market_df(n_frames=3, emotion_values={
        "Surprise": [float("nan")] * 3,
    })
    records = aggregate_emotions(market_df, "test", "A")
    assert np.isnan(records[0]["surprise_mean"])


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
