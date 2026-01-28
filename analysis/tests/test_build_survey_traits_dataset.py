"""
Purpose: Unit tests for build_survey_traits_dataset.py
Author: Claude Code
Date: 2026-01-28
"""

import pytest
from analysis.derived.build_survey_traits_dataset import (
    encode_7pt,
    encode_4pt,
    reverse_7pt,
    reverse_4pt,
    compute_extraversion,
    compute_agreeableness,
    compute_conscientiousness,
    compute_neuroticism,
    compute_openness,
    compute_impulsivity,
    compute_state_anxiety,
)
import pandas as pd


# =====
# Likert encoding tests
# =====
def test_encode_7pt_endpoints():
    """7-point Likert maps Strongly Disagree=1, Strongly Agree=7."""
    assert encode_7pt("Strongly Disagree") == 1
    assert encode_7pt("Strongly Agree") == 7


def test_encode_7pt_midpoint():
    """7-point midpoint is 4."""
    assert encode_7pt("Neither agree nor disagree") == 4


def test_encode_7pt_all_values():
    """All 7-point values map to 1-7 in order."""
    labels = [
        "Strongly Disagree", "Disagree Moderately", "Disagree a little",
        "Neither agree nor disagree", "Agree a little",
        "Agree Moderately", "Strongly Agree",
    ]
    for i, label in enumerate(labels, start=1):
        assert encode_7pt(label) == i


def test_encode_4pt_endpoints():
    """4-point Likert maps Not at all=1, Very much=4."""
    assert encode_4pt("Not at all") == 1
    assert encode_4pt("Very much") == 4


def test_encode_4pt_all_values():
    """All 4-point values map to 1-4 in order."""
    labels = ["Not at all", "Somewhat", "Moderately", "Very much"]
    for i, label in enumerate(labels, start=1):
        assert encode_4pt(label) == i


# =====
# Reverse coding tests
# =====
def test_reverse_7pt():
    """Reverse coding: 8 - value for 7-point scale."""
    assert reverse_7pt(1) == 7
    assert reverse_7pt(7) == 1
    assert reverse_7pt(4) == 4


def test_reverse_4pt():
    """Reverse coding: 5 - value for 4-point scale."""
    assert reverse_4pt(1) == 4
    assert reverse_4pt(4) == 1
    assert reverse_4pt(2) == 3


# =====
# BFI-10 trait computation tests
# =====
def make_survey_row(**overrides) -> pd.Series:
    """Create a survey row with all questions set to neutral, then apply overrides."""
    # Default: all 7-point items neutral, all 4-point items "Moderately"
    data = {}
    for i in range(1, 7):
        data[f"player.q{i}"] = "Moderately"
    for i in range(7, 25):
        data[f"player.q{i}"] = "Neither agree nor disagree"
    data["player.q25"] = 20
    data["player.q26"] = "Female"
    data.update(overrides)
    return pd.Series(data)


def test_extraversion_both_agree():
    """Both items agree strongly -> high extraversion."""
    row = make_survey_row(**{
        "player.q7": "Strongly Agree",      # forward: 7
        "player.q12": "Strongly Disagree",   # reverse: 8-1=7
    })
    assert compute_extraversion(row) == 7.0


def test_extraversion_both_disagree():
    """Both items disagree strongly -> low extraversion."""
    row = make_survey_row(**{
        "player.q7": "Strongly Disagree",    # forward: 1
        "player.q12": "Strongly Agree",      # reverse: 8-7=1
    })
    assert compute_extraversion(row) == 1.0


def test_extraversion_mixed():
    """Mixed responses -> middle score."""
    row = make_survey_row(**{
        "player.q7": "Strongly Agree",       # forward: 7
        "player.q12": "Strongly Agree",      # reverse: 8-7=1
    })
    assert compute_extraversion(row) == 4.0


def test_agreeableness_high():
    """Agreeableness: q13(+), q8(R)."""
    row = make_survey_row(**{
        "player.q13": "Strongly Agree",      # forward: 7
        "player.q8": "Strongly Disagree",    # reverse: 8-1=7
    })
    assert compute_agreeableness(row) == 7.0


def test_conscientiousness_high():
    """Conscientiousness: q9(+), q14(R)."""
    row = make_survey_row(**{
        "player.q9": "Strongly Agree",
        "player.q14": "Strongly Disagree",
    })
    assert compute_conscientiousness(row) == 7.0


def test_neuroticism_high():
    """Neuroticism: q10(+), q15(R)."""
    row = make_survey_row(**{
        "player.q10": "Strongly Agree",
        "player.q15": "Strongly Disagree",
    })
    assert compute_neuroticism(row) == 7.0


def test_openness_high():
    """Openness: q11(+), q16(R)."""
    row = make_survey_row(**{
        "player.q11": "Strongly Agree",
        "player.q16": "Strongly Disagree",
    })
    assert compute_openness(row) == 7.0


# =====
# Impulsivity tests
# =====
def test_impulsivity_all_high():
    """All items agree strongly -> maximum impulsivity."""
    overrides = {}
    # Forward items: agree strongly = 7
    for q in ["player.q18", "player.q19", "player.q23", "player.q24"]:
        overrides[q] = "Strongly Agree"
    # Reverse items: disagree strongly -> reverse(1) = 7
    for q in ["player.q17", "player.q20", "player.q21", "player.q22"]:
        overrides[q] = "Strongly Disagree"
    row = make_survey_row(**overrides)
    assert compute_impulsivity(row) == 7.0


def test_impulsivity_all_low():
    """All items disagree strongly -> minimum impulsivity."""
    overrides = {}
    for q in ["player.q18", "player.q19", "player.q23", "player.q24"]:
        overrides[q] = "Strongly Disagree"
    for q in ["player.q17", "player.q20", "player.q21", "player.q22"]:
        overrides[q] = "Strongly Agree"
    row = make_survey_row(**overrides)
    assert compute_impulsivity(row) == 1.0


def test_impulsivity_known_values():
    """Verify computation with known input values."""
    overrides = {
        "player.q17": "Agree a little",         # reverse: 8-5=3
        "player.q18": "Agree Moderately",        # forward: 6
        "player.q19": "Disagree a little",       # forward: 3
        "player.q20": "Neither agree nor disagree",  # reverse: 8-4=4
        "player.q21": "Strongly Agree",          # reverse: 8-7=1
        "player.q22": "Disagree Moderately",     # reverse: 8-2=6
        "player.q23": "Strongly Disagree",       # forward: 1
        "player.q24": "Agree a little",          # forward: 5
    }
    row = make_survey_row(**overrides)
    # Sum: 3 + 6 + 3 + 4 + 1 + 6 + 1 + 5 = 29, mean = 29/8 = 3.625
    assert compute_impulsivity(row) == pytest.approx(3.625)


# =====
# State anxiety tests
# =====
def test_state_anxiety_max():
    """Maximum anxiety: positive mood reversed high, negative mood forward high."""
    overrides = {}
    # Reverse items (positive mood): Not at all -> reverse(1)=4
    for q in ["player.q1", "player.q2", "player.q3"]:
        overrides[q] = "Not at all"
    # Forward items (negative mood): Very much = 4
    for q in ["player.q4", "player.q5", "player.q6"]:
        overrides[q] = "Very much"
    row = make_survey_row(**overrides)
    assert compute_state_anxiety(row) == 4.0


def test_state_anxiety_min():
    """Minimum anxiety: positive mood reversed low, negative mood forward low."""
    overrides = {}
    for q in ["player.q1", "player.q2", "player.q3"]:
        overrides[q] = "Very much"
    for q in ["player.q4", "player.q5", "player.q6"]:
        overrides[q] = "Not at all"
    row = make_survey_row(**overrides)
    assert compute_state_anxiety(row) == 1.0


def test_state_anxiety_known_values():
    """Verify computation with known input values."""
    overrides = {
        "player.q1": "Somewhat",      # reverse: 5-2=3
        "player.q2": "Not at all",    # reverse: 5-1=4
        "player.q3": "Moderately",    # reverse: 5-3=2
        "player.q4": "Moderately",    # forward: 3
        "player.q5": "Somewhat",      # forward: 2
        "player.q6": "Very much",     # forward: 4
    }
    row = make_survey_row(**overrides)
    # Sum: 3 + 4 + 2 + 3 + 2 + 4 = 18, mean = 18/6 = 3.0
    assert compute_state_anxiety(row) == pytest.approx(3.0)


# =====
# Score range validation
# =====
def test_bfi_score_range():
    """All BFI-10 traits should be in [1, 7]."""
    # Test with all neutral responses
    row = make_survey_row()
    for compute_fn in [
        compute_extraversion, compute_agreeableness,
        compute_conscientiousness, compute_neuroticism, compute_openness,
    ]:
        score = compute_fn(row)
        assert 1.0 <= score <= 7.0, f"{compute_fn.__name__} out of range: {score}"


def test_impulsivity_score_range():
    """Impulsivity should be in [1, 7]."""
    row = make_survey_row()
    score = compute_impulsivity(row)
    assert 1.0 <= score <= 7.0


def test_anxiety_score_range():
    """State anxiety should be in [1, 4]."""
    row = make_survey_row()
    score = compute_state_anxiety(row)
    assert 1.0 <= score <= 4.0


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
