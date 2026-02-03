"""
Purpose: Unit tests for build_personality_traits_dataset.py
Author: Claude Code
Date: 2026-02-02

Tests verify:
1. Likert encoding (4-point and 7-point scales)
2. Reverse coding logic
3. Trait calculation formulas
4. Consistency with Issue #18 trait calculation approach
"""

import pandas as pd
import numpy as np
import pytest
from analysis.derived.build_personality_traits_dataset import (
    encode_4pt,
    encode_7pt,
    reverse_4pt,
    reverse_7pt,
    calculate_state_anxiety,
    calculate_extraversion,
    calculate_agreeableness,
    calculate_conscientiousness,
    calculate_neuroticism,
    calculate_openness,
    calculate_impulsivity,
    calculate_tipi_trait,
    safe_mean,
    SCALE_4PT,
    SCALE_7PT,
)


# =====
# Test 4-point Likert encoding
# =====
class TestEncode4pt:
    """Tests for 4-point Likert scale encoding."""

    def test_not_at_all_returns_1(self):
        assert encode_4pt("Not at all") == 1

    def test_somewhat_returns_2(self):
        assert encode_4pt("Somewhat") == 2

    def test_moderately_returns_3(self):
        assert encode_4pt("Moderately") == 3

    def test_very_much_returns_4(self):
        assert encode_4pt("Very much") == 4

    def test_nan_input_returns_nan(self):
        assert np.isnan(encode_4pt(np.nan))

    def test_none_input_returns_nan(self):
        assert np.isnan(encode_4pt(None))

    def test_invalid_string_returns_nan(self):
        assert np.isnan(encode_4pt("Invalid response"))

    def test_all_valid_options_covered(self):
        """Verify all 4-point scale options are in the mapping."""
        expected_options = ["Not at all", "Somewhat", "Moderately", "Very much"]
        for option in expected_options:
            assert option in SCALE_4PT


# =====
# Test 7-point Likert encoding
# =====
class TestEncode7pt:
    """Tests for 7-point Likert scale encoding."""

    def test_strongly_disagree_returns_1(self):
        assert encode_7pt("Strongly Disagree") == 1

    def test_disagree_moderately_returns_2(self):
        assert encode_7pt("Disagree Moderately") == 2

    def test_disagree_a_little_returns_3(self):
        assert encode_7pt("Disagree a little") == 3

    def test_neither_returns_4(self):
        assert encode_7pt("Neither agree nor disagree") == 4

    def test_agree_a_little_returns_5(self):
        assert encode_7pt("Agree a little") == 5

    def test_agree_moderately_returns_6(self):
        assert encode_7pt("Agree Moderately") == 6

    def test_strongly_agree_returns_7(self):
        assert encode_7pt("Strongly Agree") == 7

    def test_nan_input_returns_nan(self):
        assert np.isnan(encode_7pt(np.nan))

    def test_invalid_string_returns_nan(self):
        assert np.isnan(encode_7pt("Invalid"))

    def test_all_valid_options_covered(self):
        """Verify all 7-point scale options are in the mapping."""
        expected_options = [
            "Strongly Disagree", "Disagree Moderately", "Disagree a little",
            "Neither agree nor disagree", "Agree a little", "Agree Moderately",
            "Strongly Agree"
        ]
        for option in expected_options:
            assert option in SCALE_7PT


# =====
# Test reverse coding
# =====
class TestReverseCoding:
    """Tests for reverse coding functions."""

    def test_reverse_4pt_min_becomes_max(self):
        """1 -> 4 (5 - 1 = 4)"""
        assert reverse_4pt(1) == 4

    def test_reverse_4pt_max_becomes_min(self):
        """4 -> 1 (5 - 4 = 1)"""
        assert reverse_4pt(4) == 1

    def test_reverse_4pt_middle_values(self):
        """2 -> 3, 3 -> 2"""
        assert reverse_4pt(2) == 3
        assert reverse_4pt(3) == 2

    def test_reverse_4pt_nan_returns_nan(self):
        assert np.isnan(reverse_4pt(np.nan))

    def test_reverse_7pt_min_becomes_max(self):
        """1 -> 7 (8 - 1 = 7)"""
        assert reverse_7pt(1) == 7

    def test_reverse_7pt_max_becomes_min(self):
        """7 -> 1 (8 - 7 = 1)"""
        assert reverse_7pt(7) == 1

    def test_reverse_7pt_middle_value(self):
        """4 stays 4 (8 - 4 = 4)"""
        assert reverse_7pt(4) == 4

    def test_reverse_7pt_all_values(self):
        """Test all 7-point values reverse correctly."""
        expected = {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
        for original, reversed_val in expected.items():
            assert reverse_7pt(original) == reversed_val

    def test_reverse_7pt_nan_returns_nan(self):
        assert np.isnan(reverse_7pt(np.nan))


# =====
# Test safe_mean
# =====
class TestSafeMean:
    """Tests for mean calculation with NaN handling."""

    def test_all_valid_values(self):
        assert safe_mean([1, 2, 3, 4, 5]) == 3.0

    def test_with_nan_values(self):
        """NaN values should be excluded from mean."""
        assert safe_mean([1, np.nan, 3]) == 2.0

    def test_all_nan_returns_nan(self):
        assert np.isnan(safe_mean([np.nan, np.nan]))

    def test_empty_list_returns_nan(self):
        assert np.isnan(safe_mean([]))

    def test_single_value(self):
        assert safe_mean([5]) == 5.0


# =====
# Test state anxiety calculation
# =====
class TestCalculateStateAnxiety:
    """Tests for state anxiety trait calculation."""

    def test_all_low_anxiety_responses(self):
        """Calm, relaxed, content (high) + not tense/upset/worried (low)."""
        responses = {
            "q1": "Very much",      # calm (reversed: 4->1)
            "q2": "Very much",      # relaxed (reversed: 4->1)
            "q3": "Very much",      # content (reversed: 4->1)
            "q4": "Not at all",     # tense (direct: 1)
            "q5": "Not at all",     # upset (direct: 1)
            "q6": "Not at all",     # worried (direct: 1)
        }
        result = calculate_state_anxiety(responses)
        assert result == 1.0  # Low anxiety

    def test_all_high_anxiety_responses(self):
        """Not calm/relaxed/content (low) + tense/upset/worried (high)."""
        responses = {
            "q1": "Not at all",     # calm (reversed: 1->4)
            "q2": "Not at all",     # relaxed (reversed: 1->4)
            "q3": "Not at all",     # content (reversed: 1->4)
            "q4": "Very much",      # tense (direct: 4)
            "q5": "Very much",      # upset (direct: 4)
            "q6": "Very much",      # worried (direct: 4)
        }
        result = calculate_state_anxiety(responses)
        assert result == 4.0  # High anxiety

    def test_moderate_anxiety_responses(self):
        """Mixed moderate responses."""
        responses = {
            "q1": "Somewhat",       # calm (reversed: 2->3)
            "q2": "Moderately",     # relaxed (reversed: 3->2)
            "q3": "Somewhat",       # content (reversed: 2->3)
            "q4": "Somewhat",       # tense (direct: 2)
            "q5": "Moderately",     # upset (direct: 3)
            "q6": "Somewhat",       # worried (direct: 2)
        }
        result = calculate_state_anxiety(responses)
        # (3 + 2 + 3 + 2 + 3 + 2) / 6 = 15/6 = 2.5
        assert result == 2.5

    def test_range_is_1_to_4(self):
        """State anxiety should always be between 1 and 4."""
        # Minimum case
        low_responses = {f"q{i}": "Very much" for i in range(1, 4)}
        low_responses.update({f"q{i}": "Not at all" for i in range(4, 7)})
        assert 1 <= calculate_state_anxiety(low_responses) <= 4

        # Maximum case
        high_responses = {f"q{i}": "Not at all" for i in range(1, 4)}
        high_responses.update({f"q{i}": "Very much" for i in range(4, 7)})
        assert 1 <= calculate_state_anxiety(high_responses) <= 4


# =====
# Test TIPI Big Five calculations
# =====
class TestCalculateTIPITraits:
    """Tests for Big Five personality trait calculations."""

    def test_tipi_trait_high_score(self):
        """High direct + low reverse = high trait score."""
        responses = {"q_direct": "Strongly Agree", "q_reverse": "Strongly Disagree"}
        # direct: 7, reverse: 8-1=7, mean = 7
        result = calculate_tipi_trait(responses, "q_direct", "q_reverse")
        assert result == 7.0

    def test_tipi_trait_low_score(self):
        """Low direct + high reverse = low trait score."""
        responses = {"q_direct": "Strongly Disagree", "q_reverse": "Strongly Agree"}
        # direct: 1, reverse: 8-7=1, mean = 1
        result = calculate_tipi_trait(responses, "q_direct", "q_reverse")
        assert result == 1.0

    def test_tipi_trait_middle_score(self):
        """Neutral responses = middle score."""
        responses = {
            "q_direct": "Neither agree nor disagree",
            "q_reverse": "Neither agree nor disagree"
        }
        # direct: 4, reverse: 8-4=4, mean = 4
        result = calculate_tipi_trait(responses, "q_direct", "q_reverse")
        assert result == 4.0


class TestCalculateExtraversion:
    """Tests for extraversion calculation."""

    def test_high_extraversion(self):
        """High on q7 (extraverted), low on q12 (reserved)."""
        responses = {
            "q7": "Strongly Agree",        # direct: 7
            "q12": "Strongly Disagree",    # reverse: 8-1=7
        }
        assert calculate_extraversion(responses) == 7.0

    def test_low_extraversion(self):
        """Low on q7 (extraverted), high on q12 (reserved)."""
        responses = {
            "q7": "Strongly Disagree",     # direct: 1
            "q12": "Strongly Agree",       # reverse: 8-7=1
        }
        assert calculate_extraversion(responses) == 1.0


class TestCalculateAgreeableness:
    """Tests for agreeableness calculation."""

    def test_high_agreeableness(self):
        """High on q13 (sympathetic), low on q8 (critical)."""
        responses = {
            "q8": "Strongly Disagree",     # reverse: 8-1=7
            "q13": "Strongly Agree",       # direct: 7
        }
        assert calculate_agreeableness(responses) == 7.0


class TestCalculateConscientiousness:
    """Tests for conscientiousness calculation."""

    def test_high_conscientiousness(self):
        """High on q9 (dependable), low on q14 (disorganized)."""
        responses = {
            "q9": "Strongly Agree",        # direct: 7
            "q14": "Strongly Disagree",    # reverse: 8-1=7
        }
        assert calculate_conscientiousness(responses) == 7.0


class TestCalculateNeuroticism:
    """Tests for neuroticism calculation."""

    def test_high_neuroticism(self):
        """High on q10 (anxious), low on q15 (calm)."""
        responses = {
            "q10": "Strongly Agree",       # direct: 7
            "q15": "Strongly Disagree",    # reverse: 8-1=7
        }
        assert calculate_neuroticism(responses) == 7.0


class TestCalculateOpenness:
    """Tests for openness calculation."""

    def test_high_openness(self):
        """High on q11 (open), low on q16 (conventional)."""
        responses = {
            "q11": "Strongly Agree",       # direct: 7
            "q16": "Strongly Disagree",    # reverse: 8-1=7
        }
        assert calculate_openness(responses) == 7.0


# =====
# Test impulsivity calculation
# =====
class TestCalculateImpulsivity:
    """Tests for impulsivity trait calculation."""

    def test_high_impulsivity(self):
        """Low on reverse items, high on direct items = high impulsivity."""
        responses = {
            "q17": "Strongly Disagree",    # plan carefully - reverse: 8-1=7
            "q18": "Strongly Agree",       # do without thinking - direct: 7
            "q19": "Strongly Agree",       # don't pay attention - direct: 7
            "q20": "Strongly Disagree",    # self-controlled - reverse: 8-1=7
            "q21": "Strongly Disagree",    # concentrate easily - reverse: 8-1=7
            "q22": "Strongly Disagree",    # careful thinker - reverse: 8-1=7
            "q23": "Strongly Agree",       # say without thinking - direct: 7
            "q24": "Strongly Agree",       # act on impulse - direct: 7
        }
        assert calculate_impulsivity(responses) == 7.0

    def test_low_impulsivity(self):
        """High on reverse items, low on direct items = low impulsivity."""
        responses = {
            "q17": "Strongly Agree",       # plan carefully - reverse: 8-7=1
            "q18": "Strongly Disagree",    # do without thinking - direct: 1
            "q19": "Strongly Disagree",    # don't pay attention - direct: 1
            "q20": "Strongly Agree",       # self-controlled - reverse: 8-7=1
            "q21": "Strongly Agree",       # concentrate easily - reverse: 8-7=1
            "q22": "Strongly Agree",       # careful thinker - reverse: 8-7=1
            "q23": "Strongly Disagree",    # say without thinking - direct: 1
            "q24": "Strongly Disagree",    # act on impulse - direct: 1
        }
        assert calculate_impulsivity(responses) == 1.0

    def test_impulsivity_item_count(self):
        """Impulsivity uses exactly 8 items (q17-q24)."""
        # Neutral on all items
        responses = {f"q{i}": "Neither agree nor disagree" for i in range(17, 25)}
        result = calculate_impulsivity(responses)
        # All 4s: (4+4+4+4+4+4+4+4)/8 = 4
        assert result == 4.0

    def test_impulsivity_range_is_1_to_7(self):
        """Impulsivity should always be between 1 and 7."""
        # Create responses for extremes
        low_imp = {f"q{i}": "Strongly Agree" for i in [17, 20, 21, 22]}
        low_imp.update({f"q{i}": "Strongly Disagree" for i in [18, 19, 23, 24]})

        high_imp = {f"q{i}": "Strongly Disagree" for i in [17, 20, 21, 22]}
        high_imp.update({f"q{i}": "Strongly Agree" for i in [18, 19, 23, 24]})

        assert 1 <= calculate_impulsivity(low_imp) <= 7
        assert 1 <= calculate_impulsivity(high_imp) <= 7


# =====
# Test consistency with Issue #18 approach
# =====
class TestIssue18Consistency:
    """
    Verify trait calculations match the approach from Issue #18.

    From Issue #18:
    - BFI-10: Mean of 2 items each with reverse coding (7-point, range 1-7)
    - Impulsivity: Mean of 8 items (4 forward, 4 reverse) (7-point, range 1-7)
    - State anxiety: Mean of 6 items (3 reverse, 3 direct) (4-point, range 1-4)
    """

    def test_big_five_uses_two_items(self):
        """Each Big Five trait should be mean of exactly 2 items."""
        # If we give neutral to one and extreme to other, mean should be
        # between the two values
        responses_ext = {"q7": "Strongly Agree", "q12": "Neither agree nor disagree"}
        result = calculate_extraversion(responses_ext)
        # direct: 7, reverse: 8-4=4, mean = 5.5
        assert result == 5.5

    def test_impulsivity_uses_eight_items(self):
        """Impulsivity should be mean of exactly 8 items."""
        # Set 4 reverse items to high (1 after reversal) and 4 direct to low (1)
        responses = {f"q{i}": "Strongly Agree" for i in [17, 20, 21, 22]}
        responses.update({f"q{i}": "Strongly Disagree" for i in [18, 19, 23, 24]})
        # Reverse items: 8-7=1 each, direct items: 1 each
        # All 8 items = 1, mean = 1
        assert calculate_impulsivity(responses) == 1.0

    def test_state_anxiety_uses_six_items(self):
        """State anxiety should be mean of exactly 6 items."""
        # All neutral-ish responses
        responses = {
            "q1": "Somewhat",       # 2 -> 3 (reversed)
            "q2": "Somewhat",       # 2 -> 3 (reversed)
            "q3": "Somewhat",       # 2 -> 3 (reversed)
            "q4": "Somewhat",       # 2 (direct)
            "q5": "Somewhat",       # 2 (direct)
            "q6": "Somewhat",       # 2 (direct)
        }
        result = calculate_state_anxiety(responses)
        # (3+3+3+2+2+2)/6 = 15/6 = 2.5
        assert result == 2.5

    def test_reverse_items_correctly_identified(self):
        """
        Verify reverse-coded items match Issue #18 specification.

        State anxiety reverse: q1, q2, q3 (positive mood)
        TIPI reverse: q12, q8, q14, q15, q16
        Impulsivity reverse: q17, q20, q21, q22
        """
        # State anxiety: positive mood items should be reverse coded
        low_anxiety_positive = {
            "q1": "Very much", "q2": "Very much", "q3": "Very much",
            "q4": "Not at all", "q5": "Not at all", "q6": "Not at all"
        }
        assert calculate_state_anxiety(low_anxiety_positive) == 1.0

        # Impulsivity: planning/self-control items should be reverse coded
        # q17 (plan), q20 (self-control), q21 (concentrate), q22 (careful)
        low_imp = {
            "q17": "Strongly Agree",       # reverse
            "q18": "Strongly Disagree",    # direct
            "q19": "Strongly Disagree",    # direct
            "q20": "Strongly Agree",       # reverse
            "q21": "Strongly Agree",       # reverse
            "q22": "Strongly Agree",       # reverse
            "q23": "Strongly Disagree",    # direct
            "q24": "Strongly Disagree",    # direct
        }
        assert calculate_impulsivity(low_imp) == 1.0


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
