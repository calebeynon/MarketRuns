"""
Purpose: Validate .tex table structure for valence-only regression outputs
         (Appendix G). Tests column headers, variable presence by column,
         absence of discrete emotions, fit statistics, and row ordering.
Author: Claude Code
Date: 2026-03-09
"""

from pathlib import Path

import pytest

from tex_table_parser import (
    find_row_by_label,
    has_value,
    label_position,
    parse_coef_rows,
    parse_fit_rows,
)

# FILE PATHS
RO_LOGIT_TEX = Path("analysis/output/tables/ro_logit_selling_position_valence_only.tex")
COX_TEX = Path("analysis/output/tables/cox_survival_regression_valence_only.tex")
ORIG_COX_TEX = Path("analysis/output/tables/cox_survival_regression.tex")
ORIG_RO_TEX = Path("analysis/output/tables/ro_logit_selling_position.tex")

# Discrete emotions that should NOT appear in valence-only tables
RO_DISCRETE_EMOTIONS = [
    "Fear (p95)", "Anger (p95)", "Contempt (p95)", "Disgust (p95)",
    "Joy (p95)", "Sadness (p95)", "Surprise (p95)", "Engagement (p95)",
]
COX_DISCRETE_EMOTIONS = [
    "Fear", "Anger", "Contempt", "Disgust",
    "Joy", "Sadness", "Surprise", "Engagement",
]

# Variables expected in both tables
RO_TRAIT_LABELS = [
    "Extraversion", "Agreeableness", "Conscientiousness",
    "Neuroticism", "Openness", "Impulsivity",
    "State anxiety", "Risk tolerance",
]
COX_TRAIT_LABELS = RO_TRAIT_LABELS
COX_CASCADE_LABELS = ["1 prior sale", "2 prior sales", "3 prior sales"]
COX_CONTROL_LABELS = [
    "Signal", "Round", "Segment 2", "Segment 3", "Segment 4",
    "Treatment 2", "Age", "Female",
]
RO_FIT_LABELS = ["Observations", "Strata", "Concordance", "Log-lik."]
COX_FIT_LABELS = ["Observations", "Events", "Participants", "Log-likelihood"]


# =====
# Main function
# =====
def main():
    pytest.main([__file__, "-v"])


# =====
# Fixtures: Ranked-order logit (valence only)
# =====
@pytest.fixture(scope="module")
def ro_tex():
    if not RO_LOGIT_TEX.exists():
        pytest.skip(f"Table not found: {RO_LOGIT_TEX}")
    return RO_LOGIT_TEX.read_text()


@pytest.fixture(scope="module")
def ro_coefs(ro_tex):
    return parse_coef_rows(ro_tex, RO_FIT_LABELS)


@pytest.fixture(scope="module")
def ro_fits(ro_tex):
    return parse_fit_rows(ro_tex)


# =====
# Fixtures: Cox survival (valence only)
# =====
@pytest.fixture(scope="module")
def cox_tex():
    if not COX_TEX.exists():
        pytest.skip(f"Table not found: {COX_TEX}")
    return COX_TEX.read_text()


@pytest.fixture(scope="module")
def cox_coefs(cox_tex):
    return parse_coef_rows(cox_tex, COX_FIT_LABELS)


@pytest.fixture(scope="module")
def cox_fits(cox_tex):
    return parse_fit_rows(cox_tex)


# =====
# Tests: RO Logit basic structure
# =====
class TestRoLogitStructure:

    def test_tex_file_exists(self):
        assert RO_LOGIT_TEX.exists()

    def test_longtable_environment(self, ro_tex):
        assert "\\begin{longtable}" in ro_tex
        assert "\\end{longtable}" in ro_tex

    def test_label_unique(self, ro_tex):
        assert "\\label{tab:ro_logit_selling_position_valence_only}" in ro_tex

    def test_caption_mentions_valence_only(self, ro_tex):
        assert "valence only" in ro_tex.lower()

    def test_four_column_numbers(self, ro_tex):
        assert "& (1) & (2) & (3) & (4)" in ro_tex

    def test_full_sample_header(self, ro_tex):
        assert "Full Sample" in ro_tex

    def test_sellers_only_header(self, ro_tex):
        assert "Sellers Only" in ro_tex


# =====
# Tests: RO Logit discrete emotions absent
# =====
class TestRoLogitNoDiscreteEmotions:

    @pytest.mark.parametrize("label", RO_DISCRETE_EMOTIONS)
    def test_discrete_emotion_absent(self, ro_coefs, label):
        row = find_row_by_label(ro_coefs, label)
        assert row is None, f"'{label}' should NOT appear in valence-only table"


# =====
# Tests: RO Logit valence row present in all 4 columns
# =====
class TestRoLogitValenceRow:

    def test_valence_present(self, ro_coefs):
        vals = find_row_by_label(ro_coefs, "Valence (p95)")
        assert vals is not None, "Valence (p95) row not found"

    def test_valence_all_four_columns(self, ro_coefs):
        vals = find_row_by_label(ro_coefs, "Valence (p95)")
        for i, v in enumerate(vals):
            assert has_value(v), f"Valence (p95) col {i+1} should have a value"


# =====
# Tests: RO Logit trait rows blank in no-traits columns
# =====
class TestRoLogitTraitRows:

    @pytest.mark.parametrize("label", RO_TRAIT_LABELS)
    def test_trait_cols_2_4_populated(self, ro_coefs, label):
        vals = find_row_by_label(ro_coefs, label)
        assert vals is not None, f"Row '{label}' not found"
        assert has_value(vals[1]), f"'{label}' col 2 should have value"
        assert has_value(vals[3]), f"'{label}' col 4 should have value"

    @pytest.mark.parametrize("label", RO_TRAIT_LABELS)
    def test_trait_cols_1_3_blank(self, ro_coefs, label):
        vals = find_row_by_label(ro_coefs, label)
        assert vals is not None, f"Row '{label}' not found"
        assert not has_value(vals[0]), f"'{label}' col 1 should be blank"
        assert not has_value(vals[2]), f"'{label}' col 3 should be blank"


# =====
# Tests: RO Logit fit statistics
# =====
class TestRoLogitFitStats:

    @pytest.mark.parametrize("label", RO_FIT_LABELS)
    def test_fit_stat_present_with_four_values(self, ro_fits, label):
        vals = find_row_by_label(ro_fits, label)
        assert vals is not None, f"Fit statistic '{label}' not found"
        for i, v in enumerate(vals):
            assert has_value(v), f"'{label}' col {i+1} should have a value"

    def test_sample_sizes_match_original(self, ro_fits):
        """Valence-only should use same sample as original (2845/486)."""
        obs = find_row_by_label(ro_fits, "Observations")
        assert obs is not None
        assert "2,845" in obs[0], f"Full col 1 should be 2,845, got {obs[0]}"
        assert "486" in obs[2], f"Sellers col 3 should be 486, got {obs[2]}"


# =====
# Tests: Cox survival basic structure
# =====
class TestCoxStructure:

    def test_tex_file_exists(self):
        assert COX_TEX.exists()

    def test_longtable_environment(self, cox_tex):
        assert "\\begin{longtable}" in cox_tex
        assert "\\end{longtable}" in cox_tex

    def test_label_unique(self, cox_tex):
        assert "\\label{tab:cox_survival_regression_valence_only}" in cox_tex

    def test_caption_mentions_valence_only(self, cox_tex):
        assert "valence only" in cox_tex.lower()

    def test_four_column_numbers(self, cox_tex):
        assert "& (1) & (2) & (3) & (4)" in cox_tex

    def test_all_sellers_header(self, cox_tex):
        assert "All Sellers" in cox_tex

    def test_first_sellers_header(self, cox_tex):
        assert "First Sellers" in cox_tex


# =====
# Tests: Cox discrete emotions absent
# =====
class TestCoxNoDiscreteEmotions:

    @pytest.mark.parametrize("label", COX_DISCRETE_EMOTIONS)
    def test_discrete_emotion_absent(self, cox_coefs, label):
        row = find_row_by_label(cox_coefs, label)
        assert row is None, f"'{label}' should NOT appear in valence-only table"


# =====
# Tests: Cox valence row present
# =====
class TestCoxValenceRow:

    def test_valence_present(self, cox_coefs):
        vals = find_row_by_label(cox_coefs, "Valence")
        assert vals is not None, "Valence row not found"

    def test_valence_all_four_columns(self, cox_coefs):
        vals = find_row_by_label(cox_coefs, "Valence")
        for i, v in enumerate(vals):
            assert has_value(v), f"Valence col {i+1} should have a value"


# =====
# Tests: Cox cascade rows cols 1-2 populated, cols 3-4 blank
# =====
class TestCoxCascadeRows:

    @pytest.mark.parametrize("label", COX_CASCADE_LABELS)
    def test_cascade_cols_1_2_populated(self, cox_coefs, label):
        vals = find_row_by_label(cox_coefs, label)
        assert vals is not None, f"Row '{label}' not found"
        assert has_value(vals[0]), f"'{label}' col 1 should have value"
        assert has_value(vals[1]), f"'{label}' col 2 should have value"

    @pytest.mark.parametrize("label", COX_CASCADE_LABELS)
    def test_cascade_cols_3_4_blank(self, cox_coefs, label):
        vals = find_row_by_label(cox_coefs, label)
        assert vals is not None, f"Row '{label}' not found"
        assert not has_value(vals[2]), f"'{label}' col 3 should be blank"
        assert not has_value(vals[3]), f"'{label}' col 4 should be blank"


# =====
# Tests: Cox control rows all 4 columns populated
# =====
class TestCoxControlRows:

    @pytest.mark.parametrize("label", COX_CONTROL_LABELS)
    def test_control_all_four_columns(self, cox_coefs, label):
        vals = find_row_by_label(cox_coefs, label)
        assert vals is not None, f"Row '{label}' not found"
        for i, v in enumerate(vals):
            assert has_value(v), f"'{label}' col {i+1} should have a value"


# =====
# Tests: Cox trait rows blank in no-traits columns
# =====
class TestCoxTraitRows:

    @pytest.mark.parametrize("label", COX_TRAIT_LABELS)
    def test_trait_cols_2_4_populated(self, cox_coefs, label):
        vals = find_row_by_label(cox_coefs, label)
        assert vals is not None, f"Row '{label}' not found"
        assert has_value(vals[1]), f"'{label}' col 2 should have value"
        assert has_value(vals[3]), f"'{label}' col 4 should have value"

    @pytest.mark.parametrize("label", COX_TRAIT_LABELS)
    def test_trait_cols_1_3_blank(self, cox_coefs, label):
        vals = find_row_by_label(cox_coefs, label)
        assert vals is not None, f"Row '{label}' not found"
        assert not has_value(vals[0]), f"'{label}' col 1 should be blank"
        assert not has_value(vals[2]), f"'{label}' col 3 should be blank"


# =====
# Tests: Cox fit statistics
# =====
class TestCoxFitStats:

    @pytest.mark.parametrize("label", COX_FIT_LABELS)
    def test_fit_stat_present_with_four_values(self, cox_fits, label):
        vals = find_row_by_label(cox_fits, label)
        assert vals is not None, f"Fit statistic '{label}' not found"
        for i, v in enumerate(vals):
            assert has_value(v), f"'{label}' col {i+1} should have a value"

    def test_sample_sizes_match_original(self, cox_fits):
        """Valence-only should use same sample as original (13,590/1,183)."""
        obs = find_row_by_label(cox_fits, "Observations")
        assert obs is not None
        assert "13,590" in obs[0], f"All Sellers col 1 should be 13,590"
        assert "1,183" in obs[2], f"First Sellers col 3 should be 1,183"

    def test_events_match_original(self, cox_fits):
        """Event counts should match original table."""
        evts = find_row_by_label(cox_fits, "Events")
        assert evts is not None
        assert "659" in evts[0], f"All Sellers events should be 659"
        assert "467" in evts[2], f"First Sellers events should be 467"


# =====
# Tests: Cox row ordering
# =====
class TestCoxRowOrdering:

    def test_cascade_before_signal(self, cox_coefs):
        last_cascade = label_position(cox_coefs, COX_CASCADE_LABELS[-1])
        signal_pos = label_position(cox_coefs, "Signal")
        assert last_cascade < signal_pos, (
            "Cascade rows should appear before Signal"
        )

    def test_valence_before_controls(self, cox_coefs):
        valence_pos = label_position(cox_coefs, "Valence")
        round_pos = label_position(cox_coefs, "Round")
        assert valence_pos < round_pos, (
            "Valence should appear before Round"
        )

    def test_controls_before_traits(self, cox_coefs):
        female_pos = label_position(cox_coefs, "Female")
        first_trait = label_position(cox_coefs, COX_TRAIT_LABELS[0])
        assert female_pos < first_trait, (
            "Control rows should appear before trait rows"
        )


# =====
# Tests: Cross-check with original tables
# =====
class TestCrossCheckOriginals:

    def test_original_cox_still_has_discrete_emotions(self):
        """Original Cox table should still have all discrete emotions."""
        if not ORIG_COX_TEX.exists():
            pytest.skip("Original Cox table not found")
        tex = ORIG_COX_TEX.read_text()
        coefs = parse_coef_rows(tex, COX_FIT_LABELS)
        for label in COX_DISCRETE_EMOTIONS:
            assert find_row_by_label(coefs, label) is not None, (
                f"Original table should still have '{label}'"
            )

    def test_original_ro_still_has_discrete_emotions(self):
        """Original RO logit table should still have all discrete emotions."""
        if not ORIG_RO_TEX.exists():
            pytest.skip("Original RO logit table not found")
        tex = ORIG_RO_TEX.read_text()
        coefs = parse_coef_rows(tex, RO_FIT_LABELS)
        for label in RO_DISCRETE_EMOTIONS:
            assert find_row_by_label(coefs, label) is not None, (
                f"Original table should still have '{label}'"
            )


# %%
if __name__ == "__main__":
    main()
