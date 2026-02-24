"""
Purpose: Validate Table 7 (.tex) structure for Cox survival regression output.
         Tests column headers, variable presence by column, fit statistics,
         and row ordering.
Author: Claude Code
Date: 2026-02-24
"""

import re
from pathlib import Path

import pytest

# FILE PATHS
TEX_PATH = Path("analysis/output/tables/cox_survival_regression.tex")

# Expected variable labels in order (as they appear in the .tex)
CASCADE_LABELS = ["1 prior sale", "2 prior sales", "3 prior sales"]
INTERACTION_LABELS = [
    r"1 prior $\times$ 1 prev.",
    r"2 prior $\times$ 1 prev.",
    r"2 prior $\times$ 2 prev.",
    r"3 prior $\times$ 1 prev.",
    r"3 prior $\times$ 2 prev.",
    r"3 prior $\times$ 3 prev.",
]
EMOTION_LABELS = [
    "Fear", "Anger", "Contempt", "Disgust",
    "Joy", "Sadness", "Surprise", "Engagement",
]
VALENCE_LABEL = "Valence"
CONTROL_LABELS = [
    "Signal", "Round", "Segment 2", "Segment 3", "Segment 4",
    "Treatment 2", "Age", "Female",
]
TRAIT_LABELS = [
    "State anxiety", "Impulsivity", "Risk tolerance",
    "Extraversion", "Agreeableness", "Neuroticism",
    "Openness", "Conscientiousness",
]
FIT_STAT_LABELS = [
    "Observations", "Events", "Participants", "Log-likelihood",
]


# =====
# Main function
# =====
def main():
    pytest.main([__file__, "-v"])


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def tex():
    if not TEX_PATH.exists():
        pytest.skip(f"Table not found: {TEX_PATH}")
    return TEX_PATH.read_text()


@pytest.fixture(scope="module")
def coef_rows(tex):
    """Extract coefficient rows as list of (label, [col1..col4])."""
    return parse_coef_rows(tex)


@pytest.fixture(scope="module")
def fit_rows(tex):
    """Extract fit statistic rows as list of (label, [col1..col4])."""
    return parse_fit_rows(tex)


# =====
# Parsing helpers
# =====
def parse_coef_rows(tex):
    """Parse coefficient rows with their 4 column values.

    Returns list of (label, [val1, val2, val3, val4]).
    """
    body = extract_table_body(tex)
    rows = []
    for line in body.split("\n"):
        parsed = parse_single_line(line)
        if parsed is not None:
            rows.append(parsed)
    return rows


def parse_single_line(line):
    """Parse one line into (label, values) or None if not a coef row."""
    line = line.strip().rstrip("\\\\").strip()
    if not line or line.startswith("\\") or line.startswith("\\emph"):
        return None
    cells = [c.strip() for c in line.split("&")]
    if len(cells) < 5:
        return None
    label = cells[0].strip()
    if not label or label.startswith("(") or label in FIT_STAT_LABELS:
        return None
    return (label, [c.strip() for c in cells[1:5]])


def extract_table_body(tex):
    """Extract content between first \\endlastfoot and \\midrule before fit."""
    start = tex.find("\\endlastfoot")
    if start == -1:
        return ""
    fit_marker = tex.find("\\emph{Fit statistics}", start)
    if fit_marker == -1:
        return tex[start:]
    return tex[start:fit_marker]


def parse_fit_rows(tex):
    """Parse fit statistic rows into list of (label, [val1..val4])."""
    match = re.search(
        r"\\emph\{Fit statistics\}.*?\n(.+?)\\midrule\s*\\midrule",
        tex, re.DOTALL,
    )
    if not match:
        return []
    rows = []
    for line in match.group(1).strip().split("\n"):
        line = line.strip().rstrip("\\\\").strip()
        if not line or line.startswith("\\"):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) >= 5:
            rows.append((cells[0].strip(), cells[1:5]))
    return rows


def find_row_by_label(coef_rows, label):
    """Find a coefficient row by its label text."""
    for row_label, vals in coef_rows:
        if row_label == label:
            return vals
    return None


def has_value(cell):
    """Check if a cell contains a numeric coefficient value."""
    return bool(re.search(r"\d", cell))


def label_position(coef_rows, label):
    """Return the index position of a label in the row list."""
    for i, (row_label, _) in enumerate(coef_rows):
        if row_label == label:
            return i
    return -1


# =====
# Tests: File existence and basic structure
# =====
class TestBasicStructure:

    def test_tex_file_exists(self):
        assert TEX_PATH.exists(), f"Missing: {TEX_PATH}"

    def test_longtable_environment(self, tex):
        assert "\\begin{longtable}" in tex
        assert "\\end{longtable}" in tex

    def test_booktabs_rules(self, tex):
        assert "\\midrule" in tex


# =====
# Tests: Column header structure
# =====
class TestColumnHeaders:

    def test_all_sellers_group_header(self, tex):
        assert "All Sellers" in tex

    def test_first_sellers_group_header(self, tex):
        assert "First Sellers" in tex

    def test_no_traits_sub_header(self, tex):
        assert "No Traits" in tex

    def test_with_traits_sub_header(self, tex):
        assert "With Traits" in tex

    def test_four_column_numbers(self, tex):
        assert "& (1) & (2) & (3) & (4)" in tex


# =====
# Tests: Emotion rows — all 4 columns have values
# =====
class TestEmotionRows:

    @pytest.mark.parametrize("label", EMOTION_LABELS)
    def test_emotion_all_four_columns(self, coef_rows, label):
        vals = find_row_by_label(coef_rows, label)
        assert vals is not None, f"Row '{label}' not found"
        for i, v in enumerate(vals):
            assert has_value(v), (
                f"'{label}' col {i+1} should have a value"
            )


# =====
# Tests: Valence row — all 4 columns have values
# =====
class TestValenceRow:

    def test_valence_all_four_columns(self, coef_rows):
        vals = find_row_by_label(coef_rows, VALENCE_LABEL)
        assert vals is not None, "Valence row not found"
        for i, v in enumerate(vals):
            assert has_value(v), (
                f"Valence col {i+1} should have a value"
            )


# =====
# Tests: Cascade rows — cols 1-2 have values, cols 3-4 blank
# =====
class TestCascadeRows:

    @pytest.mark.parametrize("label", CASCADE_LABELS)
    def test_cascade_cols_1_2_populated(self, coef_rows, label):
        vals = find_row_by_label(coef_rows, label)
        assert vals is not None, f"Row '{label}' not found"
        assert has_value(vals[0]), f"'{label}' col 1 should have value"
        assert has_value(vals[1]), f"'{label}' col 2 should have value"

    @pytest.mark.parametrize("label", CASCADE_LABELS)
    def test_cascade_cols_3_4_blank(self, coef_rows, label):
        vals = find_row_by_label(coef_rows, label)
        assert vals is not None, f"Row '{label}' not found"
        assert not has_value(vals[2]), (
            f"'{label}' col 3 should be blank"
        )
        assert not has_value(vals[3]), (
            f"'{label}' col 4 should be blank"
        )


# =====
# Tests: Interaction rows — cols 1-2 have values, cols 3-4 blank
# =====
class TestInteractionRows:

    @pytest.mark.parametrize("label", INTERACTION_LABELS)
    def test_interaction_cols_1_2_populated(self, coef_rows, label):
        vals = find_row_by_label(coef_rows, label)
        assert vals is not None, f"Row '{label}' not found"
        assert has_value(vals[0]), f"'{label}' col 1 should have value"
        assert has_value(vals[1]), f"'{label}' col 2 should have value"

    @pytest.mark.parametrize("label", INTERACTION_LABELS)
    def test_interaction_cols_3_4_blank(self, coef_rows, label):
        vals = find_row_by_label(coef_rows, label)
        assert vals is not None, f"Row '{label}' not found"
        assert not has_value(vals[2]), (
            f"'{label}' col 3 should be blank"
        )
        assert not has_value(vals[3]), (
            f"'{label}' col 4 should be blank"
        )


# =====
# Tests: Control rows — all 4 columns have values
# =====
class TestControlRows:

    @pytest.mark.parametrize("label", CONTROL_LABELS)
    def test_control_all_four_columns(self, coef_rows, label):
        vals = find_row_by_label(coef_rows, label)
        assert vals is not None, f"Row '{label}' not found"
        for i, v in enumerate(vals):
            assert has_value(v), (
                f"'{label}' col {i+1} should have a value"
            )


# =====
# Tests: Trait rows — cols 2 and 4 populated, cols 1 and 3 blank
# =====
class TestTraitRows:

    @pytest.mark.parametrize("label", TRAIT_LABELS)
    def test_trait_cols_2_4_populated(self, coef_rows, label):
        vals = find_row_by_label(coef_rows, label)
        assert vals is not None, f"Row '{label}' not found"
        assert has_value(vals[1]), f"'{label}' col 2 should have value"
        assert has_value(vals[3]), f"'{label}' col 4 should have value"

    @pytest.mark.parametrize("label", TRAIT_LABELS)
    def test_trait_cols_1_3_blank(self, coef_rows, label):
        vals = find_row_by_label(coef_rows, label)
        assert vals is not None, f"Row '{label}' not found"
        assert not has_value(vals[0]), (
            f"'{label}' col 1 should be blank"
        )
        assert not has_value(vals[2]), (
            f"'{label}' col 3 should be blank"
        )


# =====
# Tests: Fit statistics — present with 4 numeric values
# =====
class TestFitStatistics:

    @pytest.mark.parametrize("label", FIT_STAT_LABELS)
    def test_fit_stat_present(self, fit_rows, label):
        vals = find_row_by_label(fit_rows, label)
        assert vals is not None, f"Fit statistic '{label}' not found"

    @pytest.mark.parametrize("label", FIT_STAT_LABELS)
    def test_fit_stat_four_values(self, fit_rows, label):
        vals = find_row_by_label(fit_rows, label)
        assert vals is not None, f"Fit statistic '{label}' not found"
        for i, v in enumerate(vals):
            assert has_value(v), (
                f"'{label}' col {i+1} should have a numeric value"
            )


# =====
# Tests: Row ordering
# =====
class TestRowOrdering:

    def test_cascade_before_interactions(self, coef_rows):
        last_cascade = label_position(coef_rows, CASCADE_LABELS[-1])
        first_interaction = label_position(
            coef_rows, INTERACTION_LABELS[0]
        )
        assert last_cascade < first_interaction, (
            "Cascade rows should appear before interaction rows"
        )

    def test_interactions_before_emotions(self, coef_rows):
        last_interaction = label_position(
            coef_rows, INTERACTION_LABELS[-1]
        )
        first_emotion = label_position(coef_rows, EMOTION_LABELS[0])
        assert last_interaction < first_emotion, (
            "Interaction rows should appear before emotion rows"
        )

    def test_emotions_before_valence(self, coef_rows):
        last_emotion = label_position(coef_rows, EMOTION_LABELS[-1])
        valence_pos = label_position(coef_rows, VALENCE_LABEL)
        assert last_emotion < valence_pos, (
            "Emotion rows should appear before Valence"
        )

    def test_valence_before_controls(self, coef_rows):
        valence_pos = label_position(coef_rows, VALENCE_LABEL)
        first_control = label_position(coef_rows, CONTROL_LABELS[0])
        assert valence_pos < first_control, (
            "Valence should appear before control rows"
        )

    def test_controls_before_traits(self, coef_rows):
        last_control = label_position(coef_rows, CONTROL_LABELS[-1])
        first_trait = label_position(coef_rows, TRAIT_LABELS[0])
        assert last_control < first_trait, (
            "Control rows should appear before trait rows"
        )


# %%
if __name__ == "__main__":
    main()
