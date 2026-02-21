"""
Purpose: Cross-validate every number in trait_correlations.tex against
         independently computed Pearson correlations from survey_traits.csv
Author: Claude Code
Date: 2026-02-21
"""

import re
from pathlib import Path

import pandas as pd
import pytest
from scipy.stats import pearsonr

# =====
# File paths and constants
# =====
SURVEY_PATH = Path("datastore/derived/survey_traits.csv")
TEX_PATH = Path("analysis/output/tables/trait_correlations.tex")

TRAITS = [
    "extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness", "impulsivity", "state_anxiety",
    "risk_tolerance",
]


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def survey():
    return pd.read_csv(SURVEY_PATH)


@pytest.fixture(scope="module")
def tex():
    return TEX_PATH.read_text()


# =====
# LaTeX parsing helpers
# =====
def parse_correlation_rows(tex):
    """Extract data rows between midrule and bottomrule."""
    body = re.search(r"\\midrule(.+?)\\bottomrule", tex, re.DOTALL)
    assert body, "Could not find table body"
    rows = []
    for line in body.group(1).strip().split("\n"):
        line = line.strip().rstrip("\\").strip()
        if not line or line == "\\midrule":
            continue
        cells = [c.strip() for c in line.split("&")]
        rows.append(cells)
    return rows


def parse_corr_cell(cell):
    """Parse cell like '0.25**' or '-0.39***' into (float, stars_str)."""
    match = re.match(r"(-?[\d.]+)(\*{0,3})", cell.strip())
    assert match, f"Could not parse correlation cell: {cell}"
    return float(match.group(1)), match.group(2)


def _expected_stars(p_value):
    """Determine significance stars from a p-value."""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


# =====
# Tests
# =====
class TestTraitCorrelationTable:

    def test_table_exists(self):
        assert TEX_PATH.exists(), f"Missing: {TEX_PATH}"

    def test_row_count(self, tex):
        rows = parse_correlation_rows(tex)
        assert len(rows) == 8, f"Expected 8 data rows, got {len(rows)}"

    def test_diagonal_is_one(self, tex):
        rows = parse_correlation_rows(tex)
        for i, row in enumerate(rows):
            # Column 0 is the label; diagonal is at column i+1
            assert row[i + 1].strip() == "1", (
                f"Row {i} ({TRAITS[i]}): diagonal should be '1', "
                f"got '{row[i + 1].strip()}'"
            )

    def test_upper_triangle_empty(self, tex):
        rows = parse_correlation_rows(tex)
        for i, row in enumerate(rows):
            # Columns above the diagonal: j > i (1-indexed: j+1 > i+1)
            for j in range(i + 1, 8):
                cell = row[j + 1].strip()
                assert cell == "", (
                    f"Row {i}, col {j} should be empty, "
                    f"got '{cell}'"
                )

    def test_correlation_values(self, survey, tex):
        rows = parse_correlation_rows(tex)
        for i in range(1, 8):
            for j in range(i):
                corr, _ = parse_corr_cell(rows[i][j + 1])
                expected, _ = pearsonr(
                    survey[TRAITS[i]], survey[TRAITS[j]]
                )
                assert corr == pytest.approx(expected, abs=0.01), (
                    f"{TRAITS[i]} x {TRAITS[j]}: "
                    f"tex={corr}, computed={expected:.4f}"
                )

    def test_significance_stars(self, survey, tex):
        rows = parse_correlation_rows(tex)
        for i in range(1, 8):
            for j in range(i):
                _, actual_stars = parse_corr_cell(rows[i][j + 1])
                _, p_value = pearsonr(
                    survey[TRAITS[i]], survey[TRAITS[j]]
                )
                expected = _expected_stars(p_value)
                assert actual_stars == expected, (
                    f"{TRAITS[i]} x {TRAITS[j]}: "
                    f"stars='{actual_stars}', expected='{expected}' "
                    f"(p={p_value:.4f})"
                )

    def test_sample_size_in_footnote(self, survey, tex):
        expected_n = len(survey)
        assert f"N = {expected_n}" in tex, (
            f"Footnote should contain 'N = {expected_n}'"
        )
        assert expected_n == 95, (
            f"Expected 95 subjects, got {expected_n}"
        )


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
