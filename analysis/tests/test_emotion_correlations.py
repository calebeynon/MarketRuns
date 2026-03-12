"""
Purpose: Cross-validate every number in emotion_correlations.tex against
         independently computed Pearson correlations from imotions_period_emotions.csv.
         Validates table structure, correlation values, significance stars,
         data integrity, and LaTeX formatting.
Author: Claude Code
Date: 2026-03-10
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

# =====
# File paths and constants
# =====
EMOTIONS_PATH = Path("datastore/derived/imotions_period_emotions.csv")
TEX_PATH = Path("analysis/output/tables/emotion_correlations.tex")

EMOTIONS = [
    "anger_mean", "contempt_mean", "disgust_mean",
    "fear_mean", "joy_mean", "sadness_mean",
    "surprise_mean", "engagement_mean", "valence_mean",
]

NUM_EMOTIONS = 9
EXPECTED_N = 16101


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def emotions():
    return pd.read_csv(EMOTIONS_PATH)


@pytest.fixture(scope="module")
def tex():
    return TEX_PATH.read_text()


@pytest.fixture(scope="module")
def corr_matrix(emotions):
    """Pre-compute full correlation matrix with p-values."""
    matrix = {}
    for i in range(NUM_EMOTIONS):
        for j in range(NUM_EMOTIONS):
            if i == j:
                matrix[(i, j)] = (1.0, 0.0)
            else:
                r, p = pearsonr(
                    emotions[EMOTIONS[i]], emotions[EMOTIONS[j]]
                )
                matrix[(i, j)] = (r, p)
    return matrix


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
    """Parse cell like '0.25**' or '-0.39***' -> (float, str)."""
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
# Input data integrity tests
# =====
class TestInputDataIntegrity:
    """Validate the source CSV before testing the table."""

    def test_input_file_exists(self):
        assert EMOTIONS_PATH.exists(), f"Missing: {EMOTIONS_PATH}"

    def test_row_count_and_sessions(self, emotions):
        assert len(emotions) == EXPECTED_N
        assert emotions["session_id"].nunique() == 6

    def test_emotion_columns_complete(self, emotions):
        """All 9 emotion columns exist, are float64, and have no NaNs."""
        for col in EMOTIONS:
            assert col in emotions.columns, f"Missing column: {col}"
            assert emotions[col].dtype == np.float64
            assert emotions[col].isna().sum() == 0, f"{col} has NaN"

    def test_players_per_session(self, emotions):
        counts = emotions.groupby("session_id")["player"].nunique()
        for session, n in counts.items():
            assert n == 16, f"Session {session}: {n} players"

    def test_valence_can_be_negative(self, emotions):
        """Valence is the only emotion that takes negative values."""
        assert emotions["valence_mean"].min() < 0
        for col in [e for e in EMOTIONS if e != "valence_mean"]:
            assert emotions[col].min() >= 0, f"{col} has negatives"

    def test_emotions_within_expected_range(self, emotions):
        """All emotion values within [-100, 100]."""
        for col in EMOTIONS:
            assert emotions[col].min() >= -100
            assert emotions[col].max() <= 100


# =====
# Correlation computation validation
# =====
class TestCorrelationComputation:
    """Validate properties of independently computed correlations."""

    def test_bounds_and_symmetry(self, corr_matrix):
        """All correlations in [-1,1] and corr(X,Y)==corr(Y,X)."""
        for i in range(NUM_EMOTIONS):
            for j in range(i):
                r_ij = corr_matrix[(i, j)][0]
                r_ji = corr_matrix[(j, i)][0]
                assert -1 <= r_ij <= 1
                assert r_ij == pytest.approx(r_ji, abs=1e-10)

    def test_diagonal_is_exactly_one(self, corr_matrix):
        for i in range(NUM_EMOTIONS):
            assert corr_matrix[(i, i)][0] == 1.0

    def test_total_lower_triangle_pairs(self):
        n_pairs = sum(1 for i in range(NUM_EMOTIONS) for j in range(i))
        assert n_pairs == 36

    def test_known_correlations(self, emotions):
        """Spot-check known strong correlations from the data."""
        r, _ = pearsonr(emotions["fear_mean"], emotions["surprise_mean"])
        assert r > 0.9, f"fear x surprise: {r:.4f}"
        r, _ = pearsonr(emotions["valence_mean"], emotions["joy_mean"])
        assert r > 0.7, f"valence x joy: {r:.4f}"
        r, _ = pearsonr(emotions["valence_mean"], emotions["sadness_mean"])
        assert r < -0.3, f"valence x sadness: {r:.4f}"


# =====
# Table structure tests
# =====
class TestTableStructure:
    """Validate the LaTeX table format and layout."""

    def test_table_file_exists(self):
        assert TEX_PATH.exists(), f"Missing: {TEX_PATH}"

    def test_row_and_column_counts(self, tex):
        rows = parse_correlation_rows(tex)
        assert len(rows) == NUM_EMOTIONS
        for i, row in enumerate(rows):
            assert len(row) == NUM_EMOTIONS + 1, (
                f"Row {i}: {len(row)} cells, expected {NUM_EMOTIONS + 1}"
            )

    def test_diagonal_is_one(self, tex):
        rows = parse_correlation_rows(tex)
        for i, row in enumerate(rows):
            assert row[i + 1].strip() == "1", (
                f"Row {i}: diagonal='{row[i + 1].strip()}'"
            )

    def test_upper_triangle_empty(self, tex):
        rows = parse_correlation_rows(tex)
        for i, row in enumerate(rows):
            for j in range(i + 1, NUM_EMOTIONS):
                assert row[j + 1].strip() == "", (
                    f"({i},{j}) should be empty: '{row[j + 1]}'"
                )

    def test_lower_triangle_not_empty(self, tex):
        rows = parse_correlation_rows(tex)
        for i in range(1, NUM_EMOTIONS):
            for j in range(i):
                assert rows[i][j + 1].strip() != "", (
                    f"({i},{j}) lower triangle is empty"
                )

    def test_row_labels_numbered(self, tex):
        rows = parse_correlation_rows(tex)
        for i, row in enumerate(rows):
            assert row[0].strip().startswith(f"({i + 1})"), (
                f"Row {i}: '{row[0].strip()}'"
            )

    def test_column_headers_present(self, tex):
        for i in range(1, NUM_EMOTIONS + 1):
            assert f"({i})" in tex


# =====
# Correlation value cross-validation
# =====
class TestCorrelationValues:
    """Cross-validate every correlation in the table against raw data."""

    def test_all_lower_triangle_values(self, emotions, tex):
        """Every lower-triangle cell matches pearsonr within 0.01."""
        rows = parse_correlation_rows(tex)
        for i in range(1, NUM_EMOTIONS):
            for j in range(i):
                tex_corr, _ = parse_corr_cell(rows[i][j + 1])
                expected_r, _ = pearsonr(
                    emotions[EMOTIONS[i]], emotions[EMOTIONS[j]]
                )
                assert tex_corr == pytest.approx(expected_r, abs=0.01), (
                    f"{EMOTIONS[i]} x {EMOTIONS[j]}: "
                    f"tex={tex_corr}, computed={expected_r:.4f}"
                )

    def test_correlation_rounding_to_two_decimals(self, tex):
        rows = parse_correlation_rows(tex)
        for i in range(1, NUM_EMOTIONS):
            for j in range(i):
                num_match = re.match(
                    r"(-?[\d.]+)", rows[i][j + 1].strip()
                )
                assert num_match
                num_str = num_match.group(1)
                if "." in num_str:
                    decimals = len(num_str.split(".")[1])
                    assert decimals == 2, f"'{num_str}' not 2dp"

    def test_spot_checks_hardcoded(self, emotions, tex):
        """Hardcoded spot checks for specific known correlations."""
        rows = parse_correlation_rows(tex)
        checks = [
            # (row_i, col_j, emotion_i, emotion_j, expected_approx)
            (1, 1, "contempt_mean", "anger_mean", 0.08),
            (2, 1, "disgust_mean", "anger_mean", 0.55),
            (6, 4, "surprise_mean", "fear_mean", 0.94),
            (8, 5, "valence_mean", "joy_mean", 0.79),
            (8, 6, "valence_mean", "sadness_mean", -0.43),
            (8, 1, "valence_mean", "anger_mean", -0.23),
        ]
        for row_i, col_j, emo_i, emo_j, exp in checks:
            corr, _ = parse_corr_cell(rows[row_i][col_j])
            r, _ = pearsonr(emotions[emo_i], emotions[emo_j])
            assert corr == pytest.approx(exp, abs=0.01), (
                f"{emo_i} x {emo_j}: tex={corr}, expected~{exp}"
            )
            assert corr == pytest.approx(r, abs=0.01)

    def test_negative_correlations_have_minus_sign(self, emotions, tex):
        rows = parse_correlation_rows(tex)
        for i in range(1, NUM_EMOTIONS):
            for j in range(i):
                r, _ = pearsonr(
                    emotions[EMOTIONS[i]], emotions[EMOTIONS[j]]
                )
                cell = rows[i][j + 1].strip()
                if r < -0.005:
                    assert cell.startswith("-"), (
                        f"{EMOTIONS[i]} x {EMOTIONS[j]}: r={r:.4f} "
                        f"but cell '{cell}' has no minus"
                    )


# =====
# Significance star tests
# =====
class TestSignificanceStars:

    def test_all_stars_match(self, emotions, tex):
        """Every lower-triangle cell has correct significance stars."""
        rows = parse_correlation_rows(tex)
        for i in range(1, NUM_EMOTIONS):
            for j in range(i):
                _, actual_stars = parse_corr_cell(rows[i][j + 1])
                _, p = pearsonr(
                    emotions[EMOTIONS[i]], emotions[EMOTIONS[j]]
                )
                expected = _expected_stars(p)
                assert actual_stars == expected, (
                    f"{EMOTIONS[i]} x {EMOTIONS[j]}: "
                    f"'{actual_stars}' vs '{expected}' (p={p:.4e})"
                )

    def test_known_triple_star_pairs(self, tex):
        """Known strongly correlated pairs must have ***."""
        rows = parse_correlation_rows(tex)
        for i, j in [(2, 0), (6, 3), (8, 4)]:
            _, stars = parse_corr_cell(rows[i][j + 1])
            assert stars == "***", (
                f"{EMOTIONS[i]} x {EMOTIONS[j]}: '{stars}'"
            )

    def test_stars_format_valid(self, tex):
        """Star annotations contain only 0-3 asterisks."""
        rows = parse_correlation_rows(tex)
        for i in range(1, NUM_EMOTIONS):
            for j in range(i):
                cell = rows[i][j + 1].strip()
                stars = re.sub(r"-?[\d.]+", "", cell).strip()
                assert re.match(r"^\*{0,3}$", stars), (
                    f"Invalid stars '{stars}' in '{cell}'"
                )

    def test_significance_legend_present(self, tex):
        assert "p<0.01" in tex
        assert "p<0.05" in tex
        assert "p<0.1" in tex


# =====
# Footnote and formatting tests
# =====
class TestFootnoteAndFormatting:

    def test_sample_size_in_footnote(self, emotions, tex):
        """Footnote has correct N with LaTeX comma formatting."""
        assert len(emotions) == EXPECTED_N
        formatted = f"{EXPECTED_N:,}".replace(",", "{,}")
        assert f"N = {formatted}" in tex

    def test_pearson_mentioned(self, tex):
        assert "pearson" in tex.lower()

    def test_lower_triangle_mentioned(self, tex):
        assert "lower triangle" in tex.lower()

    def test_booktabs_rules(self, tex):
        assert "\\toprule" in tex
        assert "\\bottomrule" in tex
        assert "\\midrule" in tex

    def test_tabular_column_count(self, tex):
        """Tabular spec has label + 9 data columns = 10."""
        match = re.search(r"\\begin\{tabular\}\{([^}]+)\}", tex)
        assert match
        col_count = len(re.findall(r"[lcr]", match.group(1)))
        assert col_count == NUM_EMOTIONS + 1


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
