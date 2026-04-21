"""
Purpose: Validate structure of the 500ms reactive-sellers Cox survival table
         (issue #118). Confirms the .tex file exists, contains all 9 emotion
         rows, the correct label, and that the respec-dropped covariates
         (cascade dummies + cumulative-by-previous interactions) are absent.
Author: Claude Code
Date: 2026-04-21
"""

from pathlib import Path

import pytest

# FILE PATHS
REPO_ROOT = Path(__file__).resolve().parents[2]
TEX_PATH = REPO_ROOT / "analysis" / "output" / "tables" / "cox_survival_reactive_500ms.tex"

EMOTION_LABELS = [
    "Fear", "Anger", "Contempt", "Disgust",
    "Joy", "Sadness", "Surprise", "Engagement", "Valence",
]

DROPPED_COVARIATES = [
    "dummy_1_cum", "dummy_2_cum", "dummy_3_cum",
    "int_1_1", "int_2_1", "int_2_2",
    "int_3_1", "int_3_2", "int_3_3",
]

FIT_STAT_LABELS = ["Observations", "Events", "Log-likelihood"]


# =====
# Main
# =====
def main():
    """Run all tests with verbose output."""
    pytest.main([__file__, "-v"])


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def tex():
    """Load the .tex file once for all tests."""
    if not TEX_PATH.exists():
        pytest.fail(f"Expected output table missing: {TEX_PATH}")
    return TEX_PATH.read_text()


# =====
# Structural tests
# =====
def test_tex_file_exists():
    """The 500ms reactive Cox table must be written to analysis/output/tables/."""
    assert TEX_PATH.exists(), f"Missing: {TEX_PATH}"


def test_longtable_environment(tex):
    """Table must be wrapped in longtable for multi-page safety."""
    assert "\\begin{longtable}" in tex
    assert "\\end{longtable}" in tex


def test_caption_and_label(tex):
    """Caption mentions 500ms pre-click window; label is tab:cox_survival_reactive_500ms."""
    assert "500ms" in tex
    assert "\\label{tab:cox_survival_reactive_500ms}" in tex


# =====
# Content coverage
# =====
@pytest.mark.parametrize("emotion", EMOTION_LABELS)
def test_emotion_row_present(tex, emotion):
    """Each of the 9 emotion rows (incl. Valence) must appear in the coefficient block."""
    assert emotion in tex, f"Emotion label '{emotion}' missing from table"


@pytest.mark.parametrize("dropped", DROPPED_COVARIATES)
def test_dropped_covariate_absent(tex, dropped):
    """Respec removed cascade dummies and cumulative interactions — they must NOT appear."""
    assert dropped not in tex, (
        f"Dropped covariate '{dropped}' still appears in table — respec violated"
    )


@pytest.mark.parametrize("label", FIT_STAT_LABELS)
def test_fit_statistic_present(tex, label):
    """Fit rows (Observations, Events, Log-likelihood) must appear at the bottom."""
    assert label in tex, f"Fit statistic '{label}' missing from table"


# %%
if __name__ == "__main__":
    main()
