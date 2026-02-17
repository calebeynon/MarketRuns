"""
Purpose: Unit tests for visualize_first_seller_traits.R
Author: Claude Code
Date: 2026-02-16
"""

import subprocess
import pytest
import pandas as pd
from pathlib import Path

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = PROJECT_ROOT / "datastore" / "derived" / "first_seller_analysis_data.csv"
R_SCRIPT = PROJECT_ROOT / "analysis" / "analysis" / "visualize_first_seller_traits.R"
OUTPUT_BOXPLOT = PROJECT_ROOT / "analysis" / "output" / "plots" / "first_seller_trait_boxplots.pdf"

TRAITS = [
    "extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness", "impulsivity", "state_anxiety",
]
BFI_IMPULSIVITY_TRAITS = [
    "extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness", "impulsivity",
]


# =====
# Main function
# =====
def main():
    """Run all tests via pytest."""
    pytest.main([__file__, "-v"])


# =====
# Input data tests
# =====
def test_input_data_exists():
    """Input CSV for visualization script exists."""
    assert INPUT_CSV.exists(), f"Input file not found: {INPUT_CSV}"


def test_input_has_all_traits():
    """Input data contains all 7 trait columns."""
    df = pd.read_csv(INPUT_CSV)
    for trait in TRAITS:
        assert trait in df.columns, f"Missing trait column: {trait}"


def test_bfi_impulsivity_range():
    """BFI and impulsivity values are within [1, 7]."""
    df = pd.read_csv(INPUT_CSV)
    for trait in BFI_IMPULSIVITY_TRAITS:
        assert df[trait].min() >= 1.0, f"{trait} min below 1"
        assert df[trait].max() <= 7.0, f"{trait} max above 7"


def test_state_anxiety_range():
    """State anxiety values are within [1, 4]."""
    df = pd.read_csv(INPUT_CSV)
    assert df["state_anxiety"].min() >= 1.0, "state_anxiety min below 1"
    assert df["state_anxiety"].max() <= 4.0, "state_anxiety max above 4"


def test_trait_count():
    """Exactly 7 trait columns are present."""
    df = pd.read_csv(INPUT_CSV)
    present = [t for t in TRAITS if t in df.columns]
    assert len(present) == 7, f"Expected 7 traits, found {len(present)}"


def test_first_seller_label_values():
    """is_first_seller column is binary (0 or 1)."""
    df = pd.read_csv(INPUT_CSV)
    assert "is_first_seller" in df.columns, "Missing is_first_seller column"
    unique_vals = set(df["is_first_seller"].unique())
    assert unique_vals == {0, 1}, f"Expected {{0, 1}}, got {unique_vals}"


# =====
# R script execution test
# =====
def test_r_script_runs_without_error():
    """R visualization script runs without error."""
    result = subprocess.run(
        ["Rscript", str(R_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, (
        f"R script failed with stderr:\n{result.stderr}"
    )


# =====
# Output file test
# =====
def test_output_boxplot_exists():
    """Output boxplot PDF exists and has non-zero size."""
    assert OUTPUT_BOXPLOT.exists(), f"Boxplot not found: {OUTPUT_BOXPLOT}"
    assert OUTPUT_BOXPLOT.stat().st_size > 0, "Boxplot file is empty"


# %%
if __name__ == "__main__":
    main()
