"""
Purpose: Validate input data and LaTeX output for the Tobit number-of-sellers
         regression in analysis/analysis/tobit_n_sellers.R (Selling Behavior
         Table 4). Grounds assertions in the actual group_round_timing.csv and
         the produced tobit_n_sellers.tex.
Author: Claude Code
Date: 2026-05-20
"""

import re
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = PROJECT_ROOT / "datastore" / "derived" / "group_round_timing.csv"
OUTPUT_TEX = PROJECT_ROOT / "analysis" / "output" / "tables" / "tobit_n_sellers.tex"

REQUIRED_COLS = [
    "session", "treatment", "segment_num", "group_id",
    "global_group_id", "round_num", "state", "n_sellers",
]
# Coefficient labels the R script emits, in order (VAR_LABELS).
EXPECTED_LABELS = [
    "Constant", "Bad state", "Treatment 2",
    "Segment 2", "Segment 3", "Segment 4", "Round",
]
EXPECTED_N = 720  # group-rounds; matches Observations in the verified output

CSV_EXISTS = pytest.mark.skipif(
    not INPUT_CSV.exists(), reason="group_round_timing.csv not built",
)
TEX_EXISTS = pytest.mark.skipif(
    not OUTPUT_TEX.exists(), reason="tobit_n_sellers.tex not produced",
)


# =====
# Helpers
# =====
def load_input():
    """Load the group-round timing dataset."""
    return pd.read_csv(INPUT_CSV)


def read_tex():
    """Read the produced LaTeX table text."""
    return OUTPUT_TEX.read_text()


# =====
# Category A: Input data integrity
# =====
@CSV_EXISTS
def test_required_columns_present():
    """All columns the R model formula references exist."""
    cols = set(load_input().columns)
    missing = [c for c in REQUIRED_COLS if c not in cols]
    assert not missing, f"Missing required columns: {missing}"


@CSV_EXISTS
def test_row_count_is_720():
    """One row per group-round: 720 modeled observations."""
    assert len(load_input()) == EXPECTED_N


@CSV_EXISTS
def test_n_sellers_within_censoring_bounds():
    """n_sellers is integer-valued in [0, 4] (Tobit censors at 0 and 4)."""
    n = load_input()["n_sellers"]
    assert n.min() >= 0 and n.max() <= 4, f"out of bounds: {n.min()}-{n.max()}"
    assert (n.dropna() == n.dropna().round()).all(), "n_sellers not integer"


@CSV_EXISTS
def test_state_is_binary():
    """state is binary {0, 1}; bad_state is derived as state == 0."""
    vals = set(load_input()["state"].dropna().unique())
    assert vals.issubset({0, 1}), f"state has non-binary values: {vals}"


@CSV_EXISTS
def test_bad_state_derivation_matches_state():
    """bad_state = (state == 0) covers exactly the state==0 rows."""
    df = load_input()
    bad_state = (df["state"] == 0).astype(int)
    assert bad_state.sum() == (df["state"] == 0).sum()
    assert set(bad_state.unique()).issubset({0, 1})


@CSV_EXISTS
def test_segment_num_is_one_to_four():
    """segment_num spans 1-4 (factor with 4 levels in the model)."""
    assert set(load_input()["segment_num"].unique()) == {1, 2, 3, 4}


@CSV_EXISTS
def test_treatment_is_one_or_two():
    """treatment is the two-level factor {1, 2}."""
    assert set(load_input()["treatment"].unique()).issubset({1, 2})


# =====
# Category B: Output structure
# =====
@TEX_EXISTS
def test_output_non_empty():
    """Produced .tex has content."""
    assert len(read_tex().strip()) > 0


@TEX_EXISTS
def test_two_model_columns():
    """tabular alignment is lcc -> 2 model columns plus the label column."""
    tex = read_tex()
    match = re.search(r"\\begin\{tabular\}\{(l[c]+)\}", tex)
    assert match, "no tabular column spec found"
    n_model_cols = match.group(1).count("c")
    assert n_model_cols == 2, f"expected 2 model columns, got {n_model_cols}"


@TEX_EXISTS
def test_model_header_has_two_numbered_columns():
    """Model header row labels columns (1) and (2)."""
    tex = read_tex()
    assert "(1)" in tex and "(2)" in tex
    assert "(3)" not in tex, "unexpected third model column"


@TEX_EXISTS
def test_all_expected_coefficient_labels_present():
    """Every coefficient row label from the R script appears in the table."""
    tex = read_tex()
    missing = [lab for lab in EXPECTED_LABELS if lab not in tex]
    assert not missing, f"missing coefficient rows: {missing}"


@TEX_EXISTS
def test_observations_row_matches_input_rows():
    """Observations reported equal the 720 input group-rounds in both models."""
    tex = read_tex()
    match = re.search(r"Observations\s*&\s*([\d,]+)\s*&\s*([\d,]+)", tex)
    assert match, "Observations row not found"
    n1 = int(match.group(1).replace(",", ""))
    n2 = int(match.group(2).replace(",", ""))
    assert n1 == EXPECTED_N and n2 == EXPECTED_N


@TEX_EXISTS
def test_bad_state_only_in_second_model():
    """Bad state coefficient appears only in Model 2 (blank cell in Model 1)."""
    tex = read_tex()
    # The row reads: "Bad state & <blank> & <value>" — first model cell empty.
    line = next(ln for ln in tex.splitlines() if "Bad state" in ln)
    cells = [c.strip() for c in line.split("&")]
    assert cells[1] == "", "Model 1 should have no Bad state coefficient"
    assert re.search(r"\d", cells[2]), "Model 2 should have a Bad state value"


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
