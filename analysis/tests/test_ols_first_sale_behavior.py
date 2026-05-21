"""
Purpose: Validate input data, the bad_state merge, and LaTeX output for the
         pooled OLS first-sale behavior models in
         analysis/analysis/ols_first_sale_behavior.R (Selling Behavior Table 5).
         Grounds assertions in first_sale_data.csv, group_round_timing.csv, and
         the produced ols_first_sale_behavior.tex.
Author: Claude Code
Date: 2026-05-20
"""

import re
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIRST_SALE_CSV = PROJECT_ROOT / "datastore" / "derived" / "first_sale_data.csv"
TIMING_CSV = PROJECT_ROOT / "datastore" / "derived" / "group_round_timing.csv"
OUTPUT_TEX = (
    PROJECT_ROOT / "analysis" / "output" / "tables"
    / "ols_first_sale_behavior.tex"
)

MERGE_KEYS = ["session", "segment_num", "group_id", "round_num"]
FIRST_SALE_COLS = [
    *MERGE_KEYS, "treatment", "global_group_id",
    "first_sale_period", "signal_at_first_sale",
]
EXPECTED_TOTAL = 720   # one row per group-round before restriction
EXPECTED_SALES = 382   # group-rounds with a first sale; matches Observations

CSV_EXISTS = pytest.mark.skipif(
    not (FIRST_SALE_CSV.exists() and TIMING_CSV.exists()),
    reason="input CSVs not built",
)
TEX_EXISTS = pytest.mark.skipif(
    not OUTPUT_TEX.exists(), reason="ols_first_sale_behavior.tex not produced",
)


# =====
# Helpers
# =====
def load_first_sale():
    """Load the first-sale dataset."""
    return pd.read_csv(FIRST_SALE_CSV)


def merge_bad_state():
    """Replicate the R merge_bad_state(): left join state, derive bad_state."""
    fs = load_first_sale()
    timing = pd.read_csv(TIMING_CSV)
    timing["bad_state"] = (timing["state"] == 0).astype(int)
    return fs.merge(
        timing[MERGE_KEYS + ["bad_state"]], on=MERGE_KEYS, how="left",
    )


def read_tex():
    """Read the produced LaTeX table text."""
    return OUTPUT_TEX.read_text()


# =====
# Category A: Input data integrity
# =====
@CSV_EXISTS
def test_first_sale_columns_present():
    """All columns the R model and merge reference exist."""
    cols = set(load_first_sale().columns)
    missing = [c for c in FIRST_SALE_COLS if c not in cols]
    assert not missing, f"Missing required columns: {missing}"


@CSV_EXISTS
def test_first_sale_row_count():
    """One row per group-round before the sales restriction."""
    assert len(load_first_sale()) == EXPECTED_TOTAL


@CSV_EXISTS
def test_signal_at_first_sale_in_unit_interval():
    """Belief at first sale is a probability in [0, 1] when present."""
    sig = load_first_sale()["signal_at_first_sale"].dropna()
    assert sig.min() >= 0 and sig.max() <= 1, f"out of [0,1]: {sig.min()}-{sig.max()}"


@CSV_EXISTS
def test_first_sale_period_within_round_length():
    """first_sale_period is a positive period index within trading rounds."""
    per = load_first_sale()["first_sale_period"].dropna()
    assert per.min() >= 1 and per.max() <= 14, f"out of range: {per.min()}-{per.max()}"


@CSV_EXISTS
def test_merge_keys_unique_in_both_tables():
    """Merge keys identify a single row in each table (1:1 join)."""
    fs = load_first_sale()
    timing = pd.read_csv(TIMING_CSV)
    assert fs[MERGE_KEYS].drop_duplicates().shape[0] == len(fs)
    assert timing[MERGE_KEYS].drop_duplicates().shape[0] == len(timing)


# =====
# Category B: bad_state merge logic
# =====
@CSV_EXISTS
def test_merge_preserves_row_count():
    """The bad_state left join does not change the first-sale row count."""
    before = len(load_first_sale())
    after = len(merge_bad_state())
    assert after == before, f"merge changed rows: {before} -> {after}"


@CSV_EXISTS
def test_merge_introduces_no_na_on_key():
    """Every first-sale key matches a timing key — no NA bad_state."""
    merged = merge_bad_state()
    n_na = merged["bad_state"].isna().sum()
    assert n_na == 0, f"merge produced {n_na} NA bad_state values"


@CSV_EXISTS
def test_bad_state_is_binary_after_merge():
    """Merged bad_state is binary {0, 1}."""
    assert set(merge_bad_state()["bad_state"].unique()).issubset({0, 1})


# =====
# Category C: Sales restriction
# =====
@CSV_EXISTS
def test_restrict_to_sales_count():
    """Rows with non-NA signal_at_first_sale equal the model Observations."""
    merged = merge_bad_state()
    n_sales = merged["signal_at_first_sale"].notna().sum()
    assert n_sales == EXPECTED_SALES, f"expected {EXPECTED_SALES}, got {n_sales}"


@CSV_EXISTS
def test_no_signal_implies_no_first_period():
    """Group-rounds without a sale have neither belief nor period recorded."""
    fs = load_first_sale()
    no_sale = fs[fs["signal_at_first_sale"].isna()]
    assert no_sale["first_sale_period"].isna().all(), (
        "found a first_sale_period without a signal_at_first_sale"
    )


# =====
# Category D: Output structure
# =====
@TEX_EXISTS
def test_output_non_empty():
    """Produced .tex has content."""
    assert len(read_tex().strip()) > 0


@TEX_EXISTS
def test_four_model_columns():
    """tabular alignment is lcccc -> 4 model columns plus the label column."""
    tex = read_tex()
    match = re.search(r"\\begin\{tabular\}\{(l[c]+)\}", tex)
    assert match, "no tabular column spec found"
    n_model_cols = match.group(1).count("c")
    assert n_model_cols == 4, f"expected 4 model columns, got {n_model_cols}"


@TEX_EXISTS
def test_model_header_has_four_numbered_columns():
    """Model header row labels columns (1) through (4)."""
    tex = read_tex()
    for col in ["(1)", "(2)", "(3)", "(4)"]:
        assert col in tex, f"missing model column header {col}"
    assert "(5)" not in tex, "unexpected fifth model column"


@TEX_EXISTS
def test_both_dependent_variable_groupings_present():
    """Header spans both outcomes: belief and period of first sale."""
    tex = read_tex()
    assert "Belief at first sale" in tex
    assert "Period of first sale" in tex


@TEX_EXISTS
def test_observations_match_sales_count():
    """Each model reports Observations == 382 sales."""
    tex = read_tex()
    line = next(ln for ln in tex.splitlines() if "Observations" in ln)
    nums = [int(n.replace(",", "")) for n in re.findall(r"[\d,]+", line)]
    assert nums == [EXPECTED_SALES] * 4, f"Observations row: {nums}"


@TEX_EXISTS
def test_bad_state_only_in_models_two_and_four():
    """bad_state appears in Models 2 and 4 only; cells 1 and 3 are blank."""
    tex = read_tex()
    line = next(ln for ln in tex.splitlines() if "bad" in ln.lower()
                and "state" in ln.lower() and "&" in ln)
    cells = [c.strip() for c in line.split("&")]
    # cells: [label, m1, m2, m3, m4]
    assert cells[1] == "" and cells[3] == "", "Models 1 and 3 must omit bad_state"
    assert re.search(r"\d", cells[2]) and re.search(r"\d", cells[4]), (
        "Models 2 and 4 must report a bad_state coefficient"
    )


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
