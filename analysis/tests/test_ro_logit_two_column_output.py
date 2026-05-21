"""
Purpose: Validate input data and LaTeX output for the 2-column ranked-order
         logit in analysis/analysis/ro_logit_two_column.R (Selling Behavior
         Table 7). Grounds assertions in ordinal_selling_position.csv and the
         produced ro_logit_two_column.tex.

         Data-transform correctness (rank reversal, strata, z-scoring) is
         already covered by test_ro_logit_data_validation.py, which reads the
         same CSV; this module focuses on the 2-column output structure.
Author: Claude Code
Date: 2026-05-20
"""

import re
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = PROJECT_ROOT / "datastore" / "derived" / "ordinal_selling_position.csv"
OUTPUT_TEX = (
    PROJECT_ROOT / "analysis" / "output" / "tables" / "ro_logit_two_column.tex"
)

EMOTION_COLS = [
    "fear_p95", "anger_p95", "contempt_p95", "disgust_p95", "joy_p95",
    "sadness_p95", "surprise_p95", "engagement_p95", "valence_p95",
]
TRAIT_LABELS = [
    "Extraversion", "Agreeableness", "Conscientiousness",
    "Neuroticism", "Openness", "Impulsivity",
    "State anxiety", "Risk tolerance",
]
# Verified from the produced table (same sample as the 4-model RO logit):
EXPECTED_SELLERS_OBS = 486
EXPECTED_SELLERS_STRATA = 202
EXPECTED_ALL_OBS = 2845
EXPECTED_ALL_STRATA = 720

CSV_EXISTS = pytest.mark.skipif(
    not INPUT_CSV.exists(), reason="ordinal_selling_position.csv not built",
)
TEX_EXISTS = pytest.mark.skipif(
    not OUTPUT_TEX.exists(), reason="ro_logit_two_column.tex not produced",
)


# =====
# Helpers
# =====
def read_tex():
    """Read the produced LaTeX longtable text."""
    return OUTPUT_TEX.read_text()


def fit_row_numbers(label):
    """Pull the two integer values from a fit-statistics row by its label."""
    line = next(ln for ln in read_tex().splitlines() if label in ln)
    return [int(n.replace(",", "")) for n in re.findall(r"[\d,]+", line)]


# =====
# Category A: Input data integrity
# =====
@CSV_EXISTS
def test_required_columns_present():
    """All emotion, trait, and survival-key columns the script uses exist."""
    cols = set(pd.read_csv(INPUT_CSV, nrows=1).columns)
    needed = [
        "session_id", "segment", "group_id", "round", "sell_rank",
        "did_sell", "player_id", *EMOTION_COLS,
    ]
    missing = [c for c in needed if c not in cols]
    assert not missing, f"Missing required columns: {missing}"


@CSV_EXISTS
def test_did_sell_is_binary():
    """did_sell flags sellers as {0, 1}; drives the sellers-only subsample."""
    vals = set(pd.read_csv(INPUT_CSV)["did_sell"].dropna().unique())
    assert vals.issubset({0, 1}), f"did_sell non-binary: {vals}"


@CSV_EXISTS
def test_emotion_complete_row_count():
    """Emotion-complete sample is 2,845 rows (matches All Participants obs)."""
    df = pd.read_csv(INPUT_CSV).dropna(subset=EMOTION_COLS)
    assert len(df) == EXPECTED_ALL_OBS, f"got {len(df)}"


# =====
# Category B: Output structure
# =====
@TEX_EXISTS
def test_output_non_empty():
    """Produced .tex has content."""
    assert len(read_tex().strip()) > 0


@TEX_EXISTS
def test_two_model_columns():
    """longtable spec is l + 2 centered p-columns -> 2 model columns."""
    tex = read_tex()
    match = re.search(r"\\begin\{longtable\}\{l\*\{(\d+)\}", tex)
    assert match, "longtable column multiplier not found"
    assert int(match.group(1)) == 2, f"expected 2 model columns, got {match.group(1)}"


@TEX_EXISTS
def test_column_headers_are_sellers_and_all():
    """Headers label the two columns Sellers Only and All Participants."""
    tex = read_tex()
    assert "Sellers Only" in tex
    assert "All Participants" in tex


@TEX_EXISTS
def test_header_has_two_numbered_columns():
    """Numbered header runs (1)..(2); no third column."""
    tex = read_tex()
    assert "(1)" in tex and "(2)" in tex
    assert "(3)" not in tex, "unexpected third model column"


@TEX_EXISTS
def test_multicolumn_spans_are_three():
    """Footer note rows span 3 columns (label + 2 models)."""
    tex = read_tex()
    spans = set(re.findall(r"\\multicolumn\{(\d+)\}", tex))
    assert spans == {"3"}, f"unexpected multicolumn spans: {spans}"


@TEX_EXISTS
def test_emotion_and_trait_rows_present():
    """Every emotion and personality-trait label is a coefficient row."""
    tex = read_tex()
    emotion_labels = ["Fear (p95)", "Anger (p95)", "Valence (p95)"]
    missing = [lab for lab in emotion_labels + TRAIT_LABELS if lab not in tex]
    assert not missing, f"missing coefficient rows: {missing}"


# =====
# Category C: Output values match verified run
# =====
@TEX_EXISTS
def test_observations_row():
    """Observations: 486 (sellers) and 2,845 (all participants)."""
    assert fit_row_numbers("Observations") == [
        EXPECTED_SELLERS_OBS, EXPECTED_ALL_OBS,
    ]


@TEX_EXISTS
def test_strata_row():
    """Strata: 202 (sellers, >=2-seller restriction) and 720 (all)."""
    assert fit_row_numbers("Strata") == [
        EXPECTED_SELLERS_STRATA, EXPECTED_ALL_STRATA,
    ]


@TEX_EXISTS
def test_all_participants_obs_matches_emotion_complete_csv():
    """The All Participants column N equals the emotion-complete CSV rows."""
    obs = fit_row_numbers("Observations")[1]
    if INPUT_CSV.exists():
        n_csv = len(pd.read_csv(INPUT_CSV).dropna(subset=EMOTION_COLS))
        assert obs == n_csv, f"table obs {obs} != CSV emotion-complete {n_csv}"


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
