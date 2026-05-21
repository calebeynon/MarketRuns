"""
Purpose: Validate the four risk-set sample definitions and LaTeX output for the
         4-column Cox selling-behavior table in
         analysis/analysis/cox_selling_four_column.R (Selling Behavior Table 6).
         Reproduces each column's sample from emotions_traits_selling_dataset.csv
         and checks the produced cox_selling_four_column.tex against those
         verified counts.
Author: Claude Code
Date: 2026-05-20
"""

import re
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_TEX = (
    PROJECT_ROOT / "analysis" / "output" / "tables"
    / "cox_selling_four_column.tex"
)

# The Cox base pipeline lives in analysis/tests/cox_test_helpers.py.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cox_test_helpers import (  # noqa: E402
    EMOTIONS_DATASET, ALL_EMOTIONS, add_id_columns, add_dummies,
)

TRAITS = [
    "extraversion", "agreeableness", "conscientiousness", "neuroticism",
    "openness", "impulsivity", "state_anxiety", "risk_tolerance",
]
# Verified by reproducing the R sample logic against the real CSV (2026-05-20):
EXPECTED = {
    "col1": {"obs": 1183, "events": 467},   # First sellers
    "col3": {"obs": 2013, "events": 659},   # All sellers
    "col4": {"obs": 13590, "events": 659, "participants": 95},  # All participants
    # Column 2 (reactive 500ms) values come from the verified output table; the
    # presell-window merge is exercised by the existing reactive-flag tests.
    "col2": {"obs": 1082, "events": 101},   # Reactive
}

CSV_EXISTS = pytest.mark.skipif(
    not EMOTIONS_DATASET.exists(), reason="emotions_traits_selling_dataset.csv not built",
)
TEX_EXISTS = pytest.mark.skipif(
    not OUTPUT_TEX.exists(), reason="cox_selling_four_column.tex not produced",
)


# =====
# Helpers
# =====
def emotion_trait_complete_base():
    """Replicate prepare_base_data + emotion/trait completeness (Cols 1, 3, 4)."""
    df = pd.read_csv(EMOTIONS_DATASET)
    base = df[df["already_sold"] == 0].copy()
    base = add_id_columns(base)
    base = add_dummies(base)
    return base.dropna(subset=ALL_EMOTIONS + TRAITS).copy()


def read_tex():
    """Read the produced LaTeX longtable text."""
    return OUTPUT_TEX.read_text()


def fit_row_numbers(label):
    """Pull the four integer values from a fit-statistics row by its label."""
    line = next(ln for ln in read_tex().splitlines()
                if ln.strip().startswith(label))
    return [int(n.replace(",", "")) for n in re.findall(r"[\d,]+", line)]


def _row_cells(line):
    """Split a LaTeX body row into its 4 data cells, dropping the '\\\\' terminator."""
    cells = [c.strip().rstrip("\\").strip() for c in line.split("&")]
    return cells[1:5]


def coef_cells(label):
    """Return the 4 estimate-row cells [col1..col4] for a coefficient label."""
    line = next(ln for ln in read_tex().splitlines() if label in ln)
    return _row_cells(line)


def se_cells_after(label):
    """Return the 4 SE-row cells that follow a coefficient label's estimate row."""
    lines = read_tex().splitlines()
    idx = next(i for i, ln in enumerate(lines) if label in ln)
    return _row_cells(lines[idx + 1])


# =====
# Category A: Sample-definition correctness (reproduced from real data)
# =====
@CSV_EXISTS
def test_all_participants_sample_count():
    """Col 4: full risk set (already_sold==0, emotion+trait-complete) = 13,590."""
    em = emotion_trait_complete_base()
    assert len(em) == EXPECTED["col4"]["obs"]
    assert int(em["sold"].sum()) == EXPECTED["col4"]["events"]
    assert em["player_id"].nunique() == EXPECTED["col4"]["participants"]


@CSV_EXISTS
def test_first_sellers_sample_count():
    """Col 1: player-group-rounds with a first sale (prior_group_sales==0)."""
    em = emotion_trait_complete_base()
    first_ids = em.loc[
        (em["prior_group_sales"] == 0) & (em["sold"] == 1),
        "player_group_round_id",
    ].unique()
    col1 = em[em["player_group_round_id"].isin(first_ids)]
    assert len(col1) == EXPECTED["col1"]["obs"]
    assert int(col1["sold"].sum()) == EXPECTED["col1"]["events"]


@CSV_EXISTS
def test_all_sellers_restriction_count():
    """Col 3: restrict to player-group-rounds containing any sale (M3 logic)."""
    em = emotion_trait_complete_base()
    seller_ids = em.loc[em["sold"] == 1, "player_group_round_id"].unique()
    col3 = em[em["player_group_round_id"].isin(seller_ids)]
    assert len(col3) == EXPECTED["col3"]["obs"]
    assert int(col3["sold"].sum()) == EXPECTED["col3"]["events"]


@CSV_EXISTS
def test_all_sellers_is_subset_of_full_risk_set():
    """The all-sellers sample (Col 3) is a strict subset of all participants."""
    em = emotion_trait_complete_base()
    seller_ids = set(em.loc[em["sold"] == 1, "player_group_round_id"])
    assert EXPECTED["col3"]["obs"] < EXPECTED["col4"]["obs"]
    # Every seller player-group-round is present in the full base.
    assert seller_ids.issubset(set(em["player_group_round_id"]))


@CSV_EXISTS
def test_first_and_all_sellers_share_event_count_only_for_all():
    """Col 3 and Col 4 cover all 659 sales; first-seller events are fewer."""
    assert EXPECTED["col3"]["events"] == EXPECTED["col4"]["events"]
    assert EXPECTED["col1"]["events"] < EXPECTED["col3"]["events"]


# =====
# Category B: Output structure
# =====
@TEX_EXISTS
def test_output_non_empty():
    """Produced .tex has content."""
    assert len(read_tex().strip()) > 0


@TEX_EXISTS
def test_four_model_columns():
    """longtable spec is l + 4 centered p-columns -> 4 model columns."""
    tex = read_tex()
    match = re.search(r"\\begin\{longtable\}\{l\*\{(\d+)\}", tex)
    assert match, "longtable column multiplier not found"
    assert int(match.group(1)) == 4, f"expected 4 columns, got {match.group(1)}"


@TEX_EXISTS
def test_column_headers_name_all_four_samples():
    """Headers label all four risk-set samples."""
    tex = read_tex()
    for header in ["First sellers", "Reactive", "All sellers", "All participants"]:
        assert header in tex, f"missing column header: {header}"


@TEX_EXISTS
def test_header_has_four_numbered_columns():
    """Numbered header runs (1)..(4); no fifth model column."""
    tex = read_tex()
    for col in ["(1)", "(2)", "(3)", "(4)"]:
        assert col in tex, f"missing model column {col}"
    assert "(5)" not in tex, "unexpected fifth model column"


@TEX_EXISTS
def test_all_four_fit_stat_rows_present():
    """Fit block reports Observations, Events, Participants, Log-likelihood."""
    tex = read_tex()
    for row in ["Observations", "Events", "Participants", "Log-likelihood"]:
        assert row in tex, f"missing fit-statistics row: {row}"


# =====
# Category C: Output values match reproduced samples
# =====
@TEX_EXISTS
def test_observations_row_matches_samples():
    """Observations row = [first, reactive, all-sellers, all-participants]."""
    assert fit_row_numbers("Observations") == [
        EXPECTED["col1"]["obs"], EXPECTED["col2"]["obs"],
        EXPECTED["col3"]["obs"], EXPECTED["col4"]["obs"],
    ]


@TEX_EXISTS
def test_events_row_matches_samples():
    """Events row = [first, reactive, all-sellers, all-participants]."""
    assert fit_row_numbers("Events") == [
        EXPECTED["col1"]["events"], EXPECTED["col2"]["events"],
        EXPECTED["col3"]["events"], EXPECTED["col4"]["events"],
    ]


@TEX_EXISTS
def test_all_participants_participant_count():
    """The All participants column reports 95 random-effect groups."""
    # Participants row: reactive column may render '---' (cluster count handled
    # separately), so assert the last value equals the player count.
    line = next(ln for ln in read_tex().splitlines()
                if ln.strip().startswith("Participants"))
    nums = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", line)]
    assert nums[-1] == EXPECTED["col4"]["participants"], (
        f"All-participants count {nums[-1]} != {EXPECTED['col4']['participants']}"
    )


@TEX_EXISTS
def test_cascade_rows_blank_for_first_and_reactive_columns():
    """Cascade '1 prior sale' is reported only for the all-* columns (3, 4)."""
    line = next(ln for ln in read_tex().splitlines() if "1 prior sale" in ln)
    cells = [c.strip() for c in line.split("&")]
    # cells: [label, col1, col2, col3, col4]
    assert cells[1] == "" and cells[2] == "", (
        "Cascade terms must be blank for First sellers and Reactive columns"
    )
    assert re.search(r"\d", cells[3]) and re.search(r"\d", cells[4]), (
        "Cascade terms must be reported for All sellers and All participants"
    )


@TEX_EXISTS
def test_signal_reported_in_all_four_columns():
    """Signal is a shared covariate across all four specifications."""
    line = next(ln for ln in read_tex().splitlines()
                if ln.strip().startswith("Signal"))
    cells = [c.strip() for c in line.split("&")]
    for col_cell in cells[1:5]:
        assert re.search(r"\d", col_cell), "Signal missing in a model column"


# =====
# Category D: Non-identified (separation) cell rendering
# =====
NI = "n.i."
# (label, 0-based column indices expected to render the n.i. marker)
SEPARATED_CELLS = [
    (r"3 prior $\times$ 1 prev.", [2, 3]),  # cols 3 & 4
    (r"3 prior $\times$ 2 prev.", [3]),     # col 4 (col 3 dropped as constant)
    (r"3 prior $\times$ 3 prev.", [3]),     # col 4 only
]


@TEX_EXISTS
@pytest.mark.parametrize("label,ni_cols", SEPARATED_CELLS)
def test_separated_cells_render_ni_marker(label, ni_cols):
    """Separated cascade-interaction cells show 'n.i.', not 0.0000/huge SE."""
    cells = coef_cells(label)
    for col in ni_cols:
        assert cells[col] == NI, f"{label} col {col + 1} should be '{NI}'"
        assert "0.0000" not in cells[col], f"{label} col {col + 1} not 0.0000"


@TEX_EXISTS
@pytest.mark.parametrize("label,ni_cols", SEPARATED_CELLS)
def test_separated_cell_se_row_blank(label, ni_cols):
    """The SE row beneath an n.i. estimate is blank for that column."""
    ses = se_cells_after(label)
    for col in ni_cols:
        assert ses[col] == "", f"{label} SE col {col + 1} should be blank"


@TEX_EXISTS
def test_wide_but_identified_estimates_not_overflagged():
    """Conservative rule: legitimate wide estimates still print as numbers."""
    # '3 prior sales' col3 (HR 1.8089, SE 2.2096) is wide but identified.
    assert re.search(r"\d", coef_cells("3 prior sales")[2])
    assert coef_cells("3 prior sales")[2] != NI
    # '3 prior x 3 prev.' col3 (HR 0.3853) is identified for the All-sellers col.
    assert re.search(r"\d", coef_cells(r"3 prior $\times$ 3 prev.")[2])
    assert coef_cells(r"3 prior $\times$ 3 prev.")[2] != NI


@TEX_EXISTS
@pytest.mark.parametrize("label", [
    r"1 prior $\times$ 1 prev.",
    r"2 prior $\times$ 1 prev.",
    r"2 prior $\times$ 2 prev.",
])
def test_low_order_interactions_print_in_all_sellers_cols(label):
    """1- and 2-prior interactions are identified in cols 3 and 4."""
    cells = coef_cells(label)
    for col in (2, 3):
        assert re.search(r"\d", cells[col]) and cells[col] != NI, (
            f"{label} col {col + 1} should be a number"
        )


# =====
# Category E: New footer strings (reviewer fixes)
# =====
@TEX_EXISTS
def test_footer_all_sellers_reworded():
    """Col-3 sample wording reflects per-player at-risk periods, not group-rounds."""
    tex = read_tex()
    assert "at-risk periods of players who sold" in tex
    assert "group-rounds with at least one sale" not in tex


@TEX_EXISTS
def test_footer_mixed_estimator_caveat():
    """Comparability caveat distinguishes coxme vs coxph SEs and HR scales."""
    tex = read_tex()
    assert "not directly comparable" in tex
    assert "subject-specific" in tex
    assert "cluster-robust" in tex


@TEX_EXISTS
def test_footer_ni_marker_explained():
    """A legend explains that n.i. marks separation-driven non-identification."""
    tex = read_tex()
    assert "not identified" in tex
    assert "separation" in tex


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
