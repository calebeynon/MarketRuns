"""
Purpose: Verify first seller trait analysis against raw oTree session data
Author: Claude Code
Date: 2026-02-17

Cross-validates visualize_first_seller_traits.R and first_seller_descriptive_stats.R
by independently computing statistics from raw data and comparing against outputs.
"""

import subprocess

import pandas as pd
import pytest

from analysis.tests.helpers_first_seller import (
    DERIVED_DIR, DESC_R_SCRIPT, LATEX_TABLE, OUTPUT_CI_DIFF, OUTPUT_ROBUSTNESS,
    PROJECT_ROOT, ROUND_DATA_CSV, SESSION_FILES, SURVEY_TRAITS_CSV, TRAITS,
    VIZ_R_SCRIPT, build_individual_data, classify_group, compute_raw_first_seller_counts,
    compute_raw_survey_traits, count_group_rounds, p_to_stars, parse_all_sessions,
    parse_latex_differences, parse_latex_mean_sd, parse_latex_stars, welch_t_test,
)


# =====
# Main function
# =====
def main():
    """Run all tests via pytest."""
    pytest.main([__file__, "-v"])


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def parsed_sessions():
    """Parse all 6 raw oTree sessions via market_data.py."""
    sessions = parse_all_sessions()
    if sessions is None:
        pytest.skip("Not all 6 raw session files found")
    return sessions


@pytest.fixture(scope="module")
def raw_fs_counts(parsed_sessions):
    """First seller counts computed from raw oTree data."""
    return compute_raw_first_seller_counts(parsed_sessions)


@pytest.fixture(scope="module")
def raw_traits():
    """Trait scores computed from raw survey CSVs."""
    return compute_raw_survey_traits()


@pytest.fixture(scope="module")
def round_data():
    """Load derived first_seller_round_data.csv."""
    return pd.read_csv(ROUND_DATA_CSV)


@pytest.fixture(scope="module")
def survey_data():
    """Load derived survey_traits.csv."""
    return pd.read_csv(SURVEY_TRAITS_CSV)


@pytest.fixture(scope="module")
def individual_data(round_data, survey_data):
    """Build individual-level dataset matching R scripts."""
    return build_individual_data(round_data, survey_data)


@pytest.fixture(scope="module")
def latex_content():
    """Read LaTeX table output."""
    if not LATEX_TABLE.exists():
        pytest.skip(f"LaTeX table not found: {LATEX_TABLE}")
    return LATEX_TABLE.read_text()


# =====
# Section A: Cross-validate first seller counts from raw oTree data
# =====
def test_raw_total_group_rounds(parsed_sessions):
    """Raw data spans 6 sessions x 30 rounds x 4 groups = 720 group-rounds."""
    assert count_group_rounds(parsed_sessions) == 720


def test_derived_round_data_row_count(round_data):
    """Derived round data has 2880 player-round rows."""
    assert len(round_data) == 2880


def test_first_seller_counts_match_raw(raw_fs_counts, round_data):
    """First seller counts from raw oTree match derived round data."""
    derived = round_data.groupby(["session_id", "player"])["is_first_seller"].sum()
    mismatches = []
    for (sid, player), raw_n in raw_fs_counts.items():
        derived_n = derived.get((sid, player), 0)
        if raw_n != derived_n:
            mismatches.append(f"({sid},{player}): raw={raw_n}, derived={derived_n}")
    assert len(mismatches) == 0, "\n".join(mismatches[:20])


def test_binary_classification_from_raw(raw_fs_counts, survey_data):
    """Ever first seller = 83, never = 12 from raw data."""
    keys = {(r["session_id"], r["player"]) for _, r in survey_data.iterrows()}
    ever = sum(1 for k in keys if raw_fs_counts.get(k, 0) >= 1)
    assert ever == 83
    assert len(keys) - ever == 12


def test_three_group_classification_from_raw(raw_fs_counts, survey_data):
    """3-group split from raw: 0=12, 1-2=24, 3+=59."""
    keys = {(r["session_id"], r["player"]) for _, r in survey_data.iterrows()}
    groups = {"0 times": 0, "1-2 times": 0, "3+ times": 0}
    for k in keys:
        groups[classify_group(raw_fs_counts.get(k, 0))] += 1
    assert groups == {"0 times": 12, "1-2 times": 24, "3+ times": 59}


def test_grouping_requires_session_id(round_data):
    """Grouping by player alone gives 16 (wrong), not 96."""
    assert len(round_data.groupby("player")["is_first_seller"].sum()) == 16


# =====
# Section B: Cross-validate trait scores from raw survey CSVs
# =====
def test_raw_survey_yields_95(raw_traits):
    """Raw survey processing yields 95 participants."""
    assert len(raw_traits) == 95


def test_raw_traits_match_derived(raw_traits, survey_data):
    """Trait scores from raw CSVs match derived survey_traits.csv."""
    mismatches = []
    for _, row in raw_traits.iterrows():
        sid, player = row["session_id"], row["player"]
        d = survey_data[(survey_data["session_id"] == sid) & (survey_data["player"] == player)]
        if len(d) != 1:
            mismatches.append(f"({sid},{player}): missing in derived")
            continue
        for trait in TRAITS:
            if abs(row[trait] - d.iloc[0][trait]) > 1e-6:
                mismatches.append(f"({sid},{player}) {trait}: mismatch")
    assert len(mismatches) == 0, "\n".join(mismatches[:20])


def test_missing_participant_is_session_3_c(raw_traits):
    """Missing participant is session 3_11-11-tr2, player C."""
    s3 = raw_traits[raw_traits["session_id"] == "3_11-11-tr2"]
    assert "C" not in set(s3["player"])
    assert len(s3) == 15


def test_raw_traits_session_counts(raw_traits):
    """16 per session except session 3 which has 15."""
    counts = raw_traits.groupby("session_id").size()
    for sid in SESSION_FILES:
        expected = 15 if sid == "3_11-11-tr2" else 16
        assert counts[sid] == expected, f"{sid}: expected {expected}, got {counts[sid]}"


@pytest.mark.parametrize("trait", TRAITS[:6])
def test_bfi_impulsivity_range_raw(trait, raw_traits):
    """BFI/impulsivity scores from raw data in [1, 7]."""
    assert raw_traits[trait].between(1.0, 7.0).all()


def test_state_anxiety_range_raw(raw_traits):
    """State anxiety from raw data in [1, 4]."""
    assert raw_traits["state_anxiety"].between(1.0, 4.0).all()


# =====
# Section C: LaTeX table cross-validation
# =====
@pytest.mark.parametrize("trait", TRAITS)
def test_t_test_diff_matches_latex(trait, individual_data, latex_content):
    """Welch's t-test difference matches LaTeX to 3 decimal places."""
    fs = individual_data.loc[individual_data["is_ever_first_seller"], trait].values
    nfs = individual_data.loc[~individual_data["is_ever_first_seller"], trait].values
    diff, _, _, _ = welch_t_test(fs, nfs)
    diffs = parse_latex_differences(latex_content)
    label = trait.replace("_", " ").title()
    assert label in diffs, f"{label} not in LaTeX"
    assert abs(diff - diffs[label]) < 0.001


def test_significance_markers_correct(individual_data, latex_content):
    """Significance markers match scipy p-values."""
    stars = parse_latex_stars(latex_content)
    for trait in TRAITS:
        fs = individual_data.loc[individual_data["is_ever_first_seller"], trait].values
        nfs = individual_data.loc[~individual_data["is_ever_first_seller"], trait].values
        _, p_val, _, _ = welch_t_test(fs, nfs)
        label = trait.replace("_", " ").title()
        assert stars.get(label, "") == p_to_stars(p_val), f"{label}: p={p_val:.4f}"


def test_latex_mean_sd_match_python(individual_data, latex_content):
    """Mean (SD) values in LaTeX match Python calculations."""
    parsed = parse_latex_mean_sd(latex_content)
    for trait in TRAITS:
        fs = individual_data.loc[individual_data["is_ever_first_seller"], trait]
        nfs = individual_data.loc[~individual_data["is_ever_first_seller"], trait]
        p = parsed[trait]
        assert abs(fs.mean() - p["fs_mean"]) < 0.01
        assert abs(fs.std() - p["fs_sd"]) < 0.01
        assert abs(nfs.mean() - p["nfs_mean"]) < 0.01
        assert abs(nfs.std() - p["nfs_sd"]) < 0.01


def test_state_anxiety_significant(individual_data):
    """State anxiety t-test significant at p < 0.01."""
    fs = individual_data.loc[individual_data["is_ever_first_seller"], "state_anxiety"]
    nfs = individual_data.loc[~individual_data["is_ever_first_seller"], "state_anxiety"]
    _, p, _, _ = welch_t_test(fs.values, nfs.values)
    assert p < 0.01


@pytest.mark.parametrize("trait", TRAITS[:6])
def test_non_anxiety_ci_includes_zero(trait, individual_data):
    """Non-anxiety trait CIs span zero."""
    fs = individual_data.loc[individual_data["is_ever_first_seller"], trait].values
    nfs = individual_data.loc[~individual_data["is_ever_first_seller"], trait].values
    _, _, lo, hi = welch_t_test(fs, nfs)
    assert lo < 0 < hi, f"{trait}: CI=[{lo:.4f}, {hi:.4f}]"


# =====
# Regression tests from verified outputs
# =====
def test_state_anxiety_known_values(individual_data):
    """State anxiety means match verified LaTeX output."""
    fs = individual_data.loc[individual_data["is_ever_first_seller"], "state_anxiety"]
    nfs = individual_data.loc[~individual_data["is_ever_first_seller"], "state_anxiety"]
    assert fs.mean() == pytest.approx(1.749, abs=0.001)
    assert nfs.mean() == pytest.approx(1.292, abs=0.001)
    assert (fs.mean() - nfs.mean()) == pytest.approx(0.457, abs=0.001)


def test_state_anxiety_monotonic_increase(individual_data):
    """State anxiety increases monotonically across 0, 1-2, 3+ groups."""
    means = {
        g: individual_data.loc[individual_data["fs_group"] == g, "state_anxiety"].mean()
        for g in ["0 times", "1-2 times", "3+ times"]
    }
    assert means["0 times"] < means["1-2 times"] < means["3+ times"]


def test_state_anxiety_group_means_known(individual_data):
    """3-group state anxiety means match verified values."""
    expected = {"0 times": 1.2917, "1-2 times": 1.5903, "3+ times": 1.8136}
    for grp, exp in expected.items():
        val = individual_data.loc[individual_data["fs_group"] == grp, "state_anxiety"].mean()
        assert val == pytest.approx(exp, abs=0.001)


def test_individual_count_95(individual_data):
    """Inner join yields exactly 95 individuals."""
    assert len(individual_data) == 95


# =====
# Section D: LaTeX structure and R script execution
# =====
def test_latex_table_exists():
    """LaTeX table exists and is non-empty."""
    assert LATEX_TABLE.exists() and LATEX_TABLE.stat().st_size > 0


def test_latex_has_seven_rows(latex_content):
    """LaTeX table has exactly 7 data rows."""
    rows = [l for l in latex_content.split("\n") if "&" in l and "Trait" not in l]
    assert len(rows) == 7


def test_latex_headers(latex_content):
    """LaTeX table has N=95 and correct column headers."""
    assert "N = 95" in latex_content
    assert "First Sellers" in latex_content
    assert "Difference" in latex_content


def test_output_plots_exist():
    """Plot PDFs exist and are non-empty."""
    for path in [OUTPUT_CI_DIFF, OUTPUT_ROBUSTNESS]:
        assert path.exists() and path.stat().st_size > 0, f"Missing: {path}"


def test_viz_r_script_runs():
    """Visualization R script runs without error."""
    r = subprocess.run(
        ["Rscript", str(VIZ_R_SCRIPT)], capture_output=True, text=True,
        cwd=str(PROJECT_ROOT), timeout=120,
    )
    assert r.returncode == 0, f"Failed:\n{r.stderr}"


def test_desc_r_script_runs():
    """Descriptive stats R script runs without error."""
    r = subprocess.run(
        ["Rscript", str(DESC_R_SCRIPT)], capture_output=True, text=True,
        cwd=str(PROJECT_ROOT), timeout=120,
    )
    assert r.returncode == 0, f"Failed:\n{r.stderr}"


# %%
if __name__ == "__main__":
    main()
