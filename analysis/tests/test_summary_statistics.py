"""
Purpose: Verify summary_statistics.R output tables against raw input data
Author: Claude Code
Date: 2026-02-17

Cross-validates each number in the 3 LaTeX tables by independently
computing the same statistics from the CSV inputs.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# =====
# File paths
# =====
PANEL_PATH = Path("datastore/derived/individual_round_panel.csv")
SURVEY_PATH = Path("datastore/derived/survey_traits.csv")
TABLE_DIR = Path("analysis/output/tables")

DEMOGRAPHICS_TEX = TABLE_DIR / "summary_demographics.tex"
TRAITS_TEX = TABLE_DIR / "summary_traits.tex"
OUTCOMES_TEX = TABLE_DIR / "summary_market_outcomes.tex"

TRAITS = [
    "extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness", "impulsivity", "state_anxiety",
]


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(PANEL_PATH)


@pytest.fixture(scope="module")
def survey():
    df = pd.read_csv(SURVEY_PATH)
    treatments = pd.read_csv(PANEL_PATH)[["session_id", "treatment"]].drop_duplicates()
    return df.merge(treatments, on="session_id")


@pytest.fixture(scope="module")
def demographics_tex():
    return DEMOGRAPHICS_TEX.read_text()


@pytest.fixture(scope="module")
def traits_tex():
    return TRAITS_TEX.read_text()


@pytest.fixture(scope="module")
def outcomes_tex():
    return OUTCOMES_TEX.read_text()


# =====
# LaTeX parsing helpers
# =====
def parse_table_rows(tex):
    """Extract data rows between midrule and bottomrule."""
    between = re.search(r"\\midrule(.+?)\\bottomrule", tex, re.DOTALL)
    assert between, "Could not find table body"
    lines = between.group(1).strip().split("\\\\")
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split("&")]
        rows.append(cells)
    return rows


def parse_mean_sd(cell):
    """Extract (mean, sd) from '1.23 (4.56)' format."""
    match = re.match(r"([\d.]+)\s*\(([\d.]+)\)", cell)
    assert match, f"Could not parse mean(sd) from: {cell}"
    return float(match.group(1)), float(match.group(2))


# =====
# Demographics table tests
# =====
class TestDemographicsTable:
    def test_subject_counts(self, panel, demographics_tex):
        rows = parse_table_rows(demographics_tex)
        n_row = [r for r in rows if "subjects" in r[0]][0]
        expected_total = panel.drop_duplicates(subset=["session_id", "player"]).shape[0]
        expected_tr1 = panel[panel.treatment == "tr1"].drop_duplicates(
            subset=["session_id", "player"]
        ).shape[0]
        expected_tr2 = panel[panel.treatment == "tr2"].drop_duplicates(
            subset=["session_id", "player"]
        ).shape[0]
        assert n_row[1] == str(expected_total)
        assert n_row[2] == str(expected_tr1)
        assert n_row[3] == str(expected_tr2)

    def test_session_counts(self, panel, demographics_tex):
        rows = parse_table_rows(demographics_tex)
        s_row = [r for r in rows if "sessions" in r[0]][0]
        assert s_row[1] == str(panel.session_id.nunique())
        assert s_row[2] == str(
            panel[panel.treatment == "tr1"].session_id.nunique()
        )
        assert s_row[3] == str(
            panel[panel.treatment == "tr2"].session_id.nunique()
        )

    def test_age(self, survey, demographics_tex):
        rows = parse_table_rows(demographics_tex)
        age_row = [r for r in rows if "Age" in r[0]][0]

        for col_idx, subset in [(1, survey), (2, "tr1"), (3, "tr2")]:
            df = survey[survey.treatment == subset] if isinstance(subset, str) else subset
            expected_mean, expected_sd = df.age.mean(), df.age.std()
            actual_mean, actual_sd = parse_mean_sd(age_row[col_idx])
            assert actual_mean == pytest.approx(expected_mean, abs=0.01)
            assert actual_sd == pytest.approx(expected_sd, abs=0.01)

    def test_pct_female(self, survey, demographics_tex):
        rows = parse_table_rows(demographics_tex)
        fem_row = [r for r in rows if "Female" in r[0]][0]

        for col_idx, subset in [(1, survey), (2, "tr1"), (3, "tr2")]:
            df = survey[survey.treatment == subset] if isinstance(subset, str) else subset
            expected = 100 * (df.gender == "Female").mean()
            actual = float(fem_row[col_idx])
            assert actual == pytest.approx(expected, abs=0.1)


# =====
# Traits table tests
# =====
class TestTraitsTable:
    def test_trait_means_and_sds(self, survey, traits_tex):
        rows = parse_table_rows(traits_tex)
        for i, trait in enumerate(TRAITS):
            row = rows[i]
            for col_idx, subset in [(1, survey), (2, "tr1"), (3, "tr2")]:
                df = survey[survey.treatment == subset] if isinstance(subset, str) else subset
                expected_mean = df[trait].mean()
                expected_sd = df[trait].std()
                actual_mean, actual_sd = parse_mean_sd(row[col_idx])
                assert actual_mean == pytest.approx(expected_mean, abs=0.01), \
                    f"{trait} mean mismatch (col {col_idx})"
                assert actual_sd == pytest.approx(expected_sd, abs=0.01), \
                    f"{trait} SD mismatch (col {col_idx})"

    def test_trait_differences(self, survey, traits_tex):
        rows = parse_table_rows(traits_tex)
        for i, trait in enumerate(TRAITS):
            tr1_vals = survey[survey.treatment == "tr1"][trait]
            tr2_vals = survey[survey.treatment == "tr2"][trait]
            expected_diff = tr1_vals.mean() - tr2_vals.mean()
            # Parse diff column (may have significance stars)
            diff_cell = rows[i][4]
            diff_num = float(re.match(r"([+-]?[\d.]+)", diff_cell).group(1))
            assert diff_num == pytest.approx(expected_diff, abs=0.01), \
                f"{trait} diff mismatch"

    def test_significance_stars(self, survey, traits_tex):
        rows = parse_table_rows(traits_tex)
        for i, trait in enumerate(TRAITS):
            tr1_vals = survey[survey.treatment == "tr1"][trait]
            tr2_vals = survey[survey.treatment == "tr2"][trait]
            _, p_value = stats.ttest_ind(tr1_vals, tr2_vals)

            diff_cell = rows[i][4]
            if p_value < 0.01:
                assert "***" in diff_cell, f"{trait}: expected *** (p={p_value:.4f})"
            elif p_value < 0.05:
                assert "**" in diff_cell, f"{trait}: expected ** (p={p_value:.4f})"
            elif p_value < 0.1:
                assert "*" in diff_cell, f"{trait}: expected * (p={p_value:.4f})"
            else:
                assert "*" not in diff_cell, f"{trait}: unexpected stars (p={p_value:.4f})"


# =====
# Market outcomes table tests
# =====
class TestMarketOutcomesTable:
    def test_sell_rate(self, panel, outcomes_tex):
        rows = parse_table_rows(outcomes_tex)
        sell_row = [r for r in rows if "Sell rate" in r[0]][0]
        no_chat = panel[panel.segment.isin([1, 2])]
        chat = panel[panel.segment.isin([3, 4])]
        assert float(sell_row[1]) == pytest.approx(
            100 * no_chat.did_sell.mean(), abs=0.1
        )
        assert float(sell_row[2]) == pytest.approx(
            100 * chat.did_sell.mean(), abs=0.1
        )

    def test_avg_sell_period(self, panel, outcomes_tex):
        rows = parse_table_rows(outcomes_tex)
        sp_row = [r for r in rows if "sell period" in r[0]][0]
        sellers_no_chat = panel[(panel.segment.isin([1, 2])) & (panel.did_sell == 1)]
        sellers_chat = panel[(panel.segment.isin([3, 4])) & (panel.did_sell == 1)]
        assert float(sp_row[1]) == pytest.approx(
            sellers_no_chat.sell_period.mean(), abs=0.01
        )
        assert float(sp_row[2]) == pytest.approx(
            sellers_chat.sell_period.mean(), abs=0.01
        )

    def test_avg_sell_price(self, panel, outcomes_tex):
        rows = parse_table_rows(outcomes_tex)
        pr_row = [r for r in rows if "sell price" in r[0]][0]
        sellers_no_chat = panel[(panel.segment.isin([1, 2])) & (panel.did_sell == 1)]
        sellers_chat = panel[(panel.segment.isin([3, 4])) & (panel.did_sell == 1)]
        assert float(pr_row[1]) == pytest.approx(
            sellers_no_chat.sell_price.mean(), abs=0.01
        )
        assert float(pr_row[2]) == pytest.approx(
            sellers_chat.sell_price.mean(), abs=0.01
        )

    def test_full_run_pct(self, panel, outcomes_tex):
        rows = parse_table_rows(outcomes_tex)
        fr_row = [r for r in rows if "Full run" in r[0]][0]

        for col_idx, segs in [(1, [1, 2]), (2, [3, 4])]:
            group_rounds = (
                panel[panel.segment.isin(segs)]
                .groupby(["session_id", "segment", "group_id", "round"])
                .did_sell.sum()
            )
            expected = 100 * (group_rounds == 4).mean()
            assert float(fr_row[col_idx]) == pytest.approx(expected, abs=0.1)

    def test_differences_are_chat_minus_no_chat(self, outcomes_tex):
        rows = parse_table_rows(outcomes_tex)
        for row in rows:
            no_chat_val = float(row[1])
            chat_val = float(row[2])
            diff_val = float(row[3])
            assert diff_val == pytest.approx(chat_val - no_chat_val, abs=0.01)


# =====
# Table existence tests
# =====
class TestTablesExist:
    def test_demographics_exists(self):
        assert DEMOGRAPHICS_TEX.exists()

    def test_traits_exists(self):
        assert TRAITS_TEX.exists()

    def test_outcomes_exists(self):
        assert OUTCOMES_TEX.exists()


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
