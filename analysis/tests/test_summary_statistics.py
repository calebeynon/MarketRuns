"""
Purpose: Verify summary_statistics.R output tables against raw input data
Author: Claude Code
Date: 2026-02-17

Cross-validates each number in the 2 LaTeX tables by independently
computing the same statistics from the CSV inputs.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

# =====
# File paths
# =====
PANEL_PATH = Path("datastore/derived/individual_round_panel.csv")
SURVEY_PATH = Path("datastore/derived/survey_traits.csv")
TABLE_DIR = Path("analysis/output/tables")

COMBINED_TEX = TABLE_DIR / "summary_demographics_traits.tex"
SELL_RATES_TEX = TABLE_DIR / "summary_sell_rates.tex"

NO_CHAT_SEGMENTS = [1, 2]
CHAT_SEGMENTS = [3, 4]

TRAITS = [
    "extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness", "impulsivity", "state_anxiety",
    "risk_tolerance",
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
def combined_tex():
    return COMBINED_TEX.read_text()


@pytest.fixture(scope="module")
def sell_rates_tex():
    return SELL_RATES_TEX.read_text()


# =====
# LaTeX parsing helpers
# =====
def parse_all_data_rows(tex):
    """Extract all data rows from a table with multiple midrule sections."""
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


def parse_mean_sd(cell):
    """Extract (mean, sd) from '1.23 (4.56)' format."""
    match = re.match(r"([\d.]+)\s*\(([\d.]+)\)", cell)
    assert match, f"Could not parse mean(sd) from: {cell}"
    return float(match.group(1)), float(match.group(2))


# =====
# Combined demographics + traits table tests
# =====
class TestDemographicsTraitsTable:
    def test_subject_counts(self, panel, combined_tex):
        rows = parse_all_data_rows(combined_tex)
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

    def test_session_counts(self, panel, combined_tex):
        rows = parse_all_data_rows(combined_tex)
        s_row = [r for r in rows if "sessions" in r[0]][0]
        assert s_row[1] == str(panel.session_id.nunique())
        assert s_row[2] == str(
            panel[panel.treatment == "tr1"].session_id.nunique()
        )
        assert s_row[3] == str(
            panel[panel.treatment == "tr2"].session_id.nunique()
        )

    def test_age(self, survey, combined_tex):
        rows = parse_all_data_rows(combined_tex)
        age_row = [r for r in rows if "Age" in r[0]][0]

        for col_idx, subset in [(1, survey), (2, "tr1"), (3, "tr2")]:
            df = survey[survey.treatment == subset] if isinstance(subset, str) else subset
            expected_mean, expected_sd = df.age.mean(), df.age.std()
            actual_mean, actual_sd = parse_mean_sd(age_row[col_idx])
            assert actual_mean == pytest.approx(expected_mean, abs=0.01)
            assert actual_sd == pytest.approx(expected_sd, abs=0.01)

    def test_pct_female(self, survey, combined_tex):
        rows = parse_all_data_rows(combined_tex)
        fem_row = [r for r in rows if "Female" in r[0]][0]

        for col_idx, subset in [(1, survey), (2, "tr1"), (3, "tr2")]:
            df = survey[survey.treatment == subset] if isinstance(subset, str) else subset
            expected = 100 * (df.gender == "Female").mean()
            actual = float(fem_row[col_idx])
            assert actual == pytest.approx(expected, abs=0.1)

    def test_trait_means_and_sds(self, survey, combined_tex):
        rows = parse_all_data_rows(combined_tex)
        # Trait rows start after 4 demographics rows
        trait_rows = rows[4:]
        for i, trait in enumerate(TRAITS):
            row = trait_rows[i]
            for col_idx, subset in [(1, survey), (2, "tr1"), (3, "tr2")]:
                df = survey[survey.treatment == subset] if isinstance(subset, str) else subset
                expected_mean = df[trait].mean()
                expected_sd = df[trait].std()
                actual_mean, actual_sd = parse_mean_sd(row[col_idx])
                assert actual_mean == pytest.approx(expected_mean, abs=0.01), \
                    f"{trait} mean mismatch (col {col_idx})"
                assert actual_sd == pytest.approx(expected_sd, abs=0.01), \
                    f"{trait} SD mismatch (col {col_idx})"

    def test_row_count(self, combined_tex):
        """4 demographics rows + 8 trait rows = 12 total."""
        rows = parse_all_data_rows(combined_tex)
        assert len(rows) == 12, f"Expected 12 data rows, got {len(rows)}"


# =====
# Sell rates table tests
# =====
class TestSellRatesTable:
    """Validate summary_sell_rates.tex against individual_round_panel.csv."""

    COMBOS = [
        (1, "tr1", NO_CHAT_SEGMENTS),
        (2, "tr1", CHAT_SEGMENTS),
        (3, "tr2", NO_CHAT_SEGMENTS),
        (4, "tr2", CHAT_SEGMENTS),
    ]

    def _filter_subset(self, panel, treatment, segs, state=None, sellers_only=False):
        """Filter panel to a treatment x chat x state subset."""
        mask = (panel.treatment == treatment) & (panel.segment.isin(segs))
        if state is not None:
            mask &= panel.state == state
        subset = panel[mask]
        return subset[subset.did_sell == 1] if sellers_only else subset

    def _assert_block(self, panel, rows, col, stat_col, tr, segs, scale=1, sellers_only=False):
        """Assert overall, good-state, and bad-state values for one column."""
        overall_row, good_row, bad_row = rows
        for row, state in [(overall_row, None), (good_row, 1), (bad_row, 0)]:
            sub = self._filter_subset(panel, tr, segs, state, sellers_only)
            expected = scale * sub[stat_col].mean()
            assert float(row[col]) == pytest.approx(expected, abs=0.1)

    def test_sell_rate_values(self, panel, sell_rates_tex):
        rows = parse_all_data_rows(sell_rates_tex)
        block = rows[0:3]
        for col, tr, segs in self.COMBOS:
            self._assert_block(panel, block, col, "did_sell", tr, segs, scale=100)

    def test_avg_sell_period_values(self, panel, sell_rates_tex):
        rows = parse_all_data_rows(sell_rates_tex)
        block = rows[3:6]
        for col, tr, segs in self.COMBOS:
            self._assert_block(panel, block, col, "sell_period", tr, segs, sellers_only=True)

    def test_avg_sell_price_values(self, panel, sell_rates_tex):
        rows = parse_all_data_rows(sell_rates_tex)
        block = rows[6:9]
        for col, tr, segs in self.COMBOS:
            self._assert_block(panel, block, col, "sell_price", tr, segs, sellers_only=True)

    def test_row_count(self, sell_rates_tex):
        rows = parse_all_data_rows(sell_rates_tex)
        assert len(rows) == 9, f"Expected 9 data rows, got {len(rows)}"


# =====
# Table existence tests
# =====
class TestTablesExist:
    def test_combined_demographics_traits_exists(self):
        assert COMBINED_TEX.exists()

    def test_sell_rates_exists(self):
        assert SELL_RATES_TEX.exists()


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
