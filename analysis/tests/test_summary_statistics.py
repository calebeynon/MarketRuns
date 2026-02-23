"""
Purpose: Verify summary_statistics.R output tables against raw input data
Author: Claude Code
Date: 2026-02-22

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
SELLER_COUNTS_TEX = TABLE_DIR / "summary_seller_counts.tex"

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
def seller_counts_tex():
    return SELLER_COUNTS_TEX.read_text()


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
# Seller counts table tests
# =====
class TestSellerCountsTable:
    """Validate summary_seller_counts.tex against individual_round_panel.csv."""

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

    def _group_seller_counts(self, panel, treatment, segs, state=None):
        """Compute group-level seller counts for a subset."""
        sub = self._filter_subset(panel, treatment, segs, state)
        grouped = sub.groupby(
            ["session_id", "segment", "group_id", "round"]
        )["did_sell"].sum()
        return grouped

    def _count_group_rounds(self, panel, treatment, segs, state=None):
        """Count distinct group-rounds for a subset."""
        sub = self._filter_subset(panel, treatment, segs, state)
        return sub.drop_duplicates(
            subset=["session_id", "segment", "group_id", "round"]
        ).shape[0]

    def _compute_first_seller_period(self, panel, treatment, segs, state=None):
        """Compute mean of min sell_period per group-round (sellers only)."""
        sub = self._filter_subset(panel, treatment, segs, state)
        sellers = sub[sub.did_sell == 1]
        if sellers.empty:
            return None
        first_periods = sellers.groupby(
            ["session_id", "segment", "group_id", "round"]
        )["sell_period"].min()
        return first_periods.mean()

    def test_total_group_rounds(self, panel, seller_counts_tex):
        rows = parse_all_data_rows(seller_counts_tex)
        overall_row = rows[0]
        for col, tr, segs in self.COMBOS:
            expected = self._count_group_rounds(panel, tr, segs)
            assert int(overall_row[col]) == expected == 180

    def test_state_counts(self, panel, seller_counts_tex):
        rows = parse_all_data_rows(seller_counts_tex)
        good_row, bad_row = rows[1], rows[2]
        for col, tr, segs in self.COMBOS:
            expected_good = self._count_group_rounds(panel, tr, segs, state=1)
            expected_bad = self._count_group_rounds(panel, tr, segs, state=0)
            assert int(good_row[col]) == expected_good
            assert int(bad_row[col]) == expected_bad
            # Good + bad should equal total
            assert expected_good + expected_bad == 180

    def test_zero_seller_groups_values(self, panel, seller_counts_tex):
        rows = parse_all_data_rows(seller_counts_tex)
        block = rows[3:6]
        overall_row, good_row, bad_row = block
        for col, tr, segs in self.COMBOS:
            for row, state in [(overall_row, None), (good_row, 1), (bad_row, 0)]:
                counts = self._group_seller_counts(panel, tr, segs, state)
                expected = int((counts == 0).sum())
                assert int(row[col]) == expected

    def test_avg_sellers_values(self, panel, seller_counts_tex):
        rows = parse_all_data_rows(seller_counts_tex)
        block = rows[6:9]
        overall_row, good_row, bad_row = block
        for col, tr, segs in self.COMBOS:
            for row, state in [(overall_row, None), (good_row, 1), (bad_row, 0)]:
                counts = self._group_seller_counts(panel, tr, segs, state)
                expected = counts.mean()
                assert float(row[col]) == pytest.approx(expected, abs=0.01)

    def test_avg_sell_period_values(self, panel, seller_counts_tex):
        rows = parse_all_data_rows(seller_counts_tex)
        block = rows[9:12]
        overall_row, good_row, bad_row = block
        for col, tr, segs in self.COMBOS:
            for row, state in [(overall_row, None), (good_row, 1), (bad_row, 0)]:
                sub = self._filter_subset(panel, tr, segs, state, sellers_only=True)
                expected = sub["sell_period"].mean()
                assert float(row[col]) == pytest.approx(expected, abs=0.1)

    def test_avg_first_seller_period_values(self, panel, seller_counts_tex):
        rows = parse_all_data_rows(seller_counts_tex)
        block = rows[12:15]
        overall_row, good_row, bad_row = block
        for col, tr, segs in self.COMBOS:
            for row, state in [(overall_row, None), (good_row, 1), (bad_row, 0)]:
                expected = self._compute_first_seller_period(panel, tr, segs, state)
                if expected is None:
                    assert row[col] == "--"
                else:
                    assert float(row[col]) == pytest.approx(expected, abs=0.1)

    def test_row_count(self, seller_counts_tex):
        rows = parse_all_data_rows(seller_counts_tex)
        assert len(rows) == 15, f"Expected 15 data rows, got {len(rows)}"


# =====
# Table existence tests
# =====
class TestTablesExist:
    def test_combined_demographics_traits_exists(self):
        assert COMBINED_TEX.exists()

    def test_seller_counts_exists(self):
        assert SELLER_COUNTS_TEX.exists()


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
