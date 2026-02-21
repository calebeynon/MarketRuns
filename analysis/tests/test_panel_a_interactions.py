"""
Purpose: Validate prev-period dummies and interaction terms added in issue #39
Author: Claude Code
Date: 2026-02-17

Tests verify that:
1. Previous-period sale dummies are correctly constructed from lagged counts.
2. Interaction terms equal products of their component dummies.
3. Observation counts in LPM and logit tables remain unchanged.
4. Previous-period dummies match between emotions and individual_period datasets.
"""

import re
import pandas as pd
import pytest
from pathlib import Path

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DERIVED_DIR = PROJECT_ROOT / "datastore" / "derived"
EMOTIONS_DATASET = DERIVED_DIR / "emotions_traits_selling_dataset.csv"
INDIVIDUAL_PERIOD_DATASET = DERIVED_DIR / "individual_period_dataset.csv"
LPM_TABLE_FULL = (
    PROJECT_ROOT / "analysis" / "output" / "tables"
    / "unified_selling_regression_full.tex"
)
LOGIT_TABLE_FULL = (
    PROJECT_ROOT / "analysis" / "output" / "tables"
    / "unified_selling_logit_full.tex"
)

PREV_DUMMIES = ["dummy_1_prev", "dummy_2_prev", "dummy_3_prev"]
CUM_DUMMIES = ["dummy_1_cum", "dummy_2_cum", "dummy_3_cum"]
INTERACTION_TERMS = [
    "int_1_1", "int_2_1", "int_2_2",
    "int_3_1", "int_3_2", "int_3_3",
]


# =====
# Main function
# =====
def main():
    pytest.main([__file__, "-v"])


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def base_data():
    """Load and prepare base data replicating R prepare_base_data()."""
    if not EMOTIONS_DATASET.exists():
        pytest.skip(f"Dataset not found: {EMOTIONS_DATASET}")
    df = pd.read_csv(EMOTIONS_DATASET)
    df = df[df["already_sold"] == 0].copy()
    df["group_round_id"] = (
        df["session_id"] + "_"
        + df["segment"].astype(str) + "_"
        + df["group_id"].astype(str) + "_"
        + df["round"].astype(str)
    )
    df = _add_cum_dummies(df)
    df = _add_prev_period_dummies(df)
    df = _add_interaction_terms(df)
    return df


@pytest.fixture(scope="module")
def individual_period_data():
    """Load individual_period_dataset for cross-validation."""
    if not INDIVIDUAL_PERIOD_DATASET.exists():
        pytest.skip(f"Dataset not found: {INDIVIDUAL_PERIOD_DATASET}")
    df = pd.read_csv(INDIVIDUAL_PERIOD_DATASET)
    df = df[df["already_sold"] == 0].copy()
    df["group_round_id"] = (
        df["session_id"] + "_"
        + df["segment"].astype(str) + "_"
        + df["group_id"].astype(str) + "_"
        + df["round"].astype(str)
    )
    return df


# =====
# Variable construction helpers (replicate R logic)
# =====
def _add_cum_dummies(df):
    for n in [1, 2, 3]:
        df[f"dummy_{n}_cum"] = (df["prior_group_sales"] == n).astype(int)
    return df


def _compute_lagged_period_sales(df):
    ps = (
        df.groupby(["group_round_id", "period"])["sold"]
        .sum().reset_index().rename(columns={"sold": "n_sales"})
    )
    ps = ps.sort_values(["group_round_id", "period"])
    ps["prev_n_sales"] = ps.groupby("group_round_id")["n_sales"].shift(1)
    ps["prev_n_sales"] = ps["prev_n_sales"].fillna(0).astype(int)
    return ps[["group_round_id", "period", "prev_n_sales"]]


def _add_prev_period_dummies(df):
    lagged = _compute_lagged_period_sales(df)
    df = df.merge(lagged, on=["group_round_id", "period"], how="left")
    for n in [1, 2, 3]:
        df[f"dummy_{n}_prev"] = (df["prev_n_sales"] == n).astype(int)
    return df


def _add_interaction_terms(df):
    for c, p in [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)]:
        df[f"int_{c}_{p}"] = df[f"dummy_{c}_cum"] * df[f"dummy_{p}_prev"]
    return df


def _manual_prev_sales(df, row):
    if row["period"] == 1:
        return 0
    prev = df[
        (df["group_round_id"] == row["group_round_id"])
        & (df["period"] == row["period"] - 1)
    ]
    return prev["sold"].sum()


# =====
# TestPrevPeriodDummies
# =====
class TestPrevPeriodDummies:
    def test_all_binary(self, base_data):
        for col in PREV_DUMMIES:
            assert set(base_data[col].unique()).issubset({0, 1})

    def test_mutually_exclusive(self, base_data):
        total = sum(base_data[col] for col in PREV_DUMMIES)
        assert (total <= 1).all()

    def test_all_zero_when_period_one(self, base_data):
        p1 = base_data[base_data["period"] == 1]
        assert len(p1) > 0
        for col in PREV_DUMMIES:
            assert (p1[col] == 0).all(), f"{col} nonzero in period 1"

    def test_matches_manual_lag(self, base_data):
        sample = base_data.sample(min(200, len(base_data)), random_state=39)
        mismatches = sum(
            _manual_prev_sales(base_data, row) != row["prev_n_sales"]
            for _, row in sample.iterrows()
        )
        assert mismatches == 0, f"{mismatches}/200 lag mismatches"

    def test_dummy_values_match_prev_n_sales(self, base_data):
        for n in [1, 2, 3]:
            col = f"dummy_{n}_prev"
            mask = base_data["prev_n_sales"] == n
            assert (base_data.loc[mask, col] == 1).all()
            assert (base_data.loc[~mask, col] == 0).all()


# =====
# TestInteractionTerms
# =====
class TestInteractionTerms:
    def test_all_binary(self, base_data):
        for col in INTERACTION_TERMS:
            assert set(base_data[col].unique()).issubset({0, 1})

    def test_equal_product_of_components(self, base_data):
        for c, p in [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)]:
            expected = base_data[f"dummy_{c}_cum"] * base_data[f"dummy_{p}_prev"]
            assert (base_data[f"int_{c}_{p}"] == expected).all()

    def test_no_impossible_interactions(self, base_data):
        for col in INTERACTION_TERMS:
            ones = base_data[base_data[col] == 1]
            parts = col.split("_")
            cum_idx, prev_idx = int(parts[1]), int(parts[2])
            assert (ones[f"dummy_{cum_idx}_cum"] == 1).all()
            assert (ones[f"dummy_{prev_idx}_prev"] == 1).all()

    def test_zero_when_period_one(self, base_data):
        p1 = base_data[base_data["period"] == 1]
        for col in INTERACTION_TERMS:
            assert (p1[col] == 0).all(), f"{col} nonzero in period 1"


# =====
# TestObservationCountsUnchanged
# =====
class TestObservationCountsUnchanged:
    def test_lpm_panel_a_obs(self):
        assert _parse_panel_obs(LPM_TABLE_FULL, "Panel A") == [13713, 13713, 13590]

    def test_lpm_panel_b_obs(self):
        assert _parse_panel_obs(LPM_TABLE_FULL, "Panel B") == [1218, 1217, 1183]

    def test_lpm_panel_c_obs(self):
        assert _parse_panel_obs(LPM_TABLE_FULL, "Panel C") == [622, 622, 619]

    def test_logit_panel_a_obs(self):
        assert _parse_panel_obs(LOGIT_TABLE_FULL, "Panel A") == [13713, 12369, 13590]

    def test_logit_panel_b_obs(self):
        assert _parse_panel_obs(LOGIT_TABLE_FULL, "Panel B") == [1218, 1194, 1183]

    def test_logit_panel_c_obs(self):
        assert _parse_panel_obs(LOGIT_TABLE_FULL, "Panel C") == [622, 622, 619]


# =====
# TestPanelAPrevPeriodMatchesExtended
# =====
class TestPanelAPrevPeriodMatchesExtended:
    def test_prev_dummies_match_across_datasets(
        self, base_data, individual_period_data
    ):
        ipd = individual_period_data.copy()
        lagged = _compute_lagged_period_sales(ipd)
        ipd = ipd.merge(lagged, on=["group_round_id", "period"], how="left")
        merge_keys = [
            "session_id", "segment", "round", "period", "group_id", "player"
        ]
        merged = base_data.merge(
            ipd[merge_keys + ["prev_n_sales"]],
            on=merge_keys, how="inner", suffixes=("_emo", "_ipd"),
        )
        assert len(merged) > 1000, f"Only {len(merged)} matched rows"
        n_bad = (merged["prev_n_sales_emo"] != merged["prev_n_sales_ipd"]).sum()
        assert n_bad == 0, f"{n_bad}/{len(merged)} prev_n_sales mismatches"


# =====
# LaTeX table parsing helper
# =====
def _parse_panel_obs(table_path, panel_prefix):
    if not table_path.exists():
        pytest.skip(f"Table not found: {table_path}")
    text = table_path.read_text()
    panels = ["Panel A", "Panel B", "Panel C"]
    idx = next(i for i, p in enumerate(panels) if p == panel_prefix)
    start = text.index(panel_prefix)
    end = text.index(panels[idx + 1]) if idx + 1 < len(panels) else len(text)
    block = text[start:end]
    for line in block.split("\n"):
        if "Observations" in line:
            nums = re.findall(r"[\d,]+", line.replace("Observations", ""))
            return [int(n.replace(",", "")) for n in nums]
    pytest.fail(f"No Observations line in {panel_prefix} of {table_path}")


# %%
if __name__ == "__main__":
    main()
