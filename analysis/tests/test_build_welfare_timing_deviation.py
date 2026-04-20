"""
Purpose: Regression tests for build_welfare_timing_deviation.py output.
Author: Claude Code
Date: 2026-04-19

Tests validate the real CSV produced by the builder against Task #2 spec
(issue #116). Tests load the output CSV once per module via a fixture;
they do NOT re-run the builder.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
OUTPUT_PATH = DATASTORE / "derived" / "welfare_timing_deviation.csv"
EQUILIBRIUM_PATH = DATASTORE / "derived" / "equilibrium_thresholds.csv"
PERIOD_DATA_PATH = DATASTORE / "derived" / "individual_period_dataset_extended.csv"

REQUIRED_COLUMNS = [
    "session_id", "segment", "round", "player", "group_id", "global_group_id",
    "treatment", "alpha", "n", "pi_at_sale", "threshold_pi",
    "pi_deviation", "pi_dev_neg", "pi_dev_pos", "welfare", "state",
]


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def welfare_df() -> pd.DataFrame:
    """Load the welfare timing deviation dataset once per test module."""
    return pd.read_csv(OUTPUT_PATH)


@pytest.fixture(scope="module")
def equilibrium_df() -> pd.DataFrame:
    """Load the equilibrium thresholds table once per test module."""
    return pd.read_csv(EQUILIBRIUM_PATH)


@pytest.fixture(scope="module")
def periods_df() -> pd.DataFrame:
    """Load the extended period-level dataset once per test module."""
    return pd.read_csv(PERIOD_DATA_PATH)


# =====
# Test 1: Output file exists
# =====
def test_output_file_exists():
    """Output CSV must exist at the expected datastore path."""
    assert OUTPUT_PATH.exists(), f"Missing output file: {OUTPUT_PATH}"


# =====
# Test 2: Required columns present
# =====
def test_required_columns_present(welfare_df):
    """All required columns must be present in the output."""
    missing = set(REQUIRED_COLUMNS) - set(welfare_df.columns)
    assert not missing, f"Missing required columns: {missing}"


# =====
# Test 3: alpha column has exactly two values (0.0 and 0.5)
# =====
def test_alpha_values(welfare_df):
    """alpha must contain exactly {0.0, 0.5}."""
    alphas = set(welfare_df["alpha"].unique())
    assert alphas == {0.0, 0.5}, f"Unexpected alpha values: {alphas}"


# =====
# Test 4: n column only contains {1, 2, 3} (voluntary sellers only)
# =====
def test_n_values_voluntary_only(welfare_df):
    """n must be in {1, 2, 3} — n=4 forced sellers are dropped."""
    ns = set(welfare_df["n"].unique())
    assert ns == {1, 2, 3}, f"Unexpected n values (expected voluntary only): {ns}"


# =====
# Test 5: state column is all 1 (sample restriction)
# =====
def test_state_all_one(welfare_df):
    """Dataset restricted to state==1 rounds."""
    assert (welfare_df["state"] == 1).all(), "Found rows with state != 1"


# =====
# Test 6: pi_at_sale in [0, 1]
# =====
def test_pi_at_sale_in_unit_interval(welfare_df):
    """pi_at_sale is a probability and must lie in [0, 1]."""
    assert welfare_df["pi_at_sale"].between(0.0, 1.0).all(), (
        "pi_at_sale outside [0, 1]"
    )


# =====
# Test 7: pi_deviation = pi_at_sale - threshold_pi
# =====
def test_pi_deviation_formula(welfare_df):
    """pi_deviation must equal pi_at_sale - threshold_pi (float tolerance)."""
    expected = welfare_df["pi_at_sale"] - welfare_df["threshold_pi"]
    assert np.allclose(welfare_df["pi_deviation"], expected, atol=1e-10), (
        "pi_deviation does not equal pi_at_sale - threshold_pi"
    )


# =====
# Test 8: pi_dev_neg and pi_dev_pos split correctly
# =====
def test_pi_dev_split(welfare_df):
    """pi_dev_neg = min(pi_deviation, 0); pi_dev_pos = max(pi_deviation, 0)."""
    dev = welfare_df["pi_deviation"]
    expected_neg = np.minimum(dev, 0.0)
    expected_pos = np.maximum(dev, 0.0)

    assert np.allclose(welfare_df["pi_dev_neg"], expected_neg, atol=1e-10), (
        "pi_dev_neg != min(pi_deviation, 0)"
    )
    assert np.allclose(welfare_df["pi_dev_pos"], expected_pos, atol=1e-10), (
        "pi_dev_pos != max(pi_deviation, 0)"
    )
    assert np.allclose(
        welfare_df["pi_dev_neg"] + welfare_df["pi_dev_pos"],
        dev,
        atol=1e-10,
    ), "pi_dev_neg + pi_dev_pos != pi_deviation"


# =====
# Test 9: tr1 threshold matches equilibrium (random, alpha=0, n=2)
# =====
def test_tr1_threshold_matches_equilibrium(welfare_df, equilibrium_df):
    """tr1 + alpha=0 + n=2 threshold_pi matches eq table (random, 0.0, 2)."""
    expected_row = equilibrium_df[
        (equilibrium_df["alpha"] == 0.0)
        & (equilibrium_df["treatment"] == "random")
        & (equilibrium_df["n"] == 2)
    ]
    assert len(expected_row) == 1, "Expected one matching eq row"
    expected_threshold = expected_row["threshold_pi"].iloc[0]

    observed = welfare_df[
        (welfare_df["treatment"] == "tr1")
        & (welfare_df["alpha"] == 0.0)
        & (welfare_df["n"] == 2)
    ]["threshold_pi"].unique()
    assert len(observed) == 1, f"tr1/alpha=0/n=2 has multiple thresholds: {observed}"
    assert np.isclose(observed[0], expected_threshold, atol=1e-10), (
        f"tr1 threshold {observed[0]} != eq threshold {expected_threshold}"
    )


# =====
# Test 10: tr2 threshold matches equilibrium (average, alpha=0.5, n=3)
# =====
def test_tr2_threshold_matches_equilibrium(welfare_df, equilibrium_df):
    """tr2 + alpha=0.5 + n=3 threshold_pi matches eq table (average, 0.5, 3)."""
    expected_row = equilibrium_df[
        (equilibrium_df["alpha"] == 0.5)
        & (equilibrium_df["treatment"] == "average")
        & (equilibrium_df["n"] == 3)
    ]
    assert len(expected_row) == 1, "Expected one matching eq row"
    expected_threshold = expected_row["threshold_pi"].iloc[0]

    observed = welfare_df[
        (welfare_df["treatment"] == "tr2")
        & (welfare_df["alpha"] == 0.5)
        & (welfare_df["n"] == 3)
    ]["threshold_pi"].unique()
    assert len(observed) == 1, (
        f"tr2/alpha=0.5/n=3 has multiple thresholds: {observed}"
    )
    assert np.isclose(observed[0], expected_threshold, atol=1e-10), (
        f"tr2 threshold {observed[0]} != eq threshold {expected_threshold}"
    )


# =====
# Test 11: equal row counts at alpha=0 and alpha=0.5
# =====
def test_equal_row_count_per_alpha(welfare_df):
    """Long format: each seller-round contributes one row per alpha."""
    counts = welfare_df["alpha"].value_counts()
    assert counts[0.0] == counts[0.5], (
        f"Unequal rows across alphas: {counts.to_dict()}"
    )


# =====
# Test 12: no NaN in key numeric columns
# =====
def test_no_nan_in_key_columns(welfare_df):
    """Key analysis columns must be fully populated."""
    key_cols = ["pi_at_sale", "threshold_pi", "pi_deviation", "welfare", "n"]
    for col in key_cols:
        assert not welfare_df[col].isna().any(), f"NaN found in '{col}'"


# =====
# Test 13: pi_at_sale matches signal from period data for a sampled seller-round
# =====
def test_pi_at_sale_matches_period_signal(welfare_df, periods_df):
    """Cross-check: pi_at_sale equals signal on the sold==1 period row."""
    rng = np.random.default_rng(seed=42)
    sample = welfare_df.drop_duplicates(
        subset=["session_id", "segment", "round", "player"]
    ).sample(n=1, random_state=rng.integers(0, 10_000))
    row = sample.iloc[0]

    period_row = periods_df[
        (periods_df["session_id"] == row["session_id"])
        & (periods_df["segment"] == row["segment"])
        & (periods_df["round"] == row["round"])
        & (periods_df["player"] == row["player"])
        & (periods_df["sold"] == 1)
    ]
    assert len(period_row) == 1, (
        f"Expected one sold==1 period row for "
        f"{row['session_id']}/seg{row['segment']}/r{row['round']}/p{row['player']}, "
        f"got {len(period_row)}"
    )
    expected_signal = period_row["signal"].iloc[0]
    assert np.isclose(row["pi_at_sale"], expected_signal, atol=1e-10), (
        f"pi_at_sale {row['pi_at_sale']} != period signal {expected_signal}"
    )


# =====
# Test 14: global_group_id format
# =====
def test_global_group_id_format(welfare_df):
    """global_group_id must equal '{session_id}_{segment}_{group_id}'."""
    expected = (
        welfare_df["session_id"].astype(str)
        + "_"
        + welfare_df["segment"].astype(str)
        + "_"
        + welfare_df["group_id"].astype(str)
    )
    mismatches = welfare_df["global_group_id"] != expected
    assert not mismatches.any(), (
        f"{mismatches.sum()} rows have malformed global_group_id; "
        f"example: {welfare_df.loc[mismatches, 'global_group_id'].iloc[0]} "
        f"vs expected {expected[mismatches].iloc[0]}"
    )


# =====
# Test 15: welfare is constant within a group-round
# =====
def test_welfare_constant_within_group_round(welfare_df):
    """All seller-rows sharing (session, segment, round, group, alpha) must have the same welfare."""
    grouped = welfare_df.groupby(
        ["session_id", "segment", "round", "group_id", "alpha"]
    )["welfare"].nunique()
    assert (grouped == 1).all(), (
        f"welfare varies within group-round for {(grouped > 1).sum()} groups"
    )


# =====
# Test 16: welfare matches group_round_welfare source for a sampled group-round
# =====
def test_welfare_matches_source(welfare_df):
    """Cross-check: welfare equals group_round_welfare source on sampled rows."""
    source = pd.read_csv(DATASTORE / "derived" / "group_round_welfare.csv")
    sample = welfare_df.drop_duplicates(
        subset=["session_id", "segment", "round", "group_id"]
    ).sample(n=3, random_state=42)
    for _, row in sample.iterrows():
        src = source[
            (source["session"] == row["session_id"])
            & (source["segment_num"] == row["segment"])
            & (source["round_num"] == row["round"])
            & (source["group_id"] == row["group_id"])
        ]
        assert len(src) == 1 and np.isclose(row["welfare"], src["welfare"].iloc[0])


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
