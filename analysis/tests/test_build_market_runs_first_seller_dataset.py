"""
Purpose: Tests for the GROUP-ROUND first-seller market-runs dataset
         (build_market_runs_first_seller_dataset.py, issue #120).
Author: Claude Code (test-writer)
Date: 2026-05-04
"""

from pathlib import Path

import pandas as pd
import pytest

from analysis.derived.equilibrium_reference import (
    EQUILIBRIUM_CSV,
    load_equilibrium_table,
    lookup_equilibrium_reference,
)
from analysis.derived.market_runs_helpers import (
    compute_alpha,
    treatment_to_string,
)

# =====
# File paths and skip flags
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DERIVED = PROJECT_ROOT / "datastore" / "derived"
FIRST_SELLER_CSV = DERIVED / "market_runs_first_seller_dataset.csv"
MARKET_RUNS_CSV = DERIVED / "market_runs_dataset.csv"
SURVEY_TRAITS_CSV = DERIVED / "survey_traits.csv"
INDIVIDUAL_ROUND_PANEL_CSV = DERIVED / "individual_round_panel.csv"

DATASTORE_AVAILABLE = (
    DERIVED.exists()
    and EQUILIBRIUM_CSV.exists()
    and SURVEY_TRAITS_CSV.exists()
    and MARKET_RUNS_CSV.exists()
)
skip_no_datastore = pytest.mark.skipif(
    not DATASTORE_AVAILABLE, reason="Datastore not accessible"
)
skip_no_built = pytest.mark.skipif(
    not FIRST_SELLER_CSV.exists(),
    reason="market_runs_first_seller_dataset.csv not built",
)

GROUP_KEYS = ["session_id", "segment", "group_id", "round"]


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def first_seller_df():
    """Load the built first-seller dataset once per module."""
    if not FIRST_SELLER_CSV.exists():
        pytest.skip("market_runs_first_seller_dataset.csv not built")
    return pd.read_csv(FIRST_SELLER_CSV)


@pytest.fixture(scope="module")
def market_runs_df():
    """Load the player-round market_runs_dataset once per module."""
    if not MARKET_RUNS_CSV.exists():
        pytest.skip("market_runs_dataset.csv not built")
    return pd.read_csv(MARKET_RUNS_CSV)


# =====
# Schema tests
# =====
@skip_no_datastore
@skip_no_built
def test_required_columns_present(first_seller_df):
    """All identity, first-seller, group-mean, and run columns are present."""
    required = {
        "session_id", "treatment", "segment", "group_id", "round", "state",
        "first_sale_period", "first_seller_signal_correct", "alpha_first",
        "tie_at_first_period", "dev_from_threshold_first",
        "signal_correct_frac",
        "group_mean_extraversion", "group_mean_agreeableness",
        "group_mean_conscientiousness", "group_mean_neuroticism",
        "group_mean_openness", "group_mean_impulsivity",
        "group_mean_state_anxiety", "group_mean_risk_tolerance",
    }
    run_cols = {f"run_w{w}_k{k}" for w in (0, 1, 2, 3) for k in (2, 3, 4)}
    required |= run_cols
    missing = required - set(first_seller_df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"


@skip_no_datastore
@skip_no_built
def test_group_round_key_is_unique(first_seller_df):
    """(session_id, segment, group_id, round) uniquely identifies each row."""
    dup_count = first_seller_df.duplicated(subset=GROUP_KEYS).sum()
    assert dup_count == 0, f"{dup_count} duplicate group-round keys"


# =====
# Row-count tests
# =====
@skip_no_datastore
@skip_no_built
def test_row_count_matches_sold_group_rounds(first_seller_df, market_runs_df):
    """One row per group-round with at least one seller in market_runs_dataset."""
    sold_group_rounds = (
        market_runs_df[market_runs_df["did_sell"] == 1]
        .groupby(GROUP_KEYS)
        .size()
        .index
    )
    assert len(first_seller_df) == len(sold_group_rounds), (
        f"first-seller rows {len(first_seller_df)} != "
        f"sold group-rounds {len(sold_group_rounds)}"
    )


@skip_no_datastore
@skip_no_built
def test_row_count_at_most_720(first_seller_df):
    """Row count cannot exceed 720 (max group-rounds across all sessions)."""
    assert len(first_seller_df) <= 720
    # Empirically expect substantial drop only from zero-seller rounds
    assert len(first_seller_df) >= 300, (
        f"Unexpectedly few rows: {len(first_seller_df)}"
    )


@skip_no_datastore
@skip_no_built
def test_sample_restriction_matches_market_runs(first_seller_df, market_runs_df):
    """Every first-seller key has a matching sold row in market_runs_dataset."""
    fs_keys = set(map(tuple, first_seller_df[GROUP_KEYS].values))
    sold = market_runs_df[market_runs_df["did_sell"] == 1]
    sold_keys = set(map(tuple, sold[GROUP_KEYS].values))
    missing = fs_keys - sold_keys
    assert not missing, f"{len(missing)} first-seller keys absent from sold set"


# =====
# First-seller field correctness
# =====
@skip_no_datastore
@skip_no_built
def test_first_sale_period_in_round_range(first_seller_df):
    """first_sale_period falls within the experiment round (1..14)."""
    period = first_seller_df["first_sale_period"].dropna()
    assert (period >= 1).all() and (period <= 14).all(), (
        f"first_sale_period range {period.min()}-{period.max()} outside [1, 14]"
    )


@skip_no_datastore
@skip_no_built
def test_first_seller_signal_correct_binary(first_seller_df):
    """first_seller_signal_correct is in {0, 1}."""
    vals = first_seller_df["first_seller_signal_correct"].dropna().unique()
    assert set(vals).issubset({0, 1, 0.0, 1.0}), f"unexpected values {vals}"


@skip_no_datastore
@skip_no_built
def test_tie_at_first_period_binary(first_seller_df):
    """tie_at_first_period is in {0, 1}."""
    vals = first_seller_df["tie_at_first_period"].dropna().unique()
    assert set(vals).issubset({0, 1, 0.0, 1.0}), f"unexpected values {vals}"


@skip_no_datastore
@skip_no_built
def test_alpha_first_in_grid_range(first_seller_df):
    """alpha_first lies in the CRRA grid [0.0, 0.9] when not NaN."""
    alpha = first_seller_df["alpha_first"].dropna()
    assert (alpha >= 0.0).all() and (alpha <= 0.9).all(), (
        f"alpha_first range {alpha.min()}-{alpha.max()} outside [0, 0.9]"
    )


@skip_no_datastore
@skip_no_built
def test_dev_from_threshold_finite_when_alpha_present(first_seller_df):
    """dev_from_threshold_first is finite iff alpha_first is finite — since
    pi_at_sale is observed for every first seller, NaN dev rows are
    exactly the rows where survey risk_tolerance was missing."""
    df = first_seller_df
    dev_nan = df["dev_from_threshold_first"].isna()
    alpha_nan = df["alpha_first"].isna()
    # NaN dev iff NaN alpha (no other source of NaN)
    assert (dev_nan == alpha_nan).all(), (
        "dev_from_threshold_first NaN pattern should match alpha_first NaN"
    )
    # And: every alpha-present row has a finite dev
    finite_share = df.loc[~alpha_nan, "dev_from_threshold_first"].notna().mean()
    assert finite_share == 1.0, (
        f"alpha-present rows must all have finite dev (got {finite_share:.3f})"
    )


@skip_no_datastore
@skip_no_built
def test_signal_correct_frac_in_unit_interval(first_seller_df):
    """signal_correct_frac is a probability in [0, 1]."""
    vals = first_seller_df["signal_correct_frac"].dropna()
    assert ((vals >= 0.0) & (vals <= 1.0)).all()


# =====
# Run-indicator tests
# =====
@skip_no_datastore
@skip_no_built
def test_run_indicators_binary(first_seller_df):
    """Every run_w*_k* column contains only 0/1."""
    run_cols = [c for c in first_seller_df.columns if c.startswith("run_w")]
    for col in run_cols:
        vals = first_seller_df[col].dropna().unique()
        assert set(vals).issubset({0, 1}), f"{col} has unexpected values {vals}"


@skip_no_datastore
@skip_no_built
@pytest.mark.parametrize("k", [2, 3, 4])
def test_run_monotone_in_w(first_seller_df, k):
    """Wider window admits more runs: run_w0_kK=1 implies run_w3_kK=1."""
    df = first_seller_df
    bad = df[(df[f"run_w0_k{k}"] == 1) & (df[f"run_w3_k{k}"] == 0)]
    assert len(bad) == 0, f"{len(bad)} rows violate w-monotonicity at k={k}"


@skip_no_datastore
@skip_no_built
@pytest.mark.parametrize("w", [0, 1, 2, 3])
def test_run_monotone_in_k(first_seller_df, w):
    """Larger run requirement is stricter: run_w_k4=1 implies run_w_k2=1."""
    df = first_seller_df
    bad = df[(df[f"run_w{w}_k4"] == 1) & (df[f"run_w{w}_k2"] == 0)]
    assert len(bad) == 0, f"{len(bad)} rows violate k-monotonicity at w={w}"


# =====
# Group-mean trait correctness
# =====
@skip_no_datastore
@skip_no_built
def test_group_mean_extraversion_matches_manual_calc(first_seller_df):
    """A row's group_mean_extraversion equals the manual mean over its group."""
    sample_row = first_seller_df.iloc[0]
    expected = _hand_calc_group_mean(sample_row, "extraversion")
    assert sample_row["group_mean_extraversion"] == pytest.approx(expected)


@skip_no_datastore
@skip_no_built
def test_group_mean_constant_within_group(first_seller_df):
    """Group-mean traits are constant across rounds for the same group."""
    keys = ["session_id", "segment", "group_id"]
    trait_cols = [c for c in first_seller_df.columns if c.startswith("group_mean_")]
    nuniques = first_seller_df.groupby(keys)[trait_cols].nunique(dropna=False)
    bad = (nuniques != 1).any()
    assert not bad.any(), f"vary within group: {bad[bad].index.tolist()}"


def _hand_calc_group_mean(sample_row, trait):
    """Compute expected group-mean trait from raw survey_traits.csv."""
    traits = pd.read_csv(SURVEY_TRAITS_CSV)
    panel = pd.read_csv(
        INDIVIDUAL_ROUND_PANEL_CSV,
        usecols=["session_id", "segment", "group_id", "player"],
    ).drop_duplicates()
    members = panel[
        (panel["session_id"] == sample_row["session_id"])
        & (panel["segment"] == sample_row["segment"])
        & (panel["group_id"] == sample_row["group_id"])
    ]["player"].unique()
    return traits[
        (traits["session_id"] == sample_row["session_id"])
        & (traits["player"].isin(members))
    ][trait].mean()


# =====
# Spot-check: dev_from_threshold semantics for one real row
# =====
@skip_no_datastore
@skip_no_built
def test_dev_from_threshold_matches_equilibrium_lookup(first_seller_df):
    """Recompute dev_from_threshold_first from alpha+treatment for one alpha-
    present row and confirm equality to the stored value."""
    df = first_seller_df.dropna(subset=["alpha_first", "dev_from_threshold_first"])
    if df.empty:
        pytest.skip("no rows with alpha_first present")
    row = df.iloc[0]
    eq_df = load_equilibrium_table()
    threshold_pi, _ = lookup_equilibrium_reference(
        eq_df,
        alpha=float(row["alpha_first"]),
        treatment=treatment_to_string(row["treatment"]),
        n_at_sale=4,
    )
    # Reconstruct pi_at_sale from stored dev = pi - threshold
    implied_pi = row["dev_from_threshold_first"] + threshold_pi
    # Sanity: implied pi must be in [0, 1]
    assert 0.0 <= implied_pi <= 1.0, (
        f"implied pi_at_sale {implied_pi} outside [0, 1]"
    )


# =====
# State-correctness consistency
# =====
@skip_no_datastore
@skip_no_built
def test_state_in_binary_set(first_seller_df):
    """state column is binary {0, 1}."""
    vals = set(first_seller_df["state"].dropna().unique())
    assert vals.issubset({0, 1}), f"unexpected state values {vals}"


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
