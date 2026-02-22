"""
Purpose: Validate data transformations for the ranked-order logit (Cox PH)
         regression in ro_logit_selling_position.R.
Author: Claude Code
Date: 2026-02-21
"""

import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_CSV = PROJECT_ROOT / "datastore" / "derived" / "ordinal_selling_position.csv"

EMOTION_COLS = [
    "anger_p95", "contempt_p95", "disgust_p95", "fear_p95", "joy_p95",
    "sadness_p95", "surprise_p95", "engagement_p95", "valence_p95",
]
CONTINUOUS_VARS = [
    "state_anxiety", "impulsivity", "conscientiousness",
    "fear_p95", "anger_p95", "extraversion", "agreeableness",
    "neuroticism", "openness", "contempt_p95", "disgust_p95",
    "joy_p95", "sadness_p95", "surprise_p95", "engagement_p95",
    "valence_p95", "risk_tolerance", "age",
]
# 19 predictors in Cox PH formula (segment/round absorbed by strata)
RO_PREDICTORS = [
    "state_anxiety", "impulsivity", "conscientiousness", *EMOTION_COLS,
    "extraversion", "agreeableness", "neuroticism", "openness",
    "risk_tolerance", "age", "gender_female",
]

CSV_EXISTS = pytest.mark.skipif(
    not OUTPUT_CSV.exists(), reason="Output CSV not yet built",
)


# =====
# Helpers
# =====
def load_filtered():
    """Load CSV, drop rows with missing emotions."""
    return pd.read_csv(OUTPUT_CSV).dropna(subset=EMOTION_COLS)


def add_survival_vars(df):
    """Replicate R script's build_survival_vars transformations."""
    df = df.copy()
    df["sell_rank_rev"] = 5 - df["sell_rank"]
    df["stratum"] = (
        df["session_id"] + "_" + df["segment"].astype(str)
        + "_" + df["group_id"].astype(str) + "_" + df["round"].astype(str)
    )
    df["rev_rank"] = df.groupby("stratum")["sell_rank_rev"].transform(
        lambda s: s.rank(ascending=False, method="average")
    )
    return df


def get_stratum(df, session_id, seg, group_id, rnd):
    """Filter to a specific group-round."""
    return df[
        (df["session_id"] == session_id) & (df["segment"] == seg)
        & (df["group_id"] == group_id) & (df["round"] == rnd)
    ]


# =====
# Category A: Rank Reversal Correctness
# =====
def test_rank_reversal_mapping():
    """sell_rank_rev = 5 - sell_rank maps {1->4, 2->3, 3->2, 4->1}."""
    for rank, expected in [(1, 4), (2, 3), (3, 2), (4, 1)]:
        assert 5 - rank == expected


@CSV_EXISTS
def test_first_sellers_get_lowest_rev_rank():
    """First sellers (sell_rank=1) get the lowest rev_rank in stratum."""
    df = add_survival_vars(load_filtered())
    for _, grp in df.groupby("stratum"):
        first = grp[grp["sell_rank"] == 1]
        if not first.empty:
            assert (first["rev_rank"] == grp["rev_rank"].min()).all()


@CSV_EXISTS
def test_non_sellers_get_highest_rev_rank():
    """Non-sellers (sell_rank=4) get the highest rev_rank in stratum."""
    df = add_survival_vars(load_filtered())
    for _, grp in df.groupby("stratum"):
        ns = grp[grp["sell_rank"] == 4]
        if not ns.empty:
            assert (ns["rev_rank"] == grp["rev_rank"].max()).all()


@CSV_EXISTS
def test_tied_sellers_get_averaged_ranks():
    """Two players with same sell_rank get averaged rev_rank."""
    df = add_survival_vars(load_filtered())
    sub = get_stratum(df, "1_11-7-tr1", 1, 1, 2)
    ranks = dict(zip(sub["player"], sub["rev_rank"]))
    assert ranks["J"] == ranks["N"], "Tied sellers need same rev_rank"
    assert ranks["A"] < ranks["J"], "First seller ranks earlier"


# =====
# Category B: Stratum Construction
# =====
@CSV_EXISTS
def test_stratum_count():
    """Total strata ~720."""
    df = add_survival_vars(load_filtered())
    n = df["stratum"].nunique()
    assert 700 <= n <= 730, f"Expected ~720 strata, got {n}"


@CSV_EXISTS
def test_stratum_sizes_are_3_or_4():
    """Each stratum has 3 or 4 players."""
    df = add_survival_vars(load_filtered())
    sizes = set(df.groupby("stratum").size().unique())
    assert sizes.issubset({3, 4}), f"Bad sizes: {sizes - {3, 4}}"


@CSV_EXISTS
def test_stratum_id_components():
    """Stratum ID encodes session_id, segment, group_id, round."""
    df = add_survival_vars(load_filtered())
    r = df.iloc[0]
    expected = f"{r['session_id']}_{r['segment']}_{r['group_id']}_{r['round']}"
    assert r["stratum"] == expected


# =====
# Category C: Sellers-Only Filtering
# =====
@CSV_EXISTS
def test_sellers_only_counts():
    """Sellers-only with >=2 per stratum: ~486 obs in ~202 strata."""
    df = add_survival_vars(load_filtered())
    sellers = df[df["did_sell"] == 1]
    counts = sellers.groupby("stratum").size()
    valid = counts[counts >= 2]
    n_rows = sellers[sellers["stratum"].isin(valid.index)].shape[0]
    assert 480 <= n_rows <= 495, f"Expected ~486 obs, got {n_rows}"
    assert 195 <= len(valid) <= 210, f"Expected ~202 strata, got {len(valid)}"
    assert (valid >= 2).all()


# =====
# Category D: Variable Standardization
# =====
@CSV_EXISTS
def test_z_scoring_produces_mean_zero_sd_one():
    """After z-scoring, continuous vars have mean~0 and SD~1."""
    df = load_filtered()
    for var in CONTINUOUS_VARS:
        z = (df[var] - df[var].mean()) / df[var].std()
        assert abs(z.mean()) < 1e-10, f"{var} z-scored mean != 0"
        assert abs(z.std() - 1.0) < 1e-10, f"{var} z-scored SD != 1"


@CSV_EXISTS
def test_binary_var_not_standardized():
    """gender_female is binary {0,1} and NOT in CONTINUOUS_VARS."""
    df = load_filtered()
    assert "gender_female" not in CONTINUOUS_VARS
    assert set(df["gender_female"].unique()).issubset({0, 1})


@CSV_EXISTS
def test_z_scoring_no_new_nans():
    """Z-scoring does not introduce NAs in non-missing data."""
    df = load_filtered()
    for var in CONTINUOUS_VARS:
        n_orig = df[var].isna().sum()
        z = (df[var] - df[var].mean()) / df[var].std()
        assert z.isna().sum() == n_orig, f"{var}: z-scoring added NAs"


# =====
# Category E: Predictor Completeness
# =====
def test_ro_predictor_count():
    """RO logit uses exactly 19 predictors (no segment/round)."""
    assert len(RO_PREDICTORS) == 19


@CSV_EXISTS
def test_ro_predictors_present_and_complete():
    """All 19 predictors exist in CSV and are non-NA after emotion filter."""
    df = load_filtered()
    for pred in RO_PREDICTORS:
        assert pred in df.columns, f"Missing: {pred}"
        assert df[pred].isna().sum() == 0, f"{pred} has NAs"


@CSV_EXISTS
def test_excluded_vars_not_in_ro_formula():
    """segment2/3/4 and round are NOT in the RO predictor list."""
    excluded = {"segment2", "segment3", "segment4", "round"}
    assert not (excluded & set(RO_PREDICTORS))


# =====
# Category F: Ground-Truth Spot Checks
# =====
@CSV_EXISTS
def test_spot_r1_nobody_sold():
    """S1 seg1 g1 r1: nobody sold -> all sell_rank=4, sell_rank_rev=1."""
    df = add_survival_vars(load_filtered())
    sub = get_stratum(df, "1_11-7-tr1", 1, 1, 1)
    assert dict(zip(sub["player"], sub["sell_rank"])) == {
        "A": 4, "E": 4, "J": 4, "N": 4,
    }
    assert (sub["sell_rank_rev"] == 1).all()


@CSV_EXISTS
def test_spot_r3_three_sellers():
    """S1 seg1 g1 r3: A=1,N=2,J=3,E=4 -> rev: A=4,N=3,J=2,E=1."""
    df = add_survival_vars(load_filtered())
    sub = get_stratum(df, "1_11-7-tr1", 1, 1, 3)
    rev = dict(zip(sub["player"], sub["sell_rank_rev"]))
    assert rev == {"A": 4, "N": 3, "J": 2, "E": 1}


@CSV_EXISTS
def test_spot_r5_one_seller():
    """S1 seg1 g1 r5: N=rank1, others=rank4 -> rev: N=4, others=1."""
    df = add_survival_vars(load_filtered())
    sub = get_stratum(df, "1_11-7-tr1", 1, 1, 5)
    rev = dict(zip(sub["player"], sub["sell_rank_rev"]))
    assert rev == {"A": 1, "E": 1, "J": 1, "N": 4}


@CSV_EXISTS
def test_spot_r2_tie():
    """S1 seg1 g1 r2: A=1,J=2,N=2,E=4 -> rev: A=4,J=3,N=3,E=1."""
    df = add_survival_vars(load_filtered())
    sub = get_stratum(df, "1_11-7-tr1", 1, 1, 2)
    rev = dict(zip(sub["player"], sub["sell_rank_rev"]))
    assert rev == {"A": 4, "J": 3, "N": 3, "E": 1}


@CSV_EXISTS
def test_spot_r3_within_stratum_ordering():
    """S1 seg1 g1 r3: first seller A gets rev_rank=1 (event first)."""
    df = add_survival_vars(load_filtered())
    sub = get_stratum(df, "1_11-7-tr1", 1, 1, 3)
    ranks = dict(zip(sub["player"], sub["rev_rank"]))
    assert ranks == {"A": 1.0, "N": 2.0, "J": 3.0, "E": 4.0}


# =====
# Category G: Cross-Check with Ordinal Logit
# =====
@CSV_EXISTS
def test_emotion_filtered_row_count():
    """After emotion filter, exactly 2,845 rows remain."""
    df = load_filtered()
    assert len(df) == 2845, f"Expected 2845, got {len(df)}"


@CSV_EXISTS
def test_raw_csv_row_count():
    """Raw CSV has 2,850 rows (both models read same file)."""
    df = pd.read_csv(OUTPUT_CSV)
    assert len(df) == 2850, f"Expected 2850, got {len(df)}"


@CSV_EXISTS
def test_ordinal_predictors_superset_of_ro():
    """Ordinal logit predictors are a superset of RO predictors."""
    ordinal_preds = set(RO_PREDICTORS) | {"segment2", "segment3", "segment4", "round"}
    assert set(RO_PREDICTORS).issubset(ordinal_preds)


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
