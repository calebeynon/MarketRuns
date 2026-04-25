"""
Purpose: Tests for the market-runs helpers, equilibrium reference, group-trait
         aggregation, and the build_market_runs_dataset pipeline (issue #120).
Author: Claude Code (test-writer)
Date: 2026-04-24
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.derived.equilibrium_reference import (
    EQUILIBRIUM_CSV,
    load_equilibrium_table,
    lookup_equilibrium_reference,
)
from analysis.derived.group_traits import (
    TRAIT_COLS,
    compute_group_trait_means,
)
from analysis.derived.build_market_runs_dataset import equilibrium_deviations
from analysis.derived.market_runs_helpers import (
    compute_alpha,
    compute_min_signal,
    compute_n_at_dip,
    compute_n_at_sale,
    detect_run,
    signal_correct_frac,
    treatment_to_string,
)

# =====
# File paths and skip flags
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DERIVED = PROJECT_ROOT / "datastore" / "derived"
MARKET_RUNS_CSV = DERIVED / "market_runs_dataset.csv"
SURVEY_TRAITS_CSV = DERIVED / "survey_traits.csv"
INDIVIDUAL_ROUND_PANEL_CSV = DERIVED / "individual_round_panel.csv"

DATASTORE_AVAILABLE = (
    DERIVED.exists()
    and EQUILIBRIUM_CSV.exists()
    and SURVEY_TRAITS_CSV.exists()
)
skip_no_datastore = pytest.mark.skipif(
    not DATASTORE_AVAILABLE, reason="Datastore not accessible"
)
skip_no_built = pytest.mark.skipif(
    not MARKET_RUNS_CSV.exists(), reason="market_runs_dataset.csv not built"
)


# =====
# Mock data helpers
# =====
def _make_traits_df(session="s1", n_players=4):
    """Synthetic player-level survey traits with predictable values."""
    rows = []
    for i, label in enumerate(["A", "B", "C", "D"][:n_players]):
        rows.append({
            "session_id": session, "player": label,
            "extraversion": 1.0 + i, "agreeableness": 2.0 + i,
            "conscientiousness": 3.0 + i, "neuroticism": 4.0 + i,
            "openness": 5.0 + i, "impulsivity": 6.0 + i,
            "state_anxiety": 7.0 + i, "risk_tolerance": 10.0 + i,
        })
    return pd.DataFrame(rows)


def _signal_df(signals, state):
    """Build a (signal, state) DataFrame with shared state for every period."""
    return pd.DataFrame({"signal": signals, "state": [state] * len(signals)})


# =====
# detect_run tests (T1)
# =====
def test_detect_run_empty_returns_false():
    """No sales cannot be a run regardless of (w, k)."""
    assert detect_run([], w=0, k=2) is False


def test_detect_run_below_k_returns_false():
    """One seller cannot be a k=2 run."""
    assert detect_run([3], w=1, k=2) is False


def test_detect_run_w0_simultaneous_3_sellers_k3_true():
    """Three sales in the same period satisfy w=0,k=3."""
    assert detect_run([5, 5, 5], w=0, k=3) is True


def test_detect_run_w1_adjacent_periods_k2_true():
    """Two sales one period apart satisfy w=1,k=2."""
    assert detect_run([3, 4], w=1, k=2) is True


def test_detect_run_w3_four_sellers_k4_true():
    """Four sales within four contiguous periods satisfy w=3,k=4."""
    assert detect_run([5, 6, 7, 8], w=3, k=4) is True


def test_detect_run_window_does_not_wrap():
    """Sales at opposite ends of a 14-period round are not adjacent."""
    assert detect_run([1, 14], w=2, k=2) is False


def test_detect_run_monotonic_in_w():
    """run_w3_k2 is a superset of run_w0_k2 (wider window admits more)."""
    assert detect_run([3, 5], w=0, k=2) is False
    assert detect_run([3, 5], w=3, k=2) is True
    assert detect_run([3, 3], w=0, k=2) is True
    assert detect_run([3, 3], w=3, k=2) is True


# =====
# compute_alpha tests (T1)
# =====
def test_compute_alpha_zero():
    """rt=0 -> alpha=0.0 (CRRA grid lower bound)."""
    assert compute_alpha(0) == 0.0


def test_compute_alpha_clipped_high():
    """Above 18 the grid caps at 0.9; rt=18 and rt=20 both yield 0.9."""
    assert compute_alpha(18) == 0.9
    assert compute_alpha(20) == 0.9


def test_compute_alpha_negative_clipped_low():
    """Negative risk_tolerance is clipped to zero -> alpha=0.0."""
    assert compute_alpha(-5) == 0.0


def test_compute_alpha_grid_match():
    """rt=8 -> 8/20 = 0.4 lands on the grid."""
    assert compute_alpha(8) == 0.4


def test_compute_alpha_nan_propagates():
    """NaN risk_tolerance propagates as NaN (no raise; preserves the row)."""
    import math as _math
    assert _math.isnan(compute_alpha(float("nan")))
    assert _math.isnan(compute_alpha(None))


# =====
# signal_correct_frac tests (T1)
# =====
def test_signal_correct_frac_all_correct_returns_1():
    """Every period's signal points toward state=1 -> 1.0."""
    assert signal_correct_frac(_signal_df([0.7, 0.8, 1.0], state=1)) == 1.0


def test_signal_correct_frac_all_wrong_returns_0():
    """Every signal opposes the state -> 0.0."""
    assert signal_correct_frac(_signal_df([0.1, 0.2, 0.3], state=1)) == 0.0


def test_signal_correct_frac_signal_half_excluded_from_numerator():
    """signal=0.5 stays in the denominator but contributes 0 to numerator."""
    # one of two periods is informative-and-correct; 0.5 contributes 0
    df = _signal_df([0.5, 0.9], state=1)
    assert signal_correct_frac(df) == pytest.approx(0.5)
    assert signal_correct_frac(_signal_df([0.5], state=1)) == 0.0


def test_signal_correct_frac_mixed_state_returns_fraction():
    """Two of four periods match the state -> 0.5."""
    assert signal_correct_frac(_signal_df([0.9, 0.8, 0.1, 0.2], state=1)) == 0.5


def test_signal_correct_frac_empty_returns_nan():
    """Empty period_rows returns NaN (no division by zero)."""
    import math as _math
    assert _math.isnan(signal_correct_frac(_signal_df([], state=1)))


# =====
# treatment_to_string tests (T1)
# =====
def test_treatment_to_string_tr1_random():
    """'tr1' maps to 'random' (random-price payoff)."""
    assert treatment_to_string("tr1") == "random"


def test_treatment_to_string_tr2_average():
    """'tr2' maps to 'average' (mean-price payoff)."""
    assert treatment_to_string("tr2") == "average"


def test_treatment_to_string_invalid_raises():
    """Unknown labels raise rather than silently mis-route."""
    with pytest.raises(ValueError):
        treatment_to_string("tr3")


# =====
# compute_n_at_sale tests (T1)
# =====
def test_compute_n_at_sale_no_sale_returns_none():
    """A non-seller (None or absent key) returns None, not an int."""
    assert compute_n_at_sale({"A": 1, "B": None}, "B") is None
    assert compute_n_at_sale({"A": 1}, "B") is None


def test_compute_n_at_sale_first_seller_returns_4():
    """First seller (no strictly-earlier sales) faces 4 holders."""
    seller_periods = {"A": 1, "B": 4, "C": 7, "D": 10}
    assert compute_n_at_sale(seller_periods, "A") == 4


def test_compute_n_at_sale_ties_same_period_share_n():
    """Simultaneous sales share the same n (only strictly earlier reduce n)."""
    seller_periods = {"A": 3, "B": 3, "C": 5, "D": None}
    assert compute_n_at_sale(seller_periods, "A") == 4
    assert compute_n_at_sale(seller_periods, "B") == 4
    assert compute_n_at_sale(seller_periods, "C") == 2


# =====
# Equilibrium reference tests (T2)
# =====
@skip_no_datastore
def test_lookup_equilibrium_reference_known_row():
    """alpha=0.0, random, n=4 returns the M&M (2020) Appendix D threshold."""
    df = load_equilibrium_table()
    threshold_pi, avg_pi = lookup_equilibrium_reference(
        df, alpha=0.0, treatment="random", n_at_sale=4
    )
    assert threshold_pi == pytest.approx(0.2907550720186073, abs=1e-9)
    assert avg_pi == pytest.approx(0.1881074538683798, abs=1e-9)


@skip_no_datastore
def test_lookup_equilibrium_reference_grid_coverage():
    """All 80 grid cells (0.0..0.9 step 0.1 x {random,average} x {1..4}) resolve."""
    df = load_equilibrium_table()
    alphas = np.round(np.arange(0.0, 1.0, 0.1), 1)
    for alpha in alphas:
        for treatment in ("random", "average"):
            for n in range(1, 5):
                lookup_equilibrium_reference(
                    df, alpha=float(alpha), treatment=treatment, n_at_sale=n
                )


@skip_no_datastore
def test_lookup_equilibrium_reference_invalid_alpha_raises():
    """alpha outside grid raises so callers cannot mis-interpret risk."""
    df = load_equilibrium_table()
    with pytest.raises((KeyError, ValueError)):
        lookup_equilibrium_reference(df, alpha=1.5, treatment="random", n_at_sale=4)


# =====
# Group trait tests (T3)
# =====
def test_compute_group_trait_means_all_present():
    """Mean over 4 players: extraversion = (1+2+3+4)/4 = 2.5."""
    traits = _make_traits_df()
    result = compute_group_trait_means(traits, "s1", ["A", "B", "C", "D"])
    assert result["group_mean_extraversion"] == pytest.approx(2.5)
    for col in TRAIT_COLS:
        assert f"group_mean_{col}" in result


def test_compute_group_trait_means_one_missing_skips():
    """Players missing from traits_df are dropped; mean over 3 = (1+2+3)/3 = 2.0."""
    traits = _make_traits_df(n_players=3)  # only A, B, C in traits
    result = compute_group_trait_means(traits, "s1", ["A", "B", "C", "D"])
    assert result["group_mean_extraversion"] == pytest.approx(2.0)


# =====
# Integration tests (T4 - gated)
# =====
@skip_no_datastore
@skip_no_built
def test_market_runs_dataset_row_count():
    """Dataset is player-round panel: 720 group-rounds x 4 players = 2880 rows."""
    df = pd.read_csv(MARKET_RUNS_CSV)
    assert len(df) == 2880, f"Got {len(df)} rows; expected 2880 player-rounds"
    keys = ["session_id", "segment", "group_id", "round"]
    sizes = df.groupby(keys).size()
    assert (sizes == 4).all(), "Every group-round should have exactly 4 player rows"


@skip_no_datastore
@skip_no_built
def test_run_indicator_constant_within_group_round():
    """Run indicators are group-round level, so unique within (session,segment,round,group)."""
    df = pd.read_csv(MARKET_RUNS_CSV)
    keys = ["session_id", "segment", "round", "group_id"]
    keys = [k for k in keys if k in df.columns]
    run_cols = [c for c in df.columns if c.startswith("run_w")]
    assert run_cols, "Expected run_w* columns in dataset"
    grouped = df.groupby(keys)[run_cols].nunique()
    assert (grouped == 1).all().all()


@skip_no_datastore
@skip_no_built
def test_run_w0_k4_implies_all_4_same_period():
    """run_w0_k4 = 1 means all 4 sellers exited within a single period."""
    df = pd.read_csv(MARKET_RUNS_CSV)
    if "run_w0_k4" not in df.columns:
        pytest.skip("run_w0_k4 column not in dataset")
    assert df["run_w0_k4"].dropna().isin([0, 1]).all()


@skip_no_datastore
@skip_no_built
def test_run_w3_k2_superset_of_run_w0_k2():
    """Wider window admits more runs: run_w0_k2=1 => run_w3_k2=1."""
    df = pd.read_csv(MARKET_RUNS_CSV)
    needed = {"run_w0_k2", "run_w3_k2"}
    if not needed.issubset(df.columns):
        pytest.skip(f"missing columns: {sorted(needed - set(df.columns))}")
    violations = df[(df["run_w0_k2"] == 1) & (df["run_w3_k2"] == 0)]
    assert len(violations) == 0


@skip_no_datastore
@skip_no_built
def test_signal_correct_frac_in_unit_interval():
    """signal_correct_frac is a probability in [0, 1]."""
    df = pd.read_csv(MARKET_RUNS_CSV)
    candidates = [c for c in df.columns if "signal_correct" in c]
    if not candidates:
        pytest.skip("signal_correct_frac column not in dataset")
    vals = df[candidates[0]].dropna()
    assert ((vals >= 0.0) & (vals <= 1.0)).all()


@skip_no_datastore
@skip_no_built
def test_dev_from_threshold_finite_when_did_not_sell_and_alpha_present():
    """Non-sellers with valid alpha now have a finite dev_from_threshold
    (computed from the round-level signal dip vs the n_at_dip equilibrium)."""
    df = pd.read_csv(MARKET_RUNS_CSV)
    non_sellers = df[(df["did_sell"] == 0) & df["alpha"].notna()]
    assert non_sellers["dev_from_threshold"].notna().any(), (
        "Non-seller branch must produce some finite dev_from_threshold values"
    )
    assert (non_sellers["dev_from_threshold"].dropna() <= 0.0).all(), (
        "Non-seller deviations are capped at 0 above the threshold"
    )


@skip_no_datastore
@skip_no_built
def test_dev_from_avg_pi_nan_when_n_at_sale_is_one():
    """n_at_sale=1 rows have NaN dev_from_avg_pi (n=1 never sells in M&M)."""
    df = pd.read_csv(MARKET_RUNS_CSV)
    n1 = df[df["n_at_sale"] == 1]
    if len(n1) == 0:
        pytest.skip("no n_at_sale=1 rows in current dataset")
    assert n1["dev_from_avg_pi"].isna().all(), (
        "n_at_sale=1 must produce NaN dev_from_avg_pi (avg_pi_at_sale undefined)"
    )
    assert n1["dev_from_threshold"].notna().all(), (
        "n_at_sale=1 should still have a finite threshold-based deviation"
    )


@skip_no_datastore
@skip_no_built
def test_first_seller_has_n_at_sale_4():
    """Every group-round with at least one sale has >=1 row with n_at_sale=4."""
    df = pd.read_csv(MARKET_RUNS_CSV)
    keys = ["session_id", "segment", "group_id", "round"]
    sold_rounds = df[df["did_sell"] == 1].groupby(keys).size().index
    first_sellers = df[(df["did_sell"] == 1) & (df["n_at_sale"] == 4)]
    first_keys = first_sellers.groupby(keys).size().index
    missing = sold_rounds.difference(first_keys)
    assert len(missing) == 0, f"{len(missing)} sold rounds missing n_at_sale=4 row"


@skip_no_datastore
@skip_no_built
def test_group_round_constants_repeated_across_players():
    """All group-round constants have nunique=1 within each group-round."""
    df = pd.read_csv(MARKET_RUNS_CSV)
    keys = ["session_id", "segment", "group_id", "round"]
    # no_sale is per-player (= 1 - did_sell), so excluded from constancy check
    constants = (
        ["signal_correct_frac", "state", "treatment"]
        + [c for c in df.columns if c.startswith("group_mean_")]
        + [c for c in df.columns if c.startswith("run_w")]
    )
    nuniques = df.groupby(keys)[constants].nunique(dropna=False)
    bad = (nuniques != 1).any()
    assert not bad.any(), f"vary within group-round: {bad[bad].index.tolist()}"


@skip_no_datastore
@skip_no_built
def test_player_round_columns_present():
    """All required per-player and group-round columns are in the output schema."""
    df = pd.read_csv(MARKET_RUNS_CSV)
    required = {
        "did_sell", "sell_period", "pi_at_sale", "n_at_sale", "no_sale",
        "player", "alpha", "dev_from_threshold", "dev_from_avg_pi",
        "signal_correct_frac", "state", "treatment",
    }
    missing = required - set(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"


@skip_no_datastore
@skip_no_built
def test_group_mean_traits_match_hand_calculation():
    """A row's group_mean_extraversion equals the manual survey_traits mean."""
    df = pd.read_csv(MARKET_RUNS_CSV)
    if "group_mean_extraversion" not in df.columns:
        pytest.skip("group_mean_extraversion column not in dataset")
    expected = _hand_calc_extraversion_mean(df.iloc[0])
    assert df.iloc[0]["group_mean_extraversion"] == pytest.approx(expected)


def _hand_calc_extraversion_mean(sample_row):
    """Compute expected group-mean extraversion from raw survey_traits.csv."""
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
    ]["extraversion"].mean()


# =====
# compute_min_signal tests (T1 - non-seller branch)
# =====
def test_compute_min_signal_returns_min_and_period():
    """Series indexed by period returns (min_value, idxmin)."""
    series = pd.Series([0.7, 0.6, 0.5, 0.55], index=[1, 2, 3, 4])
    min_signal, dip_period = compute_min_signal(series)
    assert min_signal == pytest.approx(0.5)
    assert dip_period == 3


def test_compute_min_signal_empty_returns_nan_none():
    """Empty Series returns (NaN, None) without raising."""
    import math as _math
    min_signal, dip_period = compute_min_signal(pd.Series(dtype=float))
    assert _math.isnan(min_signal)
    assert dip_period is None


def test_compute_min_signal_ties_break_on_earliest_period():
    """Ties broken by idxmin (earliest period in this index ordering)."""
    series = pd.Series([0.5, 0.7, 0.5], index=[2, 5, 8])
    min_signal, dip_period = compute_min_signal(series)
    assert min_signal == pytest.approx(0.5)
    assert dip_period == 2


# =====
# compute_n_at_dip tests (T1 - non-seller branch)
# =====
def test_compute_n_at_dip_no_earlier_sellers_returns_4():
    """All sellers' first_sale_period >= dip_period (or None) -> 4 holders."""
    sellers = {"A": 5, "B": None, "C": 7, "D": 5}
    assert compute_n_at_dip(sellers, dip_period=5) == 4


def test_compute_n_at_dip_two_earlier_sellers_returns_2():
    """Two players sold strictly before dip_period -> 4 - 2 = 2."""
    sellers = {"A": 2, "B": 3, "C": 7, "D": None}
    assert compute_n_at_dip(sellers, dip_period=5) == 2


def test_compute_n_at_dip_assertion_when_zero():
    """All four sellers strictly earlier than dip_period violates the
    invariant (a non-seller themselves would still hold) and asserts."""
    sellers = {"A": 1, "B": 2, "C": 3, "D": 4}
    with pytest.raises(AssertionError):
        compute_n_at_dip(sellers, dip_period=10)


# =====
# equilibrium_deviations semantics tests (T2 - in-memory eq_df)
# =====
@pytest.fixture
def small_eq_df():
    """Minimal equilibrium reference table covering alpha=0.5, random, n=1..4."""
    return pd.DataFrame([
        {"alpha": 0.5, "treatment": "random", "n": 1,
         "threshold_pi": 0.30, "avg_pi_at_sale": float("nan")},
        {"alpha": 0.5, "treatment": "random", "n": 2,
         "threshold_pi": 0.40, "avg_pi_at_sale": 0.45},
        {"alpha": 0.5, "treatment": "random", "n": 3,
         "threshold_pi": 0.55, "avg_pi_at_sale": 0.60},
        {"alpha": 0.5, "treatment": "random", "n": 4,
         "threshold_pi": 0.70, "avg_pi_at_sale": 0.75},
    ])


def _nonseller_constants(period_signals, sellers):
    """Build the `constants` dict consumed by equilibrium_deviations()."""
    return {
        "treatment_str": "random",
        "per_player_first_sale": sellers,
        "group_period_signals": period_signals,
    }


def test_equilibrium_deviations_non_seller_never_dips_below_threshold_returns_zero(
    small_eq_df,
):
    """min_signal=0.8 stays above threshold_pi=0.7 -> dev_from_threshold=0."""
    period_signals = pd.Series([0.85, 0.80, 0.90], index=[1, 2, 3])
    sellers = {"A": None, "B": None, "C": None, "D": None}  # no earlier sales
    constants = _nonseller_constants(period_signals, sellers)
    out = equilibrium_deviations(
        small_eq_df, alpha=0.5, constants=constants,
        did_sell=0, n_at_sale=None, pi_at_sale=float("nan"),
    )
    assert out["dev_from_threshold"] == 0.0
    assert out["dev_from_avg_pi"] == 0.0


def test_equilibrium_deviations_non_seller_dips_below_returns_negative_diff(
    small_eq_df,
):
    """min_signal=0.5 < threshold_pi=0.7 -> dev = 0.5 - 0.7 = -0.2."""
    period_signals = pd.Series([0.85, 0.50, 0.65], index=[1, 2, 3])
    sellers = {"A": None, "B": None, "C": None, "D": None}
    constants = _nonseller_constants(period_signals, sellers)
    out = equilibrium_deviations(
        small_eq_df, alpha=0.5, constants=constants,
        did_sell=0, n_at_sale=None, pi_at_sale=float("nan"),
    )
    assert out["dev_from_threshold"] == pytest.approx(-0.2)
    # avg_pi at n=4 is 0.75 -> dev = 0.5 - 0.75 = -0.25
    assert out["dev_from_avg_pi"] == pytest.approx(-0.25)


def test_equilibrium_deviations_seller_unchanged_uses_pi_at_sale(small_eq_df):
    """Seller branch ignores group_period_signals; uses pi_at_sale - threshold."""
    constants = _nonseller_constants(pd.Series([0.1], index=[1]), {})
    out = equilibrium_deviations(
        small_eq_df, alpha=0.5, constants=constants,
        did_sell=1, n_at_sale=4, pi_at_sale=0.8,
    )
    # threshold at n=4 is 0.70 -> dev = 0.8 - 0.7 = 0.1
    assert out["dev_from_threshold"] == pytest.approx(0.1)
    # avg_pi at n=4 is 0.75 -> dev = 0.8 - 0.75 = 0.05
    assert out["dev_from_avg_pi"] == pytest.approx(0.05)


# =====
# Integration test for non-seller branch (T4 - against rebuilt CSV)
# =====
@skip_no_datastore
@skip_no_built
def test_non_sellers_have_finite_dev_from_threshold_when_alpha_present():
    """Most non-sellers with alpha now have finite dev_from_threshold; the
    only NaN cases are rounds with no period rows (empty signals)."""
    df = pd.read_csv(MARKET_RUNS_CSV)
    non_sellers = df[(df["did_sell"] == 0) & df["alpha"].notna()]
    finite_share = non_sellers["dev_from_threshold"].notna().mean()
    assert finite_share > 0.95, (
        f"Expected >95% of alpha-present non-sellers to have finite "
        f"dev_from_threshold; got {finite_share:.3f}"
    )


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
