"""
Purpose: Estimate per-participant CRRA α from selling decisions via MLE against
         the Magnani & Munro (2020) equilibrium σ grid, and invert the
         Gneezy-Potters lottery allocation to recover a survey-based α.
Author: Claude
Date: 2026-04-21
"""

from math import log
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.analysis.equilibrium_model import (
    N_INVESTORS,
    PI_0,
    _update_bad,
    _update_good,
)

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore" / "derived"
INDIVIDUAL_PERIOD_CSV = DATASTORE / "individual_period_dataset.csv"
SURVEY_TRAITS_CSV = DATASTORE / "survey_traits.csv"
SIGMA_GRID_PARQUET = DATASTORE / "equilibrium_sigma_grid.parquet"
OUTPUT_CSV = DATASTORE / "participant_risk_aversion.csv"

# CONSTANTS
ALPHA_GRID = np.round(np.arange(0.0, 1.01, 0.01), 2)
EPS = 1e-12
LR_CHI2_95 = 3.841458820694124
TREATMENT_MAP = {"tr1": "random", "tr2": "average"}
PI_ATOL = 1e-9


# =====
# Main execution flow
# =====
def main():
    """Build per-participant risk-aversion dataset and write CSV."""
    ind_per_df = pd.read_csv(INDIVIDUAL_PERIOD_CSV)
    traits_df = pd.read_csv(SURVEY_TRAITS_CSV)
    sigma_df = pd.read_parquet(SIGMA_GRID_PARQUET)

    pi_lookup = build_pi_lookup(ind_per_df)
    decisions_df = build_decision_table(ind_per_df, pi_lookup)
    sigma_lookup = build_sigma_lookup(sigma_df)

    rows = assemble_participant_rows(traits_df, decisions_df, sigma_lookup)
    out_df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(out_df)} participant rows to {OUTPUT_CSV}")


# =====
# π reconstruction via Bayesian updating from the group's public signal sequence
# =====
def build_pi_lookup(ind_per_df):
    """Map (session_id, segment, group_id, round, period) → posterior π."""
    lookup = {}
    group_keys = ["session_id", "segment", "group_id", "round"]
    for key, g in ind_per_df.groupby(group_keys):
        # Signals are group-public — all players share one π sequence per period.
        per_period = g.drop_duplicates("period").sort_values("period")
        _fill_pi_sequence(lookup, key, per_period)
    return lookup


def _fill_pi_sequence(lookup, group_key, per_period_df):
    """Replay the group's signal sequence from π₀ and store posterior at each period."""
    pi = PI_0
    for _, row in per_period_df.iterrows():
        pi = _next_pi(pi, float(row["signal"]))
        full_key = (*group_key, int(row["period"]))
        lookup[full_key] = pi


def _next_pi(pi_prev, stored_pi):
    """Apply the good- or bad-signal update consistent with the observed posterior."""
    pi_good = _update_good(pi_prev)
    pi_bad = _update_bad(pi_prev)
    if abs(stored_pi - pi_good) < abs(stored_pi - pi_bad):
        return pi_good
    return pi_bad


# =====
# Decision table: one row per hold/sell choice the participant actually made
# =====
def build_decision_table(ind_per_df, pi_lookup):
    """Filter to pre-sale rows, attach n and π for σ lookup."""
    pre = ind_per_df[ind_per_df["already_sold"] == 0].copy()
    pre["n"] = N_INVESTORS - pre["prior_group_sales"].astype(int)
    key_cols = pre[["session_id", "segment", "group_id", "round", "period"]]
    pre["pi"] = [pi_lookup[tuple(k)] for k in key_cols.itertuples(index=False, name=None)]
    keep = ["session_id", "player", "treatment", "segment", "round",
            "period", "n", "pi", "sold"]
    return pre[keep].rename(columns={"sold": "sold_indicator"})


# =====
# σ lookup: (treatment, α, n) → sorted (π, σ) arrays for strict isclose match
# =====
def build_sigma_lookup(sigma_df):
    """Group σ grid by (treatment, α, n) with pre-sorted π/σ arrays."""
    lookup = {}
    for (tr, alpha, n), g in sigma_df.groupby(["treatment", "alpha", "n"]):
        g_sorted = g.sort_values("pi")
        pi_arr = g_sorted["pi"].to_numpy()
        sigma_arr = g_sorted["sigma"].to_numpy()
        lookup[(tr, round(float(alpha), 2), int(n))] = (pi_arr, sigma_arr)
    return lookup


def sigma_at(sigma_lookup, treatment, alpha, n, pi):
    """Strict σ lookup — fail loudly on off-grid π."""
    key = (treatment, round(float(alpha), 2), int(n))
    pi_arr, sigma_arr = sigma_lookup[key]
    matches = np.where(np.isclose(pi_arr, pi, atol=PI_ATOL))[0]
    if matches.size != 1:
        raise ValueError(
            f"π={pi} off-grid for (treatment={treatment}, alpha={alpha}, n={n}); "
            f"got {matches.size} matches — regenerate equilibrium_sigma_grid.parquet"
        )
    return float(sigma_arr[matches[0]])


# =====
# MLE over α grid with 95% likelihood-ratio confidence set
# =====
def participant_log_likelihood(decisions, alpha, sigma_lookup, treatment):
    """Sum of Bernoulli log-likelihoods across a participant's hold/sell rows."""
    ll = 0.0
    for row in decisions.itertuples(index=False):
        p = sigma_at(sigma_lookup, treatment, alpha, row.n, row.pi)
        ll += _bernoulli_ll(int(row.sold_indicator), p)
    return ll


def _bernoulli_ll(sold, p):
    """Clamped Bernoulli log-likelihood for a single decision."""
    return sold * log(max(p, EPS)) + (1 - sold) * log(max(1.0 - p, EPS))


def fit_alpha_mle(decisions, treatment, sigma_lookup):
    """Grid-search α; return (argmax α, max LL, full LL profile over ALPHA_GRID)."""
    ll_profile = np.array([
        participant_log_likelihood(decisions, a, sigma_lookup, treatment)
        for a in ALPHA_GRID
    ])
    ll_max = float(ll_profile.max())
    # Break ties by smallest α — argmax returns the first maximum.
    alpha_mle = float(ALPHA_GRID[int(np.argmax(ll_profile))])
    return alpha_mle, ll_max, ll_profile


def lr_ci_from_profile(alpha_grid, ll_profile, ll_max):
    """95% likelihood-ratio set over the supplied α grid; returns (α_low, α_high)."""
    mask = 2.0 * (ll_max - ll_profile) <= LR_CHI2_95
    in_set = alpha_grid[mask]
    return float(in_set.min()), float(in_set.max())


# =====
# Gneezy-Potters inversion of lottery allocation → α_task
# =====
def compute_alpha_task(allocate):
    """Invert CRRA indifference in the Gneezy-Potters lottery (edge-aware)."""
    if pd.isna(allocate) or allocate == 0 or allocate == 20:
        return (np.nan, True)
    if allocate < 0 or allocate > 20:
        raise ValueError(f"allocate out of range: {allocate}")
    alpha = log(2.5) / log((20 + 1.5 * allocate) / (20 - allocate))
    return (float(alpha), False)


# =====
# Per-participant row assembly
# =====
def assemble_participant_rows(traits_df, decisions_df, sigma_lookup):
    """Iterate survey participants, estimate α, and build output records."""
    rows = []
    skipped = 0
    for rec in traits_df.itertuples(index=False):
        sub = _slice_participant(decisions_df, rec)
        if len(sub) == 0:
            skipped += 1
            continue
        rows.append(_estimate_one_participant(rec, sub, sigma_lookup))
    if skipped:
        print(f"Skipped {skipped} participants with zero pre-sale decisions")
    return rows


def _slice_participant(decisions_df, rec):
    """Return the decision rows belonging to a single (session, player) pair."""
    return decisions_df[
        (decisions_df["session_id"] == rec.session_id)
        & (decisions_df["player"] == rec.player)
    ]


def _estimate_one_participant(rec, sub, sigma_lookup):
    """Run MLE for one participant and package the output row."""
    tr_label = sub["treatment"].iloc[0]
    alpha_mle, ll_max, ll_profile = fit_alpha_mle(
        sub, TREATMENT_MAP[tr_label], sigma_lookup
    )
    return assemble_output_row(
        session_id=rec.session_id, player=rec.player, treatment=tr_label,
        decisions=sub, alpha_mle=alpha_mle, ll_profile=ll_profile,
        ll_max=ll_max, allocate=rec.risk_tolerance,
    )


def assemble_output_row(session_id, player, treatment, decisions,
                        alpha_mle, ll_profile, ll_max, allocate):
    """Assemble one participant's output row from pre-computed MLE inputs."""
    alpha_task, edge_flag = compute_alpha_task(allocate)
    alpha_ci_low, alpha_ci_high = lr_ci_from_profile(
        ALPHA_GRID, ll_profile, ll_max
    )
    return {
        "session_id": session_id,
        "player": player,
        "treatment": treatment,
        "n_decisions": len(decisions),
        "alpha_mle": alpha_mle,
        "alpha_ci_low": alpha_ci_low,
        "alpha_ci_high": alpha_ci_high,
        "alpha_task": alpha_task,
        "alpha_task_edge_flag": edge_flag,
    }


# %%
if __name__ == "__main__":
    main()
