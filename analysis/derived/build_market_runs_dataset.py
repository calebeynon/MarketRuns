"""
Purpose: Build the player-round market-runs dataset (issue #120). One row per
         (session, segment, group, round, player) with run indicators across
         the (w, k) sweep grid, per-player belief-space deviations from the
         M&M (2020) equilibrium, signal-accuracy, group-mean trait IVs, and
         demographics. Drives analysis/analysis/market_runs_regression.R.
Author: Claude
Date: 2026-04-25
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.derived.equilibrium_reference import (
    load_equilibrium_table, lookup_equilibrium_reference,
)
from analysis.derived.group_traits import (
    compute_group_trait_means, load_survey_traits,
)
from analysis.derived.market_runs_helpers import (
    compute_alpha, compute_min_signal, compute_n_at_dip,
    compute_n_at_sale, detect_run, signal_correct_frac,
    treatment_to_string,
)

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DERIVED = PROJECT_ROOT / "datastore" / "derived"
PANEL_CSV = DERIVED / "individual_round_panel.csv"
PERIOD_CSV = DERIVED / "individual_period_dataset.csv"
OUTPUT_CSV = DERIVED / "market_runs_dataset.csv"

# =====
# Constants
# =====
W_VALUES = [0, 1, 2, 3]
K_VALUES = [2, 3, 4]
GROUP_KEYS = ["session_id", "segment", "group_id", "round"]


# =====
# Main
# =====
# %%
def main():
    """Assemble the player-round market-runs dataset and write to CSV."""
    panel = pd.read_csv(PANEL_CSV)
    periods = pd.read_csv(PERIOD_CSV)
    traits = load_survey_traits()
    eq_df = load_equilibrium_table()
    rows = build_all_rows(panel, periods, traits, eq_df)
    out = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print_summary(out)


# =====
# Per-group-round assembly
# =====
def build_all_rows(panel, periods, traits, eq_df):
    """Iterate group-rounds; emit one row per player in each."""
    period_groups = dict(tuple(periods.groupby(GROUP_KEYS)))
    rows = []
    for keys, gr_panel in panel.groupby(GROUP_KEYS):
        gr_periods = period_groups.get(keys)
        rows.extend(build_group_round(keys, gr_panel, gr_periods, traits, eq_df))
    return rows


def build_group_round(keys, gr_panel, gr_periods, traits, eq_df):
    """One row per player in this (session, segment, group, round)."""
    session_id, segment, group_id, round_num = keys
    constants = compute_group_round_constants(
        keys, gr_panel, gr_periods, traits)
    rows = []
    for _, player_row in gr_panel.iterrows():
        rows.append(build_player_row(
            session_id, segment, group_id, round_num,
            player_row, constants, traits, eq_df))
    return rows


def compute_group_round_constants(keys, gr_panel, gr_periods, traits):
    """Group-round constants repeated across the 4 player rows."""
    session_id, _segment, _group_id, _round_num = keys
    per_player_first_sale = build_per_player_first_sale(gr_panel)
    sale_periods = sorted(p for p in per_player_first_sale.values() if p is not None)
    treatment = gr_panel["treatment"].iloc[0]
    state = int(gr_panel["state"].iloc[0])
    players = sorted(gr_panel["player"].unique().tolist())
    return {
        "per_player_first_sale": per_player_first_sale,
        "treatment": treatment,
        "treatment_str": treatment_to_string(treatment),
        "state": state,
        "run_cols": run_indicator_cols(sale_periods),
        "signal_correct_frac": compute_signal_accuracy(gr_periods, state),
        "trait_means": compute_group_trait_means(traits, session_id, players),
        "group_period_signals": build_group_period_signals(gr_periods),
    }


def build_group_period_signals(gr_periods):
    """pd.Series of public posterior pi indexed by `period_in_round`.

    Signal is public per group-period, so dedupe across the 4 players
    (keep='first'). Returns an empty Series when no period rows are
    available so non-seller deviation lookups degrade to NaN cleanly.
    """
    if gr_periods is None or gr_periods.empty:
        return pd.Series(dtype=float)
    deduped = gr_periods.drop_duplicates("period", keep="first")
    return pd.Series(
        deduped["signal"].astype(float).values,
        index=deduped["period"].astype(int).values,
    )


def build_per_player_first_sale(gr_panel):
    """{player_label -> first_sale_period} (None for non-sellers)."""
    out = {}
    for _, row in gr_panel.iterrows():
        sp = row["sell_period"]
        out[row["player"]] = int(sp) if pd.notna(sp) else None
    return out


def run_indicator_cols(sale_periods):
    """All 12 binary run indicators for the (w, k) sweep grid."""
    return {
        f"run_w{w}_k{k}": int(detect_run(sale_periods, w, k))
        for w in W_VALUES for k in K_VALUES
    }


def compute_signal_accuracy(gr_periods, state):
    """signal_correct_frac dedupes signals by period (signal is public)."""
    if gr_periods is None or gr_periods.empty:
        return float("nan")
    deduped = gr_periods.drop_duplicates("period", keep="first")
    return signal_correct_frac(deduped[["signal", "state"]])


# =====
# Per-player row
# =====
def build_player_row(session_id, segment, group_id, round_num,
                     player_row, constants, traits, eq_df):
    """One output row for a single player in a group-round."""
    player = player_row["player"]
    did_sell = int(player_row["did_sell"])
    sell_period = (int(player_row["sell_period"])
                   if pd.notna(player_row["sell_period"]) else None)
    pi_at_sale = (float(player_row["signal"])
                  if did_sell and pd.notna(player_row["signal"]) else float("nan"))
    risk_tol = lookup_player_risk_tolerance(traits, session_id, player)
    alpha = compute_alpha(risk_tol)
    base = identity_block(session_id, segment, group_id, round_num,
                          player, constants["state"], constants["treatment"])
    base.update(sale_block(constants, player, did_sell, sell_period,
                           pi_at_sale, alpha, eq_df))
    base["signal_correct_frac"] = constants["signal_correct_frac"]
    base.update(constants["trait_means"])
    base.update(constants["run_cols"])
    return base


def identity_block(session_id, segment, group_id, round_num,
                   player, state, treatment):
    """Identifier columns shared by every row."""
    return {
        "session_id": session_id, "treatment": treatment,
        "segment": int(segment), "group_id": int(group_id),
        "round": int(round_num), "player": player, "state": state,
    }


def sale_block(constants, player, did_sell, sell_period, pi_at_sale,
               alpha, eq_df):
    """Per-player sale info + belief-space equilibrium deviations.

    `no_sale` is per-player (1 iff this player did not sell), not group-level.
    """
    n_at_sale = (compute_n_at_sale(constants["per_player_first_sale"], player)
                 if did_sell else None)
    out = {
        "did_sell": did_sell, "sell_period": sell_period,
        "pi_at_sale": pi_at_sale,
        "n_at_sale": (int(n_at_sale) if n_at_sale is not None else float("nan")),
        "no_sale": int(did_sell == 0), "alpha": alpha,
    }
    out.update(equilibrium_deviations(
        eq_df, alpha, constants, did_sell, n_at_sale, pi_at_sale))
    return out


def equilibrium_deviations(eq_df, alpha, constants, did_sell,
                           n_at_sale, pi_at_sale):
    """dev_from_threshold and dev_from_avg_pi for sellers and non-sellers."""
    nan_pair = {"dev_from_threshold": float("nan"),
                "dev_from_avg_pi": float("nan")}
    if pd.isna(alpha):
        return nan_pair
    if did_sell:
        return seller_deviations(
            eq_df, alpha, constants["treatment_str"], n_at_sale, pi_at_sale)
    return nonseller_deviations(
        eq_df, alpha, constants["treatment_str"],
        constants["per_player_first_sale"],
        constants["group_period_signals"])


def seller_deviations(eq_df, alpha, treatment_str, n_at_sale, pi_at_sale):
    """Seller branch: pi_at_sale minus equilibrium reference."""
    if n_at_sale is None or pd.isna(pi_at_sale):
        return {"dev_from_threshold": float("nan"),
                "dev_from_avg_pi": float("nan")}
    threshold_pi, avg_pi = lookup_equilibrium_reference(
        eq_df, alpha, treatment_str, int(n_at_sale))
    return {
        "dev_from_threshold": pi_at_sale - threshold_pi,
        "dev_from_avg_pi": (float("nan") if pd.isna(avg_pi)
                            else pi_at_sale - avg_pi),
    }


def nonseller_deviations(eq_df, alpha, treatment_str,
                         per_player_first_sale, group_period_signals):
    """Non-seller branch: min_signal at dip vs equilibrium reference.

    Caps positive deviations at 0: a non-seller who never saw a signal
    below the threshold has zero observed deviation (we can't infer how
    much higher their willingness-to-hold was).
    """
    min_signal, dip_period = compute_min_signal(group_period_signals)
    if dip_period is None:
        return {"dev_from_threshold": float("nan"),
                "dev_from_avg_pi": float("nan")}
    n_at_dip = compute_n_at_dip(per_player_first_sale, dip_period)
    threshold_pi, avg_pi = lookup_equilibrium_reference(
        eq_df, alpha, treatment_str, int(n_at_dip))
    return {
        "dev_from_threshold": cap_nonseller_dev(min_signal, threshold_pi),
        "dev_from_avg_pi": (float("nan") if pd.isna(avg_pi)
                            else cap_nonseller_dev(min_signal, avg_pi)),
    }


def cap_nonseller_dev(min_signal, reference):
    """0 when min_signal stayed above reference; else min_signal - reference."""
    return 0.0 if min_signal >= reference else min_signal - reference


def lookup_player_risk_tolerance(traits, session_id, player):
    """Per-player risk_tolerance from survey_traits.csv (NaN if missing)."""
    match = traits[(traits["session_id"] == session_id)
                   & (traits["player"] == player)]
    if match.empty:
        return float("nan")
    return float(match["risk_tolerance"].iloc[0])


# =====
# Output
# =====
def print_summary(df):
    """Console diagnostics for the assembled dataset."""
    print(f"\nWrote {len(df)} rows ({len(df.columns)} cols) to {OUTPUT_CSV}")
    print(f"did_sell rate: {df['did_sell'].mean():.3f}")
    print(f"no_sale rate: {df['no_sale'].mean():.3f}")
    print(f"dev_from_threshold NaN: {df['dev_from_threshold'].isna().sum()}")
    print(f"dev_from_avg_pi NaN: {df['dev_from_avg_pi'].isna().sum()}")
    print("Run incidence (w, k):")
    for c in [c for c in df.columns if c.startswith("run_w")]:
        print(f"  {c}: {df[c].mean():.3f}")


# %%
if __name__ == "__main__":
    main()
