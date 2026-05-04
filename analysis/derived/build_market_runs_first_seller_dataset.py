"""
Purpose: Build the GROUP-ROUND first-seller market-runs dataset (issue #120).
         One row per (session_id, segment, group_id, round) with first-seller
         belief-space deviation, run indicators across the (w, k) sweep grid,
         signal-accuracy, and group-mean trait IVs. Drops group-rounds with
         zero sellers (first-seller fields undefined).
Author: Claude
Date: 2026-05-04
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
    compute_alpha, detect_run, signal_correct_frac, treatment_to_string,
)

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DERIVED = PROJECT_ROOT / "datastore" / "derived"
PANEL_CSV = DERIVED / "individual_round_panel.csv"
PERIOD_CSV = DERIVED / "individual_period_dataset.csv"
OUTPUT_CSV = DERIVED / "market_runs_first_seller_dataset.csv"

# =====
# Constants
# =====
W_VALUES = [0, 1, 2, 3]
K_VALUES = [2, 3, 4]
GROUP_KEYS = ["session_id", "segment", "group_id", "round"]
N_FIRST_SELLER = 4  # First seller faces 4 holders by definition.


# =====
# Main
# =====
# %%
def main():
    """Assemble the group-round first-seller dataset and write to CSV."""
    panel = pd.read_csv(PANEL_CSV)
    periods = pd.read_csv(PERIOD_CSV)
    traits = load_survey_traits()
    eq_df = load_equilibrium_table()
    rows = build_all_rows(panel, periods, traits, eq_df)
    out = pd.DataFrame(rows)
    total = len(out)
    out = out[out["_has_seller"]].drop(columns=["_has_seller"]).reset_index(drop=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print_summary(out, total_before_drop=total)


# =====
# Per-group-round assembly
# =====
def build_all_rows(panel, periods, traits, eq_df):
    """Iterate group-rounds; emit one row per (session, segment, group, round)."""
    period_groups = dict(tuple(periods.groupby(GROUP_KEYS)))
    rows = []
    for keys, gr_panel in panel.groupby(GROUP_KEYS):
        gr_periods = period_groups.get(keys)
        rows.append(build_group_round_row(
            keys, gr_panel, gr_periods, traits, eq_df))
    return rows


def build_group_round_row(keys, gr_panel, gr_periods, traits, eq_df):
    """Single output row for one (session, segment, group, round)."""
    session_id, segment, group_id, round_num = keys
    sellers = gr_panel[gr_panel["did_sell"] == 1]
    treatment = gr_panel["treatment"].iloc[0]
    state = int(gr_panel["state"].iloc[0])
    players = sorted(gr_panel["player"].unique().tolist())
    base = identity_block(session_id, segment, group_id, round_num,
                          state, treatment)
    base.update(first_seller_block(sellers, traits, session_id,
                                   treatment, state, eq_df))
    base["signal_correct_frac"] = compute_signal_accuracy(gr_periods, state)
    base.update(compute_group_trait_means(traits, session_id, players))
    base.update(run_indicator_cols(seller_periods_list(gr_panel)))
    return base


def identity_block(session_id, segment, group_id, round_num, state, treatment):
    """Identifier columns shared by every row."""
    return {
        "session_id": session_id, "treatment": treatment,
        "segment": int(segment), "group_id": int(group_id),
        "round": int(round_num), "state": state,
    }


# =====
# First-seller fields
# =====
def first_seller_block(sellers, traits, session_id, treatment, state, eq_df):
    """First-seller columns + `_has_seller` flag for the post-build drop."""
    if sellers.empty:
        return empty_first_seller_block()
    first_period = int(sellers["sell_period"].min())
    tied = sellers[sellers["sell_period"] == first_period]
    primary = tied.sort_values("player").iloc[0]
    pi_at_sale = float(primary["signal"])
    alpha = compute_alpha(lookup_player_risk_tolerance(
        traits, session_id, primary["player"]))
    return {
        "first_sale_period": first_period,
        "first_seller_signal_correct": signal_correct_flag(pi_at_sale, state),
        "alpha_first": alpha,
        "tie_at_first_period": int(len(tied) > 1),
        "pi_at_sale_first": pi_at_sale,
        "dev_from_threshold_first": dev_from_threshold(
            eq_df, alpha, treatment, pi_at_sale),
        "_has_seller": True,
    }


def empty_first_seller_block():
    """First-seller columns when the group-round has zero sellers."""
    return {
        "first_sale_period": float("nan"),
        "first_seller_signal_correct": float("nan"),
        "alpha_first": float("nan"),
        "tie_at_first_period": float("nan"),
        "pi_at_sale_first": float("nan"),
        "dev_from_threshold_first": float("nan"),
        "_has_seller": False,
    }


def signal_correct_flag(pi_at_sale, state):
    """1 if posterior pi points to the true state; 0 if pi==0.5 or opposes."""
    if pd.isna(pi_at_sale):
        return float("nan")
    if pi_at_sale > 0.5 and state == 1:
        return 1
    if pi_at_sale < 0.5 and state == 0:
        return 1
    return 0


def dev_from_threshold(eq_df, alpha, treatment, pi_at_sale):
    """pi_at_sale minus equilibrium threshold_pi at n=4 (first-seller view)."""
    if pd.isna(alpha) or pd.isna(pi_at_sale):
        return float("nan")
    threshold_pi, _ = lookup_equilibrium_reference(
        eq_df, alpha, treatment_to_string(treatment), N_FIRST_SELLER)
    return pi_at_sale - threshold_pi


def lookup_player_risk_tolerance(traits, session_id, player):
    """Per-player risk_tolerance from survey_traits.csv (NaN if missing)."""
    match = traits[(traits["session_id"] == session_id)
                   & (traits["player"] == player)]
    if match.empty:
        return float("nan")
    return float(match["risk_tolerance"].iloc[0])


# =====
# Run / signal helpers
# =====
def seller_periods_list(gr_panel):
    """Ascending list of sell_period for each seller (one entry per seller)."""
    sellers = gr_panel[gr_panel["did_sell"] == 1]
    return sorted(int(p) for p in sellers["sell_period"].dropna())


def run_indicator_cols(sale_periods):
    """All 12 binary run indicators for the (w, k) sweep grid."""
    return {
        f"run_w{w}_k{k}": int(detect_run(sale_periods, w, k))
        for w in W_VALUES for k in K_VALUES
    }


def compute_signal_accuracy(gr_periods, state):
    """signal_correct_frac dedupes signals by period (signal is public)."""
    del state  # state column already on gr_periods; kept for call symmetry
    if gr_periods is None or gr_periods.empty:
        return float("nan")
    deduped = gr_periods.drop_duplicates("period", keep="first")
    return signal_correct_frac(deduped[["signal", "state"]])


# =====
# Output
# =====
def print_summary(df, total_before_drop):
    """Console diagnostics for the assembled dataset."""
    print(f"\nGroup-rounds before drop: {total_before_drop}")
    print(f"Group-rounds after drop (zero-seller removed): {len(df)}")
    print(f"Dropped (zero sellers): {total_before_drop - len(df)}")
    print(f"Tied first-period sellers: {int(df['tie_at_first_period'].sum())}")
    print(f"NaN dev_from_threshold_first: {df['dev_from_threshold_first'].isna().sum()}")
    print(f"NaN alpha_first: {df['alpha_first'].isna().sum()}")
    print("Run incidence (w, k):")
    for c in [c for c in df.columns if c.startswith("run_w")]:
        print(f"  {c}: {df[c].mean():.3f}")
    print(f"\nWrote {len(df)} rows ({len(df.columns)} cols) to {OUTPUT_CSV}")


# %%
if __name__ == "__main__":
    main()
