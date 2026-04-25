"""
Purpose: Pure helper functions for market-run detection and group-round
         summary statistics used by the market-runs dataset and regressions
         (issue #120). No I/O; safe for import in builders, tests, and
         downstream R/Python pipelines.
Author: Claude
Date: 2026-04-24
"""

import math

import pandas as pd

# =====
# Constants
# =====
N_INVESTORS = 4  # M&M group size; matches equilibrium_model.N_INVESTORS

# Translates experimental treatment labels to the equilibrium-model kwarg.
# tr1 pays each seller a random draw from the available prices ("random");
# tr2 pays each seller the mean of those prices ("average"). See
# nonlivegame{,_tr2}/chat_noavg/__init__.py for the payoff difference.
TREATMENT_PAYOFF_MAP = {
    "tr1": "random",
    "tr2": "average",
}


# =====
# Run detection
# =====
def detect_run(seller_periods, w, k):
    """True iff any contiguous (w+1)-period window in `seller_periods`
    contains >= k entries.

    seller_periods is the ascending list of the period each distinct seller
    first sold (one entry per seller; ties allowed). The sliding window
    iterates over unique start periods p and counts entries in [p, p+w];
    w=0 means a single-period window. Empty list or len < k always returns
    False.
    """
    if not seller_periods or len(seller_periods) < k:
        return False
    periods = [int(p) for p in seller_periods]
    for start in set(periods):
        count = sum(1 for q in periods if start <= q <= start + w)
        if count >= k:
            return True
    return False


# =====
# Treatment mapping
# =====
def treatment_to_string(treatment_code):
    """Translate the experimental treatment label ('tr1'/'tr2') to the
    equilibrium-model treatment kwarg ('random'/'average').

    Raises ValueError on unknown labels so silent miscoding cannot propagate
    into equilibrium predictions.
    """
    if treatment_code in TREATMENT_PAYOFF_MAP:
        return TREATMENT_PAYOFF_MAP[treatment_code]
    raise ValueError(
        f"Unknown treatment_code {treatment_code!r}; "
        f"expected one of {sorted(TREATMENT_PAYOFF_MAP)}"
    )


# =====
# Risk-aversion mapping
# =====
def compute_alpha(risk_tolerance):
    """Map a 0-20 lottery allocation to the CRRA alpha grid.

    Formula: alpha = round(min(max(risk_tolerance, 0) / 20, 0.9), 1).
    NaN (or None) propagates as NaN so callers can skip groups with missing
    survey data without losing the row.
    """
    if risk_tolerance is None:
        return float("nan")
    value = float(risk_tolerance)
    if math.isnan(value):
        return float("nan")
    capped = min(max(value, 0.0) / 20.0, 0.9)
    return round(capped, 1)


# =====
# Signal composition
# =====
def signal_correct_frac(period_rows):
    """Fraction of group-round periods where the posterior signal points
    toward the true state.

    period_rows: pd.DataFrame with columns 'signal' (posterior pi in [0, 1])
    and 'state' (0 or 1) -- one row per period (caller dedupes across the 4
    players via drop_duplicates on period_in_round, keep='first', because
    the signal is public per group-period). A period counts as 'correct'
    iff (signal > 0.5 AND state == 1) OR (signal < 0.5 AND state == 0);
    signal == 0.5 contributes 0 to the numerator but stays in the
    denominator. Returns NaN if `period_rows` is empty.
    """
    if len(period_rows) == 0:
        return float("nan")
    sig = period_rows["signal"].astype(float)
    state = period_rows["state"].astype(int)
    correct = ((sig > 0.5) & (state == 1)) | ((sig < 0.5) & (state == 0))
    return float(correct.sum()) / len(period_rows)


# =====
# Min signal at the round-level dip (non-seller branch)
# =====
def compute_min_signal(signal_series):
    """Return (min_signal, dip_period) over a player's round.

    `signal_series` is a pandas Series indexed by `period_in_round` with the
    public posterior pi (one entry per period; caller dedupes since the
    signal is shared across the 4 players in a group-period). Ties are
    broken by the earliest period (idxmin). Returns (NaN, None) if empty.
    """
    if signal_series is None or len(signal_series) == 0:
        return float("nan"), None
    values = signal_series.astype(float)
    dip_period = int(values.idxmin())
    return float(values.min()), dip_period


# =====
# Holders remaining at the dip period (non-seller branch)
# =====
def compute_n_at_dip(seller_periods_by_player, dip_period):
    """N_INVESTORS minus sellers who sold strictly before `dip_period`.

    seller_periods_by_player: {player_id: period_first_sold}; entries with
    None (or absent keys) are treated as non-sellers. The non-seller
    themselves still holds at the dip, so the result is always >= 1.
    """
    if dip_period is None:
        return N_INVESTORS
    earlier = sum(
        1 for p in seller_periods_by_player.values()
        if p is not None and int(p) < int(dip_period)
    )
    n_at_dip = N_INVESTORS - earlier
    assert n_at_dip >= 1, (
        f"n_at_dip={n_at_dip} < 1 at dip_period={dip_period}; "
        f"sellers={seller_periods_by_player}"
    )
    return n_at_dip


# =====
# Holders remaining at sale
# =====
def compute_n_at_sale(seller_periods_by_player, player):
    """Number of holders alive when `player` sold: N_INVESTORS minus the
    count of sellers in strictly-earlier periods.

    seller_periods_by_player: {player_id: period_first_sold}; entries with
    a None value (or absent keys) are treated as non-sellers. Ties at the
    same period share the same n_at_sale because only strictly-earlier
    sales reduce n. Returns None if `player` did not sell.
    """
    own_period = seller_periods_by_player.get(player)
    if own_period is None:
        return None
    own_period = int(own_period)
    earlier = sum(
        1 for p in seller_periods_by_player.values()
        if p is not None and int(p) < own_period
    )
    return N_INVESTORS - earlier
