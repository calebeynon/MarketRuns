"""
Purpose: Manual verification of individual_period_dataset.csv against market_data.py parser.
Author: Claude Code
Date: 2025-01-14

Verifies that the generated dataset values match raw data from the market_data parser.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import market_data as md

# =====
# File paths
# =====
DATASTORE = Path("/Users/caleb/Research/marketruns/datastore")
INDIVIDUAL_PERIOD_DATA = DATASTORE / "derived" / "individual_period_dataset.csv"
SESSION_DATA = DATASTORE / "1_11-7-tr1" / "all_apps_wide_2025-11-07.csv"

# =====
# Segment name mapping (segment index to oTree segment name)
# =====
SEGMENT_MAP = {1: "chat_noavg", 2: "chat_noavg2", 3: "chat_noavg3", 4: "chat_noavg4"}


# =====
# Main function
# =====
def main():
    """Run verification of individual period dataset against raw parser data."""
    data = pd.read_csv(INDIVIDUAL_PERIOD_DATA)
    experiment = md.parse_experiment(str(SESSION_DATA))
    session = experiment.sessions[0]

    # Filter to first session only
    session_data = data[data["session_id"] == "1_11-7-tr1"]

    mismatches = {
        "sold": 0,
        "already_sold": 0,
        "prior_group_sales": 0,
        "signal": 0,
        "state": 0,
        "price": 0,
    }

    rows_checked = 0

    for _, row in session_data.iterrows():
        segment_idx = row["segment"]
        segment_name = SEGMENT_MAP[segment_idx]
        round_num = row["round"]
        period_num = row["period"]
        player_label = row["player"]
        group_id = row["group_id"]

        # Get reference data from parser
        ref = get_reference_data(
            session, segment_name, round_num, period_num, player_label, group_id
        )

        if ref is None:
            continue

        rows_checked += 1

        # Verify each field
        if row["sold"] != ref["sold_this_period"]:
            mismatches["sold"] += 1
        if row["already_sold"] != ref["already_sold"]:
            mismatches["already_sold"] += 1
        if row["prior_group_sales"] != ref["prior_group_sales"]:
            mismatches["prior_group_sales"] += 1
        if not signals_match(row["signal"], ref["signal"]):
            mismatches["signal"] += 1
        if row["state"] != ref["state"]:
            mismatches["state"] += 1
        if not prices_match(row["price"], ref["price"]):
            mismatches["price"] += 1

    print_results(mismatches, rows_checked)
    assert_no_mismatches(mismatches)


# =====
# Helper functions - reference data retrieval
# =====
def get_reference_data(session, segment_name, round_num, period_num, player_label, group_id):
    """Get reference data for a specific player-period from the parser."""
    segment = session.get_segment(segment_name)
    if not segment:
        return None

    round_obj = segment.get_round(round_num)
    if not round_obj:
        return None

    period_obj = round_obj.get_period(period_num)
    if not period_obj:
        return None

    player = period_obj.get_player(player_label)
    if not player:
        return None

    # Compute derived fields
    already_sold = compute_already_sold(round_obj, period_num, player_label)
    prior_group_sales = compute_prior_group_sales(
        segment, round_obj, period_num, player_label, group_id
    )

    return {
        "sold_this_period": 1 if player.sold_this_period else 0,
        "already_sold": already_sold,
        "prior_group_sales": prior_group_sales,
        "signal": player.signal,
        "state": player.state,
        "price": player.price,
    }


def compute_already_sold(round_obj, current_period, player_label):
    """Check if player sold in a prior period of this round."""
    for period_num in sorted(round_obj.periods.keys()):
        if period_num >= current_period:
            break
        period = round_obj.get_period(period_num)
        if player_label in period.players:
            if period.players[player_label].sold_this_period:
                return 1
    return 0


def compute_prior_group_sales(segment, round_obj, current_period, player_label, group_id):
    """Count group members who sold in prior periods (excluding self)."""
    group = segment.get_group(group_id)
    if not group:
        return 0

    prior_sales = 0
    for period_num in sorted(round_obj.periods.keys()):
        if period_num >= current_period:
            break
        period = round_obj.get_period(period_num)
        for member in group.player_labels:
            if member == player_label:
                continue
            if member in period.players and period.players[member].sold_this_period:
                prior_sales += 1
    return prior_sales


# =====
# Helper functions - comparison utilities
# =====
def signals_match(csv_signal, ref_signal):
    """Check if signals match within tolerance."""
    if pd.isna(csv_signal) and ref_signal is None:
        return True
    if pd.isna(csv_signal) or ref_signal is None:
        return False
    return abs(csv_signal - ref_signal) < 1e-9


def prices_match(csv_price, ref_price):
    """Check if prices match within tolerance."""
    if pd.isna(csv_price) and ref_price is None:
        return True
    if pd.isna(csv_price) or ref_price is None:
        return False
    return abs(csv_price - ref_price) < 1e-9


# =====
# Helper functions - output
# =====
def print_results(mismatches, rows_checked):
    """Print verification results."""
    print(f"\nVerification Results ({rows_checked} rows checked)")
    print("=" * 40)
    for field, count in mismatches.items():
        status = "OK" if count == 0 else f"MISMATCH ({count})"
        print(f"  {field}: {status}")
    print("=" * 40)


def assert_no_mismatches(mismatches):
    """Assert all fields have zero mismatches."""
    total = sum(mismatches.values())
    assert total == 0, f"Found {total} mismatches: {mismatches}"
    print("All verifications passed!")


# %%
if __name__ == "__main__":
    main()
