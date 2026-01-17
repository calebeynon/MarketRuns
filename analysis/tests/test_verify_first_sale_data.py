"""
Purpose: Verify first_sale_data.csv against market_data.py OOP module
Author: Claude Code
Date: 2025-01-11

Cross-checks the first sale dataset against the authoritative OOP data parser.
"""

import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from market_data import parse_experiment

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
FIRST_SALE_DATA = DATASTORE / "derived" / "first_sale_data.csv"

SESSIONS = {
    "1_11-7-tr1": 1,
    "2_11-10-tr2": 2,
    "3_11-11-tr2": 2,
    "4_11-12-tr1": 1,
    "5_11-14-tr2": 2,
    "6_11-18-tr1": 1,
}

SEGMENTS = ["chat_noavg", "chat_noavg2", "chat_noavg3", "chat_noavg4"]


# =====
# Main verification
# =====
def main():
    """Verify first sale data against OOP module."""
    # Load first sale dataset
    first_sale_df = pd.read_csv(FIRST_SALE_DATA)
    print(f"Loaded {len(first_sale_df)} rows from first_sale_data.csv")

    mismatches = []
    verified = 0

    for session_name, treatment in SESSIONS.items():
        print(f"\nVerifying {session_name}...")

        # Load session via OOP module
        wide_csv = list((DATASTORE / session_name).glob("all_apps_wide_*.csv"))
        if not wide_csv:
            print(f"  Warning: No wide CSV found for {session_name}")
            continue

        experiment = parse_experiment(str(wide_csv[0]))
        session = experiment.sessions[0]

        for segment_idx, segment_name in enumerate(SEGMENTS, start=1):
            segment = session.get_segment(segment_name)
            if not segment:
                continue

            for group_id, group in segment.groups.items():
                for round_num in segment.rounds.keys():
                    round_obj = segment.get_round(round_num)

                    # Get first sale info from OOP
                    oop_first_period = None
                    oop_signal = None

                    for period_num in sorted(round_obj.periods.keys()):
                        period = round_obj.get_period(period_num)
                        # Check if any group member sold this period
                        group_sellers = [
                            label for label in group.player_labels
                            if label in period.players
                            and period.players[label].sold_this_period
                        ]
                        if group_sellers:
                            oop_first_period = period_num
                            # Get signal (same for all players in group)
                            first_seller = group_sellers[0]
                            oop_signal = period.players[first_seller].signal
                            break

                    # Find matching row in first_sale_df
                    global_group_id = f"{session_name}_seg{segment_idx}_g{group_id}"
                    mask = (
                        (first_sale_df["global_group_id"] == global_group_id) &
                        (first_sale_df["round_num"] == round_num)
                    )
                    matching = first_sale_df[mask]

                    if len(matching) != 1:
                        mismatches.append({
                            "type": "missing_row",
                            "global_group_id": global_group_id,
                            "round_num": round_num,
                        })
                        continue

                    row = matching.iloc[0]
                    csv_first_period = row["first_sale_period"]
                    csv_signal = row["signal_at_first_sale"]

                    # Handle NaN comparisons
                    period_match = (
                        (pd.isna(csv_first_period) and oop_first_period is None) or
                        (csv_first_period == oop_first_period)
                    )
                    signal_match = (
                        (pd.isna(csv_signal) and oop_signal is None) or
                        (abs(csv_signal - oop_signal) < 1e-6 if oop_signal else False)
                    )

                    if not period_match or not signal_match:
                        mismatches.append({
                            "type": "value_mismatch",
                            "global_group_id": global_group_id,
                            "round_num": round_num,
                            "csv_period": csv_first_period,
                            "oop_period": oop_first_period,
                            "csv_signal": csv_signal,
                            "oop_signal": oop_signal,
                        })
                    else:
                        verified += 1

    # Report results
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    print(f"Verified: {verified}")
    print(f"Mismatches: {len(mismatches)}")

    if mismatches:
        print("\nMismatch details:")
        for m in mismatches[:10]:
            print(f"  {m}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")

    return mismatches


# =====
# Pytest tests
# =====
def test_first_sale_data_accuracy():
    """Test that first_sale_data.csv matches OOP module calculations."""
    mismatches = main()
    assert len(mismatches) == 0, f"Found {len(mismatches)} mismatches"


# %%
if __name__ == "__main__":
    mismatches = main()
