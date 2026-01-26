"""
Purpose: Verify individual_period_dataset_extended.csv against raw segment CSVs
Author: Claude
Date: 2026-01-22

Cross-references the extended dataset columns (round_payoff, sold_in_round)
against the authoritative values from raw oTree segment CSVs.

NOTE: We read directly from segment CSVs rather than market_data.py because
the round_X_payoff values are only correct at the LAST PERIOD of each round.
market_data.py reads from period 1, which is incorrect.
"""

import pandas as pd
from pathlib import Path

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
EXTENDED_DATASET = DATASTORE / "derived" / "individual_period_dataset_extended.csv"

# Session folders
SESSIONS = [
    "1_11-7-tr1",
    "2_11-10-tr2",
    "3_11-11-tr2",
    "4_11-12-tr1",
    "5_11-14-tr2",
    "6_11-18-tr1",
]

# Map segment index to segment name
SEGMENT_MAP = {
    1: "chat_noavg",
    2: "chat_noavg2",
    3: "chat_noavg3",
    4: "chat_noavg4",
}


# =====
# Main function
# =====
def main():
    """Verify extended dataset against raw segment CSVs."""
    print("Loading extended dataset...")
    df = pd.read_csv(EXTENDED_DATASET)
    print(f"  Loaded {len(df)} rows")

    print("\nVerifying columns against raw segment CSVs...")
    round_payoff_errors = verify_round_payoff(df)
    sold_in_round_errors = verify_sold_in_round(df)

    print_summary(round_payoff_errors, sold_in_round_errors, len(df))

    return round_payoff_errors, sold_in_round_errors


# =====
# Verification functions
# =====
def verify_round_payoff(df: pd.DataFrame) -> list:
    """Verify round_payoff against segment CSV round_X_payoff at last period."""
    errors = []
    checked = 0
    matched = 0

    for session_id in SESSIONS:
        session_folder = DATASTORE / session_id
        session_df = df[df["session_id"] == session_id]

        for segment_idx, segment_name in SEGMENT_MAP.items():
            csv_files = list(session_folder.glob(f"{segment_name}_*.csv"))
            if not csv_files:
                continue

            seg_df = pd.read_csv(csv_files[0])
            segment_df = session_df[session_df["segment"] == segment_idx]

            # Get expected payoffs from last period of each round
            expected = get_expected_payoffs(seg_df)

            for (round_num, player), expected_payoff in expected.items():
                player_df = segment_df[
                    (segment_df["round"] == round_num) &
                    (segment_df["player"] == player)
                ]

                if player_df.empty:
                    continue

                actual_payoff = player_df["round_payoff"].iloc[0]
                checked += 1

                if abs(actual_payoff - expected_payoff) > 0.01:
                    errors.append({
                        "session": session_id,
                        "segment": segment_idx,
                        "round": round_num,
                        "player": player,
                        "expected": expected_payoff,
                        "actual": actual_payoff,
                    })
                else:
                    matched += 1

    print(f"  round_payoff: checked {checked}, matched {matched}, errors {len(errors)}")
    return errors


def get_expected_payoffs(seg_df: pd.DataFrame) -> dict:
    """
    Extract expected payoffs from segment CSV.

    For each (round, player), get the round_X_payoff value from the LAST
    period of that round (where the final payoff is recorded).
    """
    expected = {}
    players = seg_df["participant.label"].unique()

    for player in players:
        player_df = seg_df[seg_df["participant.label"] == player]

        for round_num in player_df["player.round_number_in_segment"].unique():
            round_df = player_df[
                player_df["player.round_number_in_segment"] == round_num
            ]

            # Get last period of this round
            last_period_df = round_df[
                round_df["player.period_in_round"] ==
                round_df["player.period_in_round"].max()
            ]

            payoff_col = f"player.round_{int(round_num)}_payoff"
            if payoff_col in last_period_df.columns:
                value = last_period_df[payoff_col].iloc[0]
                if pd.notna(value):
                    expected[(int(round_num), player)] = float(value)

    return expected


def verify_sold_in_round(df: pd.DataFrame) -> list:
    """Verify sold_in_round against segment CSV sold values."""
    errors = []
    checked = 0
    matched = 0

    for session_id in SESSIONS:
        session_folder = DATASTORE / session_id
        session_df = df[df["session_id"] == session_id]

        for segment_idx, segment_name in SEGMENT_MAP.items():
            csv_files = list(session_folder.glob(f"{segment_name}_*.csv"))
            if not csv_files:
                continue

            seg_df = pd.read_csv(csv_files[0])
            segment_df = session_df[session_df["segment"] == segment_idx]

            # Get expected sold_in_round from CSV
            expected = get_expected_sold_in_round(seg_df)

            for (round_num, player), expected_sold in expected.items():
                player_df = segment_df[
                    (segment_df["round"] == round_num) &
                    (segment_df["player"] == player)
                ]

                if player_df.empty:
                    continue

                actual_sold = int(player_df["sold_in_round"].iloc[0])
                checked += 1

                if expected_sold != actual_sold:
                    errors.append({
                        "session": session_id,
                        "segment": segment_idx,
                        "round": round_num,
                        "player": player,
                        "expected": expected_sold,
                        "actual": actual_sold,
                    })
                else:
                    matched += 1

    print(f"  sold_in_round: checked {checked}, matched {matched}, errors {len(errors)}")
    return errors


def get_expected_sold_in_round(seg_df: pd.DataFrame) -> dict:
    """
    Extract expected sold_in_round from segment CSV.

    For each (round, player), check if player.sold=1 in ANY period of round.
    """
    expected = {}
    players = seg_df["participant.label"].unique()

    for player in players:
        player_df = seg_df[seg_df["participant.label"] == player]

        for round_num in player_df["player.round_number_in_segment"].unique():
            round_df = player_df[
                player_df["player.round_number_in_segment"] == round_num
            ]

            # Check if sold in any period
            sold_ever = (round_df["player.sold"] == 1).any()
            expected[(int(round_num), player)] = 1 if sold_ever else 0

    return expected


# =====
# Output
# =====
def print_summary(
    round_payoff_errors: list,
    sold_in_round_errors: list,
    total_rows: int
):
    """Print verification summary."""
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    if not round_payoff_errors and not sold_in_round_errors:
        print("✓ All checks passed!")
        print(f"  Total rows in dataset: {total_rows}")
    else:
        if round_payoff_errors:
            print(f"\n✗ round_payoff errors: {len(round_payoff_errors)}")
            for err in round_payoff_errors[:5]:
                print(f"  {err}")
            if len(round_payoff_errors) > 5:
                print(f"  ... and {len(round_payoff_errors) - 5} more")

        if sold_in_round_errors:
            print(f"\n✗ sold_in_round errors: {len(sold_in_round_errors)}")
            for err in sold_in_round_errors[:5]:
                print(f"  {err}")
            if len(sold_in_round_errors) > 5:
                print(f"  ... and {len(sold_in_round_errors) - 5} more")

    print("=" * 60)


# %%
if __name__ == "__main__":
    main()
