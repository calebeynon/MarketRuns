"""
Purpose: Build individual-period dataset for market runs analysis
Author: Claude Code
Date: 2025-01-14

Creates a player-period level dataset for analyzing selling decisions in market
experiments. Each row represents one player in one period of one round.

OUTPUT VARIABLES:
    session_id: Session identifier (e.g., "1_11-7-tr1")
    segment: Segment number within session (1-4)
    round: Round number within segment (1-14)
    period: Period within round (1-4, varies by round)
    group_id: Group identifier (1-4, fixed throughout session)
    player: Participant label
    treatment: Treatment condition ("tr1" or "tr2")
    signal: Private signal received by player (0 or 1)
    state: True state of the asset this round (0 or 1)
    price: Current market price at this period

    sold: Binary (0/1) - Did the player sell in THIS SPECIFIC period?
          This captures the selling DECISION moment, not cumulative status.
          Value is 1 only in the single period where the sale occurred.

    already_sold: Binary (0/1) - Had the player already sold in a PRIOR period
                  of this same round? This is 1 for all periods AFTER the sale.
                  IMPORTANT: sold and already_sold are mutually exclusive.
                  If sold=1, then already_sold=0 (you sell NOW, not already).
                  If already_sold=1, then sold=0 (you already sold earlier).

    prior_group_sales: Count (0-3) of OTHER group members who sold in periods
                       BEFORE this period within the same round. This variable
                       RESETS to 0 at the start of each new round.
                       Does NOT include the current player or current period.

DATA STRUCTURE:
    - 6 sessions x 4 segments x 14 rounds x ~4 periods x 4 groups x 4 players
    - Expected total: ~16,000 observations
    - Each round has variable number of periods (defined by PERIODS_PER_ROUND)
"""

import pandas as pd
from pathlib import Path

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
OUTPUT_PATH = DATASTORE / "derived" / "individual_period_dataset.csv"

# Session folders with treatment indicators
SESSIONS = {
    "1_11-7-tr1": "tr1",
    "2_11-10-tr2": "tr2",
    "3_11-11-tr2": "tr2",
    "4_11-12-tr1": "tr1",
    "5_11-14-tr2": "tr2",
    "6_11-18-tr1": "tr1",
}

SEGMENTS = ["chat_noavg", "chat_noavg2", "chat_noavg3", "chat_noavg4"]


# =====
# Main function
# =====
def main():
    """Build the individual period dataset."""
    all_records = []

    print("Processing sessions...")
    for session_name, treatment in SESSIONS.items():
        print(f"  {session_name} (treatment {treatment})")
        records = process_session(session_name, treatment)
        all_records.extend(records)
        print(f"    -> {len(records)} player-period observations")

    df = pd.DataFrame(all_records)
    print_summary_statistics(df)
    save_dataset(df)

    return df


# =====
# Session processing
# =====
def process_session(session_name: str, treatment: str) -> list[dict]:
    """Process all segments for a session, return list of player-period records."""
    session_folder = DATASTORE / session_name
    records = []

    for segment_idx, segment in enumerate(SEGMENTS, start=1):
        try:
            df = load_segment_data(session_folder, segment)
        except FileNotFoundError as e:
            print(f"    Warning: {e}")
            continue

        segment_records = process_segment(
            df, session_name, segment_idx, treatment
        )
        records.extend(segment_records)

    return records


def process_segment(
    df: pd.DataFrame,
    session_name: str,
    segment_idx: int,
    treatment: str
) -> list[dict]:
    """Process all group-rounds in a segment."""
    records = []

    group_rounds = df.groupby(
        ["group.id_in_subsession", "player.round_number_in_segment"]
    )

    for (group_id, round_num), group_df in group_rounds:
        round_records = process_group_round(
            group_df,
            session_name,
            segment_idx,
            int(group_id),
            int(round_num),
            treatment
        )
        records.extend(round_records)

    return records


# =====
# Data loading
# =====
def load_segment_data(session_folder: Path, segment: str) -> pd.DataFrame:
    """Load segment CSV file from session folder."""
    csv_files = list(session_folder.glob(f"{segment}_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found for {segment} in {session_folder}")
    return pd.read_csv(csv_files[0])


# =====
# Core processing logic
# =====
def process_group_round(
    group_df: pd.DataFrame,
    session_name: str,
    segment_idx: int,
    group_id: int,
    round_num: int,
    treatment: str
) -> list[dict]:
    """
    Process a single group-round and return player-period records.

    Core logic:
    - sold: 1 if player sold in THIS specific period (transition from 0 to 1)
    - already_sold: 1 if player sold in any prior period of this round
    - prior_group_sales: count of OTHER group members who sold before this period
    """
    records = []

    # Get unique periods and players
    periods = sorted(group_df["player.period_in_round"].unique())
    players = group_df["participant.label"].unique()

    # Build lookup: (player, period) -> sold_cumulative value
    sold_lookup = build_sold_lookup(group_df)

    # Get state (same for all players in round)
    state = int(group_df["player.state"].iloc[0])

    for period in periods:
        # Calculate prior_group_sales for this period
        prior_sales = calc_prior_group_sales(sold_lookup, players, period)

        for player in players:
            record = build_player_period_record(
                group_df, sold_lookup, session_name, segment_idx,
                round_num, period, group_id, player, treatment, state, prior_sales
            )
            records.append(record)

    return records


def build_sold_lookup(group_df: pd.DataFrame) -> dict:
    """Build a lookup dict mapping (player, period) -> cumulative sold value."""
    lookup = {}
    for _, row in group_df.iterrows():
        player = row["participant.label"]
        period = int(row["player.period_in_round"])
        sold_val = int(row["player.sold"]) if pd.notna(row["player.sold"]) else 0
        lookup[(player, period)] = sold_val
    return lookup


def calc_prior_group_sales(
    sold_lookup: dict,
    players,
    current_period: int
) -> dict:
    """
    Calculate prior group sales for each player at a given period.

    Returns dict mapping player -> count of OTHER players who sold
    in periods strictly BEFORE current_period.
    """
    # Count how many OTHER players sold before this period
    prior_sales = {}

    for player in players:
        count = 0
        for other_player in players:
            if other_player == player:
                continue
            # Check if other_player sold before current_period
            for p in range(1, current_period):
                if sold_lookup.get((other_player, p), 0) == 1:
                    count += 1
                    break  # Only count each player once
        prior_sales[player] = count

    return prior_sales


def build_player_period_record(
    group_df: pd.DataFrame,
    sold_lookup: dict,
    session_name: str,
    segment_idx: int,
    round_num: int,
    period: int,
    group_id: int,
    player: str,
    treatment: str,
    state: int,
    prior_sales: dict
) -> dict:
    """Build a single player-period record."""
    # Get current sold value
    current_sold = sold_lookup.get((player, period), 0)

    # Determine if sold THIS period (transition from 0 to 1)
    if period == 1:
        sold_this_period = current_sold
    else:
        prev_sold = sold_lookup.get((player, period - 1), 0)
        sold_this_period = 1 if (current_sold == 1 and prev_sold == 0) else 0

    # Determine if already_sold (sold in any prior period)
    already_sold = 0
    for p in range(1, period):
        if sold_lookup.get((player, p), 0) == 1:
            already_sold = 1
            break

    # Get signal and price from group_df
    player_period_row = group_df[
        (group_df["participant.label"] == player) &
        (group_df["player.period_in_round"] == period)
    ].iloc[0]

    signal = player_period_row["player.signal"]
    price = player_period_row["player.price"]

    return {
        "session_id": session_name,
        "segment": segment_idx,
        "round": round_num,
        "period": period,
        "group_id": group_id,
        "player": player,
        "treatment": treatment,
        "signal": signal,
        "state": state,
        "price": price,
        "sold": sold_this_period,
        "already_sold": already_sold,
        "prior_group_sales": prior_sales[player],
    }


# =====
# Output functions
# =====
def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics for the dataset."""
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total observations: {len(df)}")
    print(f"Unique sessions: {df['session_id'].nunique()}")
    print(f"Unique players: {df['player'].nunique()}")
    print(f"\nSales in this period: {df['sold'].sum()}")
    print(f"Already sold observations: {df['already_sold'].sum()}")
    print(f"\nBy treatment:")
    print(df.groupby("treatment")["sold"].agg(["count", "sum", "mean"]))
    print(f"\nBy segment:")
    print(df.groupby("segment")["sold"].agg(["count", "sum", "mean"]))
    print(f"\nPrior group sales distribution:")
    print(df["prior_group_sales"].value_counts().sort_index())

    # Validation checks
    print("\n" + "=" * 50)
    print("VALIDATION CHECKS")
    print("=" * 50)

    # Check sold and already_sold are mutually exclusive
    both_flags = ((df["sold"] == 1) & (df["already_sold"] == 1)).sum()
    print(f"Rows with sold=1 AND already_sold=1: {both_flags} (should be 0)")

    # Check prior_group_sales range
    pgs_min = df["prior_group_sales"].min()
    pgs_max = df["prior_group_sales"].max()
    print(f"prior_group_sales range: {pgs_min} to {pgs_max} (should be 0-3)")

    # Check all segments present
    segments = sorted(df["segment"].unique())
    print(f"Segments present: {segments} (should be [1, 2, 3, 4])")

    # Check both treatments present
    treatments = sorted(df["treatment"].unique())
    print(f"Treatments present: {treatments} (should be ['tr1', 'tr2'])")


def save_dataset(df: pd.DataFrame):
    """Save dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


# %%
if __name__ == "__main__":
    df = main()
