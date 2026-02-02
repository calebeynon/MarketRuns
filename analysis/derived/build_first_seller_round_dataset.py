"""
Purpose: Build player-round dataset identifying first sellers in each group-round
Author: Claude Code
Date: 2025-02-01

Creates a player-round level dataset where each row represents one player in one
round, with an indicator for whether they were a "first seller" for their group.

DEFINITION OF FIRST SELLER:
A "first seller" is a participant who sells BEFORE their group mates in a given round.
- Find the earliest period in which ANY player in that group sold during the round
- All players who sold in that earliest period are "first sellers"
- If no one sold, no one is marked as first seller

OUTPUT VARIABLES:
    session_id: Session identifier (e.g., "1_11-7-tr1")
    treatment: Treatment condition ("tr1" or "tr2")
    segment: Segment number within session (1-4)
    group_id: Group identifier (1-4, fixed throughout session)
    round: Round number within segment (1-14)
    player: Participant label
    public_signal: Cumulative signal shown at time of first sale (NA if no sales)
    state: True state of the asset this round (0 or 1)
    is_first_seller: Binary (1 if first seller, 0 otherwise)
    first_sale_period: Period when first sale occurred (NA if no sales)

DATA STRUCTURE:
    - 6 sessions x 4 segments x 14 rounds x 16 players
    - Expected total: ~5376 observations
"""

import pandas as pd
from pathlib import Path

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
OUTPUT_PATH = DATASTORE / "derived" / "first_seller_round_data.csv"

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
    """Build the first seller round dataset."""
    all_records = []

    print("Processing sessions...")
    for session_name, treatment in SESSIONS.items():
        print(f"  {session_name} (treatment {treatment})")
        records = process_session(session_name, treatment)
        all_records.extend(records)
        print(f"    -> {len(records)} player-round observations")

    df = pd.DataFrame(all_records)
    print_summary_statistics(df)
    save_dataset(df)

    return df


# =====
# Session processing
# =====
def process_session(session_name: str, treatment: str) -> list[dict]:
    """Process all segments for a session, return list of player-round records."""
    session_folder = DATASTORE / session_name
    records = []

    for segment_idx, segment in enumerate(SEGMENTS, start=1):
        try:
            df = load_segment_data(session_folder, segment)
        except FileNotFoundError as e:
            print(f"    Warning: {e}")
            continue

        segment_records = process_segment(df, session_name, segment_idx, treatment)
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
            group_df, session_name, segment_idx, int(group_id),
            int(round_num), treatment
        )
        records.extend(round_records)

    return records


# =====
# Data loading
# =====
def load_segment_data(session_folder: Path, segment: str) -> pd.DataFrame:
    """Load segment CSV file from session folder."""
    csv_files = sorted(session_folder.glob(f"{segment}_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found for {segment} in {session_folder}")
    if len(csv_files) > 1:
        raise ValueError(
            f"Multiple CSVs found for {segment} in {session_folder}: "
            f"{[f.name for f in csv_files]}. Expected exactly one."
        )
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
    """Process a single group-round and return player-round records."""
    players = group_df["participant.label"].unique()
    state = int(group_df["player.state"].iloc[0])

    # Find first sale info for this group-round
    first_sale_info = find_first_sale_info(group_df)
    first_sale_period = first_sale_info["first_sale_period"]
    public_signal = first_sale_info["public_signal"]
    first_sellers = first_sale_info["first_sellers"]

    records = []
    for player in players:
        is_first_seller = 1 if player in first_sellers else 0

        records.append({
            "session_id": session_name,
            "treatment": treatment,
            "segment": segment_idx,
            "group_id": group_id,
            "round": round_num,
            "player": player,
            "public_signal": public_signal,
            "state": state,
            "is_first_seller": is_first_seller,
            "first_sale_period": first_sale_period,
        })

    return records


def find_first_sale_info(group_df: pd.DataFrame) -> dict:
    """
    Find info about the first sale(s) in a group-round.

    Returns dict with:
        - first_sale_period: period when first sale occurred (None if no sales)
        - public_signal: signal value at first sale (None if no sales)
        - first_sellers: set of player labels who sold in the first sale period
    """
    # Build lookup of when each player sold
    players_with_sales = find_sellers_by_period(group_df)

    if not players_with_sales:
        return {
            "first_sale_period": None,
            "public_signal": None,
            "first_sellers": set(),
        }

    # Find minimum period where any sale occurred
    first_sale_period = min(players_with_sales.values())

    # Find all players who sold in that first period
    first_sellers = {
        player for player, period in players_with_sales.items()
        if period == first_sale_period
    }

    # Get the public signal at the first sale period
    first_period_rows = group_df[
        group_df["player.period_in_round"] == first_sale_period
    ]
    public_signal = first_period_rows["player.signal"].iloc[0]

    return {
        "first_sale_period": first_sale_period,
        "public_signal": public_signal,
        "first_sellers": first_sellers,
    }


def find_sellers_by_period(group_df: pd.DataFrame) -> dict:
    """
    Find which period each player sold in (if they sold).

    Returns dict mapping player label -> sell period (only includes sellers).
    """
    sellers = {}
    players = group_df["participant.label"].unique()

    for player in players:
        player_df = group_df[group_df["participant.label"] == player]
        sell_period = get_player_sell_period(player_df)
        if sell_period is not None:
            sellers[player] = sell_period

    return sellers


def get_player_sell_period(player_df: pd.DataFrame) -> int | None:
    """
    Find which period a player sold in.

    Returns period number when player sold, or None if never sold.
    """
    sorted_df = player_df.sort_values("player.period_in_round")
    prev_sold = 0

    for _, row in sorted_df.iterrows():
        current_sold = int(row["player.sold"]) if pd.notna(row["player.sold"]) else 0
        if current_sold == 1 and prev_sold == 0:
            return int(row["player.period_in_round"])
        prev_sold = current_sold

    return None


# =====
# Output functions
# =====
def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics for the dataset."""
    print("\n" + "=" * 50 + "\nDATASET SUMMARY\n" + "=" * 50)
    print(f"Total observations: {len(df)}")
    print(f"Unique sessions: {df['session_id'].nunique()}")
    print(f"Unique players: {df['player'].nunique()}")
    print(f"\nFirst sellers: {df['is_first_seller'].sum()}")
    print(f"First seller rate: {df['is_first_seller'].mean():.3f}")
    print("\nBy treatment:")
    print(df.groupby("treatment")["is_first_seller"].agg(["count", "sum", "mean"]))
    print("\nBy segment:")
    print(df.groupby("segment")["is_first_seller"].agg(["count", "sum", "mean"]))
    print_validation_checks(df)


def print_validation_checks(df: pd.DataFrame):
    """Print validation checks for the dataset."""
    print("\n" + "=" * 50 + "\nVALIDATION CHECKS\n" + "=" * 50)
    sales_df = df[df["first_sale_period"].notna()]
    if len(sales_df) > 0:
        print(f"first_sale_period range: {sales_df['first_sale_period'].min()}-"
              f"{sales_df['first_sale_period'].max()} (varies by round)")
    print(f"Segments present: {sorted(df['segment'].unique())} (should be [1, 2, 3, 4])")
    print(f"Treatments present: {sorted(df['treatment'].unique())} (should be ['tr1', 'tr2'])")
    invalid = df[(df["is_first_seller"] == 1) & (df["first_sale_period"].isna())]
    print(f"First sellers without first_sale_period: {len(invalid)} (should be 0)")


def save_dataset(df: pd.DataFrame):
    """Save dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


# %%
if __name__ == "__main__":
    df = main()
