"""
Purpose: Build individual-round dataset for market runs analysis
Author: Claude Code
Date: 2025-01-18

Creates a player-round level dataset. Each row represents one player in one
round, with their selling decision (which period they sold, or NA if held).

OUTPUT VARIABLES:
    session_id: Session identifier (e.g., "1_11-7-tr1")
    treatment: Treatment condition ("tr1" or "tr2")
    segment: Segment number within session (1-4)
    group_id: Group identifier (1-4, fixed throughout session)
    round: Round number within segment (1-14)
    player: Participant label
    signal: Posterior belief at time of sale (NA if player held)
    state: True state of the asset this round (0 or 1)
    sell_period: Period when player sold (NA if never sold)
    did_sell: Binary (0/1) - Did the player sell at any point in this round?
    sell_price: Price at which they sold (NA if held)

DATA STRUCTURE:
    - 6 sessions x 4 segments x variable rounds x 16 players
    - Expected total: ~2880 observations
"""

import pandas as pd
from pathlib import Path

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
OUTPUT_PATH = DATASTORE / "derived" / "individual_round_panel.csv"

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
    """Build the individual round dataset."""
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
    records = []
    players = group_df["participant.label"].unique()

    # Get state (same for all players in round)
    state = int(group_df["player.state"].iloc[0])

    for player in players:
        player_df = group_df[group_df["participant.label"] == player]
        record = build_player_round_record(
            player_df, session_name, segment_idx, group_id,
            round_num, player, treatment, state
        )
        records.append(record)

    return records


def build_player_round_record(
    player_df: pd.DataFrame,
    session_name: str,
    segment_idx: int,
    group_id: int,
    round_num: int,
    player: str,
    treatment: str,
    state: int
) -> dict:
    """Build a single player-round record."""
    sell_period, sell_price, signal_at_sale = get_player_sell_info(player_df)

    return {
        "session_id": session_name,
        "treatment": treatment,
        "segment": segment_idx,
        "group_id": group_id,
        "round": round_num,
        "player": player,
        "signal": signal_at_sale,
        "state": state,
        "sell_period": sell_period,
        "did_sell": 1 if sell_period is not None else 0,
        "sell_price": sell_price,
    }


def get_player_sell_info(player_df: pd.DataFrame) -> tuple:
    """
    Find which period a player sold in, price, and posterior belief at sale.

    Returns:
        (sell_period, sell_price, signal_at_sale) - all None if never sold
    """
    sorted_df = player_df.sort_values("player.period_in_round")
    prev_sold = 0

    for _, row in sorted_df.iterrows():
        current_sold = int(row["player.sold"]) if pd.notna(row["player.sold"]) else 0
        if current_sold == 1 and prev_sold == 0:
            period = int(row["player.period_in_round"])
            price = row["player.price"]
            signal = row["player.signal"]
            return (period, price, signal)
        prev_sold = current_sold

    return (None, None, None)


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
    print(f"\nPlayers who sold: {df['did_sell'].sum()}")
    print(f"Sell rate: {df['did_sell'].mean():.3f}")
    print(f"\nBy treatment:")
    print(df.groupby("treatment")["did_sell"].agg(["count", "sum", "mean"]))
    print(f"\nBy segment:")
    print(df.groupby("segment")["did_sell"].agg(["count", "sum", "mean"]))
    print(f"\nSell period distribution (excluding non-sellers):")
    print(df[df["did_sell"] == 1]["sell_period"].value_counts().sort_index())

    # Validation checks
    print("\n" + "=" * 50)
    print("VALIDATION CHECKS")
    print("=" * 50)

    # Check sell_period range
    sellers = df[df["did_sell"] == 1]
    if len(sellers) > 0:
        sp_min = sellers["sell_period"].min()
        sp_max = sellers["sell_period"].max()
        print(f"sell_period range: {sp_min} to {sp_max} (should be 1-4)")

    # Check consistency: did_sell vs sell_period
    inconsistent = ((df["did_sell"] == 1) & (df["sell_period"].isna())).sum()
    inconsistent += ((df["did_sell"] == 0) & (df["sell_period"].notna())).sum()
    print(f"Inconsistent did_sell/sell_period: {inconsistent} (should be 0)")

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
