"""
Purpose: Extend individual_period_dataset with round-level payoff and sold columns
Author: Claude
Date: 2026-01-22

Adds two columns to the individual period dataset:
    - round_payoff: The payoff for the round from raw oTree data (player.round_N_payoff)
    - sold_in_round: Whether player sold at any point in the round (0 or 1)
"""

from pathlib import Path
import pandas as pd

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
INPUT_PATH = DATASTORE / "derived" / "individual_period_dataset.csv"
OUTPUT_PATH = DATASTORE / "derived" / "individual_period_dataset_extended.csv"

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
    """Extend the individual period dataset with round-level columns."""
    print(f"Loading dataset from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows")

    df = add_sold_in_round(df)
    payoffs_df = load_all_round_payoffs()
    df = merge_round_payoffs(df, payoffs_df)

    validate_results(df)
    save_dataset(df)

    return df


# =====
# Compute sold_in_round
# =====
def add_sold_in_round(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sold_in_round column: 1 if player sold or already_sold in any period.

    Logic: For each (session, segment, round, player), if sold=1 OR already_sold=1
    in ANY period, then sold_in_round=1 for ALL periods in that round.
    """
    group_cols = ["session_id", "segment", "round", "player"]

    # sold=1 OR already_sold=1 means player sold at some point
    df["_sold_ever"] = (df["sold"] == 1) | (df["already_sold"] == 1)

    # Get max of _sold_ever for each round (1 if sold anytime, 0 otherwise)
    sold_by_round = df.groupby(group_cols)["_sold_ever"].max().reset_index()
    sold_by_round["sold_in_round"] = sold_by_round["_sold_ever"].astype(int)
    sold_by_round = sold_by_round.drop(columns=["_sold_ever"])

    # Merge back to original dataframe
    df = df.drop(columns=["_sold_ever"])
    df = df.merge(sold_by_round, on=group_cols, how="left")

    print(f"  Added sold_in_round: {df['sold_in_round'].sum()} player-periods with sale")
    return df


# =====
# Load round payoffs from raw data
# =====
def load_all_round_payoffs() -> pd.DataFrame:
    """Load round payoffs from all sessions and segments."""
    all_payoffs = []

    print("Loading round payoffs from raw data...")
    for session_name in SESSIONS:
        for segment_idx, segment in enumerate(SEGMENTS, start=1):
            payoffs = load_segment_payoffs(session_name, segment_idx, segment)
            if payoffs is not None:
                all_payoffs.append(payoffs)

    return pd.concat(all_payoffs, ignore_index=True)


def load_segment_payoffs(
    session_name: str, segment_idx: int, segment: str
) -> pd.DataFrame:
    """
    Load round payoffs for a single segment from raw oTree CSV.

    Returns DataFrame with: session_id, segment, round, player, round_payoff
    """
    session_folder = DATASTORE / session_name
    csv_files = list(session_folder.glob(f"{segment}_*.csv"))

    if not csv_files:
        print(f"    Warning: No CSV found for {segment} in {session_folder}")
        return None

    df = pd.read_csv(csv_files[0])
    return extract_round_payoffs(df, session_name, segment_idx)


def extract_round_payoffs(
    df: pd.DataFrame, session_name: str, segment_idx: int
) -> pd.DataFrame:
    """
    Extract round payoffs from segment data.

    For each (player, round), get the payoff from player.round_N_payoff column
    at the last period of that round. The payoff value is only correct during
    the round itself, so we must filter to rows for that specific round.
    """
    players = df["participant.label"].unique()
    rounds = df["player.round_number_in_segment"].unique()
    records = []

    for player in players:
        player_df = df[df["participant.label"] == player]
        for round_num in rounds:
            payoff = get_payoff_for_round(player_df, int(round_num))
            records.append({
                "session_id": session_name,
                "segment": segment_idx,
                "round": int(round_num),
                "player": player,
                "round_payoff": payoff,
            })

    return pd.DataFrame(records)


def get_payoff_for_round(player_df: pd.DataFrame, round_num: int) -> float:
    """
    Get the payoff for a specific round.

    Filter to rows for that round, then get the payoff value from the last
    period (max period_in_round). The payoff column is player.round_N_payoff.
    """
    round_df = player_df[player_df["player.round_number_in_segment"] == round_num]
    if round_df.empty:
        return 0.0

    # Get the last period in this round
    last_period_df = round_df[
        round_df["player.period_in_round"] == round_df["player.period_in_round"].max()
    ]

    payoff_col = f"player.round_{round_num}_payoff"
    if payoff_col in last_period_df.columns:
        value = last_period_df[payoff_col].iloc[0]
        return float(value) if pd.notna(value) else 0.0
    return 0.0


# =====
# Merge payoffs
# =====
def merge_round_payoffs(
    df: pd.DataFrame, payoffs_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge round payoffs into the main dataset."""
    merge_cols = ["session_id", "segment", "round", "player"]
    df = df.merge(payoffs_df, on=merge_cols, how="left")

    # Fill any missing payoffs with 0
    df["round_payoff"] = df["round_payoff"].fillna(0.0)

    print(f"  Added round_payoff: mean={df['round_payoff'].mean():.2f}")
    return df


# =====
# Validation
# =====
def validate_results(df: pd.DataFrame):
    """Print validation checks for the extended dataset."""
    print("\n" + "=" * 50)
    print("VALIDATION CHECKS")
    print("=" * 50)

    # Check sold_in_round consistency
    consistency_check = df.groupby(
        ["session_id", "segment", "round", "player"]
    )["sold_in_round"].nunique()
    inconsistent = (consistency_check > 1).sum()
    print(f"Rounds with inconsistent sold_in_round: {inconsistent} (should be 0)")

    # Check round_payoff consistency
    payoff_check = df.groupby(
        ["session_id", "segment", "round", "player"]
    )["round_payoff"].nunique()
    inconsistent_payoff = (payoff_check > 1).sum()
    print(f"Rounds with inconsistent round_payoff: {inconsistent_payoff} (should be 0)")

    # Check payoff distribution
    print(f"\nPayoff distribution by sold_in_round:")
    print(df.groupby("sold_in_round")["round_payoff"].describe())


# =====
# Output
# =====
def save_dataset(df: pd.DataFrame):
    """Save the extended dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved extended dataset to: {OUTPUT_PATH}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")


# %%
if __name__ == "__main__":
    df = main()
