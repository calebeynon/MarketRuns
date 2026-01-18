"""
Purpose: Build dataset for group-round selling timing analysis
Author: Claude Code
Date: 2025-01-18

Creates a group-round level dataset with seller timing information.
Each row contains selling info for up to 4 sellers ordered by period (then label).
"""

import pandas as pd
from pathlib import Path
from typing import Optional

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
OUTPUT_PATH = DATASTORE / "derived" / "group_round_timing.csv"

# Session folders with treatment indicators
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
# Main function
# =====
def main():
    """Build the group-round timing dataset."""
    all_records = []

    print("Processing sessions...")
    for session_name, treatment in SESSIONS.items():
        print(f"  {session_name} (treatment {treatment})")
        records = process_session(session_name, treatment)
        all_records.extend(records)
        print(f"    -> {len(records)} group-round observations")

    df = pd.DataFrame(all_records)
    print_summary_statistics(df)
    save_dataset(df)

    return df


# =====
# Session processing
# =====
def process_session(session_name: str, treatment: int) -> list[dict]:
    """Process all segments for a session, return list of group-round records."""
    session_folder = DATASTORE / session_name
    records = []

    for segment_idx, segment in enumerate(SEGMENTS, start=1):
        try:
            df = load_segment_data(session_folder, segment)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue

        segment_records = process_segment(
            df, session_name, treatment, segment, segment_idx
        )
        records.extend(segment_records)

    return records


def process_segment(
    df: pd.DataFrame,
    session_name: str,
    treatment: int,
    segment: str,
    segment_idx: int,
) -> list[dict]:
    """Process a single segment, return group-round records."""
    records = []
    group_rounds = df.groupby(
        ["group.id_in_subsession", "player.round_number_in_segment"]
    )

    for (group_id, round_num), group_df in group_rounds:
        record = build_group_round_record(
            group_df, session_name, treatment, segment, segment_idx,
            int(group_id), int(round_num)
        )
        records.append(record)

    return records


def build_group_round_record(
    group_df: pd.DataFrame,
    session_name: str,
    treatment: int,
    segment: str,
    segment_idx: int,
    group_id: int,
    round_num: int,
) -> dict:
    """Build a single group-round record with seller timing."""
    sellers = get_sellers_with_timing(group_df)
    state = group_df["player.state"].iloc[0]
    global_group_id = f"{session_name}_seg{segment_idx}_g{group_id}"

    record = {
        "session": session_name,
        "treatment": treatment,
        "segment": segment,
        "segment_num": segment_idx,
        "group_id": group_id,
        "global_group_id": global_group_id,
        "round_num": round_num,
        "state": int(state),
        "n_sellers": len(sellers),
    }

    # Add seller columns (up to 4 sellers)
    for i in range(4):
        seller_num = i + 1
        if i < len(sellers):
            record[f"seller_{seller_num}_period"] = sellers[i]["period"]
            record[f"seller_{seller_num}_label"] = sellers[i]["label"]
            record[f"seller_{seller_num}_signal"] = sellers[i]["signal"]
        else:
            record[f"seller_{seller_num}_period"] = None
            record[f"seller_{seller_num}_label"] = None
            record[f"seller_{seller_num}_signal"] = None

    return record


# =====
# Seller timing extraction
# =====
def get_sellers_with_timing(df: pd.DataFrame) -> list[dict]:
    """
    Extract sellers from a group-round, ordered by period then label.

    Returns list of dicts with keys: period, label, signal
    """
    sales = df[df["player.sold"] == 1].copy()

    if sales.empty:
        return []

    # Find minimum period for each player who sold
    seller_info = find_seller_info(sales)

    # Sort by period, then by label
    seller_info.sort(key=lambda x: (x["period"], x["label"]))

    return seller_info


def find_seller_info(sales_df: pd.DataFrame) -> list[dict]:
    """Extract period, label, and signal for each seller."""
    sellers = []
    grouped = sales_df.groupby("player.id_in_group")

    for _, player_sales in grouped:
        first_sale = player_sales.loc[player_sales["player.period_in_round"].idxmin()]
        sellers.append({
            "period": int(first_sale["player.period_in_round"]),
            "label": first_sale["participant.label"],
            "signal": first_sale["player.signal"],
        })

    return sellers


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
# Output and summary
# =====
def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics for the dataset."""
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total observations: {len(df)}")
    print(f"Observations with sales: {(df['n_sellers'] > 0).sum()}")
    print(f"Observations without sales: {(df['n_sellers'] == 0).sum()}")
    print(f"\nSeller count distribution:")
    print(df["n_sellers"].value_counts().sort_index())
    print(f"\nBy treatment:")
    print(df.groupby("treatment")["n_sellers"].agg(["count", "mean", "std"]))
    print(f"\nBy segment:")
    print(df.groupby("segment_num")["n_sellers"].agg(["count", "mean", "std"]))


def save_dataset(df: pd.DataFrame):
    """Save dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


# %%
if __name__ == "__main__":
    df = main()
