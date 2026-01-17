"""
Purpose: Build dataset for first sale analysis
Author: Claude Code
Date: 2025-01-11

Creates a group-round level dataset with signal at first sale for regression:
    signal_at_first_sale ~ treatment + group_FE + segment
"""

import pandas as pd
from pathlib import Path

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
OUTPUT_PATH = DATASTORE / "derived" / "first_sale_data.csv"

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
    """Build the first sale dataset."""
    all_records = []

    print("Processing sessions...")
    for session_name, treatment in SESSIONS.items():
        print(f"  {session_name} (treatment {treatment})")
        records = process_session(session_name, treatment)
        all_records.extend(records)
        print(f"    -> {len(records)} group-round observations")

    # Create DataFrame
    df = pd.DataFrame(all_records)

    # Summary statistics
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total observations: {len(df)}")
    print(f"Observations with sales: {df['signal_at_first_sale'].notna().sum()}")
    print(f"Observations without sales: {df['signal_at_first_sale'].isna().sum()}")
    print(f"\nBy treatment:")
    print(df.groupby("treatment")["signal_at_first_sale"].agg(["count", "mean", "std"]))
    print(f"\nBy segment:")
    print(df.groupby("segment_num")["signal_at_first_sale"].agg(["count", "mean", "std"]))

    # Save to csv
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")

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

        # Get unique group-round combinations
        group_rounds = df.groupby(
            ["group.id_in_subsession", "player.round_number_in_segment"]
        )

        for (group_id, round_num), group_df in group_rounds:
            first_sale_info = get_first_sale_for_group_round(group_df)

            # Create unique group identifier across sessions
            global_group_id = f"{session_name}_seg{segment_idx}_g{group_id}"

            records.append({
                "session": session_name,
                "treatment": treatment,
                "segment": segment,
                "segment_num": segment_idx,
                "group_id": int(group_id),
                "global_group_id": global_group_id,
                "round_num": int(round_num),
                "first_sale_period": first_sale_info["first_sale_period"],
                "signal_at_first_sale": first_sale_info["signal_at_first_sale"],
                "n_sellers_first_period": first_sale_info["n_sellers_first_period"],
            })

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


def get_first_sale_for_group_round(df: pd.DataFrame) -> dict:
    """
    Find the first sale in a group-round and return the signal at that time.

    Returns dict with:
        - first_sale_period: period when first sale occurred (None if no sales)
        - signal_at_first_sale: signal value at first sale
        - n_sellers_first_period: how many sold in that first period
    """
    # Filter to rows where someone sold this period
    sales = df[df["player.sold"] == 1].copy()

    if sales.empty:
        return {
            "first_sale_period": None,
            "signal_at_first_sale": None,
            "n_sellers_first_period": 0,
        }

    # Find the minimum period where a sale occurred
    first_sale_period = sales["player.period_in_round"].min()

    # Get signal at first sale (all players share same signal, take first)
    first_sale_rows = sales[sales["player.period_in_round"] == first_sale_period]
    signal_at_first_sale = first_sale_rows["player.signal"].iloc[0]
    n_sellers = len(first_sale_rows)

    return {
        "first_sale_period": first_sale_period,
        "signal_at_first_sale": signal_at_first_sale,
        "n_sellers_first_period": n_sellers,
    }


# %%
if __name__ == "__main__":
    df = main()
