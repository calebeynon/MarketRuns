"""
Purpose: Build dataset for holdout next-round analysis
Author: Claude
Date: 2026-01-22

Creates a dataset of holdouts (did not sell in round) with their next-round behavior.
Used to analyze whether holdouts who received lower payoffs are more likely to sell
in the following round.
"""

from pathlib import Path
import pandas as pd

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
INPUT_PATH = DATASTORE / "derived" / "individual_period_dataset_extended.csv"
OUTPUT_PATH = DATASTORE / "derived" / "holdout_next_round_analysis.csv"


# =====
# Main function
# =====
def main():
    """Build the holdout next-round analysis dataset."""
    print(f"Loading dataset from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows")

    df_holdouts = filter_to_holdouts(df)
    df_round = aggregate_to_round_level(df_holdouts)
    df_round = add_identifiers(df_round)
    df_round = link_next_round_behavior(df_round, df)
    df_round = compute_prior_sales(df_round, df)

    df_final = select_output_columns(df_round)
    validate_results(df_final)
    save_dataset(df_final)

    return df_final


# =====
# Filter to holdouts
# =====
def filter_to_holdouts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to holdout observations: treatment 1, state 0, did not sell.

    Holdouts are players who held their asset through the entire round
    when the state was 0 (asset worthless at liquidation).
    """
    mask = (
        (df["treatment"] == "tr1") &
        (df["state"] == 0) &
        (df["sold_in_round"] == 0)
    )
    df_holdouts = df[mask].copy()
    print(f"Filtered to {len(df_holdouts)} holdout period-observations")
    return df_holdouts


# =====
# Aggregate to round level
# =====
def aggregate_to_round_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to one row per (session, segment, round, player).

    Since round_payoff and sold_in_round are constant within a round,
    we just take the first row for each group.
    """
    group_cols = ["session_id", "segment", "round", "player", "group_id"]
    df_round = df.groupby(group_cols).first().reset_index()
    df_round = df_round[group_cols + ["round_payoff"]]
    print(f"Aggregated to {len(df_round)} holdout round-observations")
    return df_round


# =====
# Add identifiers
# =====
def add_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Add global_round and global_group_id columns."""
    df["global_round"] = (df["segment"] - 1) * 14 + df["round"]
    df["global_group_id"] = (
        df["session_id"] + "_" +
        df["segment"].astype(str) + "_" +
        df["group_id"].astype(str)
    )
    return df


# =====
# Link next round behavior
# =====
def link_next_round_behavior(
    df_holdouts: pd.DataFrame, df_full: pd.DataFrame
) -> pd.DataFrame:
    """
    Link each holdout to their sold_in_round and signal from round N+1.

    Does NOT link across segments (round 14 has no valid next round).
    Excludes observations with no valid next round.
    """
    # Build lookup for next round data
    next_round_data = build_next_round_lookup(df_full)

    # Create next round key for each holdout
    df_holdouts = df_holdouts.copy()
    df_holdouts["next_round"] = df_holdouts["round"] + 1

    # Merge to get next round behavior
    df_merged = df_holdouts.merge(
        next_round_data,
        left_on=["session_id", "segment", "next_round", "player"],
        right_on=["session_id", "segment", "round", "player"],
        how="inner",
        suffixes=("", "_next")
    )

    df_merged = df_merged.drop(columns=["next_round", "round_next"])
    print(f"Linked to next round: {len(df_merged)} observations (excluded round 14)")
    return df_merged


def build_next_round_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lookup for sold_in_round and signal from period 1 for each round.

    Signal from period 1 is always 0.5 (prior before any private signals).
    This serves as a baseline/placeholder for the next round signal.
    """
    df_tr1 = df[df["treatment"] == "tr1"].copy()

    # Get first period of each round (period 1)
    period_1 = df_tr1[df_tr1["period"] == 1].copy()
    result = period_1[
        ["session_id", "segment", "round", "player", "sold_in_round", "signal"]
    ].drop_duplicates()

    result = result.rename(columns={
        "sold_in_round": "sold_next_round",
        "signal": "signal_next_round"
    })
    return result


# =====
# Compute prior sales
# =====
def compute_prior_sales(
    df_holdouts: pd.DataFrame, df_full: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute cumulative count of rounds where player sold BEFORE current round.

    Counts across all prior segments and rounds within the same session.
    """
    # Build sales history for tr1 only
    sales_history = build_sales_history(df_full)

    # Merge prior sales
    df_merged = df_holdouts.merge(
        sales_history,
        on=["session_id", "segment", "round", "player"],
        how="left"
    )
    df_merged["prior_sales"] = df_merged["prior_sales"].fillna(0).astype(int)
    return df_merged


def build_sales_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build table of prior_sales for each (session, segment, round, player).

    prior_sales = cumulative count of rounds with sold_in_round=1 BEFORE current.
    """
    # Get one row per round with sold_in_round
    df_tr1 = df[df["treatment"] == "tr1"].copy()
    round_sales = df_tr1.groupby(
        ["session_id", "segment", "round", "player"]
    )["sold_in_round"].first().reset_index()

    # Add global_round for ordering
    round_sales["global_round"] = (
        (round_sales["segment"] - 1) * 14 + round_sales["round"]
    )

    # Sort by player and time
    round_sales = round_sales.sort_values(
        ["session_id", "player", "global_round"]
    )

    # Compute cumsum then shift within each player group
    # shift(1) moves values down, so each row gets the cumsum from prior rounds
    round_sales["prior_sales"] = round_sales.groupby(
        ["session_id", "player"]
    )["sold_in_round"].transform(lambda x: x.cumsum().shift(1, fill_value=0))

    return round_sales[["session_id", "segment", "round", "player", "prior_sales"]]


# =====
# Select output columns
# =====
def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order the final output columns."""
    output_cols = [
        "session_id", "segment", "round", "player", "group_id",
        "global_group_id", "global_round",
        "round_payoff", "sold_next_round", "signal_next_round", "prior_sales"
    ]
    return df[output_cols].copy()


# =====
# Validation
# =====
def validate_results(df: pd.DataFrame):
    """Print validation checks for the dataset."""
    print("\n" + "=" * 50)
    print("VALIDATION CHECKS")
    print("=" * 50)

    print(f"\nTotal observations: {len(df)}")
    print(f"Unique sessions: {df['session_id'].nunique()}")
    print(f"Unique players: {df.groupby('session_id')['player'].nunique().sum()}")

    print(f"\nPayoff distribution:")
    print(df["round_payoff"].value_counts().sort_index())

    print(f"\nSold next round distribution:")
    print(df["sold_next_round"].value_counts())

    print(f"\nPrior sales distribution:")
    print(df["prior_sales"].value_counts().sort_index())


# =====
# Output
# =====
def save_dataset(df: pd.DataFrame):
    """Save the dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved dataset to: {OUTPUT_PATH}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")


# %%
if __name__ == "__main__":
    df = main()
