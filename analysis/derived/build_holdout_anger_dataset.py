"""
Purpose: Build holdout dataset with iMotions anger data for DiD analysis
Author: Claude
Date: 2026-02-02

Extends the holdout_next_round_analysis dataset with anger measurements from
iMotions facial expression data. Anger is measured during the Results phase
at the end of each round when holdouts learn their payoff.
"""

from pathlib import Path
import pandas as pd

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
INPUT_HOLDOUT = DATASTORE / "derived" / "holdout_next_round_analysis.csv"
IMOTIONS_DIR = DATASTORE / "imotions"
OUTPUT_PATH = DATASTORE / "derived" / "holdout_anger_analysis.csv"

# CONSTANTS
SESSION_TO_IMOTIONS = {
    "1_11-7-tr1": 1,
    "4_11-12-tr1": 4,
    "6_11-18-tr1": 6,
}


# =====
# Main function
# =====
def main():
    """Build the holdout anger analysis dataset."""
    print(f"Loading holdout dataset from: {INPUT_HOLDOUT}")
    df = pd.read_csv(INPUT_HOLDOUT)
    print(f"Loaded {len(df)} holdout observations")

    df = filter_to_imotions_sessions(df)
    df = add_chat_available(df)
    df = add_anger_results(df)

    validate_results(df)
    save_dataset(df)

    return df


# =====
# Filter to sessions with iMotions data
# =====
def filter_to_imotions_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset to only sessions with iMotions data."""
    valid_sessions = list(SESSION_TO_IMOTIONS.keys())
    df_filtered = df[df["session_id"].isin(valid_sessions)].copy()
    print(f"Filtered to {len(df_filtered)} observations with iMotions data")
    return df_filtered


# =====
# Add chat_available indicator
# =====
def add_chat_available(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary indicator for chat availability (segments 3-4)."""
    df = df.copy()
    df["chat_available"] = (df["segment"] >= 3).astype(int)
    return df


# =====
# Add anger during Results phase
# =====
def add_anger_results(df: pd.DataFrame) -> pd.DataFrame:
    """Add mean anger during Results phase for each observation."""
    df = df.copy()
    df["anger_results"] = None
    counters = {"missing_files": 0, "missing_results": 0, "empty_anger": 0}

    for idx, row in df.iterrows():
        anger = get_anger_for_row(row, counters)
        if anger is not None:
            df.at[idx, "anger_results"] = anger

    print_missing_data_summary(**counters)
    return df


def get_anger_for_row(row: pd.Series, counters: dict) -> float:
    """Extract anger value for a single row, updating counters for missing data."""
    imotions_session = SESSION_TO_IMOTIONS[row["session_id"]]
    imotions_df = load_imotions_file(imotions_session, row["player"])

    if imotions_df is None:
        counters["missing_files"] += 1
        return None

    anger = extract_results_anger(imotions_df, row["segment"], row["round"])
    return classify_anger_result(anger, counters)


def classify_anger_result(anger: float, counters: dict) -> float:
    """Classify anger result and update counters. Returns value or None."""
    if anger is None:
        counters["missing_results"] += 1
        return None
    if pd.isna(anger):
        counters["empty_anger"] += 1
    return anger


def load_imotions_file(session: int, player: str) -> pd.DataFrame:
    """
    Load iMotions CSV for a participant.

    Returns None if file not found.
    """
    suffix = session + 2
    session_dir = IMOTIONS_DIR / str(session)

    if not session_dir.exists():
        return None

    files = list(session_dir.glob(f"*_{player}{suffix}.csv"))
    files = [f for f in files if "ExportMerge" not in f.name]

    if not files:
        return None

    return pd.read_csv(files[0], skiprows=24, encoding="utf-8-sig")


def extract_results_anger(
    df: pd.DataFrame, segment: int, round_num: int
) -> float:
    """
    Extract mean anger during Results phase.

    Annotations have format like 's1r1m4Results' (with market number).
    Returns None if no Results data found, NaN if anger values are empty.
    """
    pattern_prefix = f"s{segment}r{round_num}m"
    results_df = filter_to_results_phase(df, pattern_prefix)

    if results_df.empty:
        return None

    return compute_mean_anger(results_df)


def compute_mean_anger(df: pd.DataFrame) -> float:
    """Compute mean anger from filtered dataframe. Returns NaN if no values."""
    anger_values = df["Anger"].dropna()
    return anger_values.mean() if not anger_values.empty else float("nan")


def filter_to_results_phase(df: pd.DataFrame, pattern_prefix: str) -> pd.DataFrame:
    """
    Filter to rows where annotation matches Results phase (not ResultsWait).

    Pattern: starts with prefix (e.g. 's1r1m'), ends with 'Results' (not 'Wait').
    """
    annotation_col = "Respondent Annotations active"
    annotations = df[annotation_col].fillna("")
    mask = (
        annotations.str.startswith(pattern_prefix) &
        annotations.str.endswith("Results")
    )
    return df[mask]


def print_missing_data_summary(
    missing_files: int, missing_results: int, empty_anger: int
):
    """Print summary of missing data."""
    print("\n" + "=" * 50)
    print("MISSING DATA SUMMARY")
    print("=" * 50)
    print(f"Missing iMotions files: {missing_files}")
    print(f"Missing Results annotations: {missing_results}")
    print(f"Empty Anger values (face not detected): {empty_anger}")


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

    print(f"\nChat available distribution:")
    print(df["chat_available"].value_counts())

    valid_anger = df["anger_results"].notna()
    print(f"\nAnger data coverage: {valid_anger.sum()}/{len(df)} observations")

    if valid_anger.any():
        print(f"\nAnger statistics (Results phase):")
        print(df.loc[valid_anger, "anger_results"].describe())


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
    main()
