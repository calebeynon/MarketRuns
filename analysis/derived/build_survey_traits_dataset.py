"""
Purpose: Build survey personality traits dataset from oTree survey exports
Author: Claude Code
Date: 2026-01-28

Extracts BFI-10 personality traits, impulsivity, state anxiety, and demographics
from post-experiment survey data. Each row is one participant.

OUTPUT VARIABLES:
    session_id: Session identifier (e.g., "1_11-7-tr1")
    player: Participant label (A-R, excluding I and O)
    extraversion: BFI-10 extraversion score (1-7)
    agreeableness: BFI-10 agreeableness score (1-7)
    conscientiousness: BFI-10 conscientiousness score (1-7)
    neuroticism: BFI-10 neuroticism score (1-7)
    openness: BFI-10 openness score (1-7)
    impulsivity: Impulsivity score (1-7, mean of 8 items)
    state_anxiety: State anxiety score (1-4, mean of 6 items)
    age: Participant age
    gender: Participant gender
"""

import pandas as pd
from pathlib import Path

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
OUTPUT_PATH = DATASTORE / "derived" / "survey_traits.csv"

SESSIONS = {
    "1_11-7-tr1": "tr1",
    "2_11-10-tr2": "tr2",
    "3_11-11-tr2": "tr2",
    "4_11-12-tr1": "tr1",
    "5_11-14-tr2": "tr2",
    "6_11-18-tr1": "tr1",
}

# Likert scale mappings
LIKERT_7 = {
    "Strongly Disagree": 1,
    "Disagree Moderately": 2,
    "Disagree a little": 3,
    "Neither agree nor disagree": 4,
    "Agree a little": 5,
    "Agree Moderately": 6,
    "Strongly Agree": 7,
}

LIKERT_4 = {
    "Not at all": 1,
    "Somewhat": 2,
    "Moderately": 3,
    "Very much": 4,
}


# =====
# Main function
# =====
def main():
    """Build the survey traits dataset."""
    all_records = []

    print("Processing survey data...")
    for session_id in SESSIONS:
        print(f"  {session_id}")
        records = process_session(session_id)
        all_records.extend(records)
        print(f"    -> {len(records)} participants")

    df = pd.DataFrame(all_records)
    print_summary(df)
    save_dataset(df)

    return df


# =====
# Session processing
# =====
def process_session(session_id: str) -> list[dict]:
    """Load survey CSV for a session and extract traits for each participant."""
    session_folder = DATASTORE / session_id
    csv_files = list(session_folder.glob("survey_*.csv"))
    if not csv_files:
        print(f"    Warning: No survey CSV found in {session_folder}")
        return []

    df = pd.read_csv(csv_files[0])
    records = []
    for _, row in df.iterrows():
        if has_missing_survey_data(row):
            label = row["participant.label"]
            print(f"    Warning: Skipping {label} (missing survey responses)")
            continue
        record = extract_participant_traits(row, session_id)
        records.append(record)

    return records


def has_missing_survey_data(row: pd.Series) -> bool:
    """Check if any required survey question (q1-q24) is missing."""
    return any(pd.isna(row[f"player.q{i}"]) for i in range(1, 25))


def extract_participant_traits(row: pd.Series, session_id: str) -> dict:
    """Extract all trait scores and demographics from a single participant row."""
    return {
        "session_id": session_id,
        "player": row["participant.label"],
        "extraversion": compute_extraversion(row),
        "agreeableness": compute_agreeableness(row),
        "conscientiousness": compute_conscientiousness(row),
        "neuroticism": compute_neuroticism(row),
        "openness": compute_openness(row),
        "impulsivity": compute_impulsivity(row),
        "state_anxiety": compute_state_anxiety(row),
        "age": row["player.q25"],
        "gender": row["player.q26"],
    }


# =====
# Likert encoding helpers
# =====
def encode_7pt(value: str) -> int:
    """Encode a 7-point Likert response to numeric (1-7)."""
    return LIKERT_7[value]


def encode_4pt(value: str) -> int:
    """Encode a 4-point Likert response to numeric (1-4)."""
    return LIKERT_4[value]


def reverse_7pt(value: int) -> int:
    """Reverse code a 7-point Likert value."""
    return 8 - value


def reverse_4pt(value: int) -> int:
    """Reverse code a 4-point Likert value."""
    return 5 - value


# =====
# BFI-10 trait scoring (each = mean of 2 items)
# =====
def compute_extraversion(row: pd.Series) -> float:
    """Extraversion: q7(+), q12(R). Mean of 2 items."""
    forward = encode_7pt(row["player.q7"])
    reverse = reverse_7pt(encode_7pt(row["player.q12"]))
    return (forward + reverse) / 2


def compute_agreeableness(row: pd.Series) -> float:
    """Agreeableness: q13(+), q8(R). Mean of 2 items."""
    forward = encode_7pt(row["player.q13"])
    reverse = reverse_7pt(encode_7pt(row["player.q8"]))
    return (forward + reverse) / 2


def compute_conscientiousness(row: pd.Series) -> float:
    """Conscientiousness: q9(+), q14(R). Mean of 2 items."""
    forward = encode_7pt(row["player.q9"])
    reverse = reverse_7pt(encode_7pt(row["player.q14"]))
    return (forward + reverse) / 2


def compute_neuroticism(row: pd.Series) -> float:
    """Neuroticism: q10(+), q15(R). Mean of 2 items."""
    forward = encode_7pt(row["player.q10"])
    reverse = reverse_7pt(encode_7pt(row["player.q15"]))
    return (forward + reverse) / 2


def compute_openness(row: pd.Series) -> float:
    """Openness: q11(+), q16(R). Mean of 2 items."""
    forward = encode_7pt(row["player.q11"])
    reverse = reverse_7pt(encode_7pt(row["player.q16"]))
    return (forward + reverse) / 2


# =====
# Impulsivity scoring (mean of 8 items, 7-point scale)
# =====
def compute_impulsivity(row: pd.Series) -> float:
    """Impulsivity: forward q18,q19,q23,q24; reverse q17,q20,q21,q22."""
    forward_qs = ["player.q18", "player.q19", "player.q23", "player.q24"]
    reverse_qs = ["player.q17", "player.q20", "player.q21", "player.q22"]

    forward_vals = [encode_7pt(row[q]) for q in forward_qs]
    reverse_vals = [reverse_7pt(encode_7pt(row[q])) for q in reverse_qs]

    return sum(forward_vals + reverse_vals) / 8


# =====
# State anxiety scoring (mean of 6 items, 4-point scale)
# =====
def compute_state_anxiety(row: pd.Series) -> float:
    """State anxiety: reverse q1,q2,q3 (positive mood); forward q4,q5,q6."""
    reverse_qs = ["player.q1", "player.q2", "player.q3"]
    forward_qs = ["player.q4", "player.q5", "player.q6"]

    reverse_vals = [reverse_4pt(encode_4pt(row[q])) for q in reverse_qs]
    forward_vals = [encode_4pt(row[q]) for q in forward_qs]

    return sum(reverse_vals + forward_vals) / 6


# =====
# Output
# =====
def print_summary(df: pd.DataFrame):
    """Print summary statistics for the traits dataset."""
    print("\n" + "=" * 50)
    print("SURVEY TRAITS SUMMARY")
    print("=" * 50)
    print(f"Total participants: {len(df)}")
    print(f"Sessions: {df['session_id'].nunique()}")

    trait_cols = [
        "extraversion", "agreeableness", "conscientiousness",
        "neuroticism", "openness", "impulsivity", "state_anxiety",
    ]
    print("\nTrait score ranges:")
    for col in trait_cols:
        print(f"  {col}: {df[col].min():.2f} - {df[col].max():.2f} "
              f"(mean={df[col].mean():.2f})")

    print(f"\nAge range: {df['age'].min()} - {df['age'].max()}")
    print(f"Gender distribution:\n{df['gender'].value_counts().to_string()}")


def save_dataset(df: pd.DataFrame):
    """Save dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


# %%
if __name__ == "__main__":
    main()
