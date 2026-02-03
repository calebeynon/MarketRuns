"""
Purpose: Extract personality trait scores from post-experiment survey
Author: Claude Code
Date: 2026-02-02

Creates a player-level dataset with personality trait scores derived from:
- State Anxiety (STAI-6): q1-q6, 4-point scale
- TIPI Big Five: q7-q16, 7-point scale
- Impulsivity (BIS-style): q17-q24, 7-point scale

OUTPUT VARIABLES:
    session_id: Session identifier (e.g., "1_11-7-tr1")
    player: Participant label (A-P)
    state_anxiety: Mean of 6 items after reverse coding (range 1-4)
    extraversion: Mean of 2 items after reverse coding (range 1-7)
    agreeableness: Mean of 2 items after reverse coding (range 1-7)
    conscientiousness: Mean of 2 items after reverse coding (range 1-7)
    neuroticism: Mean of 2 items after reverse coding (range 1-7)
    openness: Mean of 2 items after reverse coding (range 1-7)
    impulsivity: Mean of 8 items after reverse coding (range 1-7)

DATA STRUCTURE:
    - 6 sessions x 16 players = 96 total participants
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
OUTPUT_PATH = DATASTORE / "derived" / "personality_traits_dataset.csv"

# Session folders
SESSIONS = [
    "1_11-7-tr1",
    "2_11-10-tr2",
    "3_11-11-tr2",
    "4_11-12-tr1",
    "5_11-14-tr2",
    "6_11-18-tr1",
]

# Likert scale mappings
SCALE_4PT = {
    "Not at all": 1,
    "Somewhat": 2,
    "Moderately": 3,
    "Very much": 4,
}

SCALE_7PT = {
    "Strongly Disagree": 1,
    "Disagree Moderately": 2,
    "Disagree a little": 3,
    "Neither agree nor disagree": 4,
    "Agree a little": 5,
    "Agree Moderately": 6,
    "Strongly Agree": 7,
}


# =====
# Main function
# =====
def main():
    """Build the personality traits dataset."""
    all_records, missing_count = collect_all_records()
    df = pd.DataFrame(all_records)
    print_summary_statistics(df, missing_count)
    validate_dataset(df)
    save_dataset(df)
    return df


def collect_all_records() -> tuple[list[dict], int]:
    """Collect records from all sessions."""
    all_records = []
    missing_count = 0

    print("Processing sessions...")
    for session_name in SESSIONS:
        print(f"  {session_name}")
        try:
            records, n_missing = process_session(session_name)
            all_records.extend(records)
            missing_count += n_missing
            print(f"    -> {len(records)} participants")
        except FileNotFoundError as e:
            print(f"    Warning: {e}")

    return all_records, missing_count


# =====
# Session processing
# =====
def process_session(session_name: str) -> tuple[list[dict], int]:
    """Process survey data for a session, return list of player records."""
    session_folder = DATASTORE / session_name
    survey_df = load_survey_data(session_folder)

    records = []
    missing_count = 0

    for _, row in survey_df.iterrows():
        player = row.get("participant.label")
        if pd.isna(player):
            continue

        record, has_missing = build_player_record(row, session_name, player)
        records.append(record)
        if has_missing:
            missing_count += 1

    return records, missing_count


def build_player_record(
    row: pd.Series,
    session_name: str,
    player: str
) -> tuple[dict, bool]:
    """Build a single player record with all personality traits."""
    responses = extract_survey_responses(row)
    has_missing = check_missing_responses(responses, player)
    traits = calculate_all_traits(responses)

    record = {
        "session_id": session_name,
        "player": player,
        **traits,
    }
    return record, has_missing


def check_missing_responses(responses: dict, player: str) -> bool:
    """Check for missing survey responses and warn if found."""
    if any(pd.isna(v) for v in responses.values()):
        print(f"      Warning: Missing survey data for {player}")
        return True
    return False


def calculate_all_traits(responses: dict) -> dict:
    """Calculate all personality trait scores from responses."""
    return {
        "state_anxiety": calculate_state_anxiety(responses),
        "extraversion": calculate_extraversion(responses),
        "agreeableness": calculate_agreeableness(responses),
        "conscientiousness": calculate_conscientiousness(responses),
        "neuroticism": calculate_neuroticism(responses),
        "openness": calculate_openness(responses),
        "impulsivity": calculate_impulsivity(responses),
    }


# =====
# Data loading
# =====
def load_survey_data(session_folder: Path) -> pd.DataFrame:
    """Load survey CSV file from session folder."""
    csv_files = sorted(session_folder.glob("survey_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No survey CSV found in {session_folder}")
    if len(csv_files) > 1:
        raise ValueError(
            f"Multiple survey CSVs found in {session_folder}: "
            f"{[f.name for f in csv_files]}. Expected exactly one."
        )
    return pd.read_csv(csv_files[0])


def extract_survey_responses(row: pd.Series) -> dict:
    """Extract survey question responses from a row."""
    responses = {}
    for i in range(1, 25):
        col_name = f"player.q{i}"
        responses[f"q{i}"] = row.get(col_name)
    return responses


# =====
# Likert encoding functions
# =====
def encode_4pt(value: str) -> float:
    """Convert 4-point Likert text to numeric (1-4)."""
    if pd.isna(value):
        return np.nan
    return SCALE_4PT.get(value, np.nan)


def encode_7pt(value: str) -> float:
    """Convert 7-point Likert text to numeric (1-7)."""
    if pd.isna(value):
        return np.nan
    return SCALE_7PT.get(value, np.nan)


def reverse_4pt(value: float) -> float:
    """Reverse code a 4-point scale item: 5 - original."""
    if pd.isna(value):
        return np.nan
    return 5 - value


def reverse_7pt(value: float) -> float:
    """Reverse code a 7-point scale item: 8 - original."""
    if pd.isna(value):
        return np.nan
    return 8 - value


# =====
# Trait calculation functions
# =====
def calculate_state_anxiety(responses: dict) -> float:
    """
    Calculate state anxiety score from q1-q6.

    Positive mood (reverse coded): q1 (calm), q2 (relaxed), q3 (content)
    Negative mood (direct): q4 (tense), q5 (upset), q6 (worried)

    Returns mean after reverse coding (range 1-4).
    """
    q1 = reverse_4pt(encode_4pt(responses["q1"]))  # calm - reverse
    q2 = reverse_4pt(encode_4pt(responses["q2"]))  # relaxed - reverse
    q3 = reverse_4pt(encode_4pt(responses["q3"]))  # content - reverse
    q4 = encode_4pt(responses["q4"])  # tense - direct
    q5 = encode_4pt(responses["q5"])  # upset - direct
    q6 = encode_4pt(responses["q6"])  # worried - direct

    items = [q1, q2, q3, q4, q5, q6]
    return safe_mean(items)


# TIPI Big Five item mappings: (direct_q, reverse_q)
TIPI_ITEMS = {
    "extraversion": ("q7", "q12"),      # extraverted vs reserved
    "agreeableness": ("q13", "q8"),     # sympathetic vs critical
    "conscientiousness": ("q9", "q14"),  # dependable vs disorganized
    "neuroticism": ("q10", "q15"),      # anxious vs calm
    "openness": ("q11", "q16"),         # open vs conventional
}


def calculate_tipi_trait(responses: dict, direct_q: str, reverse_q: str) -> float:
    """Calculate a Big Five trait from one direct and one reverse-coded item."""
    direct = encode_7pt(responses[direct_q])
    reverse = reverse_7pt(encode_7pt(responses[reverse_q]))
    return safe_mean([direct, reverse])


def calculate_extraversion(responses: dict) -> float:
    """Calculate extraversion: q7 (direct), q12 (reverse)."""
    return calculate_tipi_trait(responses, "q7", "q12")


def calculate_agreeableness(responses: dict) -> float:
    """Calculate agreeableness: q13 (direct), q8 (reverse)."""
    return calculate_tipi_trait(responses, "q13", "q8")


def calculate_conscientiousness(responses: dict) -> float:
    """Calculate conscientiousness: q9 (direct), q14 (reverse)."""
    return calculate_tipi_trait(responses, "q9", "q14")


def calculate_neuroticism(responses: dict) -> float:
    """Calculate neuroticism: q10 (direct), q15 (reverse)."""
    return calculate_tipi_trait(responses, "q10", "q15")


def calculate_openness(responses: dict) -> float:
    """Calculate openness: q11 (direct), q16 (reverse)."""
    return calculate_tipi_trait(responses, "q11", "q16")


def calculate_impulsivity(responses: dict) -> float:
    """
    Calculate impulsivity score from BIS-style items q17-q24.

    Reverse (low impulsivity): q17, q20, q21, q22
    Direct (high impulsivity): q18, q19, q23, q24

    Returns mean of 8 items (range 1-7).
    """
    q17 = reverse_7pt(encode_7pt(responses["q17"]))  # plan carefully - reverse
    q18 = encode_7pt(responses["q18"])  # do things without thinking - direct
    q19 = encode_7pt(responses["q19"])  # don't pay attention - direct
    q20 = reverse_7pt(encode_7pt(responses["q20"]))  # self-controlled - reverse
    q21 = reverse_7pt(encode_7pt(responses["q21"]))  # concentrate easily - reverse
    q22 = reverse_7pt(encode_7pt(responses["q22"]))  # careful thinker - reverse
    q23 = encode_7pt(responses["q23"])  # say things without thinking - direct
    q24 = encode_7pt(responses["q24"])  # act on spur of moment - direct

    items = [q17, q18, q19, q20, q21, q22, q23, q24]
    return safe_mean(items)


def safe_mean(items: list) -> float:
    """Calculate mean of items, handling NaN values."""
    valid_items = [x for x in items if not pd.isna(x)]
    if not valid_items:
        return np.nan
    return np.mean(valid_items)


# Trait names used for summary and validation
TRAIT_NAMES = [
    "state_anxiety", "extraversion", "agreeableness",
    "conscientiousness", "neuroticism", "openness", "impulsivity"
]


# =====
# Output functions
# =====
def print_summary_statistics(df: pd.DataFrame, missing_count: int):
    """Print summary statistics for each trait."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total participants: {len(df)}")
    print(f"Participants with missing survey data: {missing_count}")
    print_trait_table(df)


def print_trait_table(df: pd.DataFrame):
    """Print formatted table of trait statistics."""
    print("\nTrait Statistics:")
    print(f"{'Trait':<20} {'Mean':>8} {'SD':>8} {'Min':>8} {'Max':>8}")
    print("-" * 52)

    for trait in TRAIT_NAMES:
        stats = df[trait].agg(["mean", "std", "min", "max"])
        print(f"{trait:<20} {stats['mean']:>8.3f} {stats['std']:>8.3f} "
              f"{stats['min']:>8.2f} {stats['max']:>8.2f}")


# Expected trait ranges for validation
TRAIT_RANGES = {
    "state_anxiety": (1, 4),
    "extraversion": (1, 7),
    "agreeableness": (1, 7),
    "conscientiousness": (1, 7),
    "neuroticism": (1, 7),
    "openness": (1, 7),
    "impulsivity": (1, 7),
}


def validate_dataset(df: pd.DataFrame):
    """Validate dataset integrity and trait value ranges."""
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)
    validate_participant_count(df)
    validate_trait_ranges(df)
    validate_no_missing_values(df)
    validate_session_count(df)


def validate_participant_count(df: pd.DataFrame):
    """Check that we have the expected number of participants."""
    expected = 96  # 6 sessions x 16 players
    actual = len(df)
    status = "PASS" if actual == expected else "WARN"
    print(f"[{status}] Participant count: {actual} (expected {expected})")


def validate_trait_ranges(df: pd.DataFrame):
    """Check that all trait values are within expected ranges."""
    for trait, (min_exp, max_exp) in TRAIT_RANGES.items():
        actual_min, actual_max = df[trait].min(), df[trait].max()
        in_range = actual_min >= min_exp and actual_max <= max_exp
        status = "PASS" if in_range else "FAIL"
        print(f"[{status}] {trait} range: {actual_min:.2f}-{actual_max:.2f} "
              f"(expected {min_exp}-{max_exp})")


def validate_no_missing_values(df: pd.DataFrame):
    """Check for missing values in the dataset."""
    missing_any = df.isna().any().any()
    status = "WARN" if missing_any else "PASS"
    print(f"[{status}] No missing trait values: {not missing_any}")


def validate_session_count(df: pd.DataFrame):
    """Check that all sessions are present."""
    n_sessions = df["session_id"].nunique()
    status = "PASS" if n_sessions == 6 else "WARN"
    print(f"[{status}] Unique sessions: {n_sessions} (expected 6)")


def save_dataset(df: pd.DataFrame):
    """Save dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


# %%
if __name__ == "__main__":
    df = main()
