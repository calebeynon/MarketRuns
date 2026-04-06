"""
Purpose: Build pre-sell emotion dataset from iMotions facial expression data
Author: Claude Code
Date: 2026-04-05

For each sell event with a recorded click time, extracts emotion data from the
2000ms window before the sell click. Maps oTree sell_click_time to iMotions
relative timestamps using per-participant recording offsets from edited_data CSVs.

OUTPUT VARIABLES:
    session_id, segment, round, period, group_id, player, treatment, signal,
    state, price, sold, already_sold, prior_group_sales: From individual_period_dataset
    extraversion, agreeableness, conscientiousness, neuroticism, openness,
    impulsivity, state_anxiety, risk_tolerance, age, gender: From survey_traits
    sell_click_time: Unix epoch of sell click (seconds)
    window_start_ms, window_end_ms: iMotions-relative pre-sell window bounds (ms)
    n_presell_frames: Number of iMotions frames in the pre-sell window
    anger_mean, contempt_mean, disgust_mean, fear_mean, joy_mean,
    sadness_mean, surprise_mean, engagement_mean, valence_mean: Window-averaged
    global_group_id: Unique group identifier ({session_id}_{segment}_{group_id})
"""

import re
import pandas as pd
from pathlib import Path

# =====
# File paths and constants
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
IMOTIONS_DIR = DATASTORE / "imotions"
EDITED_DATA_DIR = DATASTORE / "annotations" / "edited_data"
DERIVED = DATASTORE / "derived"
INPUT_PERIOD = DERIVED / "individual_period_dataset.csv"
INPUT_TRAITS = DERIVED / "survey_traits.csv"
OUTPUT_PATH = DERIVED / "presell_emotions_traits_dataset.csv"

IMOTIONS_SESSION_MAP = {
    "1": "1_11-7-tr1",
    "2": "2_11-10-tr2",
    "3": "3_11-11-tr2",
    "4": "4_11-12-tr1",
    "5": "5_11-14-tr2",
    "6": "6_11-18-tr1",
}

SEGMENTS = {1: "chat_noavg", 2: "chat_noavg2", 3: "chat_noavg3", 4: "chat_noavg4"}

EMOTION_COLS = [
    "Anger", "Contempt", "Disgust", "Fear", "Joy",
    "Sadness", "Surprise", "Engagement", "Valence",
]

PARTICIPANT_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H",
    "J", "K", "L", "M", "N", "P", "Q", "R",
]

IMOTIONS_SKIP_ROWS = 24
PRESELL_WINDOW_MS = 2000
EXCEL_EPOCH_1970 = 25569.0
CST_OFFSET_HOURS = 6


# =====
# Main function
# =====
def main():
    """Build the pre-sell emotions dataset."""
    all_records = []

    print("Extracting pre-sell emotions...")
    for imotions_session, session_id in IMOTIONS_SESSION_MAP.items():
        print(f"  Session {imotions_session} ({session_id})")
        records = process_session(imotions_session, session_id)
        all_records.extend(records)
        print(f"    -> {len(records)} sell events with click time")

    presell_df = pd.DataFrame(all_records)
    print(f"\nTotal pre-sell records: {len(presell_df)}")

    merged = merge_all(presell_df)
    merged = add_global_group_id(merged)
    print_summary(merged)
    save_dataset(merged)

    return merged


# =====
# Session processing
# =====
def process_session(imotions_session: str, session_id: str) -> list[dict]:
    """Process all segments for one session."""
    recording_starts = load_recording_starts(imotions_session)
    imotions_cache = load_all_imotions(imotions_session)
    records = []

    for segment_num, segment_name in SEGMENTS.items():
        sell_events = load_sell_events(session_id, segment_name)
        for _, row in sell_events.iterrows():
            record = build_presell_record(
                row, session_id, segment_num,
                recording_starts, imotions_cache,
            )
            if record is not None:
                records.append(record)

    return records


def load_recording_starts(imotions_session: str) -> dict[str, float]:
    """Load recording start epoch for each participant from edited_data."""
    ed_path = EDITED_DATA_DIR / f"e{imotions_session}.csv"
    ed = pd.read_csv(ed_path)
    starts = {}

    for pid in ed["participant_id_in_session"].unique():
        recording_serial = ed.loc[
            ed["participant_id_in_session"] == pid, "RECORDING"
        ].iloc[0]
        starts[pid_to_label(int(pid))] = excel_to_epoch(recording_serial)

    return starts


def load_all_imotions(imotions_session: str) -> dict[str, pd.DataFrame]:
    """Load all iMotions CSVs for a session, keyed by player label."""
    session_dir = IMOTIONS_DIR / imotions_session
    cache = {}

    for csv_file in sorted(session_dir.glob("*.csv")):
        if csv_file.name == "ExportMerge.csv":
            continue
        label = extract_player_label(csv_file.name)
        if label is not None:
            cache[label] = load_imotions_csv(csv_file)

    return cache


# =====
# Sell event extraction
# =====
def load_sell_events(session_id: str, segment_name: str) -> pd.DataFrame:
    """Load oTree CSV and return rows with valid sell clicks."""
    session_dir = DATASTORE / session_id
    csv_files = list(session_dir.glob(f"{segment_name}_*.csv"))
    if not csv_files:
        return pd.DataFrame()

    df = pd.read_csv(csv_files[0])
    mask = (df["player.sold"] == 1) & (df["player.sell_click_time"].notna())
    return df[mask]


def build_presell_record(
    row: pd.Series, session_id: str, segment: int,
    recording_starts: dict, imotions_cache: dict,
) -> dict | None:
    """Build one pre-sell emotion record from a sell event row."""
    player = row["participant.label"]
    if player not in recording_starts or player not in imotions_cache:
        return None

    sell_click = row["player.sell_click_time"]
    window_start, window_end = compute_presell_window(
        sell_click, recording_starts[player],
    )
    emotions = extract_window_emotions(
        imotions_cache[player], window_start, window_end,
    )
    return assemble_record(
        session_id, segment, row, player, sell_click,
        window_start, window_end, emotions,
    )


def compute_presell_window(sell_click: float, rec_start: float) -> tuple:
    """Compute the iMotions-relative pre-sell window bounds."""
    click_ms = (sell_click - rec_start) * 1000
    return click_ms - PRESELL_WINDOW_MS, click_ms


def assemble_record(
    session_id, segment, row, player, sell_click,
    window_start, window_end, emotions,
) -> dict:
    """Assemble the output record dict for one sell event."""
    return {
        "session_id": session_id,
        "segment": segment,
        "round": int(row["player.round_number_in_segment"]),
        "period": int(row["player.period_in_round"]),
        "player": player,
        "sell_click_time": sell_click,
        "window_start_ms": window_start,
        "window_end_ms": window_end,
        **emotions,
    }


# =====
# iMotions data loading and emotion extraction
# =====
def load_imotions_csv(filepath: Path) -> pd.DataFrame:
    """Load an iMotions CSV, returning Timestamp and emotion columns."""
    df = pd.read_csv(
        filepath,
        skiprows=IMOTIONS_SKIP_ROWS,
        encoding="utf-8-sig",
        low_memory=False,
        usecols=["Timestamp"] + EMOTION_COLS,
    )
    return df


def extract_window_emotions(
    imotions_df: pd.DataFrame, start_ms: float, end_ms: float,
) -> dict:
    """Average emotions in the [start_ms, end_ms] window."""
    mask = (imotions_df["Timestamp"] >= start_ms) & (imotions_df["Timestamp"] <= end_ms)
    window = imotions_df[mask]

    result = {"n_presell_frames": len(window)}
    for col in EMOTION_COLS:
        if len(window) == 0:
            result[f"{col.lower()}_mean"] = float("nan")
        else:
            result[f"{col.lower()}_mean"] = pd.to_numeric(
                window[col], errors="coerce"
            ).mean()

    return result


def extract_player_label(filename: str) -> str | None:
    """Extract participant letter from filename like 001_R3.csv."""
    match = re.match(r"\d+_([A-Z])\d+\.csv", filename)
    return match.group(1) if match else None


# =====
# Conversion helpers
# =====
def pid_to_label(pid: int) -> str:
    """Convert participant_id_in_session (1-16) to letter label."""
    return PARTICIPANT_LABELS[pid - 1]


def excel_to_epoch(excel_serial: float) -> float:
    """Convert Excel serial date (CST) to Unix epoch seconds."""
    return (excel_serial - EXCEL_EPOCH_1970 + CST_OFFSET_HOURS / 24) * 86400


# =====
# Merge with period data and traits
# =====
TRAIT_COLS = [
    "session_id", "player", "extraversion", "agreeableness",
    "conscientiousness", "neuroticism", "openness",
    "impulsivity", "state_anxiety", "risk_tolerance", "age", "gender",
]

MERGE_KEYS = ["session_id", "segment", "round", "period", "player"]


def merge_all(presell_df: pd.DataFrame) -> pd.DataFrame:
    """Merge pre-sell emotions with period data and survey traits."""
    period_df = pd.read_csv(INPUT_PERIOD)
    traits_df = pd.read_csv(INPUT_TRAITS)

    # Filter period data to sell events only for merge
    merged = period_df.merge(presell_df, on=MERGE_KEYS, how="inner")
    merged = merged.merge(
        traits_df[TRAIT_COLS], on=["session_id", "player"], how="left",
    )

    print(f"Merged: {len(merged)} rows (period data x presell emotions)")
    return merged


def add_global_group_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add unique group identifier across sessions and segments."""
    df["global_group_id"] = (
        df["session_id"].astype(str) + "_"
        + df["segment"].astype(str) + "_"
        + df["group_id"].astype(str)
    )
    return df


# =====
# Output
# =====
def print_summary(df: pd.DataFrame):
    """Print summary statistics for the pre-sell emotions dataset."""
    print("\n" + "=" * 50)
    print("PRESELL EMOTIONS SUMMARY")
    print("=" * 50)
    print(f"Total sell events: {len(df)}")
    print(f"Sessions: {df['session_id'].nunique()}")
    print(f"Unique players: {df['player'].nunique()}")
    print(f"Segments: {sorted(df['segment'].unique())}")

    zero_frames = (df["n_presell_frames"] == 0).sum()
    print(f"\nn_presell_frames: {df['n_presell_frames'].min()} - "
          f"{df['n_presell_frames'].max()} "
          f"(mean={df['n_presell_frames'].mean():.1f})")
    print(f"Events with 0 frames in window: {zero_frames}")
    print_emotion_ranges(df)


def print_emotion_ranges(df: pd.DataFrame):
    """Print min/max/mean for each emotion column."""
    emotion_cols = [f"{c.lower()}_mean" for c in EMOTION_COLS]
    print("\nEmotion score ranges:")
    for col in emotion_cols:
        non_null = df[col].dropna()
        if len(non_null) > 0:
            print(f"  {col}: {non_null.min():.3f} - {non_null.max():.3f} "
                  f"(mean={non_null.mean():.3f})")


def save_dataset(df: pd.DataFrame):
    """Save dataset to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


# %%
if __name__ == "__main__":
    main()
