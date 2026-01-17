"""
Transform PageTimes CSV to edited data format.

This script processes PageTimes CSV data by:
1. Filtering to most recent session
2. Trimming to last InitializeParticipant per participant
3. Reordering from interleaved to sequential by participant
4. Adding derived time columns (date time, gmt, LOCAL, RECORDING, and unnamed L/M/N)
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dateutil import parser as date_parser

# GLOBAL CONFIGURATION
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_PAGETIMES = PROJECT_ROOT / "datastore/annotations/pagetimes/p1.csv"
DEFAULT_TIMESHEET = PROJECT_ROOT / "datastore/annotations/timesheets/t1.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "datastore/annotations/edited_data/e1.csv"
TZ_NAME = "America/Chicago"
CST_OFFSET_HOURS = 6
LETTER_TO_ID = {
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
    "J": 9, "K": 10, "L": 11, "M": 12, "N": 13, "P": 14, "Q": 15, "R": 16
}
EXCEL_EPOCH_1970 = 25569.0  # Excel serial date for 1970-01-01


def excel_serial_from_epoch_seconds(epoch_seconds):
    """Convert Unix epoch seconds to Excel serial date."""
    return epoch_seconds / 86400.0 + EXCEL_EPOCH_1970


def excel_serial_from_naive_datetime(dt):
    """Convert naive datetime to Excel serial date."""
    excel_epoch = datetime(1899, 12, 30)
    delta = dt - excel_epoch
    return delta.days + (delta.seconds + delta.microseconds / 1e6) / 86400.0


def clean_epoch_units(value):
    """Convert epoch time to seconds, auto-detecting milliseconds."""
    if value >= 1e12:
        return value / 1000.0
    return float(value)


def parse_timesheet_start_time(raw_str, tz_name):
    """Parse timesheet start time string to Excel serial date."""
    # Remove unicode control characters
    cleaned = re.sub(r'[\u200e\u200f\u202a-\u202e]', '', str(raw_str))
    cleaned = cleaned.strip()
    
    try:
        dt = date_parser.parse(cleaned, fuzzy=True)
        # Assume parsed time is in Central Time (naive)
        return excel_serial_from_naive_datetime(dt)
    except Exception as e:
        print(f"Warning: Could not parse timesheet time '{raw_str}': {e}")
        return None


def load_pagetimes(path):
    """Load PageTimes CSV with validation."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Could not find file: {path}")
        sys.exit(1)
    
    required_cols = [
        "session_code", "participant_id_in_session", "participant_code",
        "page_index", "app_name", "page_name", "epoch_time_completed"
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        print(f"Error: Missing required columns: {missing}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} rows from PageTimes CSV")
    return df


def load_timesheet(path):
    """Load timesheet Excel and extract participant recording start times."""
    try:
        df = pd.read_excel(path, header=None, engine="openpyxl")
    except FileNotFoundError:
        print(f"Error: Could not find file: {path}")
        sys.exit(1)
    
    recording_map = {}
    
    # Extract columns C (index 2) and D (index 3)
    for idx, row in df.iterrows():
        if idx < 2:  # Skip header rows
            continue
        
        participant_letter = str(row.get(2, "")).strip().upper()
        start_time_str = row.get(3)
        
        if participant_letter in LETTER_TO_ID and start_time_str:
            participant_id = LETTER_TO_ID[participant_letter]
            excel_serial = parse_timesheet_start_time(start_time_str, TZ_NAME)
            if excel_serial is not None:
                recording_map[participant_id] = excel_serial
                print(f"  Participant {participant_letter} (ID {participant_id}): "
                      f"start = {excel_serial:.6f}")
    
    print(f"Loaded {len(recording_map)} participant recording start times")
    return recording_map


def filter_most_recent_session(df):
    """Filter dataframe to only the most recent session."""
    last_code = df.loc[df["session_code"].notna(), "session_code"].iloc[-1]
    filtered = df[df["session_code"] == last_code].copy()
    print(f"Filtered to most recent session: {last_code} ({len(filtered)} rows)")
    return filtered, last_code


def trim_to_last_initialize(df):
    """Trim each participant's data to start from their last InitializeParticipant."""
    kept_frames = []
    
    for participant_id in sorted(df["participant_id_in_session"].unique()):
        participant_df = df[df["participant_id_in_session"] == participant_id]
        
        init_rows = participant_df[participant_df["page_name"] == "InitializeParticipant"]
        if len(init_rows) == 0:
            print(f"Warning: No InitializeParticipant found for participant {participant_id}, skipping")
            continue
        
        last_init_idx = init_rows.index[-1]
        kept = participant_df[participant_df.index >= last_init_idx]
        kept_frames.append(kept)
        print(f"  Participant {participant_id}: kept {len(kept)} rows from index {last_init_idx}")
    
    result = pd.concat(kept_frames, ignore_index=False)
    print(f"After trimming: {len(result)} rows total")
    return result


def reorder_sequential(df):
    """Reorder from interleaved to sequential by participant."""
    df = df.copy()
    df["_orig_index"] = df.index
    df["_order"] = df.groupby("participant_id_in_session")["_orig_index"].rank(method="first")
    df = df.sort_values(["participant_id_in_session", "_order"])
    df = df.drop(columns=["_orig_index", "_order"])
    print(f"Reordered data sequentially by participant")
    return df


def add_time_columns(df, recording_map):
    """Add derived time columns H through N."""
    df = df.copy()
    
    # Normalize epoch times
    df["_epoch_seconds"] = df["epoch_time_completed"].apply(clean_epoch_units)
    
    # H: date time
    df["date time"] = df["_epoch_seconds"].apply(excel_serial_from_epoch_seconds)
    
    # I: gmt (same as H)
    df["gmt"] = df["date time"]
    
    # J: LOCAL (CST offset)
    df["LOCAL"] = df["date time"] - (CST_OFFSET_HOURS / 24.0)
    
    # K: RECORDING (from timesheet)
    df["RECORDING"] = df["participant_id_in_session"].map(recording_map)
    
    # L: Time since recording started (LOCAL - RECORDING)
    def compute_col_l(row):
        if row["page_name"] == "InitializeParticipant":
            return ""
        if pd.isna(row["RECORDING"]):
            return ""
        return row["LOCAL"] - row["RECORDING"]
    
    df["col_L"] = df.apply(compute_col_l, axis=1)
    
    # M: Total seconds from column L
    def compute_col_m(row):
        if row["col_L"] == "":
            return ""
        l_value = row["col_L"]
        total_seconds = l_value * 86400.0
        minutes = int(total_seconds // 60)
        seconds = int(round(total_seconds % 60))
        return minutes * 60 + seconds
    
    df["col_M"] = df.apply(compute_col_m, axis=1)
    
    # N: Milliseconds (M * 1000)
    def compute_col_n(row):
        if row["col_M"] == "":
            return ""
        return int(row["col_M"]) * 1000
    
    df["col_N"] = df.apply(compute_col_n, axis=1)
    
    df = df.drop(columns=["_epoch_seconds"])
    print(f"Added derived time columns (H-N)")
    return df


def build_output_frame(df):
    """Assemble final output with proper column names."""
    output_cols = [
        "session_code",
        "participant_id_in_session",
        "participant_code",
        "page_index",
        "app_name",
        "page_name",
        "epoch_time_completed",
        "date time",
        "gmt",
        "LOCAL",
        "RECORDING",
        "col_L",
        "col_M",
        "col_N"
    ]
    
    result = df[output_cols].copy()
    
    # Rename columns to match expected output (last three are unnamed)
    result.columns = [
        "session_code",
        "participant_id_in_session",
        "participant_code",
        "page_index",
        "app_name",
        "page_name",
        "epoch_time_completed",
        "date time",
        "gmt",
        "LOCAL",
        "RECORDING",
        "",  # Unnamed column L
        "",  # Unnamed column M
        ""   # Unnamed column N
    ]
    
    return result


def write_output_csv(df, path, session_code):
    """Write output CSV with validation logging."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(path, index=False, encoding="utf-8")
    
    # Log statistics
    print(f"\n{'='*60}")
    print(f"Output written to: {path}")
    print(f"Total rows: {len(df)}")
    print(f"Session: {session_code}")
    print(f"Unique participants: {df['participant_id_in_session'].nunique()}")
    init_count = (df['page_name'] == 'InitializeParticipant').sum()
    print(f"InitializeParticipant rows: {init_count}")
    
    # Show sample data for first 3 participants
    print(f"\n{'='*60}")
    print("Sample output (first 3 rows per participant):")
    for pid in sorted(df['participant_id_in_session'].unique())[:3]:
        participant_rows = df[df['participant_id_in_session'] == pid].head(3)
        print(f"\nParticipant {pid}:")
        for _, row in participant_rows.iterrows():
            l_val = row.iloc[11] if row.iloc[11] != "" else "blank"
            m_val = row.iloc[12] if row.iloc[12] != "" else "blank"
            n_val = row.iloc[13] if row.iloc[13] != "" else "blank"
            print(f"  {row['page_name']:25s} L={str(l_val)[:12]:12s} "
                  f"M={str(m_val)[:8]:8s} N={str(n_val)[:10]:10s}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transform PageTimes CSV to edited data format with time calculations."
    )
    parser.add_argument(
        "--pagetimes",
        type=str,
        default=DEFAULT_PAGETIMES,
        help=f"Path to PageTimes CSV file (default: {DEFAULT_PAGETIMES})"
    )
    parser.add_argument(
        "--timesheet",
        type=str,
        default=DEFAULT_TIMESHEET,
        help=f"Path to timesheet Excel file (default: {DEFAULT_TIMESHEET})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Path for output CSV file (default: {DEFAULT_OUTPUT})"
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("Starting PageTimes to Edited Data transformation\n")
    print(f"Input PageTimes: {args.pagetimes}")
    print(f"Input Timesheet: {args.timesheet}")
    print(f"Output Path: {args.output}\n")
    
    # Load inputs
    df = load_pagetimes(args.pagetimes)
    recording_map = load_timesheet(args.timesheet)
    
    # Transform data
    df, session_code = filter_most_recent_session(df)
    df = trim_to_last_initialize(df)
    df = reorder_sequential(df)
    df = add_time_columns(df, recording_map)
    df = build_output_frame(df)
    
    # Write output
    write_output_csv(df, args.output, session_code)
    
    print(f"\n{'='*60}")
    print("Transformation complete!")


if __name__ == "__main__":
    main()
