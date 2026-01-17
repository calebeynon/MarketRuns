"""
Generate annotation CSV from edited_data_output.csv (unfiltered version)

Transforms page event data into annotation format with:
- Respondent name mapping (participant_id_in_session -> A1, B1, etc.)
- Marker names with occurrence counters within supergames
- Start/end times calculated from successive page timestamps
- NO duration filtering (includes all events regardless of duration)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# Global constants
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_INPUT = str((SCRIPT_DIR / "edited_data_output.csv").resolve())
DEFAULT_OUTPUT = str((SCRIPT_DIR / "annotation_generated_unfiltered.csv").resolve())
RESPONDENT_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R"]
SEG_ROUND_NUMS = [3,4,3,7,5]


def load_data(path):
    """Load CSV data with validation."""
    df = pd.read_csv(path)
    required_cols = ['participant_id_in_session', 'app_name', 'page_name']
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def detect_time_column(df):
    """Detect the rightmost numeric column as timestamp column."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        raise ValueError("No numeric columns found for timestamp")
    
    time_col = numeric_cols[-1]
    print(f"Detected time column: '{time_col}'")
    
    return time_col


def extract_session_number(filename):
    """Extract session number from filename.
    
    Looks for patterns like 'e3.csv', 'e10.csv' in the filename.
    Returns session number (1-11) or 1 as default.
    """
    import re
    
    # Try various patterns
    patterns = [
        r'e(\d+)',              # e3, e10
        r'session[_-]?(\d+)',  # session_3, session-3, session3
        r's(\d+)',              # s3
        r'_(\d+)_',             # _3_
        r'_(\d+)\.',            # _3.
    ]
    
    filename_lower = filename.lower()
    
    for pattern in patterns:
        match = re.search(pattern, filename_lower)
        if match:
            session_num = int(match.group(1))
            if 1 <= session_num <= 11:
                return session_num
    
    # Default to 1 if no session number found
    return 1


def id_to_respondent(pid, session_num):
    """Map participant_id_in_session to respondent name (A1, B1, ..., R1).
    
    Args:
        pid: participant_id_in_session (1-16)
        session_num: session number (1-11)
    """
    if pid < 1 or pid > 16:
        raise ValueError(f"participant_id_in_session must be 1-16, got {pid}")
    
    return RESPONDENT_LABELS[pid - 1] + str(session_num)


def compute_durations(df, time_col):
    """Compute start/end times and durations for each page event.
    
    Timestamps in the input data represent page completion times.
    For each event/page:
    - start_time_ms = previous page's completion time (within same participant)
    - end_time_ms = current page's completion time
    - First event per participant has start_time_ms == end_time_ms (zero duration)
    """
    # Ensure proper sorting by participant and timestamp
    df = df.sort_values(['participant_id_in_session', time_col]).reset_index(drop=True)
    
    # Convert timestamp to integer milliseconds
    df[time_col] = df[time_col].astype(int)
    
    # FIXED: start_time is previous row's timestamp, end_time is current row's timestamp
    df['start_time_ms'] = df.groupby('participant_id_in_session')[time_col].shift(1)
    df['end_time_ms'] = df[time_col]
    
    # For first row per participant, set start_time equal to end_time (zero duration)
    df['start_time_ms'] = df['start_time_ms'].fillna(df['end_time_ms']).astype(int)
    
    # Compute duration
    df['duration_ms'] = df['end_time_ms'] - df['start_time_ms']
    
    # Sanity checks
    assert (df['duration_ms'] >= 0).all(), "Found negative durations"
    
    return df


def extract_supergame_number(app_name):
    """Extract supergame number from app_name.
    
    Args:
        app_name: app name like 'supergame1', 'supergame2', etc.
    
    Returns:
        supergame number (1-4) or None if not a supergame
    """
    import re
    if pd.isna(app_name):
        return None
    
    match = re.search(r'supergame(\d+)', str(app_name).lower())
    if match:
        return int(match.group(1))
    return None


def build_marker_names(df):
    """Build marker names with supergame and round numbering.
    
    Args:
        df: DataFrame with page event data
    
    Returns:
        DataFrame with marker_name column with formats:
        - introduction/finalresults pages: just page_name
        - StartPage/RegroupingMessage: s{N}page_name (supergame only)
        - other supergame pages: s{N}r{R}page_name (supergame and round)
    """
    df = df.copy()
    
    # Extract supergame number from app_name
    df['supergame_num'] = df['app_name'].apply(extract_supergame_number)
    
    # Track round number within each participant's supergame
    # Round increments when we see 'RoundWaitPage'
    df['round_num'] = 0
    
    for participant_id in df['participant_id_in_session'].unique():
        participant_mask = df['participant_id_in_session'] == participant_id
        participant_df = df[participant_mask]
        
        round_counter = 0  # Start at 0, will increment to 1 on first round
        round_nums = []
        
        for idx, row in participant_df.iterrows():
            # Check if this is a supergame app
            if pd.notna(row['supergame_num']):
                # StartPage resets round counter but doesn't get round label
                if row['page_name'] == 'StartPage':
                    round_counter = 0
                    round_nums.append(0)  # StartPage doesn't get round label
                # RoundWaitPage indicates start of new round
                elif row['page_name'] == 'RoundWaitPage':
                    round_counter += 1
                    round_nums.append(round_counter)
                # RegroupingMessage doesn't get round label
                elif row['page_name'] == 'RegroupingMessage':
                    round_nums.append(0)
                # Regular pages get current round number (or increment if first page after StartPage)
                else:
                    # If round_counter is 0, this is the first round page, set to 1
                    if round_counter == 0:
                        round_counter = 1
                    round_nums.append(round_counter)
            else:
                # Non-supergame apps (introduction, finalresults)
                round_nums.append(0)
        
        df.loc[participant_mask, 'round_num'] = round_nums
    
    # Build marker names based on app type and page type
    def create_marker_name(row):
        page_name = row['page_name']
        supergame = row['supergame_num']
        round_num = row['round_num']
        
        # Introduction or finalresults - no prefix
        if pd.isna(supergame):
            return page_name
        
        # StartPage or RegroupingMessage - supergame only
        if page_name in ['StartPage', 'RegroupingMessage'] or round_num == 0:
            return f"s{int(supergame)}{page_name}"
        
        # Regular supergame pages - supergame and round
        return f"s{int(supergame)}r{int(round_num)}{page_name}"
    
    df['marker_name'] = df.apply(create_marker_name, axis=1)
    
    # Clean up temporary columns
    df = df.drop(columns=['supergame_num', 'round_num'])
    
    return df


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate annotation CSV from edited data output WITHOUT duration filtering."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Full path to input CSV file (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Full path to output CSV file (default: {DEFAULT_OUTPUT})"
    )
    return parser.parse_args()


def assemble_output(df):
    """Create output DataFrame with required schema."""
    output = pd.DataFrame({
        'Respondent Name': df['respondent_name'],
        'Stimulus Name': '',
        'Marker Type': 'Respondent Annotation',
        'Marker Name': df['marker_name'],
        'Start Time (ms)': df['start_time_ms'],
        'End Time (ms)': df['end_time_ms'],
        'Comment': '',
        'Color': ''
    })
    
    output = output.sort_values(['Respondent Name', 'Start Time (ms)']).reset_index(drop=True)
    
    return output


def write_output_csv(df, path):
    """Write output CSV file."""
    df.to_csv(path, index=False)
    print(f"\nOutput written to: {path}")
    print(f"Total rows: {len(df)}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Extract session number from input filename
    session_num = extract_session_number(input_path.name)
    
    print(f"Loading data from: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Detected session number: {session_num}")
    print("NOTE: This script does NOT filter by duration threshold\n")
    
    df = load_data(input_path)
    print(f"Loaded {len(df)} rows")
    
    time_col = detect_time_column(df)
    
    df = df.dropna(subset=['participant_id_in_session', 'app_name', 'page_name', time_col])
    print(f"After dropping nulls: {len(df)} rows")
    
    df['respondent_name'] = df['participant_id_in_session'].apply(lambda pid: id_to_respondent(pid, session_num))
    
    df = compute_durations(df, time_col)
    
    df = build_marker_names(df)
    
    output = assemble_output(df)
    
    print("\nSample output (first 5 rows):")
    print(output.head())
    
    write_output_csv(output, output_path)


if __name__ == "__main__":
    main()
