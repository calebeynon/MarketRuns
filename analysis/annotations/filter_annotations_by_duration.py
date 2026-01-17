"""
Filter annotation CSV by removing rows with duration < 1 second.

When a row is removed, the preceding row's end time is extended to the 
removed row's end time. This handles consecutive short-duration rows by
cascading the end time forward through all deletions.

Usage:
    python filter_annotations_by_duration.py --input <path> --output <path>
"""

import argparse
import pandas as pd
from pathlib import Path


DURATION_THRESHOLD_MS = 1000  # 1 second in milliseconds


def load_annotations(path):
    """Load annotation CSV file."""
    df = pd.read_csv(path)
    
    required_cols = ['Respondent Name', 'Start Time (ms)', 'End Time (ms)']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def filter_by_duration(df, threshold_ms=DURATION_THRESHOLD_MS):
    """Remove rows with duration < threshold and extend previous row's end time.
    
    Args:
        df: DataFrame with annotation data
        threshold_ms: Minimum duration in milliseconds (default: 1000ms = 1 second)
    
    Returns:
        Filtered DataFrame with short-duration rows removed and preceding rows extended
    """
    df = df.copy()
    
    # Calculate duration
    df['duration_ms'] = df['End Time (ms)'] - df['Start Time (ms)']
    
    # Track which rows to keep
    keep_mask = df['duration_ms'] >= threshold_ms
    
    # Process each respondent separately to maintain boundaries
    result_dfs = []
    
    for respondent in df['Respondent Name'].unique():
        respondent_df = df[df['Respondent Name'] == respondent].copy()
        respondent_keep = keep_mask[df['Respondent Name'] == respondent]
        
        # Iterate through rows and extend end times when removing short rows
        i = 0
        while i < len(respondent_df):
            if respondent_keep.iloc[i]:
                # Keep this row - check if any following rows should be removed
                current_end = respondent_df.iloc[i]['End Time (ms)']
                j = i + 1
                
                # Find consecutive short-duration rows to remove
                while j < len(respondent_df) and not respondent_keep.iloc[j]:
                    current_end = respondent_df.iloc[j]['End Time (ms)']
                    j += 1
                
                # If we removed any rows, extend this row's end time
                if j > i + 1:
                    respondent_df.iloc[i, respondent_df.columns.get_loc('End Time (ms)')] = current_end
                
                i = j
            else:
                i += 1
        
        # Filter to only kept rows
        result_dfs.append(respondent_df[respondent_keep])
    
    # Combine all respondents
    result = pd.concat(result_dfs, ignore_index=True)
    
    # Drop temporary duration column
    result = result.drop(columns=['duration_ms'])
    
    return result


def write_annotations(df, path):
    """Write annotation CSV file."""
    df.to_csv(path, index=False)
    print(f"\nOutput written to: {path}")
    print(f"Total rows: {len(df)}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter annotations by removing rows with duration < 1 second."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input annotation CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output annotation CSV file"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DURATION_THRESHOLD_MS,
        help=f"Minimum duration threshold in milliseconds (default: {DURATION_THRESHOLD_MS})"
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    threshold_ms = args.threshold
    
    print(f"Loading annotations from: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Duration threshold: {threshold_ms}ms ({threshold_ms/1000:.1f} seconds)\n")
    
    df = load_annotations(input_path)
    print(f"Loaded {len(df)} rows")
    
    # Count rows per respondent before filtering
    respondent_counts_before = df['Respondent Name'].value_counts().to_dict()
    
    filtered_df = filter_by_duration(df, threshold_ms)
    print(f"After filtering: {len(filtered_df)} rows")
    print(f"Removed: {len(df) - len(filtered_df)} rows\n")
    
    # Show summary by respondent
    respondent_counts_after = filtered_df['Respondent Name'].value_counts().to_dict()
    
    print("Summary by respondent:")
    for respondent in sorted(respondent_counts_before.keys()):
        before = respondent_counts_before.get(respondent, 0)
        after = respondent_counts_after.get(respondent, 0)
        removed = before - after
        print(f"  {respondent}: {before} → {after} (removed {removed})")
    
    write_annotations(filtered_df, output_path)


if __name__ == "__main__":
    main()
