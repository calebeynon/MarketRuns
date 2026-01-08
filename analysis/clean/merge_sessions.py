"""
Purpose: Verify and merge all_apps_wide CSV files from six experimental sessions
Author: Caleb Eynon w/ Warp
Date: 2026-01-08
"""

import pandas as pd
import sys
from pathlib import Path
import re

# Global variables
BASE_DIR = Path(__file__).parent.parent.parent / "datastore"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "datastore"
SESSION_FOLDERS = [
    "1_11-7-tr1",
    "2_11-10-tr2", 
    "3_11-11-tr2",
    "4_11-12-tr1",
    "5_11-14-tr2",
    "6_11-18-tr1"
]


def extract_treatment(folder_name):
    """Extract treatment number from folder name."""
    match = re.search(r'tr(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None


def find_csv_file(folder_path):
    """Find the all_apps_wide CSV file in the folder."""
    csv_files = list(folder_path.glob("all_apps_wide_*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No all_apps_wide CSV file found in {folder_path}")
    if len(csv_files) > 1:
        raise ValueError(f"Multiple all_apps_wide CSV files found in {folder_path}")
    return csv_files[0]


def verify_column_consistency(dataframes):
    """Verify that all dataframes have the same columns."""
    if not dataframes:
        raise ValueError("No dataframes to verify")
    
    base_columns = set(dataframes[0].columns)
    
    for i, df in enumerate(dataframes[1:], start=1):
        current_columns = set(df.columns)
        if current_columns != base_columns:
            missing = base_columns - current_columns
            extra = current_columns - base_columns
            error_msg = f"Column mismatch in dataframe {i}:\n"
            if missing:
                error_msg += f"  Missing columns: {missing}\n"
            if extra:
                error_msg += f"  Extra columns: {extra}\n"
            raise ValueError(error_msg)
    
    print("✓ All CSV files have identical columns")
    return True


def check_merge_conflicts(dataframes, folder_names):
    """Check for potential merge conflicts."""
    all_participant_codes = []
    
    for i, (df, folder) in enumerate(zip(dataframes, folder_names)):
        codes = df['participant.code'].tolist()
        all_participant_codes.extend([(code, folder) for code in codes])
    
    code_counts = {}
    for code, folder in all_participant_codes:
        if code not in code_counts:
            code_counts[code] = []
        code_counts[code].append(folder)
    
    conflicts = {code: folders for code, folders in code_counts.items() 
                 if len(folders) > 1}
    
    if conflicts:
        print("⚠ WARNING: Duplicate participant codes found:")
        for code, folders in conflicts.items():
            print(f"  {code}: appears in {folders}")
        return False
    
    print("✓ No duplicate participant codes found")
    return True


def load_and_prepare_data():
    """Load all CSV files and add treatment column."""
    dataframes = []
    folder_names = []
    
    for folder_name in SESSION_FOLDERS:
        folder_path = BASE_DIR / folder_name
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        csv_file = find_csv_file(folder_path)
        df = pd.read_csv(csv_file)
        
        treatment = extract_treatment(folder_name)
        df.insert(0, 'treatment', treatment)
        
        dataframes.append(df)
        folder_names.append(folder_name)
        
        print(f"Loaded {csv_file.name}: {len(df)} rows, treatment {treatment}")
    
    return dataframes, folder_names


def merge_data(dataframes):
    """Merge all dataframes."""
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"\n✓ Merged {len(dataframes)} files into {len(merged_df)} total rows")
    return merged_df


def save_merged_data(merged_df, output_path):
    """Save the merged dataframe to CSV."""
    merged_df.to_csv(output_path, index=False)
    print(f"✓ Saved merged data to {output_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Merging all_apps_wide CSV files")
    print("=" * 60)
    print()
    
    try:
        print("Step 1: Loading CSV files...")
        dataframes, folder_names = load_and_prepare_data()
        print()
        
        print("Step 2: Verifying column consistency...")
        verify_column_consistency(dataframes)
        print()
        
        print("Step 3: Checking for merge conflicts...")
        no_conflicts = check_merge_conflicts(dataframes, folder_names)
        print()
        
        if not no_conflicts:
            response = input("Continue with merge despite conflicts? (y/n): ")
            if response.lower() != 'y':
                print("Merge cancelled by user")
                sys.exit(1)
        
        print("Step 4: Merging data...")
        merged_df = merge_data(dataframes)
        print()
        
        output_path = OUTPUT_DIR / "all_apps_wide_merged.csv"
        print("Step 5: Saving merged data...")
        save_merged_data(merged_df, output_path)
        print()
        
        print("=" * 60)
        print("Merge completed successfully!")
        print(f"Total rows: {len(merged_df)}")
        print(f"Treatment 1: {len(merged_df[merged_df['treatment'] == 1])} rows")
        print(f"Treatment 2: {len(merged_df[merged_df['treatment'] == 2])} rows")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
