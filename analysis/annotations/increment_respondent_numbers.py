import pandas as pd
from pathlib import Path
import re
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "datastore/annotations/annotations_filtered"


def increment_respondent_name(name):
    """
    Increment the number in a respondent name by 2.
    e.g., 'A1' -> 'A3', 'B4' -> 'B6'
    """
    match = re.match(r'([A-Za-z]+)(\d+)', str(name))
    if match:
        letter = match.group(1)
        number = int(match.group(2))
        return f"{letter}{number + 2}"
    return name


def process_csv_file(filepath):
    """
    Process a single CSV file, incrementing the 'Respondent Name' column.
    """
    df = pd.read_csv(filepath)
    
    if 'Respondent Name' in df.columns:
        df['Respondent Name'] = df['Respondent Name'].apply(increment_respondent_name)
        df.to_csv(filepath, index=False)
        print(f"Processed: {filepath.name}")
    else:
        print(f"Warning: 'Respondent Name' column not found in {filepath.name}")


def main():
    """
    Process a specific CSV file in the input directory.
    """
    parser = argparse.ArgumentParser(description='Increment respondent numbers in a CSV file')
    parser.add_argument('csv_file', type=str, help='Name of the CSV file to process (e.g., "file.csv")')
    
    args = parser.parse_args()
    
    csv_filepath = INPUT_DIR / args.csv_file
    
    if not csv_filepath.exists():
        print(f"Error: File '{csv_filepath}' not found in {INPUT_DIR}")
        return
    
    if not csv_filepath.suffix.lower() == '.csv':
        print(f"Error: '{args.csv_file}' is not a CSV file")
        return
    
    process_csv_file(csv_filepath)
    print("Done!")


if __name__ == "__main__":
    main()
