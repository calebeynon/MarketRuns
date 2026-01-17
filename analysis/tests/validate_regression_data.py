"""
Purpose: Validate the prepared regression data against market_data.py parser and raw CSVs.
Author: Claude
Date: 2026-01-15

Verifies that the generated regression dataset values match raw data from the market_data parser.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import market_data as md

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
REGRESSION_DATA = DATASTORE / "derived" / "selling_period_regression_data.csv"

# Session data files
SESSION_FILES = {
    "1_11-7-tr1": DATASTORE / "1_11-7-tr1" / "all_apps_wide_2025-11-07.csv",
    "2_11-10-tr2": DATASTORE / "2_11-10-tr2" / "all_apps_wide_2025-11-10.csv",
    "3_11-11-tr2": DATASTORE / "3_11-11-tr2" / "all_apps_wide_2025-11-11.csv",
    "4_11-12-tr1": DATASTORE / "4_11-12-tr1" / "all_apps_wide_2025-11-12.csv",
    "5_11-14-tr2": DATASTORE / "5_11-14-tr2" / "all_apps_wide_2025-11-14.csv",
    "6_11-18-tr1": DATASTORE / "6_11-18-tr1" / "all_apps_wide_2025-11-18.csv",
}

# Raw segment CSV files (for raw validation)
SESSION_DATES = {
    "1_11-7-tr1": "2025-11-07",
    "2_11-10-tr2": "2025-11-10",
    "3_11-11-tr2": "2025-11-11",
    "4_11-12-tr1": "2025-11-12",
    "5_11-14-tr2": "2025-11-14",
    "6_11-18-tr1": "2025-11-18",
}

# Segment name mapping
SEGMENT_MAP = {1: "chat_noavg", 2: "chat_noavg2", 3: "chat_noavg3", 4: "chat_noavg4"}


# =====
# Main function
# =====
def main():
    """Run validation of regression dataset."""
    print("Loading regression data...")
    data = pd.read_csv(REGRESSION_DATA)
    print(f"  Loaded {len(data)} rows")

    print("\n" + "=" * 60)
    print("VALIDATION 1: Against market_data.py parser")
    print("=" * 60)
    parser_mismatches = validate_against_parser(data)

    print("\n" + "=" * 60)
    print("VALIDATION 2: Against raw CSV files")
    print("=" * 60)
    raw_mismatches = validate_against_raw_csv(data)

    print("\n" + "=" * 60)
    print("VALIDATION 3: Sale timing variables logic")
    print("=" * 60)
    timing_mismatches = validate_sale_timing_vars(data)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = parser_mismatches + raw_mismatches + timing_mismatches
    if total == 0:
        print("ALL VALIDATIONS PASSED")
    else:
        print(f"TOTAL MISMATCHES: {total}")

    return total


# =====
# Validation against market_data.py parser
# =====
def validate_against_parser(data):
    """Validate key fields against market_data.py parser."""
    mismatches = {"signal": 0, "state": 0, "price": 0, "sold": 0}
    rows_checked = 0

    # Load all sessions
    experiments = {}
    for session_id, csv_path in SESSION_FILES.items():
        if csv_path.exists():
            print(f"  Loading {session_id}...")
            experiments[session_id] = md.parse_experiment(str(csv_path))

    # Sample 200 random rows for validation
    sample = data.sample(min(200, len(data)), random_state=42)

    for _, row in sample.iterrows():
        session_id = row["session_id"]
        segment_idx = int(row["segment"])
        round_num = int(row["round"])
        period_num = int(row["period"])
        player_label = row["player"]

        experiment = experiments.get(session_id)
        if experiment is None or len(experiment.sessions) == 0:
            continue

        session = experiment.sessions[0]
        segment_name = SEGMENT_MAP[segment_idx]

        ref = get_reference_data(session, segment_name, round_num, period_num, player_label)
        if ref is None:
            continue

        rows_checked += 1

        # Verify each field
        if not floats_match(row["signal"], ref["signal"]):
            mismatches["signal"] += 1
        if row["state"] != ref["state"]:
            mismatches["state"] += 1
        if not floats_match(row["price"], ref["price"]):
            mismatches["price"] += 1
        if row["sold"] != ref["sold_this_period"]:
            mismatches["sold"] += 1

    print_mismatch_results(mismatches, rows_checked, "parser")
    return sum(mismatches.values())


def get_reference_data(session, segment_name, round_num, period_num, player_label):
    """Get reference data for a specific player-period from the parser."""
    segment = session.get_segment(segment_name)
    if not segment:
        return None

    round_obj = segment.get_round(round_num)
    if not round_obj:
        return None

    period_obj = round_obj.get_period(period_num)
    if not period_obj:
        return None

    player = period_obj.get_player(player_label)
    if not player:
        return None

    return {
        "sold_this_period": 1 if player.sold_this_period else 0,
        "signal": player.signal,
        "state": player.state,
        "price": player.price,
    }


# =====
# Validation against raw CSV files
# =====
def validate_against_raw_csv(data):
    """Validate key fields against raw oTree CSV exports."""
    mismatches = {"signal": 0, "state": 0, "price": 0}
    rows_checked = 0

    # Sample 200 random rows
    sample = data.sample(min(200, len(data)), random_state=123)

    # Cache loaded CSVs
    csv_cache = {}

    for _, row in sample.iterrows():
        session_id = row["session_id"]
        segment_idx = int(row["segment"])
        round_num = int(row["round"])
        period_num = int(row["period"])
        player_label = row["player"]

        date_suffix = SESSION_DATES.get(session_id)
        if date_suffix is None:
            continue

        segment_name = SEGMENT_MAP[segment_idx]
        csv_path = DATASTORE / session_id / f"{segment_name}_{date_suffix}.csv"

        if not csv_path.exists():
            continue

        # Load and cache CSV
        cache_key = str(csv_path)
        if cache_key not in csv_cache:
            csv_cache[cache_key] = pd.read_csv(csv_path)
        raw_df = csv_cache[cache_key]

        # Find matching row using round_number_in_segment and period_in_round
        match = raw_df[
            (raw_df["participant.label"] == player_label) &
            (raw_df["player.round_number_in_segment"] == round_num) &
            (raw_df["player.period_in_round"] == period_num)
        ]

        if len(match) == 0:
            continue

        raw_row = match.iloc[0]
        rows_checked += 1

        # Verify each field
        if not floats_match(row["signal"], raw_row["player.signal"]):
            mismatches["signal"] += 1
        if row["state"] != raw_row["player.state"]:
            mismatches["state"] += 1
        if not floats_match(row["price"], raw_row["player.price"]):
            mismatches["price"] += 1

    print_mismatch_results(mismatches, rows_checked, "raw CSV")
    return sum(mismatches.values())


# =====
# Validation of sale timing variables
# =====
def validate_sale_timing_vars(data):
    """Validate sale_prev_period and n_sales_earlier are computed correctly."""
    mismatches = {"sale_prev_period": 0, "n_sales_earlier": 0}
    rows_checked = 0

    # Group by group_round_id and check logic
    for group_round_id, group_df in data.groupby("group_round_id"):
        group_df = group_df.sort_values("period")
        periods = sorted(group_df["period"].unique())

        # Track cumulative sales by period
        sales_by_period = {}
        for period in periods:
            period_df = group_df[group_df["period"] == period]
            sales_by_period[period] = period_df["sold"].sum()

        for period in periods:
            period_df = group_df[group_df["period"] == period]

            # Expected sale_prev_period: any sale in period t-1
            if period > 1 and (period - 1) in sales_by_period:
                expected_sale_prev = 1 if sales_by_period[period - 1] > 0 else 0
            else:
                expected_sale_prev = 0

            # Expected n_sales_earlier: count of sales in periods 1 to t-2
            expected_n_earlier = sum(
                sales_by_period.get(p, 0) for p in periods if p < period - 1
            )

            # Check all players in this period
            for _, player_row in period_df.iterrows():
                rows_checked += 1
                if player_row["sale_prev_period"] != expected_sale_prev:
                    mismatches["sale_prev_period"] += 1
                if player_row["n_sales_earlier"] != expected_n_earlier:
                    mismatches["n_sales_earlier"] += 1

    print_mismatch_results(mismatches, rows_checked, "timing logic")
    return sum(mismatches.values())


# =====
# Helper functions
# =====
def floats_match(val1, val2, tol=1e-9):
    """Check if two floats match within tolerance."""
    if pd.isna(val1) and (val2 is None or pd.isna(val2)):
        return True
    if pd.isna(val1) or val2 is None or pd.isna(val2):
        return False
    return abs(val1 - val2) < tol


def print_mismatch_results(mismatches, rows_checked, source):
    """Print mismatch results for a validation."""
    print(f"  Checked {rows_checked} rows against {source}")
    for field, count in mismatches.items():
        status = "OK" if count == 0 else f"MISMATCH ({count})"
        print(f"    {field}: {status}")


# %%
if __name__ == "__main__":
    sys.exit(main())
