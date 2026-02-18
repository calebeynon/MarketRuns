"""
Purpose: Shared helpers for first seller trait verification tests
Author: Claude Code
Date: 2026-02-17

Provides fixtures, data parsing, and statistical helpers used by
test_visualize_first_seller_traits.py to cross-validate against raw data.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
import market_data as md
from analysis.derived.build_survey_traits_dataset import (
    compute_agreeableness,
    compute_conscientiousness,
    compute_extraversion,
    compute_impulsivity,
    compute_neuroticism,
    compute_openness,
    compute_state_anxiety,
)

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
DERIVED_DIR = DATASTORE / "derived"
ROUND_DATA_CSV = DERIVED_DIR / "first_seller_round_data.csv"
SURVEY_TRAITS_CSV = DERIVED_DIR / "survey_traits.csv"
VIZ_R_SCRIPT = PROJECT_ROOT / "analysis" / "analysis" / "visualize_first_seller_traits.R"
DESC_R_SCRIPT = PROJECT_ROOT / "analysis" / "analysis" / "first_seller_descriptive_stats.R"
OUTPUT_CI_DIFF = PROJECT_ROOT / "analysis" / "output" / "plots" / "first_seller_trait_ci_diff.pdf"
OUTPUT_ROBUSTNESS = PROJECT_ROOT / "analysis" / "output" / "plots" / "first_seller_trait_robustness.pdf"
LATEX_TABLE = PROJECT_ROOT / "analysis" / "output" / "tables" / "first_seller_trait_comparisons.tex"

SESSION_FILES = {
    "1_11-7-tr1": DATASTORE / "1_11-7-tr1" / "all_apps_wide_2025-11-07.csv",
    "2_11-10-tr2": DATASTORE / "2_11-10-tr2" / "all_apps_wide_2025-11-10.csv",
    "3_11-11-tr2": DATASTORE / "3_11-11-tr2" / "all_apps_wide_2025-11-11.csv",
    "4_11-12-tr1": DATASTORE / "4_11-12-tr1" / "all_apps_wide_2025-11-12.csv",
    "5_11-14-tr2": DATASTORE / "5_11-14-tr2" / "all_apps_wide_2025-11-14.csv",
    "6_11-18-tr1": DATASTORE / "6_11-18-tr1" / "all_apps_wide_2025-11-18.csv",
}
SEGMENT_MAP = {1: "chat_noavg", 2: "chat_noavg2", 3: "chat_noavg3", 4: "chat_noavg4"}

TRAITS = [
    "extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness", "impulsivity", "state_anxiety",
]


# =====
# Main function
# =====
def main():
    """Module is not runnable; import helpers from here."""
    pass


# =====
# Raw data parsing helpers
# =====
def parse_all_sessions():
    """Parse all 6 raw oTree sessions via market_data.py."""
    sessions = {}
    for session_id, csv_path in SESSION_FILES.items():
        if not csv_path.exists():
            return None
        experiment = md.parse_experiment(str(csv_path))
        sessions[session_id] = experiment.sessions[0]
    return sessions if len(sessions) == 6 else None


def compute_raw_first_seller_counts(parsed_sessions):
    """Count how many times each player was first seller from raw data."""
    counts = {}
    for session_id, session in parsed_sessions.items():
        for seg_name in SEGMENT_MAP.values():
            segment = session.get_segment(seg_name)
            if not segment:
                continue
            for group in segment.groups.values():
                for round_num in segment.rounds:
                    sellers = find_first_sellers(segment, round_num, group)
                    for label in sellers:
                        key = (session_id, label)
                        counts[key] = counts.get(key, 0) + 1
    return counts


def find_first_sellers(segment, round_num, group):
    """Find all who sold in the earliest period of a group-round."""
    round_obj = segment.get_round(round_num)
    if not round_obj:
        return []
    for period_num in sorted(round_obj.periods.keys()):
        period = round_obj.get_period(period_num)
        sellers = [
            l for l in group.player_labels
            if l in period.players and period.players[l].sold_this_period
        ]
        if sellers:
            return sellers
    return []


def compute_raw_survey_traits():
    """Compute trait scores from raw survey CSVs independently."""
    records = []
    for session_id in SESSION_FILES:
        csv_files = list((DATASTORE / session_id).glob("survey_*.csv"))
        if not csv_files:
            continue
        df = pd.read_csv(csv_files[0])
        for _, row in df.iterrows():
            if any(pd.isna(row.get(f"player.q{i}")) for i in range(1, 25)):
                continue
            records.append(build_trait_record(row, session_id))
    return pd.DataFrame(records)


def build_trait_record(row, session_id):
    """Compute all trait scores from a raw survey row."""
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
    }


# =====
# Classification and statistics helpers
# =====
def classify_group(x):
    """Replicate R's cut_fs_group factor classification."""
    if x == 0:
        return "0 times"
    elif x <= 2:
        return "1-2 times"
    return "3+ times"


def welch_t_test(fs_vals, nfs_vals):
    """Run Welch's t-test and return diff, p-value, CI bounds."""
    result = stats.ttest_ind(fs_vals, nfs_vals, equal_var=False)
    diff = fs_vals.mean() - nfs_vals.mean()
    s1_n = fs_vals.var(ddof=1) / len(fs_vals)
    s2_n = nfs_vals.var(ddof=1) / len(nfs_vals)
    se = np.sqrt(s1_n + s2_n)
    df = (s1_n + s2_n) ** 2 / (
        s1_n ** 2 / (len(fs_vals) - 1) + s2_n ** 2 / (len(nfs_vals) - 1)
    )
    crit = stats.t.ppf(0.975, df)
    return diff, result.pvalue, diff - crit * se, diff + crit * se


def p_to_stars(p):
    """Convert p-value to significance stars matching R convention."""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


# =====
# LaTeX parsing helpers
# =====
def parse_latex_differences(latex_text):
    """Extract trait name -> difference value from LaTeX data rows."""
    results = {}
    for line in latex_text.split("\n"):
        if "&" not in line or "Trait" in line:
            continue
        parts = [p.strip() for p in line.strip().rstrip("\\").strip().split("&")]
        if len(parts) >= 4:
            diff_str = re.sub(r"\*+", "", parts[3]).strip()
            try:
                results[parts[0].strip()] = float(diff_str)
            except ValueError:
                continue
    return results


def parse_latex_stars(latex_text):
    """Extract trait name -> significance stars from LaTeX."""
    results = {}
    for line in latex_text.split("\n"):
        if "&" not in line or "Trait" in line:
            continue
        parts = [p.strip() for p in line.strip().rstrip("\\").strip().split("&")]
        if len(parts) >= 4:
            star_match = re.search(r"(\*+)", parts[3].strip())
            results[parts[0].strip()] = star_match.group(1) if star_match else ""
    return results


def parse_latex_mean_sd(text):
    """Extract (mean, sd) pairs for each trait from LaTeX rows."""
    pattern = r"([\d.]+)\s*\(([\d.]+)\)"
    results = {}
    for line in text.split("\n"):
        if "&" not in line or "Trait" in line:
            continue
        parts = [p.strip() for p in line.strip().rstrip("\\").strip().split("&")]
        if len(parts) < 4:
            continue
        trait_key = parts[0].strip().lower().replace(" ", "_")
        matches = re.findall(pattern, line)
        if len(matches) >= 2:
            results[trait_key] = {
                "fs_mean": float(matches[0][0]), "fs_sd": float(matches[0][1]),
                "nfs_mean": float(matches[1][0]), "nfs_sd": float(matches[1][1]),
            }
    return results


def count_group_rounds(parsed_sessions):
    """Count total group-rounds across all sessions."""
    total = 0
    for session in parsed_sessions.values():
        for seg_name in SEGMENT_MAP.values():
            segment = session.get_segment(seg_name)
            if segment:
                total += len(segment.rounds) * len(segment.groups)
    return total


def build_individual_data(round_data, survey_data):
    """Build individual-level dataset matching R scripts' logic."""
    counts = round_data.groupby(
        ["session_id", "player"]
    )["is_first_seller"].sum().reset_index()
    counts.rename(columns={"is_first_seller": "times_first_seller"}, inplace=True)
    merged = counts.merge(survey_data, on=["session_id", "player"], how="inner")
    merged["is_ever_first_seller"] = merged["times_first_seller"] >= 1
    merged["fs_group"] = merged["times_first_seller"].apply(classify_group)
    return merged


# %%
if __name__ == "__main__":
    main()
