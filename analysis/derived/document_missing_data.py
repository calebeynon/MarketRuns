"""
Purpose: Generate comprehensive missing data report for chat mitigation analysis
Author: Claude Code
Date: 2026-02-02

Analyzes missing data patterns in the chat mitigation dataset, focusing on
iMotions emotion variables. Checks for systematic patterns that could bias
the analysis.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
DERIVED_DIR = DATASTORE / "derived"
INPUT_PATH = DERIVED_DIR / "chat_mitigation_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "analysis" / "output" / "analysis" / "missing_data_report.txt"


# =====
# Main function
# =====
def main():
    """Generate missing data documentation report."""
    print_header("MISSING DATA DOCUMENTATION REPORT")
    print(f"Dataset: {INPUT_PATH}\n")

    df = load_dataset()
    if df is None:
        return None

    lines = generate_all_sections(df)
    save_report(lines)
    return df


def load_dataset() -> pd.DataFrame | None:
    """Load dataset or return None if not found."""
    if not INPUT_PATH.exists():
        print("ERROR: Dataset not found. Run build_chat_mitigation_dataset.py first.")
        return None
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} observations\n")
    return df


def generate_all_sections(df: pd.DataFrame) -> list[str]:
    """Generate all report sections."""
    lines = []
    lines.extend(document_overall_missing(df))
    lines.extend(document_missing_by_group(df, 'session_id', "SESSION"))
    lines.extend(document_missing_by_group(df, 'segment', "SEGMENT"))
    lines.extend(document_missing_by_player(df))
    lines.extend(compare_selling_rates(df))
    lines.extend(check_systematic_patterns(df))
    return lines


# =====
# Overall missing data
# =====
def document_overall_missing(df: pd.DataFrame) -> list[str]:
    """Document overall missing rates for emotion variables."""
    print_header("1. OVERALL MISSING DATA RATES")
    lines = ["=" * 60, "1. OVERALL MISSING DATA RATES", "=" * 60, ""]

    emotion_vars = ['fear_mean', 'anger_mean', 'sadness_mean', 'joy_mean',
                    'valence_mean', 'engagement_mean']
    for var in emotion_vars:
        n_missing = df[var].isna().sum()
        pct = 100 * n_missing / len(df)
        line = f"  {var}: {n_missing} missing ({pct:.1f}%)"
        print(line)
        lines.append(line)

    has_data = df['fear_mean'].notna().sum()
    coverage = f"\nTotal with emotion data: {has_data} ({100 * has_data / len(df):.1f}%)"
    print(coverage)
    lines.append(coverage)
    return lines


# =====
# Missing by grouping variable
# =====
def document_missing_by_group(df: pd.DataFrame, group_var: str, label: str) -> list[str]:
    """Document missing rates by a grouping variable."""
    print_header(f"2. MISSING DATA BY {label}")
    lines = ["", "=" * 60, f"2. MISSING DATA BY {label}", "=" * 60, ""]

    group_stats = df.groupby(group_var).agg(
        n_obs=('fear_mean', 'size'),
        n_missing=('fear_mean', lambda x: x.isna().sum()),
        pct_missing=('fear_mean', lambda x: 100 * x.isna().mean())
    ).round(1)

    print(group_stats.to_string())
    lines.append(group_stats.to_string())

    high_missing = group_stats[group_stats['pct_missing'] > 50]
    if not high_missing.empty:
        warning = f"\nWARNING: {len(high_missing)} {label.lower()}s with >50% missing"
        print(warning)
        lines.append(warning)
    return lines


# =====
# Missing by player
# =====
def document_missing_by_player(df: pd.DataFrame) -> list[str]:
    """Document missing rates by player."""
    print_header("3. MISSING DATA BY PLAYER")
    lines = ["", "=" * 60, "3. MISSING DATA BY PLAYER", "=" * 60, ""]

    player_stats = df.groupby('player_id').agg(
        pct_missing=('fear_mean', lambda x: 100 * x.isna().mean())).round(1)

    desc = player_stats['pct_missing'].describe()
    print("Distribution of player-level missing rates:")
    print(desc.to_string())
    lines.extend(["Distribution of missing rates:", desc.to_string()])

    summary = summarize_player_coverage(player_stats)
    print(summary)
    lines.append(summary)
    return lines


def summarize_player_coverage(player_stats: pd.DataFrame) -> str:
    """Summarize player data coverage counts."""
    n_complete = (player_stats['pct_missing'] == 0).sum()
    n_partial = ((player_stats['pct_missing'] > 0) & (player_stats['pct_missing'] < 100)).sum()
    n_none = (player_stats['pct_missing'] == 100).sum()
    return f"\nComplete: {n_complete}, Partial: {n_partial}, None: {n_none}"


# =====
# Selling rate comparison
# =====
def compare_selling_rates(df: pd.DataFrame) -> list[str]:
    """Compare selling rates for observations with vs without emotion data."""
    print_header("4. SELLING RATE COMPARISON")
    lines = ["", "=" * 60, "4. SELLING RATE COMPARISON", "=" * 60, ""]

    df_at_risk = df[df['already_sold'] == 0].copy()
    df_at_risk['has_emotions'] = df_at_risk['fear_mean'].notna()

    comparison = df_at_risk.groupby('has_emotions').agg(
        n_obs=('sold', 'size'), n_sold=('sold', 'sum'), sell_rate=('sold', 'mean')).round(4)
    print(comparison.to_string())
    lines.append(comparison.to_string())

    test_result = run_chi_square_test(df_at_risk)
    if test_result:
        print(test_result)
        lines.append(test_result)
    return lines


def run_chi_square_test(df: pd.DataFrame) -> str | None:
    """Run chi-square test for selling rate differences by emotion availability."""
    with_emo = df[df['has_emotions']]['sold']
    without_emo = df[~df['has_emotions']]['sold']
    if len(without_emo) < 10 or len(with_emo) < 10:
        return None
    contingency = pd.crosstab(df['has_emotions'], df['sold'])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    sig = "SIGNIFICANT" if p_value < 0.05 else "NOT significant"
    return f"\nChi-square: chi2={chi2:.2f}, p={p_value:.4f} ({sig})"


# =====
# Systematic pattern check
# =====
def check_systematic_patterns(df: pd.DataFrame) -> list[str]:
    """Check for systematic patterns in missingness."""
    print_header("5. SYSTEMATIC PATTERN CHECK")
    lines = ["", "=" * 60, "5. SYSTEMATIC PATTERN CHECK", "=" * 60, ""]

    df_test = df.copy()
    df_test['missing_emotions'] = df_test['fear_mean'].isna().astype(int)

    print("Testing if observables predict missing emotion data:\n")
    lines.append("Testing if observables predict missing emotion data:")

    for var in ['chat_segment', 'round', 'signal', 'prior_group_sales']:
        corr = df_test['missing_emotions'].corr(df_test[var])
        t_stat = corr * np.sqrt(len(df_test) - 2) / np.sqrt(1 - corr**2)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(df_test) - 2))
        sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
        result = f"  {var}: r={corr:.4f}, p={p_value:.4f} {sig}"
        print(result)
        lines.append(result)
    return lines


# =====
# Output functions
# =====
def save_report(report_lines: list[str]):
    """Save report to text file."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"\n{'=' * 60}\nReport saved to: {OUTPUT_PATH}")


def print_header(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# %%
if __name__ == "__main__":
    main()
