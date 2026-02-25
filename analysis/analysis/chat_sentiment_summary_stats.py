"""
Purpose: Generate LaTeX summary statistics table for chat sentiment dataset
Author: Claude Code
Date: 2026-02-25

Produces summary stats with metrics as columns and nested treatment x segment
rows. Structure: Overall, Treatment 1 (Seg 3, Seg 4), Treatment 2 (Seg 3, Seg 4).
"""

import pandas as pd
from pathlib import Path

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "datastore" / "derived" / "chat_sentiment_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "analysis" / "output" / "tables" / "chat_sentiment_summary_stats.tex"

METRICS = [
    ("$N$", "N", "{:.0f}"),
    ("Mean messages", "msg_mean", "{:.1f}"),
    ("Mean sentiment", "compound_mean", "{:.3f}"),
    ("SD sentiment", "compound_sd", "{:.3f}"),
    ("Frac.\\ positive", "frac_pos", "{:.3f}"),
    ("Frac.\\ negative", "frac_neg", "{:.3f}"),
]


# =====
# Main function
# =====
def main():
    """Compute summary stats and write LaTeX table."""
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH.name}")

    rows = build_row_data(df)
    latex = build_latex_table(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(latex)
    print(f"Wrote {OUTPUT_PATH}")


# =====
# Row computation
# =====
def build_row_data(df: pd.DataFrame) -> list[tuple]:
    """Build nested row structure: (label, indent_level, stats)."""
    return [
        ("Overall", 0, compute_stats(df)),
        ("Treatment 1", 0, compute_stats(df[df["treatment"] == "tr1"])),
        ("Segment 3", 1, compute_stats(filter_df(df, "tr1", 3))),
        ("Segment 4", 1, compute_stats(filter_df(df, "tr1", 4))),
        ("Treatment 2", 0, compute_stats(df[df["treatment"] == "tr2"])),
        ("Segment 3", 1, compute_stats(filter_df(df, "tr2", 3))),
        ("Segment 4", 1, compute_stats(filter_df(df, "tr2", 4))),
    ]


def filter_df(df: pd.DataFrame, treatment: str, segment: int) -> pd.DataFrame:
    """Filter to a specific treatment x segment subset."""
    return df[(df["treatment"] == treatment) & (df["segment"] == segment)]


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics for a subset of the data."""
    return {
        "N": len(df),
        "msg_mean": df["message_count"].mean(),
        "compound_mean": df["vader_compound_mean"].mean(),
        "compound_sd": df["vader_compound_mean"].std(),
        "frac_pos": df["frac_positive"].mean(),
        "frac_neg": df["frac_negative"].mean(),
    }


# =====
# LaTeX table building
# =====
def build_latex_table(rows: list[tuple]) -> str:
    """Build the complete LaTeX table string."""
    lines = table_header()
    lines += data_rows(rows)
    lines += table_footer()
    return "\n".join(lines) + "\n"


def table_header() -> list[str]:
    """Build LaTeX table header with metric columns."""
    col_labels = " & ".join(label for label, _, _ in METRICS)
    return [
        r"\begingroup",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(METRICS) + "}",
        r"\toprule",
        f" & {col_labels} \\\\",
        r"\midrule",
    ]


def data_rows(rows: list[tuple]) -> list[str]:
    """Build data rows with indentation for nested structure."""
    lines = []
    for label, indent, stats in rows:
        lines.append(format_data_row(label, indent, stats))
    return lines


def format_data_row(label: str, indent: int, stats: dict) -> str:
    """Format a single data row with optional indentation."""
    prefix = r"\quad " * indent
    values = [fmt.format(stats[key]) for _, key, fmt in METRICS]
    return f"  {prefix}{label} & " + " & ".join(values) + " \\\\"


def table_footer() -> list[str]:
    """Build LaTeX table footer."""
    n_cols = len(METRICS) + 1
    return [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par",
        r"\vspace{2pt}",
        rf"{{\footnotesize \textit{{Note:}} Sentiment scored with VADER "
        rf"compound score ($-1$ to $+1$). "
        rf"Positive: compound $> 0.05$; Negative: compound $< -0.05$.}}",
        r"\endgroup",
    ]


# %%
if __name__ == "__main__":
    main()
