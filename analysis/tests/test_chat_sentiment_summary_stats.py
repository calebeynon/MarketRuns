"""
Purpose: Verify chat sentiment summary stats table against independent computation
Author: Claude Code
Date: 2026-02-25

Parses the LaTeX output and cross-validates every cell against values computed
directly from the derived CSV.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DERIVED_CSV = PROJECT_ROOT / "datastore" / "derived" / "chat_sentiment_dataset.csv"
TABLE_PATH = (
    PROJECT_ROOT / "analysis" / "output" / "tables"
    / "chat_sentiment_summary_stats.tex"
)

skip_no_data = pytest.mark.skipif(
    not DERIVED_CSV.exists() or not TABLE_PATH.exists(),
    reason="Derived CSV or LaTeX table not found",
)


# =====
# LaTeX parsing helpers
# =====
def parse_latex_table(path: Path) -> dict[str, list[str]]:
    """Parse the LaTeX table into {row_label: [cell_values]}."""
    text = path.read_text()
    rows = {}
    for line in text.split("\n"):
        line = line.strip()
        if "&" not in line or "toprule" in line or "midrule" in line:
            continue
        if line.startswith("&"):
            continue
        cells = [c.strip().rstrip("\\").strip() for c in line.split("&")]
        label = cells[0].strip()
        values = cells[1:]
        rows[label] = values
    return rows


def find_row(rows: dict, keyword: str) -> list[float]:
    """Find a row by keyword and parse its numeric values."""
    for label, values in rows.items():
        if keyword.lower() in label.lower():
            return [float(v) for v in values]
    raise KeyError(f"No row matching '{keyword}' in table")


# =====
# Independent computation from CSV
# =====
def compute_stat(df: pd.DataFrame, column: str, func: str) -> float:
    """Compute a single statistic from a dataframe column."""
    if func == "mean":
        return df[column].mean()
    if func == "std":
        return df[column].std()
    if func == "count":
        return float(len(df))
    raise ValueError(f"Unknown func: {func}")


def get_strata_subsets(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Return 9 subsets in the column order of the LaTeX table."""
    return [
        df,
        df[df["treatment"] == "tr1"],
        df[df["treatment"] == "tr2"],
        df[df["segment"] == 3],
        df[df["segment"] == 4],
        df[(df["treatment"] == "tr1") & (df["segment"] == 3)],
        df[(df["treatment"] == "tr1") & (df["segment"] == 4)],
        df[(df["treatment"] == "tr2") & (df["segment"] == 3)],
        df[(df["treatment"] == "tr2") & (df["segment"] == 4)],
    ]


# =====
# Tests
# =====
@skip_no_data
def test_n_values():
    """Verify N (group-segments) row matches actual counts."""
    df = pd.read_csv(DERIVED_CSV)
    rows = parse_latex_table(TABLE_PATH)
    table_values = find_row(rows, "$N$")
    expected = [float(len(s)) for s in get_strata_subsets(df)]
    assert table_values == expected


@skip_no_data
def test_mean_messages():
    """Verify mean messages row matches independent computation."""
    df = pd.read_csv(DERIVED_CSV)
    rows = parse_latex_table(TABLE_PATH)
    table_values = find_row(rows, "Mean messages")
    subsets = get_strata_subsets(df)

    for i, subset in enumerate(subsets):
        expected = round(subset["message_count"].mean(), 1)
        assert table_values[i] == expected, (
            f"Stratum {i}: table={table_values[i]}, computed={expected}"
        )


@skip_no_data
def test_mean_vader_compound():
    """Verify mean VADER compound row matches independent computation."""
    df = pd.read_csv(DERIVED_CSV)
    rows = parse_latex_table(TABLE_PATH)
    table_values = find_row(rows, "Mean VADER compound")
    subsets = get_strata_subsets(df)

    for i, subset in enumerate(subsets):
        expected = round(subset["vader_compound_mean"].mean(), 3)
        assert table_values[i] == expected, (
            f"Stratum {i}: table={table_values[i]}, computed={expected}"
        )


@skip_no_data
def test_sd_vader_compound():
    """Verify SD VADER compound row matches independent computation (ddof=1)."""
    df = pd.read_csv(DERIVED_CSV)
    rows = parse_latex_table(TABLE_PATH)
    table_values = find_row(rows, "SD VADER compound")
    subsets = get_strata_subsets(df)

    for i, subset in enumerate(subsets):
        expected = round(subset["vader_compound_mean"].std(), 3)
        assert table_values[i] == expected, (
            f"Stratum {i}: table={table_values[i]}, computed={expected}"
        )


@skip_no_data
def test_frac_positive():
    """Verify fraction positive row matches independent computation."""
    df = pd.read_csv(DERIVED_CSV)
    rows = parse_latex_table(TABLE_PATH)
    table_values = find_row(rows, "positive")
    subsets = get_strata_subsets(df)

    for i, subset in enumerate(subsets):
        expected = round(subset["frac_positive"].mean(), 3)
        assert table_values[i] == expected, (
            f"Stratum {i}: table={table_values[i]}, computed={expected}"
        )


@skip_no_data
def test_frac_negative():
    """Verify fraction negative row matches independent computation."""
    df = pd.read_csv(DERIVED_CSV)
    rows = parse_latex_table(TABLE_PATH)
    table_values = find_row(rows, "negative")
    subsets = get_strata_subsets(df)

    for i, subset in enumerate(subsets):
        expected = round(subset["frac_negative"].mean(), 3)
        assert table_values[i] == expected, (
            f"Stratum {i}: table={table_values[i]}, computed={expected}"
        )


@skip_no_data
def test_table_has_nine_columns():
    """Table should have 9 data columns (Overall + 2 treatments + 2 segments + 4 interactions)."""
    rows = parse_latex_table(TABLE_PATH)
    for label, values in rows.items():
        assert len(values) == 9, (
            f"Row '{label}' has {len(values)} columns, expected 9"
        )


@skip_no_data
def test_strata_ns_sum_correctly():
    """Treatment strata sum to overall; interaction strata sum to marginals."""
    rows = parse_latex_table(TABLE_PATH)
    n_vals = find_row(rows, "$N$")

    overall = n_vals[0]
    tr1, tr2 = n_vals[1], n_vals[2]
    seg3, seg4 = n_vals[3], n_vals[4]
    tr1s3, tr1s4, tr2s3, tr2s4 = n_vals[5], n_vals[6], n_vals[7], n_vals[8]

    assert tr1 + tr2 == overall, "Tr1 + Tr2 != Overall"
    assert seg3 + seg4 == overall, "Seg3 + Seg4 != Overall"
    assert tr1s3 + tr1s4 == tr1, "Tr1xSeg3 + Tr1xSeg4 != Tr1"
    assert tr2s3 + tr2s4 == tr2, "Tr2xSeg3 + Tr2xSeg4 != Tr2"
    assert tr1s3 + tr2s3 == seg3, "Tr1xSeg3 + Tr2xSeg3 != Seg3"
    assert tr1s4 + tr2s4 == seg4, "Tr1xSeg4 + Tr2xSeg4 != Seg4"
