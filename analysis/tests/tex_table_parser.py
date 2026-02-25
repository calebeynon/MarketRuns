"""
Purpose: Parsing helpers for LaTeX longtable .tex output files.
         Extracts coefficient rows, fit statistics, and provides
         label lookup and position utilities.
Author: Claude Code
Date: 2026-02-24
"""

import re


# =====
# Main parsing functions
# =====
def parse_coef_rows(tex, fit_stat_labels):
    """Parse coefficient rows with their 4 column values.

    Returns list of (label, [val1, val2, val3, val4]).
    """
    body = extract_table_body(tex)
    rows = []
    for line in body.split("\n"):
        parsed = parse_single_line(line, fit_stat_labels)
        if parsed is not None:
            rows.append(parsed)
    return rows


def parse_fit_rows(tex):
    """Parse fit statistic rows into list of (label, [val1..val4])."""
    match = re.search(
        r"\\emph\{Fit statistics\}.*?\n(.+?)\\midrule\s*\\midrule",
        tex, re.DOTALL,
    )
    if not match:
        return []
    rows = []
    for line in match.group(1).strip().split("\n"):
        line = line.strip().rstrip("\\\\").strip()
        if not line or line.startswith("\\"):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) >= 5:
            rows.append((cells[0].strip(), cells[1:5]))
    return rows


# =====
# Internal helpers
# =====
def extract_table_body(tex):
    """Extract content between first \\endlastfoot and fit statistics."""
    start = tex.find("\\endlastfoot")
    if start == -1:
        return ""
    fit_marker = tex.find("\\emph{Fit statistics}", start)
    if fit_marker == -1:
        return tex[start:]
    return tex[start:fit_marker]


def parse_single_line(line, fit_stat_labels):
    """Parse one line into (label, values) or None if not a coef row."""
    line = line.strip().rstrip("\\\\").strip()
    if not line or line.startswith("\\"):
        return None
    cells = [c.strip() for c in line.split("&")]
    if len(cells) < 5:
        return None
    label = cells[0].strip()
    if not label or label.startswith("(") or label in fit_stat_labels:
        return None
    return (label, [c.strip() for c in cells[1:5]])


# =====
# Lookup utilities
# =====
def find_row_by_label(rows, label):
    """Find a row by its label text."""
    for row_label, vals in rows:
        if row_label == label:
            return vals
    return None


def has_value(cell):
    """Check if a cell contains a numeric coefficient value."""
    return bool(re.search(r"\d", cell))


def label_position(rows, label):
    """Return the index position of a label in the row list."""
    for i, (row_label, _) in enumerate(rows):
        if row_label == label:
            return i
    return -1
