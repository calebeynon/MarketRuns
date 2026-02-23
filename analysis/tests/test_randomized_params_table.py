"""
Purpose: Validate randomized_params_table.py output against known segment values
         from the oTree __init__.py files.
Author: Claude Code
Date: 2026-02-22
"""

import re
from pathlib import Path

import pytest

from analysis.analysis.randomized_params_table import (
    parse_num_rounds,
    parse_periods_per_round,
)

# =====
# File paths and constants
# =====
SEGMENT_INIT_DIR = Path("nonlivegame")
TEX_PATH = Path("analysis/output/tables/randomized_params.tex")

SEGMENT_DIRS = [
    "chat_noavg",   # Segment 1
    "chat_noavg2",  # Segment 2
    "chat_noavg3",  # Segment 3
    "chat_noavg4",  # Segment 4
]

EXPECTED = {
    1: {"rounds": 10, "periods": [3, 10, 8, 4, 2, 3, 9, 7, 4, 6], "chat": "No"},
    2: {"rounds": 5,  "periods": [9, 2, 5, 6, 5],                  "chat": "No"},
    3: {"rounds": 6,  "periods": [9, 8, 4, 4, 3, 5],               "chat": "Yes"},
    4: {"rounds": 9,  "periods": [5, 3, 11, 3, 6, 14, 6, 3, 1],    "chat": "Yes"},
}


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def tex():
    return TEX_PATH.read_text()


def _read_segment_source(seg_dir):
    """Read __init__.py source for a given segment directory."""
    return (SEGMENT_INIT_DIR / seg_dir / "__init__.py").read_text()


# =====
# LaTeX parsing helpers
# =====
def parse_data_rows(tex):
    """Extract data rows between first midrule and bottomrule."""
    body = re.search(r"\\midrule(.+?)\\bottomrule", tex, re.DOTALL)
    assert body, "Could not find table body"
    rows = []
    for line in body.group(1).strip().split("\n"):
        line = line.strip().rstrip("\\").strip()
        if not line or line == "\\midrule":
            continue
        cells = [c.strip() for c in line.split("&")]
        rows.append(cells)
    return rows


def parse_segment_rows(tex):
    """Return only the 4 segment data rows (row[0] is a digit)."""
    return [r for r in parse_data_rows(tex) if re.match(r"^\d$", r[0])]


def parse_summary_rows(tex):
    """Return summary rows (Total, Average)."""
    return [r for r in parse_data_rows(tex) if not re.match(r"^\d$", r[0])]


# =====
# Tests: parsing functions
# =====
class TestParseSegmentParams:

    def test_parse_segment_1(self):
        source = _read_segment_source("chat_noavg")
        assert parse_num_rounds(source) == 10
        assert parse_periods_per_round(source) == [3, 10, 8, 4, 2, 3, 9, 7, 4, 6]

    def test_parse_segment_2(self):
        source = _read_segment_source("chat_noavg2")
        assert parse_num_rounds(source) == 5
        assert parse_periods_per_round(source) == [9, 2, 5, 6, 5]

    def test_parse_segment_3(self):
        source = _read_segment_source("chat_noavg3")
        assert parse_num_rounds(source) == 6
        assert parse_periods_per_round(source) == [9, 8, 4, 4, 3, 5]

    def test_parse_segment_4(self):
        source = _read_segment_source("chat_noavg4")
        assert parse_num_rounds(source) == 9
        assert parse_periods_per_round(source) == [5, 3, 11, 3, 6, 14, 6, 3, 1]

    def test_num_rounds_matches_periods_length(self):
        """For each segment, num_rounds must equal len(periods_per_round)."""
        for seg_dir in SEGMENT_DIRS:
            source = _read_segment_source(seg_dir)
            num_rounds = parse_num_rounds(source)
            periods = parse_periods_per_round(source)
            assert num_rounds == len(periods), (
                f"{seg_dir}: num_rounds={num_rounds} != "
                f"len(periods)={len(periods)}"
            )


# =====
# Tests: LaTeX table structure
# =====
class TestTableOutput:

    def test_output_file_exists(self):
        assert TEX_PATH.exists(), f"Missing: {TEX_PATH}"

    def test_table_has_four_segment_rows(self, tex):
        seg_rows = parse_segment_rows(tex)
        assert len(seg_rows) == 4, (
            f"Expected 4 segment rows, got {len(seg_rows)}"
        )

    def test_table_has_total_row(self, tex):
        summary = parse_summary_rows(tex)
        labels = [r[0].lower() for r in summary]
        assert any("total" in lb for lb in labels), (
            f"Expected a Total row, found: {labels}"
        )

    def test_booktabs_structure(self, tex):
        assert "\\toprule" in tex
        assert "\\midrule" in tex
        assert "\\bottomrule" in tex

    # Table columns: Segment & Chat & Rounds & Periods per Round
    #                & Total Periods & Avg. Periods per Round

    @pytest.mark.parametrize("seg_num", [1, 2, 3, 4])
    def test_segment_rounds(self, tex, seg_num):
        row = parse_segment_rows(tex)[seg_num - 1]
        expected = EXPECTED[seg_num]["rounds"]
        assert int(row[2]) == expected, (
            f"Segment {seg_num}: expected {expected} rounds, got {row[2]}"
        )

    @pytest.mark.parametrize("seg_num", [1, 2, 3, 4])
    def test_segment_total_periods(self, tex, seg_num):
        row = parse_segment_rows(tex)[seg_num - 1]
        expected = sum(EXPECTED[seg_num]["periods"])
        assert int(row[4]) == expected, (
            f"Segment {seg_num}: expected {expected} periods, got {row[4]}"
        )

    @pytest.mark.parametrize("seg_num", [1, 2, 3, 4])
    def test_segment_avg_periods(self, tex, seg_num):
        row = parse_segment_rows(tex)[seg_num - 1]
        periods = EXPECTED[seg_num]["periods"]
        expected_avg = sum(periods) / len(periods)
        actual_avg = float(row[5])
        assert actual_avg == pytest.approx(expected_avg, abs=0.05), (
            f"Segment {seg_num}: expected avg {expected_avg:.1f}, "
            f"got {actual_avg}"
        )

    @pytest.mark.parametrize("seg_num", [1, 2, 3, 4])
    def test_chat_column_values(self, tex, seg_num):
        row = parse_segment_rows(tex)[seg_num - 1]
        expected = EXPECTED[seg_num]["chat"]
        assert row[1] == expected, (
            f"Segment {seg_num}: expected chat='{expected}', "
            f"got '{row[1]}'"
        )

    def test_total_row_rounds(self, tex):
        summary = parse_summary_rows(tex)
        total_row = [r for r in summary if "total" in r[0].lower()]
        assert len(total_row) == 1, "Expected exactly one Total row"
        assert int(total_row[0][2]) == 30

    def test_total_row_periods(self, tex):
        summary = parse_summary_rows(tex)
        total_row = [r for r in summary if "total" in r[0].lower()]
        assert len(total_row) == 1, "Expected exactly one Total row"
        assert int(total_row[0][4]) == 168

    def test_average_row_avg_periods(self, tex):
        """Average of per-segment avg periods should be ~5.6."""
        summary = parse_summary_rows(tex)
        avg_row = [r for r in summary if "avg" in r[0].lower()]
        assert len(avg_row) == 1, "Expected exactly one Avg row"
        actual_avg = float(avg_row[0][5])
        assert actual_avg == pytest.approx(5.6, abs=0.1)


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
