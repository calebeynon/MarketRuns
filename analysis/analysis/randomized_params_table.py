"""
Purpose: Parse oTree segment configs and generate a LaTeX table of randomized parameters.
Author: Caleb Eynon
Date: 2026-02-22
"""

import re
from pathlib import Path

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OTREE_DIR = PROJECT_ROOT / 'nonlivegame'
SEGMENT_DIRS = ['chat_noavg', 'chat_noavg2', 'chat_noavg3', 'chat_noavg4']
OUTPUT_FILE = PROJECT_ROOT / 'analysis' / 'output' / 'tables' / 'randomized_params.tex'

# Segments 1-2 have no chat, segments 3-4 have chat
CHAT_SEGMENTS = {1: 'No', 2: 'No', 3: 'Yes', 4: 'Yes'}


# =====
# Main function
# =====
def main():
    """Parse oTree configs and write LaTeX table."""
    segments = [
        load_segment(i, dir_name)
        for i, dir_name in enumerate(SEGMENT_DIRS, start=1)
    ]

    latex = build_latex_table(segments)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(latex)
    print(f"Wrote {OUTPUT_FILE}")


# =====
# Parsing functions
# =====
def parse_num_rounds(source):
    """Extract NUM_ROUNDS_IN_SEGMENT integer from source."""
    match = re.search(
        r'NUM_ROUNDS_IN_SEGMENT\s*=\s*(\d+)', source
    )
    if not match:
        raise ValueError("NUM_ROUNDS_IN_SEGMENT not found")
    return int(match.group(1))


def parse_periods_per_round(source):
    """Extract PERIODS_PER_ROUND list from source."""
    match = re.search(
        r'PERIODS_PER_ROUND\s*=\s*\[([^\]]+)\]', source
    )
    if not match:
        raise ValueError("PERIODS_PER_ROUND not found")
    return [int(x.strip()) for x in match.group(1).split(',')]


def load_segment(seg_num, dir_name):
    """Read and validate one segment's config, return segment data."""
    init_path = OTREE_DIR / dir_name / '__init__.py'
    if not init_path.exists():
        raise FileNotFoundError(
            f"Segment {seg_num} config not found: {init_path}"
        )
    source = init_path.read_text()
    num_rounds = parse_num_rounds(source)
    periods = parse_periods_per_round(source)
    if num_rounds != len(periods):
        raise ValueError(
            f"Segment {seg_num} ({dir_name}): "
            f"NUM_ROUNDS={num_rounds} != len(PERIODS)={len(periods)}"
        )
    return build_segment_data(seg_num, num_rounds, periods)


# =====
# Data building
# =====
def build_segment_data(segment_num, num_rounds, periods):
    """Build a dict of segment summary data."""
    return {
        'segment': segment_num,
        'chat': CHAT_SEGMENTS[segment_num],
        'rounds': num_rounds,
        'periods': periods,
        'total_periods': sum(periods),
        'avg_periods': sum(periods) / len(periods),
    }


# =====
# LaTeX generation
# =====
def build_latex_table(segments):
    """Build complete booktabs LaTeX table string."""
    lines = table_header()
    for seg in segments:
        lines.append(format_segment_row(seg))
    lines += table_footer(segments)
    lines.append('')
    return '\n'.join(lines)


def table_header():
    """Return the LaTeX table header lines."""
    return [
        r'\begingroup',
        r'\centering',
        r'\small',
        r'\begin{tabular}{clcccc}',
        r'\toprule',
        r'Segment & Chat & Rounds & Periods per Round'
        r' & Total Periods & Avg. per Round \\',
        r'\midrule',
    ]


def table_footer(segments):
    """Return the LaTeX table footer lines with summary rows."""
    total_rounds = sum(s['rounds'] for s in segments)
    total_periods = sum(s['total_periods'] for s in segments)
    return [
        r'\midrule',
        format_total_row(total_rounds, total_periods),
        format_average_row(segments),
        r'\bottomrule',
        r'\end{tabular}',
        r'\endgroup',
    ]


def format_segment_row(seg):
    """Format one segment as a LaTeX table row."""
    periods_str = ', '.join(str(p) for p in seg['periods'])
    return (
        f"  {seg['segment']} & {seg['chat']} & {seg['rounds']}"
        f" & {periods_str} & {seg['total_periods']}"
        f" & {seg['avg_periods']:.1f} \\\\"
    )


def format_total_row(total_rounds, total_periods):
    """Format the Total summary row."""
    return (
        f"  Total & --- & {total_rounds}"
        f" & --- & {total_periods} & --- \\\\"
    )


def format_average_row(segments):
    """Format the Avg. per Segment summary row."""
    n = len(segments)
    avg_rounds = sum(s['rounds'] for s in segments) / n
    avg_total = sum(s['total_periods'] for s in segments) / n
    avg_per_round = (
        sum(s['avg_periods'] for s in segments) / n
    )
    return (
        f"  Avg. per Segment & --- & {avg_rounds:.1f}"
        f" & --- & {avg_total:.1f} & {avg_per_round:.1f} \\\\"
    )


# %%
if __name__ == "__main__":
    main()
