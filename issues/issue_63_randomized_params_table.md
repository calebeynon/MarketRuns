# Issue #63: Add Pre-Randomized Parameters Table to Implementation Section

## Summary

Add a table to the Design of the Experiment section showing the pre-randomized experimental parameters (rounds, periods per round, chat status) by segment.

## Motivation

The paper's implementation section needs to document the pre-randomized experimental parameters that were generated prior to running sessions. This table shows readers the structure of each segment.

## Approach

- Create a Python script that parses the 4 oTree segment `__init__.py` files via regex to extract `NUM_ROUNDS_IN_SEGMENT` and `PERIODS_PER_ROUND`
- Generate a booktabs LaTeX table with columns: Segment, Chat, Rounds, Periods per Round, Total Periods, Avg. Periods per Round
- Include Total and Average summary rows
- Merge Overleaf edits into `main.tex` (full Design of the Experiment section content)

## Key Values

| Segment | Chat | Rounds | Periods per Round | Total | Avg |
|---------|------|--------|-------------------|-------|-----|
| 1 | No | 10 | 3, 10, 8, 4, 2, 3, 9, 7, 4, 6 | 56 | 5.6 |
| 2 | No | 5 | 9, 2, 5, 6, 5 | 27 | 5.4 |
| 3 | Yes | 6 | 9, 8, 4, 4, 3, 5 | 33 | 5.5 |
| 4 | Yes | 9 | 5, 3, 11, 3, 6, 14, 6, 3, 1 | 52 | 5.8 |
| **Total** | | **30** | | **168** | |
| **Average** | | **7.5** | | **42.0** | **5.6** |

## Scripts and Outputs

### Analysis
| Script | Input | Output |
|--------|-------|--------|
| `randomized_params_table.py` | `nonlivegame/chat_noavg*/__init__.py` | `analysis/output/tables/randomized_params.tex` |

### Paper
| File | Change |
|------|--------|
| `analysis/paper/main.tex` | Added table in Section 3; merged Overleaf edits (Market Model, Experimental Implementation, equations, liquidation example) |

### Tests
| Test File | Coverage |
|-----------|----------|
| `test_randomized_params_table.py` | 28/28 passed (parsing all 4 segments, table structure, values, summary rows) |
