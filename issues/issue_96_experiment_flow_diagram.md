# Issue #96: Add Experiment Flow Diagram to Paper

## Problem
The paper lacks a visual diagram showing the structure of the experiment flow, making it harder for readers to understand how the experiment is organized.

## Solution
Add a TikZ hierarchical tree diagram to the paper that shows:
- Part 1 → Segments → Rounds → Periods
- Color-coded chat vs. no-chat segments (gray for no-chat, blue for chat)
- Ellipsis notation for variable counts (Round 1, Round 2, ..., Round R)
- Geometric distribution annotations (R ~ Geometric(0.125), T ~ Geometric(0.125))

## Files Changed
- `analysis/paper/experiment_flow_diagram.tex` — New TikZ diagram file
- `analysis/paper/main.tex` — Includes the new diagram in the experimental design section

## Expected Output
A compiled figure in the paper showing the full experiment structure as a labeled tree diagram.
