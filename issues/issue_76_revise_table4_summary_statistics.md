# Issue 76: Revise Table 4 Summary Statistics

## Summary
Revise Table 4 (seller counts and market outcomes) to add total group-round counts, good/bad state counts, and average sell period for first sellers. Remove average sell price. Reorder rows.

## Motivation
Table 4 lacked state-level breakdowns and first-seller timing information. Adding these provides reviewers with better context for interpreting selling behavior across treatment and chat conditions.

## Changes
- `analysis/analysis/summary_statistics.R` - Added `stat_total_group_rounds`, `stat_avg_first_seller_period`; removed `stat_avg_sell_price`; rewrote `write_seller_count_table` with new 15-row layout
- `analysis/output/tables/summary_seller_counts.tex` - Regenerated table with new row order
- `analysis/paper/main.tex` - Updated Table 4 caption
- `analysis/tests/test_summary_statistics.py` - 3 new tests, 1 removed, row indices updated (8/8 seller counts tests pass)
- `analysis/tests/test_seller_counts_raw_validation.py` - Added `TestSellPeriodsVsRawData` class validating sell_period, sell_price, and first-seller period against raw oTree exports (9/9 pass)

## Expected Output
- `analysis/output/tables/summary_seller_counts.tex` - 15-row table: total group-rounds, good/bad state counts, zero-seller rounds, avg sellers, avg sell period, avg first-seller period (each with good/bad sub-rows where applicable)

## Key Design Decisions
- First seller = player(s) with earliest sell_period in a group-round; all tied-first sellers count
- Zero-seller group-rounds excluded from first-seller average
- State counts are raw counts (not percentages)
- Good/bad state sub-rows retained on all statistics

## PR Type
Table Revision
