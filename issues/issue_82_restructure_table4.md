# Issue 82: Restructure Table 4 — Collapse States, Expand Seller Counts, Add All Seller Periods

## Problem

Table 4 (summary seller counts) had a structure that did not clearly communicate the distribution of group-rounds by seller count or the average sell period for each seller position.

Specific issues:
- Good/Bad state sub-rows under Total group-rounds added unnecessary row complexity
- Zero-seller group-rounds row was a single summary rather than a full breakdown by seller count (0-4)
- Avg sell period only reported the first seller's timing, omitting 2nd-4th seller positions

## Solution

Restructure Table 4 with the following changes:

1. **Remove** Good/Bad state sub-rows from the Total group-rounds block
2. **Replace** the Zero-seller row with Group-rounds by seller count (0, 1, 2, 3, 4 sellers)
3. **Replace** Avg sell period (first seller only) with Avg sell period by seller position (1st-4th)
4. Use competition ranking (`min_rank`) for tie-breaking seller position ordering
5. Preserve the Avg sellers per group-round and Avg sell period blocks unchanged

## Files Changed

| File | Description |
|------|-------------|
| `analysis/analysis/summary_statistics.R` | Updated table construction logic: added `stat_n_seller_groups`, `stat_avg_nth_seller_period`, `build_seller_count_block`, `build_seller_position_block`; removed `stat_zero_seller_groups`, `stat_avg_first_seller_period` |
| `analysis/tests/test_summary_statistics.py` | Updated tests for new 18-row structure (was 15): added seller count rows, position rows, header tests, sum integrity test |
| `analysis/output/tables/summary_seller_counts.tex` | Regenerated LaTeX output |

## Testing

- 9 `TestSellerCountsTable` tests pass, cross-validating every cell against raw panel data
- `test_seller_counts_sum_to_total` verifies 0-4 seller rows sum to 180 per column
- `test_seller_position_period_rows` independently computes seller positions using pandas `rank(method="min")`
- R and Python standards checks pass
- LaTeX compiles cleanly
