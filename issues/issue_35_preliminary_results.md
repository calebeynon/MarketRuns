# Issue #35: Add preliminary results section with summary statistics and regression tables

**Status:** In Progress
**Date Created:** 2026-02-12
**Date Completed:** 2026-02-16

## Summary

Build the preliminary Results section in `main.tex` with reorganized regression tables and updated personality trait visualizations.

## Objectives

1. Reorganize Results section to fit regression tables on 2 pages
2. Fix personality trait boxplot y-axis scales (group by measurement scale)
3. Simplify trait comparison table with significance asterisks
4. Remove redundant Treatment Effects subsection

## Implementation

### Scripts Modified
- `analysis/analysis/visualize_first_seller_traits.R` — Split boxplots into two panels by measurement scale (1-7 for BFI/impulsivity, 1-4 for state anxiety) using cowplot
- `analysis/analysis/first_seller_descriptive_stats.R` — Replaced t-stat and p-value columns with significance asterisks on Difference column
- `analysis/analysis/ordinal_logit_selling_position.R` — Compact tabular formatting for side-by-side layout
- `analysis/analysis/holdout_liquidation_regression.R` — Matched compact formatting

### Output Files
- `analysis/output/tables/first_seller_trait_comparisons.tex` — Simplified 4-column table with asterisks
- `analysis/output/tables/ordinal_logit_selling_position.tex` — Compact format
- `analysis/output/tables/holdout_liquidation_regression.tex` — Compact format
- `analysis/output/plots/first_seller_trait_boxplots.pdf` — Fixed y-axis grouping

### Paper Changes (`analysis/paper/main.tex`)
- Restructured Results into subsections: Personality Traits, Determinants of Selling Behavior
- Ordinal logit and holdout regression side-by-side on one page
- Removed Treatment Effects subsection
- Added Appendix C for selling_timing_treatment_interactions

### Tests Added
- `analysis/tests/test_visualize_first_seller_traits.py` — 8 tests covering data validation, trait ranges, R script execution, output PDF generation

## Testing

- [x] Paper compiles locally with `latexmk`
- [x] All 8 visualization tests pass
- [x] Regression tables fit on 2 pages
- [x] Boxplot y-axes grouped by measurement scale

## Related Issues

- Closes #35
- Related to #11 (holdout liquidation)
- Related to #32/#34 (ordinal logit selling position)
