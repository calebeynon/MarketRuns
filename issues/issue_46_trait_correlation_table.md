# Issue 46: Add Trait Correlation Table with Significance Stars

## Problem
The paper's summary statistics section lacked a pairwise correlation table for the 8 personality trait measures, making it difficult for readers to assess collinearity and relationships between traits.

## Proposed Solution
Create a lower-triangle Pearson correlation matrix with significance stars and integrate it into the paper between the demographics/traits table and market outcomes table.

## Changes Made

### New files
- **`analysis/analysis/trait_correlations.R`** — Computes pairwise Pearson correlations across all 8 traits using `cor.test()`, outputs lower-triangle matrix with significance stars to LaTeX
- **`analysis/tests/test_trait_correlations.py`** — Cross-validates all 28 lower-triangle correlations and significance stars against independently computed values from `scipy.stats.pearsonr`

### Modified files
- **`analysis/derived/build_survey_traits_dataset.py`** — Renamed `risky_investment` to `risk_tolerance` (merged from main)
- **`analysis/analysis/summary_statistics.R`** — Updated TRAITS/TRAIT_LABELS to use `risk_tolerance`
- **`analysis/analysis/visualize_first_seller_traits.R`** — Added `scales = "free_x"` to facet plot so risk tolerance (0-20) doesn't compress CIs of other traits (1-7, 1-4)
- **`analysis/paper/main.tex`** — Integrated trait correlation table; removed `\resizebox` wrapper that caused rendering errors; adopted main's restructured summary tables

## Outputs
- Script: `trait_correlations.R` → Output: `analysis/output/tables/trait_correlations.tex`
- Script: `visualize_first_seller_traits.R` → Output: `analysis/output/plots/first_seller_trait_robustness.pdf`, `first_seller_trait_ci_diff.pdf`

## Notes
- All 8 traits have 95 complete observations (no missing data)
- Significance levels: *** p<0.01, ** p<0.05, * p<0.1
- The `risky_investment` → `risk_tolerance` rename was adopted from main branch (issue #48)
- No borderline p-values near significance thresholds between R and Python implementations
