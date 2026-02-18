# Issue 40: Replace Section 4.2 box plots with individual-level CI difference plots

## Summary
Replace box plots and violin plots in Section 4.2 with confidence interval difference plots showing the difference in mean trait scores between first sellers and non-first sellers. Collapse the unit of observation to the individual level (N=95) rather than group-round level. Add robustness analysis with finer first-seller frequency thresholds (0, 1-2, 3+ times).

## Inputs
- `datastore/derived/first_seller_round_data.csv` - Round-level first seller data
- `datastore/derived/survey_traits.csv` - Individual-level personality trait scores

## Outputs
- `analysis/output/plots/first_seller_trait_ci_diff.pdf` - CI difference plot (first seller minus non-first seller, 95% CI)
  - Created by: `analysis/analysis/visualize_first_seller_traits.R`
- `analysis/output/plots/first_seller_trait_robustness.pdf` - Robustness plot with 0/1-2/3+ frequency groups
  - Created by: `analysis/analysis/visualize_first_seller_traits.R`
- `analysis/output/tables/first_seller_trait_comparisons.tex` - Updated comparison table (individual-level, N=95)
  - Created by: `analysis/analysis/first_seller_descriptive_stats.R`

## Changes
- **`analysis/analysis/visualize_first_seller_traits.R`** - Rewrote to produce CI difference plot and robustness plot instead of box plots, violin plots, and quartile bar charts. Uses individual-level data (1+ times = first seller).
- **`analysis/analysis/first_seller_descriptive_stats.R`** - Updated to compute individual-level descriptive statistics and t-tests rather than group-round level.
- **`analysis/paper/main.tex`** - Updated Figure/Table references in Section 4.2 to use new CI difference plot. Added Appendix D for robustness plot with frequency thresholds.
- **`analysis/tests/helpers_first_seller.py`** - New test helper module with synthetic data builders for first seller tests.
- **`analysis/tests/test_visualize_first_seller_traits.py`** - Expanded unit tests covering CI computation, robustness grouping, and plot output validation.
- **`pyproject.toml`** / **`uv.lock`** - Added new dependency.

## Deleted
- `analysis/output/plots/first_seller_trait_boxplots.pdf` - Replaced by CI difference plot
- `analysis/output/plots/first_seller_trait_violins.pdf` - Replaced by CI difference plot
- `analysis/output/plots/first_seller_rate_by_quartile.pdf` - Replaced by robustness plot
