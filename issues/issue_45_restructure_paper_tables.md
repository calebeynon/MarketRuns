# Issue 45: Restructure Paper Tables, Figures, and Regression Output

## Summary
Reorganize the paper's summary statistics, figures, and regression tables. Combine demographic tables, create treatment-by-chat outcome table, reorder and expand regression output, and redistribute content between main paper and appendix. Ensure all trait and emotion controls are both included in regressions and displayed in appendix tables.

## Changes Made

### Summary Statistics
- Combined `summary_demographics` and `summary_traits` into single table `summary_demographics_traits.tex`; removed Diff (p) column
- New sell rate table `summary_sell_rates.tex` showing sell rates, avg sell period, and avg sell price by Treatment x Chat x State
- `first_seller_trait_comparisons` moved to Appendix E with dynamic sample sizes

### Figures
- Promoted robustness plot (`first_seller_trait_robustness.pdf`) to Figure 1 in main paper
- Moved CI difference plot (`first_seller_trait_ci_diff.pdf`) to Appendix D

### Regressions — Display Changes (No Specification Changes)
- Ordinal logit (Table 3): AME-only two-column minipage layout with all 22 controls displayed; centered on page with fit statistics contained
- LPM and logit tables: swapped Panel B/C so First Sellers appear before Second Sellers
- Appendix A (LPM full) and Appendix B (logit full): Panel A now displays all 9 emotions and all 7 personality traits (previously hidden by display bug in `get_panel_vars()`)
- Appendix C (treatment-period interactions): converted to longtable for multi-page flow
- Added SE clustering notes to compact LPM and logit tables

### Scripts Modified
- `analysis/analysis/ordinal_logit_selling_position.R` — rewrote table builder for two-column AME layout; centered and compacted to fit page
- `analysis/analysis/unified_selling_logit.R` — fixed `get_panel_vars()` to use `ALL_PERSON_VARS` for full table Panel A; added clustering note to compact footer
- `analysis/analysis/unified_selling_regression_landscape.R` — added clustering note to compact footer
- `analysis/analysis/summary_statistics.R` — combined demographics/traits table; new sell rates table

### Outputs Changed
- `analysis/output/tables/ordinal_logit_selling_position.tex` — new centered layout
- `analysis/output/tables/unified_selling_logit.tex` / `unified_selling_logit_full.tex` — clustering note; full covariate display in Panel A
- `analysis/output/tables/unified_selling_regression.tex` — clustering note added
- `analysis/output/tables/selling_timing_treatment_interactions.tex` — tabular → longtable
- `analysis/output/tables/summary_demographics_traits.tex` — new combined table
- `analysis/output/tables/summary_sell_rates.tex` — new sell rate table
- `analysis/paper/main.tex` — restructured sections, appendix reordering

## Key Decisions
- All regression specifications are unchanged — every change is display/formatting only
- All regressions already estimated all 16 covariates (9 emotions + 7 traits); the display bug only affected which rows were shown in Panel A of the full appendix tables
- FE models (2) cluster SEs at `global_group_id` (session × segment × group); RE models (1, 3) use player random intercepts

## Verification
- Git diff confirms all previously-reported coefficients, SEs, and fit statistics are byte-for-byte identical
- New rows in full tables come from the same AME computation that was already running
- All 12 unit tests pass

## Related Issues
- Closes #45
