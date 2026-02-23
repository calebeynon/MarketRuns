# Issue 69: Cox Proportional Hazards Survival Regression

## Summary
Add Cox proportional hazards survival regression tables to Appendix B as a robustness check alongside the existing logit tables.

## Motivation
The existing logit regressions (Appendix B.1) model the binary selling decision per period. Cox survival regression provides a complementary analysis by treating within-round period as the survival time axis and selling as the failure event. This tests whether the logit findings are robust to a different statistical framework that naturally accounts for the sequential timing of selling decisions.

## Changes
- `analysis/analysis/cox_survival_panel_a.R` - Panel A (all participants): 3 RE Cox models (cascade, cascade+emotions, cascade+traits)
- `analysis/analysis/cox_survival_panel_b.R` - Panel B (first sellers): 3 RE Cox models (controls, emotions, traits)
- `analysis/analysis/cox_survival_regression.R` - Main orchestration: data prep, hazard ratio extraction (delta method), longtable builder
- `analysis/output/tables/cox_survival_regression.tex` - Generated LaTeX longtable with hazard ratios
- `analysis/paper/main.tex` - Split Appendix B into B.1 (Logit) and B.2 (Cox); added cross-reference in Results section

## Expected Output
- `analysis/output/tables/cox_survival_regression.tex` - Cox survival regression results table with two panels and fit statistics

## Key Design Decisions
- Reports hazard ratios (not raw coefficients)
- All models use random-effects Cox via `coxme` with `(1 | player_id)`
- Period is the survival time axis, not a covariate
- Reuses shared helpers from `selling_regression_helpers.R` and `subset_first_sellers()` from existing panel scripts

## PR Type
Robustness Check
