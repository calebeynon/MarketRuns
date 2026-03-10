# Issue 95: Add valence-only emotion regressions for Tables 6 & 7

## Problem
Tables 6 (rank-ordered logit) and 7 (Cox survival) include all 9 emotion measures
as covariates. Since Valence may be a linear combination of the other emotions,
we need alternate specifications to determine whether Valence alone captures the
same information, or whether the discrete emotions carry independent signal.

## Solution
Add two alternate specifications for each of Tables 6 and 7:
1. **Valence-only**: Regressions using only the valence emotion measure
2. **No-valence**: Regressions excluding valence, keeping 8 discrete emotions

These become Appendix G (valence-only) and Appendix H (no-valence) in the paper.

## Key Finding
Valence alone does not capture the same information as the discrete emotions.
Dropping valence barely changes fit (LL delta ~0.4–1.4), while dropping discrete
emotions hurts substantially more (LL delta ~0.7–5.7). Joy shows collinearity
with Valence but no significant sign changes across specifications.

## Files Changed

### New R scripts (8)
- `analysis/analysis/ro_logit_selling_position_valence_only.R` → `ro_logit_selling_position_valence_only.tex`
- `analysis/analysis/ro_logit_selling_position_no_valence.R` → `ro_logit_selling_position_no_valence.tex`
- `analysis/analysis/cox_survival_regression_valence_only.R` → `cox_survival_regression_valence_only.tex`
- `analysis/analysis/cox_survival_regression_no_valence.R` → `cox_survival_regression_no_valence.tex`
- `analysis/analysis/cox_survival_panel_a_valence_only.R`
- `analysis/analysis/cox_survival_panel_a_no_valence.R`
- `analysis/analysis/cox_survival_panel_b_valence_only.R`
- `analysis/analysis/cox_survival_panel_b_no_valence.R`

### Generated output (4)
- `analysis/output/tables/ro_logit_selling_position_valence_only.tex`
- `analysis/output/tables/ro_logit_selling_position_no_valence.tex`
- `analysis/output/tables/cox_survival_regression_valence_only.tex`
- `analysis/output/tables/cox_survival_regression_no_valence.tex`

### Modified
- `analysis/paper/main.tex` — Added Appendix G (valence-only) and Appendix H (no-valence)

### Tests
- `analysis/tests/test_valence_only_output.py` — 96 validation tests
