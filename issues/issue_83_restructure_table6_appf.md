# Issue 83: Restructure Table 6 and Appendix F

## Summary

Restructure Table 6 (ranked-order logit / Cox PH) and Appendix F (ordinal logit)
from an "All Emotions vs Valence" column layout to a "No Traits vs With Traits"
column layout. Also aligns Table 7 formatting, fixes a phantom rank bug in the
data pipeline, and removes Figures 2-3 from the paper.

## Motivation

The previous column structure compared emotion specifications (all discrete
emotions vs. valence composite). The new structure instead compares models that
exclude personality traits (columns 1 & 3) against models that include them
(columns 2 & 4), which better highlights the marginal contribution of trait
covariates.

## Changes

### Analysis scripts
- `analysis/analysis/ro_logit_selling_position.R` — Replace
  `PREDICTORS_EMOTIONS` / `PREDICTORS_VALENCE` with `PREDICTORS_NO_TRAITS` /
  `PREDICTORS_WITH_TRAITS`; blank trait cells in No Traits columns; reorder
  table sections (emotions/valence → demographics → traits).
- `analysis/analysis/ordinal_logit_selling_position.R` — Same restructure for
  Appendix F.
- `analysis/analysis/cox_survival_regression.R` — Reorder personality traits
  to match Tables 6/F; move `signal` above emotions in display order.

### Data pipeline
- `analysis/derived/build_ordinal_selling_position.py` — Move `merge_traits`
  before `compute_selling_ranks` to fix phantom sell ranks caused by player C
  being dropped after ranks were computed. Add phantom rank validation.

### Tests
- `analysis/tests/test_ro_logit_data_validation.py` — Update predictor lists to
  match the new No Traits / With Traits split (11 no-traits, 19 with-traits).

### Paper
- `analysis/paper/main.tex` — Comment out Figures 2 and 3 (kept for
  presentations).

## Outputs

| Output | Script |
|---|---|
| `analysis/output/tables/ro_logit_selling_position.tex` | `ro_logit_selling_position.R` |
| `analysis/output/tables/ordinal_logit_selling_position.tex` | `ordinal_logit_selling_position.R` |
| `analysis/output/tables/cox_survival_regression.tex` | `cox_survival_regression.R` |
