# Issue 68: Change logit column 2 from FE to RE with treatment dummy

## Summary
In Appendix B's logit robustness check table (`unified_selling_logit`), column 2 used a fixed effects logit (`feglm` with `player_id` absorbed) across all three panels. This was changed to a random effects logit (`glmer` with `(1 | player_id)`) so that the `treatment` dummy variable can be estimated (previously collinear with individual fixed effects).

## Background
The `treatment` variable was omitted from column 2 because it is collinear with individual fixed effects. Switching to random effects allows its inclusion alongside `age` and `gender_female`, which were also previously unestimable with FE.

## Changes Made
- Panel A/B/C m2 functions: `feglm(... | player_id)` -> `glmer(... + treatment + age + gender_female + (1 | player_id))`
- Main table builder: removed `library(fixest)`, `extract_ame_fixest()`, fixest dispatch logic
- Column header: "FE Logit" -> "RE Logit"
- Footer notes: updated to "All columns: random-intercept logit (glmer)."
- Tests: removed delta method test class (dead code), updated observation counts and header assertions

## Outputs
- `analysis/analysis/unified_selling_logit.R` + panel scripts + `datastore/derived/emotions_traits_selling_dataset.csv` -> `analysis/output/tables/unified_selling_logit.tex`
- `analysis/analysis/unified_selling_logit.R` + panel scripts + `datastore/derived/emotions_traits_selling_dataset.csv` -> `analysis/output/tables/unified_selling_logit_full.tex`

## Testing
- All 16 non-slow tests pass (`uv run pytest analysis/tests/test_unified_logit_regression.py -v -k "not slow"`)
- AME sign agreement with LPM confirmed for all panels
- Exact observation counts verified: Panel A M2: 13,590; Panel B M2: 1,183; Panel C M2: 619
- Paper compiles successfully with updated tables
- R and Python code standards checks pass on all modified files
