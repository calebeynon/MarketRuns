# Issue #31: Logit Robustness Tables with Average Marginal Effects

## Description

The unified regression table (LPM) from issue #31 uses linear probability models. As a robustness check, we need logit versions of all 9 models (3 panels x 3 columns) reporting average marginal effects (AMEs) instead of raw coefficients.

## Motivation

LPM can overstate marginal effects when baseline probabilities are low (e.g., Panel C first sellers). Logit models correctly attenuate effects in the tails via the sigmoid transformation. AMEs provide directly comparable probability-scale estimates.

## Implementation

### New Files

**`analysis/analysis/unified_selling_logit_panel_a.R`:**
- Panel A (All Participants) logit models
- M1/M3: `lme4::glmer` with random intercept on `player_id`
- M2: `fixest::feglm` with individual FE, clustered by `global_group_id`

**`analysis/analysis/unified_selling_logit_panel_b.R`:**
- Panel B (Second Sellers) logit models
- Reuses `identify_second_sellers()` and `create_prev_period_dummy()` from LPM panel B

**`analysis/analysis/unified_selling_logit_panel_c.R`:**
- Panel C (First Sellers) logit models
- Reuses `subset_first_sellers()` from LPM panel C

**`analysis/analysis/unified_selling_logit.R`:**
- Main orchestrator: AME extraction, fit statistics, LaTeX table builder
- glmer AMEs via `marginaleffects::avg_slopes()`
- feglm AMEs via manual delta method (avg_slopes cannot handle absorbed FE)
- Reports McFadden pseudo R-squared and log-likelihood

**`analysis/tests/test_unified_logit_regression.py`:**
- 20 pytest tests: convergence, AME sign agreement with LPM, observation counts, delta method validation, table structure

### Files Modified

**`analysis/paper/main.tex`:**
- Added Appendix B (Logit Robustness Check) with `\input{unified_selling_logit_full}`
- Renumbered existing Appendix B (Instructions) to Appendix C

## Expected Outputs

### Script: `analysis/analysis/unified_selling_logit.R`
- Input: `datastore/derived/emotions_traits_selling_dataset.csv`
- Outputs:
  - `analysis/output/tables/unified_selling_logit.tex` (compact, main body)
  - `analysis/output/tables/unified_selling_logit_full.tex` (full, appendix)

## Known Limitations

- Panel B glmer models (622 obs) have convergence warnings; RE model SEs are unreliable
- Panel C trait significance weakened vs LPM (expected: logit correctly attenuates low base-rate effects)
