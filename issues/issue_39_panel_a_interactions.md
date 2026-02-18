# Issue 39: Add Cumulative Interaction Terms to Panel A Regressions

## Summary
Add previous-period selling dummies and their interactions with cumulative sales dummies to Panel A ("All Participants") in the unified selling regression tables (LPM and logit). Only Panel A is modified; Panels B and C stay unchanged.

## Changes from Original Table

### Model Specification (Panel A only)
- Added 3 cumulative prior sales dummies (`dummy_1_cum`, `dummy_2_cum`, `dummy_3_cum`)
- Added 6 cumulative × previous-period interaction terms (`int_1_1` through `int_3_3`)
- Previous-period main effects are **not** included due to perfect collinearity with the interaction terms
- Panels B and C are completely unchanged in model specification

### Table Layout
- Compact table changed from a single vertical longtable to a **two-column portrait layout**
  - Panel A on the left (54% width)
  - Panels B + C stacked on the right (44% width)
- Uses `\singlespacing`, `\scriptsize`, `[t]` tabular alignment, tight `\tabcolsep`
- Variable labels shortened for compact table (e.g., "Exactly 1 prior sale" → "1 prior sale", interactions as "2 prior × 1 prev.")

### Code Refactoring
- Shared helpers extracted to `selling_regression_helpers.R` to eliminate DRY violations between LPM and logit scripts
- `unified_selling_regression.R` reduced from 290 → 191 LOC
- `unified_selling_logit.R` reduced from 345 → 231 LOC

## Scripts and Outputs

| Script | Input | Output |
|--------|-------|--------|
| `unified_selling_regression.R` | `emotions_traits_selling_dataset.csv` | `unified_selling_regression.tex`, `unified_selling_regression_full.tex` |
| `unified_selling_logit.R` | `emotions_traits_selling_dataset.csv` | `unified_selling_logit.tex`, `unified_selling_logit_full.tex` |
| `selling_regression_helpers.R` | — | Sourced by above scripts |
| `unified_selling_regression_landscape.R` | — | Sourced by LPM script for compact layout |

## Files Modified
- `analysis/analysis/selling_regression_helpers.R` (new)
- `analysis/analysis/unified_selling_regression_landscape.R` (new)
- `analysis/analysis/unified_selling_regression.R`
- `analysis/analysis/unified_selling_regression_panel_a.R`
- `analysis/analysis/unified_selling_logit.R`
- `analysis/analysis/unified_selling_logit_panel_a.R`
- `analysis/output/tables/unified_selling_regression.tex`
- `analysis/output/tables/unified_selling_regression_full.tex`
- `analysis/output/tables/unified_selling_logit.tex`
- `analysis/output/tables/unified_selling_logit_full.tex`
- `analysis/paper/main.tex`

## Testing
- `analysis/tests/test_panel_a_interactions.py` — 16 tests for interaction term algebra and data integrity
- `analysis/tests/test_raw_data_validation.py` — 25 tests validating dummies against raw data
