# Issue 75: Move Cox Survival Model to Section 4.2.2 with First-Seller Subsample

## Problem
The Cox survival model was located in Appendix B.2 and used a misspecified `Surv()` call
(right-censored format instead of counting process format). The analysis also lacked a
first-seller subsample and did not split results by emotional valence.

## Solution
- Rewrote `cox_survival_panel_a.R` and `cox_survival_panel_b.R` for a 4-column valence split
  (All Emotions / Valence x All Sellers / First Sellers)
- Rewrote `cox_survival_regression.R` for flat 4-column table output
- Fixed critical `Surv()` specification bug: changed from `Surv(period, sold)` to counting
  process format `Surv(period_start, period, sold)` to prevent risk set inflation
- Moved the Cox table from Appendix B.2 to Section 4.2.2 in `main.tex`
- Wrote 50 tests in `test_cox_survival_data.py` validating data correctness, counting process
  structure, and cross-validation against raw oTree parser
- Improved table formatting to match Table 5 style
- Added "Viewing the Paper" section to `CLAUDE.md`

## Files Modified
- `analysis/analysis/cox_survival_panel_a.R` — 2-model valence split for All Sellers
- `analysis/analysis/cox_survival_panel_b.R` — 2-model valence split for First Sellers
- `analysis/analysis/cox_survival_regression.R` — main script, 4-column table builder
- `analysis/tests/test_cox_survival_data.py` — 50 tests (new file)
- `analysis/paper/main.tex` — moved table to Section 4.2.2, removed Appendix B.2
- `CLAUDE.md` — added "Viewing the Paper" section

## Outputs
- `analysis/output/tables/cox_survival_regression.tex` — Cox regression table (4 columns)
  - Input: `datastore/derived/emotions_traits_selling_dataset.csv`
  - Script: `analysis/analysis/cox_survival_regression.R`
