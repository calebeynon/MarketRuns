# Issue 80: Restructure Table 6 and Appendix F to 4-Column Longtable Format

## Problem

Table 6 (ranked-order logit) and Appendix F (CLM robustness check) used a 2-column minipage layout where each column included all 9 emotion variables (8 discrete + valence) together. This format was inconsistent with Table 7 (Cox survival regression), which separates discrete emotions from valence into distinct columns.

## Solution

Restructured both tables from 2-column minipage to 4-column longtable format:
- (1) Full Sample, All Emotions (8 discrete emotions, no valence)
- (2) Full Sample, Valence (valence only, no discrete emotions)
- (3) Sellers Only, All Emotions
- (4) Sellers Only, Valence

Also moved the unified selling regression (former Table 8) to Appendix C to fill a gap in appendix lettering.

## Scripts

| Script | Input | Output |
|--------|-------|--------|
| `analysis/analysis/ro_logit_selling_position.R` | `datastore/derived/ordinal_selling_position.csv` | `analysis/output/tables/ro_logit_selling_position.tex` |
| `analysis/analysis/ordinal_logit_selling_position.R` | `datastore/derived/ordinal_selling_position.csv` | `analysis/output/tables/ordinal_logit_selling_position.tex` |

## Files Modified

- `analysis/analysis/ro_logit_selling_position.R` - Split 2 models into 4 (Full/Sellers x Emotions/Valence), minipage to longtable
- `analysis/analysis/ordinal_logit_selling_position.R` - Same 2-to-4 model split for CLM, minipage to longtable
- `analysis/output/tables/ro_logit_selling_position.tex` - Generated 4-column longtable output
- `analysis/output/tables/ordinal_logit_selling_position.tex` - Generated 4-column longtable output
- `analysis/paper/main.tex` - Removed table float wrappers, moved unified selling regression to Appendix C
