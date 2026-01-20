# Issue #4: Individual-Level Cascade Analysis with Selling Period Regressions

## Problem Statement
Need to analyze whether participants are more likely to sell when they observe prior sales in their group, testing for information cascade behavior at the individual-period level.

## Solution Overview
Implemented multi-level cascade analysis with both group-round and individual-period level regressions to test cascade behavior from different angles: first-sale timing at the group level and selling decisions at the individual level.

## Scripts and Output Mapping

| Script | Input | Output Table(s) |
|--------|-------|-----------------|
| `analysis/derived/build_first_sale_dataset.py` | Raw oTree CSVs | `datastore/derived/first_sale_data.csv` |
| `analysis/derived/build_individual_period_dataset.py` | Raw oTree CSVs | `datastore/derived/individual_period_dataset.csv` |
| `analysis/analysis/regression_first_sale.R` | `first_sale_data.csv` | `h2_regression_cluster.tex` |
| `analysis/analysis/selling_period_regression.R` | `individual_period_dataset.csv` | `selling_period_regression.tex` |
| `analysis/analysis/selling_period_regression_dummies.R` | `individual_period_dataset.csv` | `selling_period_regression_dummies.tex`, `selling_period_regression_dummies_appendix.tex` |
| `analysis/analysis/selling_period_regression_extended.R` | `individual_period_dataset.csv` | `selling_period_regression_first_sellers.tex`, `selling_period_regression_second_sellers.tex`, `selling_period_regression_second_sellers_interaction.tex`, `selling_period_regression_interactions.tex` |

## Key Changes

### Clustering Bug Fix (2026-01-19)
Fixed incorrect `global_group_id` construction in three R scripts. Groups are unique per segment, so clustering must include segment:

```r
# WRONG (24 clusters): session × group
df[, global_group_id := paste(session_id, group_id, sep = "_")]

# CORRECT (96 clusters): session × segment × group
df[, global_group_id := paste(session_id, segment, group_id, sep = "_")]
```

**Impact:** Only `selling_period_regression.tex` changed (uses clustered SEs). Other tables use random effects and were unaffected.

| Coefficient | Before Fix | After Fix |
|-------------|------------|-----------|
| treatment_2 | 0.017 (SE=0.010, p=0.103) | 0.018 (SE=0.009, p=0.043) |

### Group-Round Level Analysis
- Created `build_first_sale_dataset.py` to build group-round level dataset (720 observations)
- Implemented `regression_first_sale.R` with clustered standard errors
- Results: Treatment 2 shows higher probability of any sale (+12.2pp, p=0.052)

### Individual-Period Level Analysis
- Created `build_individual_period_dataset.py` to build player-period level dataset (13,728 observations after filtering)
- Implemented `selling_period_regression.R` with two LPM specifications:
  - Model 1: `n_sales_earlier` (count of sales in periods 1 to t-2)
  - Model 2: `sale_prev_period` (binary indicator for any sale in period t-1)
- Implemented `selling_period_regression_dummies.R` with random effects models:
  - Cumulative dummies: D1, D2, D3 for exactly 1, 2, or 3 prior sales
  - Previous period dummies: D1, D2, D3 for exactly 1, 2, or 3 sales in t-1
- Implemented `selling_period_regression_extended.R` with first/second seller analysis
- Added comprehensive validation scripts

## Testing
- Parser validation: 200 sampled rows verified against market_data.py
- Raw CSV validation: 200 sampled rows verified against oTree exports
- Timing logic validation: All 13,728 rows verified for sale_prev_period and n_sales_earlier
- Dummy variable tests: 15 unit tests all passing
- Edge case validation: Period 1, period 2, no-sales scenarios tested

## Key Findings

### Treatment Effect
- Treatment 2 participants are 1.6-2.6pp more likely to sell (p<0.05 after clustering fix)
- Effect operates through 3rd/4th sellers, not 1st/2nd sellers
- Interpretation: Participants dislike average pricing when they don't sell; prefer selling at $4 over waiting to share $3

### Cascade Behavior
- Cumulative earlier sales significant: -0.026*** per additional sale
- Immediate prior-period sale: marginally significant (0.019, p=0.062)
- Non-linear effect: D2 = -0.035***, D3 = -0.081***
- Cascade operates through accumulated information, not immediate panic
- Threshold effect emerges only after 2+ prior sales

### Second Sellers
- 30pp more likely to sell when first sale was in immediately previous period
- No treatment × timing interaction (p=0.86)
