# Issue #4: Individual-Level Cascade Analysis with Selling Period Regressions

## Problem Statement
Need to analyze whether participants are more likely to sell when they observe prior sales in their group, testing for information cascade behavior at the individual-period level.

## Solution Overview
Implemented multi-level cascade analysis with both group-round and individual-period level regressions to test cascade behavior from different angles: first-sale timing at the group level and selling decisions at the individual level.

## Key Changes

### Group-Round Level Analysis
- Created `create_first_sale_dataset.R` to build group-round level dataset (1,440 observations)
- Implemented `regression_first_sale.R` with clustered standard errors:
  - OLS regression testing whether first sale happens earlier when chat is enabled
  - Controls: round, segment, treatment
  - Results: Treatment 2 shows earlier first sales (-0.36 periods, p=0.073)

### Individual-Period Level Analysis
- Created `create_individual_period_dataset.R` to build player-period level dataset (13,728 observations)
- Implemented `selling_period_regression.R` with two LPM specifications:
  - Model 1: `n_sales_earlier` (count of sales in periods 1 to t-2)
  - Model 2: `sale_prev_period` (binary indicator for any sale in period t-1)
- Implemented `selling_period_regression_dummies.R` with random effects models:
  - Cumulative dummies: D1, D2, D3 for exactly 1, 2, or 3 prior sales
  - Previous period dummies: D1, D2, D3 for exactly 1, 2, or 3 sales in t-1
- Created `market_data.py` parser module for data validation
- Added comprehensive validation scripts:
  - `validate_regression_data.py` - validates against parser and raw CSVs
  - `test_regression_dummy_variables.py` - 15 unit tests for dummy variable logic
- Generated LaTeX tables for main results and appendix (with period dummies)

## Expected Outcomes
- Individual-period level dataset validated against raw data
- Regression results showing cascade behavior operates through accumulated information
- Evidence of non-linear threshold effects (significant at 2+ prior sales)
- Publication-ready LaTeX tables

## Testing
- Parser validation: 200 sampled rows verified against market_data.py
- Raw CSV validation: 200 sampled rows verified against oTree exports
- Timing logic validation: All 13,728 rows verified for sale_prev_period and n_sales_earlier
- Dummy variable tests: 15 unit tests all passing
- Edge case validation: Period 1, period 2, no-sales scenarios tested

## Key Findings

### Group-Round Level
- Treatment 2 (chat enabled) shows earlier first sales: -0.36 periods
- Suggests chat may facilitate faster cascade initiation

### Individual-Period Level
- Cumulative earlier sales significant: -0.026*** per additional sale
- Immediate prior-period sale: not significant (0.019)
- Non-linear effect: D2 = -0.035***, D3 = -0.081***
- Cascade operates through accumulated information, not immediate panic
- Threshold effect emerges only after 2+ prior sales
