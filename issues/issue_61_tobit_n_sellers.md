# Issue #61: Add Tobit Regression for Number of Sellers Per Group-Round

## Problem Statement

We need a group-round level regression to model the number of sellers (0-4) per group-round. The dependent variable is bounded at 0 and 4 with substantial left-censoring (47% of observations have zero sellers), making a tobit model appropriate. This complements the existing individual-level selling regressions by analyzing selling at the group-round level.

## Identification Strategy

Tobit regression with left bound 0 and upper bound 4, estimated via `AER::tobit()`. Cluster-robust standard errors via `sandwich::vcovCL()` clustered at the group level (`global_group_id`, 96 unique groups). Three progressively richer specifications isolate the effects of state, treatment, segment learning, and within-segment round trends.

## Changes Made

### Regression Analysis
- **Script**: `analysis/analysis/tobit_n_sellers.R`
  - **Input**: `datastore/derived/group_round_timing.csv` (720 group-round observations)
  - **Output**: `analysis/output/tables/tobit_n_sellers.tex`
  - **Model 1**: `n_sellers ~ bad_state + treatment`
  - **Model 2**: `n_sellers ~ bad_state + treatment + segment_num`
  - **Model 3**: `n_sellers ~ bad_state + treatment + segment_num + round_num`

### Paper Integration
- **File**: `analysis/paper/main.tex`
  - Merged latest Overleaf edits (Market Model, Experimental Implementation sections)
  - Added tobit table to section 4.2.1 (Group-Round-level regression analysis)

### Data Validation Tests
- **File**: `analysis/tests/test_build_group_round_timing_dataset.py`
  - Added 17 integration tests verifying all 6 columns used by the tobit regression (`n_sellers`, `state`, `treatment`, `segment_num`, `round_num`, `global_group_id`) against raw oTree exports

## Key Results

- **Bad state**: Strongly positive (~0.98, p<0.001) — more sellers when true asset value is low
- **Treatment 2**: Positive (~0.50, p<0.05) — more selling under averaged pricing
- **Segments 3 & 4**: Strongly negative (p<0.001) — fewer sellers in later segments
- **Round**: Negative (-0.074, p<0.01) — within-segment learning reduces selling
