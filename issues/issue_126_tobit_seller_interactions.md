# Issue #126: Tobit Model for Number of Sellers with Treatment2 x Segment Interactions

## Problem Statement

The group-round Tobit model from issue #61 estimates a single average Treatment 2 effect on the number of sellers. We want to test whether the average-pricing treatment (T2) deviation strengthens in the chat segments (3-4). This requires adding `treatment2 * segment_num` interaction terms to the Tobit specification so the treatment effect can vary by segment.

## Identification Strategy

Tobit regression with left bound 0 and upper bound 4, estimated via `AER::tobit()`, matching the censoring of `n_sellers` (0-4). Cluster-robust standard errors via `sandwich::vcovCL()` clustered at the group level (`global_group_id`). The `treatment2 x segment_num` interactions capture segment-specific deviations of the T2 effect relative to the baseline segment.

## Changes Made

### Regression Analysis
- **Script**: `analysis/analysis/tobit_n_sellers_interactions.R`
  - **Input**: `datastore/derived/group_round_timing.csv`
  - **Output**: `analysis/output/tables/tobit_n_sellers_interactions.tex`
  - Regressors: `bad_state`, `treatment2`, `segment_num` indicators, `treatment2:segment_num` interactions, and `round_num`
  - Reports cell counts and per-model summaries; builds a publication-ready LaTeX table with cluster-robust SEs

## Expected Output

A LaTeX regression table (`tobit_n_sellers_interactions.tex`) reporting Tobit coefficients for the bad-state indicator, Treatment 2, segment indicators, the T2 x segment interactions, and round, with cluster-robust standard errors at the group level.

## Related Issues
- Builds on: #61 (original group-round Tobit model for number of sellers)
