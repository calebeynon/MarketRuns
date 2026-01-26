# Issue #11: Holdout Liquidation Analysis

## Problem Statement

In Treatment 1 State 0 rounds, participants who hold their asset until the end of the round receive a random liquidation payoff (2, 4, 6, or 8 ECU). This analysis tests whether holdouts who received lower payoffs are more likely to sell in the following round.

## Identification Strategy

Within a group-round, holdouts receive randomly assigned liquidation payoffs. We use group-by-round fixed effects to compare only within the same group-round, isolating the random variation in payoffs.

## Changes Made

### Derived Datasets
1. **Extended individual period dataset** - Adds round-level columns (`round_payoff`, `sold_in_round`) to the existing individual period dataset
2. **Holdout next-round analysis dataset** - Filters to holdouts in Treatment 1 State 0 rounds and links to next-round selling behavior

### Regression Analysis
- Linear probability model testing whether `sold_next_round` depends on `round_payoff`
- Uses group-by-round fixed effects to exploit within-group random assignment
- Controls for prior sales in the segment

## Outputs

- **Script**: `analysis/derived/build_individual_period_dataset_extended.py`
  - **Input**: `datastore/derived/individual_period_dataset.csv`, raw session data
  - **Output**: `datastore/derived/individual_period_dataset_extended.csv`

- **Script**: `analysis/derived/build_holdout_next_round_dataset.py`
  - **Input**: `datastore/derived/individual_period_dataset_extended.csv`
  - **Output**: `datastore/derived/holdout_next_round_analysis.csv`

- **Script**: `analysis/analysis/holdout_liquidation_regression.R`
  - **Input**: `datastore/derived/holdout_next_round_analysis.csv`
  - **Output**: `analysis/output/analysis/holdout_liquidation_regression.tex`

- **Script**: `analysis/visualize_holdout_payoff_coefficients.R`
  - **Input**: `datastore/derived/holdout_next_round_analysis.csv`
  - **Output**: `analysis/output/plots/holdout_payoff_coefficients.png`

## Testing

- Unit tests for both derived dataset scripts covering edge cases
- Verification script for extended dataset validation
- All tests pass
