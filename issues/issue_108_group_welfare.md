# Issue #108: Compute group-round welfare and merge into individual-level dataset

## Problem
The analysis pipeline lacked a welfare metric at the group-round level. Welfare measures allocative efficiency: total group earnings as a fraction of maximum possible earnings (80 in state=1). This metric is needed for downstream analysis of how group communication and trading behavior affect overall market efficiency.

## Approach
Welfare is computed using a closed-form formula based on state and number of sellers:
- **State = 0**: welfare = 1.0 (no trade is first-best)
- **State = 1**: welfare = ((4 - n_sellers) × 20 + Σ prices) / 80, where prices = [8, 6, 4, 2]

The formula is treatment-invariant (TR2 redistributes among sellers but doesn't change totals). Cross-validated against actual summed `round_payoff` values from raw data — exact match for all 720 group-rounds.

## Changes Made
- **Modified** `analysis/derived/build_group_round_timing_dataset.py`: Added `PRICES` constant, `compute_welfare()` function with input validation, welfare key in `build_group_round_record()`
- **Created** `analysis/derived/build_welfare_dataset.py`: Reads welfare from `group_round_timing.csv`, cross-validates against actual payoff sums from `individual_period_dataset_extended.csv`, outputs standalone `group_round_welfare.csv` (720 rows), merges welfare into `emotions_traits_selling_dataset.csv`
- **Modified** `analysis/tests/test_build_group_round_timing_dataset.py`: Added 17 welfare tests (unit + integration)
- **Created** `analysis/tests/test_build_welfare_dataset.py`: 13 tests for welfare dataset and merge logic

## Outputs
- `datastore/derived/group_round_timing.csv` → now 22 columns (welfare added)
- `datastore/derived/group_round_welfare.csv` → 720 rows, 5 columns (session, segment_num, round_num, group_id, welfare)
- `datastore/derived/emotions_traits_selling_dataset.csv` → 16,128 rows, 35 columns (welfare merged)

## Tests
All 58 tests pass (30 new welfare tests + 28 pre-existing).
