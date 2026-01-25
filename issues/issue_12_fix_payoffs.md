# Issue #12: Fix round_payoffs extraction in market_data.py

## Description
The `round_payoffs` extraction in `market_data.py` was incorrectly reading from period 1 instead of the last period of each round. In oTree, payoff values are updated progressively during each round, and the final payoff is only available in the last period.

## Problem
When accessing `Round.round_payoffs[player_label]`, the function returned intermediate values from period 1 rather than the finalized payoff from the last period of each round.

Example: Player C, Round 1, Segment chat_noavg returned 6.0 (period 1 value) instead of 4.0 (correct final value from period 3).

## Approach
1. Track the maximum oTree period number for each market round during parsing
2. After processing all periods, extract `round_X_payoff` from the last period's columns
3. Add comprehensive tests verifying payoffs against raw CSV data

## Files Modified
- `analysis/market_data.py` - Fixed payoff extraction logic
  - Added `round_to_last_otree_period` dictionary to track max oTree period per round
  - Changed extraction to read from `{segment_name}.{last_otree_period}.player.round_{round_num}_payoff`

## Files Created
- `analysis/tests/test_round_payoffs.py` - 8 tests verifying payoffs against raw CSV

## Results
| Metric | Value |
|--------|-------|
| Existing tests | 75 passed |
| New tests | 8 passed |
| Total | 83 passed |

## Conclusion
Payoffs now correctly read from the last period of each round. Player C, Round 1, Segment chat_noavg correctly returns 4.0.
