# Issue #8: Analyze Selling Timing Differences Across Treatments

## Description
Analyze whether selling timing differs between treatments over time, using treatment × period interactions.

## Approach
1. Create dataset builders for timing analysis at segment-group level
2. Implement regression with treatment × period interactions
3. Use 96 clusters at segment-group level (groups reshuffle across segments)
4. Create diff-in-diff style visualization

## Model Specification
```r
sold ~ treatment * i(period) + signal + round + i(segment)
```
- Clustering: segment × group (96 clusters)
- Reference: Treatment 2 as baseline
- Sample: 13,728 observations (already_sold == 0)

## Files Created
- `analysis/derived/build_group_round_timing_dataset.py`
- `analysis/derived/build_individual_round_dataset.py`
- `analysis/analysis/selling_timing_treatment_interactions.R`
- `analysis/analysis/visualize_selling_timing.R`

## Results
| Test | Statistic | p-value |
|------|-----------|---------|
| Wald (Treatment 2 × Period) | F = 1.28 | 0.216 |
| Treatment 1 main effect | β = 0.027 | 0.567 |

## Conclusion
Treatment × period interactions are **not jointly significant**. Selling timing patterns do not differ substantially between treatments.
