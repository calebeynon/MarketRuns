# Issue #18: OLS Regressions for Emotions and Traits

## Problem Statement

Test whether facial emotions and personality traits predict selling probability in the market experiment, and whether these individual characteristics moderate cascade behavior (response to prior sales).

## Approach

Simplified the regression analysis to focus on theoretically motivated variables:

- **Emotions (2):** Fear, Anger
- **Traits (3):** State anxiety, Impulsivity, Conscientiousness
- **Cascade variables:** n_sales_earlier, sale_prev_period
- **Controls shown:** Signal, Period, Round, Segment, Treatment
- **Controls hidden:** Other AFFDEX emotions, other BFI-10 traits, age, gender

Key methodological choices:
- Use plm random effects (player_id index) instead of session fixed effects to allow treatment estimation
- Include all 10 interaction terms (5 emotion/trait vars × 2 cascade vars) in a single model
- Produce separate tables for first sellers and second sellers subsamples

## Outputs

| Output | Description |
|--------|-------------|
| `analysis/output/tables/selling_emotions_traits_full.tex` | Full sample with cascade interactions (N=13,590) |
| `analysis/output/tables/selling_emotions_traits_first.tex` | First sellers subsample (N=1,183) |
| `analysis/output/tables/selling_emotions_traits_second.tex` | Second sellers subsample (N=619) |

## Scripts

| Script | Input | Output |
|--------|-------|--------|
| `analysis/analysis/selling_emotions_traits_unified.R` | `datastore/derived/emotions_traits_selling_dataset.csv` | 3 LaTeX tables |

## Data Dependencies

- `datastore/derived/emotions_traits_selling_dataset.csv` - Merged dataset with period-level emotions and participant traits

## Key Results

**Full sample:**
- State anxiety × n_sales_earlier interaction is significant (−0.0094*) — anxious participants respond more negatively to accumulated prior sales
- State anxiety (0.0314***) and conscientiousness (0.0218***) are significant main effects

**First sellers:**
- Conscientiousness (0.0708***), impulsivity (0.0598**), and state anxiety (0.0713*) predict first-seller behavior
- Emotions (fear, anger) not significant

**Second sellers:**
- Recency effect (dummy_prev_period) is large and significant (0.2836***) — 28pp increase if first sale was in preceding period
- Individual emotions/traits not significant after controlling for recency

## Testing

- Script executed successfully with 13,590 observations in full sample
- LaTeX tables compile without errors
- Observation counts match expectations from prior analyses
