# Issue 111: Welfare Regression on Individual Traits

## Problem
Analyze how individual personality traits and demographics predict group-level welfare outcomes in the market experiment.

## Approach
Run a welfare regression using `fixest::feols()` with group-round welfare as the outcome variable, regressing on individual personality traits (Big Five, state anxiety, impulsivity, risk tolerance), demographics (age, gender), and experimental controls (segment, round, treatment). Cluster standard errors at the group level. Filter to state == 1 rounds only.

## Inputs
- `datastore/derived/individual_round_panel.csv` — panel data with player-level round information
- `datastore/derived/group_round_welfare.csv` — group-round welfare outcomes
- `datastore/derived/survey_traits.csv` — individual survey traits

## Outputs
- `analysis/output/tables/welfare_regression.tex` — LaTeX regression table
- `analysis/paper/main.tex` — paper updated with welfare regression subsection
