---
title: "Analysis Scripts (R)"
type: tool
tags: [regression, visualization, R, fixest, ggplot2]
summary: "R scripts in analysis/analysis/ for regressions (fixest) and visualization (ggplot2), producing LaTeX tables and PDF plots"
status: draft
last_verified: "2026-04-19"
---

## Summary

The `analysis/analysis/` directory contains R scripts that consume derived datasets and produce LaTeX regression tables and PDF/PNG plots. Regressions use the `fixest` package; all plots use `ggplot2`.

## Key Points

- Tables output to `analysis/output/tables/` as `.tex` files
- Plots output to `analysis/output/plots/` as `.pdf` or `.png` files
- No plot titles — titles go in the paper
- Helper functions shared via `selling_regression_helpers.R`

## Regression Scripts

| Script | Focus | Key Models |
|--------|-------|------------|
| `selling_period_regression.R` | Main selling timing | FE/RE panel regressions |
| `selling_period_regression_dummies.R` | Period dummies | Coefficient plots |
| `selling_period_regression_extended.R` | Extended with emotions | Adds iMotions variables |
| `unified_selling_regression.R` | Unified panel A/B/C | All specifications |
| `unified_selling_logit.R` | Logit selling model | Binary outcome |
| `cox_survival_regression.R` | Cox proportional hazard | Survival analysis |
| `ro_logit_selling_position.R` | Rank-ordered logit | Selling order |
| `ordinal_logit_selling_position.R` | Ordinal logit | Selling position |
| `first_seller_lpm_regression.R` | First seller LPM | Linear probability model |
| `holdout_liquidation_regression.R` | Holdout strategy | Next-round prediction |
| `welfare_regression.R` | Welfare on traits | OLS clustered (group) |
| `welfare_timing_deviation.R` | Round payoff on π-deviation from M&M threshold; pooled + asymmetric spline, α∈{0, 0.5} | fixest + Wald symmetry test |
| `selling_emotions_traits_unified.R` | Emotions + traits | Combined model |
| `tobit_n_sellers.R` | Tobit model | Number of sellers |
| `did_learning_communication.R` | Diff-in-diff | Learning × communication |
| `summary_statistics.R` | Descriptive stats | Summary tables |
| `trait_correlations.R` | Trait correlations | Personality measures |
| `emotion_correlations.R` | Emotion correlations | iMotions measures |
| `randomized_params_table.py` | Randomized parameters | Session params (Python) |

## Visualization Scripts

| Script | Output |
|--------|--------|
| `visualize_selling_timing.R` | Selling timing patterns |
| `visualize_holdout_payoff_coefficients.R` | Holdout payoff coefficient plot |
| `visualize_first_seller_traits.R` | First seller trait comparison |
| `visualize_did_segment_effects.R` | DiD segment effects |
| `visualize_welfare_theory.R` | Theoretical welfare plot |

## Robustness Variants

Several regressions have `_no_valence` and `_valence_only` variants (Cox survival, rank-ordered logit) for checking sensitivity to emotion valence measures.

## Related

- [Derived Datasets Pipeline](wiki/tools/derived-datasets.md)
- [Project Architecture](wiki/tools/architecture.md)
