# Issue #48: Add Risk Tolerance Control to Regressions and CI Plots

## Summary

Add the `risk_tolerance` variable (extracted from `player.allocate`, a 0-20 ECU lottery allocation from the post-experiment survey) end-to-end through the data pipeline, all regression specifications, visualization scripts, and tests.

## Motivation

Risk tolerance may confound selling behavior. Participants who are more risk-tolerant may systematically sell at different times. Including this control strengthens causal inference in the regression models.

## Approach

- Extract `risk_tolerance` from `player.allocate` in `build_survey_traits_dataset.py`
- Propagate through 3 downstream derived datasets (emotions_traits, ordinal_selling_position, first_seller_analysis)
- Include in all trait-bearing regression specifications (Model 3 for unified regressions, both models for ordinal logit)
- Add to shared helpers (`selling_regression_helpers.R`) in `SHOW_TRAITS`
- Update CI difference and robustness plots
- Update descriptive stats and summary statistics tables
- Add unit tests for extraction and range validation

## Key Findings

- **Ordinal logit full sample**: AME = -0.026, p < 0.001 — risk-tolerant participants sell earlier
- **Unified regression Panel A**: coef = -0.0014, p < 0.01
- **Logit Panel A**: AME = -0.0016, p < 0.05
- **Descriptive stats**: Non-first sellers significantly more risk-tolerant (15.33 vs 11.29 ECUs, p < 0.05)
- Not significant in LPM regression, sellers-only ordinal logit, or Panels B/C

## Scripts and Outputs

### Derived (data pipeline)
| Script | Input | Output |
|--------|-------|--------|
| `build_survey_traits_dataset.py` | `datastore/*/survey_*.csv` | `datastore/derived/survey_traits.csv` |
| `build_emotions_traits_dataset.py` | `survey_traits.csv`, period data, emotions | `datastore/derived/emotions_traits_selling_dataset.csv` |
| `build_ordinal_selling_position.py` | round data, emotions, traits | `datastore/derived/ordinal_selling_position.csv` |
| `build_first_seller_analysis_dataset.py` | first seller round data, traits | `datastore/derived/first_seller_analysis_data.csv` |

### Analysis (regressions and visualization)
| Script | Output |
|--------|--------|
| `ordinal_logit_selling_position.R` | `ordinal_logit_selling_position.tex` |
| `unified_selling_regression_panel_a/b/c.R` | `unified_selling_regression.tex`, `unified_selling_regression_full.tex` |
| `unified_selling_logit_panel_a/b/c.R` | `unified_selling_logit.tex`, `unified_selling_logit_full.tex` |
| `first_seller_lpm_regression.R` | `first_seller_lpm_regression.tex` |
| `selling_emotions_traits_unified.R` | `selling_emotions_traits_full/first/second.tex` |
| `first_seller_descriptive_stats.R` | `first_seller_trait_comparisons.tex` |
| `summary_statistics.R` | `summary_demographics_traits.tex` |
| `visualize_first_seller_traits.R` | `first_seller_trait_ci_diff.pdf`, `first_seller_trait_robustness.pdf` |

### Tests
| Test File | Coverage |
|-----------|----------|
| `test_build_survey_traits_dataset.py` | 25/25 passed (2 new: extraction + range) |
| `test_visualize_first_seller_traits.py` | 45/45 passed (updated row count 7→8, added range test) |
