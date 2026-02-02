# Issue #19: Traits Predict First Sellers

## Overview
Analyze whether personality traits predict first seller behavior in market runs experiments.

## Objectives
1. Identify first sellers in each group-round
2. Merge first seller status with personality trait data from surveys
3. Run regression analysis with session fixed effects and clustered standard errors
4. Create descriptive visualizations comparing traits across first sellers vs. non-first sellers

## Methodology

### First Seller Definition
A "first seller" is a participant who sells before their group mates in a given round. Specifically:
- Find the earliest period in which any sale occurred in the group-round
- All players who sold in that earliest period are marked as first sellers
- Multiple players can be first sellers if they sell in the same earliest period

### Data Processing
- Create player-round level dataset identifying first sellers
- Merge with personality trait data (BFI-10, Impulsivity, State Anxiety)
- Generate analysis-ready dataset with controls

### Statistical Analysis
- Linear Probability Model (LPM) with session fixed effects
- Clustered standard errors at group level
- T-test comparison tables for appendix material

### Visualization
- Box plots comparing trait distributions by first seller status
- Violin plots showing distribution shape
- Quartile bar charts for impulsivity and neuroticism
- All plots created in ggplot2 (R), no titles per project standards

## Key Findings
- Higher conscientiousness (+0.11, p<0.05) predicts first seller behavior
- Higher neuroticism (+0.03, p<0.05) predicts first seller behavior
- Higher state anxiety (+0.14, p<0.01) predicts first seller behavior
- Higher public signals (-0.23, p<0.01) reduce first selling
- Age (+0.01, p<0.05) is a positive predictor

## Outputs

### Derived Datasets (datastore/derived/)
| File | Description | Observations |
|------|-------------|--------------|
| `survey_traits.csv` | Participant trait scores | 95 |
| `first_seller_round_data.csv` | Player-round first seller identification | 2880 |
| `first_seller_analysis_data.csv` | First sellers merged with traits | 2850 |

### Python Scripts (analysis/derived/)
| Script | Input | Output |
|--------|-------|--------|
| `build_survey_traits_dataset.py` | Raw survey data | `survey_traits.csv` |
| `build_first_seller_round_dataset.py` | Raw oTree data | `first_seller_round_data.csv` |
| `build_first_seller_analysis_dataset.py` | Above outputs | `first_seller_analysis_data.csv` |

### R Analysis Scripts (analysis/analysis/)
| Script | Output |
|--------|--------|
| `first_seller_lpm_regression.R` | `analysis/output/tables/first_seller_lpm_regression.tex` |
| `first_seller_descriptive_stats.R` | `analysis/output/tables/first_seller_trait_comparisons.tex` |
| `visualize_first_seller_traits.R` | 3 PDF plots in `analysis/output/plots/` |

### Test Coverage (analysis/tests/)
| Test File | Tests |
|-----------|-------|
| `test_build_survey_traits_dataset.py` | 23 |
| `test_build_first_seller_round_dataset.py` | 25 |
| `test_build_first_seller_analysis_dataset.py` | 24 |
| **Total** | **72** |

### Output Files (analysis/output/)
**Tables:**
- `tables/first_seller_lpm_regression.tex` - LPM regression results
- `tables/first_seller_trait_comparisons.tex` - T-test comparisons (appendix)

**Plots:**
- `plots/first_seller_trait_boxplots.pdf` - Box plots by trait
- `plots/first_seller_trait_violins.pdf` - Violin plots by trait
- `plots/first_seller_rate_by_quartile.pdf` - First seller rate by trait quartile

## Dependencies
- Python: pandas, numpy, pathlib
- R: ggplot2, fixest, data.table, tidyverse
