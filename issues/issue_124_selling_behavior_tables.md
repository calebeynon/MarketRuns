# Issue 124: Build Selling Behavior Section Regression Tables (Tables 4-7)

## Summary

Implement and restructure the four regression tables for the paper's Selling Behavior section per confirmed specifications in GitHub issue #124.

## Confirmed Specifications

- **Table 4 (Tobit, `tobit_n_sellers.R` restructured)**: 2 columns. M1: `n_sellers ~ treatment + segment + round`; M2: adds `bad_state`. Factor coding; cluster-robust SE at `global_group_id`.
- **Table 5 (Pooled OLS, new `ols_first_sale_behavior.R`)**: 4 columns. M1/M2: `signal_at_first_sale ± bad_state`; M3/M4: `first_sale_period ± bad_state`. Group-level clustering.
- **Table 6 (Cox survival, new `cox_selling_four_column.R`)**: 4 columns — first sellers, reactive sellers, all sellers, all participants. Full emotion+trait spec.
- **Table 7 (rank-ordered logit, new `ro_logit_two_column.R`)**: 2 columns — sellers-only and all-participants. Full spec.

## Acceptance Criteria

- [ ] `tobit_n_sellers.tex` — 2-column Tobit table
- [ ] `ols_first_sale_behavior.tex` — 4-column OLS table
- [ ] `cox_selling_four_column.tex` — 4-column Cox table
- [ ] `ro_logit_two_column.tex` — 2-column rank-ordered logit table
- [ ] All `.tex` outputs saved to `analysis/output/tables/`
- [ ] `main.tex` wired with correct bare filenames; paper compiles clean
