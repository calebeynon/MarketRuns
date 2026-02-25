# Issue 84: Restructure Table 7 — No Traits / With Traits Column Split

## Summary
Restructure the Cox survival regression table (Table 7) from an emotions/valence column
split to a no-traits/with-traits column split. All columns now include both individual
emotions and valence simultaneously; the column distinction is whether personality traits
are included.

### Column Layout
- **Column 1 (All Sellers, No Traits):** Cascade/interactions + emotions + valence + base controls
- **Column 2 (All Sellers, With Traits):** Same + state anxiety, impulsivity, risk tolerance, Big Five
- **Column 3 (First Sellers, No Traits):** Emotions + valence + base controls (no cascade)
- **Column 4 (First Sellers, With Traits):** Same + state anxiety, impulsivity, risk tolerance, Big Five

### Scripts and Outputs
| Script | Output |
|--------|--------|
| `analysis/analysis/cox_survival_regression.R` | `analysis/output/tables/cox_survival_regression.tex` |
| `analysis/analysis/cox_survival_panel_a.R` | (sourced by main script) |
| `analysis/analysis/cox_survival_panel_b.R` | (sourced by main script) |

## Acceptance Criteria
- [x] Four columns: {All Sellers, First Sellers} x {No Traits, With Traits}
- [x] All columns include both discrete emotions and valence
- [x] No Traits columns include base controls only (signal, round, segment, treatment, age, female)
- [x] With Traits columns add all personality traits (state anxiety, impulsivity, risk tolerance, Big Five)
- [x] Column headers read "No Traits" / "With Traits"
- [x] Row ordering: cascade → emotions → valence → controls → traits
- [x] Table compiles cleanly in LaTeX
- [x] main.tex prose updated to describe the new structure
- [x] 72 output tests validate .tex column structure
- [x] All 122 Cox-related tests pass

## Files Modified
- `analysis/analysis/cox_survival_regression.R` — table builder, variable ordering, column headers
- `analysis/analysis/cox_survival_panel_a.R` — All Sellers model formulas
- `analysis/analysis/cox_survival_panel_b.R` — First Sellers model formulas
- `analysis/output/tables/cox_survival_regression.tex` — regenerated LaTeX table
- `analysis/paper/main.tex` — prose describing Table 7
- `analysis/tests/test_cox_survival_output.py` — new test file (72 tests)
