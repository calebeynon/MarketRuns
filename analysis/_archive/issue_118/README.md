# issue_118 archive

Exploratory scripts from #118 that do not feed the paper. The paper's Cox
survival table is produced by `analysis/analysis/cox_survival_normal_vs_reactive.R`
(output: `analysis/output/tables/cox_survival_normal_vs_reactive.tex`, used at
`analysis/paper/main.tex` line 504).

## Archived drivers

- `analysis/cox_survival_presell_reactive_500ms.R` — standalone 500ms reactive
  Cox table (2-column: RE Cox vs. clustered Cox). Output lived at
  `analysis/output/tables/cox_survival_reactive_500ms.tex` and was superseded by
  the side-by-side normal-vs-reactive table.
- `analysis/cox_survival_presell_reactive.R` — window sweep (100/250/500/1000 ms)
  fitting Panel R reactive-seller Cox models; prints comparisons, no table
  output.
- `analysis/cox_survival_presell_windows.R` — window sweep fitting Panel A/B
  (all sellers / first seller) Cox models; prints comparisons, no table output.

## Archived artifacts

- `output/tables/cox_survival_reactive_500ms.tex` — last-committed output of the
  500ms driver, kept for reference.

## Archived tests

- `tests/test_cox_reactive_500ms_output.py` — structural checks on the archived
  500ms `.tex`.

## Running the archived drivers

Source paths are repo-root-relative. Invoke from the repo root:

```bash
Rscript analysis/_archive/issue_118/analysis/cox_survival_presell_reactive_500ms.R
Rscript analysis/_archive/issue_118/analysis/cox_survival_presell_reactive.R
Rscript analysis/_archive/issue_118/analysis/cox_survival_presell_windows.R
```

The 500ms driver writes to `analysis/output/tables/cox_survival_reactive_500ms.tex`
when run (overwriting any stale copy); the archived `.tex` here is the version
committed before archival.
