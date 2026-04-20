# nolint start
# Purpose: Regress round payoff on deviation from M&M (2020) equilibrium sell
#   threshold pi*, separately for alpha=0 and alpha=0.5 (pooled spec only).
#   Export combined LaTeX table with methodological footnote.
# Author: Claude Code
# Date: 2026-04-19

library(data.table)
library(fixest)

# FILE PATHS
INPUT_CSV <- "datastore/derived/welfare_timing_deviation.csv"
OUTPUT_TEX <- "analysis/output/tables/welfare_timing_deviation.tex"

COEF_DICT <- c(
  "(Intercept)" = "Constant",
  pi_deviation = "$\\pi$ deviation",
  treatmenttr2 = "treatment\\_2"
)
COL_HEADERS <- c("$\\alpha=0$", "$\\alpha=0.5$")

# Footnote explaining the omission of asymmetric / indicator specifications.
# Only 6 sales at alpha=0 and 8 at alpha=0.5 fall below the equilibrium
# threshold; at alpha=0 all 6 share nearly identical pi_deviation
# (~ -0.004), so slope and indicator coefficients on the below-threshold
# region are not identified from meaningful within-sample variation.
TABLE_NOTES <- paste(
  "\\textit{Notes:} $\\pi$ deviation $= \\pi_\\text{sale} - \\pi^*$, where $\\pi_\\text{sale}$",
  "is the seller's posterior at the moment of sale and $\\pi^*$ is the M\\&M (2020)",
  "Appendix D equilibrium threshold for the seller's sale position $n$, risk",
  "aversion $\\alpha$, and treatment-specific liquidation rule (T1 $\\to$ random,",
  "T2 $\\to$ average). Fixed effects: segment, sale position $n$. Standard errors",
  "clustered at the group level. We omit asymmetric (linear-spline-at-zero) and",
  "indicator specifications for the below-threshold region because only 6 sales",
  "at $\\alpha=0$ and 8 at $\\alpha=0.5$ fall below the threshold; at $\\alpha=0$",
  "all 6 share $\\pi_\\text{deviation} \\approx -0.004$, so the below-threshold",
  "slope and indicator are not identified from within-sample variation.",
  "Significance codes: *** $p<0.01$, ** $p<0.05$, * $p<0.1$."
)

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  dt <- load_data()
  models <- fit_pooled_models(dt)
  export_table(models)
  print_diagnostics(dt, models)
  cat("Done!\n")
}

# =====
# Load and prepare regression data
# =====
load_data <- function() {
  dt <- fread(INPUT_CSV)
  dt[, segment := factor(segment)]
  dt[, n := factor(n)]
  dt[, treatment := factor(treatment)]
  cat("Loaded rows:", nrow(dt), "\n")
  return(dt)
}

# =====
# Fit pooled spec for both alpha values
# welfare ~ pi_deviation + treatment | segment + n
# =====
fit_pooled_models <- function(dt) {
  list(
    pooled_a0 = fit_pooled(dt[alpha == 0]),
    pooled_a05 = fit_pooled(dt[alpha == 0.5])
  )
}

fit_pooled <- function(dt) {
  feols(welfare ~ pi_deviation + treatment | segment + n,
    data = dt, cluster = ~global_group_id)
}

# =====
# Export 2-column LaTeX table with methodological footnote
# =====
export_table <- function(models) {
  output_dir <- dirname(OUTPUT_TEX)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  etable(
    models$pooled_a0, models$pooled_a05,
    headers = COL_HEADERS,
    dict = COEF_DICT,
    fitstat = c("n", "r2", "ar2"),
    notes = TABLE_NOTES,
    float = FALSE,
    tex = TRUE,
    style.tex = style.tex(fontsize = "scriptsize"),
    file = OUTPUT_TEX,
    replace = TRUE
  )
  cat("Table exported to:", OUTPUT_TEX, "\n")
}

# =====
# Print sample sizes and coefficient summaries to stdout
# =====
print_diagnostics <- function(dt, models) {
  for (a in c(0, 0.5)) {
    cat(sprintf("\nalpha=%.1f sample N: %d\n", a, nrow(dt[alpha == a])))
    cat(sprintf("  rows with pi_deviation < 0: %d\n",
      nrow(dt[alpha == a & pi_deviation < 0])))
  }
  cat("\n--- Coefficient summaries ---\n")
  for (nm in names(models)) {
    cat("\n[", nm, "]\n", sep = "")
    print(coeftable(models[[nm]]))
  }
}

# %%
if (!interactive()) main()
