# nolint start
# Purpose: Regress round payoff on deviation from M&M (2020) equilibrium sell
#   threshold pi*, separately for alpha=0 and alpha=0.5, with pooled and
#   asymmetric (early/late) specifications. Export combined LaTeX table and
#   print Wald symmetry p-values.
# Author: Claude Code
# Date: 2026-04-19

library(data.table)
library(fixest)
library(car)

# FILE PATHS
INPUT_CSV <- "datastore/derived/welfare_timing_deviation.csv"
OUTPUT_TEX <- "analysis/output/tables/welfare_timing_deviation.tex"

# Coefficient labels for LaTeX output
COEF_DICT <- c(
  "(Intercept)" = "Constant",
  pi_deviation = "$\\pi$ deviation",
  pi_dev_neg = "$\\pi$ deviation (early, $\\le 0$)",
  pi_dev_pos = "$\\pi$ deviation (late, $> 0$)",
  treatmenttr2 = "treatment\\_2"
)
COL_HEADERS <- c("alpha=0 pooled", "alpha=0 asym",
  "alpha=0.5 pooled", "alpha=0.5 asym")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  dt <- load_data()
  models <- fit_all_models(dt)
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
# Fit pooled and asymmetric models for both alpha values
# =====
fit_all_models <- function(dt) {
  list(
    pooled_a0 = fit_pooled(dt[alpha == 0]),
    asym_a0 = fit_asymmetric(dt[alpha == 0]),
    pooled_a05 = fit_pooled(dt[alpha == 0.5]),
    asym_a05 = fit_asymmetric(dt[alpha == 0.5])
  )
}

# =====
# Pooled spec: round_payoff ~ pi_deviation + treatment | segment + n
# =====
fit_pooled <- function(dt) {
  feols(round_payoff ~ pi_deviation + treatment | segment + n,
    data = dt, cluster = ~global_group_id)
}

# =====
# Asymmetric spec with separate early/late slopes
# =====
fit_asymmetric <- function(dt) {
  feols(round_payoff ~ pi_dev_neg + pi_dev_pos + treatment | segment + n,
    data = dt, cluster = ~global_group_id)
}

# =====
# Wald test for symmetry: H0 pi_dev_neg == pi_dev_pos
# =====
wald_symmetry <- function(model) {
  # car::linearHypothesis uses cluster-robust vcov stored in the feols object
  test <- linearHypothesis(model, "pi_dev_neg = pi_dev_pos",
    vcov. = vcov(model))
  return(test[["Pr(>Chisq)"]][2])
}

# =====
# Export combined 4-column LaTeX table
# =====
export_table <- function(models) {
  output_dir <- dirname(OUTPUT_TEX)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  etable(
    models$pooled_a0, models$asym_a0, models$pooled_a05, models$asym_a05,
    headers = COL_HEADERS,
    dict = COEF_DICT,
    fitstat = c("n", "r2", "ar2"),
    float = FALSE,
    tex = TRUE,
    style.tex = style.tex(fontsize = "scriptsize"),
    file = OUTPUT_TEX,
    replace = TRUE
  )
  cat("Table exported to:", OUTPUT_TEX, "\n")
}

# =====
# Print sample sizes, coefficients, and symmetry p-values to stdout
# =====
print_diagnostics <- function(dt, models) {
  for (a in c(0, 0.5)) {
    cat(sprintf("\nalpha=%.1f sample N: %d\n", a, nrow(dt[alpha == a])))
  }
  cat("\n--- Coefficient summaries ---\n")
  for (nm in names(models)) {
    cat("\n[", nm, "]\n", sep = "")
    print(coeftable(models[[nm]]))
  }
  cat("\n--- Wald symmetry p-values (pi_dev_neg == pi_dev_pos) ---\n")
  cat(sprintf("alpha=0 asym: p=%.4f\n", wald_symmetry(models$asym_a0)))
  cat(sprintf("alpha=0.5 asym: p=%.4f\n", wald_symmetry(models$asym_a05)))
}

# %%
if (!interactive()) main()
