# Purpose: Regress experiment-implied alpha_MLE on survey-implied alpha_task
#          with session fixed effects and session-clustered SE.
# Author:  Caleb Eynon
# Date:    2026-04-21

library(data.table)
library(fixest)

# =====
# File paths
# =====
INPUT_CSV <- "datastore/derived/participant_risk_aversion.csv"
OUTPUT_TEX <- "analysis/output/tables/risk_aversion_consistency.tex"

# =====
# Main
# =====
main <- function() {
  dt <- load_and_filter(INPUT_CSV)
  cat("Estimation sample:", nrow(dt), "participants\n")
  model <- run_regression(dt)
  print(etable(model))
  write_table(model, OUTPUT_TEX)
  cat("\nWrote:", OUTPUT_TEX, "\n")
}

# =====
# Data loading
# =====
load_and_filter <- function(path) {
  dt <- fread(path)
  dt <- dt[alpha_task_edge_flag == FALSE & !is.na(alpha_task)]
  dt[, session_id := as.factor(session_id)]
  dt
}

# =====
# Regression
# =====
run_regression <- function(dt) {
  feols(alpha_mle ~ alpha_task | session_id,
        data = dt, cluster = ~session_id)
}

# =====
# LaTeX export
# =====
write_table <- function(model, path) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  etable(
    model,
    tex = TRUE,
    file = path,
    replace = TRUE,
    title = "Consistency of experiment-implied and survey-implied CRRA risk aversion.",
    label = "tab:risk_aversion_consistency",
    notes = "Session FE; SE clustered at session level.",
    dict = c(alpha_task = "$\\alpha_{\\text{task}}$",
             alpha_mle = "$\\alpha_{\\text{MLE}}$",
             session_id = "Session")
  )
}

# =====
# Run
# =====
main()
