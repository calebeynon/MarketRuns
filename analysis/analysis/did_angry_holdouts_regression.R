# Purpose: DiD regression for angry holdouts - Issue #20
# Author: Claude
# Date: 2026-02-02
# nolint start
#
# Tests hypothesis: Holdouts' decision to sell depends on interaction between
# random liquidation payoff and anger during Results phase.
#
# IDENTIFICATION: Triple interaction between payoff (random), anger (measured),
# and chat_available (exogenous treatment timing).

library(data.table)
library(fixest)

# =====
# File paths
# =====
INPUT_DATA <- "datastore/derived/holdout_anger_analysis.csv"
OUTPUT_TEX <- "analysis/output/tables/did_angry_holdouts_regression.tex"

# =====
# Main
# =====
main <- function() {
  dt <- load_and_prepare_data()
  print_diagnostics(dt)

  model <- run_regression(dt)
  print_results(model)
  export_latex_table(model)

  return(model)
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function() {
  dt <- fread(INPUT_DATA)
  cat("Loaded", nrow(dt), "observations\n")

  dt <- drop_missing_anger(dt)
  dt <- create_factors(dt)

  return(dt)
}

drop_missing_anger <- function(dt) {
  n_before <- nrow(dt)
  dt <- dt[!is.na(anger_results)]
  n_dropped <- n_before - nrow(dt)
  cat("Dropped", n_dropped, "observations with missing anger data\n")
  cat("Remaining:", nrow(dt), "observations\n")
  return(dt)
}

create_factors <- function(dt) {
  dt[, global_group_id := as.factor(global_group_id)]
  dt[, global_round := as.factor(global_round)]
  # Create payoff factor with 2 as reference (lowest payoff)
  dt[, round_payoff_factor := relevel(as.factor(round_payoff), ref = "2")]
  return(dt)
}

# =====
# Diagnostics
# =====
print_diagnostics <- function(dt) {
  cat("\n", strrep("=", 60), "\n")
  cat("SAMPLE DIAGNOSTICS\n")
  cat(strrep("=", 60), "\n")

  print_sample_sizes(dt)
  print_anger_distribution(dt)
  print_payoff_distribution(dt)
}

print_sample_sizes <- function(dt) {
  cat("\nTotal sample size:", nrow(dt), "\n")
  cat("Unique groups:", uniqueN(dt$global_group_id), "\n")
  cat("Unique rounds:", uniqueN(dt$global_round), "\n")

  cat("\nSample by chat_available:\n")
  print(dt[, .N, by = chat_available][order(chat_available)])
}

print_anger_distribution <- function(dt) {
  cat("\nAnger distribution (Results phase):\n")
  print(summary(dt$anger_results))
}

print_payoff_distribution <- function(dt) {
  cat("\nPayoff distribution:\n")
  print(table(dt$round_payoff))
}

# =====
# Regression
# =====
run_regression <- function(dt) {
  cat("\n", strrep("=", 60), "\n")
  cat("RUNNING REGRESSION\n")
  cat(strrep("=", 60), "\n")

  cat("\nTriple interaction: payoff x anger x chat_available\n")
  cat("Reference payoff: 2 (lowest)\n")

  feols(
    sold_next_round ~ round_payoff_factor * anger_results * chat_available +
      prior_sales | global_group_id^global_round,
    cluster = ~global_group_id,
    data = dt
  )
}

# =====
# Results output
# =====
print_results <- function(model) {
  cat("\n", strrep("=", 60), "\n")
  cat("REGRESSION RESULTS\n")
  cat(strrep("=", 60), "\n")
  print(etable(model))
}

export_latex_table <- function(model) {
  # Ensure output directory exists
  dir.create(dirname(OUTPUT_TEX), recursive = TRUE, showWarnings = FALSE)

  etable(
    model,
    file = OUTPUT_TEX,
    float = FALSE,
    tex = TRUE,
    style.tex = style.tex(fontsize = "scriptsize")
  )
  cat("\nLaTeX table exported to:", OUTPUT_TEX, "\n")
}

# =====
# Run
# =====
model <- main()
