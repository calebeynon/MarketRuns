# Purpose: Regression analysis for holdout liquidation hypothesis
# Author: Caleb Eynon w/ Claude Code
# Date: 2025-01-22
# nolint start
#
# Tests hypothesis: Holdouts' decision to sell in the next round depends on
# the previous round's liquidation payoff, controlling for prior sales.
#
# IDENTIFICATION STRATEGY: Within a group-round, holdouts receive randomly
# assigned liquidation payoffs (2, 4, 6, or 8 ECU). We use group-by-round
# fixed effects to compare ONLY within the same group-round, isolating the
# random variation in payoffs.

library(data.table)
library(fixest)

# =====
# File paths
# =====
INPUT_DATA <- "datastore/derived/holdout_next_round_analysis.csv"
OUTPUT_TEX <- "analysis/output/analysis/holdout_liquidation_regression.tex"

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

  dt[, global_group_id := as.factor(global_group_id)]
  dt[, global_round := as.factor(global_round)]
  dt[, round_payoff_factor := as.factor(round_payoff)]
  dt[, received_2 := as.integer(round_payoff == 2)]

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
  print_payoff_distribution(dt)
  print_selling_rates(dt)
}

print_sample_sizes <- function(dt) {
  cat("\nSample size:", nrow(dt), "\n")
  cat("Unique groups:", uniqueN(dt$global_group_id), "\n")
  cat("Unique rounds:", uniqueN(dt$global_round), "\n")

  # Create group-round identifier for within-group-round analysis
  dt[, group_round := paste(global_group_id, global_round, sep = "_")]
  cat("Unique group-rounds:", uniqueN(dt$group_round), "\n")

  # Group-rounds with multiple holdouts (needed for within-group-round comparison)
  holdouts_per_gr <- dt[, .N, by = group_round]
  cat("Group-rounds with 2+ holdouts:", sum(holdouts_per_gr$N >= 2), "\n")

  # Group-rounds with payoff variation (identification source)
  payoff_var <- dt[, .(n_payoffs = uniqueN(round_payoff)), by = group_round]
  multi_holdout_gr <- holdouts_per_gr[N >= 2]$group_round
  var_in_multi <- payoff_var[group_round %in% multi_holdout_gr]
  cat("Group-rounds with payoff variation:", sum(var_in_multi$n_payoffs > 1), "\n")
}

print_payoff_distribution <- function(dt) {
  cat("\nDistribution of round_payoff values:\n")
  print(table(dt$round_payoff))
}

print_selling_rates <- function(dt) {
  cat("\nSelling rate by payoff level:\n")
  selling_by_payoff <- dt[, .(
    n = .N,
    sold = sum(sold_next_round),
    rate = mean(sold_next_round)
  ), by = round_payoff][order(round_payoff)]
  print(selling_by_payoff)
}

# =====
# Regression
# =====
run_regression <- function(dt) {
  cat("\n", strrep("=", 60), "\n")
  cat("RUNNING REGRESSION\n")
  cat(strrep("=", 60), "\n")

  # Group-by-round FE forces within-group-round comparison (random assignment)
  cat("\nCategorical payoff with group x round FE\n")
  feols(
    sold_next_round ~ round_payoff_factor + prior_sales | global_group_id^global_round,
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
