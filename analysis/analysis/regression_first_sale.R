# Purpose: Regression analysis for first sale hypothesis
# Author: Caleb Eynon w/ Claude Code
# Date: 2025-01-11
# nolint start
#
# Tests hypothesis: Treatment 2 participants are more likely to sell than Treatment 1
#
# Three-Part Analysis:
#   Part 1: P(any sale) ~ treatment + segment (selection equation)
#   Part 2: signal_at_first_sale ~ treatment + segment | sale occurred
#   Part 3: n_sellers ~ treatment + segment (count of first sellers)

library(data.table)
library(fixest)

# =====
# File paths
# =====
INPUT_PATH <- "datastore/derived/first_sale_data.csv"

# =====
# Main
# =====
main <- function() {
  dt <- load_and_prepare_data()

  part1_models <- run_part1_selection(dt)
  part2_models <- run_part2_outcome(dt)
  part3_models <- run_part3_n_sellers(dt)

  print_summary_tables(part1_models, part2_models, part3_models)
  create_latex_tables(part1_models, part2_models, part3_models)

  return(list(
    selection = part1_models$model1c,
    outcome = part2_models$model2c,
    logit = part1_models$model1d,
    n_sellers = part3_models$model3c
  ))
}

# =====
# Data preparation
# =====
load_and_prepare_data <- function() {
  dt <- fread(INPUT_PATH)
  cat("Loaded", nrow(dt), "observations\n")

  dt[, treatment_2 := as.integer(treatment == 2)]
  dt[, any_sale := as.integer(!is.na(signal_at_first_sale))]
  dt[, n_sellers := ifelse(is.na(n_sellers_first_period), 0, n_sellers_first_period)]
  dt[, segment_num := as.factor(segment_num)]
  dt[, global_group_id := as.factor(global_group_id)]

  cat("Observations with sales:", sum(dt$any_sale), "\n")
  cat("Observations without sales:", sum(1 - dt$any_sale), "\n")

  return(dt)
}

# =====
# Part 1: Selection - Probability of any sale
# =====
run_part1_selection <- function(dt) {
  cat("\nPART 1: SELECTION EQUATION - P(any sale)\n")
  cat("N =", nrow(dt), "\n")

  cat("  Model 1a: any_sale ~ treatment_2\n")
  model1a <- feols(any_sale ~ treatment_2, data = dt)

  cat("  Model 1b: any_sale ~ treatment_2 + segment_num\n")
  model1b <- feols(any_sale ~ treatment_2 + segment_num, data = dt)

  cat("  Model 1c: any_sale ~ treatment_2 + segment_num, cluster\n")
  model1c <- feols(
    any_sale ~ treatment_2 + segment_num,
    cluster = ~global_group_id,
    data = dt
  )

  cat("  Model 1d: any_sale ~ treatment_2 + segment_num (Logit)\n")
  model1d <- feglm(
    any_sale ~ treatment_2 + segment_num,
    cluster = ~global_group_id,
    family = binomial(link = "logit"),
    data = dt
  )

  return(list(model1a = model1a, model1b = model1b, model1c = model1c, model1d = model1d))
}

# =====
# Part 2: Outcome - Signal at first sale
# =====
run_part2_outcome <- function(dt) {
  dt_sales <- dt[any_sale == 1]

  cat("\nPART 2: OUTCOME EQUATION - signal | sale occurred\n")
  cat("N =", nrow(dt_sales), "\n")

  cat("  Model 2a: signal ~ treatment_2\n")
  model2a <- feols(signal_at_first_sale ~ treatment_2, data = dt_sales)

  cat("  Model 2b: signal ~ treatment_2 + segment_num\n")
  model2b <- feols(signal_at_first_sale ~ treatment_2 + segment_num, data = dt_sales)

  cat("  Model 2c: signal ~ treatment_2 + segment_num, cluster\n")
  model2c <- feols(
    signal_at_first_sale ~ treatment_2 + segment_num,
    cluster = ~global_group_id,
    data = dt_sales
  )

  return(list(model2a = model2a, model2b = model2b, model2c = model2c))
}

# =====
# Part 3: Number of first sellers
# =====
run_part3_n_sellers <- function(dt) {
  cat("\nPART 3: NUMBER OF FIRST SELLERS (0 if no sale)\n")
  cat("N =", nrow(dt), "\n")

  cat("  Model 3a: n_sellers ~ treatment_2\n")
  model3a <- feols(n_sellers ~ treatment_2, data = dt)

  cat("  Model 3b: n_sellers ~ treatment_2 + segment_num\n")
  model3b <- feols(n_sellers ~ treatment_2 + segment_num, data = dt)

  cat("  Model 3c: n_sellers ~ treatment_2 + segment_num, cluster\n")
  model3c <- feols(
    n_sellers ~ treatment_2 + segment_num,
    cluster = ~global_group_id,
    data = dt
  )

  return(list(model3a = model3a, model3b = model3b, model3c = model3c))
}

# =====
# Summary tables
# =====
print_summary_tables <- function(part1, part2, part3) {
  cat("\n", strrep("=", 60), "\n")
  cat("SUMMARY: Part 1 - Selection Models\n")
  cat(strrep("=", 60), "\n")
  print(etable(part1$model1a, part1$model1b, part1$model1c, part1$model1d))

  cat("\n", strrep("=", 60), "\n")
  cat("SUMMARY: Part 2 - Outcome Models\n")
  cat(strrep("=", 60), "\n")
  print(etable(part2$model2a, part2$model2b, part2$model2c))

  cat("\n", strrep("=", 60), "\n")
  cat("SUMMARY: Part 3 - Number of First Sellers\n")
  cat(strrep("=", 60), "\n")
  print(etable(part3$model3a, part3$model3b, part3$model3c))
}

# ======
# Output latex tables for paper
# ======
create_latex_tables <- function(part1, part2, part3){
  # main regressions, main part of paper
  etable(part1$model1c,part2$model2c,part3$model3c,
  file = 'analysis/output/analysis/h2_regression_cluster.tex',
  float = FALSE,
  tex = TRUE,
  style.tex = style.tex(fontsize = "scriptsize")
  )
}

# =====
# Run
# =====
model <- main()
