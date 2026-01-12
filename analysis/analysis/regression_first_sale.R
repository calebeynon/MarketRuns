# Purpose: Regression analysis for first sale hypothesis
# Author: Claude Code
# Date: 2025-01-11
# nolint start
#
# Tests hypothesis: Treatment 2 participants are less likely to sell than Treatment 1
#
# Two-Part Model:
#   Part 1: P(any sale) ~ treatment + segment (selection equation)
#   Part 2: signal_at_first_sale ~ treatment + segment | sale occurred (outcome equation)

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
  # Load data (keep all observations)
  dt <- fread(INPUT_PATH)
  cat("Loaded", nrow(dt), "observations\n")

  # Create variables
  dt[, treatment_2 := as.integer(treatment == 2)]
  dt[, any_sale := as.integer(!is.na(signal_at_first_sale))]
  dt[, segment_num := as.factor(segment_num)]
  dt[, global_group_id := as.factor(global_group_id)]

  cat("Observations with sales:", sum(dt$any_sale), "\n")
  cat("Observations without sales:", sum(1 - dt$any_sale), "\n")

  # =============================================
  # PART 1: Selection - Probability of any sale
  # =============================================
  cat("\n", strrep("=", 60), "\n")
  cat("PART 1: SELECTION EQUATION - P(any sale)\n")
  cat(strrep("=", 60), "\n")

  cat("\nModel 1a: any_sale ~ treatment (LPM)\n")
  cat(strrep("-", 40), "\n")
  model1a <- feols(any_sale ~ treatment_2, data = dt)
  print(summary(model1a))

  cat("\nModel 1b: any_sale ~ treatment + segment (LPM)\n")
  cat(strrep("-", 40), "\n")
  model1b <- feols(any_sale ~ treatment_2 + segment_num, data = dt)
  print(summary(model1b))

  cat("\nModel 1c: any_sale ~ treatment + segment, cluster by group (LPM)\n")
  cat(strrep("-", 40), "\n")
  model1c <- feols(
    any_sale ~ treatment_2 + segment_num,
    cluster = ~global_group_id,
    data = dt
  )
  print(summary(model1c))

  cat("\nModel 1d: any_sale ~ treatment + segment (Logit)\n")
  cat(strrep("-", 40), "\n")
  model1d <- feglm(
    any_sale ~ treatment_2 + segment_num,
    cluster = ~global_group_id,
    family = binomial(link = "logit"),
    data = dt
  )
  print(summary(model1d))

  # =============================================
  # PART 2: Outcome - Signal at first sale
  # =============================================
  cat("\n", strrep("=", 60), "\n")
  cat("PART 2: OUTCOME EQUATION - signal | sale occurred\n")
  cat(strrep("=", 60), "\n")

  # Subset to observations with sales
  dt_sales <- dt[any_sale == 1]

  cat("\nModel 2a: signal ~ treatment\n")
  cat(strrep("-", 40), "\n")
  model2a <- feols(signal_at_first_sale ~ treatment_2, data = dt_sales)
  print(summary(model2a))

  cat("\nModel 2b: signal ~ treatment + segment\n")
  cat(strrep("-", 40), "\n")
  model2b <- feols(
    signal_at_first_sale ~ treatment_2 + segment_num,
    data = dt_sales
  )
  print(summary(model2b))

  cat("\nModel 2c: signal ~ treatment + segment, cluster by group\n")
  cat(strrep("-", 40), "\n")
  model2c <- feols(
    signal_at_first_sale ~ treatment_2 + segment_num,
    cluster = ~global_group_id,
    data = dt_sales
  )
  print(summary(model2c))

  # =============================================
  # Summary tables
  # =============================================
  cat("\n", strrep("=", 60), "\n")
  cat("SUMMARY: Part 1 - Selection Models\n")
  cat(strrep("=", 60), "\n")
  etable(model1a, model1b, model1c, model1d)

  cat("\n", strrep("=", 60), "\n")
  cat("SUMMARY: Part 2 - Outcome Models\n")
  cat(strrep("=", 60), "\n")
  etable(model2a, model2b, model2c)

  return(list(selection = model1c, outcome = model2c, logit = model1d))
}

# =====
# Run
# =====
model <- main()
