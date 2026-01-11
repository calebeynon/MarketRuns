# Purpose: Regression analysis for first sale hypothesis
# Author: Claude Code
# Date: 2025-01-11
# nolint start
#
# Tests hypothesis: Treatment 2 participants are less likely to sell than Treatment 1
# Regression: signal_at_first_sale ~ treatment + group_FE + segment

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
  # Load data
  dt <- fread(INPUT_PATH)
  cat("Loaded", nrow(dt), "observations\n")

  # Drop observations with no sales
  dt <- dt[!is.na(signal_at_first_sale)]
  cat("Observations with sales:", nrow(dt), "\n")

  # Create treatment dummy
  dt[, treatment_2 := as.integer(treatment == 2)]

  # Convert to factors
  dt[, segment_num := as.factor(segment_num)]
  dt[, global_group_id := as.factor(global_group_id)]

  # Run regressions
  cat("\n", strrep("=", 60), "\n")
  cat("MODEL 1: signal_at_first_sale ~ treatment\n")
  cat(strrep("=", 60), "\n")
  model1 <- feols(signal_at_first_sale ~ treatment_2, data = dt)
  print(summary(model1))

  cat("\n", strrep("=", 60), "\n")
  cat("MODEL 2: signal_at_first_sale ~ treatment + segment\n")
  cat(strrep("=", 60), "\n")
  model2 <- feols(signal_at_first_sale ~ treatment_2 + segment_num, data = dt)
  print(summary(model2))

  cat("\n", strrep("=", 60), "\n")
  cat("MODEL 3: signal_at_first_sale ~ treatment + segment, cluster by group\n")
  cat(strrep("=", 60), "\n")
  model3 <- feols(
    signal_at_first_sale ~ treatment_2 + segment_num,
    cluster = ~global_group_id,
    data = dt
  )
  print(summary(model3))

  # Summary table
  cat("\n", strrep("=", 60), "\n")
  cat("SUMMARY: All Models\n")
  cat(strrep("=", 60), "\n")
  etable(model1, model2, model3)

  return(model3)
}

# =====
# Run
# =====
model <- main()
