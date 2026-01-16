# Purpose: Analyze selling decisions using LPM and Logit with fixed effects
# Author: Claude
# Date: 2026-01-15

library(data.table)
library(fixest)

# FILE PATHS
INPUT_PATH <- "datastore/derived/individual_period_dataset.csv"
OUTPUT_PATH <- "analysis/output/analysis/selling_period_regression.tex"
PREPARED_DATA_PATH <- "datastore/derived/selling_period_regression_data.csv"

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  cat("Loading data from:", INPUT_PATH, "\n")
  df <- prepare_data(INPUT_PATH)
  cat("Data dimensions after filtering:", nrow(df), "rows\n")
  print_data_summary(df)

  # Export prepared data for validation
  cat("\nExporting prepared data to:", PREPARED_DATA_PATH, "\n")
  fwrite(df, PREPARED_DATA_PATH)

  cat("\nRunning regressions...\n")
  models <- run_models(df)

  cat("\nExporting LaTeX table to:", OUTPUT_PATH, "\n")
  export_table(models, OUTPUT_PATH)

  print_results(models)
  cat("\nDone!\n")
  return(models)
}

# =====
# Data preparation
# =====
prepare_data <- function(file_path) {
  df <- fread(file_path)

  # Filter out observations where player already sold
  df <- df[already_sold == 0]

  # Create identifier variables
  df[, player_id := paste(session_id, player, sep = "_")]
  df[, global_group_id := paste(session_id, group_id, sep = "_")]
  df[, group_round_id := paste(session_id, segment, group_id, round, sep = "_")]

  # Convert to factors
  df[, segment := as.factor(segment)]
  df[, treatment := as.factor(treatment)]

  # Create sale timing variables
  df <- create_sale_timing_vars(df)

  return(df)
}

create_sale_timing_vars <- function(df) {
  # Compute total group sales per period within each group-round
  period_sales <- df[, .(n_sales = sum(sold)), by = .(group_round_id, period)]
  setorder(period_sales, group_round_id, period)

  # Shift at PERIOD level (before merging to players) to get previous period sales
  period_sales[, prev_period_n_sales := shift(n_sales, 1, type = "lag"),
               by = group_round_id]

  # Merge lagged sales back to player-level data
  df <- merge(df, period_sales[, .(group_round_id, period, prev_period_n_sales)],
              by = c("group_round_id", "period"), all.x = TRUE)

  # Sale in immediately previous period t-1 (recency effect)
  df[, sale_prev_period := as.integer(!is.na(prev_period_n_sales) &
                                        prev_period_n_sales > 0)]

  # Number of sales in periods 1 to t-2 (excludes t-1, independent of sale_prev_period)
  df[, n_sales_earlier := prior_group_sales - fifelse(is.na(prev_period_n_sales), 0L,
                                                       as.integer(prev_period_n_sales))]

  # Clean up temp columns
  df[, c("prev_period_n_sales") := NULL]

  return(df)
}

print_data_summary <- function(df) {
  cat("\nVariable summary:\n")
  cat("  n_sales_earlier: mean =", round(mean(df$n_sales_earlier), 3),
      ", max =", max(df$n_sales_earlier), "\n")
  cat("  sale_prev_period:", sum(df$sale_prev_period), "obs with sale in prev period\n")
  cat("  Segments:", paste(levels(df$segment), collapse = ", "), "\n")
  cat("  Treatments:", paste(levels(df$treatment), collapse = ", "), "\n")
  cat("  Rounds:", min(df$round), "-", max(df$round), "\n")
}

# =====
# Regression models
# =====
run_models <- function(df) {
  models <- list()

  cat("[1/1] LPM (no player FE)...\n")
  models$lpm <- feols(
    sold ~ n_sales_earlier + sale_prev_period + period + signal +
      round + segment + treatment,
    cluster = ~global_group_id,
    data = df
  )

  return(models)
}

# =====
# Output
# =====
export_table <- function(models, output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  etable(
    models$lpm,
    dict = c(
      "(Intercept)" = "Constant",
      n_sales_earlier = "n\\_sales\\_earlier",
      sale_prev_period = "sale\\_prev\\_period",
      period = "period",
      signal = "signal",
      round = "round",
      segment2 = "segment\\_2",
      segment3 = "segment\\_3",
      segment4 = "segment\\_4",
      treatmenttr2 = "treatment\\_2"
    ),
    fitstat = c("n", "r2", "ar2"),
    file = output_path,
    float = FALSE,
    tex = TRUE,
    style.tex = style.tex(fontsize = "scriptsize")
  )
  cat("Table exported to:", output_path, "\n")
}

print_results <- function(models) {
  cat("\n", strrep("=", 60), "\n")
  cat("MODEL RESULTS\n")
  cat(strrep("=", 60), "\n")

  cat("\nLPM:\n")
  print(summary(models$lpm))
}

# %%
if (!interactive()) main()
