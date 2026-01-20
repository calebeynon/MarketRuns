# Purpose: Test treatment differences in selling behavior across periods using interactions
# Author: Claude
# Date: 2026-01-19

library(data.table)
library(fixest)

# FILE PATHS
INPUT_PATH <- "datastore/derived/individual_period_dataset.csv"
OUTPUT_PATH <- "analysis/output/analysis/selling_timing_treatment_interactions.tex"

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  cat("Loading data from:", INPUT_PATH, "\n")
  df <- load_and_validate(INPUT_PATH)

  df <- prepare_data(df)
  cat("Data dimensions after filtering:", nrow(df), "rows\n")

  cat("\nEstimating model...\n")
  model <- run_model(df)

  print_diagnostics(model, df)
  run_wald_test(model)

  cat("\nExporting LaTeX table to:", OUTPUT_PATH, "\n")
  export_table(model, OUTPUT_PATH)

  cat("\nDone!\n")
}

# =====
# Data loading and validation
# =====
load_and_validate <- function(file_path) {
  df <- fread(file_path)
  required_vars <- c("session_id", "segment", "round", "period", "group_id",
                     "player", "treatment", "signal", "state", "price",
                     "sold", "already_sold", "prior_group_sales")

  validate_variables(df, required_vars)
  report_missing_values(df, required_vars)
  return(df)
}

validate_variables <- function(df, required_vars) {
  missing_vars <- setdiff(required_vars, names(df))
  if (length(missing_vars) > 0) {
    stop("Missing required variables: ", paste(missing_vars, collapse = ", "))
  }
  cat("All required variables present.\n")
}

report_missing_values <- function(df, vars) {
  cat("\nMissing values per variable:\n")
  for (v in vars) {
    n_missing <- sum(is.na(df[[v]]))
    if (n_missing > 0) cat("  ", v, ":", n_missing, "\n")
  }
  cat("  (No missing values found)\n"[all(sapply(vars, function(v) sum(is.na(df[[v]])) == 0))])
}

# =====
# Data preparation
# =====
prepare_data <- function(df) {
  n_before <- nrow(df)
  cat("\nSample size before filtering:", n_before, "\n")

  df <- df[already_sold == 0]
  cat("Sample size after filtering (already_sold == 0):", nrow(df), "\n")

  # Cluster at segment-group level (groups reshuffle across segments)
  df[, global_group_id := paste(session_id, segment, group_id, sep = "_")]
  df[, segment := as.factor(segment)]
  # Set tr2 as reference so interactions show Treatment 2 effect
  df[, treatment := relevel(as.factor(treatment), ref = "tr2")]

  return(df)
}

# =====
# Model estimation
# =====
run_model <- function(df) {
  model <- feols(
    sold ~ treatment * i(period) + signal + round + i(segment),
    cluster = ~global_group_id,
    data = df
  )
  return(model)
}

# =====
# Diagnostics
# =====
print_diagnostics <- function(model, df) {
  cat("\n", strrep("=", 60), "\n")
  cat("MODEL SUMMARY\n")
  cat(strrep("=", 60), "\n")
  print(summary(model))

  n_clusters <- length(unique(df$global_group_id))
  cat("\nNumber of clusters (global_group_id):", n_clusters, "\n")
}

run_wald_test <- function(model) {
  cat("\n", strrep("=", 60), "\n")
  cat("WALD TEST: Joint significance of treatment x period interactions\n")
  cat(strrep("=", 60), "\n")

  # Get coefficient names that match treatment:period interactions
  coef_names <- names(coef(model))
  interaction_terms <- grep("treatment.*period", coef_names, value = TRUE)

  cat("Testing coefficients:", paste(interaction_terms, collapse = ", "), "\n\n")

  wald_result <- wald(model, interaction_terms)
  print(wald_result)
}

# =====
# Output
# =====
export_table <- function(model, output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  etable(
    model,
    dict = create_coef_dict(),
    fitstat = c("n", "r2", "ar2"),
    file = output_path,
    float = FALSE,
    tex = TRUE,
    style.tex = style.tex(fontsize = "scriptsize")
  )
  cat("Table exported to:", output_path, "\n")
}

create_coef_dict <- function() {
  dict <- c(
    "(Intercept)" = "Constant",
    "treatmenttr1" = "Treatment 1",
    "signal" = "Signal",
    "round" = "Round"
  )
  # Add period dummies (1-14)
  for (p in 1:14) {
    dict[paste0("period::", p)] <- paste0("Period ", p)
    dict[paste0("treatmenttr2:period::", p)] <- paste0("Treatment 2 $\\times$ Period ", p)
  }
  # Add segment dummies
  for (s in 2:4) {
    dict[paste0("segment::", s)] <- paste0("Segment ", s)
  }
  return(dict)
}

# %%
if (!interactive()) main()
