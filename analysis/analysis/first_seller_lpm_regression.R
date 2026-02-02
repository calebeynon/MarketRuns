# Purpose: Linear Probability Model regression to predict first seller status
# Author: Caleb Eynon w/ Claude Code
# Date: 2025-02-01
# nolint start
#
# Tests hypothesis: Individual traits predict who becomes the first seller
# Uses LPM (OLS with binary outcome) with session fixed effects
# Treatment is absorbed by session FE since each session is one treatment

library(data.table)
library(fixest)

# =====
# File paths
# =====
INPUT_DATA <- "datastore/derived/first_seller_analysis_data.csv"
OUTPUT_TEX <- "analysis/output/tables/first_seller_lpm_regression.tex"

# =====
# Main
# =====
main <- function() {
  dt <- load_and_prepare_data()
  print_diagnostics(dt)

  model <- run_lpm_regression(dt)
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

  dt <- convert_factors(dt)
  dt <- create_group_identifier(dt)

  return(dt)
}

convert_factors <- function(dt) {
  dt[, session_id := as.factor(session_id)]
  dt[, segment := as.factor(segment)]
  dt[, group_id := as.factor(group_id)]
  return(dt)
}

create_group_identifier <- function(dt) {
  # Create global group ID for clustering (session + group combination)
  dt[, global_group_id := paste(session_id, group_id, sep = "_")]
  dt[, global_group_id := as.factor(global_group_id)]
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
  print_first_seller_distribution(dt)
  print_trait_summary(dt)
}

print_sample_sizes <- function(dt) {
  cat("\nSample size:", nrow(dt), "\n")
  cat("Unique sessions:", uniqueN(dt$session_id), "\n")
  cat("Unique groups:", uniqueN(dt$global_group_id), "\n")
  cat("Unique segments:", uniqueN(dt$segment), "\n")
}

print_first_seller_distribution <- function(dt) {
  cat("\nFirst seller distribution:\n")
  cat("  First sellers:", sum(dt$is_first_seller, na.rm = TRUE), "\n")
  cat("  Non-first sellers:", sum(1 - dt$is_first_seller, na.rm = TRUE), "\n")
  cat("  First seller rate:", round(mean(dt$is_first_seller, na.rm = TRUE), 3), "\n")
}

print_trait_summary <- function(dt) {
  trait_vars <- c(
    "extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness", "impulsivity", "state_anxiety"
  )
  cat("\nTrait variable means:\n")
  for (var in trait_vars) {
    cat(sprintf("  %s: %.2f\n", var, mean(dt[[var]], na.rm = TRUE)))
  }
}

# =====
# Regression
# =====
run_lpm_regression <- function(dt) {
  cat("\n", strrep("=", 60), "\n")
  cat("RUNNING LPM REGRESSION\n")
  cat(strrep("=", 60), "\n")

  # LPM with session fixed effects and group-clustered SEs
  # Treatment absorbed by session FE (each session is one treatment)
  model <- feols(
    is_first_seller ~ extraversion + agreeableness + conscientiousness +
      neuroticism + openness + impulsivity + state_anxiety +
      public_signal + segment + age + gender_female | session_id,
    cluster = ~global_group_id,
    data = dt
  )

  return(model)
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
