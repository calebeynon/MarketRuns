# Purpose: Visualize treatment × period interaction coefficients (diff-in-diff style)
# Author: Claude
# Date: 2026-01-19

library(data.table)
library(fixest)
library(ggplot2)

# FILE PATHS
INPUT_PATH <- "datastore/derived/individual_period_dataset.csv"
OUTPUT_PATH <- "analysis/output/plots/treatment_period_interactions.pdf"

# =====
# Theme for economics papers
# =====
theme_econ <- function() {
  theme_minimal() +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "gray90"),
      text = element_text(family = "serif"),
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 11)
    )
}

# =====
# Main function
# =====
main <- function() {
  cat("Loading data from:", INPUT_PATH, "\n")
  df <- prepare_data(INPUT_PATH)

  cat("Estimating model with clustered SEs...\n")
  model <- estimate_model(df)

  cat("Extracting interaction coefficients...\n")
  coef_df <- extract_interaction_coefs(model)

  cat("Creating plot...\n")
  p <- create_coef_plot(coef_df)

  cat("Saving to:", OUTPUT_PATH, "\n")
  save_plot(p, OUTPUT_PATH)

  cat("Done!\n")
  return(coef_df)
}

# =====
# Data preparation
# =====
prepare_data <- function(file_path) {
  df <- fread(file_path)
  df <- df[already_sold == 0]

  df[, global_group_id := paste(session_id, group_id, sep = "_")]
  df[, segment := as.factor(segment)]
  # Set tr2 as reference so interactions show Treatment 2 effect
  df[, treatment := relevel(as.factor(treatment), ref = "tr2")]

  cat("Sample size:", nrow(df), "observations\n")
  cat("Clusters:", length(unique(df$global_group_id)), "\n")
  return(df)
}

# =====
# Model estimation
# =====
estimate_model <- function(df) {
  model <- feols(
    sold ~ treatment * i(period) + signal + round + i(segment),
    cluster = ~global_group_id,
    data = df
  )
  return(model)
}

# =====
# Extract coefficients with clustered SEs
# =====
extract_interaction_coefs <- function(model) {
  coefs <- coef(model)
  ses <- se(model)

  # Find Treatment 2 interaction terms
  interaction_names <- grep("^treatmenttr2:period::", names(coefs), value = TRUE)
  periods <- as.integer(gsub("treatmenttr2:period::", "", interaction_names))

  coef_df <- data.table(
    period = periods,
    estimate = coefs[interaction_names],
    se = ses[interaction_names]
  )

  # 95% CI using clustered SEs
  coef_df[, ci_lower := estimate - 1.96 * se]
  coef_df[, ci_upper := estimate + 1.96 * se]

  setorder(coef_df, period)
  return(coef_df)
}

# =====
# Coefficient plot (diff-in-diff style, grayscale-safe)
# =====
create_coef_plot <- function(coef_df) {
  p <- ggplot(coef_df, aes(x = period, y = estimate)) +
    # Reference line at 0
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    # Confidence intervals
    geom_errorbar(
      aes(ymin = ci_lower, ymax = ci_upper),
      width = 0.25,
      linewidth = 0.5,
      color = "black"
    ) +
    # Point estimates
    geom_point(size = 2.5, color = "black", shape = 16) +
    # Scales
    scale_x_continuous(breaks = 1:14) +
    scale_y_continuous(
      labels = function(x) paste0(round(x * 100, 1), "pp")
    ) +
    # Labels (no title - goes in paper caption)
    labs(
      x = "Period",
      y = "Treatment 2 Effect on P(Sell)"
    ) +
    theme_econ()

  return(p)
}

# =====
# Save as PDF (vector graphics for publication)
# =====
save_plot <- function(p, output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  ggsave(output_path, p, width = 7, height = 5)
}

# %%
if (!interactive()) main()
