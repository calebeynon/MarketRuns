# Purpose: Visualize regression coefficients for holdout liquidation payoff effect
# Author: Caleb Eynon w/ Claude Code
# Date: 2025-01-22
# nolint start

library(data.table)
library(fixest)
library(ggplot2)

# =====
# File paths
# =====
INPUT_DATA <- "datastore/derived/holdout_next_round_analysis.csv"
OUTPUT_PLOT <- "analysis/output/plots/holdout_payoff_coefficients.png"

# =====
# Main
# =====
main <- function() {
  model <- run_regression()
  plot_data <- extract_coefficients(model)
  p <- create_plot(plot_data)
  save_plot(p)
}

# =====
# Regression
# =====
run_regression <- function() {
  dt <- fread(INPUT_DATA)
  dt[, global_group_id := as.factor(global_group_id)]
  dt[, global_round := as.factor(global_round)]
  dt[, round_payoff_factor := as.factor(round_payoff)]

  feols(
    sold_next_round ~ round_payoff_factor + prior_sales | global_group_id^global_round,
    cluster = ~global_group_id,
    data = dt
  )
}

# =====
# Extract coefficients and compute CIs
# =====
extract_coefficients <- function(model) {
  # Extract payoff coefficients by name for robustness
  coef_names <- names(coef(model))
  payoff_coef_names <- coef_names[startsWith(coef_names, "round_payoff_factor")]
  coefs <- coef(model)[payoff_coef_names]
  ses <- se(model)[payoff_coef_names]

  # Extract degrees of freedom dynamically (n_clusters - 1 for cluster-robust SEs)
  df <- fixest::degrees_freedom(model, type = "t")

  plot_data <- data.table(
    payoff = c(4, 6, 8),
    estimate = coefs,
    se = ses
  )

  plot_data[, ci90_low := estimate - qt(0.95, df) * se]
  plot_data[, ci90_high := estimate + qt(0.95, df) * se]
  plot_data[, ci95_low := estimate - qt(0.975, df) * se]
  plot_data[, ci95_high := estimate + qt(0.975, df) * se]

  # Add baseline
  baseline <- data.table(
    payoff = 2, estimate = 0, se = 0,
    ci90_low = 0, ci90_high = 0, ci95_low = 0, ci95_high = 0
  )
  rbind(baseline, plot_data)
}

# =====
# Create plot
# =====
create_plot <- function(plot_data) {
  ggplot(plot_data, aes(x = factor(payoff), y = estimate)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_linerange(aes(ymin = ci95_low, ymax = ci95_high),
                   color = "red", linewidth = 0.6) +
    geom_linerange(aes(ymin = ci90_low, ymax = ci90_high),
                   color = "black", linewidth = 0.8) +
    geom_point(size = 2) +
    labs(
      x = "Liquidation Payoff (ECU)",
      y = "Effect on Pr(Sell Next Market)"
    ) +
    theme_minimal() +
    theme(
      panel.grid.minor = element_blank(),
      axis.text = element_text(size = 11),
      axis.title = element_text(size = 12)
    )
}

# =====
# Save plot
# =====
save_plot <- function(p) {
  ggsave(OUTPUT_PLOT, p, width = 6, height = 4, dpi = 300)
  cat("Plot saved to:", OUTPUT_PLOT, "\n")
}

# %%
if (sys.nframe() == 0) {
  main()
}
