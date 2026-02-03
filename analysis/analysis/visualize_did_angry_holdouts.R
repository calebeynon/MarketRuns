# Purpose: Visualize DiD regression coefficients for angry holdouts
# Author: Claude
# Date: 2026-02-02
# nolint start
#
# Creates coefficient plot for payoff x anger interactions, faceted by chat availability.
# Corresponds to regression in did_angry_holdouts_regression.R

library(data.table)
library(fixest)
library(ggplot2)

# =====
# File paths
# =====
INPUT_DATA <- "datastore/derived/holdout_anger_analysis.csv"
OUTPUT_PLOT <- "analysis/output/plots/did_angry_holdouts_coefficients.png"

# =====
# Main
# =====
main <- function() {
  dt <- load_and_prepare_data()
  model <- run_regression(dt)
  plot_data <- extract_coefficients(model)
  p <- create_plot(plot_data)
  save_plot(p)
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function() {
  dt <- fread(INPUT_DATA)
  dt <- dt[!is.na(anger_results)]
  dt <- create_factors(dt)
  return(dt)
}

create_factors <- function(dt) {
  dt[, global_group_id := as.factor(global_group_id)]
  dt[, global_round := as.factor(global_round)]
  dt[, round_payoff_factor := relevel(as.factor(round_payoff), ref = "2")]
  return(dt)
}

# =====
# Regression (same as did_angry_holdouts_regression.R)
# =====
run_regression <- function(dt) {
  feols(
    sold_next_round ~ round_payoff_factor * anger_results * chat_available +
      prior_sales | global_group_id^global_round,
    cluster = ~global_group_id,
    data = dt
  )
}

# =====
# Extract coefficients for payoff x anger interactions
# =====
extract_coefficients <- function(model) {
  coef_names <- names(coef(model))
  coefs <- coef(model)
  ses <- se(model)
  df <- fixest::degrees_freedom(model, type = "t")

  # Extract payoff x anger interactions (with and without chat)
  plot_data <- build_coefficient_table(coef_names, coefs, ses, df)
  return(plot_data)
}

build_coefficient_table <- function(coef_names, coefs, ses, df) {
  # Payoff x Anger (no chat context)
  no_chat <- extract_payoff_anger_coefs(coef_names, coefs, ses, df, with_chat = FALSE)
  no_chat[, chat := "No Chat"]

  # Payoff x Anger x Chat (triple interaction adds to base interaction)
  with_chat <- extract_payoff_anger_coefs(coef_names, coefs, ses, df, with_chat = TRUE)
  with_chat[, chat := "With Chat"]

  plot_data <- rbind(no_chat, with_chat)
  plot_data[, chat := factor(chat, levels = c("No Chat", "With Chat"))]
  return(plot_data)
}

extract_payoff_anger_coefs <- function(coef_names, coefs, ses, df, with_chat) {
  payoffs <- c(4, 6, 8)
  results <- data.table(
    payoff = payoffs,
    estimate = numeric(3),
    se = numeric(3)
  )

  for (i in seq_along(payoffs)) {
    p <- payoffs[i]
    coef_info <- get_coefficient_for_payoff(coef_names, coefs, ses, p, with_chat)
    results[i, estimate := coef_info$est]
    results[i, se := coef_info$se]
  }

  results <- add_confidence_intervals(results, df)
  return(results)
}

get_coefficient_for_payoff <- function(coef_names, coefs, ses, payoff, with_chat) {
  # Base interaction: round_payoff_factorX:anger_results
  base_pattern <- paste0("round_payoff_factor", payoff, ":anger_results$")
  base_idx <- grep(base_pattern, coef_names)

  if (with_chat) {
    # Triple interaction: round_payoff_factorX:anger_results:chat_availableTRUE
    triple_pattern <- paste0("round_payoff_factor", payoff, ":anger_results:chat_available")
    triple_idx <- grep(triple_pattern, coef_names)
    # Sum coefficients for total effect when chat is available
    est <- coefs[base_idx] + coefs[triple_idx]
    # SE approximation (assumes some correlation, conservative)
    se_combined <- sqrt(ses[base_idx]^2 + ses[triple_idx]^2)
    return(list(est = est, se = se_combined))
  } else {
    return(list(est = coefs[base_idx], se = ses[base_idx]))
  }
}

add_confidence_intervals <- function(dt, df) {
  dt[, ci95_low := estimate - qt(0.975, df) * se]
  dt[, ci95_high := estimate + qt(0.975, df) * se]
  return(dt)
}

# =====
# Create plot
# =====
create_plot <- function(plot_data) {
  ggplot(plot_data, aes(x = factor(payoff), y = estimate)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_errorbar(aes(ymin = ci95_low, ymax = ci95_high), width = 0.2, linewidth = 0.6) +
    geom_point(size = 2.5) +
    facet_wrap(~chat, ncol = 2) +
    labs(x = "Liquidation Payoff (ECU)", y = "Payoff x Anger Interaction Effect") +
    theme_minimal() +
    get_plot_theme()
}

get_plot_theme <- function() {
  theme(
    panel.grid.minor = element_blank(),
    axis.text = element_text(size = 11),
    axis.title = element_text(size = 12),
    strip.text = element_text(size = 12, face = "bold")
  )
}

# =====
# Save plot
# =====
save_plot <- function(p) {
  dir.create(dirname(OUTPUT_PLOT), recursive = TRUE, showWarnings = FALSE)
  ggsave(OUTPUT_PLOT, p, width = 8, height = 5, dpi = 300)
  cat("Plot saved to:", OUTPUT_PLOT, "\n")
}

# %%
if (sys.nframe() == 0) {
  main()
}
