# Purpose: Plot theoretical welfare lines by risk-aversion parameter alpha
# Author: Claude
# Date: 2026-04-04

library(data.table)
library(ggplot2)

# FILE PATHS
OUTPUT_PATH <- "analysis/output/plots/welfare_theory.pdf"

# PARAMETERS
N_SELLERS <- 4
PRICES <- c(2, 4, 6, 8)  # p_n = 2n; price when n sellers remain
LIQUIDATION_VALUE <- 20
ALPHA_SEQ <- seq(0.01, 1, by = 0.01)
M_VALUES <- 0:4

# Colorblind-friendly palette (one per m value)
WELFARE_PALETTE <- c(
  "0" = "#0072B2",
  "1" = "#E69F00",
  "2" = "#009E73",
  "3" = "#D55E00",
  "4" = "#CC79A7"
)

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
  dt <- build_welfare_data()
  p <- create_welfare_plot(dt)
  save_plot(p, OUTPUT_PATH)
  cat("Saved welfare plot to:", OUTPUT_PATH, "\n")
}

# =====
# Compute welfare for a single (alpha, m) pair
# =====
compute_welfare <- function(alpha, m) {
  exponent <- 1 - alpha
  # Sum of p_n^{1-alpha} for remaining sellers (n = m+1 to N)
  if (m < N_SELLERS) {
    price_sum <- sum(PRICES[(m + 1):N_SELLERS]^exponent)
  } else {
    price_sum <- 0
  }
  numerator <- price_sum + m * LIQUIDATION_VALUE^exponent
  denominator <- N_SELLERS * LIQUIDATION_VALUE^exponent
  return(numerator / denominator)
}

# =====
# Build data.table of welfare values across alpha and m
# =====
build_welfare_data <- function() {
  dt <- CJ(alpha = ALPHA_SEQ, m = M_VALUES)
  dt[, welfare := mapply(compute_welfare, alpha, m)]
  dt[, m := factor(m)]
  return(dt)
}

# =====
# Create welfare plot
# =====
create_welfare_plot <- function(dt) {
  p <- ggplot(dt, aes(x = alpha, y = welfare, color = m)) +
    geom_line(linewidth = 0.8) +
    scale_color_manual(values = WELFARE_PALETTE, name = expression(italic(m))) +
    scale_y_continuous(breaks = seq(0, 1, by = 0.1)) +
    labs(
      x = expression(alpha),
      y = expression(italic(w)[italic(m)])
    ) +
    theme_econ()
  return(p)
}

# =====
# Save as PDF
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
