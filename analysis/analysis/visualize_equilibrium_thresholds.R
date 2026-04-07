# Purpose: Plot equilibrium selling thresholds (pi*) vs risk aversion (alpha)
#          by number of holders (n) and treatment
# Author: Claude
# Date: 2026-04-05

library(data.table)
library(ggplot2)

# FILE PATHS
INPUT_CSV <- "datastore/derived/equilibrium_thresholds.csv"
OUTPUT_PATH <- "analysis/output/plots/equilibrium_thresholds.pdf"

# Colorblind-friendly palette for n values
N_PALETTE <- c(
  "2" = "#0072B2",
  "3" = "#E69F00",
  "4" = "#D55E00"
)

# Treatment linetypes
TREATMENT_LINES <- c(
  "random" = "solid",
  "average" = "dashed"
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
      axis.title = element_text(size = 11),
      legend.position = "bottom",
      legend.box = "horizontal"
    )
}

# =====
# Main function
# =====
main <- function() {
  dt <- load_threshold_data(INPUT_CSV)
  p <- create_threshold_plot(dt)
  save_plot(p, OUTPUT_PATH)
  cat("Saved threshold plot to:", OUTPUT_PATH, "\n")
}

# =====
# Load and prepare data
# =====
load_threshold_data <- function(path) {
  dt <- fread(path)
  dt <- dt[n > 1]  # n=1 never sells
  dt[, n := factor(n)]
  dt[, treatment := factor(treatment, levels = c("random", "average"))]
  return(dt)
}

# =====
# Create threshold plot
# =====
create_threshold_plot <- function(dt) {
  p <- ggplot(dt, aes(x = alpha, y = threshold_pi, color = n, linetype = treatment)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 1.5) +
    scale_color_manual(values = N_PALETTE, name = expression(italic(n))) +
    scale_linetype_manual(values = TREATMENT_LINES, name = "Treatment") +
    scale_x_continuous(breaks = seq(0, 0.9, by = 0.1)) +
    scale_y_continuous(breaks = seq(0, 0.5, by = 0.1), limits = c(0, 0.55)) +
    labs(
      x = expression(alpha ~ "(risk aversion)"),
      y = expression(pi * "*" ~ "(selling threshold)")
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
