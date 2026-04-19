# Purpose: Plot average pi = P(Good) at sale by seller position across
#          alpha values, comparing Random vs Average treatment
# Author: Claude
# Date: 2026-04-08

library(data.table)
library(ggplot2)

# FILE PATHS
INPUT_CSV <- "datastore/derived/equilibrium_thresholds.csv"
OUTPUT_PATH <- "analysis/output/plots/equilibrium_avg_pi_at_sale.pdf"

# Seller position labels
SELLER_LABELS <- c(
  "4" = "1st seller (n=4)",
  "3" = "2nd seller (n=3)",
  "2" = "3rd seller (n=2)"
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
      strip.text = element_text(size = 10, face = "italic"),
      legend.position = "bottom"
    )
}

# =====
# Main function
# =====
main <- function() {
  dt <- load_data(INPUT_CSV)
  p <- create_plot(dt)
  save_plot(p, OUTPUT_PATH)
  cat("Saved avg pi plot to:", OUTPUT_PATH, "\n")
}

# =====
# Load and prepare data
# =====
load_data <- function(path) {
  dt <- fread(path)
  dt <- dt[n > 1]
  dt[, seller_label := factor(SELLER_LABELS[as.character(n)],
                              levels = SELLER_LABELS)]
  dt[, treatment := factor(treatment, levels = c("random", "average"))]
  return(dt)
}

# =====
# Create faceted plot
# =====
create_plot <- function(dt) {
  ggplot(dt, aes(x = alpha, y = avg_pi_at_sale, linetype = treatment)) +
    geom_line(color = "black", linewidth = 0.7) +
    geom_point(aes(shape = treatment), color = "black", size = 1.8) +
    scale_linetype_manual(values = TREATMENT_LINES, name = "Treatment") +
    scale_shape_manual(values = c("random" = 16, "average" = 17),
                       name = "Treatment") +
    scale_x_continuous(breaks = seq(0, 0.9, by = 0.2)) +
    scale_y_continuous(breaks = seq(0, 0.5, by = 0.05)) +
    facet_wrap(~ seller_label, nrow = 1) +
    labs(
      x = expression(alpha ~ "(risk aversion)"),
      y = expression("Avg " * pi ~ "= Pr(" * italic(z) * " = " * italic(G) * ") at sale")
    ) +
    theme_econ()
}

# =====
# Save as PDF
# =====
save_plot <- function(p, output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  ggsave(output_path, p, width = 9, height = 4.5)
}

# %%
if (!interactive()) main()
