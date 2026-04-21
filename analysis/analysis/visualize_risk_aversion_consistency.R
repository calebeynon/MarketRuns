# Purpose: Scatter plot of alpha_task vs alpha_MLE to assess consistency
#          between task-based and choice-based risk aversion estimates.
# Author: Claude
# Date: 2026-04-21

library(data.table)
library(ggplot2)

# FILE PATHS
INPUT_CSV <- "datastore/derived/participant_risk_aversion.csv"
OUTPUT_PDF <- "analysis/output/plots/risk_aversion_consistency.pdf"

# =====
# Theme for economics papers
# =====
theme_econ <- function() {
  theme_minimal(base_family = "serif") +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "gray90"),
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 11)
    )
}

# =====
# Main function
# =====
main <- function() {
  dt <- load_data(INPUT_CSV)
  corr <- cor(dt$alpha_task, dt$alpha_mle, use = "complete.obs")
  p <- create_plot(dt, corr)
  save_plot(p, OUTPUT_PDF)
  cat("Plotted", nrow(dt), "participants. Corr:",
      sprintf("%.3f", corr), "\n")
  cat("Saved to:", OUTPUT_PDF, "\n")
}

# =====
# Load and filter data (drop edge flag + NA alpha_task)
# =====
load_data <- function(path) {
  dt <- fread(path)
  dt <- dt[alpha_task_edge_flag == FALSE & !is.na(alpha_task)]
  return(dt)
}

# =====
# Create scatter plot with CI bars and 45 degree line
# =====
create_plot <- function(dt, corr) {
  x_upper <- max(1, max(dt$alpha_task, na.rm = TRUE))
  label_x <- x_upper * 0.95
  ggplot(dt, aes(x = alpha_task, y = alpha_mle)) +
    geom_errorbar(aes(ymin = alpha_ci_low, ymax = alpha_ci_high),
                  color = "gray70", width = 0) +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", color = "gray40") +
    geom_point(color = "black", size = 1.8) +
    annotate("text", x = label_x, y = 0.05, hjust = 1,
             family = "serif",
             label = sprintf("Corr: %.3f", corr)) +
    coord_cartesian(xlim = c(0, x_upper), ylim = c(0, 1)) +
    labs(x = expression(alpha[task]), y = expression(alpha[MLE])) +
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
  ggsave(output_path, p, width = 6, height = 5)
}

# %%
if (!interactive()) main()
