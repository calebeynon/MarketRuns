# Purpose: Caterpillar plot of per-participant MLE CRRA alpha with
#          likelihood-ratio 95% confidence intervals, colored by treatment.
# Author:  Caleb Eynon
# Date:    2026-04-21

library(data.table)
library(ggplot2)

# =====
# File paths
# =====
INPUT_CSV <- "datastore/derived/participant_risk_aversion.csv"
OUTPUT_PDF <- "analysis/output/plots/implied_risk_aversion.pdf"

TREATMENT_LABELS <- c(tr1 = "Random", tr2 = "Average")
TREATMENT_COLORS <- c(tr1 = "#1f77b4", tr2 = "#d62728")

# =====
# Main
# =====
main <- function() {
  dt <- load_and_sort(INPUT_CSV)
  p <- build_plot(dt)
  save_plot(p, OUTPUT_PDF)
}

# =====
# Data loading and ordering
# =====
load_and_sort <- function(path) {
  dt <- fread(path)
  setorder(dt, alpha_mle, alpha_ci_high)
  dt[, rank := .I]
  dt[, treatment_label := TREATMENT_LABELS[treatment]]
  dt
}

# =====
# Plot construction
# =====
build_plot <- function(dt) {
  x_max <- max(dt$alpha_ci_high) + 0.02
  ggplot(dt, aes(x = alpha_mle, y = rank, color = treatment_label)) +
    geom_errorbarh(aes(xmin = alpha_ci_low, xmax = alpha_ci_high),
                   height = 0, linewidth = 0.35) +
    geom_point(size = 1.0) +
    scale_x_continuous(limits = c(0, x_max),
                       breaks = seq(0, 0.5, 0.1),
                       expand = c(0, 0)) +
    scale_y_continuous(breaks = NULL, expand = expansion(add = 1)) +
    scale_color_manual(values = c(Random = TREATMENT_COLORS[["tr1"]],
                                  Average = TREATMENT_COLORS[["tr2"]]),
                       name = "Treatment") +
    labs(x = expression(alpha[MLE]), y = "Participant (sorted)") +
    theme_minimal(base_family = "serif", base_size = 10) +
    theme(panel.grid.major.y = element_blank(),
          panel.grid.minor = element_blank(),
          legend.position = "bottom")
}

# =====
# Output
# =====
save_plot <- function(p, path) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  ggsave(path, p, width = 5.5, height = 7)
}

main()
