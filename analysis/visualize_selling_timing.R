#' Purpose: Visualize selling timing distribution by treatment
#' Author: Claude Code
#' Date: 2025-01-18
#'
#' Creates a bar chart showing the distribution of sales across periods,
#' comparing Treatment 1 and Treatment 2.

library(tidyverse)

# =====
# File paths
# =====
PERIOD_DATA_PATH <- "datastore/derived/individual_period_dataset.csv"
OUTPUT_PATH <- "analysis/output/plots/selling_timing_by_treatment.png"

# =====
# Main
# =====
main <- function() {
  df <- read_csv(PERIOD_DATA_PATH, show_col_types = FALSE)

  # Compute conditional distribution: P(sell in period t | reached period t)
  dist <- compute_conditional_distribution(df)

  # Create plot
  p <- create_plot(dist)

  # Save
  ggsave(OUTPUT_PATH, p, width = 10, height = 6, dpi = 150)
  message("Saved to: ", OUTPUT_PATH)
}


# =====
# Conditional distribution calculation
# =====
compute_conditional_distribution <- function(df) {
  # Period-level data: one row per player per period
  # A player "reaches" period t if already_sold == 0 (haven't sold yet)
  # They sell this period if sold == 1 AND already_sold == 0

  df <- df %>%
    mutate(sold_this_period = (sold == 1 & already_sold == 0))

  df %>%
    group_by(treatment, period) %>%
    summarise(
      n_reached = sum(already_sold == 0),
      n_sold = sum(sold_this_period),
      .groups = "drop"
    ) %>%
    mutate(
      proportion = n_sold / n_reached,
      # 95% CI using normal approximation
      se = sqrt(proportion * (1 - proportion) / n_reached),
      ci_lower = pmax(0, proportion - 1.96 * se),
      ci_upper = pmin(1, proportion + 1.96 * se)
    ) %>%
    rename(sell_period = period)
}

# =====
# Plotting
# =====
create_plot <- function(dist) {
  # Rename treatments for display
  dist <- dist %>%
    mutate(treatment_label = case_when(
      treatment == "tr1" ~ "Treatment 1",
      treatment == "tr2" ~ "Treatment 2",
      TRUE ~ treatment
    ))

  ggplot(dist, aes(x = sell_period, y = proportion, fill = treatment_label)) +
    geom_col(position = position_dodge2(preserve = "single"), width = 0.7, alpha = 0.8) +
    geom_errorbar(
      aes(ymin = ci_lower, ymax = ci_upper, group = treatment_label),
      position = position_dodge2(preserve = "single", width = 0.7, padding = 0.2),
      width = 0.3,
      linewidth = 0.4
    ) +
    scale_fill_manual(
      values = c("Treatment 1" = "#2E86AB", "Treatment 2" = "#E94F37"),
      name = NULL
    ) +
    scale_x_continuous(breaks = 1:14) +
    labs(
      x = "Period",
      y = "P(Sell | Reached Period)"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "top",
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank()
    )
}

# %%
main()
