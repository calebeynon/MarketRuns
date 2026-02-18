# Purpose: CI difference and robustness plots for first seller personality traits
# Author: Caleb Eynon w/ Claude Code
# Date: 2026-02-17

library(data.table)
library(ggplot2)

# =====
# File paths
# =====
ROUND_DATA_PATH <- "datastore/derived/first_seller_round_data.csv"
SURVEY_TRAITS_PATH <- "datastore/derived/survey_traits.csv"
OUTPUT_DIR <- "analysis/output/plots"
OUTPUT_CI_DIFF <- file.path(OUTPUT_DIR, "first_seller_trait_ci_diff.pdf")
OUTPUT_ROBUSTNESS <- file.path(OUTPUT_DIR, "first_seller_trait_robustness.pdf")

TRAITS <- c(
  "extraversion", "agreeableness", "conscientiousness",
  "neuroticism", "openness", "impulsivity", "state_anxiety"
)

GROUP_PALETTE <- c(
  "0 times" = "#0072B2", "1-2 times" = "#E69F00", "3+ times" = "#D55E00"
)

# =====
# Main function
# =====
main <- function() {
  ensure_output_dir()
  round_data <- load_round_data()
  survey <- load_survey_traits()
  individual <- build_individual_data(round_data, survey)

  cat("Creating CI difference plot...\n")
  diff_dt <- compute_trait_differences(individual)
  p_diff <- create_ci_diff_plot(diff_dt)
  save_plot(p_diff, OUTPUT_CI_DIFF, width = 7, height = 5)

  cat("Creating robustness plot...\n")
  group_dt <- compute_group_means(individual)
  p_robust <- create_robustness_plot(group_dt)
  save_plot(p_robust, OUTPUT_ROBUSTNESS, width = 10, height = 6)

  cat("All plots saved to:", OUTPUT_DIR, "\n")
}

# =====
# Data loading
# =====
load_round_data <- function() {
  dt <- fread(ROUND_DATA_PATH)
  cat("Loaded", nrow(dt), "round-level rows\n")
  return(dt)
}

load_survey_traits <- function() {
  dt <- fread(SURVEY_TRAITS_PATH)
  cat("Loaded", nrow(dt), "survey responses\n")
  return(dt)
}

# =====
# Build individual-level dataset
# =====
build_individual_data <- function(round_data, survey) {
  counts <- round_data[,
    .(times_first_seller = sum(is_first_seller)),
    by = .(session_id, player)
  ]
  merged <- merge(counts, survey, by = c("session_id", "player"))
  merged[, is_ever_first_seller := (times_first_seller >= 1)]
  merged[, fs_group := cut_fs_group(times_first_seller)]
  cat("Individual-level N:", nrow(merged), "\n")
  return(merged)
}

cut_fs_group <- function(x) {
  factor(
    ifelse(x == 0, "0 times", ifelse(x <= 2, "1-2 times", "3+ times")),
    levels = c("0 times", "1-2 times", "3+ times")
  )
}

# =====
# Compute trait differences (ever first seller vs. never)
# =====
compute_trait_differences <- function(individual) {
  results <- rbindlist(lapply(TRAITS, function(trait) {
    compute_single_diff(individual, trait)
  }))
  return(results)
}

compute_single_diff <- function(dt, trait) {
  fs_vals <- dt[is_ever_first_seller == TRUE, get(trait)]
  nfs_vals <- dt[is_ever_first_seller == FALSE, get(trait)]
  tt <- t.test(fs_vals, nfs_vals)
  data.table(
    trait = format_trait_label(trait),
    estimate = tt$estimate[1] - tt$estimate[2],
    ci_lower = tt$conf.int[1],
    ci_upper = tt$conf.int[2]
  )
}

format_trait_label <- function(trait) {
  tools::toTitleCase(gsub("_", " ", trait))
}

# =====
# CI difference plot (Plot 1)
# =====
create_ci_diff_plot <- function(diff_dt) {
  diff_dt[, trait := factor(trait, levels = rev(diff_dt$trait))]

  ggplot(diff_dt, aes(x = estimate, y = trait)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_errorbarh(
      aes(xmin = ci_lower, xmax = ci_upper),
      height = 0.25, linewidth = 0.5
    ) +
    geom_point(size = 2.5, shape = 16) +
    labs(x = NULL, y = NULL) +
    theme_minimal() +
    theme(panel.grid.minor = element_blank())
}

# =====
# Compute group-level means and CIs for robustness plot
# =====
compute_group_means <- function(individual) {
  results <- rbindlist(lapply(TRAITS, function(trait) {
    compute_trait_group_ci(individual, trait)
  }))
  return(results)
}

compute_trait_group_ci <- function(dt, trait) {
  rbindlist(lapply(levels(dt$fs_group), function(grp) {
    vals <- dt[fs_group == grp, get(trait)]
    ci_from_values(vals, trait, grp)
  }))
}

ci_from_values <- function(vals, trait, group) {
  n <- length(vals)
  m <- mean(vals)
  se <- sd(vals) / sqrt(n)
  crit <- qt(0.975, n - 1)
  data.table(
    trait_label = format_trait_label(trait),
    fs_group = factor(group, levels = c("0 times", "1-2 times", "3+ times")),
    mean_score = m,
    ci_lower = m - crit * se,
    ci_upper = m + crit * se
  )
}

# =====
# Robustness 3-group plot (Plot 2)
# =====
create_robustness_plot <- function(group_dt) {
  ggplot(group_dt, aes(x = mean_score, y = fs_group, color = fs_group)) +
    geom_pointrange(
      aes(xmin = ci_lower, xmax = ci_upper),
      size = 0.5, fatten = 3
    ) +
    facet_wrap(~trait_label) +
    scale_color_manual(values = GROUP_PALETTE, name = "Times First Seller") +
    labs(x = "Mean Score", y = NULL) +
    theme_minimal() +
    theme(
      panel.grid.minor = element_blank(),
      legend.position = "bottom"
    )
}

# =====
# Utility functions
# =====
ensure_output_dir <- function() {
  if (!dir.exists(OUTPUT_DIR)) {
    dir.create(OUTPUT_DIR, recursive = TRUE)
  }
}

save_plot <- function(p, path, width = 8, height = 6) {
  ggsave(path, p, width = width, height = height)
  cat("Saved:", path, "\n")
}

# %%
if (sys.nframe() == 0) {
  main()
}
