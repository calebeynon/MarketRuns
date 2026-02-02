# Purpose: Descriptive statistics and t-tests comparing traits by first seller status
# Author: Caleb Eynon w/ Claude Code
# Date: 2025-02-01
# nolint start
#
# Creates appendix table: Mean (SD) of each trait by first seller status with t-tests

library(tidyverse)

# =====
# File paths
# =====
INPUT_PATH <- "datastore/derived/first_seller_analysis_data.csv"
OUTPUT_PATH <- "analysis/output/tables/first_seller_trait_comparisons.tex"

# =====
# Main
# =====
main <- function() {
  data <- load_data()
  unique_data <- get_unique_player_observations(data)

  summary_stats <- calculate_summary_stats(unique_data)
  t_test_results <- run_t_tests(unique_data)

  combined_table <- create_combined_table(summary_stats, t_test_results)

  print_results(combined_table)
  write_latex_table(combined_table)

  return(combined_table)
}

# =====
# Data loading
# =====
load_data <- function() {
  data <- read_csv(INPUT_PATH, show_col_types = FALSE)
  cat("Loaded", nrow(data), "observations\n")
  return(data)
}

# =====
# Get unique player-round observations to avoid repeated traits
# =====
get_unique_player_observations <- function(data) {
  # Each player appears multiple times per round; get one observation per player-group-round
  unique_data <- data %>%
    distinct(session_id, group_id, round, player, .keep_all = TRUE)

  cat("Unique player-round observations:", nrow(unique_data), "\n")
  cat("First sellers:", sum(unique_data$is_first_seller), "\n")
  cat("Non-first sellers:", sum(1 - unique_data$is_first_seller), "\n")

  return(unique_data)
}

# =====
# Calculate summary statistics by first seller status
# =====
calculate_summary_stats <- function(data) {
  traits <- c(
    "extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness", "impulsivity", "state_anxiety"
  )

  summary_stats <- data %>%
    group_by(is_first_seller) %>%
    summarise(
      across(all_of(traits), list(mean = mean, sd = sd), .names = "{.col}_{.fn}"),
      n = n(),
      .groups = "drop"
    )

  return(summary_stats)
}

# =====
# Run t-tests for each trait
# =====
run_t_tests <- function(data) {
  traits <- c(
    "extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness", "impulsivity", "state_anxiety"
  )

  results <- map_dfr(traits, function(trait) {
    run_single_t_test(data, trait)
  })

  return(results)
}

run_single_t_test <- function(data, trait) {
  first_sellers <- data %>% filter(is_first_seller == 1) %>% pull(!!sym(trait))
  non_first_sellers <- data %>% filter(is_first_seller == 0) %>% pull(!!sym(trait))

  test_result <- t.test(first_sellers, non_first_sellers)

  tibble(
    trait = trait,
    difference = mean(first_sellers) - mean(non_first_sellers),
    t_stat = test_result$statistic,
    p_value = test_result$p.value,
    df = test_result$parameter
  )
}

# =====
# Create combined table for output
# =====
create_combined_table <- function(summary_stats, t_test_results) {
  traits <- c(
    "extraversion", "agreeableness", "conscientiousness",
    "neuroticism", "openness", "impulsivity", "state_anxiety"
  )

  # Extract stats for first sellers (is_first_seller == 1)
  first_seller_stats <- summary_stats %>% filter(is_first_seller == 1)
  non_first_seller_stats <- summary_stats %>% filter(is_first_seller == 0)

  combined <- map_dfr(traits, function(trait) {
    fs_mean <- first_seller_stats[[paste0(trait, "_mean")]]
    fs_sd <- first_seller_stats[[paste0(trait, "_sd")]]
    nfs_mean <- non_first_seller_stats[[paste0(trait, "_mean")]]
    nfs_sd <- non_first_seller_stats[[paste0(trait, "_sd")]]

    t_test <- t_test_results %>% filter(trait == !!trait)

    tibble(
      trait = format_trait_name(trait),
      first_seller_mean_sd = format_mean_sd(fs_mean, fs_sd),
      non_first_seller_mean_sd = format_mean_sd(nfs_mean, nfs_sd),
      difference = round(t_test$difference, 3),
      t_stat = round(t_test$t_stat, 2),
      p_value = format_p_value(t_test$p_value)
    )
  })

  return(combined)
}

format_trait_name <- function(trait) {
  trait %>%
    str_replace_all("_", " ") %>%
    str_to_title()
}

format_mean_sd <- function(mean_val, sd_val) {
  sprintf("%.2f (%.2f)", mean_val, sd_val)
}

format_p_value <- function(p) {
  if (p < 0.001) return("<0.001")
  if (p < 0.01) return(sprintf("%.3f", p))
  sprintf("%.2f", p)
}

# =====
# Print results to console
# =====
print_results <- function(combined_table) {
  cat("\n", strrep("=", 80), "\n")
  cat("TRAIT COMPARISONS: First Sellers vs Non-First Sellers\n")
  cat(strrep("=", 80), "\n\n")

  print(combined_table, n = Inf)
}

# =====
# Write LaTeX table
# =====
write_latex_table <- function(combined_table) {
  # Ensure output directory exists
  dir.create(dirname(OUTPUT_PATH), showWarnings = FALSE, recursive = TRUE)

  latex_content <- generate_latex_content(combined_table)
  writeLines(latex_content, OUTPUT_PATH)

  cat("\nLaTeX table written to:", OUTPUT_PATH, "\n")
}

generate_latex_content <- function(combined_table) {
  latex <- c(
    "\\begingroup",
    "\\centering",
    "\\small",
    "\\begin{tabular}{lcccrr}",
    "\\toprule",
    paste0(
      "Trait & First Sellers & Non-First Sellers & ",
      "Difference & $t$-stat & $p$-value \\\\"
    ),
    "\\midrule"
  )

  # Add data rows
  for (i in seq_len(nrow(combined_table))) {
    row <- combined_table[i, ]
    latex <- c(latex, sprintf(
      "%s & %s & %s & %.3f & %.2f & %s \\\\",
      row$trait,
      row$first_seller_mean_sd,
      row$non_first_seller_mean_sd,
      row$difference,
      row$t_stat,
      row$p_value
    ))
  }

  latex <- c(
    latex,
    "\\bottomrule",
    "\\end{tabular}",
    "\\par\\endgroup",
    "",
    "% Note: Mean (SD) shown for each group. Two-sample t-tests used for comparisons."
  )

  return(paste(latex, collapse = "\n"))
}

# =====
# Run
# =====
results <- main()
