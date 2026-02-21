# Purpose: Descriptive statistics and t-tests comparing traits by first seller status
# Author: Caleb Eynon w/ Claude Code
# Date: 2026-02-17
#
# Creates appendix table: Mean (SD) of each trait by first seller status with t-tests

library(tidyverse)

# =====
# File paths
# =====
ROUND_DATA_PATH <- "datastore/derived/first_seller_round_data.csv"
SURVEY_TRAITS_PATH <- "datastore/derived/survey_traits.csv"
OUTPUT_PATH <- "analysis/output/tables/first_seller_trait_comparisons.tex"

TRAITS <- c(
  "extraversion", "agreeableness", "conscientiousness",
  "neuroticism", "openness", "impulsivity", "state_anxiety"
)

# =====
# Main
# =====
main <- function() {
  data <- load_data()

  summary_stats <- calculate_summary_stats(data)
  t_test_results <- run_t_tests(data)
  group_counts <- get_group_counts(summary_stats)

  combined_table <- create_combined_table(summary_stats, t_test_results)

  print_results(combined_table)
  write_latex_table(combined_table, group_counts)

  return(combined_table)
}

# =====
# Data loading and preparation
# =====
load_data <- function() {
  round_data <- read_csv(ROUND_DATA_PATH, show_col_types = FALSE)
  survey_traits <- read_csv(SURVEY_TRAITS_PATH, show_col_types = FALSE)

  individual_data <- build_individual_data(round_data, survey_traits)

  cat("Individual observations:", nrow(individual_data), "\n")
  cat("Ever first seller:", sum(individual_data$is_ever_first_seller), "\n")
  cat("Never first seller:", sum(!individual_data$is_ever_first_seller), "\n")

  return(individual_data)
}

build_individual_data <- function(round_data, survey_traits) {
  player_summary <- round_data %>%
    group_by(session_id, player) %>%
    summarise(times_first_seller = sum(is_first_seller), .groups = "drop")

  individual_data <- player_summary %>%
    inner_join(survey_traits, by = c("session_id", "player")) %>%
    mutate(is_ever_first_seller = times_first_seller >= 1)

  return(individual_data)
}

# =====
# Calculate summary statistics by first seller status
# =====
calculate_summary_stats <- function(data) {
  data %>%
    group_by(is_ever_first_seller) %>%
    summarise(
      across(all_of(TRAITS), list(mean = mean, sd = sd), .names = "{.col}_{.fn}"),
      n = n(),
      .groups = "drop"
    )
}

get_group_counts <- function(summary_stats) {
  list(
    first_sellers = summary_stats$n[summary_stats$is_ever_first_seller],
    non_first_sellers = summary_stats$n[!summary_stats$is_ever_first_seller]
  )
}

# =====
# Run t-tests for each trait
# =====
run_t_tests <- function(data) {
  map_dfr(TRAITS, function(trait) {
    run_single_t_test(data, trait)
  })
}

run_single_t_test <- function(data, trait) {
  first_sellers <- data %>% filter(is_ever_first_seller) %>% pull(!!sym(trait))
  non_first_sellers <- data %>% filter(!is_ever_first_seller) %>% pull(!!sym(trait))

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
  fs_stats <- summary_stats %>% filter(is_ever_first_seller)
  nfs_stats <- summary_stats %>% filter(!is_ever_first_seller)

  map_dfr(TRAITS, function(trait) {
    build_trait_row(trait, fs_stats, nfs_stats, t_test_results)
  })
}

build_trait_row <- function(trait_name, fs_stats, nfs_stats, t_test_results) {
  t_test <- t_test_results %>% filter(trait == trait_name)
  tibble(
    trait = format_trait_name(trait_name),
    first_seller_mean_sd = format_mean_sd(
      fs_stats[[paste0(trait_name, "_mean")]],
      fs_stats[[paste0(trait_name, "_sd")]]
    ),
    non_first_seller_mean_sd = format_mean_sd(
      nfs_stats[[paste0(trait_name, "_mean")]],
      nfs_stats[[paste0(trait_name, "_sd")]]
    ),
    difference = round(t_test$difference, 3),
    stars = format_significance(t_test$p_value)
  )
}

# =====
# Formatting helpers
# =====
format_trait_name <- function(trait) {
  trait %>%
    str_replace_all("_", " ") %>%
    str_to_title()
}

format_mean_sd <- function(mean_val, sd_val) {
  sprintf("%.2f (%.2f)", mean_val, sd_val)
}

format_significance <- function(p) {
  if (p < 0.01) return("***")
  if (p < 0.05) return("**")
  if (p < 0.1) return("*")
  return("")
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
write_latex_table <- function(combined_table, group_counts) {
  dir.create(dirname(OUTPUT_PATH), showWarnings = FALSE, recursive = TRUE)

  latex_content <- generate_latex_content(combined_table, group_counts)
  writeLines(latex_content, OUTPUT_PATH)

  cat("\nLaTeX table written to:", OUTPUT_PATH, "\n")
}

generate_latex_content <- function(combined_table, group_counts) {
  n_fs <- group_counts$first_sellers
  n_nfs <- group_counts$non_first_sellers
  header <- build_latex_header(n_fs, n_nfs)
  rows <- format_latex_rows(combined_table)
  footer <- build_latex_footer(n_fs + n_nfs)
  paste(c(header, rows, footer), collapse = "\n")
}

build_latex_header <- function(n_fs, n_nfs) {
  c("\\begingroup", "\\centering", "\\small",
    "\\begin{tabular}{lccr}", "\\toprule",
    sprintf("Trait & First Sellers (N=%d) & Non-First Sellers (N=%d) & Difference \\\\",
            n_fs, n_nfs),
    "\\midrule")
}

build_latex_footer <- function(n_total) {
  c("\\bottomrule", "\\end{tabular}", "\\par", "\\vspace{2pt}",
    sprintf("{\\footnotesize \\textit{Note:} N = %d individuals. %s}\\\\",
            n_total, "First Seller = sold first 1+ times across all rounds."),
    "{\\footnotesize Mean (SD). Two-sample $t$-tests. ***: $p<0.01$, **: $p<0.05$, *: $p<0.1$}",
    "\\endgroup")
}

format_latex_rows <- function(combined_table) {
  vapply(seq_len(nrow(combined_table)), function(i) {
    row <- combined_table[i, ]
    sprintf("%s & %s & %s & %.3f%s \\\\",
      row$trait, row$first_seller_mean_sd,
      row$non_first_seller_mean_sd, row$difference, row$stars)
  }, character(1))
}

# %%
if (sys.nframe() == 0) {
  main()
}
