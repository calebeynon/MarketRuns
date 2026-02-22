# Purpose: Produce summary statistics tables for the paper
# Author: Caleb Eynon w/ Claude Code
# Date: 2026-02-17
# nolint start
#
# Outputs 2 LaTeX tables to analysis/output/tables/:
#   summary_demographics_traits.tex — demographics + personality traits by treatment
#   summary_seller_counts.tex — avg sellers per group-round, avg sell period, avg sell price by treatment x chat x state

library(tidyverse)

# =====
# File paths
# =====
PANEL_PATH <- "datastore/derived/individual_round_panel.csv"
SURVEY_PATH <- "datastore/derived/survey_traits.csv"
OUTPUT_DIR <- "analysis/output/tables"

# =====
# Constants
# =====
TRAITS <- c(
  "extraversion", "agreeableness", "conscientiousness",
  "neuroticism", "openness", "impulsivity", "state_anxiety",
  "risk_tolerance"
)
TRAIT_LABELS <- c(
  "Extraversion", "Agreeableness", "Conscientiousness",
  "Neuroticism", "Openness", "Impulsivity", "State Anxiety",
  "Risk Tolerance"
)
TRAIT_RANGES <- c(
  "[1,7]", "[1,7]", "[1,7]",
  "[1,7]", "[1,7]", "[1,7]", "[1,4]",
  "[0,20]"
)
NO_CHAT_SEGMENTS <- c(1, 2)
CHAT_SEGMENTS <- c(3, 4)

# =====
# Main
# =====
main <- function() {
  panel <- read_csv(PANEL_PATH, show_col_types = FALSE)
  survey <- read_csv(SURVEY_PATH, show_col_types = FALSE)

  # Merge treatment info into survey data
  session_treatments <- panel %>% distinct(session_id, treatment)
  survey <- survey %>% left_join(session_treatments, by = "session_id")

  write_combined_demographics_traits_table(panel, survey)
  write_seller_count_table(panel)

  cat("All summary tables written to", OUTPUT_DIR, "\n")
}

# =====
# Table 1: Combined demographics and personality traits
# =====
write_combined_demographics_traits_table <- function(panel, survey) {
  demo_rows <- c(
    build_demo_row("$N$ (subjects)", panel, count_subjects),
    build_demo_row("$N$ (sessions)", panel, count_sessions),
    build_demo_row("Age", survey, summarise_age),
    build_demo_row("Female (\\%)", survey, pct_female)
  )
  trait_rows <- pmap_chr(
    list(TRAITS, TRAIT_LABELS, TRAIT_RANGES),
    ~build_trait_row(survey, ..1, ..2, ..3)
  )
  latex <- wrap_combined_latex(demo_rows, trait_rows)
  write_table(latex, "summary_demographics_traits.tex")
}

count_subjects <- function(data) {
  as.character(nrow(distinct(data, session_id, player)))
}
count_sessions <- function(data) as.character(n_distinct(data$session_id))
summarise_age <- function(data) format_mean_sd(mean(data$age), sd(data$age))
pct_female <- function(data) sprintf("%.1f", 100 * mean(data$gender == "Female"))

build_demo_row <- function(label, data, fn) {
  full <- fn(data)
  tr1 <- fn(data %>% filter(treatment == "tr1"))
  tr2 <- fn(data %>% filter(treatment == "tr2"))
  sprintf("  %s & & %s & %s & %s \\\\", label, full, tr1, tr2)
}

build_trait_row <- function(survey, trait, label, range) {
  full <- format_mean_sd(mean(survey[[trait]]), sd(survey[[trait]]))
  tr1_vals <- survey %>% filter(treatment == "tr1") %>% pull(!!sym(trait))
  tr2_vals <- survey %>% filter(treatment == "tr2") %>% pull(!!sym(trait))
  tr1 <- format_mean_sd(mean(tr1_vals), sd(tr1_vals))
  tr2 <- format_mean_sd(mean(tr2_vals), sd(tr2_vals))
  sprintf("  %s & %s & %s & %s & %s \\\\", label, range, full, tr1, tr2)
}

wrap_combined_latex <- function(demo_rows, trait_rows) {
  c(
    "\\begingroup", "\\centering", "\\small",
    "\\begin{tabular}{lcccc}", "\\toprule",
    " & Range & Full Sample & Treatment 1 & Treatment 2 \\\\",
    "\\midrule",
    demo_rows,
    "\\midrule",
    trait_rows,
    "\\bottomrule", "\\end{tabular}", "\\par", "\\vspace{2pt}",
    "{\\footnotesize \\textit{Note:} Trait values reported as Mean (SD). Ranges of possible values are in brackets.}\\\\",
    "{\\footnotesize Based on 95 survey respondents; one survey response from Treatment 2 was lost due to data corruption.}",
    "\\endgroup"
  ) %>% paste(collapse = "\n")
}

# =====
# Table 2: Seller counts, avg sell period, avg sell price
# =====
write_seller_count_table <- function(panel) {
  panel <- panel %>%
    mutate(chat = if_else(segment %in% CHAT_SEGMENTS, "Chat", "No Chat"))
  sellers <- panel %>% filter(did_sell == 1)
  seller_count_rows <- build_stat_block("Avg sellers per group-round", panel, stat_avg_sellers)
  period_rows <- build_stat_block("Avg sell period", sellers, stat_avg_sell_period)
  price_rows <- build_stat_block("Avg sell price", sellers, stat_avg_sell_price)
  zero_seller_rows <- build_stat_block("Zero-seller group-rounds", panel, stat_zero_seller_groups)
  rows <- c(seller_count_rows, "\\midrule", period_rows, "\\midrule", price_rows, "\\midrule", zero_seller_rows)
  latex <- wrap_seller_count_latex(rows)
  write_table(latex, "summary_seller_counts.tex")
}

stat_avg_sellers <- function(data) {
  group_counts <- data %>%
    group_by(session_id, segment, group_id, round) %>%
    summarise(n_sellers = sum(did_sell), .groups = "drop")
  sprintf("%.2f", mean(group_counts$n_sellers))
}

stat_zero_seller_groups <- function(data) {
  group_counts <- data %>%
    group_by(session_id, segment, group_id, round) %>%
    summarise(n_sellers = sum(did_sell), .groups = "drop")
  as.character(sum(group_counts$n_sellers == 0))
}

stat_avg_sell_period <- function(data) sprintf("%.1f", mean(data$sell_period))
stat_avg_sell_price <- function(data) sprintf("%.1f", mean(data$sell_price))

build_stat_block <- function(label, data, stat_fn) {
  c(
    build_stat_row(label, data, stat_fn),
    build_stat_row("\\quad Good state", data %>% filter(state == 1), stat_fn),
    build_stat_row("\\quad Bad state", data %>% filter(state == 0), stat_fn)
  )
}

build_stat_row <- function(label, data, stat_fn) {
  vals <- calc_stat_by_group(data, stat_fn)
  sprintf("  %s & %s & %s & %s & %s \\\\",
          label, vals[1], vals[2], vals[3], vals[4])
}

calc_stat_by_group <- function(data, stat_fn) {
  combos <- tibble(
    treatment = c("tr1", "tr1", "tr2", "tr2"),
    chat = c("No Chat", "Chat", "No Chat", "Chat")
  )
  pmap_chr(combos, function(treatment, chat) {
    subset <- data %>% filter(.data$treatment == .env$treatment, .data$chat == .env$chat)
    stat_fn(subset)
  })
}

wrap_seller_count_latex <- function(rows) {
  c(
    "\\begingroup", "\\centering", "\\small",
    "\\begin{tabular}{lcccc}", "\\toprule",
    " & \\multicolumn{2}{c}{Treatment 1} & \\multicolumn{2}{c}{Treatment 2} \\\\",
    "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
    " & No Chat & Chat & No Chat & Chat \\\\",
    "\\midrule",
    rows,
    "\\bottomrule", "\\end{tabular}",
    "\\endgroup"
  ) %>% paste(collapse = "\n")
}

# =====
# Shared helpers
# =====
format_mean_sd <- function(m, s) sprintf("%.2f (%.2f)", m, s)

write_table <- function(content, filename) {
  dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
  path <- file.path(OUTPUT_DIR, filename)
  writeLines(content, path)
  cat("Written:", path, "\n")
}

# %%
if (sys.nframe() == 0) main()
