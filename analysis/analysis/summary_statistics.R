# Purpose: Produce summary statistics tables for the paper (demographics, traits, market outcomes)
# Author: Caleb Eynon w/ Claude Code
# Date: 2026-02-17
# nolint start
#
# Outputs 3 LaTeX tables to analysis/output/tables/:
#   summary_demographics.tex — subject pool & demographics by treatment
#   summary_traits.tex — personality trait means (SD) by treatment with t-tests
#   summary_market_outcomes.tex — market outcomes by chat condition and treatment

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
  "neuroticism", "openness", "impulsivity", "state_anxiety"
)
TRAIT_LABELS <- c(
  "Extraversion", "Agreeableness", "Conscientiousness",
  "Neuroticism", "Openness", "Impulsivity", "State Anxiety"
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

  write_demographics_table(panel, survey)
  write_traits_table(survey)
  write_market_outcomes_table(panel)

  cat("All summary tables written to", OUTPUT_DIR, "\n")
}

# =====
# Table A: Demographics
# =====
write_demographics_table <- function(panel, survey) {
  rows <- c(
    build_demo_row("$N$ (subjects)", panel, count_subjects),
    build_demo_row("$N$ (sessions)", panel, count_sessions),
    build_demo_row("Age$^\\dagger$", survey, summarise_age),
    build_demo_row("Female (\\%)$^\\dagger$", survey, pct_female)
  )
  latex <- wrap_demographics_latex(rows)
  write_table(latex, "summary_demographics.tex")
}

count_subjects <- function(data) as.character(nrow(distinct(data, session_id, player)))
count_sessions <- function(data) as.character(n_distinct(data$session_id))
summarise_age <- function(data) format_mean_sd(mean(data$age), sd(data$age))

pct_female <- function(data) {
  sprintf("%.1f", 100 * mean(data$gender == "Female"))
}

build_demo_row <- function(label, data, fn) {
  full <- fn(data)
  tr1 <- fn(data %>% filter(treatment == "tr1"))
  tr2 <- fn(data %>% filter(treatment == "tr2"))
  sprintf("  %s & %s & %s & %s \\\\", label, full, tr1, tr2)
}

wrap_demographics_latex <- function(rows) {
  c(
    "\\begingroup", "\\centering", "\\small",
    "\\begin{tabular}{lccc}", "\\toprule",
    " & Full Sample & Treatment 1 & Treatment 2 \\\\",
    "\\midrule",
    rows,
    "\\bottomrule", "\\end{tabular}",
    "\\par", "\\vspace{2pt}",
    "{\\footnotesize $^\\dagger$Based on 95 survey respondents; one survey response was lost due to data corruption.}",
    "\\endgroup"
  ) %>% paste(collapse = "\n")
}

# =====
# Table B: Personality traits by treatment
# =====
write_traits_table <- function(survey) {
  rows <- map2_chr(TRAITS, TRAIT_LABELS, ~build_trait_row(survey, .x, .y))
  latex <- wrap_traits_latex(rows)
  write_table(latex, "summary_traits.tex")
}

build_trait_row <- function(survey, trait, label) {
  full <- format_mean_sd(mean(survey[[trait]]), sd(survey[[trait]]))
  tr1_vals <- survey %>% filter(treatment == "tr1") %>% pull(!!sym(trait))
  tr2_vals <- survey %>% filter(treatment == "tr2") %>% pull(!!sym(trait))
  tr1 <- format_mean_sd(mean(tr1_vals), sd(tr1_vals))
  tr2 <- format_mean_sd(mean(tr2_vals), sd(tr2_vals))
  tt <- t.test(tr1_vals, tr2_vals)
  diff <- sprintf("%.2f%s", tt$estimate[1] - tt$estimate[2], format_stars(tt$p.value))
  sprintf("  %s & %s & %s & %s & %s \\\\", label, full, tr1, tr2, diff)
}

wrap_traits_latex <- function(rows) {
  c(
    "\\begingroup", "\\centering", "\\small",
    "\\begin{tabular}{lcccc}", "\\toprule",
    " & Full Sample & Treatment 1 & Treatment 2 & Diff ($p$) \\\\",
    "\\midrule",
    rows,
    "\\bottomrule", "\\end{tabular}", "\\par", "\\vspace{2pt}",
    "{\\footnotesize \\textit{Note:} Mean (SD). Diff column: Treatment 1 $-$ Treatment 2 with two-sample $t$-test.}\\\\",
    "{\\footnotesize ***: $p<0.01$, **: $p<0.05$, *: $p<0.1$}\\\\",
    "{\\footnotesize Based on 95 survey respondents; one survey response was lost due to data corruption.}",
    "\\endgroup"
  ) %>% paste(collapse = "\n")
}

# =====
# Table C: Market outcomes by chat condition
# =====
write_market_outcomes_table <- function(panel) {
  panel <- panel %>% mutate(chat = if_else(segment %in% CHAT_SEGMENTS, "Chat", "No Chat"))
  group_rounds <- build_group_round_summary(panel)
  rows <- c(
    build_outcome_row("Sell rate (\\%)", panel, calc_sell_rate),
    build_outcome_row("Avg.\\ sell period", panel %>% filter(did_sell == 1), calc_mean_sell_period),
    build_outcome_row("Avg.\\ sell price", panel %>% filter(did_sell == 1), calc_mean_sell_price),
    build_outcome_row("Full run (\\%)", group_rounds, calc_full_run_pct)
  )
  latex <- wrap_outcomes_latex(rows)
  write_table(latex, "summary_market_outcomes.tex")
}

build_group_round_summary <- function(panel) {
  panel %>%
    group_by(session_id, treatment, chat, segment, group_id, round) %>%
    summarise(n_sold = sum(did_sell), .groups = "drop")
}

calc_sell_rate <- function(data) sprintf("%.1f", 100 * mean(data$did_sell))
calc_mean_sell_period <- function(data) sprintf("%.2f", mean(data$sell_period))
calc_mean_sell_price <- function(data) sprintf("%.2f", mean(data$sell_price))

calc_full_run_pct <- function(data) {
  sprintf("%.1f", 100 * mean(data$n_sold == 4))
}

build_outcome_row <- function(label, data, fn) {
  no_chat <- fn(data %>% filter(chat == "No Chat"))
  chat <- fn(data %>% filter(chat == "Chat"))
  diff <- compute_outcome_diff(no_chat, chat)
  sprintf("  %s & %s & %s & %s \\\\", label, no_chat, chat, diff)
}

compute_outcome_diff <- function(no_chat_str, chat_str) {
  diff_val <- as.numeric(chat_str) - as.numeric(no_chat_str)
  sprintf("%+.2f", diff_val)
}

wrap_outcomes_latex <- function(rows) {
  c(
    "\\begingroup", "\\centering", "\\small",
    "\\begin{tabular}{lccc}", "\\toprule",
    " & No Chat (Seg.\\ 1--2) & Chat (Seg.\\ 3--4) & Difference \\\\",
    "\\midrule",
    rows,
    "\\bottomrule", "\\end{tabular}", "\\par", "\\vspace{2pt}",
    "{\\footnotesize \\textit{Note:} Difference = Chat $-$ No Chat.}",
    "\\endgroup"
  ) %>% paste(collapse = "\n")
}

# =====
# Shared helpers
# =====
format_mean_sd <- function(m, s) sprintf("%.2f (%.2f)", m, s)

format_stars <- function(p) {
  if (p < 0.01) return("***")
  if (p < 0.05) return("**")
  if (p < 0.1) return("*")
  return("")
}

write_table <- function(content, filename) {
  dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
  path <- file.path(OUTPUT_DIR, filename)
  writeLines(content, path)
  cat("Written:", path, "\n")
}

# %%
if (sys.nframe() == 0) main()
