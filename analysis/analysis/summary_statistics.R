# Purpose: Produce summary statistics tables for the paper
# Author: Caleb Eynon w/ Claude Code
# Date: 2026-02-17
# nolint start
#
# Outputs 2 LaTeX tables to analysis/output/tables/:
#   summary_demographics_traits.tex — demographics + personality traits by treatment
#   summary_sell_rates.tex — sell rates by treatment x chat x state

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

  write_combined_demographics_traits_table(panel, survey)
  write_sell_rate_table(panel)

  cat("All summary tables written to", OUTPUT_DIR, "\n")
}

# =====
# Table 1: Combined demographics and personality traits
# =====
write_combined_demographics_traits_table <- function(panel, survey) {
  demo_rows <- c(
    build_demo_row("$N$ (subjects)", panel, count_subjects),
    build_demo_row("$N$ (sessions)", panel, count_sessions),
    build_demo_row("Age$^\\dagger$", survey, summarise_age),
    build_demo_row("Female (\\%)$^\\dagger$", survey, pct_female)
  )
  trait_rows <- map2_chr(TRAITS, TRAIT_LABELS, ~build_trait_row(survey, .x, .y))
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
  sprintf("  %s & %s & %s & %s \\\\", label, full, tr1, tr2)
}

build_trait_row <- function(survey, trait, label) {
  full <- format_mean_sd(mean(survey[[trait]]), sd(survey[[trait]]))
  tr1_vals <- survey %>% filter(treatment == "tr1") %>% pull(!!sym(trait))
  tr2_vals <- survey %>% filter(treatment == "tr2") %>% pull(!!sym(trait))
  tr1 <- format_mean_sd(mean(tr1_vals), sd(tr1_vals))
  tr2 <- format_mean_sd(mean(tr2_vals), sd(tr2_vals))
  sprintf("  %s & %s & %s & %s \\\\", label, full, tr1, tr2)
}

wrap_combined_latex <- function(demo_rows, trait_rows) {
  c(
    "\\begingroup", "\\centering", "\\small",
    "\\begin{tabular}{lccc}", "\\toprule",
    " & Full Sample & Treatment 1 & Treatment 2 \\\\",
    "\\midrule",
    demo_rows,
    "\\midrule",
    trait_rows,
    "\\bottomrule", "\\end{tabular}", "\\par", "\\vspace{2pt}",
    "{\\footnotesize \\textit{Note:} Trait values reported as Mean (SD).}\\\\",
    "{\\footnotesize $^\\dagger$Based on 95 survey respondents; one survey response was lost due to data corruption.}",
    "\\endgroup"
  ) %>% paste(collapse = "\n")
}

# =====
# Table 2: Sell rates by treatment x chat x state
# =====
write_sell_rate_table <- function(panel) {
  panel <- panel %>%
    mutate(chat = if_else(segment %in% CHAT_SEGMENTS, "Chat", "No Chat"))
  rows <- c(
    build_sell_rate_row("Sell rate (\\%)", panel),
    build_sell_rate_row("\\quad Good state", panel %>% filter(state == 1)),
    build_sell_rate_row("\\quad Bad state", panel %>% filter(state == 0))
  )
  latex <- wrap_sell_rate_latex(rows)
  write_table(latex, "summary_sell_rates.tex")
}

build_sell_rate_row <- function(label, data) {
  vals <- calc_sell_rates_by_group(data)
  sprintf("  %s & %s & %s & %s & %s \\\\",
          label, vals[1], vals[2], vals[3], vals[4])
}

calc_sell_rates_by_group <- function(data) {
  # Returns sell rates for: tr1 no chat, tr1 chat, tr2 no chat, tr2 chat
  combos <- tibble(
    treatment = c("tr1", "tr1", "tr2", "tr2"),
    chat = c("No Chat", "Chat", "No Chat", "Chat")
  )
  pmap_chr(combos, function(treatment, chat) {
    subset <- data %>% filter(.data$treatment == .env$treatment, .data$chat == .env$chat)
    sprintf("%.1f", 100 * mean(subset$did_sell))
  })
}

wrap_sell_rate_latex <- function(rows) {
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
