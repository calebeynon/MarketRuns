# Purpose: Unified regression table combining cascade effects, emotions (FE), and traits (RE)
# Author: Claude Code
# Date: 2026-02-06

library(data.table)
library(fixest)
library(plm)

# FILE PATHS
INPUT_PATH <- "datastore/derived/emotions_traits_selling_dataset.csv"
OUTPUT_PATH <- "analysis/output/tables/unified_selling_regression.tex"
OUTPUT_PATH_FULL <- "analysis/output/tables/unified_selling_regression_full.tex"

# VARIABLE LISTS
SHOW_EMOTIONS <- c("fear_mean", "anger_mean")
SHOW_TRAITS <- c("state_anxiety", "impulsivity", "conscientiousness")
HIDE_EMOTIONS <- c("contempt_mean", "disgust_mean", "joy_mean", "sadness_mean",
                   "surprise_mean", "engagement_mean", "valence_mean")
HIDE_TRAITS <- c("extraversion", "agreeableness", "neuroticism", "openness")

ALL_EMOTIONS <- c(SHOW_EMOTIONS, HIDE_EMOTIONS)
ALL_TRAITS <- c(SHOW_TRAITS, HIDE_TRAITS)

VAR_LABELS <- c(
  "(Intercept)" = "Constant",
  dummy_1_cum = "Exactly 1 prior sale",
  dummy_2_cum = "Exactly 2 prior sales",
  dummy_3_cum = "Exactly 3 prior sales",
  dummy_prev_period = "Sale in previous period",
  fear_mean = "Fear", anger_mean = "Anger",
  state_anxiety = "State anxiety", impulsivity = "Impulsivity",
  conscientiousness = "Conscientiousness",
  age = "Age", gender_female = "Female",
  signal = "Signal", period = "Period", round = "Round",
  segment2 = "Segment 2", segment3 = "Segment 3", segment4 = "Segment 4",
  treatmenttr2 = "Treatment 2"
)

# Source panel scripts
source("analysis/analysis/unified_selling_regression_panel_a.R")
source("analysis/analysis/unified_selling_regression_panel_b.R")
source("analysis/analysis/unified_selling_regression_panel_c.R")

# =====
# Main function
# =====
main <- function() {
  cat("Loading data from:", INPUT_PATH, "\n")
  df <- prepare_base_data(INPUT_PATH)
  df_full <- copy(df)
  df_em <- df[complete.cases(df[, .SD, .SDcols = ALL_EMOTIONS])]
  cat("Base data:", nrow(df), "rows | After emotion filter:", nrow(df_em), "\n")

  cat("\n=== Running Panel A: All Participants ===\n")
  panel_a <- run_panel_a(df_em)

  cat("\n=== Running Panel B: Second Sellers ===\n")
  panel_b <- run_panel_b(df_em, df_full)

  cat("\n=== Running Panel C: First Sellers ===\n")
  panel_c <- run_panel_c(df_em)

  cat("\n=== Building condensed table (main body) ===\n")
  build_unified_table(panel_a, panel_b, panel_c, compact = TRUE)

  cat("\n=== Building full table (appendix) ===\n")
  build_unified_table(panel_a, panel_b, panel_c, compact = FALSE)

  cat("Done!\n")
}

# =====
# Data preparation
# =====
prepare_base_data <- function(file_path) {
  df <- fread(file_path)
  df <- df[already_sold == 0]
  df[, player_id := paste(session_id, player, sep = "_")]
  df[, global_group_id := paste(session_id, segment, group_id, sep = "_")]
  df[, group_round_id := paste(session_id, segment, group_id, round, sep = "_")]
  df[, player_group_round_id := paste(player_id, segment, group_id, round, sep = "_")]
  df[, time_id := paste(segment, round, period, sep = "_")]
  df[, dummy_1_cum := as.integer(prior_group_sales == 1)]
  df[, dummy_2_cum := as.integer(prior_group_sales == 2)]
  df[, dummy_3_cum := as.integer(prior_group_sales == 3)]
  df[, gender_female := as.integer(gender == "Female")]
  df[, segment := as.factor(segment)]
  df[, treatment := as.factor(treatment)]
  df
}

# =====
# Coefficient extraction (handles both feols and plm)
# =====
extract_coefs <- function(model) {
  if (inherits(model, "fixest")) {
    ct <- coeftable(model)
    data.table(var = rownames(ct), est = ct[, 1], se = ct[, 2], pval = ct[, 4])
  } else {
    ct <- coef(summary(model))
    data.table(var = rownames(ct), est = ct[, 1], se = ct[, 2], pval = ct[, 4])
  }
}

extract_fit <- function(model) {
  if (inherits(model, "fixest")) {
    list(n = model$nobs, r2 = fitstat(model, "wr2")[[1]])
  } else {
    s <- summary(model)
    list(n = nobs(model), r2 = s$r.squared[1])
  }
}

# =====
# LaTeX table builder
# =====
CONTROLS <- c("signal", "period", "round", "segment2", "segment3", "segment4",
              "age", "gender_female", "(Intercept)")
EMOTION_HEADER <- "__header__Facial emotions"
TRAIT_HEADER <- "__header__Personality traits"
PERSON_VARS <- c(EMOTION_HEADER, SHOW_EMOTIONS,
                 TRAIT_HEADER, SHOW_TRAITS, "treatmenttr2")

build_unified_table <- function(panel_a, panel_b, panel_c, compact) {
  vars <- get_panel_vars(compact)
  tbl <- get_table_config(compact)
  lines <- c(build_preamble(tbl$caption, tbl$label, compact), build_col_header(compact))
  lines <- c(lines, build_panel("Panel A: All Participants", panel_a, vars$a))
  lines <- c(lines, build_panel("Panel B: Second Sellers", panel_b, vars$b))
  lines <- c(lines, build_panel("Panel C: First Sellers", panel_c, vars$c))
  lines <- c(lines, if (compact) build_footer_compact() else build_footer_full())
  write_table(lines, tbl$path)
}

get_panel_vars <- function(compact) {
  if (compact) {
    list(a = c("dummy_1_cum", "dummy_2_cum", "dummy_3_cum", PERSON_VARS),
         b = c("dummy_prev_period", PERSON_VARS),
         c = PERSON_VARS)
  } else {
    list(a = c("dummy_1_cum", "dummy_2_cum", "dummy_3_cum", PERSON_VARS, CONTROLS),
         b = c("dummy_prev_period", PERSON_VARS, CONTROLS),
         c = c(PERSON_VARS, CONTROLS))
  }
}

get_table_config <- function(compact) {
  if (compact) {
    list(caption = "Determinants of selling probability",
         label = "unified_selling_regression",
         path = OUTPUT_PATH)
  } else {
    list(caption = "Determinants of selling probability (full results)",
         label = "unified_selling_regression_full",
         path = OUTPUT_PATH_FULL)
  }
}

build_preamble <- function(caption, label, compact = FALSE) {
  font <- if (compact) "\\tiny" else "\\scriptsize"
  spacing <- if (compact) "\\renewcommand{\\arraystretch}{0.75}" else ""
  c("",
    "\\begingroup",
    font, spacing,
    "\\begin{longtable}{l*{3}{>{\\centering\\arraybackslash}p{3.2cm}}}",
    sprintf("\\caption{%s} \\label{tab:%s} \\\\", caption, label))
}

build_col_header <- function(compact = FALSE) {
  hdr <- c("   \\midrule \\midrule",
           "   & (1) & (2) & (3) \\\\",
           "   & Random Effects & Individual FE & Random Effects \\\\",
           "   \\midrule")
  if (compact) return(c(hdr, "\\endfirsthead", "\\endlastfoot"))
  c(hdr, "\\endfirsthead",
    "\\multicolumn{4}{l}{\\emph{(continued)}} \\\\",
    hdr, "\\endhead",
    "   \\midrule",
    "   \\multicolumn{4}{r}{\\emph{continued on next page}} \\\\",
    "\\endfoot", "\\endlastfoot")
}

build_panel <- function(title, models, var_order) {
  coefs <- lapply(models, extract_coefs)
  fits <- lapply(models, extract_fit)
  lines <- c(
    sprintf("\\multicolumn{4}{l}{\\emph{%s}} \\\\", title),
    "   \\midrule")
  for (v in var_order) {
    if (startsWith(v, "__header__")) {
      label <- sub("__header__", "", v)
      lines <- c(lines, sprintf("   \\emph{%s} & & & \\\\", label))
    } else {
      lines <- c(lines, format_coef_row(v, coefs))
    }
  }
  lines <- c(lines, format_fit_rows(fits))
  c(lines, "   \\midrule")
}

format_coef_row <- function(var_name, coefs_list) {
  label <- if (var_name %in% names(VAR_LABELS)) VAR_LABELS[var_name] else gsub("_", "\\\\_", var_name)
  vals <- character(3)
  ses <- character(3)
  for (i in seq_along(coefs_list)) {
    row <- coefs_list[[i]][var == var_name]
    if (nrow(row) == 0) {
      vals[i] <- ""
      ses[i] <- ""
    } else {
      vals[i] <- paste0(sprintf("%.4f", row$est), get_stars(row$pval))
      ses[i] <- paste0("(", sprintf("%.4f", row$se), ")")
    }
  }
  c(sprintf("   %-25s & %s & %s & %s \\\\", label, vals[1], vals[2], vals[3]),
    sprintf("   %-25s & %s & %s & %s \\\\", "", ses[1], ses[2], ses[3]))
}

format_fit_rows <- function(fits) {
  ns <- sapply(fits, function(f) format(f$n, big.mark = ","))
  r2s <- sapply(fits, function(f) sprintf("%.4f", f$r2))
  c("   \\midrule",
    "   \\emph{Fit statistics} & & & \\\\",
    sprintf("   Observations %s & %s & %s & %s \\\\", "", ns[1], ns[2], ns[3]),
    sprintf("   R$^2$ %s & %s & %s & %s \\\\", "", r2s[1], r2s[2], r2s[3]))
}

get_stars <- function(pval) {
  if (pval < 0.01) return("$^{***}$")
  if (pval < 0.05) return("$^{**}$")
  if (pval < 0.1) return("$^{*}$")
  ""
}

build_footer_compact <- function() {
  c(paste0("   \\multicolumn{4}{l}{\\emph{Controls: signal, period, round,",
           " segment indicators, age, gender. Full results in",
           " Appendix Table \\ref{tab:unified_selling_regression_full}.}} \\\\"),
    build_footer_common())
}

build_footer_full <- function() {
  c(paste0("   \\multicolumn{4}{l}{\\emph{Standard errors in parentheses.",
           " (1) \\& (3): RE with individual-level effects.",
           " (2): individual FE, clustered by group.}} \\\\"),
    build_footer_common())
}

build_footer_common <- function() {
  c("   \\multicolumn{4}{l}{\\emph{Signif. Codes: ***: 0.01, **: 0.05, *: 0.1}} \\\\",
    "\\end{longtable}",
    "\\endgroup",
    "",
    "")
}

write_table <- function(lines, path) {
  output_dir <- dirname(path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  writeLines(lines, path)
  cat("Table exported to:", path, "\n")
}

# %%
if (!interactive()) main()
