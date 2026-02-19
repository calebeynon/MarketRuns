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

# Source shared constants and helpers
source("analysis/analysis/selling_regression_helpers.R")

# LPM-specific: add Intercept label
VAR_LABELS["(Intercept)"] <- "Constant"

# Source panel and layout scripts
source("analysis/analysis/unified_selling_regression_panel_a.R")
source("analysis/analysis/unified_selling_regression_panel_b.R")
source("analysis/analysis/unified_selling_regression_panel_c.R")
source("analysis/analysis/unified_selling_regression_landscape.R")

# =====
# Main function
# =====
main <- function() {
  df <- prepare_base_data(INPUT_PATH)
  df_full <- copy(df)
  df_em <- df[complete.cases(df[, .SD, .SDcols = ALL_EMOTIONS])]
  cat("Base:", nrow(df), "rows | Emotion-complete:", nrow(df_em), "\n")

  panel_a <- run_panel_a(df_em)
  panel_b <- run_panel_b(df_em, df_full)
  panel_c <- run_panel_c(df_em)

  build_unified_table(panel_a, panel_b, panel_c, compact = TRUE)
  build_unified_table(panel_a, panel_b, panel_c, compact = FALSE)
  cat("Done.\n")
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
  df <- create_prev_period_dummies(df)
  df <- create_interaction_terms(df)
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

build_unified_table <- function(panel_a, panel_b, panel_c, compact) {
  if (compact) {
    build_landscape_table(panel_a, panel_b, panel_c)
    return(invisible(NULL))
  }
  vars <- get_panel_vars(compact)
  tbl <- get_table_config(compact)
  lines <- c(build_preamble(tbl$caption, tbl$label),
             build_col_header())
  lines <- c(lines, build_panel("Panel A: All Participants", panel_a, vars$a))
  lines <- c(lines, build_panel("Panel B: Second Sellers", panel_b, vars$b))
  lines <- c(lines, build_panel("Panel C: First Sellers", panel_c, vars$c))
  lines <- c(lines, build_footer_full())
  write_table(lines, tbl$path)
}

get_panel_vars <- function(compact) {
  cum_vars <- c("dummy_1_cum", "dummy_2_cum", "dummy_3_cum")
  int_vars <- c(INTERACTION_HEADER, INTERACTION_VARS)
  if (compact) {
    list(a = c(cum_vars, int_vars, PERSON_VARS),
         b = c("dummy_prev_period", PERSON_VARS),
         c = PERSON_VARS)
  } else {
    list(a = c(cum_vars, int_vars, PERSON_VARS, CONTROLS),
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

build_preamble <- function(caption, label) {
  c("", "\\begingroup", "\\scriptsize",
    "\\begin{longtable}{l*{3}{>{\\centering\\arraybackslash}p{3.2cm}}}",
    sprintf("\\caption{%s} \\label{tab:%s} \\\\", caption, label))
}

build_col_header <- function() {
  hdr <- c("   \\midrule \\midrule",
           "   & (1) & (2) & (3) \\\\",
           "   & Random Effects & Individual FE & Random Effects \\\\",
           "   \\midrule")
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

format_fit_rows <- function(fits) {
  ns <- sapply(fits, function(f) format(f$n, big.mark = ","))
  r2s <- sapply(fits, function(f) sprintf("%.4f", f$r2))
  c("   \\midrule",
    "   \\emph{Fit statistics} & & & \\\\",
    sprintf("   Observations %s & %s & %s & %s \\\\", "", ns[1], ns[2], ns[3]),
    sprintf("   R$^2$ %s & %s & %s & %s \\\\", "", r2s[1], r2s[2], r2s[3]))
}

build_footer_full <- function() {
  c(paste0("   \\multicolumn{4}{l}{\\emph{Standard errors in parentheses.",
           " (1) \\& (3): RE with individual-level effects.",
           " (2): individual FE, clustered by group.}} \\\\"),
    "   \\multicolumn{4}{l}{\\emph{Signif. Codes: ***: 0.01, **: 0.05, *: 0.1}} \\\\",
    "\\end{longtable}", "\\endgroup", "", "")
}

# %%
if (!interactive()) main()
