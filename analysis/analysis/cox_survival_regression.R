# Purpose: Cox proportional hazards survival regression (robustness check)
# Author: Claude Code
# Date: 2026-02-22

library(data.table)
library(survival)
library(coxme)

# FILE PATHS
INPUT_PATH <- "datastore/derived/emotions_traits_selling_dataset.csv"
OUTPUT_PATH <- "analysis/output/tables/cox_survival_regression.tex"

# Source shared constants and helpers
source("analysis/analysis/selling_regression_helpers.R")
source("analysis/analysis/unified_selling_regression_panel_c.R")
source("analysis/analysis/cox_survival_panel_a.R")
source("analysis/analysis/cox_survival_panel_b.R")

# Controls to display (no period — it is the survival time axis)
CONTROLS <- c("signal", "round", "segment2", "segment3", "segment4",
              "age", "gender_female")

# =====
# Main function
# =====
main <- function() {
  df <- prepare_base_data(INPUT_PATH)
  df_em <- df[complete.cases(df[, .SD, .SDcols = ALL_EMOTIONS])]
  cat("Base:", nrow(df), "| Emotion-complete:", nrow(df_em), "\n")

  panel_a <- run_cox_panel_a(df_em)
  panel_b <- run_cox_panel_b(df_em)

  lines <- build_cox_table(panel_a, panel_b)
  write_table(lines, OUTPUT_PATH)
  cat("Done.\n")
}

# =====
# Data preparation (mirrors unified_selling_logit.R)
# =====
prepare_base_data <- function(file_path) {
  df <- fread(file_path)
  df <- df[already_sold == 0]
  df[, player_id := paste(session_id, player, sep = "_")]
  df[, global_group_id := paste(session_id, segment, group_id, sep = "_")]
  df[, group_round_id := paste(session_id, segment, group_id, round,
                                sep = "_")]
  df[, player_group_round_id := paste(player_id, segment, group_id,
                                       round, sep = "_")]
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
# Coefficient extraction (hazard ratios via delta method)
# =====
extract_cox_coefs <- function(model) {
  beta <- fixef(model)
  v <- as.matrix(vcov(model))
  hr <- exp(beta)
  # Delta method: SE(exp(b)) = exp(b) * SE(b)
  hr_se <- hr * sqrt(diag(v))
  z <- beta / sqrt(diag(v))
  pval <- 2 * pnorm(-abs(z))
  # Map factor names to match VAR_LABELS
  nms <- normalize_cox_names(names(beta))
  data.table(var = nms, est = hr, se = hr_se, pval = pval)
}

# =====
# Map coxme coefficient names to table variable names
# =====
normalize_cox_names <- function(nms) {
  nms <- sub("^segment", "segment", nms)
  nms <- sub("^treatment", "treatment", nms)
  nms
}

# =====
# Fit statistics extraction
# =====
extract_cox_fit <- function(model) {
  # coxme$n is c(events, n_obs)
  events <- model$n[1]
  n <- model$n[2]
  n_groups <- length(ranef(model)$player_id)
  ll <- model$loglik[2]
  list(n = n, events = events, n_groups = n_groups, log_lik = ll)
}

# =====
# Table builder
# =====
build_cox_table <- function(panel_a, panel_b) {
  vars_a <- get_panel_a_vars()
  vars_b <- get_panel_b_vars()
  c(
    build_preamble(),
    build_col_header(),
    build_panel("Panel A: All Participants", panel_a, vars_a),
    build_panel("Panel B: First Sellers", panel_b, vars_b),
    build_footer()
  )
}

get_panel_a_vars <- function() {
  cascade <- c("dummy_1_cum", "dummy_2_cum", "dummy_3_cum")
  int_vars <- c(INTERACTION_HEADER, INTERACTION_VARS)
  c(cascade, int_vars, ALL_PERSON_VARS, CONTROLS)
}

get_panel_b_vars <- function() {
  c(ALL_PERSON_VARS, CONTROLS)
}

build_preamble <- function() {
  caption <- paste0("Determinants of selling probability --- ",
                     "Cox survival regression (hazard ratios)")
  c("", "\\begingroup", "\\scriptsize",
    "\\begin{longtable}{l*{3}{>{\\centering\\arraybackslash}p{3.2cm}}}",
    sprintf("\\caption{%s} \\label{tab:cox_survival_regression} \\\\",
            caption))
}

build_col_header <- function() {
  hdr <- c("   \\midrule \\midrule",
           "   & (1) & (2) & (3) \\\\",
           "   & RE Cox & RE Cox & RE Cox \\\\",
           "   \\midrule")
  c(hdr, "\\endfirsthead",
    "\\multicolumn{4}{l}{\\emph{(continued)}} \\\\",
    hdr, "\\endhead",
    "   \\midrule",
    "   \\multicolumn{4}{r}{\\emph{continued on next page}} \\\\",
    "\\endfoot", "\\endlastfoot")
}

build_panel <- function(title, models, var_order) {
  coefs <- lapply(models, extract_cox_coefs)
  fits <- lapply(models, extract_cox_fit)
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
  lines <- c(lines, format_cox_fit_rows(fits))
  c(lines, "   \\midrule")
}

format_cox_fit_rows <- function(fits) {
  ns <- sapply(fits, function(f) format(f$n, big.mark = ","))
  evts <- sapply(fits, function(f) format(f$events, big.mark = ","))
  ngs <- sapply(fits, function(f) format(f$n_groups, big.mark = ","))
  lls <- sapply(fits, function(f) sprintf("%.1f", f$log_lik))
  c("   \\midrule",
    "   \\emph{Fit statistics} & & & \\\\",
    sprintf("   %-25s & %s & %s & %s \\\\",
            "Observations", ns[1], ns[2], ns[3]),
    sprintf("   %-25s & %s & %s & %s \\\\",
            "Events", evts[1], evts[2], evts[3]),
    sprintf("   %-25s & %s & %s & %s \\\\",
            "Participants", ngs[1], ngs[2], ngs[3]),
    sprintf("   %-25s & %s & %s & %s \\\\",
            "Log-likelihood", lls[1], lls[2], lls[3]))
}

build_footer <- function() {
  c(paste0("   \\multicolumn{4}{l}{\\emph{Hazard ratios reported.",
           " All models: random-intercept Cox (coxme).}} \\\\"),
    paste0("   \\multicolumn{4}{l}{\\emph{HR $>$ 1: increased hazard of",
           " selling (sells sooner). HR $<$ 1: decreased hazard",
           " (sells later or not at all).}} \\\\"),
    "   \\multicolumn{4}{l}{\\emph{Signif. Codes: ***: 0.01, **: 0.05, *: 0.1}} \\\\",
    "\\end{longtable}", "\\endgroup", "", "")
}

# %%
if (!interactive()) main()
