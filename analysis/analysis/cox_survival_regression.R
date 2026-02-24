# Purpose: Cox proportional hazards survival regression (Section 4.2.2)
# Author: Claude Code
# Date: 2026-02-23

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

# Discrete emotions (excludes composite valence metric)
DISCRETE_EMOTIONS <- c("fear_mean", "anger_mean", "contempt_mean",
                        "disgust_mean", "joy_mean", "sadness_mean",
                        "surprise_mean", "engagement_mean")

# Controls to display (no period — it is the survival time axis)
CONTROLS <- c("signal", "round", "segment2", "segment3", "segment4",
              "treatmenttr2", "age", "gender_female")

# =====
# Main function
# =====
main <- function() {
  df <- prepare_base_data(INPUT_PATH)
  df_em <- df[complete.cases(df[, .SD, .SDcols = c(ALL_EMOTIONS, ALL_TRAITS)])]
  cat("Base:", nrow(df), "| Emotion+trait-complete:", nrow(df_em), "\n")

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
  df <- add_id_columns(df)
  df <- add_regression_variables(df)
  df
}

add_id_columns <- function(df) {
  df[, player_id := paste(session_id, player, sep = "_")]
  df[, global_group_id := paste(session_id, segment, group_id, sep = "_")]
  df[, group_round_id := paste(session_id, segment, group_id, round,
                                sep = "_")]
  df[, player_group_round_id := paste(player_id, segment, group_id,
                                       round, sep = "_")]
  df[, time_id := paste(segment, round, period, sep = "_")]
  df
}

add_regression_variables <- function(df) {
  df[, dummy_1_cum := as.integer(prior_group_sales == 1)]
  df[, dummy_2_cum := as.integer(prior_group_sales == 2)]
  df[, dummy_3_cum := as.integer(prior_group_sales == 3)]
  df <- create_prev_period_dummies(df)
  df <- create_interaction_terms(df)
  df[, gender_female := as.integer(gender == "Female")]
  df[, segment := as.factor(segment)]
  df[, treatment := as.factor(treatment)]
  # Counting process start time: each row covers interval (period-1, period]
  df[, period_start := period - 1L]
  df
}

# =====
# coxph starting values to stabilize coxme convergence
# =====
get_coxph_init <- function(formula, data, cap = 5) {
  fe_formula <- remove_random_effect(formula)
  m <- coxph(fe_formula, data = data)
  init <- coef(m)
  init[abs(init) > cap] <- sign(init[abs(init) > cap]) * cap
  init
}

remove_random_effect <- function(formula) {
  terms <- as.character(formula)
  rhs <- gsub("\\+\\s*\\(1\\s*\\|\\s*\\w+\\)", "", terms[3])
  as.formula(paste(terms[2], "~", rhs))
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
  events <- model$n[1]
  n <- model$n[2]
  n_groups <- length(ranef(model)$player_id)
  ll <- model$loglik[2]
  list(n = n, events = events, n_groups = n_groups, log_lik = ll)
}

# =====
# Variable order for flat 4-column table
# =====
get_var_order <- function() {
  cascade <- c("dummy_1_cum", "dummy_2_cum", "dummy_3_cum")
  int_vars <- INTERACTION_VARS
  c(cascade, int_vars, DISCRETE_EMOTIONS,
    "valence_mean", TRAIT_HEADER, ALL_TRAITS, CONTROLS)
}

# =====
# 4-column LaTeX coefficient formatting (local — does not modify shared helpers)
# =====
format_cox_coef_row <- function(var_name, coefs_list) {
  label <- if (var_name %in% names(VAR_LABELS)) {
    VAR_LABELS[var_name]
  } else {
    gsub("_", "\\\\_", var_name)
  }
  cells <- format_cox_coef_cells(var_name, coefs_list)
  c(sprintf("   %-25s & %s & %s & %s & %s \\\\",
            label, cells$v[1], cells$v[2], cells$v[3], cells$v[4]),
    sprintf("   %-25s & %s & %s & %s & %s \\\\",
            "", cells$s[1], cells$s[2], cells$s[3], cells$s[4]))
}

format_cox_coef_cells <- function(var_name, coefs_list) {
  vals <- ses <- character(4)
  for (i in seq_along(coefs_list)) {
    row <- coefs_list[[i]][var == var_name]
    if (nrow(row) == 0) next
    vals[i] <- paste0(sprintf("%.4f", row$est), get_stars(row$pval))
    ses[i] <- paste0("(", sprintf("%.4f", row$se), ")")
  }
  list(v = vals, s = ses)
}

# =====
# Table builder (single flat table, 4 columns)
# =====
build_cox_table <- function(panel_a, panel_b) {
  all_models <- list(panel_a$m1, panel_a$m2, panel_b$m1, panel_b$m2)
  coefs <- lapply(all_models, extract_cox_coefs)
  fits <- lapply(all_models, extract_cox_fit)
  var_order <- get_var_order()

  lines <- c(build_preamble(), build_col_header())
  for (v in var_order) {
    if (startsWith(v, "__header__")) {
      label <- sub("__header__", "", v)
      lines <- c(lines, sprintf("   \\emph{%s} & & & & \\\\", label))
    } else {
      lines <- c(lines, format_cox_coef_row(v, coefs))
    }
  }
  c(lines, format_cox_fit_rows(fits), build_footer())
}

build_preamble <- function() {
  caption <- "Cox survival regression (hazard ratios)"
  c("", "\\begingroup", "\\centering", "\\scriptsize",
    "\\setlength{\\LTcapwidth}{\\textwidth}",
    paste0("\\begin{longtable}{l",
           "*{4}{>{\\centering\\arraybackslash}p{2.2cm}}}"),
    sprintf("\\caption{%s} \\label{tab:cox_survival_regression} \\\\",
            caption))
}

build_col_header <- function() {
  hdr <- c(
    "   \\midrule \\midrule",
    "   & \\multicolumn{2}{c}{All Sellers} & \\multicolumn{2}{c}{First Sellers} \\\\",
    "   \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
    "   & (1) & (2) & (3) & (4) \\\\",
    "   & All Emotions & Valence & All Emotions & Valence \\\\",
    "   \\midrule")
  c(hdr, "\\endfirsthead",
    "\\multicolumn{5}{l}{\\emph{(continued)}} \\\\",
    hdr, "\\endhead",
    "   \\midrule",
    "   \\multicolumn{5}{r}{\\emph{continued on next page}} \\\\",
    "\\endfoot", "\\endlastfoot")
}

format_cox_fit_rows <- function(fits) {
  ns <- sapply(fits, function(f) format(f$n, big.mark = ","))
  evts <- sapply(fits, function(f) format(f$events, big.mark = ","))
  ngs <- sapply(fits, function(f) format(f$n_groups, big.mark = ","))
  lls <- sapply(fits, function(f) sprintf("%.1f", f$log_lik))
  c("   \\midrule",
    "   \\emph{Fit statistics} & & & & \\\\",
    sprintf("   %-25s & %s & %s & %s & %s \\\\",
            "Observations", ns[1], ns[2], ns[3], ns[4]),
    sprintf("   %-25s & %s & %s & %s & %s \\\\",
            "Events", evts[1], evts[2], evts[3], evts[4]),
    sprintf("   %-25s & %s & %s & %s & %s \\\\",
            "Participants", ngs[1], ngs[2], ngs[3], ngs[4]),
    sprintf("   %-25s & %s & %s & %s & %s \\\\",
            "Log-likelihood", lls[1], lls[2], lls[3], lls[4]))
}

build_footer <- function() {
  c("   \\midrule \\midrule",
    paste0("   \\multicolumn{5}{l}{\\emph{Hazard ratios reported.",
           " All models: random-intercept Cox (coxme).}} \\\\"),
    paste0("   \\multicolumn{5}{l}{\\emph{HR $>$ 1: increased",
           " hazard of selling. HR $<$ 1: decreased hazard.}} \\\\"),
    paste0("   \\multicolumn{5}{l}{\\emph{Signif. Codes:",
           " ***: 0.01, **: 0.05, *: 0.1}} \\\\"),
    "\\end{longtable}", "\\par\\endgroup", "", "")
}

# %%
if (!interactive()) main()
