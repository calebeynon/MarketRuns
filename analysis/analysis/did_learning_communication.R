# Purpose: DiD regression for learning vs communication effects on n_sellers
# Author: Caleb Eynon w/ Claude Code
# Date: 2026-03-10
#
# Decomposes segment effects into learning trend and chat (communication).
# FE progression: (1) tobit no FE, (2) OLS session+round FE,
# (3) OLS session_group FE, (4) OLS session_group FE + treatment x chat.
# Wald tests check parallel trends and isolate communication effect.

library(data.table)
library(AER)
library(sandwich)
library(lmtest)
library(car)
library(fixest)

# FILE PATHS
INPUT_PATH <- "datastore/derived/group_round_timing.csv"
OUTPUT_PATH <- "analysis/output/tables/did_learning_communication.tex"

# TABLE CONFIGURATION
VAR_ORDER <- c(
  "(Intercept)", "bad_state", "treatment2",
  "learning_trend", "chat", "tr2_chat",
  "round_num"
)
VAR_LABELS <- c(
  "(Intercept)"     = "Constant",
  bad_state         = "Bad state",
  treatment2        = "Treatment 2",
  learning_trend    = "Learning trend",
  chat              = "Chat (communication)",
  tr2_chat          = "Treatment 2 $\\times$ Chat",
  round_num         = "Round"
)

# =====
# Main function
# =====
main <- function() {
  dt <- load_and_prepare(INPUT_PATH)
  cat("Loaded", nrow(dt), "observations\n")

  models <- fit_models(dt)
  coef_tabs <- extract_all_coefs(models, dt)
  fits <- extract_all_fits(models, dt)
  wald_pvals <- run_wald_tests(dt)

  build_latex_table(coef_tabs, fits, wald_pvals, OUTPUT_PATH)
  cat("\nDone.\n")
}

# =====
# Data loading and preparation
# =====
load_and_prepare <- function(path) {
  dt <- fread(path)
  dt[, bad_state := as.integer(state == 0)]
  dt[, treatment := factor(treatment, levels = c(1, 2))]
  dt[, learning_trend := segment_num]
  dt[, chat := as.integer(segment_num >= 3)]
  dt[, session_group := paste(session, group_id, sep = "_")]
  dt[, tr2_chat := as.integer(treatment == 2) * chat]
  dt
}

# =====
# Model fitting
# =====
fit_models <- function(dt) {
  list(
    m1 = fit_tobit_no_fe(dt),
    m2 = fit_ols_session_round_fe(dt),
    m3 = fit_ols_group_fe(dt),
    m4 = fit_ols_group_fe_interaction(dt)
  )
}

fit_tobit_no_fe <- function(dt) {
  cat("Fitting Model 1 (tobit, no FE)...\n")
  tobit(
    n_sellers ~ bad_state + treatment + learning_trend + chat + round_num,
    left = 0, right = 4, data = dt
  )
}

fit_ols_session_round_fe <- function(dt) {
  cat("Fitting Model 2 (OLS, session + round FE)...\n")
  feols(
    n_sellers ~ bad_state + learning_trend + chat | session + round_num,
    cluster = ~session_group, data = dt
  )
}

fit_ols_group_fe <- function(dt) {
  cat("Fitting Model 3 (OLS, session_group FE)...\n")
  feols(
    n_sellers ~ bad_state + learning_trend + chat | session_group,
    cluster = ~session_group, data = dt
  )
}

fit_ols_group_fe_interaction <- function(dt) {
  cat("Fitting Model 4 (OLS, session_group FE + interaction)...\n")
  feols(
    n_sellers ~ bad_state + learning_trend + chat + tr2_chat | session_group,
    cluster = ~session_group, data = dt
  )
}

# =====
# Coefficient extraction — tobit with cluster-robust SEs
# =====
extract_tobit_coefs <- function(model, dt) {
  cl_vcov <- vcovCL(model, cluster = dt$global_group_id)
  ct <- coeftest(model, vcov. = cl_vcov)
  ct <- ct[rownames(ct) != "Log(scale)", , drop = FALSE]
  data.table(
    var  = rownames(ct),
    est  = ct[, 1],
    se   = ct[, 2],
    pval = ct[, 4]
  )
}

# =====
# Coefficient extraction — fixest OLS
# =====
extract_ols_coefs <- function(model) {
  ct <- coeftable(model)
  data.table(
    var  = rownames(ct),
    est  = ct[, 1],
    se   = ct[, 2],
    pval = ct[, 4]
  )
}

# =====
# Combine coefficient tables for all 4 models
# =====
extract_all_coefs <- function(models, dt) {
  list(
    extract_tobit_coefs(models$m1, dt),
    extract_ols_coefs(models$m2),
    extract_ols_coefs(models$m3),
    extract_ols_coefs(models$m4)
  )
}

# =====
# Fit statistics extraction
# =====
extract_tobit_fit <- function(model) {
  list(n = nobs(model), log_lik = as.numeric(logLik(model)))
}

extract_ols_fit <- function(model, dt) {
  list(
    n = model$nobs,
    r2 = fixest::r2(model, type = "ar2"),
    n_clusters = uniqueN(dt$session_group)
  )
}

extract_all_fits <- function(models, dt) {
  list(
    extract_tobit_fit(models$m1),
    extract_ols_fit(models$m2, dt),
    extract_ols_fit(models$m3, dt),
    extract_ols_fit(models$m4, dt)
  )
}

# =====
# Wald tests on segment-dummy tobit
# =====
run_wald_tests <- function(dt) {
  cat("\n", strrep("=", 60), "\n")
  cat("WALD TESTS (segment-dummy tobit, cluster-robust)\n")
  cat(strrep("=", 60), "\n")

  dt_wald <- copy(dt)
  dt_wald[, segment_num_f := factor(segment_num, levels = 1:4)]
  m_seg <- fit_segment_tobit(dt_wald)
  cl_vcov <- vcovCL(m_seg, cluster = dt_wald$global_group_id)

  comm_p <- run_communication_test(m_seg, cl_vcov)
  trends_p <- run_parallel_trends_test(m_seg, cl_vcov)
  list(communication = comm_p, parallel_trends = trends_p)
}

fit_segment_tobit <- function(dt) {
  tobit(
    n_sellers ~ bad_state + treatment + segment_num_f + round_num,
    left = 0, right = 4, data = dt
  )
}

run_communication_test <- function(model, vcov) {
  cat("\nCommunication effect (seg3 - 2*seg2 = 0):\n")
  h <- linearHypothesis(
    model, "segment_num_f3 - 2 * segment_num_f2 = 0", vcov. = vcov
  )
  print(h)
  h[2, "Pr(>Chisq)"]
}

run_parallel_trends_test <- function(model, vcov) {
  cat("\nParallel trends (seg4 - seg3 - seg2 = 0):\n")
  h <- linearHypothesis(
    model, "segment_num_f4 - segment_num_f3 - segment_num_f2 = 0", vcov. = vcov
  )
  print(h)
  h[2, "Pr(>Chisq)"]
}

# =====
# LaTeX table construction
# =====
build_latex_table <- function(coef_tabs, fits, wald_pvals, path) {
  lines <- c(
    table_preamble(),
    table_header(),
    coef_rows(VAR_ORDER, VAR_LABELS, coef_tabs),
    fit_rows(fits, wald_pvals),
    table_footer()
  )
  write_table(lines, path)
}

write_table <- function(lines, path) {
  output_dir <- dirname(path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  writeLines(lines, path)
  cat("Table exported to:", path, "\n")
}

table_preamble <- function() {
  c("", "\\begingroup", "\\centering", "\\scriptsize",
    "\\begin{tabular}{lcccc}")
}

table_header <- function() {
  c("   \\tabularnewline \\midrule \\midrule",
    paste0("   Dependent Variable: & \\multicolumn{4}",
           "{c}{Number of Sellers in Round}\\\\"),
    "   \\cmidrule(lr){2-5}",
    paste0("   Model:               & ~~~~~~(1)~~~~~~  ",
           "& ~~~~~~(2)~~~~~~  & ~~~~~~(3)~~~~~~  & ~~~~~~(4)~~~~~~\\\\"),
    "   Estimator:           & Tobit  & OLS  & OLS  & OLS\\\\",
    "   \\midrule",
    "   \\emph{Variables}\\\\")
}

coef_rows <- function(vars, labels, coef_tabs) {
  rows <- c()
  for (v in vars) {
    label <- labels[v]
    cells <- format_cells(v, coef_tabs)
    if (!any(nzchar(cells$vals))) next
    rows <- c(rows, row_line(label, cells$vals))
    rows <- c(rows, row_line("", cells$ses))
  }
  rows
}

format_cells <- function(var_name, coef_tabs) {
  n_cols <- length(coef_tabs)
  vals <- ses <- rep("", n_cols)
  for (i in seq_len(n_cols)) {
    row <- coef_tabs[[i]][var == var_name]
    if (nrow(row) == 0) next
    star <- get_stars(row$pval)
    vals[i] <- paste0(sprintf("%.4f", row$est), star)
    ses[i]  <- paste0("(", sprintf("%.4f", row$se), ")")
  }
  list(vals = vals, ses = ses)
}

get_stars <- function(pval) {
  if (pval < 0.01) return("$^{***}$")
  if (pval < 0.05) return("$^{**}$")
  if (pval < 0.1)  return("$^{*}$")
  ""
}

row_line <- function(label, cells) {
  sprintf("   %-35s & %s & %s & %s & %s\\\\",
          label, cells[1], cells[2], cells[3], cells[4])
}

# =====
# Fit statistics rows
# =====
fit_rows <- function(fits, wald_pvals) {
  ns <- sapply(fits, function(f) format(f$n, big.mark = ","))
  ll <- sprintf("%.1f", fits[[1]]$log_lik)
  r2s <- sapply(fits[2:4], function(f) sprintf("%.4f", f$r2))
  pt_p <- sprintf("%.4f", wald_pvals$parallel_trends)

  c("   \\midrule",
    "   \\emph{Fit statistics}\\\\",
    fit_stat_row("Observations", ns),
    fit_stat_row("Log-likelihood", c(ll, "", "", "")),
    fit_stat_row("Adj. $R^{2}$", c("", r2s)),
    fe_indicator_rows(),
    fit_stat_multicolumn("Parallel trends $p$", pt_p))
}

fe_indicator_rows <- function() {
  c(fit_stat_row("Session FE", c("No", "Yes", "No", "No")),
    fit_stat_row("Round FE", c("No", "Yes", "No", "No")),
    fit_stat_row("Group FE", c("No", "No", "Yes", "Yes")))
}

fit_stat_multicolumn <- function(label, val) {
  sprintf("   %-35s & \\multicolumn{4}{c}{%s}\\\\", label, val)
}

fit_stat_row <- function(label, vals) {
  sprintf("   %-35s & %s & %s & %s & %s\\\\",
          label, vals[1], vals[2], vals[3], vals[4])
}

# =====
# Table footer
# =====
table_footer <- function() {
  c("   \\midrule \\midrule",
    paste0("   \\multicolumn{5}{l}{\\emph{Cluster-robust",
           " standard errors in parentheses (group level",
           " for tobit, session-group for OLS)}}\\\\"),
    paste0("   \\multicolumn{5}{l}{\\emph{Signif. Codes:",
           " ***: 0.01, **: 0.05, *: 0.1}}\\\\"),
    "\\end{tabular}",
    "\\par\\endgroup", "", "")
}

# %%
if (!interactive()) main()
