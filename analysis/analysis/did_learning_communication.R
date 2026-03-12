# Purpose: DiD regression for learning vs communication effects on n_sellers
# Author: Caleb Eynon w/ Claude Code
# Date: 2026-03-10
#
# Decomposes segment effects into learning trend and chat (communication).
# Table displays OLS with session+round FE and treatment x chat interaction.
# Groups are reshuffled between segments, so all models cluster at global_group_id.
# Wald tests check parallel trends and isolate communication effect.

library(data.table)
library(AER)
library(sandwich)
library(lmtest)
library(car)
library(fixest)
# AER/sandwich/lmtest/car used for Wald tests on segment-dummy tobit

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

  model <- fit_ols_session_round_fe_interaction(dt)
  coef_tab <- extract_ols_coefs(model)
  fit <- extract_ols_fit(model)
  wald_pvals <- run_wald_tests(dt)

  build_latex_table(coef_tab, fit, wald_pvals, OUTPUT_PATH)
  cat("\nModel summary:\n")
  print(summary(model))
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
  dt[, tr2_chat := as.integer(treatment == 2) * chat]
  dt
}

# =====
# Model fitting
# =====
fit_ols_session_round_fe_interaction <- function(dt) {
  cat("Fitting OLS (session + round FE + interaction)...\n")
  feols(
    n_sellers ~ bad_state + learning_trend + chat + tr2_chat | session + round_num,
    cluster = ~global_group_id, data = dt
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
# Fit statistics extraction
# =====
extract_ols_fit <- function(model) {
  list(
    n = model$nobs,
    r2 = fixest::r2(model, type = "ar2")
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
build_latex_table <- function(coef_tab, fit, wald_pvals, path) {
  lines <- c(
    table_preamble(),
    table_header(),
    coef_rows(VAR_ORDER, VAR_LABELS, coef_tab),
    fit_rows(fit, wald_pvals),
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
    "\\begin{tabular}{lc}")
}

table_header <- function() {
  c("   \\tabularnewline \\midrule \\midrule",
    "   Dependent Variable: & Number of Sellers in Round\\\\",
    "   Estimator:           & OLS\\\\",
    "   \\midrule",
    "   \\emph{Variables}\\\\")
}

coef_rows <- function(vars, labels, coef_tab) {
  rows <- c()
  for (v in vars) {
    row <- coef_tab[var == v]
    if (nrow(row) == 0) next
    label <- labels[v]
    star <- get_stars(row$pval)
    val <- paste0(sprintf("%.4f", row$est), star)
    se <- paste0("(", sprintf("%.4f", row$se), ")")
    rows <- c(rows, row_line(label, val))
    rows <- c(rows, row_line("", se))
  }
  rows
}

get_stars <- function(pval) {
  if (pval < 0.01) return("$^{***}$")
  if (pval < 0.05) return("$^{**}$")
  if (pval < 0.1)  return("$^{*}$")
  ""
}

row_line <- function(label, val) {
  sprintf("   %-35s & %s\\\\", label, val)
}

# =====
# Fit statistics rows
# =====
fit_rows <- function(fit, wald_pvals) {
  pt_p <- sprintf("%.4f", wald_pvals$parallel_trends)

  c("   \\midrule",
    "   \\emph{Fit statistics}\\\\",
    row_line("Observations", format(fit$n, big.mark = ",")),
    row_line("Adj. $R^{2}$", sprintf("%.4f", fit$r2)),
    row_line("Session FE", "Yes"),
    row_line("Round FE", "Yes"),
    row_line("Parallel trends $p$", pt_p))
}

# =====
# Table footer
# =====
table_footer <- function() {
  c("   \\midrule \\midrule",
    paste0("   \\multicolumn{2}{l}{\\emph{Cluster-robust",
           " standard errors in parentheses",
           " (segment-group level)}}\\\\"),
    paste0("   \\multicolumn{2}{l}{\\emph{Signif. Codes:",
           " ***: 0.01, **: 0.05, *: 0.1}}\\\\"),
    "\\end{tabular}",
    "\\par\\endgroup", "", "")
}

# %%
if (!interactive()) main()
