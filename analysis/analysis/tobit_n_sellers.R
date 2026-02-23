# Purpose: Tobit regression for number of sellers per group-round
# Author: Caleb Eynon w/ Claude Code
# Date: 2026-02-22
#
# DV is n_sellers (0-4), censored at both boundaries.
# Cluster-robust SEs at the global_group_id level.

library(data.table)
library(AER)
library(sandwich)
library(lmtest)

# FILE PATHS
INPUT_PATH <- "datastore/derived/group_round_timing.csv"
OUTPUT_PATH <- "analysis/output/tables/tobit_n_sellers.tex"

# TABLE CONFIGURATION
VAR_ORDER <- c(
  "(Intercept)", "bad_state", "treatment2",
  "segment_num2", "segment_num3", "segment_num4",
  "round_num"
)
VAR_LABELS <- c(
  "(Intercept)" = "Constant",
  bad_state     = "Bad state",
  treatment2    = "Treatment 2",
  segment_num2  = "Segment 2",
  segment_num3  = "Segment 3",
  segment_num4  = "Segment 4",
  round_num     = "Round"
)

# =====
# Main function
# =====
main <- function() {
  dt <- load_and_prepare(INPUT_PATH)
  cat("Loaded", nrow(dt), "observations\n")

  models <- fit_models(dt)
  coef_tabs <- lapply(models, function(m) extract_coefs(m, dt))
  fits <- lapply(models, extract_fit)

  build_latex_table(coef_tabs, fits, OUTPUT_PATH)
  print_summaries(models, dt)
  cat("\nDone.\n")
}

# =====
# Data loading and preparation
# =====
load_and_prepare <- function(path) {
  dt <- fread(path)
  dt[, bad_state := as.integer(state == 0)]
  dt[, segment_num := factor(segment_num, levels = 1:4)]
  dt[, treatment := factor(treatment, levels = c(1, 2))]
  dt
}

# =====
# Model fitting
# =====
fit_models <- function(dt) {
  cat("Fitting Model 1 (baseline)...\n")
  m1 <- tobit(n_sellers ~ bad_state + treatment,
              left = 0, right = 4, data = dt)

  cat("Fitting Model 2 (+ segments)...\n")
  m2 <- tobit(n_sellers ~ bad_state + treatment + segment_num,
              left = 0, right = 4, data = dt)

  cat("Fitting Model 3 (full)...\n")
  m3 <- tobit(n_sellers ~ bad_state + treatment + segment_num + round_num,
              left = 0, right = 4, data = dt)

  list(m1 = m1, m2 = m2, m3 = m3)
}

# =====
# Coefficient extraction with cluster-robust SEs
# =====
extract_coefs <- function(model, dt) {
  cl_vcov <- vcovCL(model, cluster = dt$global_group_id)
  ct <- coeftest(model, vcov. = cl_vcov)
  # Drop Log(scale) parameter row
  ct <- ct[rownames(ct) != "Log(scale)", , drop = FALSE]
  data.table(
    var = rownames(ct),
    est = ct[, 1],
    se  = ct[, 2],
    pval = ct[, 4]
  )
}

extract_fit <- function(model) {
  list(
    n = nobs(model),
    log_lik = as.numeric(logLik(model))
  )
}

# =====
# LaTeX table
# =====
build_latex_table <- function(coef_tabs, fits, path) {
  lines <- c(
    table_preamble(),
    table_header(),
    coef_rows(VAR_ORDER, VAR_LABELS, coef_tabs),
    fit_rows(fits),
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
    "\\begin{tabular}{lccc}")
}

table_header <- function() {
  c("   \\tabularnewline \\midrule \\midrule",
    "   Dependent Variable: & \\multicolumn{3}{c}{n\\_sellers}\\\\",
    "   Model:               & (1)  & (2)  & (3)\\\\",
    "   \\midrule",
    "   \\emph{Variables}\\\\")
}

coef_rows <- function(vars, labels, coef_tabs) {
  rows <- c()
  for (v in vars) {
    label <- labels[v]
    cells <- format_cells(var_name = v, coef_tabs)
    if (!any(nzchar(cells$vals))) next
    rows <- c(rows, row_line(label, cells$vals))
    rows <- c(rows, row_line("", cells$ses))
  }
  rows
}

format_cells <- function(var_name, coef_tabs) {
  vals <- ses <- rep("", 3)
  for (i in seq_along(coef_tabs)) {
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
  sprintf("   %-25s & %s & %s & %s\\\\",
          label, cells[1], cells[2], cells[3])
}

fit_rows <- function(fits) {
  ns <- sapply(fits, function(f) format(f$n, big.mark = ","))
  lls <- sapply(fits, function(f) sprintf("%.1f", f$log_lik))
  c("   \\midrule",
    "   \\emph{Fit statistics}\\\\",
    sprintf("   %-25s & %s & %s & %s\\\\",
            "Observations", ns[1], ns[2], ns[3]),
    sprintf("   %-25s & %s & %s & %s\\\\",
            "Log-likelihood", lls[1], lls[2], lls[3]))
}

table_footer <- function() {
  c("   \\midrule \\midrule",
    paste0("   \\multicolumn{4}{l}{\\emph{Cluster-robust",
           " (global\\_group\\_id) standard errors",
           " in parentheses}}\\\\"),
    paste0("   \\multicolumn{4}{l}{\\emph{Signif. Codes:",
           " ***: 0.01, **: 0.05, *: 0.1}}\\\\"),
    "\\end{tabular}",
    "\\par\\endgroup", "", "")
}

# =====
# Console output
# =====
print_summaries <- function(models, dt) {
  cat("\n", strrep("=", 60), "\n")
  cat("MODEL SUMMARIES\n")
  cat(strrep("=", 60), "\n")
  for (name in names(models)) {
    cat("\n", name, ":\n")
    cl_vcov <- vcovCL(models[[name]], cluster = dt$global_group_id)
    print(coeftest(models[[name]], vcov. = cl_vcov))
  }
}

# %%
if (!interactive()) main()
