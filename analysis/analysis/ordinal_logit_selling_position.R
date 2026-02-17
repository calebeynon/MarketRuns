# Purpose: Ordinal logit regressions for selling position (rank 1-4)
#          Model 1: Full sample, clm (non-sellers = rank 4, no random effects)
#          Model 2: Sellers only, clmm (did_sell == 1, player random effects)
# Author: Claude Code
# Date: 2026-02-10

library(data.table)
library(ordinal)
library(marginaleffects)

# FILE PATHS
INPUT_PATH <- "datastore/derived/ordinal_selling_position.csv"
OUTPUT_PATH <- "analysis/output/tables/ordinal_logit_selling_position.tex"

# VARIABLE LISTS
SHOW_VARS <- c("state_anxiety", "impulsivity", "conscientiousness",
               "fear_p95", "anger_p95")
HIDE_VARS <- c("extraversion", "agreeableness", "neuroticism", "openness",
               "contempt_p95", "disgust_p95", "joy_p95", "sadness_p95",
               "surprise_p95", "engagement_p95", "valence_p95",
               "age", "gender_female",
               "segment2", "segment3", "segment4", "round")
ALL_PREDICTORS <- c(SHOW_VARS, HIDE_VARS)
CONTINUOUS_VARS <- c("state_anxiety", "impulsivity", "conscientiousness",
                     "fear_p95", "anger_p95",
                     "extraversion", "agreeableness", "neuroticism", "openness",
                     "contempt_p95", "disgust_p95", "joy_p95", "sadness_p95",
                     "surprise_p95", "engagement_p95", "valence_p95",
                     "age", "round")
EMOTION_COLS <- c("fear_p95", "anger_p95", "contempt_p95", "disgust_p95",
                  "joy_p95", "sadness_p95", "surprise_p95",
                  "engagement_p95", "valence_p95")
VAR_DICT <- c(state_anxiety = "State anxiety", impulsivity = "Impulsivity",
              conscientiousness = "Conscientiousness",
              fear_p95 = "Fear (p95)", anger_p95 = "Anger (p95)")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  cat("Loading data from:", INPUT_PATH, "\n")
  df <- load_and_prepare(INPUT_PATH)

  cat("\n=== Sell rank frequencies (full sample) ===\n")
  print(table(df$sell_rank))

  cat("\n--- Fitting Model 1: Full Sample (clm, no RE) ---\n")
  model1 <- fit_clm(df, "Full Sample")

  df_sellers <- df[did_sell == 1]
  cat("\n=== Sell rank frequencies (sellers only) ===\n")
  print(table(df_sellers$sell_rank))

  cat("\n--- Fitting Model 2: Sellers Only (clmm, player RE) ---\n")
  model2 <- fit_clmm(df_sellers, "Sellers Only")

  cat("\n--- Computing AMEs on P(rank=1) ---\n")
  ame1 <- compute_ames(model1)
  # marginaleffects doesn't support clmm; refit as clm for AMEs
  model2_clm <- fit_clm(df_sellers, "Sellers Only (clm for AMEs)")
  ame2 <- compute_ames(model2_clm)

  build_two_model_table(model1, model2, ame1, ame2, OUTPUT_PATH)
  cat("\nDone!\n")
}

# =====
# Data loading and preparation
# =====
load_and_prepare <- function(file_path) {
  df <- fread(file_path)
  cat("Raw data dimensions:", nrow(df), "x", ncol(df), "\n")
  n_before <- nrow(df)
  df <- df[complete.cases(df[, ..EMOTION_COLS])]
  cat("Dropped", n_before - nrow(df), "rows with missing emotions\n")
  cat("Analysis sample:", nrow(df), "rows\n")
  df[, sell_rank := ordered(factor(sell_rank))]
  df[, player_id := as.factor(player_id)]
  create_segment_dummies(df)
  standardize_continuous(df)
  return(df)
}

create_segment_dummies <- function(df) {
  df[, segment2 := as.integer(segment == 2)]
  df[, segment3 := as.integer(segment == 3)]
  df[, segment4 := as.integer(segment == 4)]
}

standardize_continuous <- function(df) {
  for (v in CONTINUOUS_VARS) df[, (v) := scale(get(v))]
}

# =====
# Model fitting
# =====
fit_clm <- function(df, label) {
  formula_str <- paste("sell_rank ~", paste(ALL_PREDICTORS, collapse = " + "))
  cat("Formula:", formula_str, "\nN observations:", nrow(df), "\n")
  model <- clm(as.formula(formula_str), data = df)
  cat("\n--- Summary:", label, "---\n")
  print(summary(model))
  return(model)
}

fit_clmm <- function(df, label) {
  formula_str <- paste("sell_rank ~", paste(ALL_PREDICTORS, collapse = " + "),
                       "+ (1 | player_id)")
  cat("Formula:", formula_str, "\nN observations:", nrow(df), "\n")
  model <- clmm(as.formula(formula_str), data = df)
  cat("\n--- Summary:", label, "---\n")
  print(summary(model))
  return(model)
}

# =====
# Average Marginal Effects on P(rank=1)
# =====
compute_ames <- function(model) {
  ame_r1 <- as.data.table(avg_slopes(model))[group == "1"]
  data.table(var = ame_r1$term, est = ame_r1$estimate,
             se = ame_r1$std.error, pval = ame_r1$p.value)
}

# =====
# Coefficient extraction for clm/clmm
# =====
extract_clmm_coefs <- function(model) {
  s <- summary(model)
  beta_dt <- extract_beta_table(s)
  # clm stores thresholds in $Theta; clmm stores them in $alpha
  thresholds <- if (!is.null(model$alpha)) model$alpha else model$Theta
  alpha_dt <- extract_alpha_table(model, thresholds)
  list(beta = beta_dt, alpha = alpha_dt)
}

extract_beta_table <- function(s) {
  beta <- as.data.frame(s$coefficients)
  data.table(var = rownames(beta), est = beta[, "Estimate"],
             se = beta[, "Std. Error"], pval = beta[, "Pr(>|z|)"])
}

# Extract threshold params; use vcov when Hessian is available
extract_alpha_table <- function(model, thresholds) {
  tnames <- names(thresholds)
  ests <- unname(thresholds)
  vc <- tryCatch(vcov(model), error = function(e) NULL)
  if (!is.null(vc) && all(tnames %in% rownames(vc))) {
    ses <- sapply(tnames, function(t) sqrt(vc[t, t]))
    pvals <- 2 * pnorm(abs(ests / ses), lower.tail = FALSE)
    return(data.table(var = tnames, est = ests, se = ses, pval = pvals))
  }
  data.table(var = tnames, est = ests, se = NaN, pval = NaN)
}

extract_clmm_fit <- function(model) {
  list(n = nobs(model), loglik = logLik(model), aic = AIC(model))
}

# =====
# LaTeX table builder (two-model side-by-side)
# =====
get_sig_stars <- function(pval) {
  if (is.nan(pval) || is.na(pval)) return("")
  if (pval < 0.01) return("$^{***}$")
  if (pval < 0.05) return("$^{**}$")
  if (pval < 0.1) return("$^{*}$")
  return("")
}

format_coef_cell <- function(coefs_dt, varname) {
  row <- coefs_dt[var == varname]
  if (nrow(row) == 0) return(c("", ""))
  val <- sprintf("%.4f%s", row$est, get_sig_stars(row$pval))
  se <- if (is.nan(row$se)) "(---)" else sprintf("(%.4f)", row$se)
  c(val, se)
}

build_two_model_table <- function(model1, model2, ame1, ame2, output_path) {
  c1 <- extract_clmm_coefs(model1)
  c2 <- extract_clmm_coefs(model2)
  f1 <- extract_clmm_fit(model1)
  f2 <- extract_clmm_fit(model2)
  lines <- build_header()
  lines <- c(lines, "   \\emph{Panel A: Log-odds coefficients}\\\\")
  lines <- append_var_rows(lines, c1$beta, c2$beta)
  lines <- append_threshold_rows(lines, c1$alpha, c2$alpha)
  lines <- append_ame_rows(lines, ame1, ame2)
  lines <- append_fit_rows(lines, f1, f2)
  lines <- append_footer(lines)
  write_table(lines, output_path)
}

header_column_block <- function() {
  c("   \\midrule \\midrule",
    "   & (1) Full & (2) Sellers\\\\",
    "   & Sample & Only\\\\",
    "   \\midrule")
}

build_header <- function() {
  col_block <- header_column_block()
  c("", "\\begingroup", "\\centering", "\\tiny",
    "\\renewcommand{\\arraystretch}{0.75}",
    "\\setlength{\\tabcolsep}{2pt}",
    "\\begin{tabular}{@{}lcc@{}}",
    col_block)
}

append_var_rows <- function(lines, dt1, dt2, vars = SHOW_VARS) {
  for (v in vars) {
    label <- if (v %in% names(VAR_DICT)) VAR_DICT[v] else gsub("_", "\\\\_", v)
    c1 <- format_coef_cell(dt1, v)
    c2 <- format_coef_cell(dt2, v)
    lines <- c(lines,
      sprintf("   %-20s& %s & %s\\\\", label, c1[1], c2[1]),
      sprintf("   %-20s& %s & %s\\\\", "", c1[2], c2[2]))
  }
  return(lines)
}

append_threshold_rows <- function(lines, alpha1, alpha2) {
  lines <- c(lines, "   \\midrule", "   \\emph{Threshold parameters}\\\\")
  for (tname in union(alpha1$var, alpha2$var)) {
    label <- gsub("\\|", "$|$", tname)
    c1 <- format_coef_cell(alpha1, tname)
    c2 <- format_coef_cell(alpha2, tname)
    lines <- c(lines,
      sprintf("   %-20s& %s & %s\\\\", label, c1[1], c2[1]),
      sprintf("   %-20s& %s & %s\\\\", "", c1[2], c2[2]))
  }
  return(lines)
}

append_ame_rows <- function(lines, ame1, ame2) {
  lines <- c(lines, "   \\midrule",
    "   \\emph{Panel B: Average Marginal Effects on P(selling first)}\\\\")
  append_var_rows(lines, ame1, ame2)
}

append_fit_rows <- function(lines, f1, f2) {
  c(lines, "   \\midrule", "   \\emph{Fit statistics}\\\\",
    "   Model         & CLM & CLMM\\\\",
    sprintf("   Observations  & %s & %s\\\\",
            format(f1$n, big.mark = ","), format(f2$n, big.mark = ",")),
    sprintf("   Log-lik.      & %.1f & %.1f\\\\",
            as.numeric(f1$loglik), as.numeric(f2$loglik)),
    sprintf("   AIC           & %.1f & %.1f\\\\", f1$aic, f2$aic))
}

append_footer <- function(lines) {
  controls_note <- paste0(
    "Controls: extraversion, agreeableness, neuroticism, openness, ",
    "contempt (p95), disgust (p95), joy (p95), sadness (p95), ",
    "surprise (p95), engagement (p95), valence (p95), age, gender, ",
    "segment dummies, round. Model 1 excludes random effects; ",
    "Model 2 includes player random effects.")
  c(lines, "   \\midrule \\midrule",
    sprintf("   \\multicolumn{3}{@{}p{\\linewidth}@{}}{\\emph{%s}}\\\\", controls_note),
    "   \\multicolumn{3}{@{}l@{}}{\\emph{Standardized coefficients (z-scored)}}\\\\",
    "   \\multicolumn{3}{@{}l@{}}{\\emph{***: 0.01, **: 0.05, *: 0.1}}\\\\",
    "\\end{tabular}", "\\endgroup", "", "")
}

write_table <- function(lines, output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  writeLines(lines, output_path)
  cat("Table exported to:", output_path, "\n")
}

# %%
if (!interactive()) main()
