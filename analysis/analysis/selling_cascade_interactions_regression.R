# Purpose: Test whether cascade effects differ by emotion/trait levels via interactions
# Author: Claude Code
# Date: 2026-01-28

library(data.table)
library(fixest)
library(plm)

# FILE PATHS
INPUT_PATH <- "datastore/derived/emotions_traits_selling_dataset.csv"
OUTPUT_CASCADE <- "analysis/output/analysis/selling_cascade_interactions.tex"
OUTPUT_SECOND <- "analysis/output/analysis/selling_second_sellers_interactions.tex"

# VARIABLE LISTS
EMOTION_VARS <- c("anger_mean", "contempt_mean", "disgust_mean", "fear_mean",
                   "joy_mean", "sadness_mean", "surprise_mean",
                   "engagement_mean", "valence_mean")
TRAIT_VARS <- c("extraversion", "agreeableness", "conscientiousness",
                "neuroticism", "openness", "impulsivity", "state_anxiety")
INTERACT_VARS <- c("anger_mean", "fear_mean", "conscientiousness", "state_anxiety")

VAR_DICT <- c(
  "(Intercept)" = "Constant",
  n_sales_earlier = "n\\_sales\\_earlier",
  sale_prev_period = "sale\\_prev\\_period",
  dummy_prev_period = "dummy\\_prev\\_period",
  anger_mean = "Anger", contempt_mean = "Contempt", disgust_mean = "Disgust",
  fear_mean = "Fear", joy_mean = "Joy", sadness_mean = "Sadness",
  surprise_mean = "Surprise", engagement_mean = "Engagement", valence_mean = "Valence",
  extraversion = "Extraversion", agreeableness = "Agreeableness",
  conscientiousness = "Conscientiousness", neuroticism = "Neuroticism",
  openness = "Openness", impulsivity = "Impulsivity", state_anxiety = "State anxiety",
  age = "Age", gender_female = "Female",
  signal = "Signal", period = "Period", round = "Round",
  segment2 = "Segment 2", segment3 = "Segment 3", segment4 = "Segment 4",
  treatmenttr2 = "Treatment 2"
)

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  cat("Loading data from:", INPUT_PATH, "\n")
  df <- prepare_base_data(INPUT_PATH)
  cat("Base data dimensions:", nrow(df), "rows\n")

  cat("\n=== Table 1: Cascade Interactions ===\n")
  run_cascade_interactions(df)

  cat("\n=== Table 2: Second Sellers Interactions ===\n")
  run_second_sellers_interactions(df)

  cat("\nDone!\n")
}

# =====
# Data preparation (reused from selling_cascade_emotions_traits_regression.R)
# =====
prepare_base_data <- function(file_path) {
  df <- fread(file_path)
  df[, player_id := paste(session_id, player, sep = "_")]
  df[, global_group_id := paste(session_id, segment, group_id, sep = "_")]
  df[, group_round_id := paste(session_id, segment, group_id, round, sep = "_")]
  df[, player_group_round_id := paste(player_id, segment, group_id, round, sep = "_")]

  df <- df[already_sold == 0]
  df <- create_sale_timing_vars(df)
  df[, gender_female := as.integer(gender == "Female")]
  df[, time_id := paste(segment, round, period, sep = "_")]

  df[, segment := as.factor(segment)]
  df[, treatment := as.factor(treatment)]
  df[, session_id := as.factor(session_id)]
  return(df)
}

create_sale_timing_vars <- function(df) {
  period_sales <- df[, .(n_sales = sum(sold)), by = .(group_round_id, period)]
  setorder(period_sales, group_round_id, period)
  period_sales[, prev_n := shift(n_sales, 1, type = "lag"), by = group_round_id]

  df <- merge(df, period_sales[, .(group_round_id, period, prev_n)],
              by = c("group_round_id", "period"), all.x = TRUE)
  df[, sale_prev_period := as.integer(!is.na(prev_n) & prev_n > 0)]
  df[, n_sales_earlier := prior_group_sales - fifelse(is.na(prev_n), 0L, as.integer(prev_n))]
  df[, prev_n := NULL]
  return(df)
}

filter_complete_emotions <- function(df) {
  df[complete.cases(df[, ..EMOTION_VARS])]
}

# =====
# Table 1: Cascade × Emotion/Trait Interactions
# =====
run_cascade_interactions <- function(df) {
  df_em <- filter_complete_emotions(df)
  cat("Sample size after emotion filter:", nrow(df_em), "\n")

  et <- paste(c(EMOTION_VARS, TRAIT_VARS, "age", "gender_female"), collapse = " + ")
  int_nse <- paste0("n_sales_earlier:", INTERACT_VARS, collapse = " + ")
  int_spp <- paste0("sale_prev_period:", INTERACT_VARS, collapse = " + ")

  m1 <- feols(as.formula(paste("sold ~ n_sales_earlier +", int_nse, "+",
    et, "+ signal + period + round + segment | session_id")),
    cluster = ~global_group_id, data = df_em)
  pdata <- pdata.frame(as.data.frame(df_em), index = c("player_id", "time_id"))
  m2 <- plm(as.formula(paste("sold ~ n_sales_earlier +", int_nse, "+",
    et, "+ signal + period + round + segment + treatment")),
    data = pdata, model = "random")
  m3 <- feols(as.formula(paste("sold ~ sale_prev_period +", int_spp, "+",
    et, "+ signal + period + round + segment | session_id")),
    cluster = ~global_group_id, data = df_em)
  m4 <- plm(as.formula(paste("sold ~ sale_prev_period +", int_spp, "+",
    et, "+ signal + period + round + segment + treatment")),
    data = pdata, model = "random")

  # Interaction variable names as they appear in model output
  int_names_nse <- paste0("n_sales_earlier:", INTERACT_VARS)
  int_names_spp <- paste0("sale_prev_period:", INTERACT_VARS)
  int_labels_nse <- paste0("n\\_sales\\_earlier $\\times$ ", VAR_DICT[INTERACT_VARS])
  int_labels_spp <- paste0("sale\\_prev\\_period $\\times$ ", VAR_DICT[INTERACT_VARS])

  # Add interaction labels to dict
  int_dict <- setNames(
    c(int_labels_nse, int_labels_spp),
    c(int_names_nse, int_names_spp)
  )

  var_order <- c("n_sales_earlier", "sale_prev_period",
    int_names_nse, int_names_spp,
    EMOTION_VARS, TRAIT_VARS, "age", "gender_female",
    "signal", "period", "round",
    "segment2", "segment3", "segment4", "treatmenttr2", "(Intercept)")

  build_combined_table(list(m1, m2, m3, m4), var_order,
    c("(1)", "(2)", "(3)", "(4)"), c(TRUE, FALSE, TRUE, FALSE),
    c("Clustered", "RE", "Clustered", "RE"), OUTPUT_CASCADE,
    extra_dict = int_dict)
}

# =====
# Table 2: Second Sellers × Emotion/Trait Interactions
# =====
run_second_sellers_interactions <- function(df) {
  second_ids <- df[prior_group_sales == 1 & sold == 1, unique(player_group_round_id)]
  df_second <- df[player_group_round_id %in% second_ids]
  df_second <- create_prev_period_dummy(df_second, df)
  df_second <- filter_complete_emotions(df_second)
  cat("Second sellers sample:", nrow(df_second), "rows\n")

  et <- paste(c(EMOTION_VARS, TRAIT_VARS, "age", "gender_female"), collapse = " + ")
  int_dpp <- paste0("dummy_prev_period:", INTERACT_VARS, collapse = " + ")

  m1 <- feols(as.formula(paste("sold ~ dummy_prev_period +", int_dpp, "+",
    et, "+ signal + round + segment | session_id")),
    cluster = ~global_group_id, data = df_second)
  pdata <- pdata.frame(as.data.frame(df_second), index = c("player_id", "time_id"))
  m2 <- plm(as.formula(paste("sold ~ dummy_prev_period +", int_dpp, "+",
    et, "+ signal + round + segment + treatment")),
    data = pdata, model = "random")

  int_names <- paste0("dummy_prev_period:", INTERACT_VARS)
  int_labels <- paste0("dummy\\_prev\\_period $\\times$ ", VAR_DICT[INTERACT_VARS])
  int_dict <- setNames(int_labels, int_names)

  var_order <- c("dummy_prev_period", int_names,
    EMOTION_VARS, TRAIT_VARS, "age", "gender_female",
    "signal", "round", "segment2", "segment3", "segment4",
    "treatmenttr2", "(Intercept)")

  build_combined_table(list(m1, m2), var_order,
    c("(1)", "(2)"), c(TRUE, FALSE),
    c("Clustered", "RE"), OUTPUT_SECOND,
    extra_dict = int_dict)
}

create_prev_period_dummy <- function(df_second, df_full) {
  first_sales <- df_full[prior_group_sales == 0 & sold == 1,
                         .(first_sale_period = min(period)), by = group_round_id]
  df_second <- merge(df_second, first_sales, by = "group_round_id", all.x = TRUE)
  df_second[, dummy_prev_period := as.integer(first_sale_period == (period - 1))]
  df_second[, first_sale_period := NULL]
  return(df_second)
}

# =====
# Coefficient extraction helpers
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
    list(n = summary(model)$nobs, r2 = r2(model, "r2"), ar2 = r2(model, "ar2"))
  } else {
    s <- summary(model)
    rv <- s$r.squared
    list(n = nobs(model), r2 = rv[1], ar2 = if (length(rv) > 1) rv[2] else NA)
  }
}

# =====
# LaTeX table builder
# =====
get_sig_stars <- function(pval) {
  if (pval < 0.01) return("$^{***}$")
  if (pval < 0.05) return("$^{**}$")
  if (pval < 0.1) return("$^{*}$")
  return("")
}

build_combined_table <- function(models, var_order, col_headers,
                                  has_session_fe, se_types, output_path,
                                  extra_dict = NULL) {
  full_dict <- c(VAR_DICT, extra_dict)
  coefs <- lapply(models, extract_coefs)
  fits <- lapply(models, extract_fit)
  nc <- length(models)

  col_spec <- paste0("l", paste(rep("c", nc), collapse = ""))
  header <- c(
    "   \\midrule \\midrule",
    sprintf("   Dependent Variable: & \\multicolumn{%d}{c}{sold}\\\\", nc),
    sprintf("   & %s\\\\", paste(col_headers, collapse = " & ")),
    "   \\midrule"
  )
  lines <- c(
    "", "\\begingroup", "\\scriptsize",
    sprintf("\\begin{longtable}{%s}", col_spec),
    header, "   \\emph{Variables}\\\\",
    "   \\endfirsthead",
    sprintf("   \\multicolumn{%d}{l}{\\emph{(continued)}}\\\\", nc + 1),
    header,
    "   \\endhead",
    "   \\midrule",
    sprintf("   \\multicolumn{%d}{r}{\\emph{continued on next page}}\\\\", nc + 1),
    "   \\endfoot",
    "   \\endlastfoot"
  )

  for (v in var_order) {
    label <- if (v %in% names(full_dict)) full_dict[v] else gsub("_", "\\\\_", v)
    vals <- vapply(coefs, function(ct) {
      row <- ct[var == v]
      if (nrow(row) == 0) return("")
      sprintf("%.4f%s", row$est, get_sig_stars(row$pval))
    }, character(1))
    ses <- vapply(coefs, function(ct) {
      row <- ct[var == v]
      if (nrow(row) == 0) return("")
      sprintf("(%.4f)", row$se)
    }, character(1))
    if (all(vals == "")) next
    lines <- c(lines,
      sprintf("   %s & %s\\\\", label, paste(vals, collapse = " & ")),
      sprintf("   & %s\\\\", paste(ses, collapse = " & ")))
  }

  lines <- c(lines, "   \\midrule", "   \\emph{Fit statistics}\\\\",
    sprintf("   Session FE & %s\\\\", paste(ifelse(has_session_fe, "Yes", "No"), collapse = " & ")),
    sprintf("   SE type & %s\\\\", paste(se_types, collapse = " & ")),
    sprintf("   Observations & %s\\\\",
      paste(vapply(fits, function(f) format(f$n, big.mark = ","), character(1)), collapse = " & ")),
    sprintf("   R$^2$ & %s\\\\",
      paste(vapply(fits, function(f) sprintf("%.5f", f$r2), character(1)), collapse = " & ")))

  ar2 <- vapply(fits, function(f) if (is.na(f$ar2)) "" else sprintf("%.5f", f$ar2), character(1))
  if (!all(ar2 == ""))
    lines <- c(lines, sprintf("   Adj. R$^2$ & %s\\\\", paste(ar2, collapse = " & ")))

  lines <- c(lines, "   \\midrule \\midrule",
    sprintf("   \\multicolumn{%d}{l}{\\emph{Signif. Codes: ***: 0.01, **: 0.05, *: 0.1}}\\\\", nc + 1),
    "\\end{longtable}", "\\endgroup", "", "")

  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  writeLines(lines, output_path)
  cat("Table exported to:", output_path, "\n")
}

# %%
if (!interactive()) main()
