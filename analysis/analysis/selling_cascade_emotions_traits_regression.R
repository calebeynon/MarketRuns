# Purpose: Extend issue #4 cascade regressions with emotions/traits controls
# Author: Claude Code
# Date: 2026-01-28

library(data.table)
library(fixest)
library(plm)

# FILE PATHS
INPUT_PATH <- "datastore/derived/emotions_traits_selling_dataset.csv"
OUTPUT_CASCADE <- "analysis/output/analysis/selling_cascade_emotions_traits.tex"
OUTPUT_FIRST <- "analysis/output/analysis/selling_first_sellers_emotions_traits.tex"
OUTPUT_SECOND <- "analysis/output/analysis/selling_second_sellers_emotions_traits.tex"

# VARIABLE LISTS
EMOTION_VARS <- c("anger_mean", "contempt_mean", "disgust_mean", "fear_mean",
                   "joy_mean", "sadness_mean", "surprise_mean",
                   "engagement_mean", "valence_mean")
TRAIT_VARS <- c("extraversion", "agreeableness", "conscientiousness",
                "neuroticism", "openness", "impulsivity", "state_anxiety")
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

  cat("\n=== Table 1: Selling Cascade + Emotions/Traits ===\n")
  run_cascade_table(df)

  cat("\n=== Table 2: First Sellers + Emotions/Traits ===\n")
  run_first_sellers_table(df)

  cat("\n=== Table 3: Second Sellers + Emotions/Traits ===\n")
  run_second_sellers_table(df)

  cat("\nDone!\n")
}

# =====
# Data preparation
# =====
prepare_base_data <- function(file_path) {
  df <- fread(file_path)
  df[, player_id := paste(session_id, player, sep = "_")]
  df[, global_group_id := paste(session_id, segment, group_id, sep = "_")]
  df[, group_round_id := paste(session_id, segment, group_id, round, sep = "_")]
  df[, player_group_round_id := paste(player_id, segment, group_id, round, sep = "_")]

  df <- df[already_sold == 0]

  # Sale timing variables computed before emotion filtering
  df <- create_sale_timing_vars(df)
  df[, gender_female := as.integer(gender == "Female")]

  # Unique time index for plm (segment-round-period is unique per player)
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
# Table 1: Cascade + Emotions/Traits
# =====
run_cascade_table <- function(df) {
  df_em <- filter_complete_emotions(df)
  cat("Sample size after emotion filter:", nrow(df_em), "\n")

  et <- paste(c(EMOTION_VARS, TRAIT_VARS, "age", "gender_female"), collapse = " + ")

  m1 <- feols(as.formula(paste("sold ~ n_sales_earlier +", et, "+ signal + period + round + segment | session_id")),
              cluster = ~global_group_id, data = df_em)
  pdata <- pdata.frame(as.data.frame(df_em), index = c("player_id", "time_id"))
  m2 <- plm(as.formula(paste("sold ~ n_sales_earlier +", et, "+ signal + period + round + segment + treatment")),
            data = pdata, model = "random")
  m3 <- feols(as.formula(paste("sold ~ sale_prev_period +", et, "+ signal + period + round + segment | session_id")),
              cluster = ~global_group_id, data = df_em)
  m4 <- plm(as.formula(paste("sold ~ sale_prev_period +", et, "+ signal + period + round + segment + treatment")),
            data = pdata, model = "random")

  var_order <- c("n_sales_earlier", "sale_prev_period", EMOTION_VARS, TRAIT_VARS,
                 "age", "gender_female", "signal", "period", "round",
                 "segment2", "segment3", "segment4", "treatmenttr2", "(Intercept)")

  build_combined_table(list(m1, m2, m3, m4), var_order,
    c("(1)", "(2)", "(3)", "(4)"), c(TRUE, FALSE, TRUE, FALSE),
    c("Clustered", "RE", "Clustered", "RE"), OUTPUT_CASCADE)
}

# =====
# Table 2: First Sellers + Emotions/Traits
# =====
run_first_sellers_table <- function(df) {
  first_ids <- df[prior_group_sales == 0 & sold == 1, unique(player_group_round_id)]
  df_first <- filter_complete_emotions(df[player_group_round_id %in% first_ids])
  cat("First sellers sample:", nrow(df_first), "rows\n")

  et <- paste(c(EMOTION_VARS, TRAIT_VARS, "age", "gender_female"), collapse = " + ")

  m1 <- feols(as.formula(paste("sold ~", et, "+ signal + round + segment | session_id")),
              cluster = ~global_group_id, data = df_first)
  pdata <- pdata.frame(as.data.frame(df_first), index = c("player_id", "time_id"))
  m2 <- plm(as.formula(paste("sold ~", et, "+ signal + round + segment + treatment")),
            data = pdata, model = "random")

  var_order <- c(EMOTION_VARS, TRAIT_VARS, "age", "gender_female", "signal", "round",
                 "segment2", "segment3", "segment4", "treatmenttr2", "(Intercept)")

  build_combined_table(list(m1, m2), var_order, c("(1)", "(2)"),
    c(TRUE, FALSE), c("Clustered", "RE"), OUTPUT_FIRST)
}

# =====
# Table 3: Second Sellers + Emotions/Traits
# =====
run_second_sellers_table <- function(df) {
  second_ids <- df[prior_group_sales == 1 & sold == 1, unique(player_group_round_id)]
  df_second <- df[player_group_round_id %in% second_ids]
  df_second <- create_prev_period_dummy(df_second, df)
  df_second <- filter_complete_emotions(df_second)
  cat("Second sellers sample:", nrow(df_second), "rows\n")

  et <- paste(c(EMOTION_VARS, TRAIT_VARS, "age", "gender_female"), collapse = " + ")

  m1 <- feols(as.formula(paste("sold ~ dummy_prev_period +", et, "+ signal + round + segment | session_id")),
              cluster = ~global_group_id, data = df_second)
  pdata <- pdata.frame(as.data.frame(df_second), index = c("player_id", "time_id"))
  m2 <- plm(as.formula(paste("sold ~ dummy_prev_period +", et, "+ signal + round + segment + treatment")),
            data = pdata, model = "random")

  var_order <- c("dummy_prev_period", EMOTION_VARS, TRAIT_VARS, "age", "gender_female",
                 "signal", "round", "segment2", "segment3", "segment4",
                 "treatmenttr2", "(Intercept)")

  build_combined_table(list(m1, m2), var_order, c("(1)", "(2)"),
    c(TRUE, FALSE), c("Clustered", "RE"), OUTPUT_SECOND)
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
                                  has_session_fe, se_types, output_path) {
  coefs <- lapply(models, extract_coefs)
  fits <- lapply(models, extract_fit)
  nc <- length(models)

  lines <- c(
    "", "\\begingroup", "\\centering", "\\scriptsize",
    sprintf("\\begin{tabular}{l%s}", paste(rep("c", nc), collapse = "")),
    "   \\tabularnewline \\midrule \\midrule",
    sprintf("   Dependent Variable: & \\multicolumn{%d}{c}{sold}\\\\", nc),
    sprintf("   & %s\\\\", paste(col_headers, collapse = " & ")),
    "   \\midrule", "   \\emph{Variables}\\\\"
  )

  for (v in var_order) {
    label <- if (v %in% names(VAR_DICT)) VAR_DICT[v] else gsub("_", "\\\\_", v)
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
    "\\end{tabular}", "\\par\\endgroup", "", "")

  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  writeLines(lines, output_path)
  cat("Table exported to:", output_path, "\n")
}

# %%
if (!interactive()) main()
