# Purpose: Unified regression analysis for selling probability with emotions, traits, and cascade interactions
# Author: Claude Code
# Date: 2026-02-01

library(data.table)
library(plm)

# FILE PATHS
INPUT_PATH <- "datastore/derived/emotions_traits_selling_dataset.csv"
OUTPUT_FULL <- "analysis/output/tables/selling_emotions_traits_full.tex"
OUTPUT_FIRST <- "analysis/output/tables/selling_emotions_traits_first.tex"
OUTPUT_SECOND <- "analysis/output/tables/selling_emotions_traits_second.tex"

# VARIABLE LISTS
SHOW_EMOTIONS <- c("fear_mean", "anger_mean")
SHOW_TRAITS <- c("state_anxiety", "impulsivity", "conscientiousness")
HIDE_EMOTIONS <- c("contempt_mean", "disgust_mean", "joy_mean", "sadness_mean",
                   "surprise_mean", "engagement_mean", "valence_mean")
HIDE_TRAITS <- c("extraversion", "agreeableness", "neuroticism", "openness")

ALL_EMOTIONS <- c(SHOW_EMOTIONS, HIDE_EMOTIONS)
ALL_TRAITS <- c(SHOW_TRAITS, HIDE_TRAITS)

VAR_DICT <- c(
  "(Intercept)" = "Constant",
  n_sales_earlier = "n\\_sales\\_earlier",
  sale_prev_period = "sale\\_prev\\_period",
  dummy_prev_period = "dummy\\_prev\\_period",
  fear_mean = "Fear", anger_mean = "Anger",
  state_anxiety = "State anxiety", impulsivity = "Impulsivity",
  conscientiousness = "Conscientiousness",
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

  cat("\n=== Table 1: Full Sample with Cascade Interactions ===\n")
  run_full_sample_table(df)

  cat("\n=== Table 2: First Sellers ===\n")
  run_first_sellers_table(df)

  cat("\n=== Table 3: Second Sellers ===\n")
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
  df <- create_sale_timing_vars(df)
  df[, gender_female := as.integer(gender == "Female")]
  df[, time_id := paste(segment, round, period, sep = "_")]

  df[, segment := as.factor(segment)]
  df[, treatment := as.factor(treatment)]
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
  df[complete.cases(df[, ..ALL_EMOTIONS])]
}

# =====
# Table 1: Full Sample with Both Cascade Variables and Interactions
# =====
run_full_sample_table <- function(df) {
  df_em <- filter_complete_emotions(df)
  cat("Sample size after emotion filter:", nrow(df_em), "\n")

  # Build interaction terms (5 emotion/trait vars x 2 cascade vars = 10 interactions)
  int_vars <- c(SHOW_EMOTIONS, SHOW_TRAITS)
  int_nse <- paste0("n_sales_earlier:", int_vars, collapse = " + ")
  int_spp <- paste0("sale_prev_period:", int_vars, collapse = " + ")

  # All controls (shown + hidden)
  all_controls <- paste(c(ALL_EMOTIONS, ALL_TRAITS, "age", "gender_female"), collapse = " + ")

  formula_str <- paste("sold ~ n_sales_earlier + sale_prev_period +",
                       int_nse, "+", int_spp, "+",
                       all_controls, "+ signal + period + round + segment + treatment")

  pdata <- pdata.frame(as.data.frame(df_em), index = c("player_id", "time_id"))
  model <- plm(as.formula(formula_str), data = pdata, model = "random")

  # Build variable order for display
  int_names_nse <- paste0("n_sales_earlier:", int_vars)
  int_names_spp <- paste0("sale_prev_period:", int_vars)
  int_labels_nse <- paste0("n\\_sales\\_earlier $\\times$ ", VAR_DICT[int_vars])
  int_labels_spp <- paste0("sale\\_prev\\_period $\\times$ ", VAR_DICT[int_vars])
  int_dict <- setNames(c(int_labels_nse, int_labels_spp), c(int_names_nse, int_names_spp))

  var_order <- c("n_sales_earlier", "sale_prev_period",
                 int_names_nse, int_names_spp,
                 SHOW_EMOTIONS, SHOW_TRAITS,
                 "signal", "period", "round",
                 "segment2", "segment3", "segment4", "treatmenttr2", "(Intercept)")

  build_single_table(model, var_order, OUTPUT_FULL, extra_dict = int_dict)
}

# =====
# Table 2: First Sellers (no cascade variables, no period)
# =====
run_first_sellers_table <- function(df) {
  first_ids <- df[prior_group_sales == 0 & sold == 1, unique(player_group_round_id)]
  df_first <- filter_complete_emotions(df[player_group_round_id %in% first_ids])
  cat("First sellers sample:", nrow(df_first), "rows\n")

  all_controls <- paste(c(ALL_EMOTIONS, ALL_TRAITS, "age", "gender_female"), collapse = " + ")
  formula_str <- paste("sold ~", all_controls, "+ signal + round + segment + treatment")

  pdata <- pdata.frame(as.data.frame(df_first), index = c("player_id", "time_id"))
  model <- plm(as.formula(formula_str), data = pdata, model = "random")

  var_order <- c(SHOW_EMOTIONS, SHOW_TRAITS, "signal", "round",
                 "segment2", "segment3", "segment4", "treatmenttr2", "(Intercept)")

  build_single_table(model, var_order, OUTPUT_FIRST)
}

# =====
# Table 3: Second Sellers (with dummy_prev_period)
# =====
run_second_sellers_table <- function(df) {
  second_ids <- df[prior_group_sales == 1 & sold == 1, unique(player_group_round_id)]
  df_second <- df[player_group_round_id %in% second_ids]
  df_second <- create_prev_period_dummy(df_second, df)
  df_second <- filter_complete_emotions(df_second)
  cat("Second sellers sample:", nrow(df_second), "rows\n")

  all_controls <- paste(c(ALL_EMOTIONS, ALL_TRAITS, "age", "gender_female"), collapse = " + ")
  formula_str <- paste("sold ~ dummy_prev_period +", all_controls,
                       "+ signal + round + segment + treatment")

  pdata <- pdata.frame(as.data.frame(df_second), index = c("player_id", "time_id"))
  model <- plm(as.formula(formula_str), data = pdata, model = "random")

  var_order <- c("dummy_prev_period", SHOW_EMOTIONS, SHOW_TRAITS, "signal", "round",
                 "segment2", "segment3", "segment4", "treatmenttr2", "(Intercept)")

  build_single_table(model, var_order, OUTPUT_SECOND)
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
# Coefficient extraction
# =====
extract_coefs <- function(model) {
  ct <- coef(summary(model))
  data.table(var = rownames(ct), est = ct[, 1], se = ct[, 2], pval = ct[, 4])
}

extract_fit <- function(model) {
  s <- summary(model)
  rv <- s$r.squared
  list(n = nobs(model), r2 = rv[1], ar2 = if (length(rv) > 1) rv[2] else NA)
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

build_single_table <- function(model, var_order, output_path, extra_dict = NULL) {
  full_dict <- c(VAR_DICT, extra_dict)
  coefs <- extract_coefs(model)
  fit <- extract_fit(model)

  lines <- build_table_header()
  lines <- append_coefficient_rows(lines, coefs, var_order, full_dict)
  lines <- append_fit_statistics(lines, fit)
  lines <- append_table_footer(lines)

  write_table(lines, output_path)
}

build_table_header <- function() {
  c("", "\\begingroup", "\\scriptsize",
    "\\begin{longtable}{lc}",
    "   \\midrule \\midrule",
    "   Dependent Variable: & sold\\\\",
    "   & (1)\\\\",
    "   \\midrule",
    "   \\emph{Variables}\\\\",
    "   \\endfirsthead",
    "   \\multicolumn{2}{l}{\\emph{(continued)}}\\\\",
    "   \\midrule \\midrule",
    "   Dependent Variable: & sold\\\\",
    "   & (1)\\\\",
    "   \\midrule",
    "   \\endhead",
    "   \\midrule",
    "   \\multicolumn{2}{r}{\\emph{continued on next page}}\\\\",
    "   \\endfoot",
    "   \\endlastfoot")
}

append_coefficient_rows <- function(lines, coefs, var_order, full_dict) {
  for (v in var_order) {
    label <- if (v %in% names(full_dict)) full_dict[v] else gsub("_", "\\\\_", v)
    row <- coefs[var == v]
    if (nrow(row) == 0) next
    val <- sprintf("%.4f%s", row$est, get_sig_stars(row$pval))
    se <- sprintf("(%.4f)", row$se)
    lines <- c(lines, sprintf("   %s & %s\\\\", label, val), sprintf("   & %s\\\\", se))
  }
  return(lines)
}

append_fit_statistics <- function(lines, fit) {
  c(lines, "   \\midrule", "   \\emph{Fit statistics}\\\\",
    "   Model & Random Effects\\\\",
    sprintf("   Observations & %s\\\\", format(fit$n, big.mark = ",")),
    sprintf("   R$^2$ & %.5f\\\\", fit$r2))
}

append_table_footer <- function(lines) {
  note <- paste("Other AFFDEX emotions, other BFI-10 traits, age, and gender",
                "also included as controls.")
  c(lines, "   \\midrule \\midrule",
    "   \\multicolumn{2}{l}{\\emph{Signif. Codes: ***: 0.01, **: 0.05, *: 0.1}}\\\\",
    sprintf("   \\multicolumn{2}{p{0.7\\linewidth}}{\\emph{Note: %s}}\\\\", note),
    "\\end{longtable}", "\\endgroup", "", "")
}

write_table <- function(lines, output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  writeLines(lines, output_path)
  cat("Table exported to:", output_path, "\n")
}

# %%
if (!interactive()) main()
