# Purpose: Shared constants and helpers for unified selling regression scripts
# Author: Claude Code
# Date: 2026-02-17

# VARIABLE LISTS
SHOW_EMOTIONS <- c("fear_mean", "anger_mean")
SHOW_TRAITS <- c("state_anxiety", "impulsivity", "conscientiousness")
HIDE_EMOTIONS <- c("contempt_mean", "disgust_mean", "joy_mean", "sadness_mean",
                   "surprise_mean", "engagement_mean", "valence_mean")
HIDE_TRAITS <- c("extraversion", "agreeableness", "neuroticism", "openness")

ALL_EMOTIONS <- c(SHOW_EMOTIONS, HIDE_EMOTIONS)
ALL_TRAITS <- c(SHOW_TRAITS, HIDE_TRAITS)

VAR_LABELS <- c(
  dummy_1_cum = "1 prior sale",
  dummy_2_cum = "2 prior sales",
  dummy_3_cum = "3 prior sales",
  int_1_1 = "1 prior $\\times$ 1 prev.",
  int_2_1 = "2 prior $\\times$ 1 prev.",
  int_2_2 = "2 prior $\\times$ 2 prev.",
  int_3_1 = "3 prior $\\times$ 1 prev.",
  int_3_2 = "3 prior $\\times$ 2 prev.",
  int_3_3 = "3 prior $\\times$ 3 prev.",
  dummy_prev_period = "Sale in previous period",
  fear_mean = "Fear", anger_mean = "Anger",
  contempt_mean = "Contempt", disgust_mean = "Disgust",
  joy_mean = "Joy", sadness_mean = "Sadness",
  surprise_mean = "Surprise", engagement_mean = "Engagement",
  valence_mean = "Valence",
  state_anxiety = "State anxiety", impulsivity = "Impulsivity",
  conscientiousness = "Conscientiousness",
  extraversion = "Extraversion", agreeableness = "Agreeableness",
  neuroticism = "Neuroticism", openness = "Openness",
  age = "Age", gender_female = "Female",
  signal = "Signal", period = "Period", round = "Round",
  segment2 = "Segment 2", segment3 = "Segment 3", segment4 = "Segment 4",
  treatmenttr2 = "Treatment 2"
)

# DISPLAY HEADERS AND VARIABLE GROUPS
EMOTION_HEADER <- "__header__Facial emotions"
TRAIT_HEADER <- "__header__Personality traits"
INTERACTION_HEADER <- "__header__Cumulative $\\times$ prev.-period interactions"
INTERACTION_VARS <- c("int_1_1", "int_2_1", "int_2_2",
                       "int_3_1", "int_3_2", "int_3_3")
PERSON_VARS <- c(EMOTION_HEADER, SHOW_EMOTIONS,
                 TRAIT_HEADER, SHOW_TRAITS, "treatmenttr2")
ALL_PERSON_VARS <- c(EMOTION_HEADER, ALL_EMOTIONS,
                     TRAIT_HEADER, ALL_TRAITS, "treatmenttr2")

# =====
# Previous-period dummy construction
# =====
create_prev_period_dummies <- function(df) {
  period_sales <- df[, .(n_sales = sum(sold)),
                     by = .(group_round_id, period)]
  setorder(period_sales, group_round_id, period)
  period_sales[, prev_n_sales := shift(n_sales, 1, type = "lag"),
               by = group_round_id]
  df <- merge(df,
              period_sales[, .(group_round_id, period, prev_n_sales)],
              by = c("group_round_id", "period"), all.x = TRUE)
  df[, prev_n_sales := fifelse(is.na(prev_n_sales), 0L,
                                as.integer(prev_n_sales))]
  df[, dummy_1_prev := as.integer(prev_n_sales == 1)]
  df[, dummy_2_prev := as.integer(prev_n_sales == 2)]
  df[, dummy_3_prev := as.integer(prev_n_sales == 3)]
  df[, prev_n_sales := NULL]
  df
}

# =====
# Interaction term construction (cumulative x previous-period)
# =====
create_interaction_terms <- function(df) {
  df[, int_1_1 := dummy_1_cum * dummy_1_prev]
  df[, int_2_1 := dummy_2_cum * dummy_1_prev]
  df[, int_2_2 := dummy_2_cum * dummy_2_prev]
  df[, int_3_1 := dummy_3_cum * dummy_1_prev]
  df[, int_3_2 := dummy_3_cum * dummy_2_prev]
  df[, int_3_3 := dummy_3_cum * dummy_3_prev]
  df
}

# =====
# LaTeX helpers
# =====
get_stars <- function(pval) {
  if (pval < 0.01) return("$^{***}$")
  if (pval < 0.05) return("$^{**}$")
  if (pval < 0.1) return("$^{*}$")
  ""
}

format_coef_row <- function(var_name, coefs_list) {
  label <- if (var_name %in% names(VAR_LABELS)) {
    VAR_LABELS[var_name]
  } else {
    gsub("_", "\\\\_", var_name)
  }
  cells <- format_coef_cells(var_name, coefs_list)
  c(sprintf("   %-25s & %s & %s & %s \\\\", label, cells$v[1], cells$v[2], cells$v[3]),
    sprintf("   %-25s & %s & %s & %s \\\\", "", cells$s[1], cells$s[2], cells$s[3]))
}

format_coef_cells <- function(var_name, coefs_list) {
  vals <- ses <- character(3)
  for (i in seq_along(coefs_list)) {
    row <- coefs_list[[i]][var == var_name]
    if (nrow(row) == 0) next
    vals[i] <- paste0(sprintf("%.4f", row$est), get_stars(row$pval))
    ses[i] <- paste0("(", sprintf("%.4f", row$se), ")")
  }
  list(v = vals, s = ses)
}

write_table <- function(lines, path) {
  output_dir <- dirname(path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  writeLines(lines, path)
  cat("Table exported to:", path, "\n")
}
