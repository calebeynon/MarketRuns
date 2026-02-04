# Purpose: Analyze whether chat mitigates negative emotion effects on selling decisions
# Author: Claude Code
# Date: 2026-02-02
#
# HYPOTHESIS: Negative emotions (fear, anger) drive selling, but chat communication
# provides an outlet that weakens this effect. We test this via emotion x chat_segment
# interactions, using player fixed effects to control for unobserved heterogeneity.
#
# INTERPRETATION:
#   - Main effect (e.g., fear_z): Effect of fear on selling in NON-chat segments
#   - Interaction (e.g., fear_z:chat_segment): How much DIFFERENT is fear effect in chat
#   - POSITIVE interaction = chat MITIGATES the negative emotion effect
#   - If fear_z > 0 (more fear -> more selling) and interaction < 0, chat reduces effect

library(data.table)
library(fixest)

# FILE PATHS
INPUT_PATH <- "datastore/derived/chat_mitigation_dataset.csv"
OUTPUT_PATH <- "analysis/output/analysis/chat_mitigation_emotions.tex"

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  print_header("CHAT MITIGATION REGRESSION ANALYSIS")
  cat("\nLoading data from:", INPUT_PATH, "\n")
  df <- load_and_prepare_data(INPUT_PATH)

  print_summary_statistics(df)
  print_emotion_by_chat_segment(df)
  print_header("RUNNING REGRESSION MODELS")

  models <- run_all_models(df)
  print_model_results(models)
  export_latex_table(models, OUTPUT_PATH)
  cat("\nDone!\n")
  return(models)
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function(file_path) {
  df <- fread(file_path)
  df <- filter_at_risk(df)
  df <- create_factors(df)
  df <- validate_required_vars(df)
  return(df)
}

filter_at_risk <- function(df) {
  n_before <- nrow(df)
  df <- df[already_sold == 0]
  cat("Filtered to at-risk periods:", nrow(df), "(dropped",
      n_before - nrow(df), "already sold)\n")
  return(df)
}

create_factors <- function(df) {
  df[, segment := as.factor(segment)]
  df[, treatment := as.factor(treatment)]
  df[, player_id := as.factor(player_id)]
  df[, global_group_id := as.factor(global_group_id)]
  return(df)
}

validate_required_vars <- function(df) {
  required <- c("sold", "fear_z", "anger_z", "sadness_z", "chat_segment",
                "signal", "prior_group_sales", "round", "player_id")
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) stop("Missing: ", paste(missing, collapse = ", "))
  return(df)
}

# =====
# Summary statistics
# =====
print_summary_statistics <- function(df) {
  print_header("SAMPLE SUMMARY")
  cat("\nSample size:", nrow(df), "player-period observations\n")
  cat("Unique players:", uniqueN(df$player_id), "\n")
  cat("Unique groups:", uniqueN(df$global_group_id), "\n")
  cat("\nSelling rate (outcome):", round(mean(df$sold), 4), "\n")
  cat("Sales this period:", sum(df$sold), "\n")
  cat("\nObservations by chat_segment:\n")
  print(df[, .(n = .N, pct_sold = round(mean(sold), 4)), by = chat_segment])
}

print_emotion_by_chat_segment <- function(df) {
  print_header("EMOTION MEANS BY CHAT SEGMENT")
  df_emo <- df[!is.na(fear_mean)]
  print(compute_emotion_summary(df_emo))
  cat("\nWithin-player variance of emotions:\n")
  print_within_player_variance(df_emo)
}

compute_emotion_summary <- function(df_emo) {
  df_emo[, .(
    n = .N,
    fear_mean = round(mean(fear_mean, na.rm = TRUE), 4),
    anger_mean = round(mean(anger_mean, na.rm = TRUE), 4),
    sadness_mean = round(mean(sadness_mean, na.rm = TRUE), 4),
    sell_rate = round(mean(sold), 4)
  ), by = chat_segment]
}

print_within_player_variance <- function(df_emo) {
  within_var <- df_emo[, .(
    fear_var = var(fear_z, na.rm = TRUE),
    anger_var = var(anger_z, na.rm = TRUE),
    sadness_var = var(sadness_z, na.rm = TRUE)
  ), by = player_id]
  cat("  fear_z:", round(mean(within_var$fear_var, na.rm = TRUE), 4), "\n")
  cat("  anger_z:", round(mean(within_var$anger_var, na.rm = TRUE), 4), "\n")
  cat("  sadness_z:", round(mean(within_var$sadness_var, na.rm = TRUE), 4), "\n")
}

# =====
# Regression models
# =====
run_all_models <- function(df) {
  models <- list()
  models$m1_baseline <- run_model_baseline(df)
  models$m2_interactions <- run_model_interactions(df)
  models$m3_max_emotions <- run_model_max_emotions(df)
  return(models)
}

run_model_baseline <- function(df) {
  cat("\n[1/3] Model 1: Baseline (emotions + controls, no interactions)...\n")
  feols(
    sold ~ fear_z + anger_z + sadness_z +
      signal + prior_group_sales + round + period + chat_segment | player_id,
    cluster = ~player_id, data = df
  )
}

run_model_interactions <- function(df) {
  cat("[2/3] Model 2: Emotion x Chat interactions (MAIN SPECIFICATION)...\n")
  feols(
    sold ~ fear_z + anger_z + sadness_z +
      fear_z:chat_segment + anger_z:chat_segment + sadness_z:chat_segment +
      signal + prior_group_sales + round + period + chat_segment | player_id,
    cluster = ~player_id, data = df
  )
}

run_model_max_emotions <- function(df) {
  cat("[3/3] Model 3: Max emotions (robustness)...\n")
  df <- standardize_max_emotions(df)
  feols(
    sold ~ fear_max_z + anger_max_z + sadness_max_z +
      fear_max_z:chat_segment + anger_max_z:chat_segment +
      sadness_max_z:chat_segment +
      signal + prior_group_sales + round + period + chat_segment | player_id,
    cluster = ~player_id, data = df
  )
}

standardize_max_emotions <- function(df) {
  df[, fear_max_z := scale(fear_max)[, 1]]
  df[, anger_max_z := scale(anger_max)[, 1]]
  df[, sadness_max_z := scale(sadness_max)[, 1]]
  return(df)
}

# =====
# Results output
# =====
print_model_results <- function(models) {
  print_header("MODEL RESULTS SUMMARY")
  cat("\nModel 1: Baseline (no interactions)\n")
  print(summary(models$m1_baseline))
  print_model_separator("Model 2: MAIN SPECIFICATION (Emotion x Chat)")
  print(summary(models$m2_interactions))
  print_interaction_interpretation(models$m2_interactions)
  print_model_separator("Model 3: Max emotions (robustness)")
  print(summary(models$m3_max_emotions))
}

print_model_separator <- function(title) {
  cat("\n", rep("-", 60), "\n", sep = "")
  cat(title, "\n")
}

print_interaction_interpretation <- function(model) {
  cat("\n--- INTERPRETATION OF MAIN SPECIFICATION ---\n")
  print_key_coefficients(model)
  cat("\nInterpretation:\n")
  cat("  - fear_z: Effect of 1 SD fear on selling probability (non-chat)\n")
  cat("  - fear_z:chat_segment: Change in fear effect when chat available\n")
  cat("  - Positive interaction = chat MITIGATES emotion-driven selling\n")
}

print_key_coefficients <- function(model) {
  coefs <- coef(model)
  se <- sqrt(diag(vcov(model)))
  vars <- c("fear_z", "anger_z", "sadness_z", "fear_z:chat_segment",
            "anger_z:chat_segment", "sadness_z:chat_segment")
  for (v in vars[vars %in% names(coefs)]) {
    cat(sprintf("  %s: %.4f (%.4f) %s\n", v, coefs[v], se[v],
                get_sig_stars(coefs[v], se[v])))
  }
}

get_sig_stars <- function(est, se) {
  t_val <- abs(est / se)
  if (t_val > 2.576) return("***")
  if (t_val > 1.96) return("**")
  if (t_val > 1.645) return("*")
  return("")
}

# =====
# LaTeX table export
# =====
export_latex_table <- function(models, output_path) {
  ensure_output_dir(output_path)
  cat("\nExporting LaTeX table to:", output_path, "\n")
  etable(
    models$m1_baseline, models$m2_interactions, models$m3_max_emotions,
    headers = c("Baseline", "Interactions", "Max Emotions"),
    dict = get_var_dict(),
    fitstat = c("n", "r2", "ar2"),
    file = output_path, float = FALSE, tex = TRUE,
    style.tex = style.tex(fontsize = "scriptsize")
  )
}

ensure_output_dir <- function(output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
}

get_var_dict <- function() {
  c(
    "fear_z" = "Fear (z)", "anger_z" = "Anger (z)", "sadness_z" = "Sadness (z)",
    "fear_max_z" = "Fear max (z)", "anger_max_z" = "Anger max (z)",
    "sadness_max_z" = "Sadness max (z)",
    "fear_z:chat_segment" = "Fear (z) $\\times$ Chat",
    "anger_z:chat_segment" = "Anger (z) $\\times$ Chat",
    "sadness_z:chat_segment" = "Sadness (z) $\\times$ Chat",
    "fear_max_z:chat_segment" = "Fear max (z) $\\times$ Chat",
    "anger_max_z:chat_segment" = "Anger max (z) $\\times$ Chat",
    "sadness_max_z:chat_segment" = "Sadness max (z) $\\times$ Chat",
    "signal" = "Signal", "prior_group_sales" = "Prior group sales",
    "round" = "Round", "period" = "Period", "chat_segment" = "Chat segment"
  )
}

# =====
# Utility functions
# =====
print_header <- function(title) {
  cat("\n", rep("=", 60), "\n", sep = "")
  cat(title, "\n")
  cat(rep("=", 60), "\n", sep = "")
}

# %%
if (!interactive()) main()
