# Purpose: Robustness checks for chat mitigation analysis
# Author: Claude Code
# Date: 2026-02-02
#
# ROBUSTNESS TESTS:
#   A. Alternative Emotion Aggregation (mean vs max)
#   B. Segment-by-Segment Analysis
#   C. Exclude Early Rounds (learning period)
#   D. Pre-Trends Check (segments 1 vs 2, both non-chat)
#   E. Alternative Clustering (player vs group-round)
#   F. Sample Attrition Check (missing iMotions data)

library(data.table)
library(fixest)

# FILE PATHS
INPUT_PATH <- "datastore/derived/chat_mitigation_dataset.csv"
OUTPUT_PATH <- "analysis/output/analysis/chat_mitigation_robustness.tex"

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  print_header("CHAT MITIGATION ROBUSTNESS CHECKS")
  df <- load_and_prepare_data(INPUT_PATH)

  results <- list(
    A = run_emotion_aggregation_check(df),
    B = run_segment_analysis(df),
    C = run_exclude_early_rounds(df),
    D = run_pretrends_check(df),
    E = run_clustering_check(df),
    F = run_attrition_check(df)
  )

  print_summary()
  export_tables(results, OUTPUT_PATH)
  cat("\nRobustness checks complete.\n")
  return(results)
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function(file_path) {
  cat("Loading data from:", file_path, "\n")
  df <- fread(file_path)[already_sold == 0]
  df[, `:=`(segment = as.factor(segment), player_id = as.factor(player_id),
            global_group_id = as.factor(global_group_id),
            group_round = paste0(global_group_id, "_", round))]
  cat("At-risk observations:", nrow(df), "\n")
  return(df)
}

# =====
# A. Alternative Emotion Aggregation
# =====
run_emotion_aggregation_check <- function(df) {
  print_header("A. ALTERNATIVE EMOTION AGGREGATION")
  df[, `:=`(fear_max_z = scale(fear_max)[,1], anger_max_z = scale(anger_max)[,1],
            sadness_max_z = scale(sadness_max)[,1])]

  mean_model <- run_main_spec(df, "mean")
  max_model <- run_main_spec(df, "max")

  cat("\n--- Mean Emotions ---\n"); print_key_coefs(mean_model, "fear_z")
  cat("\n--- Max Emotions ---\n"); print_key_coefs(max_model, "fear_max_z")
  return(list(mean = mean_model, max = max_model))
}

run_main_spec <- function(df, type = "mean") {
  if (type == "mean") {
    feols(sold ~ fear_z + anger_z + sadness_z + fear_z:chat_segment +
            anger_z:chat_segment + sadness_z:chat_segment +
            signal + prior_group_sales + round + chat_segment | player_id,
          cluster = ~player_id, data = df)
  } else {
    feols(sold ~ fear_max_z + anger_max_z + sadness_max_z +
            fear_max_z:chat_segment + anger_max_z:chat_segment +
            sadness_max_z:chat_segment +
            signal + prior_group_sales + round + chat_segment | player_id,
          cluster = ~player_id, data = df)
  }
}

# =====
# B. Segment-by-Segment Analysis
# =====
run_segment_analysis <- function(df) {
  print_header("B. SEGMENT-BY-SEGMENT ANALYSIS")
  non_chat <- run_segment_model(df[chat_segment == 0], "Non-chat (1-2)")
  chat <- run_segment_model(df[chat_segment == 1], "Chat (3-4)")
  cat("\n--- Non-Chat ---\n"); print_key_coefs(non_chat, c("fear_z", "anger_z"))
  cat("\n--- Chat ---\n"); print_key_coefs(chat, c("fear_z", "anger_z"))
  return(list(non_chat = non_chat, chat = chat))
}

run_segment_model <- function(df_sub, label) {
  cat(label, ": N =", nrow(df_sub), "\n")
  feols(sold ~ fear_z + anger_z + sadness_z + signal + prior_group_sales +
          round | player_id, cluster = ~player_id, data = df_sub)
}

# =====
# C. Exclude Early Rounds
# =====
run_exclude_early_rounds <- function(df) {
  print_header("C. EXCLUDE EARLY ROUNDS")
  df_late <- df[round >= 3]
  cat("Late rounds (3+): N =", nrow(df_late), "\n")
  model <- run_main_spec(df_late, "mean")
  cat("\n--- Results ---\n"); print_key_coefs(model, c("fear_z", "fear_z:chat_segment"))
  return(model)
}

# =====
# D. Pre-Trends Check
# =====
run_pretrends_check <- function(df) {
  print_header("D. PRE-TRENDS CHECK")
  cat("Testing if emotion effects differ between s1 and s2 (both non-chat).\n")
  df_pre <- df[segment %in% c(1, 2)]
  df_pre[, segment2 := as.integer(segment == 2)]
  cat("Pre-chat observations: N =", nrow(df_pre), "\n")

  model <- feols(sold ~ fear_z + anger_z + sadness_z + fear_z:segment2 +
                   anger_z:segment2 + sadness_z:segment2 + signal +
                   prior_group_sales + round + segment2 | player_id,
                 cluster = ~player_id, data = df_pre)
  cat("\n--- Segment 2 interactions (should NOT be significant) ---\n")
  print_key_coefs(model, c("fear_z:segment2", "anger_z:segment2"))
  return(model)
}

# =====
# E. Alternative Clustering
# =====
run_clustering_check <- function(df) {
  print_header("E. ALTERNATIVE CLUSTERING")
  player_cluster <- run_main_spec(df, "mean")
  group_round_cluster <- feols(
    sold ~ fear_z + anger_z + sadness_z + fear_z:chat_segment +
      anger_z:chat_segment + sadness_z:chat_segment +
      signal + prior_group_sales + round + chat_segment | player_id,
    cluster = ~group_round, data = df)

  cat("\n--- Player Clustering ---\n"); print_key_coefs(player_cluster, "fear_z:chat_segment")
  cat("\n--- Group-Round Clustering ---\n"); print_key_coefs(group_round_cluster, "fear_z:chat_segment")
  return(list(player = player_cluster, group_round = group_round_cluster))
}

# =====
# F. Sample Attrition Check
# =====
run_attrition_check <- function(df) {
  print_header("F. SAMPLE ATTRITION CHECK")
  df[, has_emotions := !is.na(fear_mean)]
  cat("\nBy emotion availability:\n")
  print(df[, .(n = .N, sell_rate = round(mean(sold), 4)), by = has_emotions])
  cat("\nMissing rate by segment:\n")
  print(df[, .(missing_pct = round(100 * mean(!has_emotions), 1)), by = segment])

  model <- feols(has_emotions ~ chat_segment + round + signal + prior_group_sales,
                 cluster = ~player_id, data = df)
  cat("\nPredicting missingness:\n"); print(summary(model))
  return(list(summary = df[, .N, by = has_emotions], model = model))
}

# =====
# Summary and Export
# =====
print_summary <- function() {
  print_header("ROBUSTNESS CHECK SUMMARY")
  cat("A. Emotion Aggregation: Check coefficient consistency\n",
      "B. Segment Analysis: Compare emotion coefficients across segments\n",
      "C. Early Rounds Excluded: Results should hold after learning period\n",
      "D. Pre-Trends: Segment 2 interactions should NOT be significant\n",
      "E. Clustering: Results should be robust to clustering choice\n",
      "F. Attrition: Missingness should not be strongly predicted\n", sep = "")
}

export_tables <- function(results, output_path) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  export_main_table(results, output_path)
  export_segment_table(results, gsub("\\.tex$", "_segments.tex", output_path))
  export_pretrends_table(results, gsub("\\.tex$", "_pretrends.tex", output_path))
  cat("Exported tables to:", dirname(output_path), "\n")
}

export_main_table <- function(results, path) {
  etable(results$A$mean, results$A$max, results$C,
         headers = c("Mean Emotions", "Max Emotions", "Late Rounds"),
         dict = get_var_dict(), fitstat = c("n", "r2"), file = path,
         float = FALSE, tex = TRUE, style.tex = style.tex(fontsize = "scriptsize"))
}

export_segment_table <- function(results, path) {
  etable(results$B$non_chat, results$B$chat, headers = c("Non-Chat", "Chat"),
         dict = get_var_dict(), fitstat = c("n", "r2"), file = path,
         float = FALSE, tex = TRUE, style.tex = style.tex(fontsize = "scriptsize"))
}

export_pretrends_table <- function(results, path) {
  dict <- c(get_var_dict(), "fear_z:segment2" = "Fear $\\times$ Seg2",
            "anger_z:segment2" = "Anger $\\times$ Seg2",
            "sadness_z:segment2" = "Sadness $\\times$ Seg2", "segment2" = "Segment 2")
  etable(results$D, headers = "Pre-Trends", dict = dict, fitstat = c("n", "r2"),
         file = path, float = FALSE, tex = TRUE, style.tex = style.tex(fontsize = "scriptsize"))
}

get_var_dict <- function() {
  c("fear_z" = "Fear (z)", "anger_z" = "Anger (z)", "sadness_z" = "Sadness (z)",
    "fear_max_z" = "Fear max (z)", "anger_max_z" = "Anger max (z)",
    "sadness_max_z" = "Sadness max (z)", "fear_z:chat_segment" = "Fear $\\times$ Chat",
    "anger_z:chat_segment" = "Anger $\\times$ Chat",
    "sadness_z:chat_segment" = "Sadness $\\times$ Chat",
    "fear_max_z:chat_segment" = "Fear max $\\times$ Chat",
    "anger_max_z:chat_segment" = "Anger max $\\times$ Chat",
    "sadness_max_z:chat_segment" = "Sadness max $\\times$ Chat",
    "signal" = "Signal", "prior_group_sales" = "Prior Group Sales",
    "round" = "Round", "chat_segment" = "Chat Segment")
}

# =====
# Utility functions
# =====
print_header <- function(title) {
  cat("\n", rep("=", 60), "\n", title, "\n", rep("=", 60), "\n", sep = "")
}

print_key_coefs <- function(model, vars) {
  coefs <- coef(model); se <- sqrt(diag(vcov(model)))
  for (v in vars[vars %in% names(coefs)]) {
    t_val <- abs(coefs[v] / se[v])
    sig <- ifelse(t_val > 2.576, "***", ifelse(t_val > 1.96, "**",
                                               ifelse(t_val > 1.645, "*", "")))
    cat(sprintf("  %s: %.4f (%.4f) %s\n", v, coefs[v], se[v], sig))
  }
}

# %%
if (!interactive()) main()
