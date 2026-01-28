# Purpose: OLS (LPM) regression of selling probability on emotions, traits, and controls
# Author: Claude Code
# Date: 2026-01-28

library(data.table)
library(fixest)

# FILE PATHS
INPUT_PATH <- "datastore/derived/emotions_traits_selling_dataset.csv"
OUTPUT_PATH <- "analysis/output/analysis/selling_emotions_traits_regression.tex"

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  cat("Loading data from:", INPUT_PATH, "\n")
  df <- prepare_data(INPUT_PATH)
  cat("Data dimensions after filtering:", nrow(df), "rows\n")
  print_data_summary(df)

  cat("\nRunning regressions...\n")
  models <- run_models(df)

  cat("\nExporting LaTeX table to:", OUTPUT_PATH, "\n")
  export_table(models, OUTPUT_PATH)

  print_results(models)
  cat("\nDone!\n")
  return(models)
}

# =====
# Data preparation
# =====
prepare_data <- function(file_path) {
  df <- fread(file_path)

  # Filter: only periods where player has not already sold
  df <- df[already_sold == 0]

  # Drop rows with NA emotions (needed for emotion models)
  emotion_cols <- c("anger_mean", "contempt_mean", "disgust_mean", "fear_mean",
                    "joy_mean", "sadness_mean", "surprise_mean",
                    "engagement_mean", "valence_mean")
  df <- df[complete.cases(df[, ..emotion_cols])]

  # Create gender dummy
  df[, gender_female := as.integer(gender == "Female")]

  # Ensure factor types
  df[, session_id := as.factor(session_id)]

  cat("  Rows after filtering already_sold:", sum(fread(file_path)$already_sold == 0), "\n")
  cat("  Rows after dropping NA emotions:", nrow(df), "\n")
  cat("  Unique global_group_ids:", uniqueN(df$global_group_id), "\n")

  return(df)
}

print_data_summary <- function(df) {
  cat("\nVariable summary:\n")
  cat("  Sessions:", nlevels(df$session_id), "\n")
  cat("  Segments:", paste(sort(unique(df$segment)), collapse = ", "), "\n")
  cat("  Rounds:", min(df$round), "-", max(df$round), "\n")
  cat("  Selling rate:", round(mean(df$sold), 4), "\n")

  # Emotion means
  emotion_cols <- c("anger_mean", "contempt_mean", "disgust_mean", "fear_mean",
                    "joy_mean", "sadness_mean", "surprise_mean",
                    "engagement_mean", "valence_mean")
  cat("\nEmotion means:\n")
  for (col in emotion_cols) {
    cat(sprintf("  %s: %.4f\n", col, mean(df[[col]], na.rm = TRUE)))
  }
}

# =====
# Regression models
# =====
run_models <- function(df) {
  models <- list()

  cat("[1/3] Model 1: Emotions + controls...\n")
  models$m1 <- feols(
    sold ~ anger_mean + contempt_mean + disgust_mean + fear_mean +
           joy_mean + sadness_mean + surprise_mean +
           engagement_mean + valence_mean +
           signal + period + round + segment
    | session_id,
    cluster = ~global_group_id,
    data = df
  )

  cat("[2/3] Model 2: Traits + controls...\n")
  models$m2 <- feols(
    sold ~ extraversion + agreeableness + conscientiousness +
           neuroticism + openness + impulsivity + state_anxiety +
           signal + period + round + segment
    | session_id,
    cluster = ~global_group_id,
    data = df
  )

  cat("[3/3] Model 3: Full (emotions + traits + demographics)...\n")
  models$m3 <- feols(
    sold ~ anger_mean + contempt_mean + disgust_mean + fear_mean +
           joy_mean + sadness_mean + surprise_mean +
           engagement_mean + valence_mean +
           extraversion + agreeableness + conscientiousness +
           neuroticism + openness + impulsivity + state_anxiety +
           age + gender_female +
           signal + period + round + segment
    | session_id,
    cluster = ~global_group_id,
    data = df
  )

  return(models)
}

# =====
# Output
# =====
export_table <- function(models, output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  etable(
    models$m1, models$m2, models$m3,
    headers = c("Emotions", "Traits", "Full"),
    dict = c(
      anger_mean = "Anger",
      contempt_mean = "Contempt",
      disgust_mean = "Disgust",
      fear_mean = "Fear",
      joy_mean = "Joy",
      sadness_mean = "Sadness",
      surprise_mean = "Surprise",
      engagement_mean = "Engagement",
      valence_mean = "Valence",
      extraversion = "Extraversion",
      agreeableness = "Agreeableness",
      conscientiousness = "Conscientiousness",
      neuroticism = "Neuroticism",
      openness = "Openness",
      impulsivity = "Impulsivity",
      state_anxiety = "State anxiety",
      age = "Age",
      gender_female = "Female",
      signal = "Signal",
      period = "Period",
      round = "Round",
      segment = "Segment"
    ),
    fitstat = c("n", "r2", "ar2"),
    se.below = TRUE,
    file = output_path,
    float = FALSE,
    tex = TRUE,
    style.tex = style.tex(fontsize = "scriptsize")
  )
  cat("Table exported to:", output_path, "\n")
}

print_results <- function(models) {
  cat("\n", strrep("=", 60), "\n")
  cat("MODEL RESULTS\n")
  cat(strrep("=", 60), "\n")

  cat("\nModel 1 (Emotions + controls):\n")
  print(summary(models$m1))

  cat("\nModel 2 (Traits + controls):\n")
  print(summary(models$m2))

  cat("\nModel 3 (Full):\n")
  print(summary(models$m3))
}

# %%
if (!interactive()) main()
