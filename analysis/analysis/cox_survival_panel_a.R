# Purpose: Panel A (all sellers) Cox survival regression — emotion/valence split
# Author: Claude Code
# Date: 2026-02-23

# Libraries loaded by main script (do not load here)

# =====
# Panel A: All Sellers — 2-model emotion/valence split
# =====
run_cox_panel_a <- function(df) {
  cat("Panel A (Cox) sample size:", nrow(df), "\n")

  m1 <- run_cox_panel_a_emotions(df)
  m2 <- run_cox_panel_a_valence_only(df)

  cat("\n--- Panel A Cox Model Summaries ---\n")
  cat("\nModel 1 (Discrete Emotions RE Cox):\n")
  print(summary(m1))
  cat("\nModel 2 (Valence-Only RE Cox):\n")
  print(summary(m2))

  list(m1 = m1, m2 = m2)
}

# =====
# Model 1: Discrete emotions RE Cox
# =====
run_cox_panel_a_emotions <- function(df) {
  cat("[Panel A M1] Discrete emotions RE Cox (coxme)...\n")
  coxme(
    Surv(period_start, period, sold) ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
      int_1_1 + int_2_1 + int_2_2 + int_3_1 + int_3_2 + int_3_3 +
      fear_mean + anger_mean + contempt_mean + disgust_mean +
      joy_mean + sadness_mean + surprise_mean + engagement_mean +
      signal + round + segment + treatment + risk_tolerance +
      age + gender_female +
      (1 | player_id),
    data = df
  )
}

# =====
# Model 2: Valence-only RE Cox
# =====
run_cox_panel_a_valence_only <- function(df) {
  cat("[Panel A M2] Valence-only RE Cox (coxme)...\n")
  coxme(
    Surv(period_start, period, sold) ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
      int_1_1 + int_2_1 + int_2_2 + int_3_1 + int_3_2 + int_3_3 +
      valence_mean +
      signal + round + segment + treatment + risk_tolerance +
      age + gender_female +
      (1 | player_id),
    data = df
  )
}
