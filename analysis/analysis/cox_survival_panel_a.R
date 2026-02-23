# Purpose: Panel A (all participants) Cox survival regression models
# Author: Claude Code
# Date: 2026-02-22

# Libraries loaded by main script (do not load here)

# =====
# Panel A: All Participants (Cox)
# =====
run_cox_panel_a <- function(df) {
  cat("Panel A (Cox) sample size:", nrow(df), "\n")

  m1 <- run_cox_panel_a_m1(df)
  m2 <- run_cox_panel_a_m2(df)
  m3 <- run_cox_panel_a_m3(df)

  cat("\n--- Panel A Cox Model Summaries ---\n")
  cat("\nModel 1 (Cascade RE Cox):\n")
  print(summary(m1))
  cat("\nModel 2 (Cascade + Emotions RE Cox):\n")
  print(summary(m2))
  cat("\nModel 3 (Cascade + Traits RE Cox):\n")
  print(summary(m3))

  list(m1 = m1, m2 = m2, m3 = m3)
}

# =====
# Model 1: Cascade RE Cox
# =====
run_cox_panel_a_m1 <- function(df) {
  cat("[Panel A M1] Cascade RE Cox (coxme)...\n")
  coxme(
    Surv(period, sold) ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
      int_1_1 + int_2_1 + int_2_2 + int_3_1 + int_3_2 + int_3_3 +
      signal + round + segment + treatment +
      (1 | player_id),
    data = df
  )
}

# =====
# Model 2: Cascade + Emotions RE Cox
# =====
run_cox_panel_a_m2 <- function(df) {
  cat("[Panel A M2] Cascade + Emotions RE Cox (coxme)...\n")
  coxme(
    Surv(period, sold) ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
      int_1_1 + int_2_1 + int_2_2 + int_3_1 + int_3_2 + int_3_3 +
      fear_mean + anger_mean + contempt_mean + disgust_mean +
      joy_mean + sadness_mean + surprise_mean + engagement_mean +
      valence_mean +
      signal + round + segment + treatment +
      (1 | player_id),
    data = df
  )
}

# =====
# Model 3: Cascade + Traits RE Cox
# =====
run_cox_panel_a_m3 <- function(df) {
  cat("[Panel A M3] Cascade + Traits RE Cox (coxme)...\n")
  coxme(
    Surv(period, sold) ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
      int_1_1 + int_2_1 + int_2_2 + int_3_1 + int_3_2 + int_3_3 +
      state_anxiety + impulsivity + conscientiousness +
      extraversion + agreeableness + neuroticism + openness +
      risk_tolerance + age + gender_female +
      signal + round + segment + treatment +
      (1 | player_id),
    data = df
  )
}
