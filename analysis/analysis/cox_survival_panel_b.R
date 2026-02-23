# Purpose: Panel B (first sellers) Cox survival regression models
# Author: Claude Code
# Date: 2026-02-22

# Libraries loaded by main script (do not load here)
# subset_first_sellers() is defined in unified_selling_regression_panel_c.R

# =====
# Panel B: First Sellers (Cox)
# =====
run_cox_panel_b <- function(df) {
  cat("Cox Panel B: Identifying first sellers...\n")
  df_first <- subset_first_sellers(df)

  cat("  [1/3] Model 1: Controls only (RE Cox)...\n")
  m1 <- run_cox_panel_b_m1(df_first)

  cat("  [2/3] Model 2: Emotions (RE Cox)...\n")
  m2 <- run_cox_panel_b_m2(df_first)

  cat("  [3/3] Model 3: Traits (RE Cox)...\n")
  m3 <- run_cox_panel_b_m3(df_first)

  cat("  Cox Panel B complete.\n")
  list(m1 = m1, m2 = m2, m3 = m3)
}

# =====
# Model 1: Controls only RE Cox
# =====
run_cox_panel_b_m1 <- function(df_first) {
  coxme(
    Surv(period, sold) ~ signal + round + segment + treatment +
      (1 | player_id),
    data = df_first
  )
}

# =====
# Model 2: Emotions RE Cox
# =====
run_cox_panel_b_m2 <- function(df_first) {
  coxme(
    Surv(period, sold) ~ fear_mean + anger_mean + contempt_mean +
      disgust_mean + joy_mean + sadness_mean + surprise_mean +
      engagement_mean + valence_mean +
      signal + round + segment + treatment +
      (1 | player_id),
    data = df_first
  )
}

# =====
# Model 3: Traits RE Cox
# =====
run_cox_panel_b_m3 <- function(df_first) {
  coxme(
    Surv(period, sold) ~ state_anxiety + impulsivity +
      conscientiousness + extraversion + agreeableness +
      neuroticism + openness + risk_tolerance +
      age + gender_female +
      signal + round + segment + treatment +
      (1 | player_id),
    data = df_first
  )
}
