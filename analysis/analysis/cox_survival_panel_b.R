# Purpose: Panel B (first sellers) Cox survival regression models
# Author: Claude Code
# Date: 2026-02-23

# Libraries loaded by main script (do not load here)
# subset_first_sellers() is defined in unified_selling_regression_panel_c.R

# =====
# Panel B: First Sellers (Cox)
# =====
run_cox_panel_b <- function(df) {
  cat("Cox Panel B: Identifying first sellers...\n")
  df_first <- subset_first_sellers(df)

  cat("  [1/2] Model 1: Discrete emotions (RE Cox)...\n")
  m1 <- run_cox_panel_b_m1(df_first)

  cat("  [2/2] Model 2: Valence only (RE Cox)...\n")
  m2 <- run_cox_panel_b_m2(df_first)

  cat("  Cox Panel B complete.\n")
  list(m1 = m1, m2 = m2)
}

# =====
# Model 1: Discrete emotions RE Cox (no cascade for first sellers)
# =====
run_cox_panel_b_m1 <- function(df_first) {
  f <- Surv(period_start, period, sold) ~ fear_mean + anger_mean + contempt_mean +
    disgust_mean + joy_mean + sadness_mean + surprise_mean + engagement_mean +
    state_anxiety + impulsivity + conscientiousness +
    extraversion + agreeableness + neuroticism + openness +
    risk_tolerance +
    signal + round + segment + treatment +
    age + gender_female +
    (1 | player_id)
  coxme(f, data = df_first, init = get_coxph_init(f, df_first))
}

# =====
# Model 2: Valence only RE Cox (no cascade for first sellers)
# =====
run_cox_panel_b_m2 <- function(df_first) {
  f <- Surv(period_start, period, sold) ~ valence_mean +
    state_anxiety + impulsivity + conscientiousness +
    extraversion + agreeableness + neuroticism + openness +
    risk_tolerance +
    signal + round + segment + treatment +
    age + gender_female +
    (1 | player_id)
  coxme(f, data = df_first, init = get_coxph_init(f, df_first))
}
