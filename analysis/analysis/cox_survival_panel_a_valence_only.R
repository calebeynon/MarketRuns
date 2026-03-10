# Purpose: Panel A (all sellers) Cox survival regression — valence only
# Author: Claude Code
# Date: 2026-03-09

# Libraries loaded by main script (do not load here)

# =====
# Panel A: All Sellers — 2-model no-traits/with-traits split (valence only)
# =====
run_cox_panel_a <- function(df) {
  cat("Panel A (Cox, valence only) sample size:", nrow(df), "\n")

  m1 <- run_cox_panel_a_no_traits(df)
  m2 <- run_cox_panel_a_with_traits(df)

  cat("\n--- Panel A Cox Valence-Only Model Summaries ---\n")
  cat("\nModel 1 (No Traits RE Cox):\n")
  print(summary(m1))
  cat("\nModel 2 (With Traits RE Cox):\n")
  print(summary(m2))

  list(no_traits = m1, with_traits = m2)
}

# =====
# Model 1: No traits RE Cox (valence only, no personality traits)
# =====
run_cox_panel_a_no_traits <- function(df) {
  cat("[Panel A M1] No traits RE Cox, valence only (coxme)...\n")
  f <- Surv(period_start, period, sold) ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
    int_1_1 + int_2_1 + int_2_2 + int_3_1 + int_3_2 + int_3_3 +
    valence_mean +
    signal + round + segment + treatment +
    age + gender_female +
    (1 | player_id)
  coxme(f, data = df, init = get_coxph_init(f, df))
}

# =====
# Model 2: With traits RE Cox (valence + personality traits)
# =====
run_cox_panel_a_with_traits <- function(df) {
  cat("[Panel A M2] With traits RE Cox, valence only (coxme)...\n")
  f <- Surv(period_start, period, sold) ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
    int_1_1 + int_2_1 + int_2_2 + int_3_1 + int_3_2 + int_3_3 +
    valence_mean +
    state_anxiety + impulsivity + risk_tolerance +
    conscientiousness + extraversion + agreeableness + neuroticism + openness +
    signal + round + segment + treatment +
    age + gender_female +
    (1 | player_id)
  coxme(f, data = df, init = get_coxph_init(f, df))
}
