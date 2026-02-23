# Purpose: Panel A (all participants) logit regression models for unified selling table
# Author: Claude Code
# Date: 2026-02-09

# Libraries loaded by main script (do not load here)

# =====
# Panel A: All Participants (Logit)
# =====
run_logit_panel_a <- function(df) {
  cat("Panel A (logit) sample size:", nrow(df), "\n")

  m1 <- run_logit_panel_a_m1(df)
  m2 <- run_logit_panel_a_m2(df)
  m3 <- run_logit_panel_a_m3(df)

  cat("\n--- Panel A Logit Model Summaries ---\n")
  cat("\nModel 1 (Cascade RE glmer):\n")
  print(summary(m1))
  cat("\nModel 2 (Cascade + Emotions, RE glmer):\n")
  print(summary(m2))
  cat("\nModel 3 (Cascade + Traits, RE glmer):\n")
  print(summary(m3))

  list(m1 = m1, m2 = m2, m3 = m3)
}

# =====
# Model 1: Cascade Random Effects (logit)
# =====
run_logit_panel_a_m1 <- function(df) {
  cat("[Panel A M1] Cascade RE (glmer)...\n")
  glmer(
    sold ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
      int_1_1 + int_2_1 + int_2_2 + int_3_1 + int_3_2 + int_3_3 +
      signal + period + round + segment + treatment +
      (1 | player_id),
    family = binomial,
    data = df,
    control = glmerControl(
      optimizer = "bobyqa",
      optCtrl = list(maxfun = 100000)
    )
  )
}

# =====
# Model 2: Cascade + Emotions RE (logit)
# =====
run_logit_panel_a_m2 <- function(df) {
  cat("[Panel A M2] Cascade + Emotions RE (glmer)...\n")
  glmer(
    sold ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
      int_1_1 + int_2_1 + int_2_2 + int_3_1 + int_3_2 + int_3_3 +
      fear_mean + anger_mean + contempt_mean + disgust_mean +
      joy_mean + sadness_mean + surprise_mean + engagement_mean +
      valence_mean +
      signal + period + round + segment + treatment +
      age + gender_female +
      (1 | player_id),
    family = binomial,
    data = df,
    control = glmerControl(
      optimizer = "bobyqa",
      optCtrl = list(maxfun = 100000)
    )
  )
}

# =====
# Model 3: Cascade + Traits RE (logit)
# =====
run_logit_panel_a_m3 <- function(df) {
  cat("[Panel A M3] Cascade + Traits RE (glmer)...\n")
  glmer(
    sold ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
      int_1_1 + int_2_1 + int_2_2 + int_3_1 + int_3_2 + int_3_3 +
      state_anxiety + impulsivity + conscientiousness +
      extraversion + agreeableness + neuroticism + openness +
      risk_tolerance + age + gender_female +
      signal + period + round + segment + treatment +
      (1 | player_id),
    family = binomial,
    data = df,
    control = glmerControl(
      optimizer = "bobyqa",
      optCtrl = list(maxfun = 100000)
    )
  )
}
