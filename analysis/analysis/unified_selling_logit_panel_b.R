# Purpose: Panel B (second sellers) logit regression models for unified selling table
# Author: Claude Code
# Date: 2026-02-09

# Libraries loaded by main script (do not load here)
# Reuses identify_second_sellers() and create_prev_period_dummy() from LPM panel B

# =====
# Panel B: Second Sellers (Logit)
# =====
run_logit_panel_b <- function(df, df_full) {
  cat("Logit Panel B: Identifying second sellers...\n")
  df_second <- identify_second_sellers(df)
  cat("  Second seller observations:", nrow(df_second), "\n")

  cat("  Creating dummy_prev_period...\n")
  df_second <- create_prev_period_dummy(df_second, df_full)

  cat("  Running Model 1 (Cascade RE logit)...\n")
  m1 <- run_logit_panel_b_m1(df_second)

  cat("  Running Model 2 (Cascade + Emotions FE logit)...\n")
  m2 <- run_logit_panel_b_m2(df_second)

  cat("  Running Model 3 (Cascade + Traits RE logit)...\n")
  m3 <- run_logit_panel_b_m3(df_second)

  cat("Logit Panel B complete.\n")
  list(m1 = m1, m2 = m2, m3 = m3)
}

# =====
# Model specifications
# =====
run_logit_panel_b_m1 <- function(df_second) {
  glmer(sold ~ dummy_prev_period +
        signal + period + round + segment + treatment +
        (1 | player_id),
        family = binomial, data = df_second,
        control = glmerControl(optimizer = "bobyqa",
                               optCtrl = list(maxfun = 100000)))
}

run_logit_panel_b_m2 <- function(df_second) {
  feglm(sold ~ dummy_prev_period +
        fear_mean + anger_mean + contempt_mean + disgust_mean +
        joy_mean + sadness_mean + surprise_mean +
        engagement_mean + valence_mean +
        signal + period + round + segment | player_id,
        family = binomial, cluster = ~global_group_id,
        data = df_second)
}

run_logit_panel_b_m3 <- function(df_second) {
  glmer(sold ~ dummy_prev_period +
        state_anxiety + impulsivity + conscientiousness +
        extraversion + agreeableness + neuroticism + openness +
        age + gender_female +
        signal + period + round + segment + treatment +
        (1 | player_id),
        family = binomial, data = df_second,
        control = glmerControl(optimizer = "bobyqa",
                               optCtrl = list(maxfun = 100000)))
}
