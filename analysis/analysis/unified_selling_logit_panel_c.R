# Purpose: Panel C (first sellers) logit regression models for unified selling table
# Author: Claude Code
# Date: 2026-02-09

# Libraries loaded by main script (do not load here)
# subset_first_sellers() is defined in unified_selling_regression_panel_c.R

# =====
# Panel C: First Sellers (Logit)
# =====
run_logit_panel_c <- function(df) {
  cat("Logit Panel C: Identifying first sellers...\n")
  df_first <- subset_first_sellers(df)

  cat("  [1/3] Model 1: Controls only (RE logit)...\n")
  m1 <- run_logit_panel_c_m1(df_first)

  cat("  [2/3] Model 2: Emotions (individual FE logit)...\n")
  m2 <- run_logit_panel_c_m2(df_first)

  cat("  [3/3] Model 3: Traits (RE logit)...\n")
  m3 <- run_logit_panel_c_m3(df_first)

  cat("  Logit Panel C complete.\n")
  list(m1 = m1, m2 = m2, m3 = m3)
}

# =====
# Panel C logit model specifications
# =====
run_logit_panel_c_m1 <- function(df_first) {
  glmer(
    sold ~ signal + period + round + segment + treatment + (1 | player_id),
    family = binomial,
    data = df_first,
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
  )
}

run_logit_panel_c_m2 <- function(df_first) {
  feglm(
    sold ~ fear_mean + anger_mean + contempt_mean + disgust_mean +
      joy_mean + sadness_mean + surprise_mean + engagement_mean +
      valence_mean + signal + period + round + segment | player_id,
    family = binomial,
    cluster = ~global_group_id,
    data = df_first
  )
}

run_logit_panel_c_m3 <- function(df_first) {
  glmer(
    sold ~ state_anxiety + impulsivity + conscientiousness + extraversion +
      agreeableness + neuroticism + openness + age + gender_female +
      signal + period + round + segment + treatment + (1 | player_id),
    family = binomial,
    data = df_first,
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
  )
}
