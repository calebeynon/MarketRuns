# Purpose: Panel A (all participants) regression models for unified selling table
# Author: Claude Code
# Date: 2026-02-06

# Libraries loaded by main script (do not load here)

# =====
# Panel A: All Participants
# =====
run_panel_a <- function(df) {
  df_em <- df[complete.cases(df[, .SD, .SDcols = ALL_EMOTIONS])]
  cat("Panel A sample size:", nrow(df_em), "\n")

  m1 <- run_panel_a_m1(df_em)
  m2 <- run_panel_a_m2(df_em)
  m3 <- run_panel_a_m3(df_em)

  cat("\n--- Panel A Model Summaries ---\n")
  cat("\nModel 1 (Cascade RE):\n")
  print(summary(m1))
  cat("\nModel 2 (Cascade + Emotions, individual FE):\n")
  print(summary(m2))
  cat("\nModel 3 (Cascade + Traits, RE):\n")
  print(summary(m3))

  list(m1 = m1, m2 = m2, m3 = m3)
}

# =====
# Model 1: Cascade Random Effects
# =====
run_panel_a_m1 <- function(df) {
  cat("[Panel A M1] Cascade RE...\n")
  pdata <- pdata.frame(
    as.data.frame(df),
    index = c("player_id", "time_id")
  )
  plm(
    sold ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
      signal + period + round + segment + treatment,
    data = pdata,
    model = "random"
  )
}

# =====
# Model 2: Cascade + Emotions with individual FE
# =====
run_panel_a_m2 <- function(df) {
  cat("[Panel A M2] Cascade + Emotions with individual FE...\n")
  feols(
    sold ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
      fear_mean + anger_mean + contempt_mean + disgust_mean +
      joy_mean + sadness_mean + surprise_mean + engagement_mean +
      valence_mean +
      signal + period + round + segment | player_id,
    cluster = ~global_group_id,
    data = df
  )
}

# =====
# Model 3: Cascade + Traits RE
# =====
run_panel_a_m3 <- function(df) {
  cat("[Panel A M3] Cascade + Traits RE...\n")
  pdata <- pdata.frame(
    as.data.frame(df),
    index = c("player_id", "time_id")
  )
  plm(
    sold ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
      state_anxiety + impulsivity + conscientiousness +
      extraversion + agreeableness + neuroticism + openness +
      age + gender_female +
      signal + period + round + segment + treatment,
    data = pdata,
    model = "random"
  )
}
