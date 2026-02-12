# Purpose: Panel C (first sellers) regression models for unified selling table
# Author: Claude Code
# Date: 2026-02-06

# Libraries loaded by main script (do not load here)

# =====
# Panel C: First Sellers
# =====
run_panel_c <- function(df) {
  cat("Panel C: Identifying first sellers...\n")
  df_first <- subset_first_sellers(df)

  pdata <- pdata.frame(as.data.frame(df_first), index = c("player_id", "time_id"))

  cat("  [1/3] Model 1: Controls only (RE)...\n")
  m1 <- run_panel_c_m1(pdata)

  cat("  [2/3] Model 2: Emotions (individual FE)...\n")
  m2 <- run_panel_c_m2(df_first)

  cat("  [3/3] Model 3: Traits (RE)...\n")
  m3 <- run_panel_c_m3(pdata)

  cat("  Panel C complete.\n")
  list(m1 = m1, m2 = m2, m3 = m3)
}

# =====
# First seller subsetting
# =====
subset_first_sellers <- function(df) {
  first_ids <- df[prior_group_sales == 0 & sold == 1, unique(player_group_round_id)]
  df_first <- df[player_group_round_id %in% first_ids]
  cat("  First seller player-group-rounds:", length(first_ids), "\n")
  cat("  First sellers sample size:", nrow(df_first), "\n")
  df_first
}

# =====
# Panel C model specifications
# =====
run_panel_c_m1 <- function(pdata) {
  plm(
    sold ~ signal + period + round + segment + treatment,
    data = pdata,
    model = "random"
  )
}

run_panel_c_m2 <- function(df_first) {
  feols(
    sold ~ fear_mean + anger_mean + contempt_mean + disgust_mean +
      joy_mean + sadness_mean + surprise_mean + engagement_mean +
      valence_mean + signal + period + round + segment | player_id,
    cluster = ~global_group_id,
    data = df_first
  )
}

run_panel_c_m3 <- function(pdata) {
  plm(
    sold ~ state_anxiety + impulsivity + conscientiousness + extraversion +
      agreeableness + neuroticism + openness + age + gender_female +
      signal + period + round + segment + treatment,
    data = pdata,
    model = "random"
  )
}
