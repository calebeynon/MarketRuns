# Purpose: Panel B (second sellers) regression models for unified selling table
# Author: Claude Code
# Date: 2026-02-06

# Libraries loaded by main script (do not load here)

# =====
# Panel B: Second Sellers
# =====
run_panel_b <- function(df, df_full) {
  cat("Panel B: Identifying second sellers...\n")
  df_second <- identify_second_sellers(df)
  cat("  Second seller observations:", nrow(df_second), "\n")

  cat("  Creating dummy_prev_period...\n")
  df_second <- create_prev_period_dummy(df_second, df_full)

  cat("  Running Model 1 (Cascade RE)...\n")
  m1 <- run_panel_b_m1(df_second)

  cat("  Running Model 2 (Cascade + Emotions FE)...\n")
  m2 <- run_panel_b_m2(df_second)

  cat("  Running Model 3 (Cascade + Traits RE)...\n")
  m3 <- run_panel_b_m3(df_second)

  cat("Panel B complete.\n")
  list(m1 = m1, m2 = m2, m3 = m3)
}

# =====
# Data subsetting
# =====
identify_second_sellers <- function(df) {
  second_ids <- df[prior_group_sales == 1 & sold == 1,
                   unique(player_group_round_id)]
  df[player_group_round_id %in% second_ids]
}

create_prev_period_dummy <- function(df_second, df_full) {
  first_sales <- df_full[prior_group_sales == 0 & sold == 1,
                         .(first_sale_period = min(period)),
                         by = group_round_id]
  df_second <- merge(df_second, first_sales,
                     by = "group_round_id", all.x = TRUE)
  df_second[, dummy_prev_period := as.integer(
    first_sale_period == (period - 1)
  )]
  df_second[, first_sale_period := NULL]
  df_second
}

# =====
# Model specifications
# =====
run_panel_b_m1 <- function(df_second) {
  pdata <- pdata.frame(as.data.frame(df_second),
                       index = c("player_id", "time_id"))
  plm(sold ~ dummy_prev_period +
      signal + period + round + segment + treatment,
      data = pdata, model = "random")
}

run_panel_b_m2 <- function(df_second) {
  feols(sold ~ dummy_prev_period +
        fear_mean + anger_mean + contempt_mean + disgust_mean +
        joy_mean + sadness_mean + surprise_mean +
        engagement_mean + valence_mean +
        signal + period + round + segment | player_id,
        cluster = ~global_group_id, data = df_second)
}

run_panel_b_m3 <- function(df_second) {
  pdata <- pdata.frame(as.data.frame(df_second),
                       index = c("player_id", "time_id"))
  plm(sold ~ dummy_prev_period +
      state_anxiety + impulsivity + conscientiousness +
      extraversion + agreeableness + neuroticism + openness +
      age + gender_female +
      signal + period + round + segment + treatment,
      data = pdata, model = "random")
}
