# Purpose: Panel B (first sellers) Cox survival regression — valence only
# Author: Claude Code
# Date: 2026-03-09

# Libraries loaded by main script (do not load here)
# subset_first_sellers() is defined in unified_selling_regression_panel_c.R

# =====
# Panel B: First Sellers (Cox, valence only)
# =====
run_cox_panel_b <- function(df) {
  cat("Cox Panel B (valence only): Identifying first sellers...\n")
  df_first <- subset_first_sellers(df)

  cat("  [1/2] No traits model (RE Cox)...\n")
  no_traits <- run_cox_panel_b_no_traits(df_first)

  cat("  [2/2] With traits model (RE Cox)...\n")
  with_traits <- run_cox_panel_b_with_traits(df_first)

  cat("  Cox Panel B (valence only) complete.\n")
  list(no_traits = no_traits, with_traits = with_traits)
}

# =====
# No traits: Valence only, no personality traits
# =====
run_cox_panel_b_no_traits <- function(df_first) {
  f <- Surv(period_start, period, sold) ~ valence_mean +
    signal + round + segment + treatment +
    age + gender_female +
    (1 | player_id)
  coxme(f, data = df_first, init = get_coxph_init(f, df_first))
}

# =====
# With traits: Valence + personality traits
# =====
run_cox_panel_b_with_traits <- function(df_first) {
  f <- Surv(period_start, period, sold) ~ valence_mean +
    state_anxiety + impulsivity + risk_tolerance +
    conscientiousness + extraversion + agreeableness + neuroticism + openness +
    signal + round + segment + treatment +
    age + gender_female +
    (1 | player_id)
  coxme(f, data = df_first, init = get_coxph_init(f, df_first))
}
