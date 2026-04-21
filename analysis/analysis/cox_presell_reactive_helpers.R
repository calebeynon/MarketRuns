# Purpose: Helpers for Panel R (reactive sellers) Cox survival regression (issue #118)
# Author: Claude Code
# Date: 2026-04-21

# Libraries loaded by main script (do not load here)

# =====
# Add group_sold_prev_period and reactive_sale flags
# =====
# reactive_sale == 1 iff this row is a sale (sold==1) AND any group-mate
# (including self) sold in period T-1. We first sum `sold` by
# (session_id, segment, round, group_id, period), then shift by one period.
# Period 1 has no T-1, so group_sold_prev_period is 0 there.
#
# Precondition: caller must have applied already_sold == 0 upstream. Under
# that filter, a player's own T-1 sale cannot appear at row T (the T row is
# filtered out), so summing `sold` across the whole group (including self)
# is equivalent to "any OTHER group-mate sold at T-1".
add_reactive_flag <- function(df) {
  input_n <- nrow(df)
  stopifnot(all(df$already_sold == 0))
  group_keys <- c("session_id", "segment", "round", "group_id")
  sums <- compute_group_period_sold(df, group_keys)
  lookup <- build_prev_period_lookup(sums)
  df <- attach_prev_sold(df, lookup, group_keys)
  df[, group_sold_prev_period := as.integer(!is.na(group_sold_prev_period) &
                                              group_sold_prev_period > 0)]
  df[, reactive_sale := as.integer(sold) * group_sold_prev_period]
  stopifnot(nrow(df) == input_n)
  df
}

compute_group_period_sold <- function(df, group_keys) {
  by_cols <- c(group_keys, "period")
  df[, .(group_sold = sum(as.integer(sold), na.rm = TRUE)), by = by_cols]
}

build_prev_period_lookup <- function(sums) {
  # Shift period forward by 1 so a row with period==T joins to sums at T-1.
  lookup <- copy(sums)
  lookup[, period := period + 1L]
  setnames(lookup, "group_sold", "group_sold_prev_period")
  lookup
}

attach_prev_sold <- function(df, lookup, group_keys) {
  by_cols <- c(group_keys, "period")
  merge(df, lookup, by = by_cols, all.x = TRUE, sort = FALSE)
}

# =====
# Panel R: Reactive Sellers — no-traits / with-traits / cluster split
# =====
run_cox_panel_r <- function(df) {
  cat("Panel R (Cox) sample size:", nrow(df),
      "| reactive sales:", sum(df$reactive_sale == 1, na.rm = TRUE), "\n")

  m1 <- run_cox_panel_r_no_traits(df)
  m2 <- run_cox_panel_r_with_traits(df)
  m3 <- run_cox_panel_r_cluster(df)

  cat("\n--- Panel R Cox Model Summaries ---\n")
  cat("\nModel 1 (No Traits RE Cox, reactive_sale):\n")
  print(summary(m1))
  cat("\nModel 2 (With Traits RE Cox, reactive_sale):\n")
  print(summary(m2))
  cat("\nModel 3 (Cluster-robust Cox, reactive_sale):\n")
  print(summary(m3))

  list(no_traits = m1, with_traits = m2, cluster = m3)
}

# =====
# Model 1: No traits RE Cox, event = reactive_sale
# =====
# Cascade dummies (dummy_*_cum) and interactions (int_*_*) are omitted
# because reactive_sale==1 requires prior_group_sales>=1 by definition,
# making them mechanically collinear with the event.
run_cox_panel_r_no_traits <- function(df) {
  cat("[Panel R M1] No traits RE Cox (coxme)...\n")
  f <- Surv(period_start, period, reactive_sale) ~
    fear_mean + anger_mean + contempt_mean + disgust_mean +
    joy_mean + sadness_mean + surprise_mean + engagement_mean +
    valence_mean +
    signal + round + segment + treatment +
    age + gender_female +
    (1 | player_id)
  coxme(f, data = df, init = get_coxph_init(f, df))
}

# =====
# Model 2: With traits RE Cox, event = reactive_sale
# =====
run_cox_panel_r_with_traits <- function(df) {
  cat("[Panel R M2] With traits RE Cox (coxme)...\n")
  f <- Surv(period_start, period, reactive_sale) ~
    fear_mean + anger_mean + contempt_mean + disgust_mean +
    joy_mean + sadness_mean + surprise_mean + engagement_mean +
    valence_mean +
    state_anxiety + impulsivity + risk_tolerance +
    conscientiousness + extraversion + agreeableness + neuroticism + openness +
    signal + round + segment + treatment +
    age + gender_female +
    (1 | player_id)
  coxme(f, data = df, init = get_coxph_init(f, df))
}

# =====
# Model 3: Cluster-robust Cox (group_id sandwich SEs), event = reactive_sale
# =====
run_cox_panel_r_cluster <- function(df) {
  cat("[Panel R M3] Cluster-robust Cox (coxph, cluster=group_id)...\n")
  f <- Surv(period_start, period, reactive_sale) ~
    fear_mean + anger_mean + contempt_mean + disgust_mean +
    joy_mean + sadness_mean + surprise_mean + engagement_mean +
    valence_mean +
    signal + round + segment + treatment +
    age + gender_female
  coxph(f, data = df, cluster = df$group_id)
}
