# Purpose: Normal vs Reactive-500ms Cox side-by-side table (issue #118)
# Author: Claude Code
# Date: 2026-04-21

library(data.table)
library(survival)
library(coxme)

# DO NOT REORDER. `cox_survival_regression.R` defines OUTPUT_PATH pointing at
# the main Cox table .tex. Our FILE PATHS block below intentionally shadows
# it so write_table() writes the normal-vs-reactive .tex. Do not call the
# sourced file's main() from here.
source("analysis/analysis/selling_regression_helpers.R")
source("analysis/analysis/unified_selling_regression_panel_c.R")
source("analysis/analysis/cox_survival_regression.R")
source("analysis/analysis/cox_survival_panel_a.R")
source("analysis/analysis/cox_presell_merge_helpers.R")
source("analysis/analysis/cox_presell_reactive_helpers.R")
source("analysis/analysis/cox_reactive_500ms_table_helpers.R")
source("analysis/analysis/cox_normal_vs_reactive_helpers.R")

# FILE PATHS
BASE_PATH <- "datastore/derived/emotions_traits_selling_dataset.csv"
PRESELL_PATH <- "datastore/derived/presell_emotions_traits_dataset.csv"
OUTPUT_PATH <- "analysis/output/tables/cox_survival_normal_vs_reactive.tex"
WINDOW <- 500L

# =====
# Main function
# =====
main <- function() {
  normal_model <- fit_normal_cox_cluster()
  reactive_model <- fit_reactive_cox_cluster()
  cat("\n--- Normal clustered Cox summary ---\n")
  print(summary(normal_model))
  cat("\n--- Reactive-500ms clustered Cox summary ---\n")
  print(summary(reactive_model))
  lines <- build_cox_table_normal_vs_reactive(normal_model, reactive_model)
  write_table(lines, OUTPUT_PATH)
  cat("Done.\n")
}

# =====
# Column (1): clustered coxph on the full base dataset (event = sold).
# Cluster = global_group_id (session x segment x group_id = 96 clusters).
# =====
fit_normal_cox_cluster <- function() {
  df <- prepare_base_data(BASE_PATH)
  df_em <- df[complete.cases(df[, .SD,
                                 .SDcols = c(ALL_EMOTIONS, ALL_TRAITS)])]
  nclusters <- uniqueN(df_em$global_group_id)
  cat(sprintf("[Normal] Base rows: %d | Complete: %d | Clusters: %d\n",
              nrow(df), nrow(df_em), nclusters))
  m <- coxph(normal_cox_formula(), data = df_em,
             cluster = global_group_id)
  attr(m, "nclusters") <- nclusters
  m
}

normal_cox_formula <- function() {
  Surv(period_start, period, sold) ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
    int_1_1 + int_2_1 + int_2_2 + int_3_1 + int_3_2 + int_3_3 +
    fear_mean + anger_mean + contempt_mean + disgust_mean +
    joy_mean + sadness_mean + surprise_mean + engagement_mean +
    valence_mean +
    signal + round + segment + treatment +
    age + gender_female
}

# =====
# Column (2): clustered coxph on 500ms-filtered reactive sample.
# =====
fit_reactive_cox_cluster <- function() {
  df_r <- build_reactive_500ms_sample()
  nclusters <- uniqueN(df_r$global_group_id)
  cat(sprintf("[Reactive] Clusters: %d\n", nclusters))
  m <- coxph(reactive_cox_formula(), data = df_r,
             cluster = global_group_id)
  attr(m, "nclusters") <- nclusters
  m
}

reactive_cox_formula <- function() {
  Surv(period_start, period, reactive_sale) ~
    fear_mean + anger_mean + contempt_mean + disgust_mean +
    joy_mean + sadness_mean + surprise_mean + engagement_mean +
    valence_mean +
    signal + round + segment + treatment +
    age + gender_female
}

# =====
# Build the 500ms reactive-sellers sample used by Column (2)
# =====
build_reactive_500ms_sample <- function() {
  base_dt <- prepare_base_data(BASE_PATH)
  presell_dt <- load_presell_dataset(PRESELL_PATH)
  merged <- merge_presell_window(base_dt, presell_dt, WINDOW)
  dropped <- drop_missing_window_rows(merged$dt, WINDOW, merged$frames_col)
  df_em <- dropped$dt[complete.cases(dropped$dt[, .SD,
                                                 .SDcols = c(ALL_EMOTIONS,
                                                             ALL_TRAITS)])]
  df_r <- add_reactive_flag(df_em)
  cat(sprintf("[Reactive 500ms] Rows: %d | Reactive sales: %d\n",
              nrow(df_r), sum(df_r$reactive_sale == 1, na.rm = TRUE)))
  df_r
}

# %%
if (sys.nframe() == 0L) main()
