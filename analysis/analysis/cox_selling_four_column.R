# Purpose: 4-column Cox survival table of selling behavior across risk sets (issue #124)
# Author: Claude Code
# Date: 2026-05-20

library(data.table)
library(survival)
library(coxme)

# DO NOT REORDER. cox_survival_regression.R defines OUTPUT_PATH pointing at the
# main Cox table .tex; the FILE PATHS block below intentionally shadows it (the
# shadowing gotcha is documented in cox_survival_normal_vs_reactive.R:9). Do not
# call any sourced file's main() from here.
source("analysis/analysis/selling_regression_helpers.R")
source("analysis/analysis/unified_selling_regression_panel_c.R")
source("analysis/analysis/cox_survival_regression.R")
source("analysis/analysis/cox_survival_panel_a.R")
source("analysis/analysis/cox_survival_panel_b.R")
source("analysis/analysis/cox_presell_merge_helpers.R")
source("analysis/analysis/cox_presell_reactive_helpers.R")
source("analysis/analysis/cox_reactive_500ms_table_helpers.R")
source("analysis/analysis/cox_normal_vs_reactive_helpers.R")
source("analysis/analysis/cox_selling_four_column_helpers.R")

# FILE PATHS (shadow the sourced OUTPUT_PATH after sourcing)
BASE_PATH <- "datastore/derived/emotions_traits_selling_dataset.csv"
PRESELL_PATH <- "datastore/derived/presell_emotions_traits_dataset.csv"
OUTPUT_PATH <- "analysis/output/tables/cox_selling_four_column.tex"
WINDOW <- 500L

# =====
# Main function: fit four risk-set models, assemble flat 4-column table
# =====
main <- function() {
  df_em <- load_emotion_complete_base()

  m1 <- run_cox_panel_b(df_em)$with_traits
  m2 <- fit_reactive_with_traits()
  m3 <- fit_all_sellers_with_traits(df_em)
  m4 <- run_cox_panel_a(df_em)$with_traits

  models <- list(
    coxme_model(m1), coxph_cluster_model(m2),
    coxme_model(m3), coxme_model(m4))
  lines <- build_cox_table_four_column(models)
  write_table(lines, OUTPUT_PATH)
  cat("Done.\n")
}

# =====
# Shared emotion+trait-complete base (Columns 1, 3, 4 all start here)
# =====
load_emotion_complete_base <- function() {
  df <- prepare_base_data(BASE_PATH)
  df_em <- df[complete.cases(df[, .SD,
                                .SDcols = c(ALL_EMOTIONS, ALL_TRAITS)])]
  cat("Base:", nrow(df), "| Emotion+trait-complete:", nrow(df_em), "\n")
  df_em
}

# =====
# Column (2): reactive-500ms clustered coxph, with-traits spec. Reuses the
# existing clustered estimator (run_cox_panel_r_with_traits) — NOT refit as
# coxme. nclusters is attached for the Participants row's "---" handling.
# =====
fit_reactive_with_traits <- function() {
  df_r <- build_reactive_500ms_sample()
  m <- run_cox_panel_r_with_traits(df_r)
  restricted <- restrict_to_reactive_risk_set(df_r)
  attr(m, "nclusters") <- uniqueN(restricted$global_group_id)
  m
}

# =====
# Build the 500ms reactive-sellers sample (mirrors the normal-vs-reactive main
# script; inlined here to avoid sourcing that script's main()). Overwrites
# sold==1 emotions with the 500ms pre-click window, drops sold rows lacking
# window frames, then adds the reactive-sale flags.
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

# =====
# Column (3): NEW all-sellers-only coxme. Restrict the base to player-group-
# rounds that contain at least one sale, then fit the Panel A with-traits
# spec. This isolates the timing of sales among groups where selling occurred,
# distinct from Panel A's full risk set (Column 4).
# =====
fit_all_sellers_with_traits <- function(df_em) {
  seller_ids <- df_em[sold == 1, unique(player_group_round_id)]
  df_sellers <- df_em[player_group_round_id %in% seller_ids]
  cat("All-sellers player-group-rounds:", length(seller_ids),
      "| sample rows:", nrow(df_sellers),
      "| events:", sum(df_sellers$sold == 1, na.rm = TRUE), "\n")
  f <- all_sellers_formula(df_sellers)
  coxme(f, data = df_sellers, init = get_coxph_init(f, df_sellers))
}

# =====
# Panel A with-traits spec, but drop cascade/interaction terms that are
# constant within the restricted all-sellers sample (e.g. int_3_2 == 0).
# A constant covariate aborts coxme; dropping it lets the term render blank.
# =====
all_sellers_formula <- function(df) {
  cascade <- c("dummy_1_cum", "dummy_2_cum", "dummy_3_cum")
  candidates <- c(cascade, INTERACTION_VARS)
  keep <- candidates[sapply(candidates, function(v) uniqueN(df[[v]]) > 1)]
  rhs <- paste(c(keep, DISCRETE_EMOTIONS, "valence_mean", COX_TRAITS,
                 "signal", "round", "segment", "treatment",
                 "age", "gender_female", "(1 | player_id)"),
               collapse = " + ")
  as.formula(paste("Surv(period_start, period, sold) ~", rhs))
}

# =====
# Model wrappers pairing each fit with its coefficient/fit extractors
# =====
coxme_model <- function(fit) {
  list(fit = fit, coef_fn = extract_cox_coefs, fit_fn = extract_cox_fit)
}

coxph_cluster_model <- function(fit) {
  list(fit = fit, coef_fn = extract_coxph_coefs,
       fit_fn = extract_coxph_fit_nvr)
}

# %%
if (sys.nframe() == 0L) main()
