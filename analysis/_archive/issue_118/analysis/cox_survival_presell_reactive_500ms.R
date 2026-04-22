# Purpose: 500ms Cox survival regression on reactive sellers — paper table (issue #118)
# Author: Claude Code
# Date: 2026-04-21

library(data.table)
library(survival)
library(coxme)

# DO NOT REORDER. `cox_survival_regression.R` defines its own OUTPUT_PATH
# global pointing at the main Cox table .tex. Our FILE PATHS block below
# intentionally shadows it so write_table() targets the reactive-500ms .tex.
# Reordering (or invoking the sourced file's main()) will silently clobber
# the main Cox table.
source("analysis/analysis/selling_regression_helpers.R")
source("analysis/analysis/unified_selling_regression_panel_c.R")
source("analysis/analysis/cox_survival_regression.R")
source("analysis/analysis/cox_survival_panel_a.R")
source("analysis/analysis/cox_presell_merge_helpers.R")
source("analysis/analysis/cox_presell_reactive_helpers.R")
source("analysis/analysis/cox_reactive_500ms_table_helpers.R")

# FILE PATHS
BASE_PATH <- "datastore/derived/emotions_traits_selling_dataset.csv"
PRESELL_PATH <- "datastore/derived/presell_emotions_traits_dataset.csv"
OUTPUT_PATH <- "analysis/output/tables/cox_survival_reactive_500ms.tex"
WINDOW <- 500L

# =====
# Main function
# =====
main <- function() {
  df_r <- build_reactive_sample()
  re_cox <- run_cox_panel_r_no_traits(df_r)
  cluster_cox <- run_cox_panel_r_cluster(df_r)
  print_model_summaries(re_cox, cluster_cox)
  print_ph_diagnostic(cluster_cox)
  lines <- build_cox_table_2col(re_cox, cluster_cox)
  write_table(lines, OUTPUT_PATH)
  cat("Done.\n")
}

# =====
# Build the reactive-sellers sample at the 500ms window
# =====
build_reactive_sample <- function() {
  base_dt <- prepare_base_data(BASE_PATH)
  presell_dt <- load_presell_dataset(PRESELL_PATH)
  cat(sprintf("Base rows: %d | Presell rows: %d\n",
              nrow(base_dt), nrow(presell_dt)))
  merged <- merge_presell_window(base_dt, presell_dt, WINDOW)
  dropped <- drop_missing_window_rows(merged$dt, WINDOW, merged$frames_col)
  df_em <- dropped$dt[complete.cases(dropped$dt[, .SD,
                                                 .SDcols = c(ALL_EMOTIONS,
                                                             ALL_TRAITS)])]
  cat(sprintf("[window=%dms] After complete.cases: %d rows (from %d)\n",
              WINDOW, nrow(df_em), nrow(dropped$dt)))
  df_r <- add_reactive_flag(df_em)
  cat(sprintf("[window=%dms] Reactive sales: %d of %d sold==1\n",
              WINDOW, sum(df_r$reactive_sale == 1, na.rm = TRUE),
              sum(df_r$sold == 1, na.rm = TRUE)))
  df_r
}

# =====
# Print summaries for the two fitted models
# =====
print_model_summaries <- function(re_cox, cluster_cox) {
  cat("\n--- Panel R (500ms) — No Traits RE Cox (coxme) ---\n")
  print(summary(re_cox))
  cat("\n--- Panel R (500ms) — Cluster-robust Cox (coxph) ---\n")
  print(summary(cluster_cox))
}

# =====
# Proportional-hazards diagnostic on the coxph (cluster) model
# =====
print_ph_diagnostic <- function(cluster_cox) {
  cat("\n--- cox.zph diagnostic (Schoenfeld residuals, coxph cluster) ---\n")
  zph <- cox.zph(cluster_cox)
  print(zph$table)
  invisible(zph)
}

# %%
if (sys.nframe() == 0L) main()
