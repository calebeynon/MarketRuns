# Purpose: Cox survival regression for reactive sellers with pre-sell emotion windows (issue #118)
# Author: Claude Code
# Date: 2026-04-21

library(data.table)
library(survival)
library(coxme)

# FILE PATHS
BASE_PATH <- "datastore/derived/emotions_traits_selling_dataset.csv"
PRESELL_PATH <- "datastore/derived/presell_emotions_traits_dataset.csv"

# Source dependencies (order matters: helpers -> panels -> merge helpers)
source("analysis/analysis/selling_regression_helpers.R")
source("analysis/analysis/unified_selling_regression_panel_c.R")
source("analysis/analysis/cox_survival_regression.R")
source("analysis/analysis/cox_survival_panel_a.R")
source("analysis/analysis/cox_presell_merge_helpers.R")
source("analysis/analysis/cox_presell_reactive_helpers.R")

# =====
# Main function
# =====
main <- function() {
  base_dt <- prepare_base_data(BASE_PATH)
  presell_dt <- load_presell_dataset(PRESELL_PATH)
  cat("Base rows:", nrow(base_dt),
      "| Presell rows:", nrow(presell_dt), "\n")

  results <- run_all_windows(base_dt, presell_dt)
  print_all_comparisons(results)
  cat("\nDone.\n")
}

# =====
# Run both Panel R models for each window
# =====
run_all_windows <- function(base_dt, presell_dt) {
  slots <- c("panel_r_no_traits", "panel_r_with_traits")
  results <- setNames(lapply(slots, function(s) list()), slots)
  for (w in WINDOWS) {
    fits <- run_one_window(base_dt, presell_dt, w)
    for (s in slots) results[[s]][[as.character(w)]] <- fits[[s]]
  }
  results
}

# =====
# Run Panel R on one window's merged data
# =====
# Note: add_reactive_flag is called AFTER the emotion/trait complete.cases
# filter so the T-1 group-sale count is computed on the same sample used for
# Cox estimation. This drops ~6 reactive events vs. flagging pre-filter
# (107 base -> ~101 per window) but keeps the flag internally consistent
# with the fit sample.
run_one_window <- function(base_dt, presell_dt, window) {
  cat(sprintf("\n>>> Window = %dms <<<\n", window))
  merged <- merge_presell_window(base_dt, presell_dt, window)
  dropped <- drop_missing_window_rows(merged$dt, window, merged$frames_col)
  df_em <- dropped$dt[complete.cases(dropped$dt[, .SD,
                                                 .SDcols = c(ALL_EMOTIONS,
                                                             ALL_TRAITS)])]
  cat(sprintf("[window=%dms] After complete.cases: %d rows (from %d)\n",
              window, nrow(df_em), nrow(dropped$dt)))
  df_r <- add_reactive_flag(df_em)
  cat(sprintf("[window=%dms] Reactive sales: %d of %d sold==1\n",
              window, sum(df_r$reactive_sale == 1, na.rm = TRUE),
              sum(df_r$sold == 1, na.rm = TRUE)))
  panel_r <- run_cox_panel_r(df_r)
  list(panel_r_no_traits = panel_r$no_traits,
       panel_r_with_traits = panel_r$with_traits)
}

# =====
# Print side-by-side comparisons for both model slots
# =====
print_all_comparisons <- function(results) {
  labels <- c(panel_r_no_traits = "Panel R — No Traits (Reactive)",
              panel_r_with_traits = "Panel R — With Traits (Reactive)")
  for (slot in names(labels)) {
    models_list <- lapply(as.character(WINDOWS),
                          function(w) results[[slot]][[w]])
    coefs <- collect_window_coefs(models_list, WINDOWS)
    fits <- collect_window_fits(models_list, WINDOWS)
    print_window_comparison(coefs, fits, labels[[slot]])
  }
}

# %%
if (sys.nframe() == 0L) main()
