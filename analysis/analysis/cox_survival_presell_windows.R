# Purpose: Cox survival regression with pre-click emotion windows (issue #118)
# Author: Claude Code
# Date: 2026-04-20

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
source("analysis/analysis/cox_survival_panel_b.R")
source("analysis/analysis/cox_presell_merge_helpers.R")

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
# Run all 4 models for each window
# =====
run_all_windows <- function(base_dt, presell_dt) {
  slots <- c("panel_a_no_traits", "panel_a_with_traits",
             "panel_b_no_traits", "panel_b_with_traits")
  results <- setNames(lapply(slots, function(s) list()), slots)
  for (w in WINDOWS) {
    fits <- run_one_window(base_dt, presell_dt, w)
    for (s in slots) results[[s]][[as.character(w)]] <- fits[[s]]
  }
  results
}

# =====
# Run the 4 Cox models on one window's merged data
# =====
run_one_window <- function(base_dt, presell_dt, window) {
  cat(sprintf("\n>>> Window = %dms <<<\n", window))
  merged <- merge_presell_window(base_dt, presell_dt, window)
  dropped <- drop_missing_window_rows(merged$dt, window, merged$frames_col)
  df_em <- dropped$dt[complete.cases(dropped$dt[, .SD,
                                                 .SDcols = c(ALL_EMOTIONS,
                                                             ALL_TRAITS)])]
  cat(sprintf("[window=%dms] After complete.cases: %d rows (from %d)\n",
              window, nrow(df_em), nrow(dropped$dt)))
  panel_a <- run_cox_panel_a(df_em)
  panel_b <- run_cox_panel_b(df_em)
  list(panel_a_no_traits = panel_a$no_traits,
       panel_a_with_traits = panel_a$with_traits,
       panel_b_no_traits = panel_b$no_traits,
       panel_b_with_traits = panel_b$with_traits)
}

# =====
# Print side-by-side comparisons for all 4 model slots
# =====
print_all_comparisons <- function(results) {
  labels <- c(panel_a_no_traits = "Panel A — No Traits",
              panel_a_with_traits = "Panel A — With Traits",
              panel_b_no_traits = "Panel B — No Traits",
              panel_b_with_traits = "Panel B — With Traits")
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
