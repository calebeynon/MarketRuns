# Purpose: Pooled OLS of first-sale behavior (belief and timing) by treatment
# Author: Caleb Eynon w/ Claude Code
# Date: 2026-05-20
# nolint start
#
# Among group-rounds where a sale occurred, regress two first-sale outcomes on
# treatment, segment, and round, with and without controlling for the bad state:
#   Belief at first sale  = signal_at_first_sale
#   Period of first sale  = first_sale_period
# bad_state is merged from group_round_timing.csv (bad_state = 1 when state == 0).
# Standard errors are clustered on global_group_id.

library(data.table)
library(fixest)

# =====
# File paths
# =====
FIRST_SALE_PATH <- "datastore/derived/first_sale_data.csv"
TIMING_PATH <- "datastore/derived/group_round_timing.csv"
OUTPUT_PATH <- "analysis/output/tables/ols_first_sale_behavior.tex"

# =====
# Main
# =====
main <- function() {
  dt <- load_first_sale_data()
  dt <- merge_bad_state(dt)
  dt_sales <- restrict_to_sales(dt)

  models <- run_models(dt_sales)
  write_latex_table(models)
}

# =====
# Data loading and bad_state merge
# =====
load_first_sale_data <- function() {
  dt <- fread(FIRST_SALE_PATH)
  cat("Loaded", nrow(dt), "group-round observations\n")
  return(dt)
}

merge_bad_state <- function(dt) {
  timing <- fread(TIMING_PATH)
  timing[, bad_state := as.integer(state == 0)]
  keys <- c("session", "segment_num", "group_id", "round_num")

  merged <- merge(dt, timing[, c(keys, "bad_state"), with = FALSE],
                  by = keys, all.x = TRUE, sort = FALSE)
  validate_merge(merged, nrow(dt))

  merged[, treatment := as.factor(treatment)]
  merged[, segment_num := as.factor(segment_num)]
  merged[, global_group_id := as.factor(global_group_id)]
  return(merged)
}

validate_merge <- function(merged, before) {
  # Fail loudly: the merge must be 1:1 with no unmatched key
  if (nrow(merged) != before) {
    stop(sprintf("bad_state merge changed row count: %d -> %d", before, nrow(merged)))
  }
  if (any(is.na(merged$bad_state))) {
    stop(sprintf("bad_state merge produced %d NA values on key join",
                 sum(is.na(merged$bad_state))))
  }
}

restrict_to_sales <- function(dt) {
  dt_sales <- dt[!is.na(signal_at_first_sale)]
  cat("Restricted to sales:", nrow(dt_sales), "of", nrow(dt), "observations\n")
  return(dt_sales)
}

# =====
# Models (cluster on global_group_id)
# =====
run_models <- function(dt) {
  m1 <- feols(signal_at_first_sale ~ treatment + segment_num + round_num,
              cluster = ~global_group_id, data = dt)
  m2 <- feols(signal_at_first_sale ~ treatment + segment_num + round_num + bad_state,
              cluster = ~global_group_id, data = dt)
  m3 <- feols(first_sale_period ~ treatment + segment_num + round_num,
              cluster = ~global_group_id, data = dt)
  m4 <- feols(first_sale_period ~ treatment + segment_num + round_num + bad_state,
              cluster = ~global_group_id, data = dt)
  return(list(m1 = m1, m2 = m2, m3 = m3, m4 = m4))
}

# =====
# LaTeX output
# =====
write_latex_table <- function(models) {
  etable(models$m1, models$m2, models$m3, models$m4,
         file = OUTPUT_PATH,
         replace = TRUE,
         float = FALSE,
         tex = TRUE,
         headers = list("Belief at first sale" = 2, "Period of first sale" = 2),
         style.tex = style.tex(fontsize = "scriptsize"))
  cat("Wrote", OUTPUT_PATH, "\n")
}

# =====
# Run
# =====
if (!interactive()) {
  main()
}
