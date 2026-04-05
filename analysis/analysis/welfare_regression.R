# nolint start
# Purpose: Regress group-round welfare on individual traits and controls
# Author: Claude Code
# Date: 2026-04-05

library(data.table)
library(fixest)

# FILE PATHS
INPUT_PANEL <- "datastore/derived/individual_round_panel.csv"
INPUT_WELFARE <- "datastore/derived/group_round_welfare.csv"
INPUT_TRAITS <- "datastore/derived/survey_traits.csv"
OUTPUT_TEX <- "analysis/output/tables/welfare_regression.tex"

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  dt <- load_and_merge()
  dt <- filter_state_1(dt)
  dt <- prepare_variables(dt)
  model <- run_full_traits_model(dt)
  export_table(model)
  cat("Done!\n")
}

# =====
# Load and merge all three datasets
# =====
load_and_merge <- function() {
  panel_dt <- fread(INPUT_PANEL)
  welfare_dt <- fread(INPUT_WELFARE)
  traits_dt <- fread(INPUT_TRAITS)

  setnames(welfare_dt,
    c("session", "segment_num", "round_num"),
    c("session_id", "segment", "round"))

  merged <- merge(panel_dt, welfare_dt,
    by = c("session_id", "segment", "round", "group_id"))

  merged <- merge(merged, traits_dt,
    by = c("session_id", "player"))

  cat("Merged rows:", nrow(merged), "\n")
  return(merged)
}

# =====
# Filter to state == 1 and rounds with at least one sale
# =====
filter_state_1 <- function(dt) {
  n_before <- nrow(dt)
  dt <- dt[state == 1]
  cat("Filtered state==1:", n_before, "->", nrow(dt), "rows\n")
  return(dt)
}

# =====
# Prepare regression variables
# =====
prepare_variables <- function(dt) {
  dt[, global_group_id := paste(session_id, segment, group_id, sep = "_")]
  dt[, gender_female := as.integer(gender == "Female")]
  dt[, segment := factor(segment)]
  dt[, treatment := factor(treatment)]
  return(dt)
}

# =====
# Run full traits model (no session FE, treatment as regressor)
# =====
run_full_traits_model <- function(dt) {
  feols(welfare ~ round + i(segment) + age + gender_female
    + treatment + extraversion + agreeableness + conscientiousness
    + neuroticism + openness + state_anxiety + impulsivity
    + risk_tolerance,
    data = dt, cluster = ~global_group_id)
}

# =====
# Export LaTeX table
# =====
export_table <- function(model) {
  output_dir <- dirname(OUTPUT_TEX)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  etable(
    model,
    dict = c(
      "(Intercept)" = "Constant",
      gender_female = "gender\\_female",
      state_anxiety = "state\\_anxiety",
      risk_tolerance = "risk\\_tolerance",
      treatmenttr2 = "treatment\\_2"
    ),
    fitstat = c("n", "r2", "ar2"),
    float = FALSE,
    tex = TRUE,
    style.tex = style.tex(fontsize = "scriptsize"),
    file = OUTPUT_TEX
  )
  cat("Table exported to:", OUTPUT_TEX, "\n")
}

# %%
if (!interactive()) main()
