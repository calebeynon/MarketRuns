# nolint start
# Purpose: determine if conscientiousness causes people to earn a higher payoff, sell more optimally
# Author: Caleb Eynon (Claude Code Down rn)
# Date: 2026-03-03

library(data.table)
library(ggplot2)
library(fixest)

# =====
# File paths
# =====
ind_per_dt <- fread("datastore/derived/individual_period_dataset_extended.csv")
trait_dt <- fread("datastore/derived/emotions_traits_selling_dataset.csv")

# =====
# Main function
# =====
main <- function() {
    dt <- merge_dt(ind_per_dt, trait_dt)
    dt <- collapse_to_round(dt)
    dt <- create_global_group_id(dt)
    model <- run_regression(dt)
    return(model)
}

# =====
# Merge
# =====
merge_dt <- function(ind_per_dt, trait_dt) {
  merged_dt <- merge(ind_per_dt, trait_dt, by = c('session_id','player','segment','round','period'))
  return(merged_dt)
}

# =====
# Collapse data to round level
# =====
collapse_to_round <- function(dt) {
  collapsed_dt <- dt[, .(
    payoff = mean(round_payoff),
    conscientiousness = mean(conscientiousness),
    state_anxiety = mean(state_anxiety),
    impulsivity = mean(impulsivity),
    risk_tolerance = mean(risk_tolerance),
    treatment = treatment.x[1],
    group_id = group_id.x[1],
    round = round[1],
    segment = segment[1]
  ), by = .(session_id, player, segment, round)]
  return(collapsed_dt)
}

# =====
# create global group id
# =====
create_global_group_id <- function(dt) {
  dt[, global_group_id := paste(session_id, segment, group_id, sep = "_")]
  return(dt)
}

# =====
# Regression
# =====
run_regression <- function(dt) {
  model <- feols(payoff ~ conscientiousness + state_anxiety + impulsivity + risk_tolerance  + round + segment + treatment, 
                 data = dt, 
                 cluster = ~global_group_id)
  return(model)
}
