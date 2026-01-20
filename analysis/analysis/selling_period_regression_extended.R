# Purpose: Extended cascade analysis with first/second seller and interaction specifications
# Author: Claude
# Date: 2026-01-18

library(data.table)
library(fixest)
library(plm)

# FILE PATHS
INPUT_PATH <- "datastore/derived/individual_period_dataset.csv"
OUTPUT_PATH_FIRST <- "analysis/output/analysis/selling_period_regression_first_sellers.tex"
OUTPUT_PATH_SECOND <- "analysis/output/analysis/selling_period_regression_second_sellers.tex"
OUTPUT_PATH_INTERACTIONS <- "analysis/output/analysis/selling_period_regression_interactions.tex"

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  cat("Loading data from:", INPUT_PATH, "\n")
  df <- load_data(INPUT_PATH)
  cat("Total observations:", nrow(df), "\n")

  cat("\n=== Specification 1: First Sellers ===\n")
  run_first_sellers(df)

  cat("\n=== Specification 2: Second Sellers ===\n")
  run_second_sellers(df)

  cat("\n=== Specification 3: Interaction Model ===\n")
  run_interaction_model(df)

  cat("\nDone!\n")
}

# =====
# Data loading
# =====
load_data <- function(file_path) {
  df <- fread(file_path)
  df[, player_id := paste(session_id, player, sep = "_")]
  # Groups are unique within each segment, so include segment in cluster ID
  df[, global_group_id := paste(session_id, segment, group_id, sep = "_")]
  df[, group_round_id := paste(session_id, segment, group_id, round, sep = "_")]
  df[, player_group_round_id := paste(player_id, segment, group_id, round, sep = "_")]
  df[, segment := as.factor(segment)]
  df[, treatment := as.factor(treatment)]
  return(df)
}

# =====
# Specification 1: First Sellers
# =====
run_first_sellers <- function(df) {
  # Identify players who are first sellers (prior_group_sales == 0 when they sold)
  first_seller_ids <- df[prior_group_sales == 0 & sold == 1, unique(player_group_round_id)]
  cat("Number of first seller player-group-rounds:", length(first_seller_ids), "\n")

  # Keep all observations for these players up to and including when they sell
  # already_sold == 0 means they haven't sold in a PREVIOUS period
  # This includes the period where sold == 1
  df_first <- df[player_group_round_id %in% first_seller_ids & already_sold == 0]
  cat("First sellers sample size (all periods):", nrow(df_first), "\n")

  # Verify: count how many sold == 1 observations are included
  n_sales <- sum(df_first$sold)
  cat("  - Observations where sold=1:", n_sales, "\n")
  cat("  - Observations where sold=0:", nrow(df_first) - n_sales, "\n")

  # Random effects model - no prior sales dummies since they're first
  pdata <- pdata.frame(as.data.frame(df_first), index = c("player_id", "period"))
  model <- plm(
    sold ~ signal + round + segment + treatment,
    data = pdata,
    model = "random"
  )

  cat("\nFirst Sellers Results:\n")
  print(summary(model))

  export_table_single(model, OUTPUT_PATH_FIRST, "First Sellers")
  cat("Table exported to:", OUTPUT_PATH_FIRST, "\n")
}

# =====
# Specification 2: Second Sellers
# =====
run_second_sellers <- function(df) {
  # Identify players who are second sellers (prior_group_sales == 1 when they sold)
  second_seller_ids <- df[prior_group_sales == 1 & sold == 1, unique(player_group_round_id)]
  cat("Number of second seller player-group-rounds:", length(second_seller_ids), "\n")

  # Keep all observations for these players up to and including when they sell
  df_second <- df[player_group_round_id %in% second_seller_ids & already_sold == 0]
  cat("Second sellers sample size (all periods):", nrow(df_second), "\n")

  # Verify: count how many sold == 1 observations are included
  n_sales <- sum(df_second$sold)
  cat("  - Observations where sold=1:", n_sales, "\n")
  cat("  - Observations where sold=0:", nrow(df_second) - n_sales, "\n")

  # Create dummy: 1 if the first sale was in immediately previous period
  df_second <- create_prev_period_dummy_for_second(df_second, df)

  cat("Observations with first sale in t-1:", sum(df_second$dummy_prev_period), "\n")

  # Create treatment dummy for interaction (1 if treatment 2)
  df_second[, treat2 := as.integer(treatment == "tr2")]
  df_second[, prev_x_treat := dummy_prev_period * treat2]

  cat("Interaction term (prev_period x treatment2):", sum(df_second$prev_x_treat), "\n")

  # Random effects model 1: without interaction
  pdata <- pdata.frame(as.data.frame(df_second), index = c("player_id", "period"))
  model1 <- plm(
    sold ~ dummy_prev_period + signal + round + segment + treatment,
    data = pdata,
    model = "random"
  )

  # Random effects model 2: with interaction
  model2 <- plm(
    sold ~ dummy_prev_period + treatment + prev_x_treat + signal + round + segment,
    data = pdata,
    model = "random"
  )

  cat("\nSecond Sellers Results (without interaction):\n")
  print(summary(model1))

  cat("\nSecond Sellers Results (with interaction):\n")
  print(summary(model2))

  export_table_single(model1, OUTPUT_PATH_SECOND, "Second Sellers")
  cat("Table exported to:", OUTPUT_PATH_SECOND, "\n")

  export_table_second_sellers_interaction(model1, model2,
    "analysis/output/analysis/selling_period_regression_second_sellers_interaction.tex")
  cat("Interaction table exported\n")
}

create_prev_period_dummy_for_second <- function(df_second, df_full) {
  # Get first sale period for each group-round
  first_sales <- df_full[prior_group_sales == 0 & sold == 1,
                         .(first_sale_period = min(period)),
                         by = group_round_id]

  df_second <- merge(df_second, first_sales, by = "group_round_id", all.x = TRUE)

  # Dummy = 1 if current period is immediately after first sale period
  df_second[, dummy_prev_period := as.integer(first_sale_period == (period - 1))]
  df_second[, first_sale_period := NULL]

  return(df_second)
}

# =====
# Specification 3: Interaction Model
# =====
run_interaction_model <- function(df) {
  # Filter: not already sold (includes period where they sell)
  df_int <- df[already_sold == 0]
  cat("Interaction model sample size:", nrow(df_int), "\n")

  # Create cumulative dummies
  df_int[, dummy_1_cum := as.integer(prior_group_sales == 1)]
  df_int[, dummy_2_cum := as.integer(prior_group_sales == 2)]
  df_int[, dummy_3_cum := as.integer(prior_group_sales == 3)]

  # Create previous period dummies
  df_int <- create_prev_period_dummies(df_int)

  # Create interactions
  df_int[, int_1_1 := dummy_1_cum * dummy_1_prev]
  df_int[, int_2_1 := dummy_2_cum * dummy_1_prev]
  df_int[, int_2_2 := dummy_2_cum * dummy_2_prev]
  df_int[, int_3_1 := dummy_3_cum * dummy_1_prev]
  df_int[, int_3_2 := dummy_3_cum * dummy_2_prev]
  df_int[, int_3_3 := dummy_3_cum * dummy_3_prev]

  print_interaction_summary(df_int)

  # Random effects model
  pdata <- pdata.frame(as.data.frame(df_int), index = c("player_id", "period"))
  model <- plm(
    sold ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
      int_1_1 + int_2_1 + int_2_2 + int_3_1 + int_3_2 + int_3_3 +
      signal + round + segment + treatment,
    data = pdata,
    model = "random"
  )

  cat("\nInteraction Model Results:\n")
  print(summary(model))

  export_table_interactions(model, OUTPUT_PATH_INTERACTIONS)
  cat("Table exported to:", OUTPUT_PATH_INTERACTIONS, "\n")
}

create_prev_period_dummies <- function(df) {
  # Compute total group sales per period within each group-round
  period_sales <- df[, .(n_sales = sum(sold)), by = .(group_round_id, period)]
  setorder(period_sales, group_round_id, period)

  # Shift at PERIOD level to get previous period sales
  period_sales[, prev_period_n_sales := shift(n_sales, 1, type = "lag"),
               by = group_round_id]

  # Merge back
  df <- merge(df, period_sales[, .(group_round_id, period, prev_period_n_sales)],
              by = c("group_round_id", "period"), all.x = TRUE)

  # Create dummies
  df[, prev_period_n_sales := fifelse(is.na(prev_period_n_sales), 0L,
                                      as.integer(prev_period_n_sales))]
  df[, dummy_1_prev := as.integer(prev_period_n_sales == 1)]
  df[, dummy_2_prev := as.integer(prev_period_n_sales == 2)]
  df[, dummy_3_prev := as.integer(prev_period_n_sales == 3)]

  df[, prev_period_n_sales := NULL]
  return(df)
}

print_interaction_summary <- function(df) {
  cat("\nCumulative dummy summary:\n")
  cat("  dummy_1_cum:", sum(df$dummy_1_cum), "\n")
  cat("  dummy_2_cum:", sum(df$dummy_2_cum), "\n")
  cat("  dummy_3_cum:", sum(df$dummy_3_cum), "\n")

  cat("\nPrevious period dummy summary:\n")
  cat("  dummy_1_prev:", sum(df$dummy_1_prev), "\n")
  cat("  dummy_2_prev:", sum(df$dummy_2_prev), "\n")
  cat("  dummy_3_prev:", sum(df$dummy_3_prev), "\n")

  cat("\nInteraction summary:\n")
  cat("  int_1_1 (cum1 x prev1):", sum(df$int_1_1), "\n")
  cat("  int_2_1 (cum2 x prev1):", sum(df$int_2_1), "\n")
  cat("  int_2_2 (cum2 x prev2):", sum(df$int_2_2), "\n")
  cat("  int_3_1 (cum3 x prev1):", sum(df$int_3_1), "\n")
  cat("  int_3_2 (cum3 x prev2):", sum(df$int_3_2), "\n")
  cat("  int_3_3 (cum3 x prev3):", sum(df$int_3_3), "\n")
}

# =====
# Table export functions
# =====
export_table_single <- function(model, output_path, title) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  m_summary <- summary(model)
  m_coefs <- coef(m_summary)

  latex_lines <- build_table_header(title)
  latex_lines <- c(latex_lines, format_coef_rows(m_coefs))
  latex_lines <- c(latex_lines, build_table_footer(model, m_summary))

  writeLines(latex_lines, output_path)
}

export_table_interactions <- function(model, output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  m_summary <- summary(model)
  m_coefs <- coef(m_summary)

  latex_lines <- build_table_header("Interaction Model")
  latex_lines <- c(latex_lines, format_coef_rows_interactions(m_coefs))
  latex_lines <- c(latex_lines, build_table_footer(model, m_summary))

  writeLines(latex_lines, output_path)
}

export_table_second_sellers_interaction <- function(model1, model2, output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  m1_summary <- summary(model1)
  m2_summary <- summary(model2)
  m1_coefs <- coef(m1_summary)
  m2_coefs <- coef(m2_summary)

  latex_lines <- c(
    "",
    "\\begingroup",
    "\\centering",
    "\\scriptsize",
    "\\begin{tabular}{lcc}",
    "   \\tabularnewline \\midrule \\midrule",
    "   \\multicolumn{3}{c}{Second Sellers}\\\\",
    "   \\midrule",
    "   Dependent Variable: & \\multicolumn{2}{c}{sold}\\\\",
    "   Model:               & (1)             & (2)\\\\",
    "   \\midrule",
    "   \\emph{Variables}\\\\"
  )

  # Helper to format a row with two columns
  format_row <- function(label, c1, se1, c2, se2) {
    val1 <- if (is.na(c1)) "" else {
      sig1 <- get_sig_stars(c1, se1)
      paste0(format(round(c1, 4), nsmall = 4), sig1)
    }
    val2 <- if (is.na(c2)) "" else {
      sig2 <- get_sig_stars(c2, se2)
      paste0(format(round(c2, 4), nsmall = 4), sig2)
    }
    se1_str <- if (is.na(se1)) "" else paste0("(", format(round(se1, 4), nsmall = 4), ")")
    se2_str <- if (is.na(se2)) "" else paste0("(", format(round(se2, 4), nsmall = 4), ")")
    c(
      sprintf("   %-25s & %s & %s\\\\", label, val1, val2),
      sprintf("   %-25s & %s & %s\\\\", "", se1_str, se2_str)
    )
  }

  # Variable order for second sellers interaction table
  vars <- c("(Intercept)", "dummy_prev_period", "treatmenttr2", "prev_x_treat",
            "signal", "round", "segment2", "segment3", "segment4")
  labels <- c("Constant", "dummy\\_prev\\_period", "treatment\\_2",
              "prev\\_period $\\times$ treatment\\_2",
              "signal", "round", "segment\\_2", "segment\\_3", "segment\\_4")

  for (i in seq_along(vars)) {
    v <- vars[i]
    lbl <- labels[i]
    c1 <- if (v %in% rownames(m1_coefs)) m1_coefs[v, 1] else NA
    se1 <- if (v %in% rownames(m1_coefs)) m1_coefs[v, 2] else NA
    c2 <- if (v %in% rownames(m2_coefs)) m2_coefs[v, 1] else NA
    se2 <- if (v %in% rownames(m2_coefs)) m2_coefs[v, 2] else NA
    latex_lines <- c(latex_lines, format_row(lbl, c1, se1, c2, se2))
  }

  # Fit statistics
  latex_lines <- c(latex_lines,
    "   \\midrule",
    "   \\emph{Fit statistics}\\\\",
    sprintf("   Observations & %s & %s\\\\",
            format(nobs(model1), big.mark = ","),
            format(nobs(model2), big.mark = ",")),
    sprintf("   R$^2$ & %.5f & %.5f\\\\",
            m1_summary$r.squared[1],
            m2_summary$r.squared[1]),
    "   \\midrule \\midrule",
    "   \\multicolumn{3}{l}{\\emph{Random effects (individual level) SE in parentheses}}\\\\",
    "   \\multicolumn{3}{l}{\\emph{Signif. Codes: ***: 0.01, **: 0.05, *: 0.1}}\\\\",
    "\\end{tabular}",
    "\\par\\endgroup",
    "",
    ""
  )

  writeLines(latex_lines, output_path)
  cat("Table exported to:", output_path, "\n")
}

build_table_header <- function(title) {
  c(
    "",
    "\\begingroup",
    "\\centering",
    "\\scriptsize",
    "\\begin{tabular}{lc}",
    "   \\tabularnewline \\midrule \\midrule",
    sprintf("   \\multicolumn{2}{c}{%s}\\\\", title),
    "   \\midrule",
    "   Dependent Variable: & sold\\\\",
    "   \\midrule",
    "   \\emph{Variables}\\\\"
  )
}

format_coef_rows <- function(coefs) {
  lines <- c()
  for (var in rownames(coefs)) {
    label <- format_var_name(var)
    val <- coefs[var, 1]
    se <- coefs[var, 2]
    sig <- get_sig_stars(val, se)
    lines <- c(lines,
               sprintf("   %-25s & %s%s\\\\", label, format(round(val, 4), nsmall = 4), sig),
               sprintf("   %-25s & (%s)\\\\", "", format(round(se, 4), nsmall = 4)))
  }
  return(lines)
}

format_coef_rows_interactions <- function(coefs) {
  lines <- c()
  # Define display order for interaction model
  var_order <- c("(Intercept)", "dummy_1_cum", "dummy_2_cum", "dummy_3_cum",
                 "int_1_1", "int_2_1", "int_2_2", "int_3_1", "int_3_2", "int_3_3",
                 "signal", "round", "segment2", "segment3", "segment4", "treatmenttr2")

  for (var in var_order) {
    if (var %in% rownames(coefs)) {
      label <- format_var_name(var)
      val <- coefs[var, 1]
      se <- coefs[var, 2]
      sig <- get_sig_stars(val, se)
      lines <- c(lines,
                 sprintf("   %-25s & %s%s\\\\", label, format(round(val, 4), nsmall = 4), sig),
                 sprintf("   %-25s & (%s)\\\\", "", format(round(se, 4), nsmall = 4)))
    }
  }
  return(lines)
}

format_var_name <- function(var) {
  mapping <- list(
    "(Intercept)" = "Constant",
    "dummy_1_cum" = "dummy\\_1\\_cum",
    "dummy_2_cum" = "dummy\\_2\\_cum",
    "dummy_3_cum" = "dummy\\_3\\_cum",
    "dummy_1_prev" = "dummy\\_1\\_prev",
    "dummy_prev_period" = "dummy\\_prev\\_period",
    "int_1_1" = "cum1 $\\times$ prev1",
    "int_2_1" = "cum2 $\\times$ prev1",
    "int_2_2" = "cum2 $\\times$ prev2",
    "int_3_1" = "cum3 $\\times$ prev1",
    "int_3_2" = "cum3 $\\times$ prev2",
    "int_3_3" = "cum3 $\\times$ prev3",
    "signal" = "signal",
    "round" = "round",
    "segment2" = "segment\\_2",
    "segment3" = "segment\\_3",
    "segment4" = "segment\\_4",
    "treatmenttr2" = "treatment\\_2"
  )
  if (var %in% names(mapping)) return(mapping[[var]])
  return(gsub("_", "\\\\_", var))
}

get_sig_stars <- function(val, se) {
  t_stat <- abs(val / se)
  if (t_stat > 2.576) return("$^{***}$")
  if (t_stat > 1.96) return("$^{**}$")
  if (t_stat > 1.645) return("$^{*}$")
  return("")
}

build_table_footer <- function(model, m_summary) {
  c(
    "   \\midrule",
    "   \\emph{Fit statistics}\\\\",
    sprintf("   Observations & %s\\\\", format(nobs(model), big.mark = ",")),
    sprintf("   R$^2$ & %.5f\\\\", m_summary$r.squared[1]),
    "   \\midrule \\midrule",
    "   \\multicolumn{2}{l}{\\emph{Random effects (individual level) SE in parentheses}}\\\\",
    "   \\multicolumn{2}{l}{\\emph{Signif. Codes: ***: 0.01, **: 0.05, *: 0.1}}\\\\",
    "\\end{tabular}",
    "\\par\\endgroup",
    "",
    ""
  )
}

# %%
if (!interactive()) main()
