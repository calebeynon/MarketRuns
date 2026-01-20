# Purpose: Analyze selling decisions using dummy variables for cascade intensity
# Author: Claude
# Date: 2026-01-16

library(data.table)
library(fixest)
library(plm)

# FILE PATHS
INPUT_PATH <- "datastore/derived/individual_period_dataset.csv"
OUTPUT_PATH <- "analysis/output/analysis/selling_period_regression_dummies.tex"
OUTPUT_PATH_APPENDIX <- "analysis/output/analysis/selling_period_regression_dummies_appendix.tex"

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  cat("Loading data from:", INPUT_PATH, "\n")
  df <- prepare_data(INPUT_PATH)
  cat("Data dimensions after filtering:", nrow(df), "rows\n")
  print_data_summary(df)

  cat("\nRunning regressions...\n")
  models <- run_models(df)

  cat("\nExporting LaTeX table to:", OUTPUT_PATH, "\n")
  export_table(models, OUTPUT_PATH, include_period_dummies = FALSE)

  cat("\nExporting appendix LaTeX table to:", OUTPUT_PATH_APPENDIX, "\n")
  export_table(models, OUTPUT_PATH_APPENDIX, include_period_dummies = TRUE)

  print_results(models)
  cat("\nDone!\n")
  return(models)
}

# =====
# Data preparation
# =====
prepare_data <- function(file_path) {
  df <- fread(file_path)

  # Filter out observations where player already sold
  df <- df[already_sold == 0]

  # Create identifier variables
  df[, player_id := paste(session_id, player, sep = "_")]
  # Groups are unique within each segment, so include segment in cluster ID
  df[, global_group_id := paste(session_id, segment, group_id, sep = "_")]
  df[, group_round_id := paste(session_id, segment, group_id, round, sep = "_")]

  # Convert to factors
  df[, segment := as.factor(segment)]
  df[, treatment := as.factor(treatment)]

  # Create cumulative sale dummies (based on all previous periods 1 to t-1)
  df[, dummy_1_cum := as.integer(prior_group_sales == 1)]
  df[, dummy_2_cum := as.integer(prior_group_sales == 2)]
  df[, dummy_3_cum := as.integer(prior_group_sales == 3)]

  # Create previous period sale dummies (based on sales in period t-1 only)
  df <- create_prev_period_dummies(df)

  return(df)
}

create_prev_period_dummies <- function(df) {
  # Compute total group sales per period within each group-round
  period_sales <- df[, .(n_sales = sum(sold)), by = .(group_round_id, period)]
  setorder(period_sales, group_round_id, period)

  # Shift at PERIOD level to get previous period sales
  period_sales[, prev_period_n_sales := shift(n_sales, 1, type = "lag"),
               by = group_round_id]

  # Merge lagged sales back to player-level data
  df <- merge(df, period_sales[, .(group_round_id, period, prev_period_n_sales)],
              by = c("group_round_id", "period"), all.x = TRUE)

  # Create dummies for exactly 1, 2, 3 sales in previous period
  df[, prev_period_n_sales := fifelse(is.na(prev_period_n_sales), 0L,
                                      as.integer(prev_period_n_sales))]
  df[, dummy_1_prev := as.integer(prev_period_n_sales == 1)]
  df[, dummy_2_prev := as.integer(prev_period_n_sales == 2)]
  df[, dummy_3_prev := as.integer(prev_period_n_sales == 3)]

  # Clean up temp columns
  df[, c("prev_period_n_sales") := NULL]

  return(df)
}

print_data_summary <- function(df) {
  cat("\nCumulative dummy summary:\n")
  cat("  dummy_1_cum:", sum(df$dummy_1_cum), "obs with exactly 1 prior sale\n")
  cat("  dummy_2_cum:", sum(df$dummy_2_cum), "obs with exactly 2 prior sales\n")
  cat("  dummy_3_cum:", sum(df$dummy_3_cum), "obs with exactly 3 prior sales\n")

  cat("\nPrevious period dummy summary:\n")
  cat("  dummy_1_prev:", sum(df$dummy_1_prev), "obs with exactly 1 sale in t-1\n")
  cat("  dummy_2_prev:", sum(df$dummy_2_prev), "obs with exactly 2 sales in t-1\n")
  cat("  dummy_3_prev:", sum(df$dummy_3_prev), "obs with exactly 3 sales in t-1\n")
}

# =====
# Regression models
# =====
run_models <- function(df) {
  models <- list()

  cat("[1/4] Cumulative dummies (OLS)...\n")
  models$m1 <- feols(
    sold ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
      period + signal + round + segment + treatment,
    cluster = ~global_group_id,
    data = df
  )

  cat("[2/4] Cumulative dummies (Random Effects)...\n")
  pdata <- pdata.frame(as.data.frame(df), index = c("player_id", "period"))
  models$m2 <- plm(
    sold ~ dummy_1_cum + dummy_2_cum + dummy_3_cum +
      period + signal + round + segment + treatment,
    data = pdata,
    model = "random"
  )

  cat("[3/4] Previous period dummies (OLS)...\n")
  models$m3 <- feols(
    sold ~ dummy_1_prev + dummy_2_prev + dummy_3_prev +
      period + signal + round + segment + treatment,
    cluster = ~global_group_id,
    data = df
  )

  cat("[4/4] Previous period dummies (Random Effects)...\n")
  models$m4 <- plm(
    sold ~ dummy_1_prev + dummy_2_prev + dummy_3_prev +
      period + signal + round + segment + treatment,
    data = pdata,
    model = "random"
  )

  return(models)
}

# =====
# Output
# =====
export_table <- function(models, output_path, include_period_dummies = FALSE) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Extract coefficients and standard errors from plm random effects models
  m2_summary <- summary(models$m2)
  m4_summary <- summary(models$m4)

  # Build manual LaTeX table
  latex_lines <- c(
    "",
    "\\begingroup",
    "\\centering",
    "\\scriptsize",
    "\\begin{tabular}{lcc}",
    "   \\tabularnewline \\midrule \\midrule",
    "   Dependent Variable: & \\multicolumn{2}{c}{sold}\\\\",
    "   Model:               & (1)             & (2)\\\\",
    "   \\midrule",
    "   \\emph{Variables}\\\\"
  )

  # Helper function to format coefficient row
  format_coef_row <- function(label, m2_coef, m2_se, m4_coef, m4_se) {
    # Handle NA values
    val2 <- if (is.na(m2_coef)) "" else {
      sig2 <- ifelse(abs(m2_coef / m2_se) > 2.576, "$^{***}$",
                     ifelse(abs(m2_coef / m2_se) > 1.96, "$^{**}$",
                            ifelse(abs(m2_coef / m2_se) > 1.645, "$^{*}$", "")))
      paste0(format(round(m2_coef, 4), nsmall = 4), sig2)
    }

    val4 <- if (is.na(m4_coef)) "" else {
      sig4 <- ifelse(abs(m4_coef / m4_se) > 2.576, "$^{***}$",
                     ifelse(abs(m4_coef / m4_se) > 1.96, "$^{**}$",
                            ifelse(abs(m4_coef / m4_se) > 1.645, "$^{*}$", "")))
      paste0(format(round(m4_coef, 4), nsmall = 4), sig4)
    }

    se2 <- if (is.na(m2_se)) "" else paste0("(", format(round(m2_se, 4), nsmall = 4), ")")
    se4 <- if (is.na(m4_se)) "" else paste0("(", format(round(m4_se, 4), nsmall = 4), ")")

    c(
      sprintf("   %-20s & %s & %s\\\\", label, val2, val4),
      sprintf("   %-20s & %s & %s\\\\", "", se2, se4)
    )
  }

  # Get coefficients
  m2_coefs <- coef(m2_summary)
  m4_coefs <- coef(m4_summary)

  # Add rows for each variable
  latex_lines <- c(latex_lines,
                   format_coef_row("Constant", m2_coefs["(Intercept)", 1], m2_coefs["(Intercept)", 2],
                                   m4_coefs["(Intercept)", 1], m4_coefs["(Intercept)", 2]))

  if ("dummy_1_cum" %in% rownames(m2_coefs)) {
    latex_lines <- c(latex_lines,
                     format_coef_row("dummy\\_1\\_cum", m2_coefs["dummy_1_cum", 1], m2_coefs["dummy_1_cum", 2],
                                     NA, NA))
  }
  if ("dummy_2_cum" %in% rownames(m2_coefs)) {
    latex_lines <- c(latex_lines,
                     format_coef_row("dummy\\_2\\_cum", m2_coefs["dummy_2_cum", 1], m2_coefs["dummy_2_cum", 2],
                                     NA, NA))
  }
  if ("dummy_3_cum" %in% rownames(m2_coefs)) {
    latex_lines <- c(latex_lines,
                     format_coef_row("dummy\\_3\\_cum", m2_coefs["dummy_3_cum", 1], m2_coefs["dummy_3_cum", 2],
                                     NA, NA))
  }

  if ("dummy_1_prev" %in% rownames(m4_coefs)) {
    latex_lines <- c(latex_lines,
                     format_coef_row("dummy\\_1\\_prev", NA, NA,
                                     m4_coefs["dummy_1_prev", 1], m4_coefs["dummy_1_prev", 2]))
  }
  if ("dummy_2_prev" %in% rownames(m4_coefs)) {
    latex_lines <- c(latex_lines,
                     format_coef_row("dummy\\_2\\_prev", NA, NA,
                                     m4_coefs["dummy_2_prev", 1], m4_coefs["dummy_2_prev", 2]))
  }
  if ("dummy_3_prev" %in% rownames(m4_coefs)) {
    latex_lines <- c(latex_lines,
                     format_coef_row("dummy\\_3\\_prev", NA, NA,
                                     m4_coefs["dummy_3_prev", 1], m4_coefs["dummy_3_prev", 2]))
  }

  # Add period dummies if requested
  if (include_period_dummies) {
    for (i in 2:14) {
      period_name <- paste0("period", i)
      if (period_name %in% rownames(m2_coefs) && period_name %in% rownames(m4_coefs)) {
        latex_lines <- c(latex_lines,
                         format_coef_row(paste0("period\\_", i),
                                         m2_coefs[period_name, 1], m2_coefs[period_name, 2],
                                         m4_coefs[period_name, 1], m4_coefs[period_name, 2]))
      }
    }
  }

  # Add other control variables
  if ("signal" %in% rownames(m2_coefs)) {
    latex_lines <- c(latex_lines,
                     format_coef_row("signal", m2_coefs["signal", 1], m2_coefs["signal", 2],
                                     m4_coefs["signal", 1], m4_coefs["signal", 2]))
  }
  if ("round" %in% rownames(m2_coefs)) {
    latex_lines <- c(latex_lines,
                     format_coef_row("round", m2_coefs["round", 1], m2_coefs["round", 2],
                                     m4_coefs["round", 1], m4_coefs["round", 2]))
  }
  if ("segment2" %in% rownames(m2_coefs)) {
    latex_lines <- c(latex_lines,
                     format_coef_row("segment\\_2", m2_coefs["segment2", 1], m2_coefs["segment2", 2],
                                     m4_coefs["segment2", 1], m4_coefs["segment2", 2]))
  }
  if ("segment3" %in% rownames(m2_coefs)) {
    latex_lines <- c(latex_lines,
                     format_coef_row("segment\\_3", m2_coefs["segment3", 1], m2_coefs["segment3", 2],
                                     m4_coefs["segment3", 1], m4_coefs["segment3", 2]))
  }
  if ("segment4" %in% rownames(m2_coefs)) {
    latex_lines <- c(latex_lines,
                     format_coef_row("segment\\_4", m2_coefs["segment4", 1], m2_coefs["segment4", 2],
                                     m4_coefs["segment4", 1], m4_coefs["segment4", 2]))
  }
  if ("treatmenttr2" %in% rownames(m2_coefs)) {
    latex_lines <- c(latex_lines,
                     format_coef_row("treatment\\_2", m2_coefs["treatmenttr2", 1], m2_coefs["treatmenttr2", 2],
                                     m4_coefs["treatmenttr2", 1], m4_coefs["treatmenttr2", 2]))
  }

  # Add fit statistics
  latex_lines <- c(latex_lines,
                   "   \\midrule",
                   "   \\emph{Fit statistics}\\\\",
                   sprintf("   Observations         & %s & %s\\\\",
                           format(nobs(models$m2), big.mark = ","),
                           format(nobs(models$m4), big.mark = ",")),
                   sprintf("   R$^2$                & %.5f & %.5f\\\\",
                           m2_summary$r.squared[1],
                           m4_summary$r.squared[1]),
                   "   \\midrule \\midrule",
                   "   \\multicolumn{3}{l}{\\emph{Random effects (individual level) standard-errors in parentheses}}\\\\",
                   "   \\multicolumn{3}{l}{\\emph{Signif. Codes: ***: 0.01, **: 0.05, *: 0.1}}\\\\",
                   "\\end{tabular}",
                   "\\par\\endgroup",
                   "",
                   ""
  )

  # Write to file
  writeLines(latex_lines, output_path)
  cat("Table exported to:", output_path, "\n")
}

print_results <- function(models) {
  cat("\n", strrep("=", 60), "\n")
  cat("MODEL RESULTS\n")
  cat(strrep("=", 60), "\n")

  cat("\nModel 1 (Cumulative dummies - OLS):\n")
  print(summary(models$m1))

  cat("\nModel 2 (Cumulative dummies - Random Effects):\n")
  print(summary(models$m2))

  cat("\nModel 3 (Previous period dummies - OLS):\n")
  print(summary(models$m3))

  cat("\nModel 4 (Previous period dummies - Random Effects):\n")
  print(summary(models$m4))
}

# %%
if (!interactive()) main()
