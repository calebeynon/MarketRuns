# Purpose: Analyze emotion spikes when group member sold in prior period
# Author: Claude Code
# Date: 2026-02-02
#
# Three specifications for Fear and Anger:
#   Spec 1: Do emotions spike when sale_prev_period = 1?
#   Spec 2: Does emotion mediate the selling decision?
#   Spec 3: Do traits moderate the emotional response? (Random Effects)

library(data.table)
library(fixest)
library(plm)

# FILE PATHS
INPUT_PATH <- "datastore/derived/emotion_spikes_analysis_dataset.csv"
OUTPUT_DIR <- "analysis/output/tables"

# =====
# Main function
# =====
main <- function() {
  cat("Loading data from:", INPUT_PATH, "\n")
  df <- prepare_data(INPUT_PATH)

  cat("\n", strrep("=", 60), "\n")
  cat("SAMPLE SUMMARY\n")
  cat(strrep("=", 60), "\n")
  print_data_summary(df)

  cat("\n", strrep("=", 60), "\n")
  cat("RUNNING REGRESSIONS\n")
  cat(strrep("=", 60), "\n")

  # Run all specifications
  results <- list()

  cat("\n--- Specification 1: Do emotions spike? ---\n")
  results$spec1_fear <- run_spec1(df, "fear_max")
  results$spec1_anger <- run_spec1(df, "anger_max")

  cat("\n--- Specification 2: Does emotion mediate selling? ---\n")
  results$spec2_fear <- run_spec2(df, "fear_max")
  results$spec2_anger <- run_spec2(df, "anger_max")

  cat("\n--- Specification 3: Do traits moderate emotional response? ---\n")
  results$spec3_fear <- run_spec3(df, "fear_max")
  results$spec3_anger <- run_spec3(df, "anger_max")

  cat("\n--- ROBUSTNESS: Using 95th percentile instead of max ---\n")
  results$robust_spec1_fear <- run_spec1(df, "fear_p95")
  results$robust_spec1_anger <- run_spec1(df, "anger_p95")
  results$robust_spec3_fear <- run_spec3(df, "fear_p95")
  results$robust_spec3_anger <- run_spec3(df, "anger_p95")

  cat("\n", strrep("=", 60), "\n")
  cat("EXPORTING TABLES\n")
  cat(strrep("=", 60), "\n")
  export_all_tables(results)

  cat("\n", strrep("=", 60), "\n")
  cat("KEY RESULTS\n")
  cat(strrep("=", 60), "\n")
  print_key_results(results)

  return(results)
}

# =====
# Data preparation
# =====
prepare_data <- function(file_path) {
  df <- fread(file_path)

  # Convert to factors
  df[, segment := as.factor(segment)]
  df[, treatment := as.factor(treatment)]

  # Standardize traits for interpretability
  df[, neuroticism_z := scale(neuroticism)[, 1]]
  df[, impulsivity_z := scale(impulsivity)[, 1]]
  df[, state_anxiety_z := scale(state_anxiety)[, 1]]

  return(df)
}

print_data_summary <- function(df) {
  cat("\nObservations:", nrow(df), "\n")
  cat("Unique players:", uniqueN(df$player_id), "\n")
  cat("Unique groups:", uniqueN(df$global_group_id), "\n")

  cat("\nsale_prev_period distribution:\n")
  print(table(df$sale_prev_period))

  cat("\nEmotion statistics:\n")
  cat("  fear_max: mean =", round(mean(df$fear_max, na.rm = TRUE), 3),
      ", sd =", round(sd(df$fear_max, na.rm = TRUE), 3), "\n")
  cat("  anger_max: mean =", round(mean(df$anger_max, na.rm = TRUE), 3),
      ", sd =", round(sd(df$anger_max, na.rm = TRUE), 3), "\n")

  cat("\nTrait coverage:\n")
  cat("  neuroticism:", sum(!is.na(df$neuroticism)), "/", nrow(df), "\n")
  cat("  impulsivity:", sum(!is.na(df$impulsivity)), "/", nrow(df), "\n")
  cat("  state_anxiety:", sum(!is.na(df$state_anxiety)), "/", nrow(df), "\n")
}

# =====
# Specification 1: Do emotions spike when sale_prev_period = 1?
# =====
run_spec1 <- function(df, emotion_var) {
  cat("  Running Spec 1 for", emotion_var, "...\n")

  formula_str <- paste0(
    emotion_var, " ~ sale_prev_period + signal + prior_group_sales + ",
    "period + round | player_id + segment"
  )

  model <- feols(
    as.formula(formula_str),
    cluster = ~global_group_id,
    data = df
  )

  cat("    sale_prev_period coef:", round(coef(model)["sale_prev_period"], 4),
      "(p =", round(pvalue(model)["sale_prev_period"], 4), ")\n")

  return(model)
}

# =====
# Specification 2: Does emotion mediate the selling decision?
# =====
run_spec2 <- function(df, emotion_var) {
  cat("  Running Spec 2 for", emotion_var, "...\n")

  # Create interaction term
  df_temp <- copy(df)
  df_temp[, emotion := get(emotion_var)]
  df_temp[, interaction := sale_prev_period * emotion]

  formula_str <- paste0(
    "sold ~ sale_prev_period + ", emotion_var, " + interaction + ",
    "signal + prior_group_sales + period + round | player_id + segment"
  )

  model <- feols(
    as.formula(formula_str),
    cluster = ~global_group_id,
    data = df_temp
  )

  cat("    sale_prev_period coef:", round(coef(model)["sale_prev_period"], 4), "\n")
  cat("    ", emotion_var, " coef:", round(coef(model)[emotion_var], 4), "\n")
  cat("    interaction coef:", round(coef(model)["interaction"], 4),
      "(p =", round(pvalue(model)["interaction"], 4), ")\n")

  return(model)
}

# =====
# Specification 3: Do traits moderate emotional response?
# Uses Random Effects (plm) because traits are time-invariant
# =====
run_spec3 <- function(df, emotion_var) {
  cat("  Running Spec 3 for", emotion_var, "(Random Effects)...\n")

  # Filter to observations with valid traits
  df_valid <- df[!is.na(neuroticism_z) & !is.na(impulsivity_z) & !is.na(state_anxiety_z)]
  cat("    Observations with valid traits:", nrow(df_valid), "\n")

  # Create interaction terms
  df_valid[, sp_neuroticism := sale_prev_period * neuroticism_z]
  df_valid[, sp_impulsivity := sale_prev_period * impulsivity_z]
  df_valid[, sp_state_anxiety := sale_prev_period * state_anxiety_z]

  # Create unique observation ID for panel (player_id is individual, obs_id is time)
  df_valid[, obs_id := .I]

  # Create panel data frame with unique time index
  pdata <- pdata.frame(
    as.data.frame(df_valid),
    index = c("player_id", "obs_id")
  )

  formula_str <- paste0(
    emotion_var, " ~ sale_prev_period + ",
    "neuroticism_z + impulsivity_z + state_anxiety_z + ",
    "sp_neuroticism + sp_impulsivity + sp_state_anxiety + ",
    "signal + prior_group_sales + round + segment"
  )

  model <- plm(
    as.formula(formula_str),
    data = pdata,
    model = "random"
  )

  # Print key coefficients
  summ <- summary(model)
  coefs <- coef(summ)

  cat("    sale_prev_period:", round(coefs["sale_prev_period", 1], 4),
      "(p =", round(coefs["sale_prev_period", 4], 4), ")\n")
  cat("    sp_neuroticism:", round(coefs["sp_neuroticism", 1], 4),
      "(p =", round(coefs["sp_neuroticism", 4], 4), ")\n")
  cat("    sp_impulsivity:", round(coefs["sp_impulsivity", 1], 4),
      "(p =", round(coefs["sp_impulsivity", 4], 4), ")\n")
  cat("    sp_state_anxiety:", round(coefs["sp_state_anxiety", 1], 4),
      "(p =", round(coefs["sp_state_anxiety", 4], 4), ")\n")

  return(model)
}

# =====
# Export tables
# =====
export_all_tables <- function(results) {
  if (!dir.exists(OUTPUT_DIR)) {
    dir.create(OUTPUT_DIR, recursive = TRUE)
  }

  # Spec 1: Emotion comparison (Fear and Anger)
  export_spec1_table(results$spec1_fear, results$spec1_anger,
                     file.path(OUTPUT_DIR, "emotion_spikes_spec1.tex"))

  # Spec 2: Mediation (Fear and Anger)
  export_spec2_table(results$spec2_fear, results$spec2_anger,
                     file.path(OUTPUT_DIR, "emotion_spikes_spec2.tex"))

  # Spec 3: Trait moderation (Fear and Anger)
  export_spec3_table(results$spec3_fear, results$spec3_anger,
                     file.path(OUTPUT_DIR, "emotion_spikes_spec3.tex"))

  # Robustness: 95th percentile
  if (!is.null(results$robust_spec1_fear)) {
    export_spec1_table(results$robust_spec1_fear, results$robust_spec1_anger,
                       file.path(OUTPUT_DIR, "emotion_spikes_spec1_p95.tex"))
    export_spec3_table(results$robust_spec3_fear, results$robust_spec3_anger,
                       file.path(OUTPUT_DIR, "emotion_spikes_spec3_p95.tex"))
  }
}

export_spec1_table <- function(model_fear, model_anger, output_path) {
  etable(
    model_fear, model_anger,
    title = "Specification 1: Do Emotions Spike When sale\\_prev\\_period = 1?",
    headers = c("Fear (max)", "Anger (max)"),
    dict = c(
      sale_prev_period = "sale\\_prev\\_period",
      prior_group_sales = "prior\\_group\\_sales"
    ),
    fitstat = c("n", "r2"),
    notes = "Player and segment fixed effects. Clustered SEs by group.",
    file = output_path,
    float = FALSE,
    tex = TRUE,
    style.tex = style.tex(fontsize = "scriptsize")
  )
  cat("  Exported:", output_path, "\n")
}

export_spec2_table <- function(model_fear, model_anger, output_path) {
  etable(
    model_fear, model_anger,
    title = "Specification 2: Does Emotion Mediate the Selling Decision?",
    headers = c("Fear (max)", "Anger (max)"),
    dict = c(
      sale_prev_period = "sale\\_prev\\_period",
      prior_group_sales = "prior\\_group\\_sales",
      fear_max = "fear\\_max",
      anger_max = "anger\\_max",
      interaction = "sale\\_prev\\_period $\\times$ emotion"
    ),
    fitstat = c("n", "r2"),
    notes = "Dependent variable: sold. Player and segment FE. Clustered SEs by group.",
    file = output_path,
    float = FALSE,
    tex = TRUE,
    style.tex = style.tex(fontsize = "scriptsize")
  )
  cat("  Exported:", output_path, "\n")
}

export_spec3_table <- function(model_fear, model_anger, output_path) {
  # plm models need manual table construction
  fear_summ <- summary(model_fear)
  anger_summ <- summary(model_anger)

  fear_coefs <- coef(fear_summ)
  anger_coefs <- coef(anger_summ)

  # Build LaTeX table
  latex_lines <- c(
    "",
    "\\begingroup",
    "\\centering",
    "\\scriptsize",
    "\\begin{tabular}{lcc}",
    "   \\tabularnewline \\midrule \\midrule",
    "   Dependent Variable: & Fear (max) & Anger (max)\\\\",
    "   \\midrule",
    "   \\emph{Variables}\\\\"
  )

  # Helper function to format coefficient row
  format_row <- function(label, fear_row, anger_row) {
    format_val <- function(coef, se) {
      if (is.na(coef)) return(c("", ""))
      sig <- ifelse(abs(coef / se) > 2.576, "$^{***}$",
                    ifelse(abs(coef / se) > 1.96, "$^{**}$",
                           ifelse(abs(coef / se) > 1.645, "$^{*}$", "")))
      c(paste0(format(round(coef, 4), nsmall = 4), sig),
        paste0("(", format(round(se, 4), nsmall = 4), ")"))
    }

    fear_fmt <- format_val(fear_row[1], fear_row[2])
    anger_fmt <- format_val(anger_row[1], anger_row[2])

    c(sprintf("   %-25s & %s & %s\\\\", label, fear_fmt[1], anger_fmt[1]),
      sprintf("   %-25s & %s & %s\\\\", "", fear_fmt[2], anger_fmt[2]))
  }

  # Key variables
  vars <- c("sale_prev_period", "sp_neuroticism", "sp_impulsivity", "sp_state_anxiety",
            "neuroticism_z", "impulsivity_z", "state_anxiety_z",
            "signal", "prior_group_sales", "round")

  labels <- c("sale\\_prev\\_period", "sale\\_prev\\_period $\\times$ neuroticism",
              "sale\\_prev\\_period $\\times$ impulsivity",
              "sale\\_prev\\_period $\\times$ state\\_anxiety",
              "neuroticism (z)", "impulsivity (z)", "state\\_anxiety (z)",
              "signal", "prior\\_group\\_sales", "round")

  for (i in seq_along(vars)) {
    v <- vars[i]
    if (v %in% rownames(fear_coefs) && v %in% rownames(anger_coefs)) {
      latex_lines <- c(latex_lines,
                       format_row(labels[i], fear_coefs[v, ], anger_coefs[v, ]))
    }
  }

  # Fit statistics
  latex_lines <- c(latex_lines,
                   "   \\midrule",
                   "   \\emph{Fit statistics}\\\\",
                   sprintf("   Observations & %s & %s\\\\",
                           format(nobs(model_fear), big.mark = ","),
                           format(nobs(model_anger), big.mark = ",")),
                   sprintf("   R$^2$ & %.4f & %.4f\\\\",
                           fear_summ$r.squared[1], anger_summ$r.squared[1]),
                   "   \\midrule \\midrule",
                   "   \\multicolumn{3}{l}{\\emph{Random effects model. Standard errors in parentheses.}}\\\\",
                   "   \\multicolumn{3}{l}{\\emph{Signif. Codes: ***: 0.01, **: 0.05, *: 0.1}}\\\\",
                   "\\end{tabular}",
                   "\\par\\endgroup",
                   ""
  )

  writeLines(latex_lines, output_path)
  cat("  Exported:", output_path, "\n")
}

# =====
# Print key results
# =====
print_key_results <- function(results) {
  cat("\n--- Spec 1: Do emotions spike? ---\n")
  cat("Fear: sale_prev_period =",
      round(coef(results$spec1_fear)["sale_prev_period"], 4),
      "(p =", round(pvalue(results$spec1_fear)["sale_prev_period"], 4), ")\n")
  cat("Anger: sale_prev_period =",
      round(coef(results$spec1_anger)["sale_prev_period"], 4),
      "(p =", round(pvalue(results$spec1_anger)["sale_prev_period"], 4), ")\n")

  cat("\n--- Spec 2: Does emotion mediate selling? ---\n")
  cat("Fear interaction =",
      round(coef(results$spec2_fear)["interaction"], 4),
      "(p =", round(pvalue(results$spec2_fear)["interaction"], 4), ")\n")
  cat("Anger interaction =",
      round(coef(results$spec2_anger)["interaction"], 4),
      "(p =", round(pvalue(results$spec2_anger)["interaction"], 4), ")\n")

  cat("\n--- Spec 3: Trait moderation (interactions) ---\n")
  fear_coefs <- coef(summary(results$spec3_fear))
  anger_coefs <- coef(summary(results$spec3_anger))

  for (trait in c("sp_neuroticism", "sp_impulsivity", "sp_state_anxiety")) {
    cat(trait, ":\n")
    cat("  Fear:", round(fear_coefs[trait, 1], 4),
        "(p =", round(fear_coefs[trait, 4], 4), ")\n")
    cat("  Anger:", round(anger_coefs[trait, 1], 4),
        "(p =", round(anger_coefs[trait, 4], 4), ")\n")
  }

  # Robustness results
  if (!is.null(results$robust_spec1_fear)) {
    cat("\n--- ROBUSTNESS: 95th percentile ---\n")
    cat("Spec 1 (p95):\n")
    cat("  Fear: sale_prev_period =",
        round(coef(results$robust_spec1_fear)["sale_prev_period"], 4),
        "(p =", round(pvalue(results$robust_spec1_fear)["sale_prev_period"], 4), ")\n")
    cat("  Anger: sale_prev_period =",
        round(coef(results$robust_spec1_anger)["sale_prev_period"], 4),
        "(p =", round(pvalue(results$robust_spec1_anger)["sale_prev_period"], 4), ")\n")
  }
}

# %%
if (!interactive()) main()
