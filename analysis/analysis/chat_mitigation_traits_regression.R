# Purpose: Analyze whether personality trait effects on selling are moderated by chat
# Author: Claude Code
# Date: 2026-02-02
#
# IDENTIFICATION STRATEGY:
# Traits (impulsivity, neuroticism) are TIME-INVARIANT within players.
# Player fixed effects ABSORB trait main effects. We use random effects for
# trait main effects, and note that trait x chat_segment interactions CAN be
# identified with player FE since chat_segment varies within player.

library(data.table)
library(fixest)
library(plm)

# FILE PATHS
INPUT_PATH <- "datastore/derived/chat_mitigation_dataset.csv"
OUTPUT_PATH <- "analysis/output/analysis/chat_mitigation_traits.tex"

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  print_header("CHAT MITIGATION: PERSONALITY TRAITS ANALYSIS")
  df <- load_and_prepare_data(INPUT_PATH)
  print_summary_statistics(df)

  models <- run_all_models(df)
  print_model_results(models)
  export_latex_table(models, OUTPUT_PATH)
  return(models)
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function(file_path) {
  cat("Loading data from:", file_path, "\n")
  df <- fread(file_path)
  n_before <- nrow(df)
  df <- df[already_sold == 0]
  cat("Filtered to at-risk:", nrow(df), "(dropped", n_before - nrow(df), ")\n")
  df[, c("segment", "treatment", "player_id", "global_group_id") :=
       .(as.factor(segment), as.factor(treatment),
         as.factor(player_id), as.factor(global_group_id))]
  validate_required_vars(df)
  return(df)
}

validate_required_vars <- function(df) {
  required <- c("sold", "impulsivity_z", "neuroticism_z", "state_anxiety",
                "fear_z", "anger_z", "sadness_z", "chat_segment",
                "signal", "prior_group_sales", "round", "player_id")
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) stop("Missing: ", paste(missing, collapse = ", "))
}

# =====
# Summary statistics
# =====
print_summary_statistics <- function(df) {
  print_header("SAMPLE SUMMARY")
  cat("N =", nrow(df), "| Players:", uniqueN(df$player_id),
      "| Sell rate:", round(mean(df$sold), 4), "\n")
  cat("\nBy chat_segment:\n")
  print(df[, .(n = .N, pct_sold = round(mean(sold), 4)), by = chat_segment])
  cat("\nTraits are TIME-INVARIANT: RE for main effects, FE for interactions.\n")
}

# =====
# Regression models
# =====
run_all_models <- function(df) {
  list(
    m1_traits_re = run_model_traits_re(df),
    m2_interactions = run_model_trait_interactions(df),
    m3_combined = run_model_combined(df),
    m4_fe_interactions = run_model_fe_interactions(df)
  )
}

run_model_traits_re <- function(df) {
  cat("\n[1/4] Model 1: Traits + Controls (Random Effects)...\n")
  pdata <- pdata.frame(as.data.frame(df), index = c("player_id", "period"))
  plm(sold ~ impulsivity_z + neuroticism_z + state_anxiety + fear_z + anger_z +
        sadness_z + signal + prior_group_sales + round + chat_segment + treatment,
      data = pdata, model = "random")
}

run_model_trait_interactions <- function(df) {
  cat("[2/4] Model 2: Trait x Chat Interactions (RE)...\n")
  pdata <- pdata.frame(as.data.frame(df), index = c("player_id", "period"))
  plm(sold ~ impulsivity_z + neuroticism_z + state_anxiety +
        impulsivity_z:chat_segment + neuroticism_z:chat_segment +
        state_anxiety:chat_segment + fear_z + anger_z + sadness_z +
        signal + prior_group_sales + round + chat_segment + treatment,
      data = pdata, model = "random")
}

run_model_combined <- function(df) {
  cat("[3/4] Model 3: Combined Emotions + Traits Interactions (RE)...\n")
  pdata <- pdata.frame(as.data.frame(df), index = c("player_id", "period"))
  plm(sold ~ fear_z + anger_z + sadness_z + impulsivity_z + neuroticism_z +
        fear_z:chat_segment + anger_z:chat_segment + sadness_z:chat_segment +
        impulsivity_z:chat_segment + neuroticism_z:chat_segment +
        signal + prior_group_sales + round + chat_segment + treatment,
      data = pdata, model = "random")
}

run_model_fe_interactions <- function(df) {
  cat("[4/4] Model 4: Trait x Chat with Player FE...\n")
  feols(sold ~ impulsivity_z:chat_segment + neuroticism_z:chat_segment +
          state_anxiety:chat_segment + fear_z + anger_z + sadness_z +
          signal + prior_group_sales + round + chat_segment | player_id,
        cluster = ~player_id, data = df)
}

# =====
# Results output
# =====
print_model_results <- function(models) {
  print_header("MODEL RESULTS")
  for (name in names(models)) {
    cat("\n---", name, "---\n")
    print(summary(models[[name]]))
  }
  print_interpretation_note()
}

print_interpretation_note <- function() {
  cat("\n--- INTERPRETATION ---\n")
  cat("Main effect (e.g., impulsivity_z): Effect in NON-chat segments\n")
  cat("Interaction (e.g., impulsivity_z:chat_segment): Change when chat available\n")
  cat("POSITIVE interaction = chat MITIGATES trait effect on selling\n")
}

# =====
# LaTeX table export
# =====
export_latex_table <- function(models, output_path) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  export_re_table(models, output_path)
  export_fe_table(models$m4_fe_interactions, gsub("\\.tex$", "_fe.tex", output_path))
}

export_re_table <- function(models, output_path) {
  sums <- lapply(models[1:3], summary)
  header <- build_table_header()
  rows <- build_coef_rows(sums)
  footer <- build_table_footer(models[1:3], sums)
  writeLines(c(header, rows, footer), output_path)
  cat("Exported RE table:", output_path, "\n")
}

build_table_header <- function() {
  c("", "\\begingroup", "\\centering", "\\scriptsize", "\\begin{tabular}{lccc}",
    "   \\tabularnewline \\midrule \\midrule",
    "   \\multicolumn{4}{c}{Personality Traits and Chat Mitigation}\\\\",
    "   \\midrule", "   Dependent Variable: & \\multicolumn{3}{c}{sold}\\\\",
    "   Model: & (1) Traits & (2) Interactions & (3) Combined\\\\", "   \\midrule")
}

build_coef_rows <- function(sums) {
  vars <- get_var_labels()
  coefs <- lapply(sums, function(s) coef(s))
  rows <- c()
  for (i in seq_along(vars$name)) {
    rows <- c(rows, format_row(vars$label[i], coefs, vars$name[i]))
  }
  return(rows)
}

get_var_labels <- function() {
  list(
    name = c("impulsivity_z", "neuroticism_z", "state_anxiety", "fear_z",
             "anger_z", "sadness_z", "impulsivity_z:chat_segment",
             "neuroticism_z:chat_segment", "state_anxiety:chat_segment",
             "fear_z:chat_segment", "anger_z:chat_segment", "sadness_z:chat_segment",
             "signal", "prior_group_sales", "round", "chat_segment"),
    label = c("Impulsivity (z)", "Neuroticism (z)", "State Anxiety", "Fear (z)",
              "Anger (z)", "Sadness (z)", "Impulsivity $\\times$ Chat",
              "Neuroticism $\\times$ Chat", "State Anxiety $\\times$ Chat",
              "Fear $\\times$ Chat", "Anger $\\times$ Chat", "Sadness $\\times$ Chat",
              "Signal", "Prior Group Sales", "Round", "Chat Segment")
  )
}

format_row <- function(label, coefs, var) {
  vals <- sapply(coefs, function(c) get_coef_str(c, var))
  ses <- sapply(coefs, function(c) get_se_str(c, var))
  c(sprintf("   %-28s & %s & %s & %s\\\\", label, vals[1], vals[2], vals[3]),
    sprintf("   %-28s & %s & %s & %s\\\\", "", ses[1], ses[2], ses[3]))
}

get_coef_str <- function(coefs, var) {
  if (!(var %in% rownames(coefs))) return("")
  val <- coefs[var, 1]; se <- coefs[var, 2]
  sig <- ifelse(abs(val/se) > 2.576, "$^{***}$",
                ifelse(abs(val/se) > 1.96, "$^{**}$",
                       ifelse(abs(val/se) > 1.645, "$^{*}$", "")))
  paste0(format(round(val, 4), nsmall = 4), sig)
}

get_se_str <- function(coefs, var) {
  if (!(var %in% rownames(coefs))) return("")
  paste0("(", format(round(coefs[var, 2], 4), nsmall = 4), ")")
}

build_table_footer <- function(models, sums) {
  obs <- sapply(models, nobs)
  r2s <- sapply(sums, function(s) s$r.squared[1])
  c("   \\midrule", "   \\emph{Fit statistics}\\\\",
    sprintf("   Observations & %s & %s & %s\\\\", format(obs[1], big.mark = ","),
            format(obs[2], big.mark = ","), format(obs[3], big.mark = ",")),
    sprintf("   R$^2$ & %.4f & %.4f & %.4f\\\\", r2s[1], r2s[2], r2s[3]),
    "   \\midrule \\midrule",
    "   \\multicolumn{4}{l}{\\emph{Random effects; SE in parentheses}}\\\\",
    "   \\multicolumn{4}{l}{\\emph{Signif.: ***: 0.01, **: 0.05, *: 0.1}}\\\\",
    "\\end{tabular}", "\\par\\endgroup", "", "")
}

export_fe_table <- function(model, output_path) {
  dict <- c("impulsivity_z:chat_segment" = "Impulsivity $\\times$ Chat",
            "neuroticism_z:chat_segment" = "Neuroticism $\\times$ Chat",
            "state_anxiety:chat_segment" = "State Anxiety $\\times$ Chat",
            "fear_z" = "Fear (z)", "anger_z" = "Anger (z)", "sadness_z" = "Sadness (z)",
            "signal" = "Signal", "prior_group_sales" = "Prior Group Sales",
            "round" = "Round", "chat_segment" = "Chat Segment")
  etable(model, headers = "FE Interactions", dict = dict, fitstat = c("n", "r2", "ar2"),
         file = output_path, float = FALSE, tex = TRUE,
         style.tex = style.tex(fontsize = "scriptsize"))
  cat("Exported FE table:", output_path, "\n")
}

# =====
# Utility
# =====
print_header <- function(title) {
  cat("\n", rep("=", 60), "\n", title, "\n", rep("=", 60), "\n", sep = "")
}

# %%
if (!interactive()) main()
