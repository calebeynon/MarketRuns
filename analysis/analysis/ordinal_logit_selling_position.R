# Purpose: Ordinal logit regressions for selling position (rank 1-4)
#          Model 1: Full sample, clm (non-sellers = rank 4, no random effects)
#          Model 2: Sellers only, clmm (did_sell == 1, player random effects)
#          Output: Two-column minipage AME table with all controls displayed
# Author: Claude Code
# Date: 2026-02-19

library(data.table)
library(ordinal)
library(marginaleffects)

# FILE PATHS
INPUT_PATH <- "datastore/derived/ordinal_selling_position.csv"
OUTPUT_PATH <- "analysis/output/tables/ordinal_logit_selling_position.tex"

# VARIABLE LISTS
CONTINUOUS_VARS <- c("state_anxiety", "impulsivity", "conscientiousness",
                     "fear_p95", "anger_p95",
                     "extraversion", "agreeableness", "neuroticism", "openness",
                     "contempt_p95", "disgust_p95", "joy_p95", "sadness_p95",
                     "surprise_p95", "engagement_p95", "valence_p95",
                     "age", "round")
EMOTION_COLS <- c("fear_p95", "anger_p95", "contempt_p95", "disgust_p95",
                  "joy_p95", "sadness_p95", "surprise_p95",
                  "engagement_p95", "valence_p95")
ALL_PREDICTORS <- c("state_anxiety", "impulsivity", "conscientiousness",
                    EMOTION_COLS, "extraversion", "agreeableness",
                    "neuroticism", "openness",
                    "age", "gender_female",
                    "segment2", "segment3", "segment4", "round")

# Display labels for all predictors
VAR_DICT <- c(
  state_anxiety = "State anxiety", impulsivity = "Impulsivity",
  conscientiousness = "Conscientiousness",
  fear_p95 = "Fear (p95)", anger_p95 = "Anger (p95)",
  contempt_p95 = "Contempt (p95)", disgust_p95 = "Disgust (p95)",
  joy_p95 = "Joy (p95)", sadness_p95 = "Sadness (p95)",
  surprise_p95 = "Surprise (p95)", engagement_p95 = "Engagement (p95)",
  valence_p95 = "Valence (p95)",
  extraversion = "Extraversion", agreeableness = "Agreeableness",
  neuroticism = "Neuroticism", openness = "Openness",
  age = "Age", gender_female = "Female",
  segment2 = "Segment 2", segment3 = "Segment 3",
  segment4 = "Segment 4", round = "Round"
)

# Ordered variable groups for display
KEY_TRAITS <- c("state_anxiety", "impulsivity", "conscientiousness")
OTHER_TRAITS <- c("extraversion", "agreeableness", "neuroticism", "openness")
CONTROLS <- c("age", "gender_female", "segment2", "segment3",
              "segment4", "round")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  cat("Loading data from:", INPUT_PATH, "\n")
  df <- load_and_prepare(INPUT_PATH)

  cat("\n--- Fitting Model 1: Full Sample (clm, no RE) ---\n")
  model1 <- fit_clm(df, "Full Sample")

  df_sellers <- df[did_sell == 1]
  cat("\n--- Fitting Model 2: Sellers Only (clmm, player RE) ---\n")
  model2 <- fit_clmm(df_sellers, "Sellers Only")

  cat("\n--- Computing AMEs on P(rank=1) ---\n")
  ame1 <- compute_ames(model1)
  model2_clm <- fit_clm(df_sellers, "Sellers Only (clm for AMEs)")
  ame2 <- compute_ames(model2_clm)

  build_minipage_table(model1, model2, ame1, ame2, OUTPUT_PATH)
  cat("\nDone!\n")
}

# =====
# Data loading and preparation
# =====
load_and_prepare <- function(file_path) {
  df <- fread(file_path)
  cat("Raw data dimensions:", nrow(df), "x", ncol(df), "\n")
  n_before <- nrow(df)
  df <- df[complete.cases(df[, ..EMOTION_COLS])]
  cat("Dropped", n_before - nrow(df), "rows with missing emotions\n")
  df[, sell_rank := ordered(factor(sell_rank))]
  df[, player_id := as.factor(player_id)]
  create_segment_dummies(df)
  standardize_continuous(df)
  return(df)
}

create_segment_dummies <- function(df) {
  df[, segment2 := as.integer(segment == 2)]
  df[, segment3 := as.integer(segment == 3)]
  df[, segment4 := as.integer(segment == 4)]
}

standardize_continuous <- function(df) {
  for (v in CONTINUOUS_VARS) df[, (v) := scale(get(v))]
}

# =====
# Model fitting
# =====
fit_clm <- function(df, label) {
  fml <- paste("sell_rank ~", paste(ALL_PREDICTORS, collapse = " + "))
  cat("Formula:", fml, "\nN observations:", nrow(df), "\n")
  model <- clm(as.formula(fml), data = df)
  cat("\n--- Summary:", label, "---\n")
  print(summary(model))
  return(model)
}

fit_clmm <- function(df, label) {
  fml <- paste("sell_rank ~", paste(ALL_PREDICTORS, collapse = " + "),
               "+ (1 | player_id)")
  cat("Formula:", fml, "\nN observations:", nrow(df), "\n")
  model <- clmm(as.formula(fml), data = df)
  cat("\n--- Summary:", label, "---\n")
  print(summary(model))
  return(model)
}

# =====
# Average Marginal Effects on P(rank=1)
# =====
compute_ames <- function(model) {
  ame_r1 <- as.data.table(avg_slopes(model))[group == "1"]
  data.table(var = ame_r1$term, est = ame_r1$estimate,
             se = ame_r1$std.error, pval = ame_r1$p.value)
}

# =====
# LaTeX formatting helpers
# =====
get_sig_stars <- function(pval) {
  if (is.nan(pval) || is.na(pval)) return("")
  if (pval < 0.01) return("$^{***}$")
  if (pval < 0.05) return("$^{**}$")
  if (pval < 0.1) return("$^{*}$")
  return("")
}

format_coef_cell <- function(coefs_dt, varname) {
  row <- coefs_dt[var == varname]
  if (nrow(row) == 0) return(c("", ""))
  val <- sprintf("%.4f%s", row$est, get_sig_stars(row$pval))
  se <- sprintf("(%.4f)", row$se)
  c(val, se)
}

get_var_label <- function(v) {
  if (v %in% names(VAR_DICT)) return(VAR_DICT[v])
  gsub("_", "\\_", v, fixed = TRUE)
}

# =====
# Build single minipage tabular for one model
# =====
build_single_tabular <- function(title, ame_dt) {
  col_spec <- ">{\\raggedright\\arraybackslash}p{3cm}>{\\centering\\arraybackslash}p{1.8cm}"
  lines <- c(
    sprintf("\\begin{tabular}[t]{%s}", col_spec),
    sprintf("\\multicolumn{2}{@{}l}{\\emph{%s}} \\\\", title),
    "   \\midrule \\midrule")
  lines <- append_section(lines, "Key traits", KEY_TRAITS, ame_dt)
  lines <- append_section(lines, "Facial emotions", EMOTION_COLS, ame_dt)
  lines <- append_section(lines, "Other personality traits", OTHER_TRAITS, ame_dt)
  lines <- append_section(lines, "Controls", CONTROLS, ame_dt)
  c(lines, "   \\midrule", "\\end{tabular}")
}

append_section <- function(lines, header, vars, ame_dt) {
  lines <- c(lines, sprintf("   \\emph{%s} & \\\\", header))
  for (v in vars) {
    cell <- format_coef_cell(ame_dt, v)
    label <- get_var_label(v)
    lines <- c(lines,
      sprintf("   %-25s& %s\\\\", label, cell[1]),
      sprintf("   %-25s& %s\\\\", "", cell[2]))
  }
  lines
}

# =====
# Build two-column minipage table
# =====
build_minipage_table <- function(m1, m2, ame1, ame2, output_path) {
  f1 <- extract_fit(m1)
  f2 <- extract_fit(m2)
  lines <- build_table_preamble()
  lines <- c(lines, build_minipages(ame1, ame2))
  lines <- c(lines, "", build_fit_line(f1, f2))
  lines <- c(lines, build_footer())
  write_table(lines, output_path)
}

build_table_preamble <- function() {
  c("", "\\begingroup", "\\centering", "\\scriptsize",
    "\\renewcommand{\\arraystretch}{0.85}",
    "\\setlength{\\tabcolsep}{2pt}")
}

build_minipages <- function(ame1, ame2) {
  left <- build_single_tabular("(1) Full Sample CLM", ame1)
  right <- build_single_tabular("(2) Sellers Only CLMM", ame2)
  c("\\begin{minipage}[t]{0.48\\linewidth}",
    left,
    "\\end{minipage}%",
    "\\hfill",
    "\\begin{minipage}[t]{0.48\\linewidth}",
    right,
    "\\end{minipage}")
}

extract_fit <- function(model) {
  list(n = nobs(model), loglik = logLik(model), aic = AIC(model))
}

build_fit_line <- function(f1, f2) {
  c("\\vspace{0.5em}",
    "\\begin{tabular}{@{}lcc@{}}",
    "   \\emph{Fit statistics} & (1) Full Sample & (2) Sellers Only \\\\",
    "   \\midrule",
    "   Model & CLM & CLMM \\\\",
    sprintf("   Observations & %s & %s \\\\",
            format(f1$n, big.mark = ","), format(f2$n, big.mark = ",")),
    sprintf("   Log-lik. & %.1f & %.1f \\\\",
            as.numeric(f1$loglik), as.numeric(f2$loglik)),
    sprintf("   AIC & %.1f & %.1f \\\\", f1$aic, f2$aic),
    "\\end{tabular}")
}

build_footer <- function() {
  c("\\vspace{0.5em}",
    paste0("\\emph{All coefficients are average marginal effects on ",
           "P(selling first), standardized (z-scored).}"),
    "",
    paste0("\\emph{Model 1 excludes random effects; ",
           "Model 2 includes player random effects ",
           "(AMEs computed via clm refit).}"),
    "",
    "\\emph{Signif. Codes: ***: 0.01, **: 0.05, *: 0.1}",
    "\\endgroup", "", "")
}

write_table <- function(lines, output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  writeLines(lines, output_path)
  cat("Table exported to:", output_path, "\n")
}

# %%
if (!interactive()) main()
