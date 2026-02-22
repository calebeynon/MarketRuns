# Purpose: Ranked-order logit (Cox PH with strata) for selling position
#          Model 1: Full sample (all observations)
#          Model 2: Sellers only (did_sell == 1, strata with >= 2 obs)
#          Output: Two-column minipage table with log-hazard ratios
# Author: Claude Code
# Date: 2026-02-21

library(data.table)
library(survival)

# FILE PATHS
INPUT_PATH <- "datastore/derived/ordinal_selling_position.csv"
OUTPUT_PATH <- "analysis/output/tables/ro_logit_selling_position.tex"

# VARIABLE LISTS
CONTINUOUS_VARS <- c("state_anxiety", "impulsivity", "conscientiousness",
                     "fear_p95", "anger_p95",
                     "extraversion", "agreeableness", "neuroticism", "openness",
                     "contempt_p95", "disgust_p95", "joy_p95", "sadness_p95",
                     "surprise_p95", "engagement_p95", "valence_p95",
                     "risk_tolerance", "age")
EMOTION_COLS <- c("fear_p95", "anger_p95", "contempt_p95", "disgust_p95",
                  "joy_p95", "sadness_p95", "surprise_p95",
                  "engagement_p95", "valence_p95")

# Segment and round excluded -- absorbed by strata
ALL_PREDICTORS <- c("state_anxiety", "impulsivity", "conscientiousness",
                    EMOTION_COLS, "extraversion", "agreeableness",
                    "neuroticism", "openness", "risk_tolerance",
                    "age", "gender_female")

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
  risk_tolerance = "Risk tolerance",
  age = "Age", gender_female = "Female"
)

# Ordered variable groups for display
KEY_TRAITS <- c("state_anxiety", "impulsivity", "conscientiousness",
                "risk_tolerance")
OTHER_TRAITS <- c("extraversion", "agreeableness", "neuroticism", "openness")
CONTROLS <- c("age", "gender_female")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  cat("Loading data from:", INPUT_PATH, "\n")
  df <- load_and_prepare(INPUT_PATH)

  cat("\n--- Fitting Model 1: Full Sample (Cox PH) ---\n")
  m1 <- fit_cox(df, "Full Sample")

  df_sellers <- prepare_sellers(df)
  cat("\n--- Fitting Model 2: Sellers Only (Cox PH) ---\n")
  m2 <- fit_cox(df_sellers, "Sellers Only")

  coefs1 <- extract_coefs(m1)
  coefs2 <- extract_coefs(m2)
  n_strata <- c(uniqueN(df$stratum), uniqueN(df_sellers$stratum))
  build_minipage_table(m1, m2, coefs1, coefs2, n_strata, OUTPUT_PATH)
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
  create_segment_dummies(df)
  standardize_continuous(df)
  build_survival_vars(df)
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

build_survival_vars <- function(df) {
  df[, sell_rank_rev := 5L - sell_rank]
  df[, stratum := paste(session_id, segment, group_id, round, sep = "_")]
  df[, rev_rank := rank(-1 * sell_rank_rev, ties.method = "average"),
     by = stratum]
  df[, status := 1L]
}

# =====
# Sellers-only data preparation
# =====
prepare_sellers <- function(df) {
  df_s <- df[did_sell == 1]
  # Recompute ranks within strata for sellers only
  df_s[, rev_rank := rank(-1 * sell_rank_rev, ties.method = "average"),
       by = stratum]
  strata_counts <- df_s[, .N, by = stratum]
  valid <- strata_counts[N >= 2, stratum]
  df_s <- df_s[stratum %in% valid]
  cat("Sellers only:", nrow(df_s), "obs in",
      length(valid), "strata\n")
  return(df_s)
}

# =====
# Model fitting
# =====
fit_cox <- function(df, label) {
  fml <- build_cox_formula()
  cat("N observations:", nrow(df), "\n")
  cat("N strata:", uniqueN(df$stratum), "\n")
  model <- coxph(fml, data = df, cluster = player_id)
  cat("\n--- Summary:", label, "---\n")
  print(summary(model))
  return(model)
}

build_cox_formula <- function() {
  rhs <- paste(c(ALL_PREDICTORS, "strata(stratum)"), collapse = " + ")
  as.formula(paste("Surv(rev_rank, status) ~", rhs))
}

# =====
# Coefficient extraction
# =====
extract_coefs <- function(model) {
  s <- summary(model)
  data.table(
    var = rownames(s$coefficients),
    est = s$coefficients[, "coef"],
    se = s$coefficients[, "robust se"],
    pval = s$coefficients[, "Pr(>|z|)"]
  )
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
build_single_tabular <- function(title, coefs_dt) {
  col_spec <- paste0(">{\\raggedright\\arraybackslash}p{2.8cm}",
                     ">{\\centering\\arraybackslash}p{1.8cm}")
  lines <- c(
    sprintf("\\begin{tabular}[t]{%s}", col_spec),
    sprintf("\\multicolumn{2}{@{}l}{\\emph{%s}} \\\\", title),
    "   \\midrule \\midrule")
  lines <- append_section(lines, "Key traits", KEY_TRAITS, coefs_dt)
  lines <- append_section(lines, "Facial emotions", EMOTION_COLS, coefs_dt)
  lines <- append_section(lines, "Other personality traits",
                          OTHER_TRAITS, coefs_dt)
  lines <- append_section(lines, "Controls", CONTROLS, coefs_dt)
  c(lines, "   \\midrule", "\\end{tabular}")
}

append_section <- function(lines, header, vars, coefs_dt) {
  lines <- c(lines, sprintf("   \\emph{%s} & \\\\", header))
  for (v in vars) {
    cell <- format_coef_cell(coefs_dt, v)
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
build_minipage_table <- function(m1, m2, coefs1, coefs2,
                                n_strata, output_path) {
  f1 <- extract_fit(m1, n_strata[1])
  f2 <- extract_fit(m2, n_strata[2])
  lines <- build_table_preamble()
  lines <- c(lines, build_minipages(coefs1, coefs2))
  lines <- c(lines, "", build_fit_line(f1, f2))
  lines <- c(lines, build_footer())
  write_table(lines, output_path)
}

build_table_preamble <- function() {
  c("", "\\begin{center}", "\\begingroup", "\\tiny",
    "\\renewcommand{\\arraystretch}{0.7}",
    "\\setlength{\\tabcolsep}{2pt}")
}

build_minipages <- function(coefs1, coefs2) {
  left <- build_single_tabular("(1) Full Sample", coefs1)
  right <- build_single_tabular("(2) Sellers Only", coefs2)
  c("\\begin{minipage}[t]{0.42\\linewidth}",
    "\\centering", left, "\\end{minipage}%",
    "\\hspace{1.5em}",
    "\\begin{minipage}[t]{0.42\\linewidth}",
    "\\centering", right, "\\end{minipage}")
}

extract_fit <- function(model, n_strata) {
  s <- summary(model)
  list(
    n = model$n,
    n_strata = n_strata,
    concordance = s$concordance["C"],
    loglik = model$loglik[2]
  )
}

build_fit_line <- function(f1, f2) {
  c("\\vspace{0.1em}",
    "\\begin{center}",
    "\\begin{tabular}{lcc}",
    paste0("   \\emph{Fit statistics} & (1) Full Sample",
           " & (2) Sellers Only \\\\"),
    "   \\midrule",
    sprintf("   Observations & %s & %s \\\\",
            format(f1$n, big.mark = ","),
            format(f2$n, big.mark = ",")),
    sprintf("   Strata & %s & %s \\\\",
            format(f1$n_strata, big.mark = ","),
            format(f2$n_strata, big.mark = ",")),
    sprintf("   Concordance & %.3f & %.3f \\\\",
            f1$concordance, f2$concordance),
    sprintf("   Log-lik. & %.1f & %.1f \\\\",
            f1$loglik, f2$loglik),
    "\\end{tabular}",
    "\\end{center}")
}

build_footer <- function() {
  c("\\vspace{0.1em}",
    "\\begin{minipage}{0.95\\linewidth}",
    "\\tiny",
    paste0("\\emph{Coefficients are log-hazard ratios ",
           "(positive = earlier selling), standardized (z-scored). ",
           "Standard errors clustered at the player level.} "),
    paste0("\\emph{Each stratum is a unique session-segment-group-round; ",
           "segment and round controls are absorbed by stratification.} "),
    paste0("\\emph{Model 2 restricted to sellers only, ",
           "excluding strata with fewer than two sellers.} "),
    "\\emph{Signif. Codes: ***: 0.01, **: 0.05, *: 0.1}",
    "\\end{minipage}",
    "\\endgroup", "\\end{center}", "", "")
}

write_table <- function(lines, output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  writeLines(lines, output_path)
  cat("Table exported to:", output_path, "\n")
}

# %%
if (!interactive()) main()
