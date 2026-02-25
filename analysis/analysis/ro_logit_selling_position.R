# Purpose: Ranked-order logit (Cox PH with strata) for selling position.
#          4 models (Full/Sellers x No Traits/With Traits), 4-column longtable.
# Author: Claude Code  |  Date: 2026-02-24

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

DISCRETE_EMOTIONS <- c("fear_p95", "anger_p95", "contempt_p95", "disgust_p95",
                        "joy_p95", "sadness_p95", "surprise_p95",
                        "engagement_p95")
EMOTIONS_AND_VALENCE <- c(DISCRETE_EMOTIONS, "valence_p95")
DEMOGRAPHICS <- c("age", "gender_female")
PERSONALITY_TRAITS <- c("state_anxiety", "impulsivity", "risk_tolerance",
                         "extraversion", "agreeableness", "conscientiousness",
                         "neuroticism", "openness")
PREDICTORS_NO_TRAITS <- c(EMOTIONS_AND_VALENCE, DEMOGRAPHICS)
PREDICTORS_WITH_TRAITS <- c(PREDICTORS_NO_TRAITS, PERSONALITY_TRAITS)

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

# Display groups for table sections (DEMOGRAPHICS defined above)

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  cat("Loading data from:", INPUT_PATH, "\n")
  df <- load_and_prepare(INPUT_PATH)
  df_sellers <- prepare_sellers(df)

  m1 <- fit_cox(df, PREDICTORS_NO_TRAITS, "Full Sample - No Traits")
  m2 <- fit_cox(df, PREDICTORS_WITH_TRAITS, "Full Sample - With Traits")
  m3 <- fit_cox(df_sellers, PREDICTORS_NO_TRAITS, "Sellers Only - No Traits")
  m4 <- fit_cox(df_sellers, PREDICTORS_WITH_TRAITS, "Sellers Only - With Traits")

  coefs <- lapply(list(m1, m2, m3, m4), extract_coefs)
  n_str <- c(uniqueN(df$stratum), uniqueN(df$stratum),
             uniqueN(df_sellers$stratum), uniqueN(df_sellers$stratum))
  fits <- mapply(extract_fit, list(m1, m2, m3, m4), n_str, SIMPLIFY = FALSE)
  build_longtable(coefs, fits, OUTPUT_PATH)
  cat("\nDone!\n")
}

# =====  Data loading and preparation
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
  for (v in CONTINUOUS_VARS) df[, (v) := as.vector(scale(get(v)))]
}

build_survival_vars <- function(df) {
  df[, sell_rank_rev := 5L - sell_rank]
  df[, stratum := paste(session_id, segment, group_id, round, sep = "_")]
  df[, rev_rank := rank(-1 * sell_rank_rev, ties.method = "average"),
     by = stratum]
  df[, status := 1L]
}

# =====  Sellers-only data preparation
prepare_sellers <- function(df) {
  df_s <- df[did_sell == 1]
  df_s[, rev_rank := rank(-1 * sell_rank_rev, ties.method = "average"),
       by = stratum]
  strata_counts <- df_s[, .N, by = stratum]
  valid <- strata_counts[N >= 2, stratum]
  df_s <- df_s[stratum %in% valid]
  cat("Sellers only:", nrow(df_s), "obs in",
      length(valid), "strata\n")
  return(df_s)
}

# =====  Model fitting
build_cox_formula <- function(predictors) {
  rhs <- paste(c(predictors, "strata(stratum)"), collapse = " + ")
  as.formula(paste("Surv(rev_rank, status) ~", rhs))
}

fit_cox <- function(df, predictors, label) {
  fml <- build_cox_formula(predictors)
  cat("\n--- Fitting:", label, "---\n")
  cat("N observations:", nrow(df), "\n")
  cat("N strata:", uniqueN(df$stratum), "\n")
  model <- coxph(fml, data = df, cluster = player_id)
  print(summary(model))
  return(model)
}

# =====  Coefficient extraction
extract_coefs <- function(model) {
  s <- summary(model)
  data.table(
    var = rownames(s$coefficients),
    est = s$coefficients[, "coef"],
    se = s$coefficients[, "robust se"],
    pval = s$coefficients[, "Pr(>|z|)"]
  )
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

# =====  LaTeX formatting helpers
get_sig_stars <- function(pval) {
  if (is.nan(pval) || is.na(pval)) return("")
  if (pval < 0.01) return("$^{***}$")
  if (pval < 0.05) return("$^{**}$")
  if (pval < 0.1) return("$^{*}$")
  return("")
}

get_var_label <- function(v) {
  if (v %in% names(VAR_DICT)) return(VAR_DICT[v])
  gsub("_", "\\_", v, fixed = TRUE)
}

format_cell <- function(coefs_dt, varname) {
  row <- coefs_dt[var == varname]
  if (nrow(row) == 0) return(c("", ""))
  val <- sprintf("%.4f%s", row$est, get_sig_stars(row$pval))
  se <- sprintf("(%.4f)", row$se)
  c(val, se)
}

# =====  Longtable construction
build_longtable <- function(coefs, fits, output_path) {
  lines <- c(longtable_header(), longtable_body(coefs),
             longtable_fit(fits), longtable_footer())
  write_table(lines, output_path)
}

longtable_header <- function() {
  cap <- paste0("\\caption{Ranked-order logit: selling position ",
                "within group-round} \\label{tab:ro_logit_selling_position} \\\\")
  hdr <- c("", "\\begingroup", "\\centering", "\\scriptsize",
    "\\setlength{\\LTcapwidth}{\\textwidth}",
    "\\begin{longtable}{l*{4}{>{\\centering\\arraybackslash}p{2.2cm}}}", cap)
  c(hdr, col_headers("\\endfirsthead"),
    "\\multicolumn{5}{l}{\\emph{(continued)}} \\\\",
    col_headers("\\endhead"),
    "   \\midrule",
    "   \\multicolumn{5}{r}{\\emph{continued on next page}} \\\\",
    "\\endfoot", "\\endlastfoot")
}

col_headers <- function(end_cmd) {
  c("   \\midrule \\midrule",
    "   & \\multicolumn{2}{c}{Full Sample} & \\multicolumn{2}{c}{Sellers Only} \\\\",
    "   \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
    "   & (1) & (2) & (3) & (4) \\\\",
    "   & No Traits & With Traits & No Traits & With Traits \\\\",
    "   \\midrule", end_cmd)
}

longtable_body <- function(coefs) {
  lines <- c()
  lines <- add_section(lines, EMOTIONS_AND_VALENCE, coefs, "all")
  lines <- add_section(lines, DEMOGRAPHICS, coefs, "all")
  lines <- add_section(lines, PERSONALITY_TRAITS, coefs, "traits")
  lines
}

add_section <- function(lines, vars, coefs, type) {
  for (v in vars) lines <- c(lines, format_var_row(v, coefs, type))
  lines
}

format_var_row <- function(varname, coefs, type) {
  cells <- lapply(coefs, format_cell, varname)
  vals <- build_row_values(cells, type)
  label <- get_var_label(varname)
  c(sprintf("   %-25s& %s & %s & %s & %s \\\\",
            label, vals$est[1], vals$est[2], vals$est[3], vals$est[4]),
    sprintf("   %-25s& %s & %s & %s & %s \\\\",
            "", vals$se[1], vals$se[2], vals$se[3], vals$se[4]))
}

build_row_values <- function(cells, type) {
  est <- sapply(cells, `[`, 1)
  se <- sapply(cells, `[`, 2)
  # Big Five rows blank in no-traits columns (1 and 3)
  if (type == "traits") {
    est[c(1, 3)] <- ""
    se[c(1, 3)] <- ""
  }
  list(est = est, se = se)
}

longtable_fit <- function(fits) {
  ns <- sapply(fits, function(f) format(f$n, big.mark = ","))
  strata <- sapply(fits, function(f) format(f$n_strata, big.mark = ","))
  conc <- sapply(fits, function(f) sprintf("%.3f", f$concordance))
  lls <- sapply(fits, function(f) sprintf("%.1f", f$loglik))
  fmt_row <- function(label, vals) {
    sprintf("   %-25s& %s & %s & %s & %s \\\\",
            label, vals[1], vals[2], vals[3], vals[4])
  }
  c("   \\midrule",
    "   \\emph{Fit statistics} & & & & \\\\",
    fmt_row("Observations", ns),
    fmt_row("Strata", strata),
    fmt_row("Concordance", conc),
    fmt_row("Log-lik.", lls))
}

longtable_footer <- function() {
  note <- function(txt) {
    sprintf("   \\multicolumn{5}{l}{\\emph{%s}} \\\\", txt)
  }
  c("   \\midrule \\midrule",
    note("Coefficients are log-hazard ratios (positive = earlier selling), standardized (z-scored)."),
    note("Standard errors clustered at the player level. Strata: session-segment-group-round."),
    note("Round, segment, and treatment absorbed by strata (session-segment-group-round)."),
    note("Sellers Only restricted to strata with $\\geq$ 2 sellers."),
    note("Signif. Codes: ***: 0.01, **: 0.05, *: 0.1"),
    "\\end{longtable}", "\\par\\endgroup", "", "")
}

write_table <- function(lines, output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  writeLines(lines, output_path)
  cat("Table exported to:", output_path, "\n")
}

# %%
if (!interactive()) main()
