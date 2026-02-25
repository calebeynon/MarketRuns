# Purpose: Ordinal logit regressions for selling position (rank 1-4)
#          4 CLM models: Full/Sellers x No Traits/With Traits
#          Output: 4-column longtable with AMEs on P(rank=1)
# Author: Claude Code
# Date: 2026-02-24

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
                     "risk_tolerance", "age", "round")
EMOTION_COLS <- c("fear_p95", "anger_p95", "contempt_p95", "disgust_p95",
                  "joy_p95", "sadness_p95", "surprise_p95",
                  "engagement_p95", "valence_p95")
DISCRETE_EMOTIONS <- c("fear_p95", "anger_p95", "contempt_p95", "disgust_p95",
                        "joy_p95", "sadness_p95", "surprise_p95",
                        "engagement_p95")

EMOTIONS_AND_VALENCE <- c(DISCRETE_EMOTIONS, "valence_p95")
DEMOGRAPHICS_AND_DESIGN <- c("age", "gender_female", "treatment_2",
                              "segment2", "segment3", "segment4", "round")
PERSONALITY_TRAITS <- c("state_anxiety", "impulsivity", "risk_tolerance",
                         "extraversion", "agreeableness", "conscientiousness",
                         "neuroticism", "openness")
PREDICTORS_NO_TRAITS <- c(EMOTIONS_AND_VALENCE, DEMOGRAPHICS_AND_DESIGN)
PREDICTORS_WITH_TRAITS <- c(PREDICTORS_NO_TRAITS, PERSONALITY_TRAITS)

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
  risk_tolerance = "Risk tolerance",
  age = "Age", gender_female = "Female", treatment_2 = "Treatment 2",
  segment2 = "Segment 2", segment3 = "Segment 3",
  segment4 = "Segment 4", round = "Round"
)

# Ordered variable groups for display
CONTROLS <- c("age", "gender_female", "treatment_2", "segment2",
              "segment3", "segment4", "round")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  cat("Loading data from:", INPUT_PATH, "\n")
  df <- load_and_prepare(INPUT_PATH)
  df_sellers <- df[did_sell == 1]

  m1 <- fit_clm(df, PREDICTORS_NO_TRAITS, "Full Sample - No Traits")
  m2 <- fit_clm(df, PREDICTORS_WITH_TRAITS, "Full Sample - With Traits")
  m3 <- fit_clm(df_sellers, PREDICTORS_NO_TRAITS, "Sellers Only - No Traits")
  m4 <- fit_clm(df_sellers, PREDICTORS_WITH_TRAITS, "Sellers Only - With Traits")

  ames <- lapply(list(m1, m2, m3, m4), compute_ames)
  fits <- lapply(list(m1, m2, m3, m4), extract_fit)
  lines <- build_longtable(ames, fits)
  write_table(lines, OUTPUT_PATH)
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
  df[, treatment_2 := as.integer(treatment == "tr2")]
}

standardize_continuous <- function(df) {
  for (v in CONTINUOUS_VARS) df[, (v) := scale(get(v))]
}

# =====
# Model fitting
# =====
fit_clm <- function(df, predictors, label) {
  fml <- paste("sell_rank ~", paste(predictors, collapse = " + "))
  cat("\n--- Fitting:", label, "---\n")
  cat("Formula:", fml, "\nN observations:", nrow(df), "\n")
  model <- clm(as.formula(fml), data = df)
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

get_var_label <- function(v) {
  if (v %in% names(VAR_DICT)) return(VAR_DICT[v])
  gsub("_", "\\_", v, fixed = TRUE)
}

format_4col_cells <- function(var_name, ames_list) {
  vals <- ses <- character(4)
  for (i in seq_along(ames_list)) {
    row <- ames_list[[i]][var == var_name]
    if (nrow(row) == 0) next
    vals[i] <- paste0(sprintf("%.4f", row$est), get_sig_stars(row$pval))
    ses[i] <- paste0("(", sprintf("%.4f", row$se), ")")
  }
  list(v = vals, s = ses)
}

format_coef_row <- function(var_name, ames_list, type = "all") {
  label <- get_var_label(var_name)
  cells <- format_4col_cells(var_name, ames_list)
  if (type == "traits") {
    cells$v[c(1, 3)] <- ""
    cells$s[c(1, 3)] <- ""
  }
  c(sprintf("   %-25s & %s & %s & %s & %s \\\\",
            label, cells$v[1], cells$v[2], cells$v[3], cells$v[4]),
    sprintf("   %-25s & %s & %s & %s & %s \\\\",
            "", cells$s[1], cells$s[2], cells$s[3], cells$s[4]))
}

# =====
# Longtable structure
# =====
build_longtable <- function(ames, fits) {
  lines <- c(build_preamble(), build_col_header())
  lines <- append_section(lines, EMOTIONS_AND_VALENCE, ames, "all")
  lines <- append_section(lines, CONTROLS, ames, "all")
  lines <- append_section(lines, PERSONALITY_TRAITS, ames, "traits")
  c(lines, format_fit_rows(fits), build_footer())
}

build_preamble <- function() {
  c("",
    "\\begingroup", "\\centering", "\\scriptsize",
    "\\setlength{\\LTcapwidth}{\\textwidth}",
    paste0("\\begin{longtable}{l",
           "*{4}{>{\\centering\\arraybackslash}p{2.2cm}}}"),
    paste0("\\caption{Ordinal logit (CLM): selling position ",
           "within group-round}",
           " \\label{tab:ordinal_logit_selling_position} \\\\"))
}

build_col_header <- function() {
  hdr <- c(
    "   \\midrule \\midrule",
    paste0("   & \\multicolumn{2}{c}{Full Sample}",
           " & \\multicolumn{2}{c}{Sellers Only} \\\\"),
    "   \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
    "   & (1) & (2) & (3) & (4) \\\\",
    "   & No Traits & With Traits & No Traits & With Traits \\\\",
    "   \\midrule")
  c(hdr, "\\endfirsthead",
    "\\multicolumn{5}{l}{\\emph{(continued)}} \\\\",
    hdr, "\\endhead",
    "   \\midrule",
    "   \\multicolumn{5}{r}{\\emph{continued on next page}} \\\\",
    "\\endfoot", "\\endlastfoot")
}

append_section <- function(lines, vars, ames, type = "all") {
  for (v in vars) lines <- c(lines, format_coef_row(v, ames, type))
  lines
}

extract_fit <- function(model) {
  list(n = nobs(model), loglik = logLik(model), aic = AIC(model))
}

format_fit_rows <- function(fits) {
  ns <- sapply(fits, function(f) format(f$n, big.mark = ","))
  lls <- sapply(fits, function(f) sprintf("%.1f", as.numeric(f$loglik)))
  aics <- sapply(fits, function(f) sprintf("%.1f", f$aic))
  c("   \\midrule",
    "   \\emph{Fit statistics} & & & & \\\\",
    sprintf("   %-25s & %s & %s & %s & %s \\\\",
            "Model", "CLM", "CLM", "CLM", "CLM"),
    sprintf("   %-25s & %s & %s & %s & %s \\\\",
            "Observations", ns[1], ns[2], ns[3], ns[4]),
    sprintf("   %-25s & %s & %s & %s & %s \\\\",
            "Log-lik.", lls[1], lls[2], lls[3], lls[4]),
    sprintf("   %-25s & %s & %s & %s & %s \\\\",
            "AIC", aics[1], aics[2], aics[3], aics[4]))
}

build_footer <- function() {
  c("   \\midrule \\midrule",
    paste0("   \\multicolumn{5}{l}{\\emph{Average marginal effects",
           " on P(selling first). Cumulative link model (CLM).}} \\\\"),
    paste0("   \\multicolumn{5}{l}{\\emph{All predictors",
           " standardized (z-scored).}} \\\\"),
    paste0("   \\multicolumn{5}{l}{\\emph{Signif. Codes:",
           " ***: 0.01, **: 0.05, *: 0.1}} \\\\"),
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
