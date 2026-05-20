# Purpose: 4-column LaTeX table helpers for the selling-behavior Cox table (issue #124)
# Author: Claude Code
# Date: 2026-05-20

# Libraries loaded by main script (do not load here)
# Reuses VAR_LABELS, get_stars (selling_regression_helpers.R), DISCRETE_EMOTIONS,
# COX_CONTROLS, COX_TRAITS, INTERACTION_VARS (cox_survival_regression.R + helpers),
# extract_cox_coefs / extract_cox_fit (coxme) and extract_coxph_coefs /
# extract_coxph_fit (clustered coxph).

# =====
# Unified variable order spanning every coefficient any of the four models
# can report: cascade + interactions + signal + 8 emotions + valence +
# controls + traits. Absent coefficients render as blank cells.
# =====
get_var_order_four_column <- function() {
  cascade <- c("dummy_1_cum", "dummy_2_cum", "dummy_3_cum")
  c(cascade, INTERACTION_VARS, "signal", DISCRETE_EMOTIONS,
    "valence_mean", COX_CONTROLS, COX_TRAITS)
}

# =====
# Assemble the flat 4-column longtable. Each model carries its own coefficient
# extractor + fit extractor (coxme vs clustered coxph) supplied by the caller.
# =====
build_cox_table_four_column <- function(models) {
  coefs <- lapply(models, function(m) m$coef_fn(m$fit))
  fits <- lapply(models, function(m) m$fit_fn(m$fit))
  lines <- c(build_preamble_4col(), build_col_header_4col())
  for (v in get_var_order_four_column()) {
    lines <- c(lines, format_cox_coef_row_4col(v, coefs))
  }
  c(lines, format_cox_fit_rows_4col(fits), build_footer_4col())
}

# =====
# 4-column coefficient row formatting (estimate row + SE row)
# =====
format_cox_coef_row_4col <- function(var_name, coefs_list) {
  label <- if (var_name %in% names(VAR_LABELS)) {
    VAR_LABELS[var_name]
  } else {
    gsub("_", "\\\\_", var_name)
  }
  cells <- format_cox_coef_cells_4col(var_name, coefs_list)
  c(sprintf("   %-25s & %s & %s & %s & %s \\\\",
            label, cells$v[1], cells$v[2], cells$v[3], cells$v[4]),
    sprintf("   %-25s & %s & %s & %s & %s \\\\",
            "", cells$s[1], cells$s[2], cells$s[3], cells$s[4]))
}

format_cox_coef_cells_4col <- function(var_name, coefs_list) {
  vals <- ses <- character(4)
  for (i in seq_along(coefs_list)) {
    row <- coefs_list[[i]][var == var_name]
    if (nrow(row) == 0) next
    if (is_non_identified(var_name, row$est, row$se)) {
      vals[i] <- NON_IDENTIFIED_MARK
      next
    }
    vals[i] <- paste0(sprintf("%.4f", row$est), get_stars(row$pval))
    ses[i] <- paste0("(", sprintf("%.4f", row$se), ")")
  }
  list(v = vals, s = ses)
}

# =====
# Detect hazard-ratio cells that are non-identified due to (near-)complete
# separation in sparse cascade-interaction cells. Two signatures: (a) the HR
# is driven to the 0/Inf boundary (rounds to 0.0000 or is astronomically
# large, or the log-HR SE is implausibly wide); (b) a 3-prior-sale interaction
# whose HR blows up above 5 — these terms have only a handful of events and
# their estimates are separation artifacts, not informative effects. The rule
# is conservative: legitimate wide estimates (e.g. cascade dummies, the 1- and
# 2-prior interactions) print normally.
# =====
NON_IDENTIFIED_MARK <- "n.i."

is_non_identified <- function(var_name, est, se) {
  log_hr_se <- se / est
  boundary <- est < 5e-5 | est > 1e4 |
    !is.finite(log_hr_se) | log_hr_se > 50
  sparse_interaction <- var_name %in% INTERACTION_VARS & est > 5
  boundary | sparse_interaction
}

# =====
# Preamble: longtable wrapper, caption, label
# =====
build_preamble_4col <- function() {
  caption <- paste0("Cox survival regression of selling behavior",
                    " across risk sets (hazard ratios)")
  c("", "\\begingroup", "\\centering", "\\scriptsize",
    "\\setlength{\\LTcapwidth}{\\textwidth}",
    paste0("\\begin{longtable}{l",
           "*{4}{>{\\centering\\arraybackslash}p{2.2cm}}}"),
    sprintf("\\caption{%s} \\label{tab:cox_selling_four_column} \\\\",
            caption))
}

# =====
# Column header: First sellers | Reactive | All sellers | All participants
# =====
build_col_header_4col <- function() {
  hdr <- c(
    "   \\midrule \\midrule",
    "   & (1) & (2) & (3) & (4) \\\\",
    paste0("   & First sellers & Reactive & All sellers",
           " & All participants \\\\"),
    "   \\midrule")
  c(hdr, "\\endfirsthead",
    "\\multicolumn{5}{l}{\\emph{(continued)}} \\\\",
    hdr, "\\endhead",
    "   \\midrule",
    "   \\multicolumn{5}{r}{\\emph{continued on next page}} \\\\",
    "\\endfoot", "\\endlastfoot")
}

# =====
# Fit-statistic rows. The reactive column is clustered coxph: its random-effect
# participant count is NA, so we report its cluster count instead via "---".
# =====
format_cox_fit_rows_4col <- function(fits) {
  ns <- sapply(fits, function(f) format(f$n, big.mark = ","))
  evts <- sapply(fits, function(f) format(f$events, big.mark = ","))
  ngs <- sapply(fits, function(f) {
    if (is.na(f$n_groups)) "---" else format(f$n_groups, big.mark = ",")
  })
  lls <- sapply(fits, function(f) sprintf("%.1f", f$log_lik))
  c("   \\midrule",
    "   \\emph{Fit statistics} & & & & \\\\",
    sprintf("   %-25s & %s & %s & %s & %s \\\\",
            "Observations", ns[1], ns[2], ns[3], ns[4]),
    sprintf("   %-25s & %s & %s & %s & %s \\\\",
            "Events", evts[1], evts[2], evts[3], evts[4]),
    sprintf("   %-25s & %s & %s & %s & %s \\\\",
            "Participants", ngs[1], ngs[2], ngs[3], ngs[4]),
    sprintf("   %-25s & %s & %s & %s & %s \\\\",
            "Log-likelihood", lls[1], lls[2], lls[3], lls[4]))
}

# =====
# Footer notes documenting each column's sample and estimator
# =====
build_footer_4col <- function() {
  c("   \\midrule \\midrule",
    paste0("   \\multicolumn{5}{l}{\\emph{Hazard ratios reported. All",
           " specifications include emotions, valence, and personality",
           " traits.}} \\\\"),
    footer_estimator_note_4col(),
    footer_sample_note_4col(),
    footer_caveat_note_4col(),
    paste0("   \\multicolumn{5}{l}{\\emph{HR $>$ 1: increased hazard of",
           " selling. HR $<$ 1: decreased hazard.}} \\\\"),
    paste0("   \\multicolumn{5}{l}{\\emph{Signif. Codes:",
           " ***: 0.01, **: 0.05, *: 0.1}} \\\\"),
    "\\end{longtable}", "\\par\\endgroup", "", "")
}

footer_estimator_note_4col <- function() {
  paste0("   \\multicolumn{5}{p{0.95\\textwidth}}{\\emph{Columns (1),",
         " (3), (4): random-intercept Cox (coxme) with a player random",
         " effect; the Participants row counts random-effect groups.",
         " Column (2): Cox (coxph) with cluster-robust SEs (clustered",
         " at session $\\times$ segment $\\times$ group); the Participants",
         " row reports the cluster count.}} \\\\")
}

footer_sample_note_4col <- function() {
  paste0("   \\multicolumn{5}{p{0.95\\textwidth}}{\\emph{Samples:",
         " (1) first sellers (first sale within each group-round);",
         " (2) reactive sellers (rows with group\\_sold\\_prev\\_period",
         " $=$ 1, 500ms pre-click emotion window); (3) all sellers",
         " (at-risk periods of players who sold --- each",
         " player-group-round ending in that player's sale);",
         " (4) all participants (full risk set). Cascade and interaction",
         " rows are blank where the sample's design omits them.}} \\\\")
}

# =====
# FIX 2: estimators differ across columns, so cross-column significance is not
# directly comparable (subject-specific coxme HRs vs marginal coxph HRs;
# model-based vs cluster-robust SEs). FIX 3: ``n.i.'' marks separation cells.
# =====
footer_caveat_note_4col <- function() {
  paste0("   \\multicolumn{5}{p{0.95\\textwidth}}{\\emph{SEs are",
         " model-based for the coxme columns (1, 3, 4) and cluster-robust",
         " for the reactive coxph column (2); hazard ratios are conditional",
         " (subject-specific) for coxme vs marginal for coxph, so",
         " cross-column significance is not directly comparable.",
         " ``n.i.'' marks a coefficient that is not identified due to",
         " (near-)complete separation in sparse cascade-interaction",
         " cells.}} \\\\")
}
