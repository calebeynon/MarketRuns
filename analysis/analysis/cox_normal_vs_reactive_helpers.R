# Purpose: 2-column table helpers — Normal vs Reactive-500ms Cox (issue #118)
# Author: Claude Code
# Date: 2026-04-21

# Libraries loaded by main script (do not load here)

# =====
# Variable order: cascade + interactions + signal + emotions + controls.
# Cascade and interaction rows will be blank in the reactive column because
# reactive_sale == 1 requires prior group sales >= 1 (mechanical collinearity).
# Traits omitted because neither displayed model uses them.
# =====
get_var_order_normal_vs_reactive <- function() {
  cascade <- c("dummy_1_cum", "dummy_2_cum", "dummy_3_cum")
  c(cascade, INTERACTION_VARS, "signal",
    DISCRETE_EMOTIONS, "valence_mean", COX_CONTROLS)
}

# =====
# 2-column row formatting (reused from cox_reactive_500ms_table_helpers.R
# is not sourced here; inlined to keep this script self-contained for the
# normal-vs-reactive table).
# =====
format_nvr_coef_row <- function(var_name, coefs_list) {
  label <- if (var_name %in% names(VAR_LABELS)) {
    VAR_LABELS[var_name]
  } else {
    gsub("_", "\\\\_", var_name)
  }
  cells <- format_nvr_coef_cells(var_name, coefs_list)
  c(sprintf("   %-25s & %s & %s \\\\",
            label, cells$v[1], cells$v[2]),
    sprintf("   %-25s & %s & %s \\\\",
            "", cells$s[1], cells$s[2]))
}

format_nvr_coef_cells <- function(var_name, coefs_list) {
  vals <- ses <- character(2)
  for (i in seq_along(coefs_list)) {
    row <- coefs_list[[i]][var == var_name]
    if (nrow(row) == 0) next
    vals[i] <- paste0(sprintf("%.4f", row$est), get_stars(row$pval))
    ses[i] <- paste0("(", sprintf("%.4f", row$se), ")")
  }
  list(v = vals, s = ses)
}

# =====
# Assemble the full 2-column LaTeX table (both columns are clustered coxph;
# extract_coxph_coefs sourced from 500ms helpers; cluster count read from
# the "nclusters" attribute attached in the main script).
# =====
build_cox_table_normal_vs_reactive <- function(normal_model, reactive_model) {
  coefs <- list(extract_coxph_coefs(normal_model),
                extract_coxph_coefs(reactive_model))
  fits <- list(extract_coxph_fit_nvr(normal_model),
               extract_coxph_fit_nvr(reactive_model))
  lines <- c(build_preamble_nvr(), build_col_header_nvr())
  for (v in get_var_order_normal_vs_reactive()) {
    lines <- c(lines, format_nvr_coef_row(v, coefs))
  }
  c(lines, format_fit_rows_nvr(fits), build_footer_nvr())
}

# =====
# Cluster count lives on the fitted model as attr(m, "nclusters"), set by
# the caller. Read it here into the fit list's n_groups slot.
# =====
extract_coxph_fit_nvr <- function(model) {
  base <- extract_coxph_fit(model)
  nc <- attr(model, "nclusters")
  base$n_groups <- if (is.null(nc)) NA_integer_ else nc
  base
}

# =====
# Preamble: table float + minipage + tabular*; @{\extracolsep{\fill}} spreads
# the two data columns across \textwidth (no right-side whitespace).
# \arraystretch compacts rows so the whole table fits on one page.
# =====
build_preamble_nvr <- function() {
  caption <- paste0("Cox survival regression: all-sellers (normal)",
                    " vs.\\ reactive sellers (500ms pre-click window)")
  c("", "\\begin{table}[!htbp]", "\\centering",
    sprintf("\\caption{%s} \\label{tab:cox_normal_vs_reactive}", caption),
    "\\begin{minipage}{\\textwidth}", "\\centering",
    "\\scriptsize", "\\renewcommand{\\arraystretch}{0.9}",
    paste0("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}",
           " l c c @{\\hspace{4em}}}"))
}

# =====
# Column header: two models identified by sample (all vs reactive)
# =====
build_col_header_nvr <- function() {
  c("   \\toprule \\toprule",
    "   & (1) & (2) \\\\",
    "   & Normal & Reactive \\\\",
    "   & All sellers & 500ms pre-click \\\\",
    "   \\midrule")
}

# =====
# Fit-statistic rows (Observations / Events / Participants / Log-likelihood)
# =====
format_fit_rows_nvr <- function(fits) {
  ns <- sapply(fits, function(f) format(f$n, big.mark = ","))
  evts <- sapply(fits, function(f) format(f$events, big.mark = ","))
  ngs <- sapply(fits, function(f) {
    if (is.na(f$n_groups)) "---" else format(f$n_groups, big.mark = ",")
  })
  lls <- sapply(fits, function(f) sprintf("%.1f", f$log_lik))
  c("   \\midrule",
    "   \\emph{Fit statistics} & & \\\\",
    sprintf("   %-25s & %s & %s \\\\", "Observations", ns[1], ns[2]),
    sprintf("   %-25s & %s & %s \\\\", "Events", evts[1], evts[2]),
    sprintf("   %-25s & %s & %s \\\\", "Clusters", ngs[1], ngs[2]),
    sprintf("   %-25s & %s & %s \\\\", "Log-likelihood", lls[1], lls[2]),
    "   \\bottomrule \\bottomrule")
}

# =====
# Footer: close the tabular*, then drop the note block into a separate
# minipage beneath it. Paragraph text wraps naturally to \textwidth and the
# table body is no longer constrained to fit the notes.
# =====
build_footer_nvr <- function() {
  c("\\end{tabular*}",
    "\\vspace{0.4em}",
    "\\begin{minipage}{\\textwidth}",
    "\\footnotesize",
    build_footer_notes_nvr(),
    "\\end{minipage}",
    "\\end{minipage}",
    "\\end{table}", "", "")
}

build_footer_notes_nvr <- function() {
  c(paste0("\\emph{Notes}: Hazard ratios reported. Both columns: Cox ",
           "(coxph) with cluster-robust standard errors, clustered at the ",
           "session $\\times$ segment $\\times$ group level ",
           "(global\\_group\\_id). Column (1) event = any sale; sample = ",
           "full emotions+traits dataset. Column (2) event = reactive sale ",
           "(sold in the period immediately after a group-mate's sale); ",
           "sample = 500ms pre-click emotion window. HR $>$ 1: increased ",
           "hazard; HR $<$ 1: decreased hazard. Cascade and cumulative-by-",
           "previous-period interaction terms are omitted from Column (2) ",
           "because reactive\\_sale $=$ 1 implies prior group sales ",
           "$\\geq 1$ (mechanical collinearity)."),
    "",
    "\\emph{Signif.\\ Codes}: ***: 0.01, **: 0.05, *: 0.1.")
}
