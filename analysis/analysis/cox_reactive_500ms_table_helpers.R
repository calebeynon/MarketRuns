# Purpose: 2-column LaTeX table helpers for reactive sellers 500ms Cox table (issue #118)
# Author: Claude Code
# Date: 2026-04-21

# Libraries loaded by main script (do not load here)

# =====
# Variable order for the 2-column table (cascade/interactions removed — these
# covariates were dropped in the respecified Panel R model because
# reactive_sale==1 implies prior_group_sales>=1, inducing collinearity).
# Traits are excluded because neither displayed model uses them.
# =====
get_var_order_reactive <- function() {
  c("signal", DISCRETE_EMOTIONS, "valence_mean", COX_CONTROLS)
}

# =====
# Cluster-robust coxph coefficient extractor (mirrors extract_cox_coefs
# but uses coef()/vcov() directly; cluster-robust vcov is already baked in
# when coxph was fit with cluster=).
# =====
extract_coxph_coefs <- function(model) {
  beta <- coef(model)
  v <- as.matrix(vcov(model))
  hr <- exp(beta)
  hr_se <- hr * sqrt(diag(v))
  z <- beta / sqrt(diag(v))
  pval <- 2 * pnorm(-abs(z))
  nms <- normalize_cox_names(names(beta))
  data.table(var = nms, est = hr, se = hr_se, pval = pval)
}

# =====
# Fit statistics for coxph (no random-effects groups)
# =====
extract_coxph_fit <- function(model) {
  events <- model$nevent
  n <- model$n
  ll <- model$loglik[2]
  list(n = n, events = events, n_groups = NA_integer_, log_lik = ll)
}

# =====
# 2-column coefficient row formatting
# =====
format_cox_coef_row_2col <- function(var_name, coefs_list) {
  label <- if (var_name %in% names(VAR_LABELS)) {
    VAR_LABELS[var_name]
  } else {
    gsub("_", "\\\\_", var_name)
  }
  cells <- format_cox_coef_cells_2col(var_name, coefs_list)
  c(sprintf("   %-25s & %s & %s \\\\",
            label, cells$v[1], cells$v[2]),
    sprintf("   %-25s & %s & %s \\\\",
            "", cells$s[1], cells$s[2]))
}

format_cox_coef_cells_2col <- function(var_name, coefs_list) {
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
# 2-column table builder
# =====
build_cox_table_2col <- function(re_cox, cluster_cox) {
  coefs <- list(extract_cox_coefs(re_cox),
                extract_coxph_coefs(cluster_cox))
  fits <- list(extract_cox_fit(re_cox),
               extract_coxph_fit(cluster_cox))
  lines <- c(build_preamble_2col(), build_col_header_2col())
  for (v in get_var_order_reactive()) {
    lines <- c(lines, format_cox_coef_row_2col(v, coefs))
  }
  c(lines, format_cox_fit_rows_2col(fits), build_footer_2col())
}

# =====
# Preamble: longtable wrapper, caption, label
# =====
build_preamble_2col <- function() {
  caption <- paste0("Cox survival regression on reactive sellers",
                    " (500ms pre-click emotion window)")
  c("", "\\begingroup", "\\centering", "\\scriptsize",
    "\\setlength{\\LTcapwidth}{\\textwidth}",
    paste0("\\begin{longtable}{l",
           "*{2}{>{\\centering\\arraybackslash}p{3.0cm}}}"),
    sprintf("\\caption{%s} \\label{tab:cox_survival_reactive_500ms} \\\\",
            caption))
}

# =====
# Column header: RE Cox vs Clustered Cox
# =====
build_col_header_2col <- function() {
  hdr <- c(
    "   \\midrule \\midrule",
    "   & (1) & (2) \\\\",
    "   & RE Cox & Clustered Cox \\\\",
    "   \\midrule")
  c(hdr, "\\endfirsthead",
    "\\multicolumn{3}{l}{\\emph{(continued)}} \\\\",
    hdr, "\\endhead",
    "   \\midrule",
    "   \\multicolumn{3}{r}{\\emph{continued on next page}} \\\\",
    "\\endfoot", "\\endlastfoot")
}

# =====
# Fit statistic rows. Participants shown only for RE Cox; "—" for coxph.
# =====
format_cox_fit_rows_2col <- function(fits) {
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
    sprintf("   %-25s & %s & %s \\\\", "Participants", ngs[1], ngs[2]),
    sprintf("   %-25s & %s & %s \\\\", "Log-likelihood", lls[1], lls[2]))
}

# =====
# Footer note describing the reactive-sale event
# =====
build_footer_2col <- function() {
  c("   \\midrule \\midrule",
    paste0("   \\multicolumn{3}{l}{\\emph{Hazard ratios reported.",
           " Column (1): random-intercept Cox (coxme) with player random effect.",
           " Column (2): Cox (coxph) with group-clustered robust SEs.}} \\\\"),
    paste0("   \\multicolumn{3}{l}{\\emph{Event: reactive sale",
           " (sold in period immediately after a group-mate's sale).",
           " Non-reactive sales included in risk set as non-events.}} \\\\"),
    paste0("   \\multicolumn{3}{l}{\\emph{HR $>$ 1: increased hazard of",
           " reactive sale. HR $<$ 1: decreased hazard.}} \\\\"),
    paste0("   \\multicolumn{3}{l}{\\emph{Signif. Codes:",
           " ***: 0.01, **: 0.05, *: 0.1}} \\\\"),
    "\\end{longtable}", "\\par\\endgroup", "", "")
}
