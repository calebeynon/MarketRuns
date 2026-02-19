# Purpose: Two-column table builder for compact unified selling regression
# Author: Claude Code
# Date: 2026-02-18

# Globals and helpers from main script are available via source()

# =====
# Main two-column table builder
# =====
build_landscape_table <- function(panel_a, panel_b, panel_c) {
  vars <- get_panel_vars(compact = TRUE)
  tbl <- get_table_config(compact = TRUE)
  left <- build_col_panel("Panel A: All Participants",
                          panel_a, vars$a, wide = TRUE)
  right_b <- build_col_panel("Panel B: Second Sellers",
                             panel_b, vars$b, wide = FALSE)
  right_c <- build_col_panel("Panel C: First Sellers",
                             panel_c, vars$c, wide = FALSE)
  lines <- build_twocol_wrapper(tbl, left, right_b, right_c)
  write_table(lines, tbl$path)
}

# =====
# Wrapper (Panel A left, B+C stacked right)
# =====
build_twocol_wrapper <- function(tbl, left, right_b, right_c) {
  c("",
    build_table_header(tbl),
    build_minipages(left, right_b, right_c),
    "",
    build_twocol_footer(),
    "\\end{table}", "", "")
}

build_table_header <- function(tbl) {
  c("\\begin{table}[H]",
    "\\singlespacing",
    "\\scriptsize",
    "\\setlength{\\tabcolsep}{2pt}",
    "\\renewcommand{\\arraystretch}{0.85}",
    sprintf("\\caption{%s}", tbl$caption),
    sprintf("\\label{tab:%s}", tbl$label),
    "\\centering")
}

build_minipages <- function(left, right_b, right_c) {
  c("\\begin{minipage}[t]{0.54\\linewidth}",
    left,
    "\\end{minipage}%",
    "\\hfill",
    "\\begin{minipage}[t]{0.44\\linewidth}",
    right_b,
    "\\vspace{0.5em}",
    right_c,
    "\\end{minipage}")
}

# =====
# Column panel tabular (with [t] for top alignment)
# =====
build_col_panel <- function(title, models, var_order, wide) {
  coefs <- lapply(models, extract_coefs)
  fits <- lapply(models, extract_fit)
  col_spec <- get_col_spec(wide)
  lines <- c(
    sprintf("\\begin{tabular}[t]{%s}", col_spec),
    sprintf("\\multicolumn{4}{@{}l}{\\emph{%s}} \\\\", title),
    "   \\midrule \\midrule",
    "   & (1) & (2) & (3) \\\\",
    "   & RE & Indiv. FE & RE \\\\",
    "   \\midrule")
  lines <- append_coef_rows(lines, var_order, coefs)
  lines <- c(lines, format_fit_rows(fits))
  c(lines, "   \\midrule", "\\end{tabular}")
}

get_col_spec <- function(wide) {
  lbl <- if (wide) "p{3.5cm}" else "p{2.3cm}"
  dcol <- if (wide) "p{1.4cm}" else "p{1.3cm}"
  dcol_full <- paste0(">{\\centering\\arraybackslash}", dcol)
  paste0(">{\\raggedright\\arraybackslash}", lbl,
         "*{3}{", dcol_full, "}")
}

append_coef_rows <- function(lines, var_order, coefs) {
  for (v in var_order) {
    if (startsWith(v, "__header__")) {
      label <- sub("__header__", "", v)
      lines <- c(lines, sprintf("   \\emph{%s} & & & \\\\", label))
    } else {
      lines <- c(lines, format_coef_row(v, coefs))
    }
  }
  lines
}

# =====
# Footer note
# =====
build_twocol_footer <- function() {
  note <- paste0(
    "\\footnotesize ",
    "\\emph{Controls: signal, period, round, segment indicators, ",
    "age, gender. Additional emotion controls (contempt, disgust, ",
    "joy, sadness, surprise, engagement, valence) and personality ",
    "traits (extraversion, agreeableness, neuroticism, openness) ",
    "included but not displayed. ",
    "Full results in Appendix ",
    "Table~\\ref{tab:unified_selling_regression_full}.}")
  c("\\vspace{0.5em}",
    note,
    "",
    paste0("\\footnotesize ",
           "\\emph{Signif. Codes: ***: 0.01, **: 0.05, *: 0.1}"))
}
