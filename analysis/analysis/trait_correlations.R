# Purpose: Produce pairwise correlation table for the 8 trait measures
# Author: Caleb Eynon w/ Claude Code
# Date: 2026-02-20
# nolint start
#
# Outputs LaTeX table to analysis/output/tables/:
#   trait_correlations.tex — lower-triangle correlation matrix with significance stars

library(tidyverse)

# =====
# File paths
# =====
SURVEY_PATH <- "datastore/derived/survey_traits.csv"
OUTPUT_DIR <- "analysis/output/tables"

# =====
# Constants
# =====
TRAITS <- c(
  "extraversion", "agreeableness", "conscientiousness",
  "neuroticism", "openness", "impulsivity", "state_anxiety",
  "risky_investment"
)
TRAIT_LABELS <- c(
  "Extraversion", "Agreeableness", "Conscientiousness",
  "Neuroticism", "Openness", "Impulsivity", "State Anxiety",
  "Risky Investment"
)
TRAIT_SHORT <- c("(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)")

# =====
# Main
# =====
main <- function() {
  survey <- read_csv(SURVEY_PATH, show_col_types = FALSE)
  trait_data <- survey[, TRAITS]

  # Compute correlation matrix and p-value matrix
  n <- length(TRAITS)
  cor_mat <- matrix(NA, n, n)
  p_mat <- matrix(NA, n, n)

  for (i in 1:n) {
    for (j in 1:n) {
      test <- cor.test(trait_data[[TRAITS[i]]], trait_data[[TRAITS[j]]])
      cor_mat[i, j] <- test$estimate
      p_mat[i, j] <- test$p.value
    }
  }

  # Build LaTeX table (lower triangle only)
  latex <- build_correlation_latex(cor_mat, p_mat)
  write_table(latex, "trait_correlations.tex")
  cat("Trait correlation table written to", OUTPUT_DIR, "\n")
}

# =====
# LaTeX table builder
# =====
format_stars <- function(p) {
  if (p < 0.01) return("***")
  if (p < 0.05) return("**")
  if (p < 0.1) return("*")
  return("")
}

build_correlation_latex <- function(cor_mat, p_mat) {
  n <- nrow(cor_mat)

  # Column alignment: label + n numeric columns
  col_spec <- paste0("l", paste(rep("c", n), collapse = ""))

  # Header row with short labels
  header <- paste0("  & ", paste(TRAIT_SHORT, collapse = " & "), " \\\\")

  # Build each row: label, then lower-triangle entries, diagonal = 1
  rows <- character(n)
  for (i in 1:n) {
    cells <- character(n)
    for (j in 1:n) {
      if (j < i) {
        # Lower triangle: show correlation with stars
        stars <- format_stars(p_mat[i, j])
        cells[j] <- sprintf("%.2f%s", cor_mat[i, j], stars)
      } else if (j == i) {
        cells[j] <- "1"
      } else {
        # Upper triangle: leave blank
        cells[j] <- ""
      }
    }
    rows[i] <- sprintf("  %s %s & %s \\\\",
                        TRAIT_SHORT[i], TRAIT_LABELS[i],
                        paste(cells, collapse = " & "))
  }

  # Assemble full table
  lines <- c(
    "\\begingroup", "\\centering", "\\small",
    sprintf("\\begin{tabular}{%s}", col_spec),
    "\\toprule",
    header,
    "\\midrule",
    rows,
    "\\bottomrule",
    "\\end{tabular}",
    "\\par", "\\vspace{2pt}",
    "{\\footnotesize \\textit{Note:} Pairwise Pearson correlations ($N = 95$). Lower triangle reported.}\\\\",
    "{\\footnotesize ***: $p<0.01$, **: $p<0.05$, *: $p<0.1$}",
    "\\endgroup"
  )

  paste(lines, collapse = "\n")
}

# =====
# Shared helpers
# =====
write_table <- function(content, filename) {
  dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
  path <- file.path(OUTPUT_DIR, filename)
  writeLines(content, path)
  cat("Written:", path, "\n")
}

# %%
if (sys.nframe() == 0) main()
