# Purpose: Produce pairwise correlation table for the 9 AFFDEX emotion measures
# Author: Caleb Eynon w/ Claude Code
# Date: 2026-03-10
# nolint start
#
# Outputs LaTeX table to analysis/output/tables/:
#   emotion_correlations.tex — lower-triangle correlation matrix with significance stars

library(tidyverse)

# =====
# File paths
# =====
EMOTIONS_PATH <- "datastore/derived/imotions_period_emotions.csv"
OUTPUT_DIR <- "analysis/output/tables"

# =====
# Constants
# =====
EMOTIONS <- c(
  "anger_mean", "contempt_mean", "disgust_mean",
  "fear_mean", "joy_mean", "sadness_mean",
  "surprise_mean", "engagement_mean", "valence_mean"
)
EMOTION_LABELS <- c(
  "Anger", "Contempt", "Disgust",
  "Fear", "Joy", "Sadness",
  "Surprise", "Engagement", "Valence"
)
EMOTION_SHORT <- c(
  "(1)", "(2)", "(3)", "(4)", "(5)",
  "(6)", "(7)", "(8)", "(9)"
)

# =====
# Main
# =====
main <- function() {
  emotions <- read_csv(EMOTIONS_PATH, show_col_types = FALSE)
  emotion_data <- emotions[, EMOTIONS] |> drop_na()
  matrices <- compute_correlation_matrices(emotion_data)
  n_obs <- nrow(emotion_data)
  latex <- build_correlation_latex(matrices$cor, matrices$p, n_obs)
  write_table(latex, "emotion_correlations.tex")
  cat("Emotion correlation table written to", OUTPUT_DIR, "\n")
}

# =====
# Correlation computation
# =====
compute_correlation_matrices <- function(emotion_data) {
  n <- length(EMOTIONS)
  cor_mat <- matrix(NA, n, n)
  p_mat <- matrix(NA, n, n)
  for (i in 1:n) {
    for (j in 1:n) {
      test <- cor.test(emotion_data[[EMOTIONS[i]]], emotion_data[[EMOTIONS[j]]])
      cor_mat[i, j] <- test$estimate
      p_mat[i, j] <- test$p.value
    }
  }
  list(cor = cor_mat, p = p_mat)
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

build_correlation_latex <- function(cor_mat, p_mat, n_obs) {
  n <- nrow(cor_mat)
  col_spec <- paste0("l", paste(rep("c", n), collapse = ""))
  header <- paste0("  & ", paste(EMOTION_SHORT, collapse = " & "), " \\\\")
  rows <- build_data_rows(cor_mat, p_mat)
  assemble_latex(col_spec, header, rows, n_obs)
}

build_data_rows <- function(cor_mat, p_mat) {
  n <- nrow(cor_mat)
  rows <- character(n)
  for (i in 1:n) {
    cells <- build_row_cells(cor_mat, p_mat, i, n)
    rows[i] <- sprintf("  %s %s & %s \\\\",
                        EMOTION_SHORT[i], EMOTION_LABELS[i],
                        paste(cells, collapse = " & "))
  }
  rows
}

build_row_cells <- function(cor_mat, p_mat, i, n) {
  cells <- character(n)
  for (j in 1:n) {
    if (j < i) {
      stars <- format_stars(p_mat[i, j])
      cells[j] <- sprintf("%.2f%s", cor_mat[i, j], stars)
    } else if (j == i) {
      cells[j] <- "1"
    } else {
      cells[j] <- ""
    }
  }
  cells
}

assemble_latex <- function(col_spec, header, rows, n_obs) {
  n_formatted <- gsub(",", "{,}", format(n_obs, big.mark = ","))
  note <- sprintf(
    "{\\footnotesize \\textit{Note:} Pairwise Pearson correlations ($N = %s$). Lower triangle reported.}\\\\",
    n_formatted
  )
  lines <- c(
    "\\begingroup", "\\centering", "\\small",
    sprintf("\\begin{tabular}{%s}", col_spec),
    "\\toprule", header, "\\midrule", rows, "\\bottomrule",
    "\\end{tabular}", "\\par", "\\vspace{2pt}",
    note,
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
