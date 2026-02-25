# Purpose: Grouped bar chart of mean VADER compound sentiment by treatment x segment
# Author: Claude
# Date: 2026-02-25

library(data.table)
library(ggplot2)

# FILE PATHS
INPUT_PATH <- "datastore/derived/chat_sentiment_dataset.csv"
OUTPUT_PATH <- "analysis/output/plots/chat_sentiment_by_treatment.pdf"

# =====
# Theme for economics papers
# =====
theme_econ <- function() {
  theme_minimal() +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "gray90"),
      text = element_text(family = "serif"),
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 11)
    )
}

# =====
# Main function
# =====
main <- function() {
  cat("Loading data from:", INPUT_PATH, "\n")
  df <- fread(INPUT_PATH)

  cat("Observations:", nrow(df), "\n")
  summary_dt <- compute_summary(df)
  cat("Group means:\n")
  print(summary_dt)

  p <- create_bar_chart(summary_dt)

  cat("Saving to:", OUTPUT_PATH, "\n")
  save_plot(p, OUTPUT_PATH)
  cat("Done!\n")
}

# =====
# Compute mean and SE by treatment x segment
# =====
compute_summary <- function(df) {
  summary_dt <- df[, .(
    mean_sentiment = mean(vader_compound_mean),
    se_sentiment = sd(vader_compound_mean) / sqrt(.N),
    n = .N
  ), by = .(treatment, segment)]

  # Readable labels
  summary_dt[, treatment_label := fifelse(
    treatment == "tr1", "Treatment 1", "Treatment 2"
  )]
  summary_dt[, segment_label := paste("Segment", segment)]
  return(summary_dt)
}

# =====
# Grouped bar chart with error bars
# =====
create_bar_chart <- function(summary_dt) {
  dodge <- position_dodge(0.7)
  p <- ggplot(summary_dt, aes(
    x = treatment_label, y = mean_sentiment, fill = segment_label
  )) +
    geom_col(position = dodge, width = 0.6) +
    geom_errorbar(aes(
      ymin = mean_sentiment - se_sentiment,
      ymax = mean_sentiment + se_sentiment
    ), position = dodge, width = 0.2, linewidth = 0.5) +
    scale_fill_manual(values = c("gray70", "gray30")) +
    labs(x = "Treatment", y = "Mean VADER Compound Sentiment", fill = NULL) +
    theme_econ() +
    theme(legend.position = "bottom")
  return(p)
}

# =====
# Save as PDF (vector graphics for publication)
# =====
save_plot <- function(p, output_path) {
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  ggsave(output_path, p, width = 7, height = 5)
}

# %%
if (!interactive()) main()
