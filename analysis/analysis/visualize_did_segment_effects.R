# Purpose: Event-study plot of segment coefficients from tobit model (Table 5 Model 3)
# Author: Caleb Eynon w/ Claude Code
# Date: 2026-03-10

library(data.table)
library(AER)
library(sandwich)
library(lmtest)
library(ggplot2)

# FILE PATHS
INPUT_PATH <- "datastore/derived/group_round_timing.csv"
OUTPUT_PATH <- "analysis/output/plots/did_segment_effects.pdf"

# =====
# Main function
# =====
main <- function() {
  dt <- load_and_prepare(INPUT_PATH)
  model <- fit_tobit(dt)
  coef_df <- extract_segment_coefs(model, dt)
  p <- create_event_study_plot(coef_df)
  save_plot(p, OUTPUT_PATH)
  cat("Done. Saved to:", OUTPUT_PATH, "\n")
}

# =====
# Data loading and preparation
# =====
load_and_prepare <- function(path) {
  dt <- fread(path)
  dt[, bad_state := as.integer(state == 0)]
  dt[, segment_num := factor(segment_num, levels = 1:4)]
  dt[, treatment := factor(treatment, levels = c(1, 2))]
  dt
}

# =====
# Model fitting (Table 5 Model 3 specification)
# =====
fit_tobit <- function(dt) {
  tobit(n_sellers ~ bad_state + treatment + segment_num + round_num,
        left = 0, right = 4, data = dt)
}

# =====
# Extract segment coefficients with cluster-robust SEs
# =====
extract_segment_coefs <- function(model, dt) {
  cl_vcov <- vcovCL(model, cluster = dt$global_group_id)
  ct <- coeftest(model, vcov. = cl_vcov)
  coef_df <- build_segment_table(ct)
  cat("Segment coefficients:\n")
  print(coef_df)
  coef_df
}

build_segment_table <- function(ct) {
  seg_names <- c("segment_num2", "segment_num3", "segment_num4")
  coef_df <- data.table(
    segment = 2:4,
    estimate = ct[seg_names, 1],
    se = ct[seg_names, 2]
  )
  # Add segment 1 as reference (coefficient = 0, no uncertainty)
  ref <- data.table(segment = 1, estimate = 0, se = 0)
  coef_df <- rbind(ref, coef_df)
  coef_df[, ci_lower := estimate - 1.96 * se]
  coef_df[, ci_upper := estimate + 1.96 * se]
  setorder(coef_df, segment)
}

# =====
# Theme for economics papers (serif, minimal grid)
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
# Build annotations for segment transitions
# =====
build_annotations <- function(coef_df) {
  # Y position for annotations: above the highest CI
  y_top <- max(coef_df$ci_upper) + 0.15 * diff(range(coef_df$ci_lower, coef_df$ci_upper))
  y_text <- y_top + 0.08 * diff(range(coef_df$ci_lower, coef_df$ci_upper))

  data.table(
    x = c(1.5, 2.5, 3.5),
    xstart = c(1, 2, 3),
    xend = c(2, 3, 4),
    y = y_top,
    y_text = y_text,
    label = c("Learning", "Learning +\nCommunication", "Learning")
  )
}

# =====
# Create event-study coefficient plot
# =====
create_event_study_plot <- function(coef_df) {
  ann <- build_annotations(coef_df)

  p <- ggplot(coef_df, aes(x = segment, y = estimate)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_vline(xintercept = 2.5, linetype = "dashed", color = "gray40") +
    geom_errorbar(
      aes(ymin = ci_lower, ymax = ci_upper),
      width = 0.15, linewidth = 0.5, color = "gray30"
    ) +
    geom_point(size = 2.5, color = "black", shape = 16) +
    add_transition_annotations(ann) +
    scale_x_continuous(breaks = 1:4, labels = paste("Segment", 1:4)) +
    labs(x = NULL, y = "Coefficient (relative to Segment 1)") +
    theme_econ()

  p
}

# =====
# Add bracket annotations for segment transitions
# =====
add_transition_annotations <- function(ann) {
  list(
    bracket_segments(ann),
    bracket_ticks(ann, "xstart"),
    bracket_ticks(ann, "xend"),
    bracket_labels(ann),
    chat_label()
  )
}

bracket_segments <- function(ann) {
  geom_segment(
    data = ann,
    aes(x = xstart, xend = xend, y = y, yend = y),
    inherit.aes = FALSE, linewidth = 0.4, color = "gray30"
  )
}

bracket_ticks <- function(ann, col) {
  geom_segment(
    data = ann,
    aes(x = .data[[col]], xend = .data[[col]],
        y = y - 0.03, yend = y),
    inherit.aes = FALSE, linewidth = 0.4, color = "gray30"
  )
}

bracket_labels <- function(ann) {
  geom_text(
    data = ann,
    aes(x = x, y = y_text, label = label),
    inherit.aes = FALSE, size = 3, family = "serif", lineheight = 0.85
  )
}

chat_label <- function() {
  annotate(
    "text", x = 2.55, y = -Inf, label = "Chat introduced",
    hjust = 0, vjust = -0.5, size = 2.8, family = "serif",
    fontface = "italic", color = "gray30"
  )
}

# =====
# Save plot as PDF
# =====
save_plot <- function(p, path) {
  output_dir <- dirname(path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  ggsave(path, p, width = 7, height = 5)
}

# %%
if (!interactive()) main()
