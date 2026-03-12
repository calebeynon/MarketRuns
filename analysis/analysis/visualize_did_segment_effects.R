# Purpose: DiD-style plot decomposing segment effects into learning vs communication
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
  p <- create_did_plot(coef_df)
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
# Model fitting (tobit with segment dummies)
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
  ref <- data.table(segment = 1, estimate = 0, se = 0)
  coef_df <- rbind(ref, coef_df)
  coef_df[, ci_lower := estimate - 1.96 * se]
  coef_df[, ci_upper := estimate + 1.96 * se]
  setorder(coef_df, segment)
}

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
# Counterfactual 1: extrapolate 1->2 learning slope forward
# =====
build_counterfactual_12 <- function(coef_df) {
  slope <- coef_df[segment == 2, estimate]
  data.table(segment = 1:4, cf = slope * (0:3))
}

# =====
# Counterfactual 2: use 3->4 learning slope, extended over full domain
# =====
build_counterfactual_34 <- function(coef_df) {
  seg2 <- coef_df[segment == 2, estimate]
  seg3 <- coef_df[segment == 3, estimate]
  seg4 <- coef_df[segment == 4, estimate]
  slope <- seg4 - seg3
  # Anchor at seg 2; project back to seg 1 and forward to seg 4
  data.table(segment = 1:4, cf = seg2 + slope * (-1:2))
}

# =====
# Counterfactual trend layers
# =====
counterfactual_layers <- function(cf12, cf34) {
  list(
    # CF1 (1->2 slope): dashed post-treatment only
    geom_line(
      data = cf12[segment >= 2], aes(x = segment, y = cf),
      inherit.aes = FALSE, linetype = "dashed", color = "gray50", linewidth = 0.5
    ),
    geom_point(
      data = cf12[segment >= 3], aes(x = segment, y = cf),
      inherit.aes = FALSE, shape = 1, size = 2.5, color = "gray50"
    ),
    # CF2 (3->4 slope): dashed post-treatment only
    geom_line(
      data = cf34[segment >= 2], aes(x = segment, y = cf),
      inherit.aes = FALSE, linetype = "dashed", color = "gray40", linewidth = 0.5
    ),
    geom_point(
      data = cf34[segment >= 3], aes(x = segment, y = cf),
      inherit.aes = FALSE, shape = 2, size = 2.5, color = "gray40"
    )
  )
}

# =====
# Communication effect arrows from each counterfactual to actual seg 3
# =====
communication_arrow <- function(cf12_seg3, cf34_seg3, actual3) {
  list(
    # Arrow from CF1
    annotate(
      "segment", x = 3.08, xend = 3.08, y = cf12_seg3, yend = actual3,
      color = "#CC4400", linewidth = 0.6,
      arrow = arrow(ends = "both", length = unit(0.06, "inches"))
    ),
    # Arrow from CF2
    annotate(
      "segment", x = 2.92, xend = 2.92, y = cf34_seg3, yend = actual3,
      color = "#005599", linewidth = 0.6,
      arrow = arrow(ends = "both", length = unit(0.06, "inches"))
    )
  )
}

# =====
# Actual estimate layers (trajectory, CIs, points)
# =====
actual_layers <- function() {
  list(
    geom_line(color = "black", linewidth = 0.5),
    geom_errorbar(
      aes(ymin = ci_lower, ymax = ci_upper),
      width = 0.12, linewidth = 0.5, color = "gray30"
    ),
    geom_point(size = 2.5, shape = 16)
  )
}

# =====
# Text labels for chat introduction and counterfactuals
# =====
text_labels <- function(cf12_seg4_y, cf34_seg4_y) {
  list(
    annotate(
      "text", x = 2.55, y = -Inf, label = "Chat introduced",
      hjust = 0, vjust = -0.5, size = 2.8, family = "serif",
      fontface = "italic", color = "gray50"
    ),
    annotate(
      "text", x = 4.08, y = cf12_seg4_y,
      label = "CF: 1\u21922\nslope",
      hjust = 0, size = 2.8, family = "serif",
      color = "gray50", lineheight = 0.85
    ),
    annotate(
      "text", x = 4.08, y = cf34_seg4_y,
      label = "CF: 3\u21924\nslope",
      hjust = 0, size = 2.8, family = "serif",
      color = "gray40", lineheight = 0.85
    )
  )
}

# =====
# Assemble the DiD decomposition plot
# =====
create_did_plot <- function(coef_df) {
  cf12 <- build_counterfactual_12(coef_df)
  cf34 <- build_counterfactual_34(coef_df)
  cf12_seg3 <- cf12[segment == 3, cf]
  cf34_seg3 <- cf34[segment == 3, cf]
  a3 <- coef_df[segment == 3, estimate]

  ggplot(coef_df, aes(x = segment, y = estimate)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray70") +
    geom_vline(xintercept = 2.5, linetype = "dotted", color = "gray50") +
    counterfactual_layers(cf12, cf34) +
    communication_arrow(cf12_seg3, cf34_seg3, a3) +
    actual_layers() +
    text_labels(cf12[segment == 4, cf], cf34[segment == 4, cf]) +
    scale_x_continuous(
      breaks = 1:4, labels = paste("Segment", 1:4),
      expand = expansion(mult = c(0.05, 0.2))
    ) +
    labs(x = NULL, y = "Change in number of sellers") +
    theme_econ()
}

# =====
# Save plot
# =====
save_plot <- function(p, path) {
  output_dir <- dirname(path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  ggsave(path, p, width = 7, height = 5)
}

# %%
if (!interactive()) main()
