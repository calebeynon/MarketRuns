# Purpose: Create descriptive visualizations of trait distributions by first seller status
# Author: Claude Code
# Date: 2026-02-01

library(data.table)
library(ggplot2)

# =====
# File paths
# =====
INPUT_PATH <- "datastore/derived/first_seller_analysis_data.csv"
OUTPUT_DIR <- "analysis/output/plots"
OUTPUT_BOXPLOTS <- file.path(OUTPUT_DIR, "first_seller_trait_boxplots.pdf")
OUTPUT_VIOLINS <- file.path(OUTPUT_DIR, "first_seller_trait_violins.pdf")
OUTPUT_QUARTILE <- file.path(OUTPUT_DIR, "first_seller_rate_by_quartile.pdf")

# Traits to visualize
TRAITS <- c(
  "extraversion", "agreeableness", "conscientiousness",
  "neuroticism", "openness", "impulsivity", "state_anxiety"
)

# Key traits for quartile analysis (hypothesis-driven)
KEY_TRAITS <- c("impulsivity", "neuroticism")

# Colorblind-friendly palette for first seller status
CB_PALETTE <- c("No" = "#0072B2", "Yes" = "#D55E00")

# =====
# Main function
# =====
main <- function() {
  ensure_output_dir()
  df <- load_and_prepare_data()

  cat("Creating box plots...\n")
  p_box <- create_boxplots(df)
  save_plot(p_box, OUTPUT_BOXPLOTS, width = 10, height = 8)

  cat("Creating violin plots...\n")
  p_violin <- create_violins(df)
  save_plot(p_violin, OUTPUT_VIOLINS, width = 10, height = 8)

  cat("Creating quartile bar chart...\n")
  p_quartile <- create_quartile_plot(df)
  save_plot(p_quartile, OUTPUT_QUARTILE, width = 8, height = 5)

  cat("All plots saved to:", OUTPUT_DIR, "\n")
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function() {
  df <- fread(INPUT_PATH)

  # Create readable first seller label
  df[, first_seller_label := ifelse(is_first_seller == 1, "Yes", "No")]
  df[, first_seller_label := factor(first_seller_label, levels = c("No", "Yes"))]

  cat("Loaded", nrow(df), "observations\n")
  cat("First sellers:", sum(df$is_first_seller), "\n")
  return(df)
}

# =====
# Reshape data to long format for faceted plots
# =====
reshape_to_long <- function(df) {
  id_cols <- c("session_id", "segment", "group_id", "round", "player",
               "is_first_seller", "first_seller_label")

  df_long <- melt(
    df,
    id.vars = id_cols,
    measure.vars = TRAITS,
    variable.name = "trait",
    value.name = "score"
  )

  # Clean trait names for display
  df_long[, trait_label := gsub("_", " ", trait)]
  df_long[, trait_label := tools::toTitleCase(trait_label)]

  return(df_long)
}

# =====
# Plot 1: Faceted box plots by trait
# =====
create_boxplots <- function(df) {
  df_long <- reshape_to_long(df)

  ggplot(df_long, aes(x = first_seller_label, y = score, fill = first_seller_label)) +
    geom_boxplot(alpha = 0.7, outlier.size = 1) +
    facet_wrap(~trait_label, scales = "free_y", ncol = 4) +
    scale_fill_manual(values = CB_PALETTE, name = "First Seller") +
    labs(x = "First Seller", y = "Trait Score") +
    theme_minimal() +
    theme(
      panel.grid.minor = element_blank(),
      strip.text = element_text(size = 10, face = "bold"),
      legend.position = "bottom"
    )
}

# =====
# Plot 2: Faceted violin plots by trait
# =====
create_violins <- function(df) {
  df_long <- reshape_to_long(df)

  ggplot(df_long, aes(x = first_seller_label, y = score, fill = first_seller_label)) +
    geom_violin(alpha = 0.7, draw_quantiles = c(0.25, 0.5, 0.75)) +
    facet_wrap(~trait_label, scales = "free_y", ncol = 4) +
    scale_fill_manual(values = CB_PALETTE, name = "First Seller") +
    labs(x = "First Seller", y = "Trait Score") +
    theme_minimal() +
    theme(
      panel.grid.minor = element_blank(),
      strip.text = element_text(size = 10, face = "bold"),
      legend.position = "bottom"
    )
}

# =====
# Plot 3: First seller rate by trait quartile
# =====
create_quartile_plot <- function(df) {
  quartile_data <- compute_quartile_rates(df)

  ggplot(quartile_data, aes(x = quartile, y = first_seller_rate, fill = trait_label)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    geom_text(
      aes(label = sprintf("%.1f%%", first_seller_rate * 100)),
      position = position_dodge(width = 0.9),
      vjust = -0.5,
      size = 3
    ) +
    scale_fill_manual(
      values = c("Impulsivity" = "#E69F00", "Neuroticism" = "#009E73"),
      name = "Trait"
    ) +
    scale_y_continuous(
      labels = scales::percent_format(),
      limits = c(0, max(quartile_data$first_seller_rate) * 1.15)
    ) +
    labs(x = "Trait Quartile", y = "First Seller Rate") +
    theme_minimal() +
    theme(
      panel.grid.minor = element_blank(),
      legend.position = "bottom"
    )
}

# =====
# Compute first seller rates by quartile for key traits
# =====
compute_quartile_rates <- function(df) {
  results <- lapply(KEY_TRAITS, function(trait) {
    compute_single_trait_quartiles(df, trait)
  })
  rbindlist(results)
}

compute_single_trait_quartiles <- function(df, trait) {
  # Make a copy to avoid modifying the original
  df_copy <- copy(df)

  # Assign quartiles directly based on trait values
  df_copy[, quartile := cut(
    get(trait),
    breaks = quantile(get(trait), probs = 0:4/4, na.rm = TRUE),
    labels = c("Q1 (Low)", "Q2", "Q3", "Q4 (High)"),
    include.lowest = TRUE
  )]

  # Compute first seller rates by quartile
  rates <- df_copy[, .(
    n_first_sellers = sum(is_first_seller),
    n_total = .N,
    first_seller_rate = mean(is_first_seller)
  ), by = quartile]

  rates[, trait := trait]
  rates[, trait_label := tools::toTitleCase(gsub("_", " ", trait))]

  return(rates)
}

# =====
# Utility functions
# =====
ensure_output_dir <- function() {
  if (!dir.exists(OUTPUT_DIR)) {
    dir.create(OUTPUT_DIR, recursive = TRUE)
    cat("Created output directory:", OUTPUT_DIR, "\n")
  }
}

save_plot <- function(p, path, width = 8, height = 6) {
  ggsave(path, p, width = width, height = height)
  cat("Saved:", path, "\n")
}

# %%
if (sys.nframe() == 0) {
  main()
}
