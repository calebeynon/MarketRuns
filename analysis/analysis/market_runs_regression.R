# nolint start
# Purpose: Regress market run indicators on group-level traits, equilibrium
#          deviation, no-sale flag, signal composition, signal accuracy, and
#          treatment / segment controls. Sweeps the (w, k) run-detection grid
#          and writes 3 stacked panels (one per k) of LPM + logit columns to
#          a single .tex at analysis/output/tables/market_runs_regression.tex.
#          Cluster: group_round_id. Logit reports raw log-odds.
# Author: Claude Code (impl-regression teammate, issue #120)
# Date: 2026-04-24

library(data.table)
library(fixest)

# FILE PATHS
INPUT_CSV <- "datastore/derived/market_runs_dataset.csv"
OUTPUT_TEX <- "analysis/output/tables/market_runs_regression.tex"

# Sweep grid: window length is w + 1 periods, threshold is k distinct sellers.
W_VALUES <- c(0, 1, 2, 3)
K_VALUES <- c(2, 3, 4)

# Footnote appended to a panel when one or more logit columns were dropped.
NONCONV_FOOTNOTE <- "\\footnotesize\\textit{Logit columns omitted where feglm did not converge (separation).}\\normalsize"

# Right-hand side: group-mean traits, equilibrium IVs, signal accuracy, controls.
# Column names are dictated by the T3 trait aggregator + T4 builder.
RHS <- paste(
  "group_mean_extraversion + group_mean_agreeableness + group_mean_conscientiousness",
  "+ group_mean_neuroticism + group_mean_openness + group_mean_impulsivity",
  "+ group_mean_state_anxiety + group_mean_risk_tolerance",
  "+ dev_from_threshold + dev_from_avg_pi + no_sale",
  "+ signal_correct_frac + alpha + treatment + segment"
)

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  dt <- load_and_prepare()
  print_iv_collinearity(dt)
  panel_k2 <- run_panel_for_k(dt, 2)
  panel_k3 <- run_panel_for_k(dt, 3)
  panel_k4 <- run_panel_for_k(dt, 4)
  export_stacked_panels(panel_k2, panel_k3, panel_k4)
  cat("Done!\n")
}

# =====
# Load CSV and build the cluster identifier
# =====
load_and_prepare <- function() {
  if (!file.exists(INPUT_CSV)) {
    stop(sprintf(
      "Input CSV missing: %s. Run build_market_runs_dataset.py first.",
      INPUT_CSV
    ))
  }
  dt <- fread(INPUT_CSV)
  cat("Loaded", nrow(dt), "group-round rows from", INPUT_CSV, "\n")
  dt[, group_round_id := paste(session_id, segment, group_id, round, sep = "_")]
  dt[, segment := factor(segment)]
  dt[, treatment := factor(treatment)]
  return(dt)
}

# =====
# Diagnostic: collinearity between the two equilibrium-deviation IVs
# =====
print_iv_collinearity <- function(dt) {
  rho <- cor(dt$dev_from_threshold, dt$dev_from_avg_pi, use = "complete.obs")
  cat(sprintf("dev_from_threshold vs dev_from_avg_pi correlation: %.4f\n", rho))
  if (abs(rho) > 0.95) {
    warning(sprintf("Near-collinearity (|rho| = %.3f); check IV stability", rho))
  }
}

# =====
# Run all 4 LPM + 4 logit models for one panel (one k value)
# =====
run_panel_for_k <- function(dt, k) {
  cat(sprintf("\n--- Panel k = %d ---\n", k))
  lpms <- estimate_grid(dt, k, fit_lpm)
  logits <- estimate_grid(dt, k, fit_logit)
  print_cluster_counts(dt, k)
  list(lpms = lpms, logits = logits)
}

# =====
# Apply estimator across all w values for fixed k
# =====
estimate_grid <- function(dt, k, fit_fn) {
  lapply(W_VALUES, function(w) {
    depvar <- sprintf("run_w%d_k%d", w, k)
    fit_fn(dt, depvar)
  })
}

# =====
# LPM estimator with group_round-clustered SE
# =====
fit_lpm <- function(dt, depvar) {
  formula_obj <- as.formula(paste(depvar, "~", RHS))
  feols(formula_obj, data = dt, cluster = ~group_round_id)
}

# =====
# Logit estimator (raw log-odds) with group_round-clustered SE.
# Returns NULL if feglm fails to converge (e.g., quasi-separation), so
# panel_to_tex can drop the column rather than print absurd coefficients.
# =====
fit_logit <- function(dt, depvar) {
  formula_obj <- as.formula(paste(depvar, "~", RHS))
  fit <- tryCatch(
    feglm(
      formula_obj, data = dt,
      family = binomial(link = "logit"),
      cluster = ~group_round_id
    ),
    error = function(e) {
      cat(sprintf("[warn] logit errored for %s: %s\n", depvar, conditionMessage(e)))
      NULL
    }
  )
  if (is.null(fit)) return(NULL)
  if (!isTRUE(fit$convStatus)) {
    cat(sprintf("[warn] logit did not converge for %s\n", depvar))
    return(NULL)
  }
  fit
}

# =====
# Report N clusters per (w, k) cell; warn if cluster count is low
# =====
print_cluster_counts <- function(dt, k) {
  for (w in W_VALUES) {
    depvar <- sprintf("run_w%d_k%d", w, k)
    n_clust <- dt[!is.na(get(depvar)), uniqueN(group_round_id)]
    cat(sprintf("  k=%d, w=%d: %d clusters\n", k, w, n_clust))
    if (n_clust < 30) {
      warning(sprintf("k=%d, w=%d: only %d clusters (< 30)", k, w, n_clust))
    }
  }
}

# =====
# Write the 3 stacked panels (k=2, 3, 4) into a single .tex
# =====
export_stacked_panels <- function(p2, p3, p4) {
  output_dir <- dirname(OUTPUT_TEX)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  tex_lines <- c(
    "\\textbf{Panel A: k = 2 (any 2 distinct sellers in window)}\\\\",
    panel_to_tex(p2),
    "\\medskip",
    "\\textbf{Panel B: k = 3 (any 3 distinct sellers in window)}\\\\",
    panel_to_tex(p3),
    "\\medskip",
    "\\textbf{Panel C: k = 4 (all 4 distinct sellers in window)}\\\\",
    panel_to_tex(p4)
  )
  writeLines(tex_lines, OUTPUT_TEX)
  cat("Table exported to:", OUTPUT_TEX, "\n")
}

# =====
# Render one panel (LPM + surviving Logit columns) as a TeX character block.
# Logit columns whose feglm did not converge are dropped (returned NULL by
# fit_logit); a footnote line is appended when at least one was dropped.
# =====
panel_to_tex <- function(panel) {
  keep <- !sapply(panel$logits, is.null)
  logits_kept <- panel$logits[keep]
  models <- c(panel$lpms, logits_kept)
  hdrs <- panel_headers(length(panel$lpms), W_VALUES[keep])
  tex <- etable(
    models, headers = hdrs,
    fitstat = c("n", "r2"),
    tex = TRUE, float = FALSE,
    style.tex = style.tex(fontsize = "scriptsize")
  )
  if (any(!keep)) tex <- c(tex, NONCONV_FOOTNOTE)
  tex
}

# =====
# Build the etable headers list given LPM count and surviving logit windows
# =====
panel_headers <- function(n_lpm, w_logit) {
  list(
    "Estimator" = c(rep("LPM", n_lpm), rep("Logit", length(w_logit))),
    "Window w" = c(as.character(W_VALUES), as.character(w_logit))
  )
}

# %%
if (!interactive()) main()
