# nolint start
# Purpose: Regress market run indicators on first-seller deviation, first-seller
#          signal correctness, first-sale period, alpha_first, group-mean traits,
#          signal accuracy, tie indicator, and treatment / segment controls.
#          Sweeps the (w, k) run-detection grid and writes 3 stacked panels
#          (one per k) of LPM + logit columns to a single .tex at
#          analysis/output/tables/market_runs_first_seller_regression.tex.
#          Cluster: multi-way on session_id + group_id. Logit reports raw log-odds.
# Author: Claude Code (impl-regression teammate, issue #120)
# Date: 2026-05-04

library(data.table)
library(fixest)

# FILE PATHS
INPUT_CSV <- "datastore/derived/market_runs_first_seller_dataset.csv"
OUTPUT_TEX <- "analysis/output/tables/market_runs_first_seller_regression.tex"

# Sweep grid: window length is w + 1 periods, threshold is k distinct sellers.
W_VALUES <- c(0, 1, 2, 3)
K_VALUES <- c(2, 3, 4)

# Footnote appended to a panel when one or more logit columns were dropped.
NONCONV_FOOTNOTE <- "\\footnotesize\\textit{Logit columns omitted where feglm did not converge (separation).}\\normalsize"

# Right-hand side: indicator for the first seller liquidating above the M&M (2020)
# equilibrium threshold pi*(alpha, n=4). sold_above_eq=1 is "panicked early"; the
# omitted reference (sold_below_eq=1, "held past threshold") is captured by the
# intercept since the two indicators sum to 1 by construction (no exact-eq rows
# in the data make them perfect complements). Treatment and segment retained as
# standard factor controls.
RHS <- "sold_above_eq + pi_at_sale_first + treatment + segment"

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
  dt <- load_and_prepare()
  panel_k2 <- run_panel_for_k(dt, 2)
  panel_k3 <- run_panel_for_k(dt, 3)
  panel_k4 <- run_panel_for_k(dt, 4)
  export_stacked_panels(panel_k2, panel_k3, panel_k4)
  cat("Done!\n")
}

# =====
# Load CSV and coerce factor controls
# =====
load_and_prepare <- function() {
  if (!file.exists(INPUT_CSV)) {
    stop(sprintf(
      "Input CSV missing: %s. Run build_market_runs_first_seller_dataset.py first.",
      INPUT_CSV
    ))
  }
  dt <- fread(INPUT_CSV)
  cat("Loaded", nrow(dt), "group-round rows from", INPUT_CSV, "\n")
  dt[, sold_above_eq := as.integer(dev_from_threshold_first > 0)]
  dt[, sold_below_eq := as.integer(dev_from_threshold_first < 0)]
  dt[, segment := factor(segment)]
  dt[, treatment := factor(treatment)]
  return(dt)
}

# =====
# Run all LPM + logit models for one panel (one k value)
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
# LPM estimator with multi-way clustered SE (session + group)
# =====
fit_lpm <- function(dt, depvar) {
  formula_obj <- as.formula(paste(depvar, "~", RHS))
  feols(formula_obj, data = dt, cluster = ~session_id + group_id)
}

# =====
# Logit estimator (raw log-odds) with multi-way clustered SE.
# Returns NULL if feglm fails to converge (e.g., quasi-separation), so
# panel_to_tex can drop the column rather than print absurd coefficients.
# =====
fit_logit <- function(dt, depvar) {
  formula_obj <- as.formula(paste(depvar, "~", RHS))
  fit <- tryCatch(
    feglm(
      formula_obj, data = dt,
      family = binomial(link = "logit"),
      cluster = ~session_id + group_id
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
# Report cluster counts per (w, k) cell; warn if either dimension is thin
# =====
print_cluster_counts <- function(dt, k) {
  for (w in W_VALUES) {
    depvar <- sprintf("run_w%d_k%d", w, k)
    sub <- dt[!is.na(get(depvar))]
    n_sess <- sub[, uniqueN(session_id)]
    n_grp <- sub[, uniqueN(group_id)]
    cat(sprintf("  k=%d, w=%d: %d sessions, %d groups\n", k, w, n_sess, n_grp))
    if (n_sess < 10) {
      warning(sprintf("k=%d, w=%d: only %d sessions (< 10)", k, w, n_sess))
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
