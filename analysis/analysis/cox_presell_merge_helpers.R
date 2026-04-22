# Purpose: Merge helpers + console-print helpers for windowed Cox presell regressions
# Author: Claude Code
# Date: 2026-04-20

# Libraries loaded by main script (do not load here)

# =====
# Constants
# =====
WINDOWS <- c(50, 100, 500, 1000, 2000)

BASE_EMOTION_COLS <- c("fear_mean", "anger_mean", "contempt_mean",
                       "disgust_mean", "joy_mean", "sadness_mean",
                       "surprise_mean", "engagement_mean", "valence_mean")

MERGE_KEYS <- c("session_id", "segment", "round", "period",
                "group_id", "player")

# =====
# Load presell dataset
# =====
load_presell_dataset <- function(path) {
  fread(path)
}

# =====
# Merge window-specific emotions onto base data and overwrite sold==1 rows
# =====
merge_presell_window <- function(base_dt, presell_dt, window) {
  window_cols <- paste0(BASE_EMOTION_COLS, "_", window, "ms")
  frames_col <- paste0("n_frames_", window, "ms")
  sub <- build_presell_subset(presell_dt, window_cols, frames_col)
  sub <- align_key_types(sub, base_dt)
  merged <- merge(base_dt, sub, by = MERGE_KEYS, all.x = TRUE, sort = FALSE)
  stopifnot(nrow(merged) == nrow(base_dt))
  merged <- overwrite_sold_emotions(merged)
  n_sold <- sum(merged$sold == 1, na.rm = TRUE)
  n_matched <- sum(merged$sold == 1 & !is.na(merged[[frames_col]]))
  list(dt = merged, window = window, frames_col = frames_col,
       n_sold = n_sold, n_matched = n_matched)
}

build_presell_subset <- function(presell_dt, window_cols, frames_col) {
  sub <- presell_dt[, c(MERGE_KEYS, window_cols, frames_col), with = FALSE]
  setnames(sub, window_cols, paste0(BASE_EMOTION_COLS, "_presell"))
  sub
}

align_key_types <- function(sub, base_dt) {
  for (k in MERGE_KEYS) {
    if (is.factor(base_dt[[k]]) && !is.factor(sub[[k]])) {
      sub[, (k) := factor(as.character(sub[[k]]), levels = levels(base_dt[[k]]))]
    }
  }
  sub
}

overwrite_sold_emotions <- function(merged) {
  for (col in BASE_EMOTION_COLS) {
    pc <- paste0(col, "_presell")
    mask <- merged$sold == 1 & !is.na(merged[[pc]])
    if (any(mask)) merged[mask, (col) := get(pc)]
  }
  merged[, (paste0(BASE_EMOTION_COLS, "_presell")) := NULL]
  merged
}

# =====
# Drop sold==1 rows where the window had no frames
# =====
drop_missing_window_rows <- function(merged_dt, window, frames_col) {
  mask_drop <- merged_dt$sold == 1 &
    (is.na(merged_dt[[frames_col]]) | merged_dt[[frames_col]] == 0)
  n_dropped <- sum(mask_drop)
  filtered <- merged_dt[!mask_drop]
  cat(sprintf(
    "[window=%dms] Dropped %d sold==1 rows with no window data\n",
    window, n_dropped))
  list(dt = filtered, n_dropped = n_dropped)
}

# =====
# Collect (HR, SE, p) for each emotion across windows for one model slot
# =====
collect_window_coefs <- function(models_list, windows) {
  out <- list()
  for (i in seq_along(windows)) {
    w <- windows[i]
    m <- models_list[[i]]
    if (is.null(m)) {
      out[[as.character(w)]] <- NULL
    } else {
      out[[as.character(w)]] <- extract_cox_coefs(m)
    }
  }
  out
}

# =====
# Collect fit stats per window for one model slot
# =====
collect_window_fits <- function(models_list, windows) {
  out <- list()
  for (i in seq_along(windows)) {
    w <- windows[i]
    m <- models_list[[i]]
    out[[as.character(w)]] <- if (is.null(m)) NULL else extract_cox_fit(m)
  }
  out
}

# =====
# Side-by-side console comparison for one model slot
# =====
print_window_comparison <- function(coefs_by_window, fits_by_window,
                                    model_label) {
  cat(sprintf("\n=== %s ===\n", model_label))
  cat(format_window_header(), "\n")
  for (em in BASE_EMOTION_COLS) {
    cat(format_window_row(em, coefs_by_window), "\n")
  }
  cat("---\n")
  cat("Fit statistics\n")
  cat(format_fit_row("N", fits_by_window, "n"), "\n")
  cat(format_fit_row("Events", fits_by_window, "events"), "\n")
  cat(format_fit_row("LogLik", fits_by_window, "log_lik"), "\n")
}

format_window_header <- function() {
  cells <- sapply(WINDOWS, function(w) {
    formatC(sprintf("%dms (HR, SE, p)", w), width = 24, flag = "-")
  })
  paste0(formatC("EMOTION", width = 18, flag = "-"), "| ",
         paste(cells, collapse = "| "))
}

format_window_row <- function(emotion, coefs_by_window) {
  cells <- sapply(WINDOWS, function(w) {
    coefs <- coefs_by_window[[as.character(w)]]
    formatC(format_emotion_cell(emotion, coefs), width = 24, flag = "-")
  })
  paste0(formatC(emotion, width = 18, flag = "-"), "| ",
         paste(cells, collapse = "| "))
}

format_emotion_cell <- function(emotion, coefs) {
  if (is.null(coefs)) return("--")
  row <- coefs[var == emotion]
  if (nrow(row) == 0) return("--")
  sprintf("%.4f, %.4f, %.3f", row$est, row$se, row$pval)
}

format_fit_row <- function(label, fits_by_window, field) {
  cells <- sapply(WINDOWS, function(w) {
    fit <- fits_by_window[[as.character(w)]]
    formatC(format_fit_cell(fit, field), width = 24, flag = "-")
  })
  paste0(formatC(label, width = 18, flag = "-"), "| ",
         paste(cells, collapse = "| "))
}

format_fit_cell <- function(fit, field) {
  if (is.null(fit)) return("--")
  val <- fit[[field]]
  if (field == "log_lik") sprintf("%.1f", val) else format(val, big.mark = ",")
}
