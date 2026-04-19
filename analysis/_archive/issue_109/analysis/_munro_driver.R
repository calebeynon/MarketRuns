# Purpose: Thin R driver for Munro's MixedStrat_low_RA.R replication check.
# Sources the original file and writes the final sigma and V tables to CSV
# so Python can load them for numerical comparison.
#
# Invoked by compare_munro_replication.py via Rscript.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript _munro_driver.R <munro_ra.R> <out_csv>")
}
munro_script <- args[[1]]
out_csv <- args[[2]]

if (!file.exists(munro_script)) {
  stop(sprintf("Munro script not found: %s", munro_script))
}

# Source Munro's script without printing to console (suppress via sink)
sink(tempfile())
source(munro_script, local = FALSE)
sink()

# After sourcing, the following are in the global env:
#   beliefs, sig (final n=4 strategy), V_1..V_4, W_1..W_4
# Because sig is overwritten inside the script, we need to reconstruct the
# per-n sigma that was actually used. Munro hard-codes these; we replay.
sig_n2 <- as.numeric(beliefs >= 0.90)
sig_n2[length(beliefs) - 5] <- 0.453

sig_n3 <- as.numeric(beliefs >= 0.82)
sig_n3[length(beliefs) - 6] <- 0.402

sig_n4 <- as.numeric(beliefs >= 0.68)
sig_n4[length(beliefs) - 7] <- 0.11
sig_n4[length(beliefs) - 6] <- 0.989

df <- data.frame(
  belief = beliefs,
  sigma_n2 = sig_n2,
  sigma_n3 = sig_n3,
  sigma_n4 = sig_n4,
  V_1 = V_1,
  V_2 = V_2,
  V_3 = V_3,
  V_4 = V_4
)
write.csv(df, out_csv, row.names = FALSE)
cat(sprintf("Wrote %d rows to %s\n", nrow(df), out_csv))
