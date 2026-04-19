# Purpose: Munro-style value iteration with a DENSE reachable-belief grid
# (matching our Python solver's build_belief_grid(t_max=20)) and EXACT
# continuous Bayesian updating via linear interpolation of V. Sigma is
# derived by uniroot() on the indifference condition. Runs at alpha=0.5,
# random treatment, with OUR prices [2,4,6,8] mapped into Munro's
# descending convention as c(8,6,4,2). Writes per-n sigma and V to CSV
# for cross-validation against the Python solver.
#
# Author: r-densifier
# Date: 2026-04-16

# =====
# Parameters (match Munro's MixedStrat_low_RA.R, our prices)
# =====
L <- 0.125
MUB <- 0.675
MUG <- 1 - MUB
A <- 0.5
V_LIQ <- 20
PI_0 <- 0.5
T_MAX <- 20

PRICES <- c(8, 6, 4, 2)

TOL_V <- 1e-6
TOL_SIG <- 1e-7
MAX_ITER <- 2000

# Resolve output path relative to the project root. Run from project root:
#   Rscript analysis/_archive/issue_109/analysis/_munro_style_solver.R
OUT_CSV <- "analysis/_archive/issue_109/output/munro_style_our_prices.csv"

# Reachable-belief grid in P(Bad). Mirrors equilibrium_model.build_belief_grid
# (axis there is P(Good)); we build directly in P(Bad) by iterating bad/good
# Bayes updates from pi_0=0.5, then take sorted unique values.
BELIEFS <- local({
  pi_vec <- PI_0
  pi_cur <- PI_0
  for (k in seq_len(T_MAX)) {
    pi_cur <- pi_cur * (1 - MUB) / (pi_cur * (1 - MUB) + (1 - pi_cur) * MUB)
    pi_vec <- c(pi_vec, pi_cur)
  }
  pi_cur <- PI_0
  for (k in seq_len(T_MAX)) {
    pi_cur <- pi_cur * MUB / (pi_cur * MUB + (1 - pi_cur) * (1 - MUB))
    pi_vec <- c(pi_vec, pi_cur)
  }
  sort(unique(pi_vec))
})
NUM_BELIEFS <- length(BELIEFS)

PROB_B <- BELIEFS * MUB + (1 - BELIEFS) * MUG
PROB_G <- BELIEFS * (1 - MUB) + (1 - BELIEFS) * (1 - MUG)
# Exact posterior beliefs (P(Bad)) after bad/good signal.
PI_GIVEN_B <- BELIEFS * MUB / PROB_B
PI_GIVEN_G <- BELIEFS * (1 - MUB) / PROB_G

# =====
# Main function
# =====
main <- function() {
  u_fn <- make_utility(A)
  res <- solve_all(u_fn)
  write_results(res)
  print_summary(res)
}

# =====
# Utility
# =====
make_utility <- function(alpha) {
  if (abs(alpha - 1) < 1e-10) return(function(x) log(x))
  function(x) x^(1 - alpha) / (1 - alpha)
}

# =====
# Per-n price utilities (for given n holders, j other sellers)
# =====
sell_price_util <- function(n, j, u_fn) {
  start <- 5 - n
  end <- start + j
  mean(u_fn(PRICES[start:end]))
}

forced_liq_util <- function(n, u_fn) {
  start <- 5 - n
  mean(u_fn(PRICES[start:4]))
}

# =====
# s_probs matrix for a given sigma vector (length NUM_BELIEFS)
# =====
build_s_probs <- function(sigma_vec, n) {
  mat <- matrix(nrow = n, ncol = NUM_BELIEFS)
  for (j in 0:(n - 1)) {
    mat[j + 1, ] <- choose(n - 1, j) * sigma_vec^j * (1 - sigma_vec)^(n - 1 - j)
  }
  mat
}

# =====
# Continuous continuation value: linear interpolation of V at exact Bayes
# posteriors, weighted by signal probabilities.
# =====
cont_value_vec <- function(v_n) {
  v_b <- approx(BELIEFS, v_n, PI_GIVEN_B, rule = 2)$y
  v_g <- approx(BELIEFS, v_n, PI_GIVEN_G, rule = 2)$y
  PROB_B * v_b + PROB_G * v_g
}

# =====
# Single Bellman update for V_n given sigma, V_n itself, and W_{<n}
# Returns list(new_V_n, exp_price_vec, exp_W_vec, W_n)
# =====
bellman_update <- function(n, sigma_vec, v_n, w_lower, u_fn) {
  cont_v <- cont_value_vec(v_n)
  term_value <- (1 - BELIEFS) * u_fn(V_LIQ) + BELIEFS * forced_liq_util(n, u_fn)
  w_n <- L * term_value + (1 - L) * cont_v
  exp_price_vec <- compute_exp_price(n, sigma_vec, u_fn)
  exp_w_vec <- compute_exp_w(n, sigma_vec, w_n, w_lower)
  new_v_n <- pmax(exp_price_vec, exp_w_vec)
  list(V = new_v_n, exp_price = exp_price_vec, exp_w = exp_w_vec, W = w_n)
}

compute_exp_price <- function(n, sigma_vec, u_fn) {
  s_probs <- build_s_probs(sigma_vec, n)
  price_utils <- sapply(0:(n - 1), function(j) sell_price_util(n, j, u_fn))
  price_mat <- matrix(rep(price_utils, NUM_BELIEFS), nrow = n, byrow = FALSE)
  colSums(s_probs * price_mat)
}

compute_exp_w <- function(n, sigma_vec, w_n, w_lower) {
  s_probs <- build_s_probs(sigma_vec, n)
  w_stack <- rbind(w_n, w_lower)
  colSums(s_probs * w_stack)
}

# =====
# Scalar indifference diff (for uniroot) at a single belief index i
# =====
indiff_diff <- function(sigma_scalar, i, n, v_n, w_lower, u_fn) {
  sig_vec <- rep(sigma_scalar, NUM_BELIEFS)
  cont_v <- cont_value_vec(v_n)
  term_value <- (1 - BELIEFS) * u_fn(V_LIQ) + BELIEFS * forced_liq_util(n, u_fn)
  w_n <- L * term_value + (1 - L) * cont_v
  ep <- compute_exp_price(n, sig_vec, u_fn)[i]
  ew <- compute_exp_w(n, sig_vec, w_n, w_lower)[i]
  ep - ew
}

# =====
# Derive sigma vector given current V_n and W_{<n} via uniroot per belief
# =====
derive_sigma <- function(n, v_n, w_lower, u_fn) {
  sig <- numeric(NUM_BELIEFS)
  for (i in seq_len(NUM_BELIEFS)) {
    sig[i] <- sigma_at_belief(i, n, v_n, w_lower, u_fn)
  }
  sig
}

sigma_at_belief <- function(i, n, v_n, w_lower, u_fn) {
  d0 <- indiff_diff(0, i, n, v_n, w_lower, u_fn)
  d1 <- indiff_diff(1, i, n, v_n, w_lower, u_fn)
  if (d0 <= 0) return(0)
  if (d1 >= 0) return(1)
  r <- uniroot(indiff_diff, interval = c(0, 1), i = i, n = n,
               v_n = v_n, w_lower = w_lower, u_fn = u_fn,
               tol = 1e-12)
  r$root
}

# =====
# One-player base case (no strategic sigma)
# =====
solve_n1 <- function(u_fn) {
  sell_now <- u_fn(PRICES[4])
  hold <- (1 - BELIEFS) * u_fn(V_LIQ) + BELIEFS * u_fn(PRICES[4])
  v_1 <- pmax(sell_now, hold)
  term_value <- (1 - BELIEFS) * u_fn(V_LIQ) + BELIEFS * u_fn(PRICES[4])
  cont_v <- cont_value_vec(v_1)
  w_1 <- L * term_value + (1 - L) * cont_v
  list(V = v_1, W = w_1, sigma = rep(0, NUM_BELIEFS), iters = 1)
}

# =====
# Multi-player solver: outer loop alternates V-update and sigma-update
# =====
solve_n <- function(n, w_lower, u_fn) {
  v_n <- init_v(n, u_fn)
  sigma_vec <- as.numeric(BELIEFS >= 0.5)
  for (iter in seq_len(MAX_ITER)) {
    step <- bellman_update(n, sigma_vec, v_n, w_lower, u_fn)
    new_v <- step$V
    new_sigma <- derive_sigma(n, new_v, w_lower, u_fn)
    dv <- max(abs(new_v / pmax(abs(v_n), 1e-12) - 1))
    ds <- max(abs(new_sigma - sigma_vec))
    v_n <- new_v
    sigma_vec <- new_sigma
    if (dv < TOL_V && ds < TOL_SIG) {
      final <- bellman_update(n, sigma_vec, v_n, w_lower, u_fn)
      return(list(V = final$V, W = final$W, sigma = sigma_vec, iters = iter))
    }
  }
  stop(sprintf("n=%d did not converge in %d iterations", n, MAX_ITER))
}

init_v <- function(n, u_fn) {
  start <- 5 - n
  sell_all <- mean(u_fn(PRICES[start:4]))
  sell_now <- u_fn(PRICES[start])
  hold <- BELIEFS * sell_all + (1 - BELIEFS) * u_fn(V_LIQ)
  pmax(sell_now, hold)
}

# =====
# Solve all 4 holder counts and collect results
# =====
solve_all <- function(u_fn) {
  res_1 <- solve_n1(u_fn)
  w_stack <- matrix(res_1$W, nrow = 1)
  res_2 <- solve_n(2, w_stack, u_fn)
  w_stack <- rbind(res_2$W, w_stack)
  res_3 <- solve_n(3, w_stack, u_fn)
  w_stack <- rbind(res_3$W, w_stack)
  res_4 <- solve_n(4, w_stack, u_fn)
  list(n1 = res_1, n2 = res_2, n3 = res_3, n4 = res_4)
}

# =====
# Output
# =====
write_results <- function(res) {
  rows <- list()
  for (n in 1:4) {
    key <- sprintf("n%d", n)
    rows[[n]] <- data.frame(
      n = n, belief = BELIEFS,
      sigma = res[[key]]$sigma, V = res[[key]]$V
    )
  }
  df <- do.call(rbind, rows)
  dir.create(dirname(OUT_CSV), showWarnings = FALSE, recursive = TRUE)
  write.csv(df, OUT_CSV, row.names = FALSE)
  cat(sprintf("Wrote %d rows to %s\n", nrow(df), OUT_CSV))
}

print_summary <- function(res) {
  sample_idx <- round(seq(1, NUM_BELIEFS, length.out = 5))
  for (n in 1:4) {
    key <- sprintf("n%d", n)
    cat(sprintf("--- n=%d  iters=%d\n", n, res[[key]]$iters))
    for (i in sample_idx) {
      cat(sprintf("  belief=%.4f  sigma=%.4f  V=%.4f\n",
                  BELIEFS[i], res[[key]]$sigma[i], res[[key]]$V[i]))
    }
  }
}

main()
