"""
Purpose: Core Magnani & Munro (2020) equilibrium model — Bellman equation
         solver for a market runs experiment with N=4 investors.
Author: Claude
Date: 2026-04-05
"""

import numpy as np
from math import comb, log
from scipy.optimize import brentq

# CONSTANTS
N_INVESTORS = 4
PRICES = [2, 4, 6, 8]  # p_n = 2n
FINAL_VALUE = 20  # v, liquidation value in good state
MU_B = 0.675  # P(bad signal | Bad state)
MU_G = 0.325  # P(good signal | Bad state)
LAMBDA = 0.125  # per-period termination probability
PI_0 = 0.5  # initial belief


# =====
# Main function
# =====
def main():
    """Demo: solve equilibrium for alpha=0.5, random treatment."""
    result = solve_equilibrium(alpha=0.5, treatment="random")
    grid = result["belief_grid"]
    print("Equilibrium solved. Belief grid size:", len(grid))
    for n in range(1, N_INVESTORS + 1):
        thresholds = grid[result["sigma"][n] > 0.01]
        if len(thresholds) > 0:
            print(f"  n={n}: selling starts at pi={thresholds[0]:.4f}")
        else:
            print(f"  n={n}: never sells")


# =====
# Belief grid construction
# =====
def build_belief_grid(t_max=40):
    """Build reachable belief grid from net bad signal counts.

    Since good and bad signals are exact inverses (mu_B + mu_G = 1),
    belief depends only on the net number of bad signals, giving
    2*t_max + 1 distinct beliefs.
    """
    pi = PI_0
    beliefs = [pi]
    for _ in range(t_max):
        pi = _update_bad(pi)
        beliefs.append(pi)
    pi = PI_0
    for _ in range(t_max):
        pi = _update_good(pi)
        beliefs.append(pi)
    return np.sort(np.unique(beliefs))


def _update_good(pi):
    """Bayesian update after a good signal."""
    num = pi * (1 - MU_G)
    return num / (num + (1 - pi) * MU_G)


def _update_bad(pi):
    """Bayesian update after a bad signal."""
    num = pi * (1 - MU_B)
    return num / (num + (1 - pi) * MU_B)


# =====
# Utility functions
# =====
def crra_utility(x, alpha):
    """CRRA utility: x^(1-a)/(1-a). Risk-neutral if a~0, log if a~1."""
    if abs(alpha) < 1e-10:
        return float(x)
    if abs(alpha - 1.0) < 1e-10:
        return log(x)
    return x ** (1 - alpha) / (1 - alpha)


# =====
# Payoff calculations
# =====
def rho(n, k, alpha, treatment):
    """Payoff from k simultaneous sellers when n are holding."""
    prices = [PRICES[n - 1 - j] for j in range(k)]
    if treatment == "random":
        return sum(crra_utility(p, alpha) for p in prices) / k
    return crra_utility(sum(prices) / k, alpha)


def forced_liquidation(m, alpha, treatment):
    """L(m): expected utility from forced liquidation with m holders."""
    prices = [PRICES[i] for i in range(m)]
    if treatment == "random":
        return sum(crra_utility(p, alpha) for p in prices) / m
    return crra_utility(sum(prices) / m, alpha)


# =====
# Sell and hold expected utilities
# =====
def compute_u_sell(n, sigma, alpha, treatment):
    """Expected utility of selling given n holders and sell probability sigma."""
    total = 0.0
    for j in range(n):
        prob = comb(n - 1, j) * sigma**j * (1 - sigma) ** (n - 1 - j)
        total += prob * rho(n, j + 1, alpha, treatment)
    return total


def compute_u_hold(n, pi_prime, sigma, alpha, treatment, v_table, belief_grid):
    """Expected utility of holding given n holders and updated belief."""
    total = 0.0
    for j in range(n):
        prob = comb(n - 1, j) * sigma**j * (1 - sigma) ** (n - 1 - j)
        m = n - j  # remaining holders after j others sell
        total += prob * _h_value(m, pi_prime, alpha, treatment, v_table, belief_grid)
    return total


def _h_value(m, pi_prime, alpha, treatment, v_table, belief_grid):
    """H(m, pi'): continuation value combining termination and next period."""
    term_good = pi_prime * crra_utility(FINAL_VALUE, alpha)
    term_bad = (1 - pi_prime) * forced_liquidation(m, alpha, treatment)
    v_interp = np.interp(pi_prime, belief_grid, v_table[m])
    return LAMBDA * (term_good + term_bad) + (1 - LAMBDA) * v_interp


# =====
# Equilibrium sigma finder
# =====
def find_sigma(n, pi_prime, alpha, treatment, v_table, belief_grid):
    """Find equilibrium selling probability sigma for given state."""
    if n == 1:
        return 0.0
    u_sell_0 = compute_u_sell(n, 0.0, alpha, treatment)
    u_hold_0 = compute_u_hold(n, pi_prime, 0.0, alpha, treatment, v_table, belief_grid)
    if u_sell_0 <= u_hold_0:
        return 0.0
    u_sell_1 = compute_u_sell(n, 1.0, alpha, treatment)
    u_hold_1 = compute_u_hold(n, pi_prime, 1.0, alpha, treatment, v_table, belief_grid)
    if u_sell_1 >= u_hold_1:
        return 1.0
    return _root_find_sigma(n, pi_prime, alpha, treatment, v_table, belief_grid)


def _root_find_sigma(n, pi_prime, alpha, treatment, v_table, belief_grid):
    """Brentq root-finding for interior equilibrium sigma."""
    def diff(s):
        return (compute_u_sell(n, s, alpha, treatment)
                - compute_u_hold(n, pi_prime, s, alpha, treatment, v_table, belief_grid))
    return brentq(diff, 0.0, 1.0, xtol=1e-12)


def find_continuous_threshold(n, alpha, treatment, v_table, belief_grid):
    """Find exact pi* where U_sell(sigma=0) = U_hold(sigma=0) via root-finding."""
    if n == 1:
        return 0.0
    u_sell_0 = compute_u_sell(n, 0.0, alpha, treatment)

    def diff(pi):
        return u_sell_0 - compute_u_hold(
            n, pi, 0.0, alpha, treatment, v_table, belief_grid)

    lo, hi = float(belief_grid[0]), float(belief_grid[-1])
    if diff(lo) <= 0:
        return 0.0
    if diff(hi) > 0:
        return hi
    return brentq(diff, lo, hi, xtol=1e-12)


# =====
# Main solver
# =====
def _init_v_table(n_beliefs, alpha, v_init):
    """Initialize value table for value iteration."""
    if v_init == "high":
        v_start = crra_utility(FINAL_VALUE, alpha)
        return {n: np.full(n_beliefs, v_start) for n in range(1, N_INVESTORS + 1)}
    return {n: np.zeros(n_beliefs) for n in range(1, N_INVESTORS + 1)}


def solve_equilibrium(alpha, treatment, t_max=40, tol=1e-8, max_iter=1000,
                      v_init="low"):
    """Solve for symmetric Markov-perfect equilibrium via value iteration.

    v_init: "low" (V=0, converges from run direction) or
            "high" (V=u(v), converges from no-run direction).
    """
    belief_grid = build_belief_grid(t_max)
    v_table = _init_v_table(len(belief_grid), alpha, v_init)
    sigma_table = {n: np.zeros(len(belief_grid)) for n in range(1, N_INVESTORS + 1)}
    for _ in range(max_iter):
        v_old = {n: v_table[n].copy() for n in v_table}
        w_table = _compute_w_table(belief_grid, alpha, treatment, v_table, sigma_table)
        _update_v_table(belief_grid, v_table, w_table)
        max_diff = max(np.max(np.abs(v_table[n] - v_old[n])) for n in v_table)
        if max_diff < tol:
            break
    return {"sigma": sigma_table, "v_table": v_table, "belief_grid": belief_grid}


def _compute_w_table(belief_grid, alpha, treatment, v_table, sigma_table):
    """Compute W(n, pi') = sigma * U_sell + (1-sigma) * U_hold for all states."""
    w_table = {n: np.zeros(len(belief_grid)) for n in range(1, N_INVESTORS + 1)}
    for n in range(1, N_INVESTORS + 1):
        for i, pi_prime in enumerate(belief_grid):
            sig = find_sigma(n, pi_prime, alpha, treatment, v_table, belief_grid)
            sigma_table[n][i] = sig
            u_s = compute_u_sell(n, sig, alpha, treatment)
            u_h = compute_u_hold(n, pi_prime, sig, alpha, treatment, v_table, belief_grid)
            w_table[n][i] = sig * u_s + (1 - sig) * u_h
    return w_table


def _update_v_table(belief_grid, v_table, w_table):
    """Update V[n][pi] using signal transition probabilities."""
    for n in range(1, N_INVESTORS + 1):
        for i, pi in enumerate(belief_grid):
            pi_g = _update_good(pi)
            pi_b = _update_bad(pi)
            p_good = pi * (1 - MU_G) + (1 - pi) * MU_G
            p_bad = 1 - p_good
            w_g = np.interp(pi_g, belief_grid, w_table[n])
            w_b = np.interp(pi_b, belief_grid, w_table[n])
            v_table[n][i] = p_good * w_g + p_bad * w_b


# %%
if __name__ == "__main__":
    main()
