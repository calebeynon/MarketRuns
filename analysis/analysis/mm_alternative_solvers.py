"""
Purpose: Alternative equilibrium solvers for M&M Appendix D replication.
         Implements policy iteration, Howard improvement, and convention
         variants to diagnose sigma discrepancies vs Magnani & Munro (2020).
Author: Claude
Date: 2026-04-07
"""

import numpy as np
from math import comb
from scipy.optimize import brentq

from analysis.analysis.equilibrium_model import (
    LAMBDA,
    MU_B,
    MU_G,
    N_INVESTORS,
    FINAL_VALUE,
    build_belief_grid,
    crra_utility,
    forced_liquidation,
    compute_u_sell,
    compute_u_hold,
    find_sigma,
    _update_good,
    _update_bad,
)

# SOLVER PARAMETERS
TOL = 1e-8
MAX_ITER = 2000
T_MAX = 40


# =====
# Common helpers
# =====
def _init_tables(t_max):
    """Initialize grid, V, and sigma tables."""
    grid = build_belief_grid(t_max)
    n_b = len(grid)
    v = {n: np.zeros(n_b) for n in range(1, N_INVESTORS + 1)}
    s = {n: np.zeros(n_b) for n in range(1, N_INVESTORS + 1)}
    return grid, v, s


def _check_convergence(v_table, v_old):
    """Return max absolute difference between current and old V."""
    return max(np.max(np.abs(v_table[n] - v_old[n])) for n in v_table)


def _make_result(sigma_table, v_table, grid):
    """Package solver output."""
    return {"sigma": sigma_table, "v_table": v_table, "belief_grid": grid}


def _update_v_from_w(grid, v_table, w_table):
    """Update V from W using signal transition probabilities."""
    for n in range(1, N_INVESTORS + 1):
        for i, pi in enumerate(grid):
            pi_g = _update_good(pi)
            pi_b = _update_bad(pi)
            p_good = pi * (1 - MU_G) + (1 - pi) * MU_G
            p_bad = 1 - p_good
            w_g = np.interp(pi_g, grid, w_table[n])
            w_b = np.interp(pi_b, grid, w_table[n])
            v_table[n][i] = p_good * w_g + p_bad * w_b


# =====
# Policy iteration solver
# =====
def solve_policy_iteration(alpha, treatment, t_max=T_MAX):
    """Solve via policy iteration: exact V solve per sigma update."""
    grid, v_table, sigma_table = _init_tables(t_max)
    for _ in range(MAX_ITER):
        sigma_old = {n: sigma_table[n].copy() for n in sigma_table}
        _update_sigma_table(grid, alpha, treatment, v_table, sigma_table)
        _solve_v_exactly(grid, alpha, treatment, v_table, sigma_table)
        if _check_convergence(sigma_table, sigma_old) < TOL:
            break
    return _make_result(sigma_table, v_table, grid)


def _update_sigma_table(grid, alpha, treatment, v_table, sigma_table):
    """Update sigma at every grid point using current V."""
    for n in range(1, N_INVESTORS + 1):
        for i, pi_prime in enumerate(grid):
            sigma_table[n][i] = find_sigma(
                n, pi_prime, alpha, treatment, v_table, grid
            )


def _solve_v_exactly(grid, alpha, treatment, v_table, sigma_table):
    """Solve V = T @ W exactly for fixed sigma, from n=1 upward."""
    t_mat = _build_transition_matrix(grid)
    for n in range(1, N_INVESTORS + 1):
        const, coeff = _build_w_decomposition(
            n, grid, alpha, treatment, v_table, sigma_table
        )
        a_mat = np.eye(len(grid)) - t_mat @ coeff
        v_table[n] = np.linalg.solve(a_mat, t_mat @ const)


def _build_w_decomposition(n, grid, alpha, treatment, v_table, sigma_table):
    """Decompose W into constant + V[n]-linear parts for linear solve."""
    n_b = len(grid)
    const = np.zeros(n_b)
    coeff = np.zeros((n_b, n_b))
    for j in range(n_b):
        sig = sigma_table[n][j]
        u_s = compute_u_sell(n, sig, alpha, treatment)
        u_c, u_cf = _decompose_u_hold(
            n, grid[j], sig, alpha, treatment, v_table, grid
        )
        const[j] = sig * u_s + (1 - sig) * u_c
        coeff[j, :] = (1 - sig) * u_cf
    return const, coeff


def _decompose_u_hold(n, pi_prime, sigma, alpha, treatment, v_table, grid):
    """Decompose U_hold into constant + linear coefficients in V[n]."""
    n_b = len(grid)
    total_const = 0.0
    total_coeffs = np.zeros(n_b)
    for j in range(n):
        prob = comb(n - 1, j) * sigma**j * (1 - sigma) ** (n - 1 - j)
        m = n - j
        h_c, h_cf = _decompose_h(
            m, pi_prime, alpha, treatment, v_table, grid, n
        )
        total_const += prob * h_c
        total_coeffs += prob * h_cf
    return total_const, total_coeffs


def _decompose_h(m, pi_prime, alpha, treatment, v_table, grid, target_n):
    """Decompose H(m, pi') into const + linear in V[target_n]."""
    n_b = len(grid)
    term = LAMBDA * (
        pi_prime * crra_utility(FINAL_VALUE, alpha)
        + (1 - pi_prime) * forced_liquidation(m, alpha, treatment)
    )
    if m != target_n:
        v_interp = np.interp(pi_prime, grid, v_table[m])
        return term + (1 - LAMBDA) * v_interp, np.zeros(n_b)
    coeffs = _interp_weights(grid, pi_prime, 1 - LAMBDA)
    return term, coeffs


def _interp_weights(grid, pi_target, scale):
    """Compute linear interpolation weights for pi_target on grid."""
    n_b = len(grid)
    coeffs = np.zeros(n_b)
    idx = np.searchsorted(grid, pi_target, side="right") - 1
    idx = max(0, min(idx, n_b - 2))
    frac = (pi_target - grid[idx]) / (grid[idx + 1] - grid[idx])
    frac = np.clip(frac, 0.0, 1.0)
    coeffs[idx] = scale * (1 - frac)
    coeffs[idx + 1] = scale * frac
    return coeffs


def _build_transition_matrix(grid):
    """Build signal transition matrix T[i,j] = P(pi_j | pi_i)."""
    n_b = len(grid)
    t_mat = np.zeros((n_b, n_b))
    for i, pi in enumerate(grid):
        pi_g = _update_good(pi)
        pi_b = _update_bad(pi)
        p_good = pi * (1 - MU_G) + (1 - pi) * MU_G
        _add_interp(t_mat[i], grid, pi_g, p_good)
        _add_interp(t_mat[i], grid, pi_b, 1 - p_good)
    return t_mat


def _add_interp(row, grid, pi_target, weight):
    """Add interpolation weights for pi_target to a transition row."""
    n_b = len(grid)
    idx = np.searchsorted(grid, pi_target, side="right") - 1
    idx = max(0, min(idx, n_b - 2))
    frac = (pi_target - grid[idx]) / (grid[idx + 1] - grid[idx])
    frac = np.clip(frac, 0.0, 1.0)
    row[idx] += weight * (1 - frac)
    row[idx + 1] += weight * frac


# =====
# Howard improvement (multi-sweep) solver
# =====
def solve_howard(alpha, treatment, sweeps_per_update=10, t_max=T_MAX):
    """Value iteration with multiple V-sweeps per sigma update."""
    grid, v_table, sigma_table = _init_tables(t_max)
    for _ in range(MAX_ITER):
        v_old = {n: v_table[n].copy() for n in v_table}
        w = _compute_w_updating_sigma(
            grid, alpha, treatment, v_table, sigma_table
        )
        _update_v_from_w(grid, v_table, w)
        _do_extra_sweeps(
            sweeps_per_update - 1, grid, alpha, treatment,
            v_table, sigma_table,
        )
        if _check_convergence(v_table, v_old) < TOL:
            break
    return _make_result(sigma_table, v_table, grid)


def _do_extra_sweeps(n_sweeps, grid, alpha, treatment, v_table, sigma_table):
    """Run additional V-sweeps with fixed sigma."""
    for _ in range(n_sweeps):
        w = _compute_w_fixed_sigma(
            grid, alpha, treatment, v_table, sigma_table
        )
        _update_v_from_w(grid, v_table, w)


def _compute_w_updating_sigma(grid, alpha, treatment, v_table, sigma_table):
    """Compute W while updating sigma."""
    return _compute_w_core(grid, alpha, treatment, v_table, sigma_table, True)


def _compute_w_fixed_sigma(grid, alpha, treatment, v_table, sigma_table):
    """Compute W with fixed sigma."""
    return _compute_w_core(
        grid, alpha, treatment, v_table, sigma_table, False
    )


def _compute_w_core(grid, alpha, treatment, v_table, sigma_table, update_sig):
    """Core W computation, optionally updating sigma."""
    w_table = {n: np.zeros(len(grid)) for n in range(1, N_INVESTORS + 1)}
    for n in range(1, N_INVESTORS + 1):
        for i, pi_prime in enumerate(grid):
            if update_sig:
                sig = find_sigma(
                    n, pi_prime, alpha, treatment, v_table, grid
                )
                sigma_table[n][i] = sig
            else:
                sig = sigma_table[n][i]
            u_s = compute_u_sell(n, sig, alpha, treatment)
            u_h = compute_u_hold(
                n, pi_prime, sig, alpha, treatment, v_table, grid
            )
            w_table[n][i] = sig * u_s + (1 - sig) * u_h
    return w_table


# =====
# Discount convention variant
# =====
def solve_discount_variant(alpha, treatment, t_max=T_MAX):
    """V = lambda*E[terminal] + (1-lambda)*E[W], H has no lambda."""
    grid, v_table, sigma_table = _init_tables(t_max)
    for _ in range(MAX_ITER):
        v_old = {n: v_table[n].copy() for n in v_table}
        w = _compute_w_discount(grid, alpha, treatment, v_table, sigma_table)
        _update_v_discount(grid, alpha, treatment, v_table, w)
        if _check_convergence(v_table, v_old) < TOL:
            break
    return _make_result(sigma_table, v_table, grid)


def _find_sigma_discount(n, pi_prime, alpha, treatment, v_table, grid):
    """Find sigma using discount variant U_hold."""
    if n == 1:
        return 0.0
    u_s0 = compute_u_sell(n, 0.0, alpha, treatment)
    u_h0 = _u_hold_discount(n, pi_prime, 0.0, v_table, grid)
    if u_s0 <= u_h0:
        return 0.0
    u_s1 = compute_u_sell(n, 1.0, alpha, treatment)
    u_h1 = _u_hold_discount(n, pi_prime, 1.0, v_table, grid)
    if u_s1 >= u_h1:
        return 1.0

    def diff(s):
        return compute_u_sell(n, s, alpha, treatment) - _u_hold_discount(
            n, pi_prime, s, v_table, grid
        )

    return brentq(diff, 0.0, 1.0, xtol=1e-12)


def _u_hold_discount(n, pi_prime, sigma, v_table, grid):
    """U_hold using continuation V directly (no lambda in H)."""
    total = 0.0
    for j in range(n):
        prob = comb(n - 1, j) * sigma**j * (1 - sigma) ** (n - 1 - j)
        m = n - j
        total += prob * np.interp(pi_prime, grid, v_table[m])
    return total


def _compute_w_discount(grid, alpha, treatment, v_table, sigma_table):
    """Compute W for discount variant."""
    w_table = {n: np.zeros(len(grid)) for n in range(1, N_INVESTORS + 1)}
    for n in range(1, N_INVESTORS + 1):
        for i, pi_prime in enumerate(grid):
            sig = _find_sigma_discount(
                n, pi_prime, alpha, treatment, v_table, grid
            )
            sigma_table[n][i] = sig
            u_s = compute_u_sell(n, sig, alpha, treatment)
            u_h = _u_hold_discount(n, pi_prime, sig, v_table, grid)
            w_table[n][i] = sig * u_s + (1 - sig) * u_h
    return w_table


def _update_v_discount(grid, alpha, treatment, v_table, w_table):
    """V = lambda*E[terminal] + (1-lambda)*E[W] in discount variant."""
    for n in range(1, N_INVESTORS + 1):
        for i, pi in enumerate(grid):
            pi_g, pi_b = _update_good(pi), _update_bad(pi)
            p_good = pi * (1 - MU_G) + (1 - pi) * MU_G
            w_g = np.interp(pi_g, grid, w_table[n])
            w_b = np.interp(pi_b, grid, w_table[n])
            term = pi * crra_utility(FINAL_VALUE, alpha) + (
                1 - pi) * forced_liquidation(n, alpha, treatment)
            v_table[n][i] = LAMBDA * term + (1 - LAMBDA) * (
                p_good * w_g + (1 - p_good) * w_b
            )


# =====
# Alternative W formula (W = U_sell at indifference)
# =====
def solve_w_at_indifference(alpha, treatment, t_max=T_MAX):
    """Test W = U_sell = U_hold at indifference instead of mixture."""
    grid, v_table, sigma_table = _init_tables(t_max)
    for _ in range(MAX_ITER):
        v_old = {n: v_table[n].copy() for n in v_table}
        w = _compute_w_indifference(
            grid, alpha, treatment, v_table, sigma_table
        )
        _update_v_from_w(grid, v_table, w)
        if _check_convergence(v_table, v_old) < TOL:
            break
    return _make_result(sigma_table, v_table, grid)


def _compute_w_indifference(grid, alpha, treatment, v_table, sigma_table):
    """Compute W using U_sell at interior sigma (indifference)."""
    w_table = {n: np.zeros(len(grid)) for n in range(1, N_INVESTORS + 1)}
    for n in range(1, N_INVESTORS + 1):
        for i, pi_prime in enumerate(grid):
            sig = find_sigma(n, pi_prime, alpha, treatment, v_table, grid)
            sigma_table[n][i] = sig
            u_s = compute_u_sell(n, sig, alpha, treatment)
            u_h = compute_u_hold(
                n, pi_prime, sig, alpha, treatment, v_table, grid
            )
            # At indifference: W = U_sell = U_hold
            if 0 < sig < 1:
                w_table[n][i] = u_s
            else:
                w_table[n][i] = sig * u_s + (1 - sig) * u_h
    return w_table
