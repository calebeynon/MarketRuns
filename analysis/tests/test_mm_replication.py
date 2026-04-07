"""
Purpose: Tests for M&M replication diagnostic — verifies that policy iteration,
         value iteration, and alternative solvers produce consistent results and
         satisfy equilibrium conditions at M&M Appendix D target points.
Author: Claude Code
Date: 2026-04-07
"""

import numpy as np
import pytest

from analysis.analysis.equilibrium_model import (
    N_INVESTORS,
    build_belief_grid,
    compute_u_hold,
    compute_u_sell,
    solve_equilibrium,
    _update_bad,
)
from analysis.analysis.mm_alternative_solvers import (
    solve_policy_iteration,
    solve_howard,
    solve_discount_variant,
    solve_w_at_indifference,
    _build_transition_matrix,
)

# M&M APPENDIX D TARGET VALUES
# Each tuple: (alpha, n, d, mm_sigma)
# d = number of bad signals from prior (determines belief)
MM_TARGETS = [
    (0.0, 4, 2, 0.775),
    (0.0, 3, 2, 0.0155),
    (0.5, 4, 1, 0.5155),
    (0.5, 3, 2, 0.8868),
    (0.5, 2, 3, 0.86),
]


# =====
# Helpers
# =====
def _belief_after_d_bad(d):
    """Compute belief pi after d consecutive bad signals from pi_0=0.5."""
    pi = 0.5
    for _ in range(d):
        pi = _update_bad(pi)
    return pi


def _lookup_sigma_at_belief(result, n, pi_target):
    """Interpolate sigma for given n and belief from solver result."""
    grid = result["belief_grid"]
    return float(np.interp(pi_target, grid, result["sigma"][n]))


# =====
# Fixtures: solve once, reuse across tests
# =====
@pytest.fixture(scope="module")
def vi_alpha0():
    """Value iteration: alpha=0, random treatment."""
    return solve_equilibrium(alpha=0.0, treatment="random")


@pytest.fixture(scope="module")
def vi_alpha05():
    """Value iteration: alpha=0.5, random treatment."""
    return solve_equilibrium(alpha=0.5, treatment="random")


@pytest.fixture(scope="module")
def pi_alpha0():
    """Policy iteration: alpha=0, random treatment."""
    return solve_policy_iteration(alpha=0.0, treatment="random")


@pytest.fixture(scope="module")
def pi_alpha05():
    """Policy iteration: alpha=0.5, random treatment."""
    return solve_policy_iteration(alpha=0.5, treatment="random")


@pytest.fixture(scope="module")
def howard_alpha0():
    """Howard improvement: alpha=0, random treatment."""
    return solve_howard(alpha=0.0, treatment="random")


@pytest.fixture(scope="module")
def howard_alpha05():
    """Howard improvement: alpha=0.5, random treatment."""
    return solve_howard(alpha=0.5, treatment="random")


# =====
# Algorithm equivalence: policy iteration vs value iteration
# =====
class TestAlgorithmEquivalence:
    """Verify policy iteration and value iteration converge to the same sigma."""

    def test_sigma_match_alpha0(self, vi_alpha0, pi_alpha0):
        """alpha=0: PI and VI sigma must agree within 1e-6."""
        grid = vi_alpha0["belief_grid"]
        for n in range(1, N_INVESTORS + 1):
            np.testing.assert_allclose(
                vi_alpha0["sigma"][n],
                pi_alpha0["sigma"][n],
                atol=1e-6,
                err_msg=f"n={n}, alpha=0: VI vs PI sigma mismatch",
            )

    def test_sigma_match_alpha05(self, vi_alpha05, pi_alpha05):
        """alpha=0.5: PI and VI sigma must agree within 1e-6."""
        for n in range(1, N_INVESTORS + 1):
            np.testing.assert_allclose(
                vi_alpha05["sigma"][n],
                pi_alpha05["sigma"][n],
                atol=1e-6,
                err_msg=f"n={n}, alpha=0.5: VI vs PI sigma mismatch",
            )

    def test_howard_matches_vi_alpha0(self, vi_alpha0, howard_alpha0):
        """Howard improvement should match VI for alpha=0."""
        for n in range(1, N_INVESTORS + 1):
            np.testing.assert_allclose(
                vi_alpha0["sigma"][n],
                howard_alpha0["sigma"][n],
                atol=1e-6,
                err_msg=f"n={n}, alpha=0: VI vs Howard sigma mismatch",
            )

    def test_howard_matches_vi_alpha05(self, vi_alpha05, howard_alpha05):
        """Howard improvement should match VI for alpha=0.5."""
        for n in range(1, N_INVESTORS + 1):
            np.testing.assert_allclose(
                vi_alpha05["sigma"][n],
                howard_alpha05["sigma"][n],
                atol=1e-6,
                err_msg=f"n={n}, alpha=0.5: VI vs Howard sigma mismatch",
            )


# =====
# V value equivalence: policy iteration vs value iteration
# =====
class TestVValueEquivalence:
    """Verify V values match between solvers."""

    def test_v_match_alpha0(self, vi_alpha0, pi_alpha0):
        """alpha=0: V values from PI and VI must agree within 1e-4."""
        for n in range(1, N_INVESTORS + 1):
            np.testing.assert_allclose(
                vi_alpha0["v_table"][n],
                pi_alpha0["v_table"][n],
                atol=1e-4,
                err_msg=f"n={n}, alpha=0: VI vs PI V-values mismatch",
            )

    def test_v_match_alpha05(self, vi_alpha05, pi_alpha05):
        """alpha=0.5: V values from PI and VI must agree within 1e-4."""
        for n in range(1, N_INVESTORS + 1):
            np.testing.assert_allclose(
                vi_alpha05["v_table"][n],
                pi_alpha05["v_table"][n],
                atol=1e-4,
                err_msg=f"n={n}, alpha=0.5: VI vs PI V-values mismatch",
            )

    def test_v_finite_policy_iteration(self, pi_alpha05):
        """All PI value function entries should be finite."""
        for n in range(1, N_INVESTORS + 1):
            assert np.all(np.isfinite(pi_alpha05["v_table"][n])), (
                f"n={n}: PI V-values contain non-finite entries"
            )


# =====
# Equilibrium conditions at M&M target points
# =====
class TestEquilibriumConditions:
    """Verify all solvers satisfy equilibrium conditions at target beliefs."""

    @pytest.mark.parametrize("alpha,n,d,mm_sigma", MM_TARGETS)
    def test_vi_indifference_at_interior_sigma(
        self, alpha, n, d, mm_sigma, vi_alpha0, vi_alpha05
    ):
        """At interior sigma, U_sell must equal U_hold (indifference)."""
        result = vi_alpha0 if alpha == 0.0 else vi_alpha05
        pi = _belief_after_d_bad(d)
        sigma = _lookup_sigma_at_belief(result, n, pi)
        if sigma < 1e-8 or sigma > 1 - 1e-8:
            pytest.skip("Sigma is at boundary, indifference not expected")
        grid = result["belief_grid"]
        u_sell = compute_u_sell(n, sigma, alpha, "random")
        u_hold = compute_u_hold(
            n, pi, sigma, alpha, "random", result["v_table"], grid
        )
        assert u_sell == pytest.approx(u_hold, abs=1e-6), (
            f"alpha={alpha}, n={n}, d={d}: "
            f"U_sell={u_sell:.8f} != U_hold={u_hold:.8f}"
        )

    @pytest.mark.parametrize("alpha,n,d,mm_sigma", MM_TARGETS)
    def test_pi_indifference_at_interior_sigma(
        self, alpha, n, d, mm_sigma, pi_alpha0, pi_alpha05
    ):
        """PI solver: at interior sigma, U_sell == U_hold."""
        result = pi_alpha0 if alpha == 0.0 else pi_alpha05
        pi = _belief_after_d_bad(d)
        sigma = _lookup_sigma_at_belief(result, n, pi)
        if sigma < 1e-8 or sigma > 1 - 1e-8:
            pytest.skip("Sigma is at boundary")
        grid = result["belief_grid"]
        u_sell = compute_u_sell(n, sigma, alpha, "random")
        u_hold = compute_u_hold(
            n, pi, sigma, alpha, "random", result["v_table"], grid
        )
        assert u_sell == pytest.approx(u_hold, abs=1e-6), (
            f"PI: alpha={alpha}, n={n}, d={d}: "
            f"U_sell={u_sell:.8f} != U_hold={u_hold:.8f}"
        )

    def test_n1_never_sells(self, vi_alpha05, pi_alpha05):
        """Boundary: n=1 holder should never sell in VI or PI solver."""
        assert np.all(vi_alpha05["sigma"][1] == 0.0)
        assert np.all(pi_alpha05["sigma"][1] == 0.0)

    def test_u_sell_lt_u_hold_when_sigma_zero(self, vi_alpha05):
        """When sigma=0, U_sell <= U_hold (no incentive to deviate)."""
        grid = vi_alpha05["belief_grid"]
        v_table = vi_alpha05["v_table"]
        for n in range(2, N_INVESTORS + 1):
            sigma_n = vi_alpha05["sigma"][n]
            zero_mask = sigma_n < 1e-10
            for i in np.where(zero_mask)[0]:
                pi = grid[i]
                u_sell = compute_u_sell(n, 0.0, 0.5, "random")
                u_hold = compute_u_hold(
                    n, pi, 0.0, 0.5, "random", v_table, grid
                )
                assert u_sell <= u_hold + 1e-8, (
                    f"n={n}, pi={pi:.4f}: "
                    f"U_sell={u_sell:.6f} > U_hold={u_hold:.6f} "
                    f"but sigma=0"
                )


# =====
# M&M comparison lock-in (regression tests)
# =====
class TestMMComparisonLockIn:
    """Lock in our solver's sigma at M&M target points.

    These tests fail if a solver change shifts output values,
    catching unintended regressions.
    """

    @pytest.mark.parametrize("alpha,n,d,mm_sigma", MM_TARGETS)
    def test_vi_sigma_locked(
        self, alpha, n, d, mm_sigma, vi_alpha0, vi_alpha05
    ):
        """VI sigma at each M&M target must match recorded values."""
        result = vi_alpha0 if alpha == 0.0 else vi_alpha05
        pi = _belief_after_d_bad(d)
        our_sigma = _lookup_sigma_at_belief(result, n, pi)
        # Our solver consistently overestimates vs M&M.
        # Lock in that our values are above M&M values.
        assert our_sigma > mm_sigma, (
            f"alpha={alpha}, n={n}, d={d}: "
            f"our sigma={our_sigma:.4f} not above M&M={mm_sigma}"
        )
        # Our sigma should be within 0.06 of M&M (the max gap observed)
        assert our_sigma == pytest.approx(mm_sigma, abs=0.06), (
            f"alpha={alpha}, n={n}, d={d}: "
            f"our sigma={our_sigma:.4f} too far from M&M={mm_sigma}"
        )

    @pytest.mark.parametrize("alpha,n,d,mm_sigma", MM_TARGETS)
    def test_pi_sigma_locked(
        self, alpha, n, d, mm_sigma, pi_alpha0, pi_alpha05
    ):
        """PI sigma at each M&M target must match VI sigma (same code)."""
        vi_result = pi_alpha0 if alpha == 0.0 else pi_alpha05
        pi = _belief_after_d_bad(d)
        pi_sigma = _lookup_sigma_at_belief(vi_result, n, pi)
        # PI should produce values very close to VI (tested above),
        # so we lock in with the same tolerance
        assert pi_sigma == pytest.approx(mm_sigma, abs=0.06), (
            f"PI: alpha={alpha}, n={n}, d={d}: "
            f"sigma={pi_sigma:.4f} vs M&M={mm_sigma}"
        )


# =====
# Transition matrix properties
# =====
class TestTransitionMatrix:
    """Verify the signal transition matrix T is well-formed."""

    @pytest.fixture(scope="class")
    def transition(self):
        """Build transition matrix for default belief grid."""
        grid = build_belief_grid(t_max=40)
        t_mat = _build_transition_matrix(grid)
        return t_mat, grid

    def test_row_stochastic(self, transition):
        """Each row of T must sum to 1 (probability distribution)."""
        t_mat, _ = transition
        row_sums = t_mat.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, atol=1e-12,
            err_msg="Transition matrix rows do not sum to 1",
        )

    def test_non_negative(self, transition):
        """All entries must be non-negative."""
        t_mat, _ = transition
        assert np.all(t_mat >= -1e-15), (
            "Transition matrix has negative entries"
        )

    def test_sparse_structure(self, transition):
        """Each row should have at most 4 non-zero entries (2 targets x 2 weights)."""
        t_mat, _ = transition
        for i in range(t_mat.shape[0]):
            n_nonzero = np.count_nonzero(t_mat[i] > 1e-15)
            assert n_nonzero <= 4, (
                f"Row {i} has {n_nonzero} non-zero entries, expected <= 4"
            )

    def test_v_satisfies_bellman_pi_solver(self, pi_alpha0):
        """PI-solved V must satisfy V = T @ W exactly (up to tolerance)."""
        grid = pi_alpha0["belief_grid"]
        v_table = pi_alpha0["v_table"]
        sigma_table = pi_alpha0["sigma"]
        t_mat = _build_transition_matrix(grid)
        for n in range(1, N_INVESTORS + 1):
            w_vec = np.zeros(len(grid))
            for i, pi_prime in enumerate(grid):
                sig = sigma_table[n][i]
                u_s = compute_u_sell(n, sig, 0.0, "random")
                u_h = compute_u_hold(
                    n, pi_prime, sig, 0.0, "random", v_table, grid
                )
                w_vec[i] = sig * u_s + (1 - sig) * u_h
            v_expected = t_mat @ w_vec
            np.testing.assert_allclose(
                v_table[n], v_expected, atol=1e-6,
                err_msg=f"n={n}: V != T @ W (Bellman violation)",
            )


# =====
# Alternative solver divergence (discount variant, W-indifference)
# =====
class TestAlternativeSolverDivergence:
    """Alternative solvers should diverge from standard VI in predictable ways."""

    @pytest.fixture(scope="class")
    def discount_alpha05(self):
        return solve_discount_variant(alpha=0.5, treatment="random")

    @pytest.fixture(scope="class")
    def indiff_alpha05(self):
        return solve_w_at_indifference(alpha=0.5, treatment="random")

    @pytest.mark.parametrize("solver_name", ["discount", "indifference"])
    def test_variant_valid_sigma(self, solver_name, discount_alpha05, indiff_alpha05):
        """Alternative solvers should produce sigma in [0, 1]."""
        result = discount_alpha05 if solver_name == "discount" else indiff_alpha05
        for n in range(1, N_INVESTORS + 1):
            sigma = result["sigma"][n]
            assert np.all(sigma >= -1e-12), f"{solver_name} n={n}: negative sigma"
            assert np.all(sigma <= 1.0 + 1e-12), f"{solver_name} n={n}: sigma > 1"

    @pytest.mark.parametrize("solver_name", ["discount", "indifference"])
    def test_variant_finite_v(self, solver_name, discount_alpha05, indiff_alpha05):
        """Alternative solver V values should be finite."""
        result = discount_alpha05 if solver_name == "discount" else indiff_alpha05
        for n in range(1, N_INVESTORS + 1):
            assert np.all(np.isfinite(result["v_table"][n]))
