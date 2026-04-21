"""
Purpose: Unit tests for equilibrium_model.py — Bellman equation solver
         for the Magnani & Munro (2020) market runs model.
Author: Claude Code
Date: 2026-04-05
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.analysis.equilibrium_model import (
    build_belief_grid,
    compute_u_hold,
    compute_u_sell,
    crra_utility,
    find_sigma,
    forced_liquidation,
    rho,
    solve_equilibrium,
    FINAL_VALUE,
    MU_B,
    MU_G,
    N_INVESTORS,
    PI_0,
    PRICES,
    _update_bad,
    _update_good,
)

SIGMA_GRID_PARQUET = Path("datastore/derived/equilibrium_sigma_grid.parquet")


# =====
# Belief updating tests
# =====
class TestBeliefUpdating:
    """Bayesian belief updates after good and bad signals."""

    def test_good_signal_increases_belief(self):
        """A good signal should increase pi (belief that state is Good)."""
        pi = 0.5
        assert _update_good(pi) > pi

    def test_bad_signal_decreases_belief(self):
        """A bad signal should decrease pi."""
        pi = 0.5
        assert _update_bad(pi) < pi

    def test_symmetric_updates_from_half(self):
        """From pi=0.5, good signal -> 0.675, bad signal -> 0.325."""
        pi_g = _update_good(0.5)
        pi_b = _update_bad(0.5)
        assert pi_g == pytest.approx(0.675, abs=1e-10)
        assert pi_b == pytest.approx(0.325, abs=1e-10)

    def test_good_and_bad_are_symmetric(self):
        """pi_g(0.5) + pi_b(0.5) = 1 (symmetry around 0.5)."""
        pi_g = _update_good(0.5)
        pi_b = _update_bad(0.5)
        assert pi_g + pi_b == pytest.approx(1.0, abs=1e-12)

    def test_monotonicity_good_signal(self):
        """Good update is monotonically increasing in pi."""
        pis = [0.1, 0.3, 0.5, 0.7, 0.9]
        updated = [_update_good(p) for p in pis]
        for i in range(len(updated) - 1):
            assert updated[i] < updated[i + 1]

    def test_monotonicity_bad_signal(self):
        """Bad update is monotonically increasing in pi."""
        pis = [0.1, 0.3, 0.5, 0.7, 0.9]
        updated = [_update_bad(p) for p in pis]
        for i in range(len(updated) - 1):
            assert updated[i] < updated[i + 1]


# =====
# Belief grid tests
# =====
class TestBeliefGrid:
    """Construction of the reachable belief grid."""

    def test_grid_contains_prior(self):
        """Grid must contain the initial belief pi_0 = 0.5."""
        grid = build_belief_grid(t_max=10)
        assert PI_0 in grid

    def test_grid_is_sorted(self):
        """Grid must be sorted in ascending order."""
        grid = build_belief_grid(t_max=10)
        assert np.all(np.diff(grid) > 0)

    def test_grid_values_in_unit_interval(self):
        """All beliefs must be in (0, 1)."""
        grid = build_belief_grid(t_max=40)
        assert np.all(grid > 0)
        assert np.all(grid < 1)

    def test_grid_grows_with_t_max(self):
        """Larger t_max produces a (weakly) larger grid."""
        small = build_belief_grid(t_max=5)
        large = build_belief_grid(t_max=20)
        assert len(large) >= len(small)


# =====
# CRRA utility tests
# =====
class TestCRRAUtility:
    """CRRA utility function u(x) = x^(1-a)/(1-a)."""

    def test_risk_neutral(self):
        """alpha=0 (risk neutral): u(x) = x."""
        assert crra_utility(5.0, 0.0) == pytest.approx(5.0)
        assert crra_utility(100.0, 0.0) == pytest.approx(100.0)

    def test_alpha_half(self):
        """alpha=0.5: u(x) = x^0.5 / 0.5 = 2*sqrt(x)."""
        assert crra_utility(4.0, 0.5) == pytest.approx(2 * 2.0)  # 2*sqrt(4) = 4.0
        assert crra_utility(9.0, 0.5) == pytest.approx(2 * 3.0)  # 2*sqrt(9) = 6.0

    def test_log_utility(self):
        """alpha=1: u(x) = ln(x)."""
        assert crra_utility(1.0, 1.0) == pytest.approx(0.0)
        assert crra_utility(np.e, 1.0) == pytest.approx(1.0)

    def test_monotonicity(self):
        """u(a) < u(b) for a < b, any alpha >= 0."""
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0, 2.0]:
            assert crra_utility(2.0, alpha) < crra_utility(4.0, alpha)


# =====
# Rho (voluntary selling payoff) tests
# =====
class TestRho:
    """Payoff from voluntary selling."""

    def test_single_seller_random_equals_average(self):
        """With k=1 seller, random and average are identical."""
        for alpha in [0.0, 0.5, 1.0]:
            r_rand = rho(4, 1, alpha, "random")
            r_avg = rho(4, 1, alpha, "average")
            assert r_rand == pytest.approx(r_avg, abs=1e-12)

    def test_single_seller_gets_top_price(self):
        """Single seller from n=4 holders gets p_4 = 8."""
        assert rho(4, 1, 0.0, "random") == pytest.approx(8.0)
        assert rho(3, 1, 0.0, "random") == pytest.approx(6.0)

    def test_jensen_inequality_average_gt_random(self):
        """For alpha>0, k>1: rho_average > rho_random (Jensen's inequality)."""
        for n in [2, 3, 4]:
            for k in range(2, n + 1):
                r_rand = rho(n, k, 0.5, "random")
                r_avg = rho(n, k, 0.5, "average")
                assert r_avg > r_rand, (
                    f"Jensen violated: n={n}, k={k}"
                )

    def test_risk_neutral_equal_expected_values(self):
        """alpha=0: random and average give same payoff (no Jensen effect)."""
        for n in [2, 3, 4]:
            for k in range(1, n + 1):
                r_rand = rho(n, k, 0.0, "random")
                r_avg = rho(n, k, 0.0, "average")
                assert r_rand == pytest.approx(r_avg, abs=1e-12)

    def test_all_sell_from_n4(self):
        """All 4 sell from n=4: prices are 8,6,4,2; avg payoff = u(5)."""
        # risk neutral: average of 8,6,4,2 = 5
        assert rho(4, 4, 0.0, "random") == pytest.approx(5.0)
        assert rho(4, 4, 0.0, "average") == pytest.approx(5.0)


# =====
# Forced liquidation tests
# =====
class TestForcedLiquidation:
    """L(m): expected utility from forced liquidation with m holders."""

    def test_single_holder(self):
        """L(1) = u(p_1) = u(2) for both treatments."""
        for treatment in ["random", "average"]:
            assert forced_liquidation(1, 0.0, treatment) == pytest.approx(2.0)
            assert forced_liquidation(1, 0.5, treatment) == pytest.approx(
                crra_utility(2.0, 0.5)
            )

    def test_jensen_inequality_forced(self):
        """For alpha>0, m>1: L_average > L_random (Jensen's inequality)."""
        for m in [2, 3, 4]:
            l_rand = forced_liquidation(m, 0.5, "random")
            l_avg = forced_liquidation(m, 0.5, "average")
            assert l_avg > l_rand, f"Jensen violated: m={m}"

    def test_risk_neutral_treatments_equal(self):
        """alpha=0: forced liquidation identical across treatments."""
        for m in range(1, N_INVESTORS + 1):
            l_rand = forced_liquidation(m, 0.0, "random")
            l_avg = forced_liquidation(m, 0.0, "average")
            assert l_rand == pytest.approx(l_avg, abs=1e-12)


# =====
# Equilibrium condition tests
# =====
class TestEquilibriumConditions:
    """Structural properties of the equilibrium solution."""

    @pytest.fixture(scope="class")
    def eq_risk_averse_random(self):
        """Solve equilibrium: alpha=0.5, random treatment."""
        return solve_equilibrium(alpha=0.5, treatment="random")

    @pytest.fixture(scope="class")
    def eq_risk_averse_average(self):
        """Solve equilibrium: alpha=0.5, average treatment."""
        return solve_equilibrium(alpha=0.5, treatment="average")

    @pytest.fixture(scope="class")
    def eq_risk_neutral(self):
        """Solve equilibrium: alpha=0, random treatment."""
        return solve_equilibrium(alpha=0.0, treatment="random")

    def test_last_holder_never_sells(self, eq_risk_averse_random):
        """sigma(1, pi) = 0 for all pi (boundary condition)."""
        sigma_1 = eq_risk_averse_random["sigma"][1]
        assert np.all(sigma_1 == 0.0)

    def test_sigma_weakly_decreasing_in_pi(self, eq_risk_averse_random):
        """Higher belief in Good state -> less selling."""
        grid = eq_risk_averse_random["belief_grid"]
        for n in range(2, N_INVESTORS + 1):
            sigma_n = eq_risk_averse_random["sigma"][n]
            # Check weakly decreasing (allow small numerical tolerance)
            diffs = np.diff(sigma_n)
            assert np.all(diffs <= 1e-8), (
                f"n={n}: sigma not weakly decreasing in pi"
            )

    def test_thresholds_increase_with_n(self, eq_risk_averse_random):
        """Selling threshold pi* increases with n: pi*(2) < pi*(3) < pi*(4)."""
        grid = eq_risk_averse_random["belief_grid"]
        thresholds = {}
        for n in range(2, N_INVESTORS + 1):
            sigma_n = eq_risk_averse_random["sigma"][n]
            selling_beliefs = grid[sigma_n > 0.01]
            if len(selling_beliefs) > 0:
                thresholds[n] = selling_beliefs[-1]  # highest pi where selling
            else:
                thresholds[n] = 0.0
        # More holders -> willing to sell at higher beliefs
        for n in range(2, N_INVESTORS):
            assert thresholds[n] <= thresholds[n + 1] + 1e-8, (
                f"Threshold for n={n} ({thresholds[n]:.4f}) > "
                f"n={n+1} ({thresholds[n+1]:.4f})"
            )

    def test_sigma_in_unit_interval(self, eq_risk_averse_random):
        """All sigma values must be in [0, 1]."""
        for n in range(1, N_INVESTORS + 1):
            sigma_n = eq_risk_averse_random["sigma"][n]
            assert np.all(sigma_n >= -1e-12)
            assert np.all(sigma_n <= 1.0 + 1e-12)


# =====
# Risk neutrality equivalence tests
# =====
class TestRiskNeutralEquivalence:
    """alpha=0: random and average treatments must produce identical sigma."""

    @pytest.fixture(scope="class")
    def eq_neutral_random(self):
        return solve_equilibrium(alpha=0.0, treatment="random")

    @pytest.fixture(scope="class")
    def eq_neutral_average(self):
        return solve_equilibrium(alpha=0.0, treatment="average")

    def test_sigma_identical_across_treatments(
        self, eq_neutral_random, eq_neutral_average
    ):
        """No Jensen effect under risk neutrality."""
        for n in range(1, N_INVESTORS + 1):
            np.testing.assert_allclose(
                eq_neutral_random["sigma"][n],
                eq_neutral_average["sigma"][n],
                atol=1e-6,
                err_msg=f"n={n}: sigma differs across treatments at alpha=0",
            )


# =====
# M&M Table 2 validation (approximate)
# =====
class TestMagnaniMunroTable2:
    """Approximate validation against Magnani & Munro (2020) Table 2."""

    @pytest.fixture(scope="class")
    def eq_mm(self):
        """Solve with alpha=0.5, random — the baseline M&M parameterization."""
        return solve_equilibrium(alpha=0.5, treatment="random")

    def test_n4_threshold_approx(self, eq_mm):
        """n=4, alpha=0.5, random: threshold pi ~ 0.325 (= 1 - 0.678 M&M)."""
        grid = eq_mm["belief_grid"]
        sigma_4 = eq_mm["sigma"][4]
        # Find highest pi where sigma > 0.01
        selling_beliefs = grid[sigma_4 > 0.01]
        if len(selling_beliefs) > 0:
            threshold = selling_beliefs[-1]
        else:
            threshold = 0.0
        # M&M report threshold in terms of P(Bad) ~ 0.678, so P(Good) ~ 0.322
        assert threshold == pytest.approx(0.325, abs=0.02), (
            f"n=4 threshold pi={threshold:.4f}, expected ~0.325"
        )

    def test_n2_has_lower_threshold_than_n4(self, eq_mm):
        """n=2 threshold should be lower than n=4 threshold."""
        grid = eq_mm["belief_grid"]

        def get_threshold(n):
            selling = grid[eq_mm["sigma"][n] > 0.01]
            return selling[-1] if len(selling) > 0 else 0.0

        assert get_threshold(2) < get_threshold(4)


# =====
# Solver convergence tests
# =====
class TestSolverConvergence:
    """Value iteration convergence properties."""

    def test_solver_returns_expected_keys(self):
        """Result dict has sigma, v_table, belief_grid."""
        result = solve_equilibrium(alpha=0.0, treatment="random")
        assert "sigma" in result
        assert "v_table" in result
        assert "belief_grid" in result

    def test_v_table_has_all_n(self):
        """V table has entries for n=1..4."""
        result = solve_equilibrium(alpha=0.0, treatment="random")
        for n in range(1, N_INVESTORS + 1):
            assert n in result["v_table"]
            assert n in result["sigma"]

    def test_v_values_finite(self):
        """All value function entries should be finite."""
        result = solve_equilibrium(alpha=0.5, treatment="random")
        for n in range(1, N_INVESTORS + 1):
            assert np.all(np.isfinite(result["v_table"][n]))


# =====
# Wide α grid + persisted σ parquet (issue #117, Task 1 outputs)
# =====
class TestAlphaGridAndSigmaArtifact:
    """ALPHA_VALUES widened to 101 points; σ grid persisted to parquet."""

    def test_alpha_values_wide_grid(self):
        """ALPHA_VALUES must be 101 points in [0, 1], 2-dp rounded, no dupes."""
        from analysis.analysis.simulate_equilibrium import ALPHA_VALUES
        values = list(ALPHA_VALUES)
        assert len(values) == 101
        assert min(values) == pytest.approx(0.0, abs=1e-12)
        assert max(values) == pytest.approx(1.0, abs=1e-12)
        # All at most 2 decimal places
        for v in values:
            assert round(v, 2) == pytest.approx(v, abs=1e-12), (
                f"ALPHA_VALUES contains non-2dp value: {v}"
            )
        assert len(set(values)) == len(values), "ALPHA_VALUES has duplicates"

    def test_sigma_grid_parquet_schema(self):
        """σ grid parquet: schema, value ranges, coverage of treatments and n."""
        if not SIGMA_GRID_PARQUET.exists():
            pytest.skip(
                "parquet not yet generated; run simulate_equilibrium.py first"
            )
        df = pd.read_parquet(SIGMA_GRID_PARQUET)
        assert set(df.columns) == {"treatment", "alpha", "n", "pi", "sigma"}
        assert df["sigma"].min() >= -1e-9
        assert df["sigma"].max() <= 1.0 + 1e-9
        assert set(df["n"].unique()) == {1, 2, 3, 4}
        assert set(df["treatment"].unique()) == {"random", "average"}
        # α grid must have 101 unique values
        assert df["alpha"].nunique() == 101
