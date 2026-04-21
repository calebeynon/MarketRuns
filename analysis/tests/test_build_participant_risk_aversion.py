"""
Purpose: Unit tests for build_participant_risk_aversion.py — the per-participant
         CRRA alpha MLE builder (issue #117).
Author: Claude Code
Date: 2026-04-21
"""

from math import log

import numpy as np
import pandas as pd
import pytest

# The builder module does not exist until Task 2 finishes. Use importorskip so
# test collection does not fail during Phase 1; Phase 3 runs the file end-to-end.
brab = pytest.importorskip(
    "analysis.derived.build_participant_risk_aversion",
    reason="builder module not yet created (Task 2 pending)",
)


# =====
# Pinned numerical expectations (hand-computed once, pasted as literals)
# =====
# alpha_task(a=5)  = ln(2.5) / ln((20 + 1.5*5)  / (20 - 5))  = ln(2.5)/ln(27.5/15)
ALPHA_TASK_A5 = 1.5116921430427592
# alpha_task(a=10) = ln(2.5) / ln((20 + 1.5*10) / (20 - 10)) = ln(2.5)/ln(3.5)
ALPHA_TASK_A10 = 0.7314158822675504
# alpha_task(a=15) = ln(2.5) / ln((20 + 1.5*15) / (20 - 15)) = ln(2.5)/ln(8.5)
ALPHA_TASK_A15 = 0.42816000154742495


# =====
# Gneezy–Potters alpha inversion
# =====
class TestAlphaTaskInversion:
    """compute_alpha_task: invert Gneezy–Potters allocate -> CRRA alpha."""

    def test_alpha_task_at_a_5(self):
        """allocate=5 maps to alpha = ln(2.5)/ln(1.8333…)."""
        alpha, edge = brab.compute_alpha_task(5.0)
        assert alpha == pytest.approx(ALPHA_TASK_A5, abs=1e-10)
        assert edge is False

    def test_alpha_task_at_a_10(self):
        """allocate=10 maps to alpha = ln(2.5)/ln(3.5)."""
        alpha, edge = brab.compute_alpha_task(10.0)
        assert alpha == pytest.approx(ALPHA_TASK_A10, abs=1e-10)
        assert edge is False

    def test_alpha_task_at_a_15(self):
        """allocate=15 maps to alpha = ln(2.5)/ln(8.5)."""
        alpha, edge = brab.compute_alpha_task(15.0)
        assert alpha == pytest.approx(ALPHA_TASK_A15, abs=1e-10)
        assert edge is False

    def test_alpha_task_edge_a_0(self):
        """allocate=0 is a corner — edge flag True, alpha NaN."""
        alpha, edge = brab.compute_alpha_task(0.0)
        assert np.isnan(alpha)
        assert edge is True

    def test_alpha_task_edge_a_20(self):
        """allocate=20 is a corner — edge flag True, alpha NaN."""
        alpha, edge = brab.compute_alpha_task(20.0)
        assert np.isnan(alpha)
        assert edge is True

    def test_alpha_task_edge_nan(self):
        """NaN allocate is treated as edge, not a crash."""
        alpha, edge = brab.compute_alpha_task(float("nan"))
        assert np.isnan(alpha)
        assert edge is True

    @pytest.mark.parametrize("allocate", [-1.0, 25.0, -0.01, 20.01])
    def test_alpha_task_out_of_range_raises(self, allocate):
        """Allocations outside [0, 20] must fail loudly."""
        with pytest.raises(ValueError):
            brab.compute_alpha_task(allocate)


# =====
# Likelihood correctness
# =====
class TestLikelihoodCorrectness:
    """participant_log_likelihood: σ-lookup driven Bernoulli LL."""

    @pytest.fixture
    def tiny_sigma_lookup(self):
        """Hand-built σ lookup for treatment='random', alpha=0.30.
        Keys: (treatment, round(alpha, 2), n) -> (pi_sorted, sigma_sorted).
        Contains three (n, π) combos with known σ."""
        return {
            ("random", 0.30, 4): (
                np.array([0.10, 0.30, 0.50]),
                np.array([0.20, 0.50, 0.80]),
            ),
        }

    @pytest.fixture
    def tiny_decisions(self):
        """Three decisions at (n=4, π ∈ {0.10, 0.30, 0.50}), sold=[0, 1, 0]."""
        return pd.DataFrame({
            "n": [4, 4, 4],
            "pi": [0.10, 0.30, 0.50],
            "sold_indicator": [0, 1, 0],
        })

    def test_log_likelihood_closed_form(self, tiny_decisions, tiny_sigma_lookup):
        """LL = log(1-0.2) + log(0.5) + log(1-0.8) at alpha=0.30."""
        expected = log(1 - 0.20) + log(0.50) + log(1 - 0.80)
        got = brab.participant_log_likelihood(
            tiny_decisions, alpha=0.30, sigma_lookup=tiny_sigma_lookup,
            treatment="random",
        )
        assert got == pytest.approx(expected, abs=1e-10)

    def test_log_likelihood_epsilon_floor(self):
        """sigma=0 with sold=1 must floor at log(EPS), not -inf or NaN."""
        lookup = {
            ("random", 0.30, 4): (np.array([0.50]), np.array([0.0])),
        }
        decisions = pd.DataFrame({
            "n": [4], "pi": [0.50], "sold_indicator": [1],
        })
        got = brab.participant_log_likelihood(
            decisions, alpha=0.30, sigma_lookup=lookup, treatment="random",
        )
        assert got == pytest.approx(log(brab.EPS), abs=1e-12)
        assert np.isfinite(got)


# =====
# Likelihood-ratio CI from LL profile
# =====
class TestLRCIFromProfile:
    """lr_ci_from_profile: LR 95% set = {α : 2·(LL_max − LL(α)) ≤ 3.84146}."""

    def test_parabolic_profile_symmetric_ci(self):
        """Synthetic parabola with 2·ΔLL=3.0 at α=0.4 and α=0.6 → CI spans [0.40, 0.60]."""
        grid = np.round(np.arange(0.0, 1.01, 0.01), 2)
        ll_max = -10.0
        # Build parabola: 2·(ll_max − ll(α)) = c·(α − 0.5)^2
        # We need c·0.01 = 3.0 at α=0.4/0.6 (distance 0.1) → c = 300
        c = 300.0
        ll_profile = ll_max - 0.5 * c * (grid - 0.5) ** 2
        ci_low, ci_high = brab.lr_ci_from_profile(grid, ll_profile, ll_max)
        # Boundary is exactly at α=0.40/0.60. 2·ΔLL at α=0.39 ≈ 363·0.0121 = 3.63 > 3.84? No:
        # c=300 → at α=0.40 2·ΔLL=300·0.01=3.0 (≤3.84 -> in). At α=0.39 2·ΔLL=300·0.0121=3.63 (in).
        # So with this c the CI is slightly wider than [0.40, 0.60].
        # Tighten: find analytically the grid points where 2·ΔLL ≤ 3.84146.
        crit = brab.LR_CHI2_95
        half_width = np.sqrt(crit / c)  # α such that c·(α-0.5)^2 = crit
        expected_low = 0.5 - half_width
        expected_high = 0.5 + half_width
        # Snap to grid
        expected_low_snap = grid[grid >= expected_low - 1e-12].min()
        expected_high_snap = grid[grid <= expected_high + 1e-12].max()
        assert ci_low == pytest.approx(expected_low_snap, abs=1e-9)
        assert ci_high == pytest.approx(expected_high_snap, abs=1e-9)

    def test_flat_profile_spans_full_grid(self):
        """Flat LL profile → every α is in the 95% set → CI = [0.0, 1.0]."""
        grid = np.round(np.arange(0.0, 1.01, 0.01), 2)
        ll_profile = np.full_like(grid, -5.0)
        ci_low, ci_high = brab.lr_ci_from_profile(grid, ll_profile, ll_max=-5.0)
        assert ci_low == pytest.approx(0.0, abs=1e-12)
        assert ci_high == pytest.approx(1.0, abs=1e-12)

    def test_ci_always_contains_alpha_mle(self):
        """The MLE itself satisfies 2·ΔLL=0 ≤ 3.84, so always in the CI."""
        grid = np.round(np.arange(0.0, 1.01, 0.01), 2)
        ll_profile = np.linspace(-20.0, -10.0, grid.size)
        ll_max = ll_profile.max()
        alpha_mle = grid[np.argmax(ll_profile)]
        ci_low, ci_high = brab.lr_ci_from_profile(grid, ll_profile, ll_max)
        assert ci_low <= alpha_mle <= ci_high


# =====
# π reconstruction from signal sequence
# =====
class TestPiReconstruction:
    """build_pi_lookup reproduces Bayesian posteriors after each signal."""

    @pytest.fixture
    def three_period_fixture(self):
        """One group, one round, signals [1, 1, 0] replicated across 4 players."""
        players = ["A", "B", "C", "D"]
        rows = []
        signals = [1, 1, 0]  # good, good, bad
        for period, sig in enumerate(signals, start=1):
            for player in players:
                rows.append({
                    "session_id": "sess1",
                    "segment": "chat_noavg",
                    "round": 1,
                    "period": period,
                    "group_id": 1,
                    "player": player,
                    "treatment": "tr1",
                    "signal": sig,
                    "state": 1,
                    "price": 8,
                    "sold": 0,
                    "already_sold": 0,
                    "prior_group_sales": 0,
                })
        return pd.DataFrame(rows)

    def test_pi_after_good_good_bad(self, three_period_fixture):
        """π sequence matches _update_good / _update_bad of equilibrium_model."""
        from analysis.analysis.equilibrium_model import (
            _update_good, _update_bad, PI_0,
        )
        expected_pi1 = _update_good(PI_0)
        expected_pi2 = _update_good(expected_pi1)
        expected_pi3 = _update_bad(expected_pi2)
        lookup = brab.build_pi_lookup(three_period_fixture)
        key_base = ("sess1", "chat_noavg", 1, 1)
        assert lookup[(*key_base, 1)] == pytest.approx(expected_pi1, abs=1e-12)
        assert lookup[(*key_base, 2)] == pytest.approx(expected_pi2, abs=1e-12)
        assert lookup[(*key_base, 3)] == pytest.approx(expected_pi3, abs=1e-12)


# =====
# σ strict lookup
# =====
class TestSigmaAtStrictLookup:
    """sigma_at: on-grid π returns σ; off-grid π raises ValueError."""

    @pytest.fixture
    def lookup(self):
        """(treatment='random', alpha=0.50, n=4) with three exact π grid points."""
        return {
            ("random", 0.50, 4): (
                np.array([0.325, 0.5, 0.675]),
                np.array([0.10, 0.40, 0.90]),
            ),
        }

    def test_sigma_at_exact_grid_point(self, lookup):
        """π=0.5 exactly present → returns σ=0.40."""
        got = brab.sigma_at(lookup, treatment="random", alpha=0.50, n=4, pi=0.5)
        assert got == pytest.approx(0.40, abs=1e-12)

    def test_sigma_at_off_grid_raises(self, lookup):
        """π=0.55 (not on grid) raises ValueError with actionable message."""
        with pytest.raises(ValueError, match="off-grid"):
            brab.sigma_at(
                lookup, treatment="random", alpha=0.50, n=4, pi=0.55,
            )

    def test_sigma_at_missing_key_raises(self, lookup):
        """Missing (treatment, alpha, n) key raises (fail loudly)."""
        with pytest.raises((KeyError, ValueError)):
            brab.sigma_at(
                lookup, treatment="average", alpha=0.50, n=4, pi=0.5,
            )


# =====
# Edge flag pass-through
# =====
class TestEdgeFlagPassesThroughToOutput:
    """assemble_output_row preserves alpha_task_edge_flag and NaN alpha_task."""

    def test_edge_allocate_zero(self):
        """allocate=0 → alpha_task NaN, edge flag True in output row."""
        grid = np.round(np.arange(0.0, 1.01, 0.01), 2)
        ll_profile = np.linspace(-12.0, -10.0, grid.size)
        decisions = pd.DataFrame({
            "n": [4, 4], "pi": [0.5, 0.675], "sold_indicator": [0, 0],
        })
        row = brab.assemble_output_row(
            session_id="sess1", player="A", treatment="tr1",
            decisions=decisions,
            alpha_mle=1.0, ll_profile=ll_profile, ll_max=ll_profile.max(),
            allocate=0.0,
        )
        assert np.isnan(row["alpha_task"])
        assert row["alpha_task_edge_flag"] is True or row["alpha_task_edge_flag"] == True  # noqa: E712
        assert row["session_id"] == "sess1"
        assert row["player"] == "A"
        assert row["treatment"] == "tr1"

    def test_edge_allocate_twenty(self):
        """allocate=20 → alpha_task NaN, edge flag True."""
        grid = np.round(np.arange(0.0, 1.01, 0.01), 2)
        ll_profile = np.linspace(-12.0, -10.0, grid.size)
        decisions = pd.DataFrame({
            "n": [4], "pi": [0.5], "sold_indicator": [0],
        })
        row = brab.assemble_output_row(
            session_id="sess1", player="B", treatment="tr2",
            decisions=decisions,
            alpha_mle=0.5, ll_profile=ll_profile, ll_max=ll_profile.max(),
            allocate=20.0,
        )
        assert np.isnan(row["alpha_task"])
        assert bool(row["alpha_task_edge_flag"]) is True

    def test_interior_allocate_sets_alpha_task(self):
        """allocate=10 → alpha_task = ln(2.5)/ln(3.5), edge flag False."""
        grid = np.round(np.arange(0.0, 1.01, 0.01), 2)
        ll_profile = np.linspace(-12.0, -10.0, grid.size)
        decisions = pd.DataFrame({
            "n": [4], "pi": [0.5], "sold_indicator": [0],
        })
        row = brab.assemble_output_row(
            session_id="sess1", player="C", treatment="tr1",
            decisions=decisions,
            alpha_mle=0.7, ll_profile=ll_profile, ll_max=ll_profile.max(),
            allocate=10.0,
        )
        assert row["alpha_task"] == pytest.approx(ALPHA_TASK_A10, abs=1e-10)
        assert bool(row["alpha_task_edge_flag"]) is False


# =====
# n_decisions count from decision table
# =====
class TestNDecisionsCount:
    """build_decision_table keeps only already_sold==0 rows for a participant."""

    @pytest.fixture
    def three_round_fixture(self):
        """One player (A) in 3 rounds × 4 periods. A sells in period 2 of round 2.
        Expected at-risk (already_sold==0) rows for A = 4 + 2 + 4 = 10.
        Populate B/C/D as always-holding so prior_group_sales stays 0 for A."""
        players = ["A", "B", "C", "D"]
        rows = []
        for rnd in (1, 2, 3):
            for period in (1, 2, 3, 4):
                for player in players:
                    a_already = 1 if (rnd == 2 and period > 2) else 0
                    a_sells_now = 1 if (rnd == 2 and period == 2 and player == "A") else 0
                    already = a_already if player == "A" else 0
                    sold = a_sells_now if player == "A" else 0
                    prior_group_sales = 1 if (
                        rnd == 2 and period > 2 and player != "A"
                    ) else 0
                    rows.append({
                        "session_id": "sess1",
                        "segment": "chat_noavg",
                        "round": rnd,
                        "period": period,
                        "group_id": 1,
                        "player": player,
                        "treatment": "tr1",
                        "signal": 1,
                        "state": 1,
                        "price": 8,
                        "sold": sold,
                        "already_sold": already,
                        "prior_group_sales": prior_group_sales,
                    })
        return pd.DataFrame(rows)

    def test_n_decisions_for_player_a(self, three_round_fixture):
        """A has 4 (rnd1) + 2 (rnd2 p1-2) + 4 (rnd3) = 10 at-risk decisions."""
        pi_lookup = brab.build_pi_lookup(three_round_fixture)
        decisions = brab.build_decision_table(three_round_fixture, pi_lookup)
        a_decisions = decisions[decisions["player"] == "A"]
        assert len(a_decisions) == 10

    def test_n_decisions_has_n_column_equals_investors_minus_prior(
        self, three_round_fixture
    ):
        """n = N_INVESTORS - prior_group_sales (project constant)."""
        pi_lookup = brab.build_pi_lookup(three_round_fixture)
        decisions = brab.build_decision_table(three_round_fixture, pi_lookup)
        # All A decisions have prior_group_sales == 0 → n == 4
        a_decisions = decisions[decisions["player"] == "A"]
        assert (a_decisions["n"] == brab.N_INVESTORS).all()
