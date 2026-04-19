"""
Purpose: Tests for robustness checks — verify grid sensitivity and seed
         sensitivity produce stable results.
Author: Claude
Date: 2026-04-08
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Archived: load the archived robustness module as a sibling package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "robustness"))

from analysis.analysis.equilibrium_model import solve_equilibrium  # noqa: E402
from robustness_checks import (  # noqa: E402
    _net_bad_to_belief,
    _simulate_with_seed,
    MM_TARGETS,
    N_SIMULATIONS,
)


# =====
# Grid sensitivity tests
# =====
class TestGridSensitivity:
    """Sigma must be stable across grid resolutions."""

    @pytest.fixture(scope="class")
    def sigma_by_tmax(self):
        """Solve at t_max=20 and t_max=80, return sigma at MM targets."""
        results = {}
        for t_max in [20, 80]:
            for alpha in [0.0, 0.5]:
                result = solve_equilibrium(
                    alpha=alpha, treatment="random", t_max=t_max
                )
                results[(alpha, t_max)] = result
        return results

    def test_sigma_converged_by_tmax20(self, sigma_by_tmax):
        """Sigma at t_max=20 and t_max=80 must agree within 0.001."""
        for alpha, n, d in MM_TARGETS:
            pi = _net_bad_to_belief(d)
            r20 = sigma_by_tmax[(alpha, 20)]
            r80 = sigma_by_tmax[(alpha, 80)]
            s20 = float(np.interp(pi, r20["belief_grid"], r20["sigma"][n]))
            s80 = float(np.interp(pi, r80["belief_grid"], r80["sigma"][n]))
            assert s20 == pytest.approx(s80, abs=0.001), (
                f"a={alpha}, n={n}, d={d}: sigma differs between "
                f"t_max=20 ({s20:.4f}) and t_max=80 ({s80:.4f})"
            )


# =====
# Seed sensitivity tests
# =====
class TestSeedSensitivity:
    """Simulation results must be stable across random seeds."""

    @pytest.fixture(scope="class")
    def pbads_across_seeds(self):
        """Run simulation with 10 seeds, return avg P(Bad) per position."""
        result = solve_equilibrium(alpha=0.5, treatment="random", t_max=40)
        grid = result["belief_grid"]
        sigma_table = result["sigma"]
        all_pbads = {k: [] for k in range(1, 4)}
        for seed in range(1, 11):
            pbads = _simulate_with_seed(seed, grid, sigma_table)
            for k, pb in pbads.items():
                all_pbads[k].append(pb)
        return all_pbads

    def test_first_sale_sd_below_threshold(self, pbads_across_seeds):
        """1st sale P(Bad) SD across seeds must be < 0.005."""
        vals = np.array(pbads_across_seeds[1])
        sd = np.std(vals, ddof=1)
        assert sd < 0.005, f"1st sale SD = {sd:.4f}, expected < 0.005"

    def test_all_positions_finite(self, pbads_across_seeds):
        """All seed runs must produce finite P(Bad) values."""
        for k in range(1, 4):
            vals = np.array(pbads_across_seeds[k])
            assert np.all(np.isfinite(vals)), f"Position {k} has non-finite values"

    def test_range_below_threshold(self, pbads_across_seeds):
        """Full range across seeds must be < 0.02 for all positions."""
        for k in range(1, 4):
            vals = np.array(pbads_across_seeds[k])
            rng = np.max(vals) - np.min(vals)
            assert rng < 0.02, (
                f"Position {k} range = {rng:.4f}, expected < 0.02"
            )


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
