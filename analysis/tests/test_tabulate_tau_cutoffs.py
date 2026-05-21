"""
Purpose: Unit tests for equilibrium tau-cutoff extraction.
Author: Claude
Date: 2026-05-20
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from tabulate_tau_cutoffs import extract_cutoffs, nearest_alpha


def _slice(pi_sigma):
    """Build a (pi, sigma) slice DataFrame from a list of pairs."""
    return pd.DataFrame(pi_sigma, columns=["pi", "sigma"])


def test_tau_bar_is_highest_belief_with_positive_sigma():
    sub = _slice([(0.10, 1.0), (0.20, 0.4), (0.30, 0.0), (0.40, 0.0)])
    tau_bar, _ = extract_cutoffs(sub)
    assert tau_bar == 0.20


def test_tau_under_is_highest_belief_with_full_run():
    sub = _slice([(0.10, 1.0), (0.20, 1.0), (0.30, 0.4), (0.40, 0.0)])
    _, tau_under = extract_cutoffs(sub)
    assert tau_under == 0.20


def test_tau_bar_at_least_tau_under():
    sub = _slice([(0.05, 1.0), (0.15, 0.5), (0.25, 0.0)])
    tau_bar, tau_under = extract_cutoffs(sub)
    assert tau_bar >= tau_under


def test_no_selling_returns_nan():
    sub = _slice([(0.10, 0.0), (0.20, 0.0)])
    tau_bar, tau_under = extract_cutoffs(sub)
    assert np.isnan(tau_bar) and np.isnan(tau_under)


def test_selling_but_never_certain_gives_nan_tau_under():
    sub = _slice([(0.10, 0.6), (0.20, 0.3), (0.30, 0.0)])
    tau_bar, tau_under = extract_cutoffs(sub)
    assert tau_bar == 0.20 and np.isnan(tau_under)


def test_tiny_sigma_below_tolerance_is_not_selling():
    sub = _slice([(0.10, 5e-7), (0.20, 0.0)])
    tau_bar, _ = extract_cutoffs(sub)
    assert np.isnan(tau_bar)


def test_nearest_alpha_snaps_to_grid():
    grid = pd.DataFrame({"alpha": [0.0, 0.1, 0.5, 0.9]})
    assert nearest_alpha(grid, 0.5) == 0.5
    assert nearest_alpha(grid, 0.48) == 0.5
    assert nearest_alpha(grid, 0.0) == 0.0
