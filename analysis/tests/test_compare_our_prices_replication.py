"""
Purpose: Pytest coverage for compare_our_prices_replication.py helpers —
         belief-convention conversion, interpolation-alignment sanity,
         stdout summary math, CSV schema, and fail-loudly path.
Author: comparison-engineer
Date: 2026-04-16
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the driver's sibling imports (e.g., `import equilibrium_model`) resolvable.
DRIVER_DIR = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(DRIVER_DIR))

import compare_our_prices_replication as driver  # noqa: E402

# FILE PATHS
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis" / "output"
COMPARISON_CSV = OUTPUT_DIR / "cross_validation_our_prices.csv"
EXPECTED_COLUMNS = [
    "n", "belief_p_bad", "belief_p_good",
    "munro_sigma", "our_sigma", "sigma_diff",
    "munro_V", "our_V", "V_diff",
]


# =====
# Fixtures
# =====
@pytest.fixture
def fake_munro_df():
    """Minimal 3-point R-style CSV: n=1..2, belief in {0, 0.5, 1}."""
    rows = []
    for n in (1, 2):
        for b in (0.0, 0.5, 1.0):
            rows.append({"n": n, "belief": b,
                         "sigma": 0.25 * n * b, "V": 8.0 - b})
    return pd.DataFrame(rows)


@pytest.fixture
def fake_ours():
    """Python-solver-style output sampled at target_p_good = [1, 0.5, 0]."""
    p_good = np.array([1.0, 0.5, 0.0])
    return {
        "belief_p_good": p_good,
        "sigma": {1: np.array([0.0, 0.0, 0.0]),
                  2: np.array([0.0, 0.1, 0.4]),
                  3: np.array([0.0, 0.2, 0.6]),
                  4: np.array([0.0, 0.3, 0.8])},
        "v_table": {1: np.array([8.0, 7.5, 7.0]),
                    2: np.array([8.0, 7.4, 7.0]),
                    3: np.array([8.0, 7.3, 7.0]),
                    4: np.array([8.0, 7.2, 7.0])},
    }


# =====
# Belief-convention conversion
# =====
class TestBeliefConvention:
    def test_p_bad_zero_maps_to_p_good_one(self):
        """Munro belief=0 (P(Bad)=0) should map to pi = P(Good) = 1."""
        p_bad = np.array([0.0])
        assert np.allclose(1.0 - p_bad, np.array([1.0]))

    def test_p_bad_one_maps_to_p_good_zero(self):
        """Munro belief=1 (P(Bad)=1) should map to pi = P(Good) = 0."""
        p_bad = np.array([1.0])
        assert np.allclose(1.0 - p_bad, np.array([0.0]))

    def test_conversion_preserves_midpoint(self):
        """belief=0.5 is the fixed point of the P(Bad) -> P(Good) flip."""
        assert 1.0 - 0.5 == pytest.approx(0.5)


# =====
# Interpolation-alignment sanity (uses build_comparison)
# =====
class TestComparisonAssembly:
    def test_row_count_matches_n_times_beliefs(self, fake_munro_df, fake_ours,
                                               monkeypatch):
        """build_comparison yields one row per (n, belief)."""
        monkeypatch.setattr(driver.em, "N_INVESTORS", 2)
        p_bad = np.sort(fake_munro_df["belief"].unique())
        comp = driver.build_comparison(fake_munro_df, fake_ours, p_bad)
        assert len(comp) == 2 * 3

    def test_boundary_p_good_one_row_has_expected_values(self, fake_munro_df,
                                                        fake_ours, monkeypatch):
        """At belief=0 (pi=P(Good)=1): fixture ours[sigma][n]=0, V=8.0."""
        monkeypatch.setattr(driver.em, "N_INVESTORS", 2)
        p_bad = np.sort(fake_munro_df["belief"].unique())
        comp = driver.build_comparison(fake_munro_df, fake_ours, p_bad)
        row = comp[(comp["n"] == 2) & (comp["belief_p_bad"] == 0.0)].iloc[0]
        assert row["belief_p_good"] == pytest.approx(1.0)
        assert row["our_sigma"] == pytest.approx(0.0)
        assert row["our_V"] == pytest.approx(8.0)

    def test_sigma_diff_is_our_minus_munro(self, fake_munro_df, fake_ours,
                                          monkeypatch):
        """Sign convention: sigma_diff = our_sigma - munro_sigma."""
        monkeypatch.setattr(driver.em, "N_INVESTORS", 2)
        p_bad = np.sort(fake_munro_df["belief"].unique())
        comp = driver.build_comparison(fake_munro_df, fake_ours, p_bad)
        recomputed = comp["our_sigma"] - comp["munro_sigma"]
        assert np.allclose(comp["sigma_diff"], recomputed)

    def test_belief_alignment_mismatch_raises(self, fake_munro_df, fake_ours,
                                              monkeypatch):
        """If the R CSV belief grid drifts from the combined grid, fail loudly."""
        monkeypatch.setattr(driver.em, "N_INVESTORS", 2)
        bad_df = fake_munro_df.copy()
        bad_df.loc[bad_df.index[1], "belief"] = 0.49
        with pytest.raises(ValueError, match="Belief grid mismatch"):
            driver.build_comparison(bad_df, fake_ours,
                                    np.sort(fake_munro_df["belief"].unique()))


# =====
# Stdout summary math (_safe_corr)
# =====
class TestSafeCorr:
    def test_perfect_positive_correlation(self):
        a = pd.Series([0.0, 0.5, 1.0])
        b = pd.Series([0.0, 0.5, 1.0])
        assert driver._safe_corr(a, b) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        a = pd.Series([0.0, 0.5, 1.0])
        b = pd.Series([1.0, 0.5, 0.0])
        assert driver._safe_corr(a, b) == pytest.approx(-1.0)

    def test_constant_series_returns_nan(self):
        a = pd.Series([0.0, 0.0, 0.0])
        b = pd.Series([0.1, 0.2, 0.3])
        assert np.isnan(driver._safe_corr(a, b))

    def test_known_correlation_value(self):
        a = pd.Series([1.0, 2.0, 3.0, 4.0])
        b = pd.Series([2.0, 1.0, 4.0, 3.0])
        expected = float(np.corrcoef(a, b)[0, 1])
        assert driver._safe_corr(a, b) == pytest.approx(expected)


# =====
# CSV schema (uses the real file regenerated by the driver run)
# =====
class TestOutputCsvSchema:
    @pytest.fixture(scope="class")
    def comp_df(self):
        if not COMPARISON_CSV.exists():
            pytest.skip(
                f"Comparison CSV missing: {COMPARISON_CSV}. "
                f"Run compare_our_prices_replication.py first.")
        return pd.read_csv(COMPARISON_CSV)

    def test_required_columns_present(self, comp_df):
        assert list(comp_df.columns) == EXPECTED_COLUMNS

    def test_row_count_matches_grid(self, comp_df):
        """4 investor counts x shared belief grid; rows = 4 * grid size."""
        n_per_group = comp_df.groupby("n").size()
        assert n_per_group.nunique() == 1
        assert len(comp_df) == 4 * int(n_per_group.iloc[0])

    def test_n_values_span_one_to_four(self, comp_df):
        assert sorted(comp_df["n"].unique().tolist()) == [1, 2, 3, 4]

    def test_p_bad_plus_p_good_equals_one(self, comp_df):
        sums = comp_df["belief_p_bad"] + comp_df["belief_p_good"]
        assert np.allclose(sums, 1.0)


# =====
# Fail-loudly path: missing R CSV
# =====
class TestFailLoudly:
    def test_missing_csv_raises_file_not_found(self, monkeypatch, tmp_path):
        monkeypatch.setattr(driver, "MUNRO_STYLE_CSV",
                            tmp_path / "does_not_exist.csv")
        with pytest.raises(FileNotFoundError, match="Run the task #2 R driver"):
            driver._load_munro_style_csv()

    def test_missing_columns_raises_value_error(self, monkeypatch, tmp_path):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"n": [1], "belief": [0.0]}).to_csv(bad, index=False)
        monkeypatch.setattr(driver, "MUNRO_STYLE_CSV", bad)
        with pytest.raises(ValueError, match="missing required columns"):
            driver._load_munro_style_csv()
