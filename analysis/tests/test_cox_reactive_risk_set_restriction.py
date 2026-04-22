"""
Purpose: Tests for reactive risk set restriction parity in Cox pre-sell models (issue #118).
Validates that restricting to `group_sold_prev_period == 1` matches the full-sample
`reactive_sale` event count after the 500ms pre-sell merge pipeline.
Author: Codex
Date: 2026-04-22
"""

from pathlib import Path
import sys

import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from reactive_flag_helpers import add_reactive_flag
from presell_merge_helpers import merge_presell_window, drop_missing_window_rows

# FILE PATHS
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CSV = REPO_ROOT / "datastore" / "derived" / "emotions_traits_selling_dataset.csv"
PRESELL_CSV = REPO_ROOT / "datastore" / "derived" / "presell_emotions_traits_dataset.csv"

# CONSTANTS
ALL_EMOTIONS = [
    "fear_mean",
    "anger_mean",
    "contempt_mean",
    "disgust_mean",
    "joy_mean",
    "sadness_mean",
    "surprise_mean",
    "engagement_mean",
    "valence_mean",
]
ALL_TRAITS = [
    "state_anxiety",
    "impulsivity",
    "risk_tolerance",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    "openness",
]


# =====
# Main
# =====
def main():
    """Run all tests with verbose output."""
    pytest.main([__file__, "-v"])


# =====
# Helpers
# =====
def restrict_to_reactive_risk_set(df):
    """Mirror the R helper used by reactive Cox models."""
    return df[df["group_sold_prev_period"] == 1].copy()


def _load_real_inputs():
    """Load base and presell inputs or skip if either is unavailable."""
    if not BASE_CSV.exists():
        pytest.skip(f"Base dataset missing: {BASE_CSV}")
    if not PRESELL_CSV.exists():
        pytest.skip(f"Presell dataset missing: {PRESELL_CSV}")
    return pd.read_csv(BASE_CSV), pd.read_csv(PRESELL_CSV)


def _prepare_base_data(base_df):
    """Match the Cox base-data preparation before the pre-sell merge."""
    base = base_df[base_df["already_sold"] == 0].copy()
    base["player_id"] = base["session_id"] + "_" + base["player"].astype(str)
    base["global_group_id"] = (
        base["session_id"] + "_" + base["segment"].astype(str) + "_" + base["group_id"].astype(str)
    )
    base["group_round_id"] = base["global_group_id"] + "_" + base["round"].astype(str)
    return base


def _drop_missing_controls(df):
    """Keep the estimation sample where all emotion and trait controls are observed."""
    controls = np.array(ALL_EMOTIONS + ALL_TRAITS, dtype=object)
    return df.dropna(subset=controls.tolist()).copy()


# =====
# Fixture
# =====
@pytest.fixture(scope="module")
def reactive_sample_500ms():
    """Build the unrestricted 500ms reactive sample used for parity checks."""
    base, presell = _load_real_inputs()
    prepared = _prepare_base_data(base)
    merged = merge_presell_window(prepared, presell, 500)
    filtered, _ = drop_missing_window_rows(merged, 500)
    complete = _drop_missing_controls(filtered)
    return add_reactive_flag(complete)


# =====
# Tests
# =====
def test_restriction_keeps_only_gspp_eq_1(reactive_sample_500ms):
    """Restricted sample must contain only rows exposed to a prior-period group sale."""
    restricted = restrict_to_reactive_risk_set(reactive_sample_500ms)
    assert (restricted["group_sold_prev_period"] == 1).all()


def test_event_parity(reactive_sample_500ms):
    """Restricted sold events must match full-sample reactive_sale events."""
    restricted = restrict_to_reactive_risk_set(reactive_sample_500ms)
    restricted_events = int(restricted["sold"].sum())
    full_reactive_events = int(reactive_sample_500ms["reactive_sale"].sum())
    assert restricted_events == full_reactive_events


def test_restricted_is_strict_subset(reactive_sample_500ms):
    """Restriction should remove rows from the unrestricted sample."""
    restricted = restrict_to_reactive_risk_set(reactive_sample_500ms)
    assert len(restricted) < len(reactive_sample_500ms)


def test_restriction_idempotent(reactive_sample_500ms):
    """Applying the same risk-set restriction twice should not change results."""
    restricted_once = restrict_to_reactive_risk_set(reactive_sample_500ms)
    restricted_twice = restrict_to_reactive_risk_set(restricted_once)
    assert len(restricted_once) == len(restricted_twice)
    assert int(restricted_once["sold"].sum()) == int(restricted_twice["sold"].sum())


if __name__ == "__main__":
    main()
