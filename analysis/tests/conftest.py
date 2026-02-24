"""
Purpose: Shared pytest fixtures for Cox survival data tests
Author: Claude Code
Date: 2026-02-23
"""

import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import market_data as md
from cox_test_helpers import (
    EMOTIONS_DATASET, SESSION_FILES, ALL_EMOTIONS,
    add_id_columns, add_dummies, add_prev_period_dummies,
    add_interaction_terms,
)


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def raw_data():
    """Load raw emotions_traits_selling_dataset.csv."""
    if not EMOTIONS_DATASET.exists():
        pytest.skip(f"Dataset not found: {EMOTIONS_DATASET}")
    return pd.read_csv(EMOTIONS_DATASET)


@pytest.fixture(scope="module")
def base_data(raw_data):
    """Replicate R prepare_base_data(): filter already_sold == 0."""
    df = raw_data[raw_data["already_sold"] == 0].copy()
    df = add_id_columns(df)
    df = add_dummies(df)
    df = add_prev_period_dummies(df)
    df = add_interaction_terms(df)
    return df


@pytest.fixture(scope="module")
def emotion_filtered(base_data):
    """Emotion-complete cases — the sample passed to both panels."""
    return base_data.dropna(subset=ALL_EMOTIONS).copy()


@pytest.fixture(scope="module")
def first_seller_data(emotion_filtered):
    """First-seller subsample: player-group-rounds where someone sold
    with prior_group_sales == 0, keeping all at-risk periods."""
    first_ids = emotion_filtered.loc[
        (emotion_filtered["prior_group_sales"] == 0)
        & (emotion_filtered["sold"] == 1),
        "player_group_round_id"
    ].unique()
    return emotion_filtered[
        emotion_filtered["player_group_round_id"].isin(first_ids)
    ].copy()


@pytest.fixture(scope="module")
def parsed_experiments():
    """Load raw session data via market_data parser."""
    experiments = {}
    for session_id, csv_path in SESSION_FILES.items():
        if csv_path.exists():
            experiments[session_id] = md.parse_experiment(str(csv_path))
    if not experiments:
        pytest.skip("No raw session files found")
    return experiments
