"""
Purpose: Group-level survey-trait aggregation for market runs analysis (#120)
Author: Claude Code
Date: 2026-04-25

Loads the player-level survey-traits dataset and computes raw group means of
8 personality / behavioral traits over the 4 group members for use as IVs in
the market-runs regression. NaN-tolerant: missing players are skipped; if all
4 are missing the function returns NaN per trait and warns to stderr.
"""

import sys
import pandas as pd
from pathlib import Path

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SURVEY_TRAITS_CSV = PROJECT_ROOT / "datastore" / "derived" / "survey_traits.csv"

# =====
# Schema
# =====
TRAIT_COLS = [
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "neuroticism",
    "openness",
    "impulsivity",
    "state_anxiety",
    "risk_tolerance",
]

REQUIRED_COLS = ["session_id", "player"] + TRAIT_COLS


# =====
# Loader
# =====
def load_survey_traits(csv_path: Path = SURVEY_TRAITS_CSV) -> pd.DataFrame:
    """Load survey_traits.csv; validate session_id, player, and 8 TRAIT_COLS."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Survey traits CSV missing: {csv_path}. "
            "Run analysis/derived/build_survey_traits_dataset.py first."
        )
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"survey_traits.csv missing required columns: {missing}. "
            f"Got: {df.columns.tolist()}"
        )
    return df


# =====
# Group aggregation
# =====
def compute_group_trait_means(
    traits_df: pd.DataFrame,
    session_id: str,
    player_labels: list[str],
) -> dict[str, float]:
    """Raw mean of TRAIT_COLS over the 4-player group.

    Skips players missing from traits_df for this session. If none of the
    players are present, warns to stderr and returns NaN per trait. Always
    returns a dict with exactly 8 ``group_mean_{trait}`` keys.
    """
    subset = traits_df[
        (traits_df["session_id"] == session_id)
        & (traits_df["player"].isin(player_labels))
    ]
    if subset.empty:
        msg = f"no traits for session={session_id} players={player_labels}"
        print(f"WARNING: {msg}; returning NaN means.", file=sys.stderr)
        return {f"group_mean_{t}": float("nan") for t in TRAIT_COLS}
    return {f"group_mean_{t}": float(subset[t].mean()) for t in TRAIT_COLS}
