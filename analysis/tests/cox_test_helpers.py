"""
Purpose: Shared constants and helpers for Cox survival data tests
Author: Claude Code
Date: 2026-02-23
"""

from pathlib import Path

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
DERIVED_DIR = DATASTORE / "derived"
EMOTIONS_DATASET = DERIVED_DIR / "emotions_traits_selling_dataset.csv"

SESSION_FILES = {
    "1_11-7-tr1": DATASTORE / "1_11-7-tr1" / "all_apps_wide_2025-11-07.csv",
    "2_11-10-tr2": DATASTORE / "2_11-10-tr2" / "all_apps_wide_2025-11-10.csv",
    "3_11-11-tr2": DATASTORE / "3_11-11-tr2" / "all_apps_wide_2025-11-11.csv",
    "4_11-12-tr1": DATASTORE / "4_11-12-tr1" / "all_apps_wide_2025-11-12.csv",
    "5_11-14-tr2": DATASTORE / "5_11-14-tr2" / "all_apps_wide_2025-11-14.csv",
    "6_11-18-tr1": DATASTORE / "6_11-18-tr1" / "all_apps_wide_2025-11-18.csv",
}
SEGMENT_MAP = {
    1: "chat_noavg", 2: "chat_noavg2",
    3: "chat_noavg3", 4: "chat_noavg4",
}

ALL_EMOTIONS = [
    "fear_mean", "anger_mean", "contempt_mean", "disgust_mean",
    "joy_mean", "sadness_mean", "surprise_mean",
    "engagement_mean", "valence_mean",
]
NONVALENCE_EMOTIONS = [
    "fear_mean", "anger_mean", "contempt_mean", "disgust_mean",
    "sadness_mean", "surprise_mean", "engagement_mean",
]
COX_CONTROLS = [
    "signal", "round", "segment", "treatment", "age", "gender_female",
]
CASCADE_DUMMIES = ["dummy_1_cum", "dummy_2_cum", "dummy_3_cum"]
INTERACTION_VARS = [
    "int_1_1", "int_2_1", "int_2_2",
    "int_3_1", "int_3_2", "int_3_3",
]


# =====
# Data preparation helpers
# =====
def _compute_prev_sales(df):
    """Compute prev-period sales count per group-round."""
    period_sales = (
        df.groupby(["group_round_id", "period"])["sold"]
        .sum().reset_index()
        .rename(columns={"sold": "n_sales"})
        .sort_values(["group_round_id", "period"])
    )
    period_sales["prev_n_sales"] = (
        period_sales.groupby("group_round_id")["n_sales"].shift(1)
    )
    period_sales["prev_n_sales"] = (
        period_sales["prev_n_sales"].fillna(0).astype(int)
    )
    return period_sales[["group_round_id", "period", "prev_n_sales"]]


def add_prev_period_dummies(df):
    """Replicate create_prev_period_dummies() from R helpers."""
    prev_sales = _compute_prev_sales(df)
    df = df.merge(prev_sales, on=["group_round_id", "period"], how="left")
    df["prev_n_sales"] = df["prev_n_sales"].fillna(0).astype(int)
    df["dummy_1_prev"] = (df["prev_n_sales"] == 1).astype(int)
    df["dummy_2_prev"] = (df["prev_n_sales"] == 2).astype(int)
    df["dummy_3_prev"] = (df["prev_n_sales"] == 3).astype(int)
    df = df.drop(columns=["prev_n_sales"])
    return df


def add_interaction_terms(df):
    """Replicate create_interaction_terms() from R helpers."""
    df["int_1_1"] = df["dummy_1_cum"] * df["dummy_1_prev"]
    df["int_2_1"] = df["dummy_2_cum"] * df["dummy_1_prev"]
    df["int_2_2"] = df["dummy_2_cum"] * df["dummy_2_prev"]
    df["int_3_1"] = df["dummy_3_cum"] * df["dummy_1_prev"]
    df["int_3_2"] = df["dummy_3_cum"] * df["dummy_2_prev"]
    df["int_3_3"] = df["dummy_3_cum"] * df["dummy_3_prev"]
    return df


def add_id_columns(df):
    """Add composite ID columns to base data."""
    df["player_id"] = df["session_id"] + "_" + df["player"].astype(str)
    df["group_round_id"] = (
        df["session_id"] + "_" + df["segment"].astype(str) + "_"
        + df["group_id"].astype(str) + "_" + df["round"].astype(str)
    )
    df["player_group_round_id"] = (
        df["player_id"] + "_" + df["segment"].astype(str) + "_"
        + df["group_id"].astype(str) + "_" + df["round"].astype(str)
    )
    return df


def add_dummies(df):
    """Add cascade, gender, and period_start columns."""
    df["dummy_1_cum"] = (df["prior_group_sales"] == 1).astype(int)
    df["dummy_2_cum"] = (df["prior_group_sales"] == 2).astype(int)
    df["dummy_3_cum"] = (df["prior_group_sales"] == 3).astype(int)
    df["gender_female"] = (df["gender"] == "Female").astype(int)
    df["period_start"] = df["period"] - 1
    return df
