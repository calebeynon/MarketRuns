"""
Purpose: Unit tests for build_first_seller_analysis_dataset.py
Author: Claude Code
Date: 2026-02-01
"""

import pytest
import pandas as pd
from pathlib import Path
from analysis.derived.build_first_seller_analysis_dataset import (
    load_first_seller_data,
    load_and_prepare_traits,
    merge_datasets,
    finalize_columns,
    validate_dataset,
)


# =====
# Test fixtures
# =====
@pytest.fixture
def sample_first_seller_df():
    """Create sample first seller round data."""
    return pd.DataFrame({
        "session_id": ["1_11-7-tr1", "1_11-7-tr1", "2_11-10-tr2"],
        "treatment": ["tr1", "tr1", "tr2"],
        "segment": [1, 1, 2],
        "group_id": [1, 1, 1],
        "round": [1, 2, 1],
        "player": ["A", "B", "A"],
        "public_signal": [None, 0.5, None],
        "state": [0, 1, 1],
        "is_first_seller": [1, 0, 1],
        "first_sale_period": [1.0, 2.0, 1.0],
    })


@pytest.fixture
def sample_traits_df():
    """Create sample survey traits data."""
    return pd.DataFrame({
        "session_id": ["1_11-7-tr1", "1_11-7-tr1", "2_11-10-tr2"],
        "player": ["A", "B", "A"],
        "extraversion": [5.5, 3.5, 4.0],
        "agreeableness": [4.0, 5.5, 6.5],
        "conscientiousness": [5.0, 6.5, 6.0],
        "neuroticism": [6.5, 1.5, 1.5],
        "openness": [7.0, 5.5, 4.5],
        "impulsivity": [4.125, 1.5, 2.625],
        "state_anxiety": [3.17, 1.83, 1.67],
        "age": [18.0, 28.0, 23.0],
        "gender": ["Female", "Female", "Male"],
    })


# =====
# Data loading tests
# =====
def test_load_first_seller_data_returns_dataframe():
    """Load function returns a DataFrame."""
    df = load_first_seller_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_first_seller_data_has_required_columns():
    """First seller data contains all required columns."""
    df = load_first_seller_data()
    required_cols = [
        "session_id", "treatment", "segment", "group_id", "round",
        "player", "public_signal", "state", "is_first_seller", "first_sale_period"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_load_and_prepare_traits_creates_gender_female():
    """Gender variable is converted to binary gender_female."""
    df = load_and_prepare_traits()
    assert "gender_female" in df.columns
    assert "gender" not in df.columns  # Original dropped


def test_load_and_prepare_traits_gender_binary():
    """Gender_female is 0 or 1 only."""
    df = load_and_prepare_traits()
    assert set(df["gender_female"].unique()).issubset({0, 1})


# =====
# Merge tests
# =====
def test_merge_on_session_and_player(sample_first_seller_df, sample_traits_df):
    """Merge happens on session_id and player."""
    # Prepare traits with gender_female
    traits = sample_traits_df.copy()
    traits["gender_female"] = (traits["gender"] == "Female").astype(int)
    traits = traits.drop(columns=["gender"])

    merged = merge_datasets(sample_first_seller_df, traits)

    assert len(merged) == 3
    assert "extraversion" in merged.columns


def test_merge_drops_unmatched_rows():
    """Rows without matching traits are dropped."""
    first_seller = pd.DataFrame({
        "session_id": ["1_11-7-tr1", "1_11-7-tr1"],
        "player": ["A", "Z"],  # Z has no traits
        "treatment": ["tr1", "tr1"],
        "segment": [1, 1],
        "group_id": [1, 1],
        "round": [1, 1],
        "public_signal": [None, None],
        "state": [0, 0],
        "is_first_seller": [1, 0],
        "first_sale_period": [1.0, None],
    })
    traits = pd.DataFrame({
        "session_id": ["1_11-7-tr1"],
        "player": ["A"],
        "extraversion": [5.5],
        "agreeableness": [4.0],
        "conscientiousness": [5.0],
        "neuroticism": [6.5],
        "openness": [7.0],
        "impulsivity": [4.125],
        "state_anxiety": [3.17],
        "age": [18.0],
        "gender_female": [1],
    })

    merged = merge_datasets(first_seller, traits)
    assert len(merged) == 1
    assert merged.iloc[0]["player"] == "A"


# =====
# Column finalization tests
# =====
def test_finalize_columns_correct_order(sample_first_seller_df, sample_traits_df):
    """Final columns are in correct order."""
    traits = sample_traits_df.copy()
    traits["gender_female"] = (traits["gender"] == "Female").astype(int)
    traits = traits.drop(columns=["gender"])

    merged = merge_datasets(sample_first_seller_df, traits)
    final = finalize_columns(merged)

    expected_columns = [
        "session_id", "treatment", "segment", "group_id", "round", "player",
        "public_signal", "state", "is_first_seller", "first_sale_period",
        "extraversion", "agreeableness", "conscientiousness", "neuroticism",
        "openness", "impulsivity", "state_anxiety", "age", "gender_female"
    ]
    assert list(final.columns) == expected_columns


def test_finalize_columns_count(sample_first_seller_df, sample_traits_df):
    """Final dataset has exactly 19 columns."""
    traits = sample_traits_df.copy()
    traits["gender_female"] = (traits["gender"] == "Female").astype(int)
    traits = traits.drop(columns=["gender"])

    merged = merge_datasets(sample_first_seller_df, traits)
    final = finalize_columns(merged)

    assert len(final.columns) == 19


# =====
# Validation tests
# =====
def test_validate_no_missing_traits(capsys):
    """Validation passes when no missing trait values."""
    df = pd.DataFrame({
        "session_id": ["1_11-7-tr1"],
        "extraversion": [5.5],
        "agreeableness": [4.0],
        "conscientiousness": [5.0],
        "neuroticism": [6.5],
        "openness": [7.0],
        "impulsivity": [4.125],
        "state_anxiety": [3.17],
        "age": [18.0],
        "gender_female": [1],
    })
    validate_dataset(df)
    captured = capsys.readouterr()
    assert "OK: No missing values" in captured.out


def test_validate_catches_missing_traits(capsys):
    """Validation warns when trait values are missing."""
    df = pd.DataFrame({
        "session_id": ["1_11-7-tr1"],
        "extraversion": [None],  # Missing
        "agreeableness": [4.0],
        "conscientiousness": [5.0],
        "neuroticism": [6.5],
        "openness": [7.0],
        "impulsivity": [4.125],
        "state_anxiety": [3.17],
        "age": [18.0],
        "gender_female": [1],
    })
    validate_dataset(df)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "extraversion" in captured.out


# =====
# Integration tests (using actual data files)
# =====
def test_output_file_exists_after_build():
    """Output file should exist in derived folder."""
    output_path = (
        Path(__file__).resolve().parent.parent.parent
        / "datastore" / "derived" / "first_seller_analysis_data.csv"
    )
    assert output_path.exists(), f"Output file not found: {output_path}"


def test_output_no_missing_trait_columns():
    """No missing values in trait columns of output file."""
    output_path = (
        Path(__file__).resolve().parent.parent.parent
        / "datastore" / "derived" / "first_seller_analysis_data.csv"
    )
    df = pd.read_csv(output_path)

    trait_cols = [
        "extraversion", "agreeableness", "conscientiousness",
        "neuroticism", "openness", "impulsivity", "state_anxiety",
        "age", "gender_female"
    ]
    for col in trait_cols:
        assert df[col].isna().sum() == 0, f"Missing values in {col}"


def test_output_gender_female_binary():
    """gender_female in output is 0 or 1 only."""
    output_path = (
        Path(__file__).resolve().parent.parent.parent
        / "datastore" / "derived" / "first_seller_analysis_data.csv"
    )
    df = pd.read_csv(output_path)
    assert set(df["gender_female"].unique()) == {0, 1}


def test_output_is_first_seller_binary():
    """is_first_seller is 0 or 1."""
    output_path = (
        Path(__file__).resolve().parent.parent.parent
        / "datastore" / "derived" / "first_seller_analysis_data.csv"
    )
    df = pd.read_csv(output_path)
    assert set(df["is_first_seller"].unique()) == {0, 1}


# =====
# Raw data verification tests
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"


def test_gender_matches_raw_survey_session1():
    """Verify gender_female matches raw survey q26 for session 1."""
    # Load raw survey data
    raw_survey = pd.read_csv(DATASTORE / "1_11-7-tr1" / "survey_2025-11-07.csv")
    output = pd.read_csv(DATASTORE / "derived" / "first_seller_analysis_data.csv")

    # Filter to session 1 and get unique player-level data
    session1_output = output[output["session_id"] == "1_11-7-tr1"]
    session1_output = session1_output.drop_duplicates(subset=["player"])

    for _, row in raw_survey.iterrows():
        player = row["participant.label"]
        raw_gender = row["player.q26"]

        player_output = session1_output[session1_output["player"] == player]
        if len(player_output) == 0:
            continue  # Player may have been dropped due to missing data

        expected_female = 1 if raw_gender == "Female" else 0
        actual_female = player_output.iloc[0]["gender_female"]
        assert actual_female == expected_female, (
            f"Player {player}: expected gender_female={expected_female}, "
            f"got {actual_female} (raw gender: {raw_gender})"
        )


def test_age_matches_raw_survey_session1():
    """Verify age matches raw survey q25 for session 1."""
    raw_survey = pd.read_csv(DATASTORE / "1_11-7-tr1" / "survey_2025-11-07.csv")
    output = pd.read_csv(DATASTORE / "derived" / "first_seller_analysis_data.csv")

    session1_output = output[output["session_id"] == "1_11-7-tr1"]
    session1_output = session1_output.drop_duplicates(subset=["player"])

    for _, row in raw_survey.iterrows():
        player = row["participant.label"]
        raw_age = row["player.q25"]

        player_output = session1_output[session1_output["player"] == player]
        if len(player_output) == 0:
            continue

        actual_age = player_output.iloc[0]["age"]
        assert actual_age == raw_age, (
            f"Player {player}: expected age={raw_age}, got {actual_age}"
        )


def test_specific_participant_traits_session1_A():
    """Verify traits for participant A in session 1 match expected values.

    Raw survey responses for Player A (session 1_11-7-tr1):
    - q7 (extraversion+): Agree Moderately (6)
    - q12 (extraversion-): Disagree a little (3) -> reversed: 8-3=5
    - Extraversion = (6+5)/2 = 5.5
    """
    output = pd.read_csv(DATASTORE / "derived" / "first_seller_analysis_data.csv")
    player_a = output[
        (output["session_id"] == "1_11-7-tr1") & (output["player"] == "A")
    ].iloc[0]

    # Extraversion computed from q7 and q12
    assert player_a["extraversion"] == pytest.approx(5.5, abs=0.01)
    assert player_a["age"] == 18.0
    assert player_a["gender_female"] == 1


def test_specific_participant_traits_session1_E():
    """Verify traits for participant E (male) in session 1."""
    output = pd.read_csv(DATASTORE / "derived" / "first_seller_analysis_data.csv")
    player_e = output[
        (output["session_id"] == "1_11-7-tr1") & (output["player"] == "E")
    ].iloc[0]

    # E is male based on raw survey
    assert player_e["gender_female"] == 0
    assert player_e["age"] == 20.0


def test_correct_number_of_participants_dropped():
    """Verify approximately 1 participant dropped due to missing survey data.

    The first_seller_round_data has 6 sessions x 16 players = 96 unique participants.
    The survey_traits has 95 participants (1 dropped for missing survey responses).
    """
    first_seller = pd.read_csv(DATASTORE / "derived" / "first_seller_round_data.csv")
    output = pd.read_csv(DATASTORE / "derived" / "first_seller_analysis_data.csv")

    unique_first_seller = first_seller.groupby(
        ["session_id", "player"]
    ).ngroups
    unique_output = output.groupby(["session_id", "player"]).ngroups

    # Expect 1 participant dropped
    participants_dropped = unique_first_seller - unique_output
    assert participants_dropped == 1, (
        f"Expected 1 participant dropped, got {participants_dropped}"
    )


def test_all_sessions_represented():
    """Verify all 6 sessions are present in output."""
    output = pd.read_csv(DATASTORE / "derived" / "first_seller_analysis_data.csv")
    expected_sessions = {
        "1_11-7-tr1", "2_11-10-tr2", "3_11-11-tr2",
        "4_11-12-tr1", "5_11-14-tr2", "6_11-18-tr1"
    }
    actual_sessions = set(output["session_id"].unique())
    assert actual_sessions == expected_sessions


def test_treatment_assignment_correct():
    """Verify treatment assignment matches session naming convention."""
    output = pd.read_csv(DATASTORE / "derived" / "first_seller_analysis_data.csv")

    # tr1 sessions
    tr1_sessions = ["1_11-7-tr1", "4_11-12-tr1", "6_11-18-tr1"]
    for session in tr1_sessions:
        session_data = output[output["session_id"] == session]
        assert all(session_data["treatment"] == "tr1"), (
            f"Session {session} should be tr1"
        )

    # tr2 sessions
    tr2_sessions = ["2_11-10-tr2", "3_11-11-tr2", "5_11-14-tr2"]
    for session in tr2_sessions:
        session_data = output[output["session_id"] == session]
        assert all(session_data["treatment"] == "tr2"), (
            f"Session {session} should be tr2"
        )


def test_row_count_matches_first_seller_minus_dropped():
    """Verify output row count equals first_seller rows minus dropped participants."""
    first_seller = pd.read_csv(DATASTORE / "derived" / "first_seller_round_data.csv")
    output = pd.read_csv(DATASTORE / "derived" / "first_seller_analysis_data.csv")
    survey_traits = pd.read_csv(DATASTORE / "derived" / "survey_traits.csv")

    # Get participants in traits (these are the ones we keep)
    traits_participants = set(
        survey_traits.apply(
            lambda r: (r["session_id"], r["player"]), axis=1
        )
    )

    # Count first_seller rows for participants in traits
    first_seller["key"] = first_seller.apply(
        lambda r: (r["session_id"], r["player"]), axis=1
    )
    expected_rows = first_seller[
        first_seller["key"].isin(traits_participants)
    ].shape[0]

    assert len(output) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(output)}"
    )


def test_trait_ranges_valid():
    """Verify trait values are within expected ranges."""
    output = pd.read_csv(DATASTORE / "derived" / "first_seller_analysis_data.csv")

    # BFI-10 traits: 1-7 scale
    bfi_traits = [
        "extraversion", "agreeableness", "conscientiousness",
        "neuroticism", "openness"
    ]
    for trait in bfi_traits:
        assert output[trait].min() >= 1.0, f"{trait} below minimum 1.0"
        assert output[trait].max() <= 7.0, f"{trait} above maximum 7.0"

    # Impulsivity: 1-7 scale
    assert output["impulsivity"].min() >= 1.0
    assert output["impulsivity"].max() <= 7.0

    # State anxiety: 1-4 scale
    assert output["state_anxiety"].min() >= 1.0
    assert output["state_anxiety"].max() <= 4.0


def test_first_seller_data_preserved():
    """Verify first seller flags match source data after merge."""
    first_seller = pd.read_csv(DATASTORE / "derived" / "first_seller_round_data.csv")
    output = pd.read_csv(DATASTORE / "derived" / "first_seller_analysis_data.csv")

    # Check a specific known case from session 1
    # Round 2, group 1: Player A sold in period 1 (is_first_seller=1)
    session1_round2 = output[
        (output["session_id"] == "1_11-7-tr1") &
        (output["segment"] == 1) &
        (output["group_id"] == 1) &
        (output["round"] == 2)
    ]

    first_seller_round2 = first_seller[
        (first_seller["session_id"] == "1_11-7-tr1") &
        (first_seller["segment"] == 1) &
        (first_seller["group_id"] == 1) &
        (first_seller["round"] == 2)
    ]

    # Merge should preserve is_first_seller values
    for _, fs_row in first_seller_round2.iterrows():
        player = fs_row["player"]
        out_row = session1_round2[session1_round2["player"] == player]
        if len(out_row) == 0:
            continue  # Dropped participant

        assert out_row.iloc[0]["is_first_seller"] == fs_row["is_first_seller"], (
            f"is_first_seller mismatch for player {player}"
        )


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
