"""
Purpose: Validate seller count metrics against raw oTree session exports
Author: Claude Code
Date: 2026-02-22

Cross-validates the derived individual_round_panel.csv seller counts
against the raw oTree segment CSVs. This catches bugs in the derivation
pipeline that same-source tests would miss.
"""

import pandas as pd
import pytest
from pathlib import Path

# =====
# File paths
# =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASTORE = PROJECT_ROOT / "datastore"
PANEL_PATH = DATASTORE / "derived" / "individual_round_panel.csv"

SEGMENTS = ["chat_noavg", "chat_noavg2", "chat_noavg3", "chat_noavg4"]

# Two sessions to validate: one per treatment
VALIDATION_SESSIONS = {
    "1_11-7-tr1": "tr1",
    "2_11-10-tr2": "tr2",
}


# =====
# Main function
# =====
def main():
    pytest.main([__file__, "-v"])


# =====
# Helpers
# =====
def load_raw_segment(session_id, segment_name):
    """Load a raw oTree segment CSV for a session."""
    session_dir = DATASTORE / session_id
    csv_files = sorted(session_dir.glob(f"{segment_name}_*.csv"))
    if not csv_files:
        pytest.skip(f"Raw CSV not found: {session_dir}/{segment_name}_*.csv")
    return pd.read_csv(csv_files[0])


def count_sellers_from_raw(raw_df):
    """Count sellers per group-round from raw oTree period-level data.

    A player sold in a round if player.sold == 1 in any period of that round.
    """
    player_sold = raw_df.groupby(
        ["group.id_in_subsession", "player.round_number_in_segment",
         "participant.label"]
    )["player.sold"].max().reset_index()

    group_round_sellers = player_sold.groupby(
        ["group.id_in_subsession", "player.round_number_in_segment"]
    )["player.sold"].sum().reset_index()

    group_round_sellers.columns = ["group_id", "round", "n_sellers"]
    return group_round_sellers


def get_state_from_raw(raw_df):
    """Extract the state for each group-round from raw data."""
    states = raw_df.groupby(
        ["group.id_in_subsession", "player.round_number_in_segment"]
    )["player.state"].first().reset_index()
    states.columns = ["group_id", "round", "state"]
    return states


def get_derived_group_round(panel, session_id, seg_idx, group_id, round_num):
    """Filter panel to a specific group-round."""
    mask = (
        (panel.session_id == session_id)
        & (panel.segment == seg_idx)
        & (panel.group_id == group_id)
        & (panel["round"] == round_num)
    )
    return panel[mask]


def get_derived_segment(panel, session_id, seg_idx):
    """Filter panel to a specific session-segment."""
    return panel[
        (panel.session_id == session_id) & (panel.segment == seg_idx)
    ]


def _raw_counts_with_state(raw_df):
    """Get seller counts merged with state per group-round."""
    raw_counts = count_sellers_from_raw(raw_df)
    raw_states = get_state_from_raw(raw_df)
    return raw_counts.merge(raw_states, on=["group_id", "round"])


def _assert_zero_counts_match(raw_merged, derived, session_id, seg_idx, state):
    """Assert zero-seller group-round count matches for a given state."""
    raw_zeros = (
        raw_merged[raw_merged.state == state]["n_sellers"] == 0
    ).sum()
    derived_state = derived[derived.state == state]
    derived_zeros = (
        derived_state.groupby(["group_id", "round"])["did_sell"].sum() == 0
    ).sum()
    assert raw_zeros == derived_zeros, (
        f"{session_id} seg {seg_idx} state {state}: "
        f"raw={raw_zeros} != derived={derived_zeros}"
    )


def _assert_four_players_raw(raw_df, session_id, seg_idx):
    """Assert raw oTree data has 4 unique players per group-round."""
    raw_players = raw_df.groupby(
        ["group.id_in_subsession", "player.round_number_in_segment"]
    )["participant.label"].nunique()
    assert (raw_players == 4).all(), (
        f"{session_id} seg {seg_idx}: raw group-rounds without 4 players"
    )


def _check_states(panel, session_id, seg_idx, raw_states):
    """Compare state values for one segment, return mismatch list."""
    mismatches = []
    for _, raw_row in raw_states.iterrows():
        derived = get_derived_group_round(
            panel, session_id, seg_idx, raw_row["group_id"], raw_row["round"]
        )
        if len(derived) == 0:
            continue
        if derived["state"].iloc[0] != raw_row["state"]:
            mismatches.append({
                "session": session_id, "segment": seg_idx,
                "group": raw_row["group_id"], "round": raw_row["round"],
                "raw": raw_row["state"], "derived": derived["state"].iloc[0],
            })
    return mismatches


def compare_field_per_group_round(panel, raw_segments, field_fn):
    """Apply field_fn to each group-round, return list of mismatch dicts."""
    mismatches = []
    for session_id in VALIDATION_SESSIONS:
        for seg_idx, raw_df in raw_segments[session_id].items():
            segment_mismatches = _compare_segment(
                panel, session_id, seg_idx, raw_df, field_fn
            )
            mismatches.extend(segment_mismatches)
    return mismatches


def _compare_segment(panel, session_id, seg_idx, raw_df, field_fn):
    """Compare one segment's group-rounds via field_fn."""
    mismatches = []
    raw_counts = count_sellers_from_raw(raw_df)
    for _, raw_row in raw_counts.iterrows():
        derived = get_derived_group_round(
            panel, session_id, seg_idx,
            raw_row["group_id"], raw_row["round"],
        )
        result = field_fn(raw_df, raw_row, derived)
        if result is not None:
            mismatches.append({
                "session": session_id, "segment": seg_idx,
                "group": raw_row["group_id"],
                "round": raw_row["round"], **result,
            })
    return mismatches


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def panel():
    if not PANEL_PATH.exists():
        pytest.skip(f"Panel not found: {PANEL_PATH}")
    return pd.read_csv(PANEL_PATH)


@pytest.fixture(scope="module")
def raw_segments():
    """Load all raw segment CSVs for validation sessions."""
    data = {}
    for session_id in VALIDATION_SESSIONS:
        session_dir = DATASTORE / session_id
        if not session_dir.exists():
            pytest.skip(f"Raw session dir not found: {session_dir}")
        data[session_id] = {}
        for seg_idx, seg_name in enumerate(SEGMENTS, start=1):
            data[session_id][seg_idx] = load_raw_segment(session_id, seg_name)
    return data


# =====
# Per group-round seller count validation
# =====
class TestSellerCountsVsRawData:
    """Compare derived panel seller counts to raw oTree exports."""

    def test_seller_counts_match_per_group_round(self, panel, raw_segments):
        """Every group-round seller count in derived panel matches raw data."""
        def check(raw_df, raw_row, derived):
            derived_sellers = derived["did_sell"].sum()
            if derived_sellers != raw_row["n_sellers"]:
                return {"raw": raw_row["n_sellers"], "derived": derived_sellers}
            return None

        mismatches = compare_field_per_group_round(panel, raw_segments, check)
        assert not mismatches, (
            f"{len(mismatches)} seller count mismatches:\n"
            + "\n".join(str(m) for m in mismatches[:10])
        )

    def test_state_values_match_raw(self, panel, raw_segments):
        """State values in derived panel match raw oTree data."""
        mismatches = []
        for session_id in VALIDATION_SESSIONS:
            for seg_idx, raw_df in raw_segments[session_id].items():
                raw_states = get_state_from_raw(raw_df)
                segment_mismatches = _check_states(
                    panel, session_id, seg_idx, raw_states
                )
                mismatches.extend(segment_mismatches)
        assert not mismatches, (
            f"{len(mismatches)} state mismatches:\n"
            + "\n".join(str(m) for m in mismatches[:10])
        )

    def test_player_count_per_group_round(self, panel, raw_segments):
        """Each group-round should have exactly 4 players in both sources."""
        for session_id in VALIDATION_SESSIONS:
            for seg_idx, raw_df in raw_segments[session_id].items():
                _assert_four_players_raw(raw_df, session_id, seg_idx)
                derived = get_derived_segment(panel, session_id, seg_idx)
                counts = derived.groupby(["group_id", "round"]).size()
                assert (counts == 4).all(), (
                    f"{session_id} seg {seg_idx}: derived != 4 players"
                )


# =====
# Aggregate metric validation
# =====
class TestAggregateMetricsVsRawData:
    """Validate aggregate seller count metrics computed from raw data."""

    def test_avg_sellers_from_raw(self, panel, raw_segments):
        """Avg sellers per group-round from raw matches derived panel."""
        for session_id, treatment in VALIDATION_SESSIONS.items():
            for seg_idx, raw_df in raw_segments[session_id].items():
                raw_counts = count_sellers_from_raw(raw_df)
                raw_avg = raw_counts["n_sellers"].mean()

                derived = panel[
                    (panel.session_id == session_id)
                    & (panel.segment == seg_idx)
                ]
                derived_avg = derived.groupby(
                    ["group_id", "round"]
                )["did_sell"].sum().mean()

                assert raw_avg == pytest.approx(derived_avg, abs=0.001), (
                    f"{session_id} seg {seg_idx}: "
                    f"raw avg={raw_avg:.4f} != derived avg={derived_avg:.4f}"
                )

    def test_zero_seller_rounds_from_raw(self, panel, raw_segments):
        """Zero-seller group-round counts from raw match derived panel."""
        for session_id in VALIDATION_SESSIONS:
            for seg_idx, raw_df in raw_segments[session_id].items():
                raw_counts = count_sellers_from_raw(raw_df)
                raw_zeros = (raw_counts["n_sellers"] == 0).sum()

                derived = panel[
                    (panel.session_id == session_id)
                    & (panel.segment == seg_idx)
                ]
                derived_zeros = (
                    derived.groupby(["group_id", "round"])["did_sell"]
                    .sum() == 0
                ).sum()

                assert raw_zeros == derived_zeros, (
                    f"{session_id} seg {seg_idx}: "
                    f"raw zeros={raw_zeros} != derived zeros={derived_zeros}"
                )

    def test_zero_seller_rounds_by_state(self, panel, raw_segments):
        """Zero-seller counts by state from raw match derived panel."""
        for session_id in VALIDATION_SESSIONS:
            for seg_idx, raw_df in raw_segments[session_id].items():
                raw_merged = _raw_counts_with_state(raw_df)
                derived = get_derived_segment(panel, session_id, seg_idx)
                for state in [0, 1]:
                    _assert_zero_counts_match(
                        raw_merged, derived, session_id, seg_idx, state
                    )


# %%
if __name__ == "__main__":
    main()
