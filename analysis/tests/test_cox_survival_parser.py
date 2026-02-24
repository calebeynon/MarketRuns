"""
Purpose: Cross-validate Cox survival data against raw oTree parser
Author: Claude Code
Date: 2026-02-23

Tests cross-validate sold, signal, and first-seller identity against
the market_data.py parser operating on raw session exports.
"""

import pytest
from cox_test_helpers import SEGMENT_MAP


# =====
# Main function
# =====
def main():
    """Run all tests with verbose output."""
    pytest.main([__file__, "-v"])


# =====
# Shared parser lookup helpers
# =====
def _lookup_player(row, parsed_experiments):
    """Return PlayerPeriodData for a dataset row, or None."""
    exp = parsed_experiments.get(row["session_id"])
    if exp is None or not exp.sessions:
        return None
    session = exp.sessions[0]
    seg = session.get_segment(SEGMENT_MAP[int(row["segment"])])
    rnd = seg.get_round(int(row["round"])) if seg else None
    per = rnd.get_period(int(row["period"])) if rnd else None
    return per.get_player(row["player"]) if per else None


def _get_round_from_row(row, parsed_experiments):
    """Return (session, seg_name, rnd) for a dataset row."""
    exp = parsed_experiments.get(row["session_id"])
    if exp is None or not exp.sessions:
        return None, None, None
    session = exp.sessions[0]
    seg_name = SEGMENT_MAP[int(row["segment"])]
    seg = session.get_segment(seg_name)
    rnd = seg.get_round(int(row["round"])) if seg else None
    return session, seg_name, rnd


def _get_player_group(session, segment_name, player_label):
    """Look up which group a player belongs to in a segment."""
    segment = session.get_segment(segment_name)
    if not segment:
        return None
    for group in segment.groups.values():
        if player_label in group.player_labels:
            return group.group_id
    return None


# =====
# Cross-validation: sold field
# =====
class TestRawParserSold:
    """Cross-validate sold against market_data.py parser."""

    def test_sold_matches_parser_all_sellers(
        self, emotion_filtered, parsed_experiments
    ):
        """Validate sold field for 200 random All Sellers rows."""
        sample = emotion_filtered.sample(200, random_state=42)
        mismatches, checked = _validate_sold(sample, parsed_experiments)
        assert checked > 100, f"Only checked {checked} rows"
        assert mismatches == 0, f"{mismatches}/{checked} mismatches"

    def test_sold_matches_parser_first_sellers(
        self, first_seller_data, parsed_experiments
    ):
        """Validate sold field for 200 random first-seller rows."""
        n = min(200, len(first_seller_data))
        sample = first_seller_data.sample(n, random_state=43)
        mismatches, checked = _validate_sold(sample, parsed_experiments)
        assert checked > 50, f"Only checked {checked} rows"
        assert mismatches == 0, f"{mismatches}/{checked} mismatches"


def _validate_sold(sample, parsed_experiments):
    """Check sold field against parser for a sample of rows."""
    mismatches = checked = 0
    for _, row in sample.iterrows():
        player = _lookup_player(row, parsed_experiments)
        if player is None:
            continue
        checked += 1
        parser_sold = 1 if player.sold_this_period else 0
        if row["sold"] != parser_sold:
            mismatches += 1
    return mismatches, checked


# =====
# Cross-validation: signal
# =====
class TestRawParserSignal:
    """Cross-validate signal against market_data.py parser."""

    def test_signal_matches_parser(
        self, emotion_filtered, parsed_experiments
    ):
        """Validate signal field for 200 random rows."""
        sample = emotion_filtered.sample(200, random_state=99)
        mismatches = checked = 0
        for _, row in sample.iterrows():
            player = _lookup_player(row, parsed_experiments)
            if player is None or player.signal is None:
                continue
            checked += 1
            if abs(row["signal"] - player.signal) > 0.001:
                mismatches += 1
        assert checked > 100, f"Only checked {checked}"
        assert mismatches == 0, f"{mismatches}/{checked} mismatches"


# =====
# Cross-validation: first-seller identity
# =====
class TestRawParserFirstSeller:
    """Verify first-seller identity against raw session data."""

    def test_first_seller_identity_matches_parser(
        self, first_seller_data, parsed_experiments
    ):
        """Verify the player who sold is the first seller in raw data."""
        sold_rows = first_seller_data[first_seller_data["sold"] == 1]
        mismatches, checked = _check_first_seller_periods(
            sold_rows, parsed_experiments
        )
        assert checked > 200, f"Only checked {checked}"
        assert mismatches == 0, f"{mismatches}/{checked} mismatches"

    def test_no_prior_sales_in_raw_data(
        self, first_seller_data, parsed_experiments
    ):
        """Verify no group sales before first-seller sale period."""
        sold_rows = first_seller_data[
            first_seller_data["sold"] == 1
        ].sample(min(100, len(first_seller_data)), random_state=77)
        mismatches, checked = _check_no_prior_sales(
            sold_rows, parsed_experiments
        )
        assert checked > 50, f"Only checked {checked}"
        assert mismatches == 0, f"{mismatches}/{checked} had prior sales"


def _check_first_seller_periods(sold_rows, parsed_experiments):
    """Check that each first-seller sale matches raw data."""
    mismatches = checked = 0
    for _, row in sold_rows.iterrows():
        session, seg_name, rnd = _get_round_from_row(
            row, parsed_experiments
        )
        if rnd is None:
            continue
        first_period = _find_first_sale_period(
            rnd, session, int(row["group_id"]), seg_name
        )
        if first_period is None:
            continue
        checked += 1
        if int(row["period"]) != first_period:
            mismatches += 1
    return mismatches, checked


def _check_no_prior_sales(sold_rows, parsed_experiments):
    """Check that no group sales occurred before each first sale."""
    mismatches = checked = 0
    for _, row in sold_rows.iterrows():
        session, seg_name, rnd = _get_round_from_row(
            row, parsed_experiments
        )
        if rnd is None:
            continue
        sales = _count_sales_before(
            rnd, session, int(row["group_id"]),
            seg_name, int(row["period"]),
        )
        checked += 1
        if sales != 0:
            mismatches += 1
    return mismatches, checked


def _find_first_sale_period(rnd, session, group_id, seg_name):
    """Find the first period with any sale in a group-round."""
    for period_num in sorted(rnd.periods.keys()):
        period = rnd.get_period(period_num)
        if not period:
            continue
        for label, pdata in period.players.items():
            if pdata.sold_this_period:
                pg = _get_player_group(session, seg_name, label)
                if pg == group_id:
                    return period_num
    return None


def _count_sales_before(rnd, session, group_id, seg_name, before):
    """Count group sales before a given period."""
    total = 0
    for period_num in sorted(rnd.periods.keys()):
        if period_num >= before:
            break
        period = rnd.get_period(period_num)
        if not period:
            continue
        for label, pdata in period.players.items():
            if pdata.sold_this_period:
                pg = _get_player_group(session, seg_name, label)
                if pg == group_id:
                    total += 1
    return total


# %%
if __name__ == "__main__":
    main()
