# Issue 112: Build Pre-Sell-Click Emotions Dataset

## Summary

Create a derived dataset capturing facial emotion data from the milliseconds before each sell button click. The existing `emotions_traits_selling_dataset.csv` averages emotions over entire market periods, but analyzing the emotional state at the moment of a selling decision requires emotions from a narrow window just before the `sell_click_time`.

## Approach

For each sell event (where `sold=1` and `sell_click_time` is non-null), convert the oTree `sell_click_time` (Unix timestamp) to the iMotions relative timestamp using each participant's recording start time, then extract emotion frames from configurable windows (500ms, 1000ms, 2000ms, 5000ms) before the click. Average those frames and merge with the existing period/traits data.

## Changes Made

- `analysis/derived/build_presell_emotions.py`: New script that constructs the pre-sell emotions dataset with multi-window support (500ms, 1000ms, 2000ms, 5000ms). Identifies sell click events, converts timestamps between oTree and iMotions coordinate systems, extracts emotion frames from configurable windows, and merges with experiment/trait data.
- `analysis/tests/test_build_presell_emotions.py`: Integration tests verifying timestamp conversion, window extraction, merge logic, and multi-window output correctness.
- `pyproject.toml`: Updated project dependencies.

## Output

- Derived dataset saved to `datastore/derived/presell_emotions_traits_dataset.csv` containing emotion measures for each pre-sell event across multiple time windows.
