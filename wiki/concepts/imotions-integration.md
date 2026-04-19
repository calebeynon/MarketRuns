---
title: "iMotions Facial-Emotion Data Integration"
type: concept
tags: [imotions, emotions, facial-recognition, data-integration, affectiva]
summary: "How iMotions facial-recording data joins to oTree player-period rows: period offset, timestamp conversion, aggregation windows"
status: active
last_verified: "2026-04-19"
---

## Summary

iMotions records facial expression scores (Affectiva: 8 discrete emotions + valence) frame-by-frame per participant during the experiment. These frames must be aligned to oTree market periods to produce the player-period emotion panel used in regressions. Integration is fragile — at least three bugs (#25, #32, #112) have come from misaligned timestamps or wrong aggregation.

## Files

- **Per-participant CSV**: `datastore/imotions/<session>/<index>_<label>.csv`
- **Annotations**: `datastore/annotations/` plus per-participant edited annotations marking `MarketPeriod` start/end events
- **Aggregated output**: `datastore/derived/imotions_period_emotions.csv` (player × period × emotion means)
- **Pre-sell windows**: `datastore/derived/presell_emotions_traits_dataset.csv` (multi-window 50/100/500/1000/2000 ms aggregates before each sell-click)

## The Period Offset (Issue #25)

Annotations use `m{N}` event markers that map to oTree `period N - 1`:

| Annotation tag | oTree period |
|----------------|--------------|
| `m2` | 1 |
| `m3` | 2 |
| `m4` | 3 |

The `generate_annotations_unfiltered_v2.py` script pre-increments the marker counter before recording the first MarketPeriod. **Any new script that joins iMotions to oTree must apply this offset.** Issues #18 and #19 had to be audited for it.

## Timestamp Conversion (Issue #112)

- oTree `sell_click_time` is **Unix epoch seconds** (UTC).
- iMotions timestamps are **relative milliseconds** from each participant's recording start.
- Conversion requires the per-participant recording start time, recovered from the edited_data annotations.

The pre-sell emotions builder (`build_presell_emotions_traits.py`, PR #113) handles this conversion.

## Aggregation Windows

Pre-sell emotion features come in multiple time-window means before the sell-click:

| Window | Use case |
|--------|----------|
| 50 ms | Microexpressions |
| 100 ms | Short-window |
| 500 ms | Standard |
| 1000 ms | One-second pre-click |
| 2000 ms | Two-second pre-click (default in many specs) |

For each window, columns follow the pattern `<emotion>_mean_<window>` and `n_frames_<window>` (frame count within the window, used as a quality filter).

For period-level aggregates (no sell event), `imotions_period_emotions.csv` uses a single mean over all frames within the oTree period.

## Missingness

Many sessions/participants have partial or missing iMotions data (recording failures, calibration issues, post-hoc filtering). `analysis/derived/build_imotions_missing_summary.py` produces `datastore/derived/imotions_missing_summary.csv` reporting per-session missingness. Use this when interpreting any emotion-conditioned regression.

The 95th-percentile aggregation choice was the subject of issue #32 — use the canonical aggregator in the existing builders rather than rolling your own.

## Discrete Emotions vs. Valence

The paper reports both:
- **Discrete-emotion specifications** (anger, contempt, disgust, fear, joy, sadness, surprise, engagement) → e.g., main Cox model
- **Valence-only specifications** (composite positivity score) → Appendix G robustness
- **No-valence specifications** (discrete emotions without composite) → Appendix H robustness

R scripts have `_valence_only` and `_no_valence` suffixed variants for each main regression where this matters.

## Joining Pattern

Standard join keys:
```
session_id × segment × round × period × player
```
plus `sell_click_time` for the pre-sell variants.

`global_group_id` (= session × segment × group_id) is the correct cluster ID since groups reshuffle between segments.

## Related

- [Data Structure](data-structure.md)
- [Derived Datasets Pipeline](../tools/derived-datasets.md)
- [Known Quirks](../methods/known-quirks.md)
- [Main Paper Results](../papers/main-paper-results.md)
