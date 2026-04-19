---
title: "Data Structure: Raw + Derived"
type: concept
tags: [data, datastore, schemas, derived-data, raw-data]
summary: "Datastore layout, raw oTree session-folder structure, and the schemas of the major derived datasets"
status: active
last_verified: "2026-04-19"
---

## Summary

`datastore/` is a symlink to a Box folder shared with collaborators. It holds raw oTree exports per session, iMotions facial-recording exports per session, and a `derived/` subdirectory of analysis-ready CSVs produced by `analysis/derived/build_*.py` scripts.

## Top-Level Layout

```
datastore/   (symlink → /Users/caleb/Library/CloudStorage/Box-Box/.../BankMarketRuns)
├── 1_11-7-tr1/        # Session 1 (T1, 2025-11-07)
├── 2_11-10-tr2/       # Session 2 (T2, 2025-11-10)
├── 3_11-11-tr2/
├── 4_11-12-tr1/
├── 5_11-14-tr2/
├── 6_11-18-tr1/       # Session 6
├── 11-5-tr1/          # Pilot data (excluded from main analysis)
├── oct 10 pilot/      # Earlier pilot (excluded)
├── annotations/       # iMotions annotation files
├── codes/             # Participant code lookups
├── derived/           # Processed datasets (CSV) — see below
├── imotions/          # iMotions facial-recording exports per session
│   ├── 1/  2/  3/  4/  5/  6/   # one folder per session
├── instruction video/
├── IRB/
├── literature/
├── papers/
└── Market Runs Presentation.pdf
```

## Raw Session Folder

Each `<n>_<date>-<treatment>/` contains oTree-exported CSVs:

```
all_apps_wide_<date>.csv       # one row per participant, all variables across apps
chat_noavg_<date>.csv          # segment 1 panel
chat_noavg2_<date>.csv         # segment 2 panel
chat_noavg3_<date>.csv         # segment 3 panel
chat_noavg4_<date>.csv         # segment 4 panel
ChatMessages-<date>.csv        # chat log (channel, nickname, body, timestamp)
final_<date>.csv               # post-experiment summary app
PageTimes-<date>.csv           # page-load timestamps (used for iMotions alignment)
quiz_<date>.csv                # comprehension quiz responses
survey_<date>.csv              # personality/demographic survey
videos/                        # raw webcam recordings (not committed)
```

`market_data.parse_experiment(csv_path, chat_path)` consumes these into a hierarchical object structure (see [Market Data Parser](../tools/market-data-parser.md)).

## iMotions Folder

```
datastore/imotions/<session_num>/
├── 001_R3.csv      # one CSV per participant, named <participant_index>_<participant_label>.csv
├── 002_Q3.csv
├── ...
```

Per-participant CSVs contain frame-by-frame facial-emotion scores (Affectiva), edited annotations marking MarketPeriod start/end, and recording metadata. **iMotions period offset**: annotations use `m{N}` mapping to oTree period `N-1` — see [iMotions Integration](imotions-integration.md).

## Derived Datasets (`datastore/derived/`)

All produced by `analysis/derived/build_*.py` scripts. The full list and the script that builds each is in [Derived Datasets Pipeline](../tools/derived-datasets.md). Below are the schemas of the most-used datasets.

### `individual_period_dataset.csv` (player × period panel)

```
session_id, segment, round, period, group_id, player, treatment,
signal, state, price, sold, already_sold, prior_group_sales
```
- **`sold`** = 1 only in the period the player sells (event indicator)
- **`already_sold`** = 1 if player sold in any prior period of this round
- **`prior_group_sales`** = number of teammates who sold before this period in this round

### `individual_round_panel.csv` (player × round panel)

```
session_id, treatment, segment, group_id, round, player,
signal, state, sell_period, did_sell, sell_price
```
- **`sell_period`** = NA if player held the asset to terminal
- **`sell_price`** = liquidation price actually received (T2 may be the average across same-period sellers)

### `group_round_welfare.csv` (group × round)

```
session, segment_num, round_num, group_id, welfare
```
- **Welfare** = 1.0 identically when `state == 0` (variation only in good-state rounds — restrict regressions to z=1)

### `imotions_period_emotions.csv` (player × period emotion aggregates)

```
session_id, segment, round, period, player,
anger_mean, contempt_mean, disgust_mean, fear_mean, joy_mean,
sadness_mean, surprise_mean, engagement_mean, valence_mean, n_frames
```
- One row per player-period-frame-window aggregate (mean over all frames within the period)

### `presell_emotions_traits_dataset.csv` (event-level pre-sell emotions)

The widest derived dataset. Per-period observations (with optional `sell_click_time`) joined to multi-window emotion aggregates (50ms / 100ms / 500ms / 1000ms / 2000ms windows before sell-click) and player traits. Used for emotion regressions where timing matters.

Key columns beyond the standard panel:
```
sell_click_time, n_frames_<window>, <emotion>_mean_<window>,
extraversion, agreeableness, conscientiousness, neuroticism,
openness, impulsivity, state_anxiety, risk_tolerance,
age, gender, global_group_id
```

### `survey_traits.csv` (player-level)

```
session_id, player, extraversion, agreeableness, conscientiousness,
neuroticism, openness, impulsivity, state_anxiety, risk_tolerance,
age, gender
```

### Other notable derived files

| File | Purpose |
|------|---------|
| `first_sale_data.csv` / `first_sales.csv` | Per-round first-sale events |
| `first_seller_round_data.csv` | Group-round first-seller record |
| `first_seller_analysis_data.csv` | First-seller events joined with traits |
| `holdout_next_round_analysis.csv` | Lagged liquidation-payoff effects |
| `holdout_signal_at_sale.csv` | Signal value at the moment of sale |
| `holdout_anger_analysis.csv` | Anger-specific holdout subset |
| `ordinal_selling_position.csv` | Selling-order rank per group-round |
| `chat_mitigation_dataset.csv`, `chat_sentiment_dataset.csv` | Chat-related analyses |
| `emotion_spikes_analysis_dataset.csv` | Emotion-spike detection |
| `equilibrium_thresholds.csv` | Numerical M&M (2020) Appendix D thresholds |
| `melted_data.csv` | Long-form melted dataset for misc plotting |
| `imotions_missing_summary.csv` | Per-session iMotions missingness summary |

## Group-Identifier Caveat

Groups are reshuffled between segments. Use `global_group_id` (constructed in derived datasets as `session × segment × group_id` → unique 96-cluster ID) for clustering. Clustering at session-level alone understates SEs; clustering at `session_group` is methodologically invalid (links different participants across segments).

## Related

- [Market Data Parser](../tools/market-data-parser.md)
- [Derived Datasets Pipeline](../tools/derived-datasets.md)
- [iMotions Integration](imotions-integration.md)
- [Known Quirks & Gotchas](../methods/known-quirks.md)
