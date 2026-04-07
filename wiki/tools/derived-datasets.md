---
title: "Derived Datasets Pipeline"
type: tool
tags: [data-pipeline, derived-data, python, datasets]
summary: "Python scripts in analysis/derived/ that transform raw parsed data into analysis-ready CSV/parquet datasets"
status: draft
last_verified: "2026-04-06"
---

## Summary

The `analysis/derived/` directory contains Python scripts that use `market_data.py` to parse raw oTree exports and produce flat, analysis-ready datasets saved to `datastore/derived/`. Each script focuses on one analytical unit (individual-period, individual-round, first-seller, etc.).

## Key Points

- All scripts follow the pattern: parse raw data → compute derived variables → write to `datastore/derived/`
- Each script has a corresponding test in `analysis/tests/`
- Scripts are run with `uv run python analysis/derived/<script>.py`

## Dataset Builders

| Script | Output | Unit of Analysis |
|--------|--------|-----------------|
| `build_individual_period_dataset.py` | Individual-period panel | Player × period |
| `build_individual_period_dataset_extended.py` | Extended period panel (with emotions) | Player × period |
| `build_individual_round_dataset.py` | Individual-round panel | Player × round |
| `build_first_sale_dataset.py` | First sale events | Player × round (sellers only) |
| `build_first_seller_round_dataset.py` | First seller in each round | Group × round |
| `build_first_seller_analysis_dataset.py` | First seller traits analysis | First seller events with traits |
| `build_holdout_next_round_dataset.py` | Holdout prediction data | Player × round |
| `build_ordinal_selling_position.py` | Selling order position | Seller × round |
| `build_survey_traits_dataset.py` | Survey personality traits | Player-level |
| `build_emotions_traits_dataset.py` | iMotions emotion + traits | Player-level |
| `build_imotions_period_emotions.py` | Period-level emotions | Player × period |
| `build_imotions_missing_summary.py` | iMotions data missingness | Summary statistics |
| `build_group_round_timing_dataset.py` | Group-round selling timing | Group × round |
| `build_welfare_dataset.py` | Welfare computation | Group × round |

## Data Dependencies

```
Raw oTree CSVs (datastore/<session>/)
    └── market_data.py (parse_experiment)
        ├── build_individual_period_dataset.py
        │   └── build_individual_period_dataset_extended.py (adds emotions)
        ├── build_individual_round_dataset.py
        ├── build_first_sale_dataset.py
        │   └── build_first_seller_round_dataset.py
        │       └── build_first_seller_analysis_dataset.py (adds survey traits)
        ├── build_holdout_next_round_dataset.py
        ├── build_ordinal_selling_position.py
        ├── build_survey_traits_dataset.py
        ├── build_imotions_period_emotions.py
        ├── build_group_round_timing_dataset.py
        └── build_welfare_dataset.py
```

## Related

- [Market Data Parser](wiki/tools/market-data-parser.md)
- [Analysis Scripts](wiki/tools/analysis-scripts.md)
