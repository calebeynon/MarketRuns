---
title: "Derived Datasets Pipeline"
type: tool
tags: [data-pipeline, derived-data, python, datasets]
summary: "Python scripts in analysis/derived/ that transform raw parsed data into analysis-ready CSV/parquet datasets"
status: draft
last_verified: "2026-04-21"
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
| `build_participant_risk_aversion.py` | Participant-level CRRA α (experiment MLE + survey-inverted) | Participant |

## Non-Builder Derived Artifacts

Some `datastore/derived/` artifacts are produced by analysis scripts rather than dedicated builders:

| Artifact | Producer | Columns | Approx. rows |
|----------|----------|---------|--------------|
| `equilibrium_sigma_grid.parquet` | `analysis/analysis/simulate_equilibrium.py` | treatment, alpha, n, pi, sigma | 328,328 (1001 α × 2 treatments × 4 n × 41 π) |

## Participant Risk Aversion Dataset

`datastore/derived/participant_risk_aversion.csv` (95 rows, participant-granularity) gives each participant's experiment-implied CRRA α with a confidence interval, plus a separately computed task-implied α from the survey lottery:

- `alpha_mle` — grid-search MLE over α ∈ {0.000, 0.001, …, 1.000} against the Magnani & Munro (2020) equilibrium σ grid (`equilibrium_sigma_grid.parquet`), using the participant's hold/sell decisions. Visualized in §5.5 (`implied_risk_aversion.pdf` / `visualize_implied_risk_aversion.R`).
- `alpha_task` — Gneezy-Potters lottery inversion from `risk_tolerance` ∈ {0, …, 20} via α = log(2.5) / log((20 + 1.5a)/(20 − a)). Set to NaN and `alpha_task_edge_flag=True` when `risk_tolerance ∈ {0, 20}`. Retained for reference; not used in the paper because its support (≈[0.24, 7.41] for integer non-edge allocations) lies largely outside the MLE grid [0, 1].

Columns: `session_id`, `player`, `treatment` (tr1/tr2), `n_decisions`, `alpha_mle`, `alpha_ci_low`, `alpha_ci_high` (95% likelihood-ratio set), `alpha_task`, `alpha_task_edge_flag`.

Inputs to `build_participant_risk_aversion.py`:
- `individual_period_dataset.csv` — decision rows (`already_sold == 0`)
- `survey_traits.csv` — `risk_tolerance`
- `equilibrium_sigma_grid.parquet` — σ(α, n, π) lookup

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

analysis/analysis/simulate_equilibrium.py
    └── equilibrium_sigma_grid.parquet
        └── build_participant_risk_aversion.py (+ individual_period_dataset.csv, survey_traits.csv)
            └── participant_risk_aversion.csv
```

## Related

- [Market Data Parser](wiki/tools/market-data-parser.md)
- [Analysis Scripts](wiki/tools/analysis-scripts.md)
