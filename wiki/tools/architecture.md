---
title: "Project Architecture"
type: tool
tags: [architecture, project-structure, otree, analysis]
summary: "Top-level directory layout, tech stack, and how the experiment, analysis, and paper components connect"
status: draft
last_verified: "2026-04-06"
---

## Summary

MarketRuns is an experimental economics platform built with oTree (Python) for studying market dynamics, information cascades, and trading behavior. The project has three major components: the experiment application, the analysis pipeline, and the paper.

## Key Points

- **Tech stack**: Python 3.10+, oTree 5.x, R (ggplot2/fixest for analysis), LaTeX (paper), uv (dependency management)
- **Three experiment variants**: `nonlivegame/` (Treatment 1), `nonlivegame_tr2/` (Treatment 2), `livegame/` (live version)
- **Data flows one way**: raw oTree exports in `datastore/` → derived datasets via Python scripts → regression/visualization in R → LaTeX tables/plots → paper

## Directory Layout

```
marketruns/
├── nonlivegame/          # Main oTree experiment (Treatment 1)
│   ├── settings.py       # Session config, app sequence, rooms
│   ├── quiz/             # Pre-experiment comprehension quiz
│   ├── chat_noavg/       # Segment 1 (no chat, baseline)
│   ├── chat_noavg2/      # Segment 2 (no chat)
│   ├── chat_noavg3/      # Segment 3 (with chat)
│   ├── chat_noavg4/      # Segment 4 (with chat)
│   ├── survey/           # Post-experiment survey
│   └── final/            # Final results and payoffs
├── nonlivegame_tr2/      # Treatment 2 variant
├── livegame/             # Live experiment version
├── analysis/
│   ├── market_data.py    # Core data parsing module (hierarchical OOP structure)
│   ├── derived/          # Python scripts that build derived datasets
│   ├── analysis/         # R scripts for regressions and visualization
│   ├── tests/            # pytest unit tests
│   ├── output/
│   │   ├── tables/       # LaTeX regression tables (.tex)
│   │   └── plots/        # Figures (.pdf, .png)
│   └── paper/            # LaTeX paper source (syncs to Overleaf)
├── datastore/            # Symlinked to Box cloud storage (raw + derived data)
│   ├── <session_folders> # Raw oTree CSV exports per session
│   └── derived/          # Processed datasets (CSV/parquet)
├── pyproject.toml        # uv project config (otree, matplotlib, scipy)
└── wiki/                 # Project knowledge base
```

## Data Flow

1. **Raw data** (`datastore/<session>/`) — oTree CSV exports from lab sessions
2. **Parsing** (`analysis/market_data.py`) — hierarchical Python objects: Experiment → Session → Segment → Round → Period → Player
3. **Derived datasets** (`analysis/derived/build_*.py`) — flatten parsed data into analysis-ready CSVs in `datastore/derived/`
4. **Regression & visualization** (`analysis/analysis/*.R`) — fixest regressions and ggplot2 plots
5. **Output** (`analysis/output/tables/` and `analysis/output/plots/`) — LaTeX tables and PDF figures
6. **Paper** (`analysis/paper/main.tex`) — includes tables/plots via `\input{}` and `\includegraphics{}`

## Key Configuration

- **16 participants per session**, grouped into 4 fixed groups of 4
- **Signal accuracy**: 67.5% (`PCORRECT = 0.675`)
- **Initial price**: $8, decreases by $2 per seller
- **Participation fee**: $7.50, real-world rate: $0.25/point
- **Session flow**: `quiz → chat_noavg → chat_noavg2 → chat_noavg3 → chat_noavg4 → survey → final`

## Related

- [Market Data Parser](wiki/tools/market-data-parser.md)
- [Derived Datasets Pipeline](wiki/tools/derived-datasets.md)
- [Analysis Scripts](wiki/tools/analysis-scripts.md)
