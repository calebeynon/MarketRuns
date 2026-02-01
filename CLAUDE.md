# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## GitHub

- **Repository owner**: `calebeynon`

## Project Overview

MarketRuns is an experimental economics platform built with oTree for studying market dynamics, information cascades, and trading behavior. The platform runs multi-period market experiments with integrated chat systems and real-time signal processing.

## Commands

### Development Server
```bash
cd nonlivegame && otree devserver     # Treatment 1 (main)
cd nonlivegame_tr2 && otree devserver # Treatment 2
```

### Dependency Management (uv)
```bash
uv sync                    # Install dependencies
uv run python <script>     # Run Python scripts
```
## Architecture

### Experiment Directories
- **`nonlivegame/`** - Main experimental framework (Treatment 1)
- **`nonlivegame_tr2/`** - Treatment 2 variant
- **`livegame/`** - Live experiment version

### Analysis Directory Structure
```
analysis/
├── derived/          # Scripts that create/modify datasets
├── analysis/         # Regression and statistical analysis scripts
├── tests/            # Unit tests
└── market_data.py    # Core data parsing module
```

### Data Storage (`datastore/`)
```
datastore/            # Symlinked to Box cloud storage
├── 1_11-7-tr1/       # Session folders (raw oTree exports)
├── 2_11-10-tr2/
├── ...
└── derived/          # Processed datasets (output from analysis/derived scripts)
```

### oTree App Structure (nonlivegame)
Each segment is a separate oTree app with its own `__init__.py`:

```
nonlivegame/
├── settings.py           # Session config, app sequence, rooms
├── quiz/                 # Pre-experiment comprehension quiz
├── chat_noavg/           # Segment 1 (no chat)
├── chat_noavg2/          # Segment 2 (no chat)
├── chat_noavg3/          # Segment 3 (with chat)
├── chat_noavg4/          # Segment 4 (with chat)
├── survey/               # Post-experiment survey
└── final/                # Final results and payoffs
```

Session flow: `quiz → chat_noavg → chat_noavg2 → chat_noavg3 → chat_noavg4 → survey → final`

### Key oTree Concepts in This Codebase

**Rounds vs Periods**: oTree's `round_number` represents periods. Custom round/period tracking uses:
- `round_number_in_segment` - Which trading round (1-14) within the segment
- `period_in_round` - Which period within that round
- `PERIODS_PER_ROUND` list in constants determines round boundaries

**Group Matrix**: Fixed grouping defined in `creating_session()`:
```python
grouping = [[1,5,9,13],[2,6,10,14],[3,7,11,15],[4,8,12,16]]  # 4 groups of 4
```

**Participant State**: Uses `participant.vars` for cross-page state:
- `sold`, `payoff`, `price`, `signal`
- `signal_history`, `price_history`
- `pay_list`, `pay_list_random`

### Data Analysis Module (`analysis/market_data.py`)

Hierarchical data structure:
```
MarketRunsExperiment
└── Session
    └── Segment (chat_noavg, chat_noavg2, etc.)
        ├── Group (4 groups, constant across rounds)
        └── Round (1-14 per segment)
            ├── Period
            │   └── PlayerPeriodData
            └── ChatMessage
```

Key classes: `MarketRunsExperiment`, `Session`, `Segment`, `Round`, `Period`, `PlayerPeriodData`, `ChatMessage`, `Group`

### Market Mechanics
- **Signal accuracy**: 67.5% (`PCORRECT = 0.675`)
- **Initial price**: $8, decreases by $2 per seller
- **State**: Binary (0 or 1), determines liquidation value
- **Bayesian updating**: Signal history updated via `set_signal()` function

## Data Paths

- **Raw session data**: `datastore/<session_folder>/` (e.g., `datastore/1_11-7-tr1/`)
- **Derived datasets**: `datastore/derived/` (created by scripts in `analysis/derived/`)

## Visualization Standards

- **All visualizations must be created using ggplot2 in R**
- **Never include plot titles** - titles go in the paper/document, not the figure
- Visualization scripts go in `analysis/` with descriptive names (e.g., `visualize_selling_timing.R`)
- Output plots saved to `analysis/output/plots/`

## Analysis Output Standards

- **LaTeX tables**: Save to `analysis/output/tables/` (not `analysis/output/analysis/`)
- **Plots**: Save to `analysis/output/plots/`
