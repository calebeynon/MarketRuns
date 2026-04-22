---
title: "Getting Started: New Collaborator Onboarding"
type: skill
tags: [onboarding, getting-started, setup, workflow, orientation]
summary: "Day-1 orientation for a new collaborator: what the project is, how to set it up, the repo layout, data pipeline, and the commit/PR conventions"
status: active
last_verified: "2026-04-22"
---

## Summary

MarketRuns is an experimental economics project built on [oTree](https://www.otree.org/) that studies **market runs** — situations where investors rush to sell because prices fall as liquidity disappears. The experiment combines repeated trading markets, a within-session chat manipulation, a between-subject payoff treatment, survey traits, and iMotions facial-emotion measures. This article gets you from "clone the repo" to "running a derived-data script and reading the paper" in under an hour.

## Key Points

- Install with `uv sync`. Never install Python packages globally.
- Two experiment trees exist: `nonlivegame/` (Treatment 1, random payoff) and `nonlivegame_tr2/` (Treatment 2, average payoff). They are separate — not a config flag.
- `datastore/` is a Box symlink, not a local folder. Don't reorganize files there.
- Analysis flows: raw oTree CSV → `analysis/market_data.py` parser → `analysis/derived/build_*.py` → `datastore/derived/*.csv` → `analysis/analysis/*.R` → `analysis/output/{tables,plots}/` → `analysis/paper/main.tex`.
- Branch naming: `issue_<N>_<short_slug>`. PR body MUST start with `Closes #N`. Always go through `/pr-generator`.
- Python: `main()` first, `pathlib` only, ≤300 LOC/file, ≤20 LOC/function.
- R: `ggplot2` only, never put titles on plots — they belong in the paper.

## Details

### 1. Environment Setup

```bash
git clone git@github.com:calebeynon/marketruns.git
cd marketruns
uv sync
```

- Run any Python tool with `uv run python <script.py>`. To add a dep: `uv add <package>`.
- Conda fallback is only for environments without `uv`; this repo uses `uv`.
- Data lives behind a Box symlink at `datastore/`. Set up Box Drive locally (shared folder `SharedFolder_Caleb_Paan/BankMarketRuns`) before the symlink resolves.

### 2. Reading Order (Day 1)

Read these files, in order. You'll understand the project end-to-end after.

1. `CLAUDE.md` — project conventions, architecture notes, session flow, terminology.
2. [Project Architecture](../tools/architecture.md) — top-down picture of experiment, analysis, and paper.
3. [Experiment Design](../concepts/experiment-design.md) — 2×2 design, groups, signals, prices.
4. [Data Structure](../concepts/data-structure.md) — raw vs derived layout.
5. [Derived Datasets Pipeline](../tools/derived-datasets.md) — what each builder produces.
6. `analysis/paper/main.tex` (abstract + intro) — the research question the code serves.

### 3. Top-Level Directory Map

| Path | Purpose | When you'll touch it |
|---|---|---|
| `nonlivegame/` | oTree app tree for Treatment 1 (random payoff) | Running/editing baseline experiment |
| `nonlivegame_tr2/` | oTree app tree for Treatment 2 (average payoff) | Working on the Tr2 variant |
| `livegame/` | Separate live-session version | Rare — only for live experiment configs |
| `analysis/market_data.py` | Core parser: raw CSV → Python objects | Anytime you load session data in Python |
| `analysis/derived/` | Scripts that build analysis-ready CSVs | Adding a new dataset or fixing a column |
| `analysis/analysis/` | R scripts (fixest regressions, ggplot2 figures) | Running regressions, making plots |
| `analysis/tests/` | pytest suite | Running tests, adding tests for a new builder |
| `analysis/paper/` | LaTeX sources and compiled PDF | Editing the paper, adding tables/figures |
| `analysis/output/tables/` | Generated `.tex` tables | Never edit by hand — regenerate via R/Python |
| `analysis/output/plots/` | Generated `.pdf` figures | Never edit by hand — regenerate via R |
| `datastore/` | Box-backed raw session folders + `derived/` | Reading data only — don't reorganize |
| `wiki/` | This knowledge base | Update when your change affects documented content |

### 4. Day-1 Commands

```bash
# Dev servers (one per treatment tree)
cd nonlivegame && otree devserver       # Treatment 1
cd nonlivegame_tr2 && otree devserver   # Treatment 2

# Tests
uv run pytest analysis/tests

# Build a derived dataset
uv run python analysis/derived/build_individual_period_dataset.py

# Compile the paper — ALL FOUR PASSES REQUIRED
cd analysis/paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
open main.pdf
```

Skipping any LaTeX pass produces `?` for citations and cross-references.

### 5. Data Pipeline at 10,000 ft

```text
datastore/<session>/*.csv  (oTree raw exports)
    → analysis/market_data.py (parser)
    → analysis/derived/build_*.py
    → datastore/derived/*.csv | *.parquet
    → analysis/analysis/*.R
    → analysis/output/{tables,plots}/
    → analysis/paper/main.tex

datastore/imotions/<session>/*.csv + datastore/annotations/
    → analysis/derived/build_imotions_period_emotions.py
    → datastore/derived/imotions_period_emotions.csv
    → merged into downstream analyses
```

#### Session folder naming

Pattern: `<sessionNumber>_<month-day>-<treatment>` → `1_11-7-tr1`, `2_11-10-tr2`, etc. Current sessions run `1_11-7-tr1` through `6_11-18-tr1`. Pilot folders (`11-5-tr1/`, `oct 10 pilot/`) are **excluded** from the paper sample.

#### Derived datasets you'll use most often

| File | One row = | Builder |
|---|---|---|
| `datastore/derived/individual_period_dataset.csv` | player × period | `build_individual_period_dataset.py` |
| `datastore/derived/individual_round_panel.csv` | player × round | `build_individual_round_dataset.py` |
| `datastore/derived/survey_traits.csv` | participant | `build_survey_traits_dataset.py` |
| `datastore/derived/imotions_period_emotions.csv` | player × period emotion aggregate | `build_imotions_period_emotions.py` |
| `datastore/derived/first_seller_analysis_data.csv` | player × round (first-seller analysis) | `build_first_seller_analysis_dataset.py` |

See [Derived Datasets Pipeline](../tools/derived-datasets.md) for the full catalogue.

#### First-look recipe

```python
from pathlib import Path
from analysis.market_data import parse_experiment

# Replace filenames with the actual CSVs in the session folder
session_dir = Path("datastore/1_11-7-tr1")
exp = parse_experiment(
    session_dir / "all_apps_wide_2025-11-07.csv",
    session_dir / "ChatMessages-2025-11-07.csv",
)
session = exp.sessions[0]
print(list(session.segments.keys()))
# ['chat_noavg', 'chat_noavg2', 'chat_noavg3', 'chat_noavg4']

import pandas as pd
df = pd.read_csv("datastore/derived/individual_period_dataset.csv")
print(df[["session_id", "segment", "round", "period", "group_id", "player", "sold"]].head())
```

For the full object model (`MarketRunsExperiment → Session → Segment → Round → Period → PlayerPeriodData`), see [Market Data Parser](../tools/market-data-parser.md).

### 6. Key Concepts to Internalize

- **Rounds vs periods vs segments** — oTree's native `round_number` is NOT the economic round. Use `round_number_in_segment` and `period_in_round` from the parser. Segments are the four major experiment blocks (`chat_noavg{,2,3,4}`). See [Experiment Design](../concepts/experiment-design.md).
- **`participant.vars`** — cross-page state carrier (`sold`, `payoff`, `signal_history`, payoff lists).
- **Group matrix** — starts as `[[1,5,9,13],[2,6,10,14],[3,7,11,15],[4,8,12,16]]`, reshuffles between segments. "Group 1" in segment 1 ≠ "Group 1" in segment 3.
- **Signal mechanism** — binary public signal each period with `PCORRECT = 0.675`; beliefs update Bayesianly.
- **iMotions integration** — annotation marker `mN` maps to oTree period `N-1`. Missingness is common; check `imotions_missing_summary.csv` for coverage. See [iMotions Integration](../concepts/imotions-integration.md).

### 7. Workflow Rules

#### Branch naming

```text
issue_<issue_number>_<short_snake_case_slug>
```

Recent examples: `issue_109_equilibrium_simulation`, `issue_112_presell_emotions`, `issue_100_did_learning_communication`. Start from `/start-issue` which creates the issue, branch, and worktree in one step.

#### Commits

- Include the issue number (`(#109)`).
- Bulleted outline of changes in the body.
- **Never commit files >1MB, never commit `.csv` files.**

Example (commit `06e9f01`):
```text
Implement M&M (2020) Appendix D equilibrium simulation (#109) (#115)

* Add #109 equilibrium simulation with Bellman solver and per-seller predictions
```

#### Pull requests

- Use `/pr-generator` — don't create PRs by hand.
- PR body MUST start with `Closes #N`. Omitting it leaves the linked issue open after merge.
- Include a summary, a test-plan checklist, and map each output to its generating script + inputs.

#### Python style (`~/.claude/rules/python-standards.md`)

- `main()` is the **first** function defined.
- File-path globals in `ALL_CAPS` near the top, always `pathlib.Path`.
- End with `if __name__ == "__main__": main()`.
- ≤300 LOC/file, ≤20 LOC/function. Fail loudly — no bare `except`.
- Validate: `uv run python ~/.claude/scripts/check_python_standards.py <script.py>`.

#### R style

- All plots in `ggplot2`. Never put titles on figures (titles live in the paper).
- Plots → `analysis/output/plots/`. Tables → `analysis/output/tables/`.
- Validate: `Rscript ~/.claude/scripts/check_r_standards.R <script.R>`.

#### Paper & Overleaf

- `analysis/paper/` syncs to Overleaf automatically on push to `main` via `.github/workflows/sync-overleaf.yml`.
- `\input{bare_filename}` — no extension, no directory prefix.
- `\includegraphics{filename.pdf}` — extension **required** (Overleaf sync silently drops figures without it).
- Pull Overleaf edits with the rsync approach — see [Overleaf Sync](overleaf-sync.md). Don't use `git subtree pull`.

#### Wiki maintenance

If your PR changes data structure, experiment design, paper content, iMotions integration, architecture, or workflow conventions, update the matching wiki article **in the same PR**. Bump `last_verified` to today. After merge, run `/kb sync` to publish.

### 8. Top 5 Gotchas

1. **`nonlivegame/` vs `nonlivegame_tr2/` are separate trees**, not a config switch. Check which one you're editing.
2. **Pilot sessions are excluded** from analysis (`datastore/11-5-tr1/`, `datastore/oct 10 pilot/`).
3. **Group identity doesn't persist across segments** — use a global group id for clustering, not raw group index.
4. **iMotions missingness is common** — don't assume every period has emotion data.
5. **`\includegraphics` needs extensions; `\input` doesn't.** Mixing these breaks Overleaf sync silently.

More at [Known Quirks](../methods/known-quirks.md).

## Related

- [Project Architecture](../tools/architecture.md)
- [Experiment Design](../concepts/experiment-design.md)
- [Data Structure](../concepts/data-structure.md)
- [Derived Datasets Pipeline](../tools/derived-datasets.md)
- [Market Data Parser](../tools/market-data-parser.md)
- [iMotions Integration](../concepts/imotions-integration.md)
- [Known Quirks](../methods/known-quirks.md)
- [Overleaf Sync](overleaf-sync.md)
- [Main Paper Results](../papers/main-paper-results.md)

## Sources

- `CLAUDE.md` (project root)
- `~/.claude/CLAUDE.md` and `~/.claude/rules/*.md` (global conventions)
- `git log` and merged PR history on `calebeynon/marketruns`
- Existing wiki articles listed above
