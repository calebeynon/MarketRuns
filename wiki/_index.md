# Project Wiki

<!-- AUTO:stats -->
Total articles: 11
<!-- /AUTO:stats -->

## Articles by Type
<!-- AUTO:listing -->

### concept
- [Experiment Design](concepts/experiment-design.md) — 2x2 mixed design (liquidation payoff × communication), signal mechanism, group structure, segment treatments, and payoff structure
- [Data Structure: Raw + Derived](concepts/data-structure.md) — Datastore layout, raw oTree session-folder structure, and the schemas of the major derived datasets
- [iMotions Facial-Emotion Data Integration](concepts/imotions-integration.md) — How iMotions facial-recording data joins to oTree player-period rows: period offset, timestamp conversion, aggregation windows

### tool
- [Project Architecture](tools/architecture.md) — Top-level directory layout, tech stack, and how the experiment, analysis, and paper components connect
- [Market Data Parser](tools/market-data-parser.md) — Core Python module (analysis/market_data.py) that parses raw oTree CSV exports into a hierarchical object structure
- [Derived Datasets Pipeline](tools/derived-datasets.md) — Python scripts in analysis/derived/ that transform raw parsed data into analysis-ready CSV/parquet datasets
- [Analysis Scripts (R)](tools/analysis-scripts.md) — R scripts in analysis/analysis/ for regressions (fixest) and visualization (ggplot2), producing LaTeX tables and PDF plots

### skill
- [Getting Started: New Collaborator Onboarding](skills/getting-started.md) — Day-1 orientation for a new collaborator: what the project is, how to set it up, the repo layout, data pipeline, and the commit/PR conventions
- [Overleaf Sync Workflow](skills/overleaf-sync.md) — How to pull and push paper changes between the local repo and Overleaf using the configured git remote

### method
- [Known Quirks, Past Bugs, and Methodology Decisions](methods/known-quirks.md) — Non-obvious data quirks, past bug fixes, and methodology decisions distilled from closed issues and merged PRs

### paper
- [Main Paper: To Sell or Not to Sell — Results & Structure](papers/main-paper-results.md) — Hypotheses, main results, table inventory, and section structure of the working paper (Eynon & Jindapon)

### idea
- [Recent Activity Snapshot (Apr 2026)](roadmap/recent-activity.md) — Snapshot of in-flight work and recently merged direction as of April 2026 — welfare, equilibrium simulation, pre-sell emotions, DiD

<!-- /AUTO:listing -->
