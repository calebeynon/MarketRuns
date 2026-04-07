---
title: "Market Data Parser"
type: tool
tags: [data-parsing, market-data, python, core-module]
summary: "Core Python module (analysis/market_data.py) that parses raw oTree CSV exports into a hierarchical object structure"
status: draft
last_verified: "2026-04-06"
---

## Summary

`analysis/market_data.py` is the foundational data module. It reads raw oTree CSV exports and constructs a hierarchical object graph that all downstream derived dataset scripts depend on.

## Key Points

- Entry point: `parse_experiment(csv_path, chat_path)` returns a `MarketRunsExperiment`
- Also exports data as flat DataFrames via `experiment.as_dataframe(level='period'|'round')`
- Chat data is parsed separately via `parse_chat_data()` and attached to the experiment
- Custom `ParsingError` exception for data issues

## Class Hierarchy

```
MarketRunsExperiment
├── Session (session_code, participant_labels, metadata)
│   └── Segment (name: chat_noavg, chat_noavg2, etc.)
│       ├── Group (group_id, player_labels — constant within segment)
│       └── Round (round_number_in_segment: 1-14)
│           ├── Period (period_in_round)
│           │   └── PlayerPeriodData (label, sold, signal, price, state, payoff, sell_click_time)
│           └── ChatMessage (nickname, body, timestamp, participant_code)
```

## Important Data Concepts

- **oTree round vs. custom round**: oTree's `round_number` counts total periods. The parser maps these to `round_number_in_segment` (1-14) and `period_in_round` using `PERIODS_PER_ROUND`
- **sold vs. sold_this_period**: `sold` is cumulative within a round (0 or 1); `sold_this_period` is True only in the period the player actually clicked sell
- **Round payoffs**: extracted from `round_N_payoff` fields on the last oTree period of each round
- **Groups**: detected from `group.id_in_subsession` columns; remain fixed within a segment
- **Chat channels**: pattern `1-{segment}-{channel_number}`, mapped to groups by matching player nicknames

## Usage

```python
from analysis.market_data import parse_experiment

exp = parse_experiment('datastore/raw/1_tr_data.csv', 'datastore/raw/1_tr_chat.csv')
session = exp.sessions[0]
segment = session.get_segment('chat_noavg')
round_obj = segment.get_round(1)
period = round_obj.get_period(1)
player = period.get_player('A')
```

## Related

- [Project Architecture](wiki/tools/architecture.md)
- [Derived Datasets Pipeline](wiki/tools/derived-datasets.md)
