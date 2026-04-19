---
title: "Experiment Design"
type: concept
tags: [experiment, design, treatments, signals, payoffs, otree]
summary: "2x2 mixed design (liquidation payoff × communication), signal mechanism, group structure, segment treatments, and payoff structure"
status: active
last_verified: "2026-04-19"
---

## Summary

MarketRuns runs a 4-segment laboratory experiment in oTree where 16 participants per session trade in 4-person groups across multiple rounds. The design has two manipulations: liquidation payoff method (between-subjects) and pre-play communication (within-subjects).

## 2×2 Mixed Design

| Dimension | Type | Levels | Implementation |
|-----------|------|--------|----------------|
| Liquidation payoff | Between-subjects | Random (T1) vs. Average (T2) | T1 = `nonlivegame/`, T2 = `nonlivegame_tr2/` |
| Communication | Within-subjects | No-chat (segments 1–2) vs. Chat (segments 3–4) | `chat_noavg`, `chat_noavg2` vs. `chat_noavg3`, `chat_noavg4` |

## Sample

- **96 participants** total (6 sessions × 16 each)
- 48 per treatment (3 sessions per treatment)
- TIDE Lab, University of Alabama, ages 18–34
- 4 fixed groups of 4 per session, **regrouped each segment** so no two subjects ever paired twice
- Default grouping (segment 1): `[[1,5,9,13],[2,6,10,14],[3,7,11,15],[4,8,12,16]]`

## Round and Period Structure

- Each segment has an uncertain number of rounds (geometric, λ = 0.125)
- Each round is a market with an uncertain number of periods (geometric, λ = 0.125, periods last 4 sec)
- `PERIODS_PER_ROUND` (in app constants) records the realized period count per round
- Round termination = either signal hits the terminal probability or the random termination fires

## Signal Mechanism

- Binary state z ∈ {0, 1}; prior π(0) = 0.5
- Signal accuracy: **PCORRECT = μ_B = μ_G = 0.675** (HIGH info quality from Magnani 2020)
- Public binary signal each period; Bayesian posterior updated via the closed-form formula in main.tex Eq. \ref{eq:bayes} (lines 189–197)
- Posterior π converges almost surely to the true state

## Price and Payoff Structure

- Initial price `p_4 = 8`, falls $2 per seller: `p_n = 8 - 2(4 - n)` where n = sellers remaining
- Good-state final value `v = 20`
- **T2 (average)**: simultaneous sellers receive average price `p̄_{N,M}` instead of position-specific prices
- ECU conversion: 4 ECU = $1
- Earnings = $7.50 participation + 1 random round per segment (4 rounds total) summed for Part 1 + Part 2 (20 ECU + Gneezy–Potters risk task at 2.5×)
- Empirical earnings range: $11.00 – $36.25, mean **$23.57**

## Personality and Emotion Measurement

- **Big Five**: TIPI (Gosling 2003)
- **Impulsivity**: Barratt-style items
- **Anxiety**: STAI-style items (state anxiety)
- **Risk tolerance**: Gneezy–Potters lottery elicitation (`player.allocate`, 0–20 ECU)
- **Emotions**: Affectiva facial-reading via iMotions, 8 discrete emotions (anger, contempt, disgust, fear, joy, sadness, surprise, engagement) plus composite valence

## Session Flow

```
quiz → chat_noavg → chat_noavg2 → chat_noavg3 → chat_noavg4 → survey → final
```

## Key Constants

| Constant | Value | Where |
|----------|-------|-------|
| `PCORRECT` | 0.675 | App constants |
| Initial price | 8 ECU | Market mechanics |
| Price drop per seller | 2 ECU | `p_n = 8 - 2(4-n)` |
| Good-state value | 20 ECU | Terminal payoff |
| Round termination λ | 0.125 | Geometric |
| Period termination λ | 0.125 | Geometric |
| Period length | 4 sec | Real time |
| Group size | 4 | Fixed within segment |
| ECU rate | 4 ECU = $1 | Conversion |

## Related

- [Project Architecture](../tools/architecture.md)
- [Main Paper Results](../papers/main-paper-results.md)
- [Data Structure](data-structure.md)
- [iMotions Integration](imotions-integration.md)
