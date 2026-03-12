# Issue 100: DiD Analysis Separating Learning from Communication Effects

## Problem

Segments 1-2 have no chat; Segments 3-4 introduce chat. Observed increases in selling
behavior across segments could reflect learning (repeated exposure to the market) rather
than the communication channel itself. Without disentangling these channels, we cannot
attribute changes in `n_sellers` to chat-driven coordination versus pure experience
accumulation.

## Solution

Implement a Difference-in-Differences (DiD) framework at the segment level:
- `learning_trend` is a continuous segment counter (1–4) capturing experience
- `chat` is an indicator for Segments 3-4 (when communication is enabled)
- OLS with session and round fixed effects isolates the chat effect conditional on trend
- Cluster at `global_group_id` (96 segment-group clusters) because groups are reshuffled
  between segments, making session_group FE invalid
- Wald tests on a segment-dummy tobit validate parallel trends (p=0.76) and test the
  communication effect (p=0.44)
- An event-study plot of segment-dummy coefficients visualizes the trajectory

## Key Finding

The chat coefficient is negative but marginally significant (p=0.087) in the OLS model
with session+round FE and treatment×chat interaction. Point estimates are consistent with
a communication effect, but statistical power is insufficient to cleanly separate learning
from communication given only four segments.

## Files Changed

### New R scripts (2)
- `analysis/analysis/did_learning_communication.R` → `analysis/output/tables/did_learning_communication.tex`
- `analysis/analysis/visualize_did_segment_effects.R` → `analysis/output/plots/did_segment_effects.pdf`

### Generated output (2)
- `analysis/output/tables/did_learning_communication.tex` — single-column LaTeX table (OLS session+round FE + treatment×chat)
- `analysis/output/plots/did_segment_effects.pdf` — event-study coefficient plot with segment-transition annotations

### Modified
- `analysis/paper/main.tex` — added new subsubsection with the DiD table and event-study figure
