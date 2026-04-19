---
title: "Known Quirks, Past Bugs, and Methodology Decisions"
type: method
tags: [gotchas, bugs, methodology, data-quirks, conventions]
summary: "Non-obvious data quirks, past bug fixes, and methodology decisions distilled from closed issues and merged PRs"
status: active
last_verified: "2026-04-19"
---

## Summary

A reference of pitfalls and decisions that recur in this codebase. Read this *before* writing new derived datasets, regressions, or modifying paper tables — many of these have caused regressions.

## Data Exclusions

- **Pilot data is excluded**: `datastore/11-5-tr1/` and `datastore/oct 10 pilot/` are pilot sessions and must be skipped by all production scripts. Numbered session folders (`1_…` through `6_…`) are the analysis sample.
- **Welfare regressions: state=1 only (#108)**: Welfare is identically 1.0 for all `state==0` rows. Only good-state (z=1) rounds have variation; restrict regressions accordingly.
- **First-seller subset helper (#70, #78)**: Use `subset_first_sellers()` in `analysis/analysis/unified_selling_logit_panel_c.R`. Don't rebuild the filter inline.

## Bug-Class Quirks (read carefully before touching the related code)

### Period offset in iMotions annotations (#25)
Annotation `m{N}` maps to oTree period `N - 1`. Any new join script must apply this offset. Issues #18 and #19 had to be re-audited after this was discovered.

### `round_payoffs` extraction (#12)
`market_data.py` previously read `player.round_X_payoff` from the *first* period of the segment, returning intermediate values. Final values are only valid at the **last period of each round**. Period 1 had `round_1_payoff=6.0` but the correct value (period 3) was `4.0`; period 4+ resets to 0.0.

### Cox `Surv()` specification (PR #78)
`Surv(period, sold)` (interval form) is **wrong** — it treats every row as right-censored from t=0. Correct form is the counting-process specification:
```r
Surv(period_start, period, sold)   # period_start := period - 1L
```
This changed all hazard ratio estimates in Table 7.

### Treatment × FE collinearity (#68)
`treatment` is session-level constant → perfectly collinear with `player_id` fixed effects. The original `feglm(... | player_id)` couldn't include it. Switched to `glmer(... + (1|player_id))` so the treatment dummy could be estimated (PR #71). Trade-off accepted vs. consistency concern in #31.

### Phantom selling ranks (PR #88)
`build_ordinal_selling_position.py` does an *inner* trait merge — players missing survey data get dropped. If `merge_traits` runs *after* `compute_selling_ranks`, group-rounds end up missing rank-1. Fix: merge traits **first**, then compute ranks. A `validate_no_phantom_ranks` assertion guards against regression.

### Pre-sell timestamp conversion (#112, PR #113)
oTree `sell_click_time` = Unix epoch seconds. iMotions timestamps = relative ms from per-participant recording start. Converting requires per-participant start time from edited_data annotations. See `build_presell_emotions_traits.py`.

### Period as a control variable (#26)
Several individual-period regressions were missing `period`, which matters because price decreases over periods. Always include `period` (or period FE) in player-period specs.

### Duplicate bib entry blocked bibtex (PR #115)
`gneezy1997experiment` was duplicated in `refs.bib`, silently breaking compilation. Watch for duplicates after Overleaf pulls.

## Methodology Decisions

- **CLM over CLMM for sellers-only ordinal logit (#56, PR #58)**: `marginaleffects` doesn't support `clmm` objects. Both columns now use plain CLM with a `treatment_2` dummy.
- **DiD clustering at `global_group_id` (PR #105)**: 96 segment-group clusters. Groups reshuffle between segments, so session-level clustering understates SEs and `session_group` FE is invalid (links different participants across segments). Earlier multi-column variants with group FE were removed.
- **Treatment-invariance of welfare (#108)**: T2 redistributes among sellers but doesn't change group totals; same closed-form formula applies in both treatments.
- **Equilibrium replication (#109)**: T1 vs T2 collapse at α=0 (validation check); divergence comes only through risk aversion via Jensen's inequality. Python solver cross-validated against R `uniroot()` on a 41-point belief grid to ~5e-7 precision.
- **`risk_tolerance` derivation**: comes from `player.allocate` (0–20 ECU lottery). Was previously named `risky_investment` — expect old names in stale branches (#48).

## Workflow Patterns

### Overleaf sync noise
At least 12 closed "Overleaf has unmerged edits — sync blocked" issues (#47, #50, #54, #59, #72, #73, #79, #89, #90, #91, #92, #93, #94, #102; **#104 currently open**). Use the configured `overleaf` git remote (`git fetch overleaf`, diff `overleaf/master:<file>`) — **don't** clone with a token. See [Overleaf Sync](../skills/overleaf-sync.md) and `feedback_overleaf_remote.md`.

### Bare filenames in `.tex`
Overleaf editors may add `plots/` or `tables/` prefixes — these break locally because `\graphicspath` and `\input@path` already include them. PRs #64 and #98 had to fix these. Always use bare filenames.

### Table restructuring is recurring
Tables 4, 6, and 7 each went through multiple column restructures (#82/#85, #80/#81, #83/#88, #84/#86) toward "no-traits / with-traits" or "all-emotions / valence-only" 4-column layouts. Changes always cascade: R script → regenerated `.tex` → `main.tex` → tests.

### Generated `.tex` files are committed artifacts
Files in `analysis/output/tables/` are committed but never edit by hand — they get regenerated by R scripts.

### `Closes #N` is mandatory in PR bodies
Per `feedback_pr_template.md`. PR #114 merged without it and left issue #111 open.

## Cross-Cutting Themes

- **Tightly-coupled stack**: most PRs touch the chain `derived script → derived dataset → R analysis → .tex table → main.tex → tests`. Plan for the cascade.
- **Cross-validation against raw exports** is the standard test pattern (e.g., `test_seller_counts_raw_validation.py`, `test_cox_survival_parser.py`, the 17 integration tests added in #66). Same-source tests are insufficient.
- **Group reassignment between segments** is the source of multiple bugs. Always cluster at `global_group_id` (96 clusters), not `session_group`.
- **iMotions integration is fragile**: period offset (#25), timestamp conversion (#112), 95th-percentile aggregation (#32) have all caused issues.
- **Code standards enforcement**: PRs #78, #88, #110, #115 explicitly note splitting test files or refactoring helpers to satisfy 300-line / 20-line limits.

## Related

- [Data Structure](../concepts/data-structure.md)
- [iMotions Integration](../concepts/imotions-integration.md)
- [Overleaf Sync](../skills/overleaf-sync.md)
- [Derived Datasets Pipeline](../tools/derived-datasets.md)
