---
title: "Recent Activity Snapshot (Apr 2026)"
type: idea
tags: [roadmap, recent, status, snapshot]
summary: "Snapshot of in-flight work and recently merged direction as of April 2026 — welfare, equilibrium simulation, pre-sell emotions, DiD"
status: active
last_verified: "2026-04-19"
---

## Summary

This is a point-in-time snapshot of where the project is. Recent merges (Jan–Apr 2026) have focused on welfare analysis, equilibrium simulation replication, the pre-sell emotions pipeline, and DiD decomposition of the chat-treatment effect. The thesis template was added in early April.

> Snapshot: 2026-04-19. For *current* state always run `git log` and `gh pr list` — do not rely on this article for live status.

## Recently Completed

| PR | Title | Theme |
|----|-------|-------|
| #115 | Implement M&M (2020) Appendix D equilibrium simulation (#109) | Equilibrium |
| #113 | Add pre-sell emotions dataset builder (#112) | iMotions / emotions |
| #114 | Add welfare regression on individual traits (#111) | Welfare |
| #110 | Add group-round welfare computation and merge (#108) | Welfare |
| #107 | Add theoretical welfare plot and paper section (#106) | Welfare |
| #105 | DiD analysis separating learning from communication effects (#100) | DiD / chat |
| #103 | Add facial emotion correlation table to paper (#101) | Emotions |
| #99 | Valence-only and no-valence alternate specs for Tables 6 & 7 | Robustness |
| #98 | Add experiment flow diagram to paper (#96) | Paper |

## Open Issues (as of snapshot)

- **#104** — Overleaf has unmerged edits — sync blocked (recurring)
- **#87** — Analyze chat sentiment and its effect on selling behavior

## Active Workstreams (inferred from recent direction)

- **Welfare analysis**: theoretical plot → group-round computation → individual-trait regression. Restricted to good-state rounds (state=1).
- **Equilibrium simulation**: M&M (2020) Appendix D — CRRA grid × treatment-specific payoffs, cross-validated Python vs. R.
- **Pre-sell emotions pipeline**: timestamp-aligned multi-window emotion features (50/100/500/1000/2000 ms) for sell-click events.
- **Chat sentiment** (open #87): NLP analysis of chat-treatment messages.
- **Thesis preparation**: UA thesis template added (commits `ae4d828`, `d84a90d`).

## Things That Have Stabilized

- Table layouts (Tables 4, 6, 7) — multiple restructures complete, current layouts in production.
- Cox survival robustness (Appendix B.2).
- Rank-ordered logit + CLM specifications.
- iMotions period-emotion aggregation pipeline.

## Things to Re-Verify When Returning

- Run `git log --oneline -20` and `gh pr list` at session start
- Check #104 sync state if touching paper
- Verify `datastore/derived/` still contains the file your script consumes — Box-symlinked, may be stale on a fresh machine

## Related

- [Main Paper Results](../papers/main-paper-results.md)
- [Known Quirks](../methods/known-quirks.md)
- [Project Architecture](../tools/architecture.md)
