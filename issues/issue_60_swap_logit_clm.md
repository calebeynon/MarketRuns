# Issue #60: Swap Ranked-Order Logit and CLM Results

## Summary

Move the ranked-order logit regression table from Appendix F into the main results section, and move the CLM (Cumulative Link Model) table from the main results section into Appendix F as a robustness check.

## Motivation

The ranked-order logit is the preferred primary specification for modeling selling position within group-rounds. The CLM was previously placed in the main text but is better suited as a robustness check in the appendix.

## Approach

- Swap `\input{}`, `\label{}`, and `\caption{}` between the main results table block and the Appendix F table block in `analysis/paper/main.tex`
- Update the Appendix F heading from "Ranked-Order Logit Robustness Check" to "CLM Robustness Check"
- No analysis scripts or output table files need modification (filenames unchanged)

## Additional Changes

- Pulled latest Overleaf edits (market model section, experimental implementation, treatment details, liquidation example)
- Fixed `interface.jpeg` image path to use bare filename per `\graphicspath` convention
- Added paper compilation and Overleaf pull instructions to `CLAUDE.md`

## Files Modified

- `analysis/paper/main.tex` — table swap + Overleaf content merge + image path fix
- `analysis/output/plots/interface.jpeg` — added (referenced by new Overleaf content)
- `CLAUDE.md` — added paper compilation and Overleaf pull instructions

## Testing

- Paper compiles without errors via full `pdflatex → bibtex → pdflatex → pdflatex` sequence
- Ranked-order logit table appears in main results section
- CLM table appears in Appendix F
- All citations and cross-references resolve (no question marks)
