# Issue 109: Implement Equilibrium Simulation with CRRA Grid and Treatment-Specific Payoffs

## Summary

Implement the Magnani & Munro (2020) Appendix D equilibrium simulation for the
market runs experiment. Solves for symmetric mixed-strategy equilibria by
discretizing the belief space (π) and iterating over Bellman equations
(Eqs. 3–4 from the paper).

## Motivation

The paper needs a quantitative theoretical benchmark — equilibrium selling
thresholds σ(n, π) — to compare against observed experimental behavior across
treatments and seller positions.

## Key Extensions Beyond M&M (2020)

1. Sweep over a grid of CRRA parameters α ∈ {0, 0.1, …, 0.9} instead of only
   the single α = 0.5 used in the paper.
2. Compute equilibria separately for the **Random** and **Average** treatments,
   which differ only in the simultaneous-seller payoff rule:
   - **Random**: expected utility = avg of u(price) across positions
   - **Average**: expected utility = u(avg price) — risk pooling via Jensen's
3. Under risk neutrality (α = 0) both treatments yield identical predictions;
   divergence is mediated entirely through risk aversion.

## Acceptance Criteria

- [ ] Belief grid generator: produces the discrete π lattice from Bayesian
      updating with μ_B = 0.675
- [ ] Equilibrium solver: finds σ(n, π) for n ∈ {1..4} satisfying conditions
      1–4 from the paper
- [ ] Parameterized by CRRA coefficient α and treatment type (random vs average)
- [ ] ρ function correctly implements random vs average payoff logic
- [ ] Simulation engine: runs 10,000 games per (α, treatment) pair, reports avg
      π at first sale
- [ ] Output table of equilibrium thresholds across α × treatment grid
- [ ] Results for α = 0 (risk neutral) match between treatments (validation)
- [ ] Results for α = 0.5, random treatment reproduce M&M Table 2 values
      (π ≈ 0.678 for HIGH NoCB)
- [ ] Unit tests for belief updating, payoff functions, and equilibrium conditions

## Deliverables

- `analysis/analysis/equilibrium_model.py` — Bellman/value-iteration solver
  with CRRA utility and derived sigma strategies
- `analysis/analysis/simulate_equilibrium.py` — per-seller equilibrium simulation
- `analysis/analysis/tabulate_equilibrium.py` — threshold tabulation to LaTeX
- `analysis/analysis/_munro_style_solver.R` — Munro-style R cross-validation
  solver deriving sigma via `uniroot()` on a dense 41-point reachable-belief grid
- `analysis/analysis/compare_our_prices_replication.py` — Python vs R
  cross-validation script
- `analysis/tests/test_compare_our_prices_replication.py` — pytest suite
- `analysis/analysis/export_full_sigmas.py` — full alpha-sweep sigma export
  across Random/Average treatments
- `analysis/output/bellman_equations.tex/.pdf` — LaTeX math documentation
- `analysis/output/plots/equilibrium_avg_pi_at_sale.pdf` — avg π at sale plot
- `analysis/output/plots/equilibrium_thresholds.pdf` — threshold plot
- `analysis/output/tables/equilibrium_thresholds.tex` — thresholds table
- Paper updates in `analysis/paper/main.tex` and `refs.bib`

## Cross-Validation Result

Python vs R solvers agree on sigma to machine precision (|dσ| max ≈ 5e-7) at
α = 0.5 with our native prices.
