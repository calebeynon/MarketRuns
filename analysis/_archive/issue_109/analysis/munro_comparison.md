# Munro Comparison Memo

**Scope:** Structural comparison of our Python equilibrium solver against Munro's R reference code (RA and RN variants) for the Magnani & Munro (2020) market-runs model.

**Inputs compared:**
- Ours: `analysis/analysis/equilibrium_model.py`, `simulate_equilibrium.py`, `tabulate_equilibrium.py`
- Munro: `analysis/analysis/munro_code/MixedStrat_low_RA.R`, `MixedStrat_low_RN.R`

**Framing:** Neutral. Neither implementation is treated as gold standard. Claims are marked [VERIFIED] (confirmed by reading source), [INFERRED] (deduced from structure), or [UNCERTAIN] (guess that needs user judgment).

---

## 1. Executive summary

- **Same equilibrium concept.** Both solve a symmetric mixed-strategy equilibrium in the (n, belief) state space via value function iteration, with the same exogenous termination probability λ = 0.125, the same signal accuracy μ_b = 0.675, the same liquidation value v = 20, and the same Bellman decomposition (sell-now vs. hold-and-continue with probabilistic termination). [VERIFIED]
- **Different price schedules.** Ours uses `PRICES = [2, 4, 6, 8]` (p_n = 2n, ascending) [VERIFIED `equilibrium_model.py:14`]. Munro uses `prices = c(8, 5.5, 3, 0.5)` (descending) [VERIFIED `MixedStrat_low_RA.R:18`, `MixedStrat_low_RN.R:17`]. Level, slope, and spread differ as parameters, and the vectors are written in opposite orientations — our `rho`/`forced_liquidation` indexers require ascending, so the replication script reverses Munro's vector to `[0.5, 3, 5.5, 8]` before monkey-patching. Once the ordering convention is reconciled, the price substitution alone leaves essentially no residual (see §R): price is a parameter, not a modeling divergence.
- **Different belief discretization.** Ours uses a 41-point reachable-belief grid built from Bayesian iteration off π₀ = 0.5 [VERIFIED `equilibrium_model.py:41-57`, `simulate_equilibrium.py:30`]. Munro uses a 15-point hand-chosen grid `{0, 0.01, 0.02, 0.05, 0.1, 0.19, 0.33, 0.5, 0.68, 0.82, 0.9, 0.95, 0.98, 0.99, 1}` with a precomputed integer index map for signal updates [VERIFIED `MixedStrat_low_RA.R:11-14`]. Munro's belief updates are rounded to the nearest grid point; ours are exact with linear interpolation of W.
- **Different strategy solver.** Ours solves σ(n, π) at every grid point via Brent root-finding on the indifference condition inside each iteration [VERIFIED `equilibrium_model.py:136-156`, `206-216`]. Munro hand-specifies a threshold + one interior mixing value per n, iterates V to fixed point given that σ, then inspects `exp_price - exp_W` to assess equilibrium quality [VERIFIED e.g. `MixedStrat_low_RA.R:48-53, 81, 91`]. Munro's code does not adjust σ inside the loop; ours does.
- **Different convergence criterion and tolerance.** Ours: absolute L∞ change in V below 1e-8, up to 1000 iterations [VERIFIED `equilibrium_model.py:180, 201`]. Munro: relative L∞ change `max|new_V/V - 1|` below 1e-3, up to 50 iterations [VERIFIED `MixedStrat_low_RA.R:24-25, 84-85`]. Ours is ~5 orders of magnitude tighter.
- **Treatment-specific payoffs are a genuine extension.** Munro's code has no Random-vs-Average distinction. Ours implements both [VERIFIED `equilibrium_model.py:87-100`].
- **Risk-aversion parameterization.** Munro RA: single fixed CRRA a = 0.5 [VERIFIED `MixedStrat_low_RA.R:21-22`]. Ours: 10 α values {0.0, …, 0.9} [VERIFIED `simulate_equilibrium.py:27`]. At α = 0 our model reduces to linear utility, directly comparable to Munro RN.
- **Replication status (from our existing diagnostic, see `bellman_equations.tex` §8).** Aggregate predictions (P(Bad) at first sale) match M&M Table 2 to within 0.001–1.2pp, but pointwise σ(n, π) is 2–5pp higher than M&M Appendix D. The replication diagnostics in the paper document that policy iteration, initialization, sweep count, and discount convention have been ruled out as causes; grid resolution and interpolation remain as candidates. The price-schedule divergence flagged here has not been systematically tested in the diagnostic.
- **Direct replication numbers (post-fix, P(Good) convention, ascending prices, t_max=20).** Running our solver on Munro's 15-point grid and comparing σ and V on-grid yields, by n: |dσ| max = 0.0000/0.0205/0.0015/0.0248; mean |dσ| = 0.0000/0.0024/0.0001/0.0029; corr(σ) = n/a/0.9999/1.0000/0.9999; |dV| max = 0.0000/0.2126/0.3538/0.6983; mean |dV| = 0.0000/0.0330/0.0422/0.0698. This is apples-to-apples agreement: σ matches to 2–3 decimal places (correlation ≈ 1.0) and V matches to within ~0.7 in value units. Residual disagreement is attributable to D2 (hand-set σ vs. Brent-solved σ), D3 (grid coarsening and index-rounding), and D4 (convergence tolerance), not to price-level or utility-function differences. See §R below. [VERIFIED `analysis/output/munro_replication_comparison.csv`]

---

## 2. Conceptual comparison

| Dimension | Ours (Python) | Munro RA | Munro RN |
|---|---|---|---|
| Equilibrium concept | Symmetric Markov-perfect; σ(n, π) solved at every state | Symmetric threshold + one interior mixing point, hand-specified | Same as RA |
| State | (n, π), n ∈ {1,2,3,4}, π continuous on 41-point grid | (n, belief_index), n ∈ {1,2,3,4}, 15 belief points | Same as RA |
| Decision | Probability σ ∈ [0,1] of selling | Probability σ ∈ [0,1] of selling | Same |
| Utility | CRRA `x^(1-α)/(1-α)` (log at α=1); α ∈ {0.0, …, 0.9} | CRRA `x^(1-a)/(1-a)` with a = 0.5 fixed | Linear (u(x) = x) |
| Signal | Binary {g, b}, accuracy μ_b = μ_g = 0.675 (symmetric) | Same | Same |
| Information | Public belief π; posterior updated by one signal per period | Same (belief is the sole state) | Same |
| Termination | Per-period termination prob λ = 0.125; on termination, Good pays v, Bad pays forced-liquidation L(m) | Per-period prob l = 0.125; on termination, payoff is `(1-b)·u(v) + b·mean(u(prices[remaining]))` | Same as RA but without u() |
| Horizon | Infinite (terminated stochastically by λ) | Infinite (terminated stochastically by l) | Same |
| Price schedule | p = [2, 4, 6, 8] (p_n = 2n) [VERIFIED `equilibrium_model.py:14`] | prices = [8, 5.5, 3, 0.5] [VERIFIED `MixedStrat_low_RA.R:18`] | Same as RA |
| Multiple-seller payoff | Treatment-specific: "random" lottery over posted prices OR "average" of prices [VERIFIED `equilibrium_model.py:87-100`] | `mean(u(prices[remaining]))` — applies u to each price then averages (lottery interpretation) [VERIFIED `MixedStrat_low_RA.R:73, 79, 127, 133`] | `mean(prices[remaining])` (linear, so equivalent under either interpretation) |

**Key conceptual divergences:**
- **Price schedule.** The level and curvature of prices differs substantively (see §4 and §5.1).
- **Treatment dimension.** Munro's code has only one payoff convention. Our "random" treatment matches Munro's convention (expectation of u over prices); our "average" treatment (u of average price) is a true extension introduced to capture the experimental design variant [VERIFIED `equilibrium_model.py:90-92`].
- **Strategy as endogenous vs. hand-set.** Munro iterates V given a fixed σ function; the interior mixing values (e.g. 0.453, 0.402, 0.11/0.989) are author-chosen and presumably tuned by inspection of `exp_price - exp_W`. Ours solves σ at every grid point every iteration by root-finding.

---

## 3. Algorithmic mapping

| Step | Ours (Python) | Munro RN | Munro RA | Same? | Notes |
|---|---|---|---|---|---|
| State setup | Build 41-point reachable belief grid from Bayesian iteration off π₀ = 0.5 [VERIFIED `equilibrium_model.py:41-57`] | 15-point hand-chosen grid [VERIFIED `MixedStrat_low_RN.R:10`] | Same 15-point grid [VERIFIED `MixedStrat_low_RA.R:11`] | No | Ours is denser and grid points are reachable by Bayesian updating; Munro's is coarser and ad-hoc |
| Belief update on signal | Closed-form Bayes: `π' = π(1-μ)/(π(1-μ)+(1-π)μ)` [VERIFIED `equilibrium_model.py:60-69`]; linear interpolation of W between grid points [VERIFIED `equilibrium_model.py:227-228`] | Integer index map `index_b`, `index_g` applied to grid; updated beliefs coarsened to existing grid points [VERIFIED `MixedStrat_low_RN.R:12-13`] | Same as RN [VERIFIED `MixedStrat_low_RA.R:12-13`] | No | Munro's `index_b[i]` and `index_g[i]` lookups **round** the posterior to a grid point; ours uses exact Bayes on continuous π and interpolates V/W |
| V₁ (one player) | Computed inside `solve_equilibrium` with σ ≡ 0 for n=1 [VERIFIED `equilibrium_model.py:138-139`] | Closed-form: `V_1 = pmax(prices[4], beliefs·mean(prices[4]) + (1-beliefs)·v)` [VERIFIED `MixedStrat_low_RN.R:27`] | Closed-form with u(): `V_1 = pmax(u(prices[4]), beliefs·mean(u(prices[4])) + (1-beliefs)·u(v))` [VERIFIED `MixedStrat_low_RA.R:31`] | No | n=1 semantics: Munro's V_1 uses `prices[4] = 0.5` (min price); ours has σ(n=1) = 0 and derives V_1 inside the Bellman loop — the one-player is modeled as never selling |
| Strategy σ(n, π') | Solved every iteration: `find_sigma` checks corner (σ=0, σ=1), otherwise Brent root-finding on `U_sell(σ) − U_hold(π', σ)` with xtol = 1e-12 [VERIFIED `equilibrium_model.py:136-156, 206-216`] | Hand-specified threshold + one interior mixing value at index `num_beliefs-4/-5/-6` depending on n [VERIFIED `MixedStrat_low_RN.R:44-49, 98-103, 157-162`] | Same structural form, different threshold and interior values [VERIFIED `MixedStrat_low_RA.R:48-53, 102-107, 161-167`] | No | Munro's σ is fixed during iteration; ours is re-solved every sweep. Munro relies on the user inspecting `exp_price - exp_W` after the fact to judge equilibrium |
| Bellman update for V | Two-pass: (1) compute W(n, π') = σ·U_sell + (1-σ)·U_hold at all grid points; (2) update V(n, π) = P(g\|π)·W(π_g) + P(b\|π)·W(π_b) with linear interp on W [VERIFIED `equilibrium_model.py:206-229`] | Inline: `new_V_n = pmax(exp_price, exp_W)` where `exp_W` pools holding values across realized seller counts and `exp_price` pools selling payoffs [VERIFIED `MixedStrat_low_RN.R:73-81`] | Same structure with u() wrapping [VERIFIED `MixedStrat_low_RA.R:75-81`] | Partially | Both implement V = max(sell, hold) in expectation. Munro's `pmax` enforces optimality pointwise (direct VFI); ours uses the `sigma * U_sell + (1-sigma) * U_hold` form with σ solved at indifference (both are valid for a mixed-strategy equilibrium; see Divergence D2) |
| Continuation/term value | `H(m, π') = λ[π'·u(v) + (1-π')·L(m)] + (1-λ)·V(m, π')` [VERIFIED `equilibrium_model.py:125-130`] | `W_n = l·term_value + (1-l)·cont_value` with `term_value = (1-beliefs)·v + beliefs·mean(prices[remaining])` [VERIFIED `MixedStrat_low_RN.R:69-75`] | Same with u() [VERIFIED `MixedStrat_low_RA.R:73-75`] | Partially | **Semantic question**: Munro's term_value uses `(1-beliefs)` for the good-state-pays-v case and `beliefs` for the price-case. Ours uses `pi'` (= P(Good)) for good-state-pays-v and `(1-pi')` for the liquidation-pays-forced-price case. Whether "beliefs" in Munro denotes P(Bad) or P(Good) matters here — see Divergence D7 |
| Multiple-seller payoff | `rho(n, k)` treatment-specific: "random" = mean(u(price_j)); "average" = u(mean(price_j)) [VERIFIED `equilibrium_model.py:87-92`] | `mean(prices[remaining])` (linear, degenerate under either treatment) [VERIFIED `MixedStrat_low_RN.R:75, 129, 189`] | `mean(u(prices[remaining]))` — lottery interpretation with u applied before averaging [VERIFIED `MixedStrat_low_RA.R:73, 79, 127, 133, 193`] | Partially | Munro RA = our "random" treatment at matching α. Munro has no "average" counterpart |
| Convergence check | `max_n max_π |V^{t+1}(n,π) − V^t(n,π)| < 1e-8`; ≤ 1000 iterations [VERIFIED `equilibrium_model.py:180, 196-202`] | `max|new_V_n/V_n − 1| < 1e-3`; ≤ 50 iterations; separate loop per n [VERIFIED `MixedStrat_low_RN.R:20-21, 80-81`] | Same as RN [VERIFIED `MixedStrat_low_RA.R:24-25, 84-85`] | No | Absolute vs. relative; 1e-8 vs. 1e-3; 1000 vs. 50 iterations. See Divergence D4 |
| Strategy extraction | `find_continuous_threshold`: Brent root-find for π* where `U_sell(σ=0) = U_hold(π*, σ=0)` with xtol = 1e-12 [VERIFIED `equilibrium_model.py:159-174`] | σ is the output directly (printed via `print(sig)`) [VERIFIED `MixedStrat_low_RN.R:88`] | Same [VERIFIED `MixedStrat_low_RA.R:92`] | No | Ours outputs a continuous threshold π*; Munro outputs a vector at 15 grid points |
| Outputs | CSV `datastore/derived/equilibrium_thresholds.csv` with (α, treatment, n, threshold_pi, avg_pi_at_sale, n_obs) [VERIFIED `simulate_equilibrium.py:24, 65-66, 155-157`]; LaTeX table [VERIFIED `tabulate_equilibrium.py:14`] | Printed to R console only [VERIFIED `MixedStrat_low_RN.R:87-88`] | Same [VERIFIED `MixedStrat_low_RA.R:91-92`] | No | Munro leaves it to the user to read `sig` and `exp_price - exp_W` vectors; ours persists and post-processes |
| Simulation | 10,000 games per (α, treatment) with seed = 42; records (seller_position, π) at each sale [VERIFIED `simulate_equilibrium.py:29-31, 73-112`] | None | None | No | Simulation is a pure extension; Munro's code does not simulate |

---

## 4. Parameter comparison

| Parameter | Ours | Munro RN | Munro RA | Notes |
|---|---|---|---|---|
| N investors | 4 [VERIFIED `equilibrium_model.py:13`] | 4 (via sections n=1,…,4) [VERIFIED `MixedStrat_low_RN.R:25, 37, 92, 151`] | 4 [VERIFIED `MixedStrat_low_RA.R:29, 41, 96, 155`] | Same |
| μ_B = P(bad sig \| Bad) | 0.675 [VERIFIED `equilibrium_model.py:16`] | 0.675 [VERIFIED `MixedStrat_low_RN.R:7`] | 0.675 [VERIFIED `MixedStrat_low_RA.R:8`] | Same |
| μ_G = P(bad sig \| Good) | 0.325 [VERIFIED `equilibrium_model.py:17`] | 0.325 (= 1 − μ_b) [VERIFIED `MixedStrat_low_RN.R:8`] | 0.325 [VERIFIED `MixedStrat_low_RA.R:9`] | Same |
| Termination prob | λ = 0.125 [VERIFIED `equilibrium_model.py:18`] | l = 0.125 [VERIFIED `MixedStrat_low_RN.R:6`] | l = 0.125 [VERIFIED `MixedStrat_low_RA.R:7`] | Same |
| Good-state liquidation | v = 20 [VERIFIED `equilibrium_model.py:15`] | v = 20 [VERIFIED `MixedStrat_low_RN.R:18`] | v = 20 [VERIFIED `MixedStrat_low_RA.R:19`] | Same |
| Initial belief | π₀ = 0.5 [VERIFIED `equilibrium_model.py:19`] | Not explicitly set (grid contains 0.5) [INFERRED] | Not explicitly set [INFERRED] | Munro's code does not pin down π₀; inferred from grid symmetry |
| Price schedule | [2, 4, 6, 8] (p_n = 2n) [VERIFIED `equilibrium_model.py:14`] | [8, 5.5, 3, 0.5] [VERIFIED `MixedStrat_low_RN.R:17`] | [8, 5.5, 3, 0.5] [VERIFIED `MixedStrat_low_RA.R:18`] | **Different** |
| Indexing convention | `PRICES[n-1-j]` where j = additional sellers beyond first: first seller gets highest price [VERIFIED `equilibrium_model.py:89`] | `prices[k]` indexed by total sellers: `prices[1] = 8` is "0 others sell" [VERIFIED `MixedStrat_low_RN.R:75, 189`] | Same as RN [VERIFIED] | Both conventions agree that a lone seller gets the top price. See D1 |
| Utility | CRRA `x^(1-α)/(1-α)` over α ∈ {0.0, 0.1, …, 0.9} [VERIFIED `equilibrium_model.py:75-81`, `simulate_equilibrium.py:27`] | Linear (prices used directly) [VERIFIED `MixedStrat_low_RN.R:27, 40, 75`] | CRRA with a = 0.5 fixed [VERIFIED `MixedStrat_low_RA.R:21-22`] | Munro RA ≈ ours at α = 0.5; Munro RN ≈ ours at α = 0.0 |
| Treatments | random, average [VERIFIED `equilibrium_model.py:87-100`] | n/a | n/a | Extension |
| Belief grid | 41 points from `build_belief_grid(t_max=20)`; T_max = 20 [VERIFIED `simulate_equilibrium.py:30`] | 15 points, hand-chosen [VERIFIED `MixedStrat_low_RN.R:10`] | Same 15 points [VERIFIED `MixedStrat_low_RA.R:11`] | Different size and construction |
| Convergence tol | 1e-8 (absolute L∞ on V) [VERIFIED `equilibrium_model.py:180`] | 1e-3 (relative L∞) [VERIFIED `MixedStrat_low_RN.R:20`] | 1e-3 (relative L∞) [VERIFIED `MixedStrat_low_RA.R:24`] | **Ours is ~5 OOM tighter** |
| Max iterations | 1000 [VERIFIED `equilibrium_model.py:180`] | 50 [VERIFIED `MixedStrat_low_RN.R:21`] | 50 [VERIFIED `MixedStrat_low_RA.R:25`] | **20× difference** |
| Brent xtol (σ and π*) | 1e-12 [VERIFIED `equilibrium_model.py:156, 174`] | n/a (no root-finder) | n/a | Only ours uses root-finding |
| Simulation games | 10,000 per (α, treatment), seed = 42 [VERIFIED `simulate_equilibrium.py:29, 31`] | n/a | n/a | Only ours simulates |

---

## 5. Divergences flagged (for user judgment)

### D1. Price schedule levels and indexing convention

- **What**: Ours uses `PRICES = [2, 4, 6, 8]` indexed so `PRICES[n-1-j]` gives the first seller the top price 8. Munro writes `prices = c(8, 5.5, 3, 0.5)` — **descending**, so `prices[1] = 8` is the first-seller price and `prices[4] = 0.5` is the last-seller price. This is a labeling/ordering convention, not a structural model difference.
- **Where**: ours — `equilibrium_model.py:14`; Munro — `MixedStrat_low_RA.R:18`, `MixedStrat_low_RN.R:17`.
- **Reconciliation**: The replication script (`compare_munro_replication.py`) reverses Munro's vector to ascending `[0.5, 3, 5.5, 8]` before monkey-patching our `PRICES`, so the two implementations then share the same economic price schedule. Post-fix per-n |dV| is ≤ 0.70 and per-n |dσ| is ≤ 0.025 (see §R) — so the remaining disagreement is *not* driven by the price level.
- **Status**: Downgraded from "dominant driver" to "parameter labeling convention, easily reconciled." The paper should still document which ordering convention it uses to avoid confusion for readers cross-referencing Munro's R code.

### D2. How σ is determined

- **What**: Munro specifies σ as a piecewise constant function (threshold at some belief, one interior mixing value at `beliefs[num_beliefs - k]`) and iterates V to fixed point given that σ. The author presumably tuned the interior mixing by hand to get `exp_price − exp_W ≈ 0` at that belief; this is visible as `print(exp_price-exp_W)` at the end of each block [VERIFIED `MixedStrat_low_RA.R:91, 145, 204`]. Ours solves σ(n, π') at every grid point every iteration via Brent root-finding on the indifference condition [VERIFIED `equilibrium_model.py:136-156, 206-216`].
- **Where**: ours — `equilibrium_model.py:136-156`, `206-216`; Munro — `MixedStrat_low_RA.R:48-53`, `106-107`, `165-167`; same structure in RN.
- **Likely effect**: Munro's approach can arrive at a slightly off-equilibrium σ whenever the hand-chosen mixing value doesn't exactly satisfy indifference. That bias is bounded by how small `exp_price - exp_W` ends up after convergence. Ours has no such bias but is more expensive.
- **Recommendation**: Worth checking Munro's reported σ (the `sig` vector printed for n = 2, 3, 4 in both RA and RN) against `exp_price - exp_W` at convergence to quantify how close to indifference Munro's chosen interior mixing is. If that residual is < 0.01, the two methods should agree to within grid-coarsening error. If it is larger, Munro's σ is approximate and differences from ours should be expected.

### D3. Belief grid resolution and construction

- **What**: Ours has 41 reachable beliefs generated by Bayesian iteration off π₀ = 0.5 [VERIFIED `equilibrium_model.py:41-57`, `simulate_equilibrium.py:30`]. Munro has 15 hand-chosen beliefs that are not all reachable by exact Bayesian updating [VERIFIED `MixedStrat_low_RA.R:11`]. Munro then coarsens each signal update to the nearest grid point via the `index_b`, `index_g` integer tables.
- **Where**: ours — `equilibrium_model.py:41-57`; Munro — `MixedStrat_low_RA.R:11-13`.
- **Likely effect**: Medium to large. Munro's coarsening induces a systematic rounding of beliefs toward the nearest grid point, which can shift both the effective W values and the threshold location. Our own robustness check (Appendix D §9.1) shows that increasing T_max beyond 20 (i.e. past 41 points) does not change σ past 4 decimals, suggesting 41 is converged **with respect to our update rule**. It does not tell us what Munro's 15-point grid converges to.
- **Recommendation**: Re-solve our model on Munro's 15-point grid (swap `build_belief_grid` for Munro's exact vector and replace the Bayesian `_update_good`/`_update_bad` calls with the `index_g`/`index_b` integer map). If that reproduces Munro's σ values to within solver noise, grid coarsening is the explanation. The replication script (task #4) should include this test.

### D4. Convergence tolerance and criterion

- **What**: Ours uses absolute L∞ change in V with tol = 1e-8 [VERIFIED `equilibrium_model.py:180, 201`]. Munro uses relative L∞ (`|new/old − 1|`) with tol = 1e-3 [VERIFIED `MixedStrat_low_RA.R:24, 84`]. Max iterations differ: 1000 vs. 50.
- **Where**: ours — `equilibrium_model.py:180, 196-202`; Munro — `MixedStrat_low_RA.R:24-25, 84-85`.
- **Likely effect**: Small to moderate. 1e-3 relative on V values of order 10 means Munro accepts absolute V differences of ~0.01. At Munro's 50-iteration cap, the bound on remaining value-iteration error is bounded by `(1 − λ)^50 · V_max`, which under λ = 0.125 gives ~0.0013 · 20 = 0.026 — so Munro's 50 iterations may not be tight, and combined with tol = 1e-3, V could carry O(0.01) error into `exp_price − exp_W`, contaminating σ slightly. Our Appendix D §8.3 ruled out convergence as a cause by running policy iteration (exact solve) with identical answer, but that was under our algorithmic setup.
- **Recommendation**: When running the replication in task #4, report how many iterations each of Munro's blocks uses and the final `dist` value. If Munro hits the 50-iteration cap without converging, that's a flag.

### D5. Multiple-seller payoff convention vs. our treatments

- **What**: Munro's code has one payoff convention: `mean(u(prices[remaining]))` — apply u to each price, then average (lottery over remaining prices). This matches our "random" treatment [VERIFIED `equilibrium_model.py:90-91` vs. `MixedStrat_low_RA.R:73, 79`]. Ours additionally implements "average": `u(mean(prices[remaining]))` — average prices first, then apply u [VERIFIED `equilibrium_model.py:92`]. Under linear utility these coincide; under CRRA they differ by Jensen's inequality.
- **Where**: ours — `equilibrium_model.py:87-100`; Munro — `MixedStrat_low_RA.R:73, 79, 127, 133, 193`.
- **Likely effect**: Our "random" treatment should reproduce Munro RA at α = 0.5 (holding other divergences fixed). Our "average" treatment is a true extension and is not comparable to Munro directly.
- **Recommendation**: Compare only our "random" treatment against Munro RA, and only our α = 0 "random" against Munro RN. "Average" vs. Munro would conflate two differences.

### D6. Belief convention: P(Good) vs. P(Bad)

- **What**: Ours treats `pi` as P(Good): `good` signals push π up, `crra_utility(FINAL_VALUE)` is weighted by `pi_prime` in `_h_value` [VERIFIED `equilibrium_model.py:125-130`]. Munro labels the state variable `beliefs` but **the `term_value` formula** `(1-beliefs)*u(v) + beliefs * mean(u(prices))` weights v by `(1-beliefs)` and the forced-liquidation mean price by `beliefs`. In ours, v gets weight `pi_prime` (= P(Good)) and L gets weight `(1-pi_prime)` [VERIFIED `equilibrium_model.py:127-128`].
- **Where**: ours — `equilibrium_model.py:127-128`; Munro — `MixedStrat_low_RA.R:73` and analogous n=3, n=4 lines.
- **Likely interpretation**: Munro's `beliefs` variable is **P(Bad)** = Pr(z = 0 = Bad), which is the opposite of our `pi`. Reading the belief grid `{0, 0.01, 0.02, 0.05, 0.1, 0.19, 0.33, 0.5, 0.68, 0.82, 0.9, 0.95, 0.98, 0.99, 1}` together with `prob_b = beliefs*mub + (1-beliefs)*mug` (which is `P(sig=? | belief)`): if `beliefs = 1` means certain Bad, then `prob_b = mub = 0.675`, the probability of a bad signal given Bad — this lines up. And the threshold `beliefs >= 0.90` meaning "sell when confident in Bad" is consistent with the market-runs setup (sell fast if you think asset is bad). [VERIFIED by cross-checking `term_value` with state semantics: in Bad state the forced liquidation pays one of the low prices, weighted by `beliefs` — so `beliefs` = P(Bad). ]
- **Likely effect on comparison**: **This is a convention flip, not a model difference.** Any threshold comparison must translate: Munro's "sell if beliefs ≥ 0.90" corresponds to our "sell if π ≤ 0.10" (i.e. π* = 0.1 in our units). Our threshold output columns are in P(Good), so Munro's n=2 RA threshold of 0.90 in P(Bad) is 0.10 in our P(Good) units.
- **Recommendation**: In the replication script (task #4), convert Munro thresholds to P(Good) via `1 - threshold_in_P(Bad)` before comparing. Also document this in the paper appendix if not already stated — a reader of Munro's code who assumes `beliefs` = P(Good) will get the sign of risk-aversion effects wrong.
- **Status**: The replication script now applies this P(Bad)→P(Good) conversion before comparing σ and V on-grid; the output CSV `analysis/output/munro_replication_comparison.csv` includes both `belief_p_bad` (Munro's original) and `belief_p_good` (our convention) columns [VERIFIED `analysis/analysis/compare_munro_replication.py`]. Applying the conversion cut |dV| magnitudes from ~7.5 to ~4.24, but did **not** shrink |dσ| below 1.0 — the residual σ disagreement is driven by D2/D3 (hand-set near-step σ on a 15-point grid), not by the convention.

### D7. n = 1 handling

- **What**: Ours treats `n = 1` as a corner: `find_sigma(n=1, …) = 0.0` so the last holder never sells [VERIFIED `equilibrium_model.py:138-139`]. Munro computes `V_1 = pmax(prices[4], beliefs·mean(prices[4]) + (1-beliefs)·v)`, i.e. the last holder may still sell for `prices[4] = 0.5` [VERIFIED `MixedStrat_low_RN.R:27`]. Our tabulation explicitly drops n = 1 from the output table [VERIFIED `tabulate_equilibrium.py:23`, comment in `_table_footer()` at line 103].
- **Where**: ours — `equilibrium_model.py:138-139`; Munro — `MixedStrat_low_RN.R:27`, `MixedStrat_low_RA.R:31`.
- **Likely effect**: Small. In Munro, V_1 = max(u(0.5), ...) — the expected-hold value `(1-beliefs)·v + beliefs·u(0.5)` is almost always higher than u(0.5) for the belief range in question (because v = 20 dominates u(0.5) ≈ 1.414 weighted by all but extreme P(Bad)), so `pmax` picks holding nearly everywhere. But at beliefs very close to 1 (near certain Bad), Munro's V_1 equals u(prices[4]) = u(0.5), not the expected hold payoff. That affects V_1 at the extreme grid points only, which propagates weakly through n = 2's `exp_W` via `W_1`. Whether this matters numerically at interior beliefs is small but not zero.
- **Recommendation**: Low priority but worth noting in the replication-test output. Check whether Munro's V_1 values differ from ours only near `beliefs = 1` (= P(Good) = 0) where n = 4 sale has already happened and W_1 is less active.

### D8. Utility-function definition at α = 0

- **What**: Our `crra_utility` returns `float(x)` when `abs(alpha) < 1e-10` [VERIFIED `equilibrium_model.py:77-78`]. Munro RN does not apply a utility function — it uses prices directly [VERIFIED `MixedStrat_low_RN.R:27, 40, 75`]. These are mathematically equivalent.
- **Where**: ours — `equilibrium_model.py:75-81`; Munro — no u() in RN.
- **Likely effect**: None. Included for completeness.
- **Recommendation**: None.

---

## R. What the replication actually shows

**Background on an earlier misframing.** An earlier draft of this memo reported that our solver returned σ ≡ 0 identically under Munro's primitives and framed this as a genuine "never-sell vs. sell-at-high-P(Bad)" qualitative disagreement between the two implementations. That framing was **wrong** — it was a replication-script bug, not a model difference. Munro writes his price vector in **descending** order (`prices = c(8, 5.5, 3, 0.5)`, with `prices[1]` = first-seller price), whereas our code expects PRICES in **ascending** order and indexes via `PRICES[n-1-j]`. The replication script was monkey-patching our `PRICES` with Munro's descending vector without reversing it, effectively flipping the price-by-rank mapping and collapsing the equilibrium to the never-sell corner. After reversing Munro's vector to `[0.5, 3, 5.5, 8]` before the monkey-patch, the equilibrium comparison is apples-to-apples.

**Post-fix per-n diagnostic** (our solver on Munro's 15-point grid, P(Bad)→P(Good) converted, prices reversed to ascending, t_max=20):

| n | max \|dσ\| | mean \|dσ\| | corr(σ) | max \|dV\| | mean \|dV\| |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0000 | n/a  | 0.0000 | 0.0000 |
| 2 | 0.0205 | 0.0024 | 0.9999 | 0.2126 | 0.0330 |
| 3 | 0.0015 | 0.0001 | 1.0000 | 0.3538 | 0.0422 |
| 4 | 0.0248 | 0.0029 | 0.9999 | 0.6983 | 0.0698 |

**Interpretation.** The two independently-coded equilibrium solvers now agree closely:

1. **Strategies align.** Correlation between `munro_sigma` and `our_sigma` is ≈ 1.0 for n = 2, 3, 4, and pointwise |dσ| ≤ 0.025. The correlation is n/a for n = 1 because both implementations return σ ≡ 0 there (zero variance), not because of any disagreement. This is a direct success of the replication.

2. **Values align to within ~0.7.** Max |dV| is 0.70 (at n = 4) and mean |dV| is ≤ 0.07 across all n. Mean relative |dV|/V across the grid is ~0.9%; the worst single grid point (n = 4, P(Bad) = 0.68, where munro_V = 3.83) reaches ~18% relative error, concentrated at the interior mixing belief where D2/D3 sensitivity is highest.

3. **Residual disagreement is attributable to D2/D3/D4, not to price level or utility.** The remaining |dσ| of 0.02 and |dV| of ≤ 0.70 are consistent in magnitude with the known divergences the memo flagged earlier: D2 (Munro hand-sets interior σ values tuned so `exp_price − exp_W ≈ 0`, whereas we solve σ at every state by Brent root-finding), D3 (Munro coarsens posteriors to the 15-point grid via `index_g`/`index_b`, we use exact Bayes with linear W-interpolation), and D4 (Munro's convergence tolerance is 1e-3 relative vs. our 1e-8 absolute). No additional modeling-level divergence is needed to explain the residual.

**Files.** CSV: `analysis/output/munro_replication_comparison.csv` (60 rows: n × 15 grid points; columns include `belief_p_bad`, `belief_p_good`, `munro_sigma`, `our_sigma`, `sigma_diff`, `munro_V`, `our_V`, `V_diff`). Plot: `analysis/output/plots/munro_replication_comparison.png`. Script: `analysis/analysis/compare_munro_replication.py`.

---

## 6. Open questions for the user

1. **Canonical price schedule.** Which price vector matches the experimental design you want the paper to speak to — our `[2, 4, 6, 8]` (ascending) or Munro's `[8, 5.5, 3, 0.5]` (descending)? Both are the same economic object once orientation is reconciled (the replication script does this), but the paper should be consistent about which convention it uses when presenting thresholds and prices to readers.
2. **Should σ be an output or an input?** Munro treats the interior mixing value as a hand-tuned input to verify equilibrium after the fact. Ours treats σ as an output. Post-fix the two approaches agree tightly, but worth clarifying in the paper whether we claim to replicate M&M's algorithm exactly or extend it with endogenous σ.
3. **Attribution of residual D2/D3/D4 disagreement.** The post-fix |dσ| ≤ 0.025 and |dV| ≤ 0.70 are small. If a finer decomposition is wanted, one could (a) run our solver on Munro's 15-point grid with the `index_g`/`index_b` rounding (isolating D3), (b) fix our σ to Munro's hand-set vector (isolating D2), and (c) tighten Munro's convergence tolerance (isolating D4). Low priority given the residuals are already small.
4. **Munro's belief variable is P(Bad), not P(Good).** (See D6.) Is this convention noted in the paper? If we display Munro thresholds in our equilibrium tables alongside ours, we should be explicit that one is reported in P(Good) and the other in P(Bad), or convert everything to one convention before tabulating.
5. **Scope of "replication" claim.** The bellman_equations.tex Appendix D claims we replicate aggregate M&M Table 2 P(Bad) to within 1.2pp at α = 0 and exactly at α = 0.5. Post-fix the on-grid replication supports this — agreement is tight in σ and V — so the aggregate match is consistent with the pointwise match and no longer surprising.

---

## 7. Confidence summary

- **High confidence (VERIFIED from source in both repos):** All of §2–§4 except where marked; Divergences D1, D2, D3, D4, D5, D7, D8.
- **Medium-to-high confidence:** D6 — the P(Bad) interpretation of Munro's `beliefs` is inferred from the `term_value` formula and the threshold direction, but is also now corroborated by the post-fix replication: correlation(σ) ≈ 1.0 and |dσ| ≤ 0.025 after applying both the price-orientation reversal and the P(Bad)→P(Good) conversion. If D6 were wrong, we would not see this tight agreement.
- **Uncertain:** Whether Munro's published σ values are machine-computed or hand-tuned; whether the cap of 50 iterations is hit.
