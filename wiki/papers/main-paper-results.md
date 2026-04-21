---
title: "Main Paper: To Sell or Not to Sell — Results & Structure"
type: paper
tags: [paper, results, hypotheses, tables, market-runs]
summary: "Hypotheses, main results, table inventory, and section structure of the working paper (Eynon & Jindapon)"
status: active
last_verified: "2026-04-21"
---

## Citation

Eynon, C. and Jindapon, P. *To Sell or not to Sell: The Psychology of Market Runs.* Working paper. Source: `analysis/paper/main.tex` (~996 lines).

## Abstract (one-line)

A 2×2 experimental study (liquidation payoff × communication) of the psychological predictors — emotions, traits, luck, peer cascades — of market runs in a Bernardo–Welch setting.

## Hypotheses

LaTeX `\label`s reuse `chat_treatment` for several hypotheses; numbering in the prose is inconsistent. The substantive hypotheses are:

- **H1 (emotions)**: Joy reduces selling; fear, anger, engagement increase it.
- **H2 (price treatment)**: Subjects sell less under average-price (T2) than random-price (T1) — pooling reduces variance and encourages cooperation.
- **H3 (chat)**: Subjects sell less in the chat treatment than the no-chat treatment.
- **H4 (traits)**: Less selling — agreeableness, risk tolerance, openness. More selling — impulsivity, anxiety, conscientiousness.

## Main Results

- **Cascade effects** (Table 8 / Cox): 2 or 3 prior sellers in the immediately previous period significantly raise the selling hazard despite the public signal. Total prior-sales effect is negative (price channel).
- **Lucky / unlucky liquidation payoffs** (Table 7 / `holdout_liquidation_regression`): Receiving payoff 8 vs. 2 in random liquidation makes a participant **9.23% less likely to sell next round**; coefficients monotonic in payoff.
- **Emotions** (Tables 5, 8): Joy positive but marginal in full sample, becomes negative for sellers-only. Disgust positive in sellers subsample. Valence composite significantly negative on hazard. Fear and anger NOT significant — partial contradiction of H1.
- **Personality traits** (Tables 5, 8): Conscientiousness, neuroticism, anxiety raise hazard (consistent with H4). Risk tolerance lowers hazard (marginal). Agreeableness becomes significantly negative in sellers subsample. Effects ~3× stronger in sellers-only.
- **Chat treatment** (Tables 4, 6, 8): Chat segments reduce sellers per group-round by 1.07 and 1.29 (Tobit). DiD decomposition (Table 6) isolates pure communication effect = -0.4863 sellers (marginally significant, parallel-trends p=0.7553). Supports H3.
- **Average vs. random price**: T2 *increased* sellers by 0.4952 in Tobit (Table 4) — **contradicts H2**. Cox shows no significant treatment effect.
- **Signal/round**: Higher signal sharply lowers hazard; selling falls with round number (learning).
- **Welfare** (Table 9): OLS of group-round welfare on traits, restricted to good-state (z=1) rounds, clustered at session×segment×group.
- **Risk Aversion Consistency** (§5.5): The two independent α estimates — `alpha_mle` from selling behavior vs. `alpha_task` from the Gneezy-Potters survey lottery — are essentially uncorrelated. Raw correlation = 0.027 (n = 59 after dropping 36 participants with edge-flagged lottery allocations). Regressing `alpha_mle` on `alpha_task` with session FE and session-clustered SE yields coef = −0.0072 (SE 0.0145), R² = 0.136, within-R² = 0.003. The survey-based instrument does not predict behavior-based risk aversion in this setting.

## Table Inventory

LaTeX inputs are bare filenames; sources live in `analysis/output/tables/<name>.tex`.

| # | Label | Contents | Source script | Type |
|---|-------|----------|---------------|------|
| 1 | `randomized_params` | Pre-randomized experimental parameters by segment | `analysis/analysis/randomized_params_table.py` | Descriptive |
| 2 | `summary_demographics_traits` | Subject demographics + trait means by treatment | `analysis/analysis/summary_statistics.R` | Summary |
| 3 | `trait_correlations` | 8-trait pairwise correlations | `analysis/analysis/trait_correlations.R` | Correlation |
| 4 | `emotion_correlations` | 9 facial-emotion period-level correlations | `analysis/analysis/emotion_correlations.R` | Correlation |
| 5 | `summary_seller_counts` | Group-round seller counts/timing/prices by treatment×chat | `analysis/analysis/summary_statistics.R` | Summary |
| 6 | `tobit_n_sellers` | Tobit on # sellers/group-round; covariates: BadState, Treatment2, segment dummies, Round | `analysis/analysis/tobit_n_sellers.R` | Tobit (0,4) |
| 7 | `ro_logit_selling_position` | Rank-ordered logit / Cox-style hazard, stratified by session-segment-group-round | `analysis/analysis/ro_logit_selling_position.R` | RO logit |
| 8 | `did_learning_communication` | DiD decomposing learning vs. communication | `analysis/analysis/did_learning_communication.R` | DiD with FE |
| 9 | `cox_survival_regression` | Mixed-effects Cox PH; cascades, emotions, traits, controls | `analysis/analysis/cox_survival_regression.R` | coxme |
| 10 | `holdout_liquidation_regression` | Effect of holdout payoff on next-round sale; group×round FE | `analysis/analysis/holdout_liquidation_regression.R` | LPM |
| 11 | `welfare_regression` | OLS welfare on traits (z=1 only); SE clustered session×segment×group | `analysis/analysis/welfare_regression.R` | OLS clustered |
| 12 | `risk_aversion_consistency` | α_MLE on α_task; session FE, session-clustered SE; uses `participant_risk_aversion.csv`; accompanied by `risk_aversion_consistency.pdf` (`visualize_risk_aversion_consistency.R`) | `analysis/analysis/risk_aversion_consistency.R` | OLS with FE |
| App | `equilibrium_thresholds` | Avg equilibrium π at sale by seller position k and α (10k sims, both treatments) | `analysis/analysis/simulate_equilibrium.py` + `tabulate_equilibrium.py` | Numerical |
| App G | `*_valence_only` | Valence-only emotion robustness | `*_valence_only.R` | Robustness |
| App H | `*_no_valence` | Discrete-emotions-only robustness | `*_no_valence.R` | Robustness |

## Section Structure (line numbers in main.tex)

- §1 Introduction — line 95
- §2 Related Literature — line 117 (2.1 Market Runs 120, 2.2 Bank Runs 127, 2.3 Psychology 139)
- §3 Design of the Experiment — line 178 (3.1 Market Model 182, 3.2 Welfare 207, 3.3 Equilibrium 230, 3.4 Implementation 246, 3.5 Data Collection 302)
- §4 Hypotheses — line 308
- §5 Main Results — line 348 (5.1 Summary Stats 351, 5.2 Selling Behavior 388 [group-round 390, DiD 439, player-period 494], 5.3 Relative Income 545, 5.4 Welfare 569, 5.5 Risk Aversion Consistency 580)
- §6 Discussion and Conclusion — line 601
- Bibliography — lines 624–625
- Appendix G Valence-Only — line 704
- Appendix H No-Valence — line 714
- Appendix Equilibrium Predictions — line 724
- Appendix Instructions — line 735
- Appendices A–F (lines 655–697) commented out

## Related

- [Project Architecture](../tools/architecture.md)
- [Experiment Design](../concepts/experiment-design.md)
- [Derived Datasets Pipeline](../tools/derived-datasets.md)
- [Analysis Scripts](../tools/analysis-scripts.md)
