# CANONICAL.md

Single source of truth. Every number carries its source file and enough
reproduction detail to re-derive it. Prose anywhere in the repo that
contradicts a row here is wrong.

**Frame rule: headline QLIKE = 2823-row full stream; per-fold analysis =
fold-covered 2772 rows. Never mix.**

**Fold definition** (the most load-bearing row): `retrain_policies.csv` has no
fold column. Folds are reconstructed by joining its date index against
`concept_drift.csv` `test_start`/`test_end` (11 folds × 252 rows = 2772; the
remaining 51 stream rows are out-of-fold). This is the only fold definition in
the repo. Equivalent generator: `src/splits.py::walk_forward_splits(3831,
min_train=1008, test_size=252, embargo=21)`.

All values below were reproduced from a fresh top-to-bottom run on 2026-08-08.

## Stream vs fold-frame QLIKE, all 8 policies

Source: `retrain_policies.csv` (per-day `loss` column per policy).
How: stream = mean over all 2823 rows; folds = mean over the 2772 fold-covered
rows (fold membership from `concept_drift.csv`, see above).

| policy | QLIKE stream (2823) | QLIKE folds (2772) | stream margin vs no_retrain | fold margin vs no_retrain |
|---|---|---|---|---|
| `no_retrain` | 0.074554 | 0.075454 | — | — |
| `adwin_directional` | 0.074478 | 0.075558 | **−7.56e-05** | **+1.04e-04** (sign flips) |
| `adwin_raw` | 0.074650 | 0.075714 | +9.57e-05 | +2.60e-04 |
| `psi_budget_matched` | 0.076179 | 0.077288 | +1.63e-03 | +1.83e-03 |
| `periodic_21` | 0.077097 | 0.078209 | +2.54e-03 | +2.75e-03 |
| `periodic_63` | 0.076546 | 0.077640 | +1.99e-03 | +2.19e-03 |
| `periodic_252` | 0.077058 | 0.078161 | +2.50e-03 | +2.71e-03 |
| `oracle_single_retrain` | 0.073698 | 0.074764 | −8.56e-04 | −6.90e-04 |

| claim | value | source | how |
|---|---|---|---|
| out-of-fold offset, ALL policies | −1.6e-04 to −2.1e-04 | `retrain_policies.csv` | (stream margin − fold margin) per policy; 51 out-of-fold rows; near-uniform, not adwin-specific; adwin_directional's own offset −1.79e-04 |
| out-of-fold QLIKE, no_retrain | 0.025634 | `retrain_policies.csv` | mean loss over the 51 out-of-fold rows; vs 0.0157–0.0171 for every retraining policy |
| adwin_directional is the only sign flip | fold deficit +1.04e-04 < offset 1.79e-04 | `retrain_policies.csv` | no other policy's fold margin is small enough for the offset to flip it |
| **within folds, no policy beats no_retrain** | fold margins all ≥ +1.04e-04 (oracle excepted, lookahead) | `retrain_policies.csv` | table above |

## Oracle (reference line, not a ceiling)

| claim | value | source | how |
|---|---|---|---|
| oracle QLIKE stream | 0.073698 | `retrain_policies.csv` / `retrain_summary.csv` | lookahead-constructed (single retrain at first adwin_raw fire in f07, 2022-03-22); excluded from any policy ranking |
| oracle f07 differential | −0.01732 | `retrain_policies.csv` | fold-mean loss minus no_retrain fold-mean, fold 7 |
| oracle f08 differential | +0.01333 | `retrain_policies.csv` | same, fold 8 — the transient post-retrain tax |
| oracle f09 / f10 differential | −0.00086 / −0.00273 | `retrain_policies.csv` | tax reverses within one fold; no 4-year persistence |
| oracle is not an upper bound | periodic_21 f07 = −0.02386 vs oracle −0.01732 | `retrain_policies.csv` | a no-lookahead policy beats the lookahead reference inside f07 |

## f08 two-tier structure

Source: `retrain_policies.csv` fold-mean differentials + fire locations
(`retrain_triggered` per policy, fold via `concept_drift.csv`).

| claim | value | how |
|---|---|---|
| 0 fires at f08 (`oracle`, `psi_budget_matched`, `adwin_directional`) | +0.01333 / +0.01328 / +0.01349 | pure carry-over from the f07 retrain, ≈ +0.0133 |
| ≥1 fire at f08 (`periodic_21/63/252`, `adwin_raw`) | +0.02048 / +0.02040 / +0.01911 / +0.02019 | carry-over plus ≈ +0.006 in-fold level shift |
| extra cost not proportional to fire count | 1 fire: +0.0058; 12 fires: +0.0072 | periodic_252 (1 f08 fire) vs periodic_21 (12 f08 fires), minus the +0.01333 tier |

## Unanimous-sign folds

Source: `retrain_policies.csv` fold-mean differentials, policies with
differential ≠ 0 in that fold.

| claim | value | how |
|---|---|---|
| unanimous gain folds (ex-f00) | {f01, f07, f10} | f01: −0.0067 to −0.0082; f07: −0.0154 to −0.0239; f10: −0.0005 to −0.0031 |
| unanimous loss folds | {f03, f08} | f03: +0.0108 to +0.0244; f08: see two-tier table |
| f00 | unanimous loss, n_active = 2 | cold start; only periodic_21 (+0.0134) and periodic_63 (+0.0077) fire |
| f01 gains but is not flagged | z = +0.5202, p = .3015, drift = False | `concept_drift.csv` fold 1 — the trigger is not a sufficient statistic for when retraining helps |

## Concept drift (stage 06)

| claim | value | source | how |
|---|---|---|---|
| f07 drift z | +2.0751 | `concept_drift.csv` | fold 7; two-proportion z vs calm baseline p0 = 0.3923, normal reference, n_eff = 12 (t₁₁ not applied) |
| f07 one-sided p | .018989 | `concept_drift.csv` | 1 − Φ(2.0751); CSV stores 0.0190 (4 dp) |
| multiplicity verdict | 0 rejections under Bonferroni, Holm, BH | derived | m = 11, α = .05; all three share threshold α/m = .004545 at the minimum p; survival needs z ≥ 2.6086 |
| second-smallest p | .1684 (f02, z = +0.9607) | `concept_drift.csv` | 8.9× the f07 p — isolated peak, not a field of candidates |
| t₁₁ caveat | f07 p ≈ .031 under t₁₁ | derived (caveat, not a result) | exact reference for a mean over 12 blocks; verdict unchanged |
| P(X) cannot separate f07 from f08 | rv21 PSI 10.84 vs 10.09 (ranks 1, 2 of 11) | `covariate_shift.csv` | trailing-252 reference; stage 05 intensity table |
| P(Y|X) separates them | z = +2.0751 vs −0.2920 | `concept_drift.csv` | folds 7 and 8 |
| sign lens vs magnitude lens disagree | f06: loss_rate 0.5000, z = +0.7001; f00: loss_rate 0.4048, z = +0.0824 | `concept_drift.csv` | loss_rate = daily sign over 252 days; z = magnitude-weighted over 12 blocks |

## Uncertainty (stage 07)

| claim | value | source | how |
|---|---|---|---|
| ACI worst-fold coverage | 0.817 at f09 = −8.3pp | `uncertainty.csv` | `cov_aci` column, delay = 21, γ = 0.01, nominal 0.90 |
| offline worst | 0.631 at f07 (Bayesian) = −26.9pp | `uncertainty.csv` | worst entry over cov_split / cov_norm_garch / cov_bayes across folds |
| worst-case reduction | 3.3× | derived | 26.9pp / 8.3pp |
| ACI mean / f07 coverage | 0.881 / 0.925 | `uncertainty.csv` | mean over 11 folds; fold 7 |
| f09 breakdown is a calibration artifact, not drift | split 0.655, norm 0.639, Bayesian 0.929 at f09; drift z = −0.6055 | `uncertainty.csv` + `concept_drift.csv` | f08's small residuals compress the rolling calibration quantile; the window-free Bayesian method is unaffected |

## Retired (do not resurrect)

CSCV/PBO and Deflated Sharpe (7 trials → statistic uninterpretable; notebook
09 deleted), "oracle ceiling = 0.0728", the 4-year window-contamination story,
"f07 is the sole gain source", winner's-curse framing, the joint block-shuffle
null, the spread-ratio law, any "P(Y|X) triggers beat no_retrain on average"
claim (closed by the FIT_END = 987 embargo fix).
