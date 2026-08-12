# Volatility forecasting under concept drift: when does retraining pay?

Retraining a deployed volatility model is not free: it costs a refit, and — as
this project measures — it can make the model worse for the following year.
The question here is whether drift detection on the conditional distribution
P(Y|X) identifies *when* retraining pays better than covariate-shift detection
on the input distribution P(X). The detectors (DDM, ADWIN), the conformal
machinery, and the GARCH likelihood are implemented from scratch in `src/` and
validated against reference implementations where one exists.

## Result

No retraining policy beat the static baseline within the validated fold
structure. On the full 2823-day stream the best drift-triggered policy,
`adwin_directional`, edged out `no_retrain` by 7.6e-5 QLIKE — but that entire
margin came from 51 days that fall outside the 11 test folds, days on which
every retraining policy scores well. Restricted to the 2772 fold-covered days,
`adwin_directional` trailed `no_retrain` by 1.04e-4. The headline win was a
frame artifact.

What survived is a separation result. Folds f07 (Feb 2022–Feb 2023) and f08
(the year after) look almost identical to a P(X) detector — rv21 PSI intensity
10.84 vs 10.09, ranks 1 and 2 of 11 — yet retraining paid −0.0154 to −0.0239
QLIKE in f07 and cost +0.0133 to +0.0205 in f08. The P(Y|X) statistic told
them apart: drift z = +2.0751 at f07 vs −0.2920 at f08. That separation is the
contribution. Under multiple-testing correction across 11 folds the f07 result
did not survive (one-sided p = .0190 vs α/m = .004545 for Bonferroni, Holm,
and BH alike), so it is reported as exploratory evidence, not a confirmed
detection.

## Data and setup

One asset: QQQ daily realized volatility, 2011-01-03 to 2026-05-28, fetched
from Yahoo Finance by notebook 01. The forecast target is 21-day-forward RV
(h = 21); features are the HAR components rv1/rv5/rv21 plus VIX, 10y/3m
yields, curve slope, an HYG/LQD credit proxy, and TLT/GLD returns. Evaluation
is walk-forward: 11 folds, expanding training window from 1008 rows, 252-day
test blocks, and a 21-day embargo so no training label overlaps a test window.
Because consecutive h = 21 targets share 20 of their 21 days, each 252-day
fold carries only n_eff = 252/21 = 12 independent blocks, and all fold-level
tests use that effective sample size.

## Pipeline

One notebook per stage; each writes the file the next stage reads.

1. `01_fetch_raw` — download 8 price/index series → `raw_close.csv`.
2. `02_build_target` — features and forward RV targets → `dataset.csv`.
3. `03_garch_baseline` — persistence, HAR, and a hand-rolled GARCH(1,1)
   checked against the `arch` package; fold-mean h21 RMSE 0.0049 (HAR).
4. `04_seq_models` — an LSTM matches HAR (0.0050 vs 0.0049) and is dropped.
5. `05_covariate_shift` — KS and PSI per fold and feature → `covariate_shift.csv`.
6. `06_concept_drift` — DDM/ADWIN streams and the fold-local z-test
   → `concept_drift.csv` (also the repo's fold definition).
7. `07_uncertainty` — split/normalized/adaptive conformal and Bayesian NIG
   intervals → `uncertainty.csv`.
8. `08_adaptive_retrain` — eight retraining policies through a no-look-ahead
   simulator → `retrain_policies.csv`, `retrain_summary.csv`.

## Three findings that survived scrutiny

**The post-retrain cost is a one-fold transient, not permanent contamination.**
A single retrain inside f07 gained −0.01732 there, cost +0.01333 in f08, and
was back to −0.00086 by f09. Policies that also fire inside f08 pay an extra
level shift of about +0.006 that does not scale with fire count (1 fire:
+0.0058; 12 fires: +0.0072). An earlier draft blamed a 1008-row training
window "retaining crisis data for four years"; the oracle's time profile
falsified that.

**The drift trigger is not a sufficient statistic for when retraining helps.**
Every policy that fired in f01 gained (−0.0067 to −0.0082 QLIKE), yet the
P(Y|X) test saw nothing there (z = +0.5202, p = .3015). Gains also recurred at
f10 without a flag. Detected drift and profitable retraining overlap at f07,
but neither implies the other.

**Coverage breakdown does not mean drift.** The worst interval-coverage fold,
f09, has a negative drift z; its collapse (split conformal 0.655 vs 0.90
nominal) traces to f08's unusually small residuals compressing the
calibration quantile. The Bayesian method, which uses no rolling calibration
window, covered 0.929 there. ACI's real contribution is a worst-case bound:
worst fold −8.3pp vs −26.9pp for offline methods, a 3.3× reduction. A
coverage-based retrain trigger would fire at exactly the wrong time.

## What was wrong and got fixed

Two errors shaped this project more than any success. First, the original
static HAR was fit on the full first 1008 rows, so its last 21 training labels
overlapped the first test days; fixing the embargo (`FIT_END = 987`) reversed
the then-headline result "P(Y|X) triggers beat the static baseline on
average" — `adwin_raw` went from winner to loser. Second, the stream-vs-fold
frame effect described above flipped the sign of the remaining 7.6e-5 win.
Both fixes are kept visible in the notebooks rather than smoothed over, and
`CANONICAL.md` now pins every headline number to its source file. The oracle
policy was also relabeled: it is a lookahead-constructed reference line, not a
ceiling — `periodic_21` beat it inside f07 (−0.02386 vs −0.01732).

## Limitations

One asset, 11 folds, 7 candidate policies. Differences of order 1e-4 QLIKE
among policies are not separable from selection effects at that trial count,
so no policy ranking is claimed; an earlier CSCV/PBO analysis was removed for
the same reason. Fold-level p-values use a normal reference where t₁₁ is
exact — that moves f07 from p ≈ .019 to roughly .03 and changes no verdict.
Rank order among policies also depends on the evaluation frame (stream vs
fold-covered) and on the loss lens (daily sign vs block magnitude:
f06 has a loss rate of exactly 0.5000 yet z = +0.7001); the differences are
smaller than the frame effect, so no winner is declared.

## Reproduction

Python ≥ 3.10 with `numpy<2`, `pandas`, `scipy`, `scikit-learn`,
`matplotlib`, `yfinance`, `arch`, `torch`. Run the notebooks in order,
01 through 08, in Colab or locally — each setup cell detects its environment.
Notebook 01 fetches the raw data (not committed, ~600 KB); everything
downstream is deterministic given `raw_close.csv`, and the full run takes
under two minutes on a laptop CPU, LSTM stage included.
Numbers cited above were reproduced from a fresh fetch on 2026-08-08;
`CANONICAL.md` lists each claim with its source file and derivation.
