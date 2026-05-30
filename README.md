# Robust Volatility Forecasting under Concept Drift

Forecasting QQQ realized volatility, and detecting when the market's
underlying input-output relationship changes (concept drift) so a model
can adapt instead of silently degrading.

## Core question
Markets behave differently in calm vs turbulent regimes. Can a model
detect that the rules themselves changed, not just that inputs shifted,
and recover on its own?

## Setup
- Target: QQQ daily realized volatility (RV, N=21), multi-horizon h in {1, 5, 21}
- Inputs: QQQ returns/RV, VIX, rates (10y/3m), credit spread (HYG/LQD), TLT, GLD
- Period: 2011-2026, daily
- Forecast target uses a future window (t+1..t+h); inputs use info up to t only

## Pipeline (one notebook = one stage)
- 01_fetch_raw      raw price/index download
- 02_build_target   realized vol targets + aligned features
- 03_garch_baseline GARCH(1,1) statistical baseline
- 04_seq_models     LSTM / Transformer
- 05_covariate_shift  input-distribution shift detection
- 06_concept_drift  P(Y|X) change detection (main contribution)
- 07_uncertainty    conformal / Bayesian intervals
- 08_adaptive_retrain  drift-triggered retraining
- 09_eval_report    PBO, deflated Sharpe, final comparison

## Reproduce
Run notebooks in order. Data is fetched by 01_fetch_raw and not committed
(see .gitignore); each stage writes one output consumed by the next.

## Status
Work in progress.
