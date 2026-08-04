# CANONICAL.md

Single source of truth. Every number carries its source and reproduction.
Paste this into a new session instead of a prose status block.

**Frame rule: headline = 2823-row stream; per-fold = 2772 rows. Never mix.**

| claim | value | source | how |
|---|---|---|---|
| QLIKE stream, no_retrain | `0.074554` | `retrain_policies.csv` | mean loss, all 2823 rows |
| QLIKE stream, adwin_directional | `0.074478` | `retrain_policies.csv` | mean loss, all 2823 rows |
| margin adwin_dir vs base, STREAM | `-7.56e-05` | `retrain_policies.csv` | 2823-row frame; WIN |
| margin adwin_dir vs base, FOLDS | `+1.04e-04` | `retrain_policies.csv` | 2772-row frame; LOSS -- sign flips |
| out-of-fold offset, ALL policies | `~-1.6e-04` | `frame_decomposition.csv` | 51 rows x -0.009; near-uniform, not adwin-specific |
| out-of-fold QLIKE, no_retrain | `0.025634` | `frame_decomposition.csv` | vs 0.0157-0.0171 for all retraining policies |
| oracle QLIKE stream | `0.073698` | `retrain_policies.csv` | reference line, NOT a ceiling |
| f07 drift z | `+2.0751` | `concept_drift.csv` | fold==7, normal ref, n_eff=12 (t11 not applied) |
| f07 one-sided p | `0.018989` | `concept_drift.csv` | vs alpha/m = .004545; fails Bonf/Holm/BH |
| f01 drift z | `+0.5202` | `concept_drift.csv` | drift=False yet all firing policies gain |
| oracle f07 differential | `-0.01732` | `fold_loss_differential.csv` | vs no_retrain, fold mean |
| oracle f08 differential | `+0.01333` | `fold_loss_differential.csv` | transient tax, reverses by f09 |
| fold definition | `11 folds` | `concept_drift.csv` | test_start/test_end; ONLY fold definition -- rp has no fold column |

| unanimous folds (ex-f00) | `gain {1,7,10} / loss {3,8}` | `fold_loss_differential.csv` | sign agreement among policies with diff != 0 |
| f00 | `unanimous loss, n_active=2` | `fold_loss_differential.csv` | cold-start; only periodic_21/63 fire |
