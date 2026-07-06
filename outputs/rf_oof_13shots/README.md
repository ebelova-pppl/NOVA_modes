# RF OOF 13-Shot Audit

Inference-independent RF label audit on the current
`training_labels/tae_like_train.csv` list after merging:

- refreshed `nstx_135388`
- new `nstxuG121123J38`
- reviewed `nstxuG121123Q62`
- reviewed `nstxuG142301Y93`

Inputs:

- labels: `training_labels/tae_like_train.csv`
- model template: `models/nova_mode_classifier.joblib`
- feature schema: `rf_w_star_max_22_v2`

Results:

- rows: 2610
- labels: 606 `good`, 2004 `bad`
- OOF CM: `[[1967, 37], [91, 515]]`
- accuracy: `0.951`
- GOOD precision/recall/F1: `0.933 / 0.850 / 0.889`
- strong suspect rows: 43

Files:

- `oof_table.csv`: one OOF probability/prediction per labeled mode
- `oof_suspects.csv`: strong RF/manual disagreements using default
  `thr_low=0.2`, `thr_high=0.8`
