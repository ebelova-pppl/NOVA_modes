# Model Checkpoints

Top-level model files are the active v2 TAE-like checkpoints:

- `nova_mode_classifier.joblib` — Random Forest trained on
  `training_labels/tae_like_v2_nonG.csv`.
- `nova_cnn_raw.pt` — raw CNN trained on
  `training_labels/tae_like_v2_nonG.csv`.
- `nova_mode_classifier_bundle.joblib` — RF training arrays and schema
  metadata for the active checkpoint.

Current active-checkpoint metadata:

- RF: trained on 2900 rows: 594 `good`, 2306 `bad`; production
  `rf_w_star_max_22_v2` schema with 22 features.
- Raw CNN: full-list refit on 2900 rows after a held-out split check; best
  split accuracy `0.9534` at epoch 13; `M_target=100`, `R_target=201`,
  robust normalization, OneCycleLR peak LR `0.02`, gradient clipping `1.0`.
- Raw CNN final prediction-health metadata reports no collapse: 594 predicted
  GOOD among 2900 samples, matching the 594 true GOOD labels.

Current active SHA-256 checksums:

```text
2a96699bba6bb92d44c9f5b09373e35c3011c8d5bdeab297519e7cd69f5e6023  nova_mode_classifier.joblib
0cacd4f1b31347050c1192c109058c3a89c84e3f186ea0899e2a89f95a068629  nova_mode_classifier_bundle.joblib
29643ff060aa77e4d7624063803f199918c5a81a2a7e997f732b2481a6c0af49  nova_cnn_raw.pt
```

The active raw-CNN checkpoint uses OneCycleLR plus gradient clipping for both
split training and its 80-epoch full-data refit.

Historical four-shot RF, raw CNN, straightened CNN, hybrid CNN, and LOSO
checkpoints are archived under `old_4shots_models/`.

The pre-v2 canonical checkpoints were archived before retraining on
`training_labels/tae_like_v2_nonG.csv`:

- `pre_v2_nonG_20260810/nova_mode_classifier.joblib`
- `pre_v2_nonG_20260810/nova_mode_classifier_bundle.joblib`
- `pre_v2_nonG_20260810/nova_cnn_raw.pt`

The active RF checkpoint uses schema `rf_w_star_max_22_v2`: compared with the
previous 22-feature checkpoint, raw `omega` was removed and the
continuum-crossing feature `W_star_max` was added.

The default `sort_shot_mixed.py` fusion thresholds were selected from four-shot
LOSO checks. Expanded 10-shot LOSO makes raw CNN strongest overall, while the
combined policy retains better GOOD recall on the sparse NSTX-U G-case group.
Threshold retuning remains pending that tradeoff.
