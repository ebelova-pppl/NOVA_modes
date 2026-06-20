# Model Checkpoints

Top-level model files are the active expanded 10-shot TAE-like checkpoints:

- `nova_mode_classifier.joblib` — Random Forest trained on
  `training_labels/tae_like_train.csv`.
- `nova_cnn_raw.pt` — raw CNN trained on `training_labels/tae_like_train.csv`.

Current expanded-set checks:

- RF OOF: CM `[[1404, 43], [93, 585]]`, accuracy `0.94`, GOOD recall `0.86`.
- Raw CNN held-out: CM `[[290, 5], [8, 121]]`, accuracy `0.969`, GOOD
  precision/recall/F1 `0.960 / 0.938 / 0.949`.

The active raw-CNN checkpoint uses OneCycleLR plus gradient clipping for both
split training and its 80-epoch full-data refit. The final refit used all 2,125
labels and ended at loss `0.0008`.

Historical four-shot RF, raw CNN, straightened CNN, hybrid CNN, and LOSO
checkpoints are archived under `old_4shots_models/`.

The default `sort_shot_mixed.py` fusion thresholds were selected from four-shot
LOSO checks. Expanded 10-shot LOSO makes raw CNN strongest overall, while the
combined policy retains better GOOD recall on the sparse NSTX-U G-case group.
Threshold retuning remains pending that tradeoff.
