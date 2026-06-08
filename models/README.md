# Model Checkpoints

Top-level model files are the active expanded 10-shot TAE-like checkpoints:

- `nova_mode_classifier.joblib` — Random Forest trained on
  `training_labels/tae_like.csv`.
- `nova_cnn_raw.pt` — raw CNN trained on `training_labels/tae_like.csv`.

Current expanded-set checks:

- RF OOF: CM `[[1404, 43], [93, 585]]`, accuracy `0.94`, GOOD recall `0.86`.
- Raw CNN held-out: CM `[[288, 7], [14, 115]]`, accuracy `0.95`, GOOD recall
  `0.89`.

Historical four-shot RF, raw CNN, straightened CNN, hybrid CNN, and LOSO
checkpoints are archived under `old_4shots_models/`.

The default `sort_shot_mixed.py` fusion thresholds were selected from four-shot
LOSO checks. Revalidate or retune that policy before treating it as final for
the expanded 10-shot models.
