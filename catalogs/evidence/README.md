# Training review fingerprints

`training_input_fingerprints.csv` preserves the 2,327 `mode_key` and
`input_fingerprint` pairs from the September 13, 2026 training audit. These
are saved mode-plus-continuum fingerprints; they are not recomputed from
current raw files. Human decisions still come from
[`training_labels/tae_like_train.csv`](../../training_labels/tae_like_train.csv).

The original full audit, `outputs/review_morphology_v13_20260913/training_comparison.csv`,
was already ignored. Keeping the two required columns here makes review
matching available in a fresh checkout. The companion metadata records both
source and extracted checksums. No labels or accepted-mode selections changed.

The existing October 26 catalog metadata retains its original source paths
and checksums as historical provenance. Future catalog regeneration uses this
file through [`configs/catalog_review_sources.csv`](../../configs/catalog_review_sources.csv).
