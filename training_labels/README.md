# Training Label Lists

This directory contains version-controlled mode-label CSV files used for
training, splitting, and auditing classifier datasets.

Paths in active training CSVs should be stored relative to `$NOVA_DATA`, for
example `nstx_120113/N5/egn05w.1234E+02`.

## Active training list

### `tae_like.csv`

Current active TAE-like good/bad training list for RF and CNN retraining.

Columns:
- `path`
- `validity`
- `family`
- `signed_delta`
- `fraction_below_upper2`
- `gap_region`
- `error`

Current checked contents:
- 1085 labeled modes
- labels: 426 `good`, 659 `bad`
- shots: `nstx_120113`, `nstx_135388`, `nstx_141711`, `nstxu_204202`

This is the list to use when retraining the current RF baseline before checking
new labels. Do not merge staged NSTX-U labels into this file until the new
label review is complete.

## Archived four-shot lists

### `old_4shots_tae_only_labels/`

Older TAE-only training lists from before the mixed TAE/EAE workflow.

Contents:
- `train_master.csv`
- `train_master_full_paths.csv`

These are historical inputs only; they are not the current default training
pool.

### `old_4shots_mixed_labels/`

Previous four-shot mixed TAE/EAE lists and derived audit files.

Contents include:
- `all_modes.csv`
- `all_modes_tae_eae_split.csv`
- `tae_like_loso_train_excluding_*.csv`
- `eae_like.csv`
- `mixed_tae_like.csv`
- `bad_tae_like.csv`
- cleanup/audit lists for the `nstx_135388` small-`n_m` file issue

Use this directory for historical audit, regeneration, or LOSO references. The
root of `training_labels/` intentionally no longer carries these older mixed
lists as active files.

## Staged six-shot NSTX-U labels

Six additional NSTX-U shots have staged TAE-like label lists in the shared
`nova2/metadata` area:

- `tae_like_6new_NG.csv`
- `tae_like_6new_not_cleaned_NG.csv`
- per-shot `*_tae_eae_split/` directories

These files are not version-controlled training inputs yet. They are for label
review with the current four-shot RF/CNN models.

Checked staged-label summary:
- cleaned staged list: 1040 rows, 284 `good`, 756 `bad`
- not-cleaned staged list: 1041 rows, with one duplicate mode
- all cleaned staged paths resolve to files under `$NOVA_DATA` by
  `shot/N/file` suffix
- the per-shot TAE/EAE split outputs contain 10 additional TAE-like modes that
  are not in the cleaned staged label list, so only the TAE-like subset has
  been labeled so far

The staged CSVs currently contain absolute source paths from the labeling
environment. Tools such as `label_modes_fast.py` can match them by stable
`shot/N/file` suffix when given `--mode-list`, but training inputs should be
converted to relative `$NOVA_DATA` paths before any future merge.

Example review command for one `N` directory:

```bash
python "$NOVA_REPO/scripts/label_modes_fast.py" \
  "$NOVA_DATA/nstxuE202855A01t020/N1" \
  --mode-list /path/to/nova2/metadata/tae_like_6new_NG.csv \
  --rf-model "$NOVA_MODELS/nova_rf_tae_like_full.joblib"
```
