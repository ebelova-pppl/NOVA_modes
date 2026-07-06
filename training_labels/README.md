# Training Label Lists

This directory contains version-controlled mode-label CSV files used for
training, splitting, and auditing classifier datasets.

Paths in active training CSVs should be stored relative to `$NOVA_DATA`, for
example `nstx_120113/N5/egn05w.1234E+02`.

## Active training list

### `tae_like_train.csv`

Current active expanded TAE-like good/bad training list for RF and CNN
retraining.

Columns:
- `path`
- `validity`
- `family`
- `signed_delta`
- `fraction_below_upper2`
- `gap_region`
- `error`

Current checked contents:
- 2610 labeled modes
- labels: 606 `good`, 2004 `bad`
- shots: `nstx_120113`, `nstx_135388`, `nstx_141711`, `nstxu_204202`,
  `nstxuE202855A01t020`, `nstxuE204669M03t025`, `nstxuE205052A01t022`,
  `nstxuG121123K51`, `nstxuG133964S31`, `nstxuG142301H47`,
  `nstxuG121123J38`, `nstxuG121123Q62`, `nstxuG142301Y93`

This is the list to use when retraining the expanded RF and CNN models.
The active RF checkpoint has been retrained on this 2610-row / 13-shot list.
The active raw-CNN checkpoint was trained before the 2026-07-06 merges, on the
previous 2125-row / 10-shot version; CNN retraining is pending a GPU
allocation.

### `tae_like_4old.csv`

Backup copy of the original four-shot TAE-like training list before the
six-shot merge.

### `tae_like_6new.csv`

Reviewed six-shot NSTX-U TAE-like list that was appended to
`tae_like_train.csv`.

The bare filename `tae_like.csv` is intentionally not used for the canonical
training list anymore, because `split_tae_eae.py` and `sort_shot_mixed.py`
write TAE-like output/audit files with that name in their own output
directories.

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

## Six-shot NSTX-U component list

Six additional NSTX-U shots have a cleaned staged TAE-like label list:

- `tae_like_6new.csv`

This list uses the same full schema as `tae_like_train.csv`. The `family`
column is set to `tae` for `good` rows and `none` for `bad` rows;
`signed_delta`, `fraction_below_upper2`, `gap_region`, and `error` were
restored from the per-shot split outputs in the shared `nova2/metadata` area by
matching the stable `shot/N/file` suffix.

Related metadata/audit files in the shared `nova2/metadata` area:

- `tae_like_6new_NG.csv`
- `tae_like_6new_not_cleaned_NG.csv`
- per-shot `*_tae_eae_split/` directories

The six-shot list has been merged into `tae_like_train.csv`.

Checked staged-label summary:
- cleaned staged list: 1040 rows, 252 `good`, 788 `bad`
- not-cleaned staged list: 1041 rows, with one duplicate mode
- all cleaned staged paths resolve to files under `$NOVA_DATA` by
  `shot/N/file` suffix
- the per-shot TAE/EAE split outputs contain 10 additional TAE-like modes that
  are not in the cleaned staged label list because they were marked `skip`
  during labeling; these are intentionally excluded from training

The shared metadata CSVs currently contain absolute source paths from the
labeling environment. The staged `training_labels/tae_like_6new.csv` file uses
relative `$NOVA_DATA` paths and is kept as the reviewed six-shot component
list.

Example review command for one `N` directory:

```bash
python "$NOVA_REPO/scripts/label_modes_fast.py" \
  "$NOVA_DATA/nstxuE202855A01t020/N1" \
  --mode-list "$NOVA_REPO/training_labels/tae_like_6new.csv" \
  --rf-model "$NOVA_MODELS/nova_rf_tae_like_full.joblib"
```

## Three-shot NSTX-U review list

Three additional NSTX-U G-case shots have a separate review-stage list:

- `tae_like_3new.csv`

This original combined list is not merged into `tae_like_train.csv` as-is. It
is intentionally still blocked because it includes `nstxuG121123N75`, whose
modes still need recalculation with the corrected q profile. The already
reviewed `nstxuG121123Q62` and `nstxuG142301Y93` rows were split into
`tae_like_2new.csv` and merged into the active training list. After the N75
recalculation, review the affected labels again before creating a replacement
N75 component.

Current checked contents:
- 523 labeled modes
- labels: 14 `good`, 509 `bad`
- shots: `nstxuG121123Q62`, `nstxuG121123N75`, `nstxuG142301Y93`
- per-shot counts:
  - `nstxuG121123Q62`: 241 rows, 13 `good`, 228 `bad`
  - `nstxuG121123N75`: 176 rows, 0 `good`, 176 `bad`; blocked pending
    recalculation
  - `nstxuG142301Y93`: 106 rows, 1 `good`, 105 `bad`
- paths are relative to `$NOVA_DATA`
- no duplicate paths
- all paths resolve under `$NOVA_DATA`

The source per-shot label files live beside the shot directories in the shared
`nova2/data` area and use Flux/DiTw absolute paths:

- `nstxuG121123Q62_mode_labels_clean.csv`
- `nstxuG121123N75_mode_labels_clean.csv`
- `nstxuG142301Y93_mode_labels_clean.csv`

Example review command:

```bash
python "$NOVA_REPO/viz/view_modes_csv.py" \
  "$NOVA_REPO/training_labels/tae_like_3new.csv" \
  --base_dir "$NOVA_DATA"
```

### `tae_like_2new.csv`

Reviewed two-shot component split from `tae_like_3new.csv`, excluding blocked
`nstxuG121123N75`. This list uses the full active training schema and has been
merged into `tae_like_train.csv`.

Current checked contents:
- 347 labeled modes
- labels: 14 `good`, 333 `bad`
- `nstxuG121123Q62`: 241 rows, 13 `good`, 228 `bad`
- `nstxuG142301Y93`: 106 rows, 1 `good`, 105 `bad`
- paths are relative to `$NOVA_DATA`
- no duplicate paths
- all paths resolve under `$NOVA_DATA`

## Refreshed / new co-worker labeled component lists

Two additional TAE-like review-stage lists were generated from the per-shot
`*_tae_eae_split/tae_like.csv` files in `$CFS/m314/nova2/data` and the
corresponding `*_mode_labels_clean.csv` hand labels:

- `tae_like_nstx_135388.csv`
- `tae_like_nstxuG121123J38.csv`

These lists were accepted for training and merged into `tae_like_train.csv` on
2026-07-06. They use relative `$NOVA_DATA` paths and the same full schema as
the active training list:
`path,validity,family,signed_delta,fraction_below_upper2,gap_region,error`.
The source split CSVs still contain Flux/DiTw absolute paths; the staged
review files in this directory do not.

Current checked contents after final manual review:
- `tae_like_nstx_135388.csv`: 344 TAE-like rows, 122 `good`, 222 `bad`
- `tae_like_nstxuG121123J38.csv`: 174 TAE-like rows, 6 `good`, 168 `bad`

Merge details:
- old `nstx_135388` rows removed from `tae_like_train.csv`: 380 rows, 185
  `good`, 195 `bad`
- refreshed `nstx_135388` rows added: 344 rows, 122 `good`, 222 `bad`
- new `nstxuG121123J38` rows added: 174 rows, 6 `good`, 168 `bad`

The continuum-crossing mismatch issue is considered resolved for modes
recalculated with the correct q profile.

Example review commands:

```bash
python "$NOVA_REPO/viz/view_modes_csv.py" \
  "$NOVA_REPO/training_labels/tae_like_nstx_135388.csv" \
  --base_dir "$NOVA_DATA"

python "$NOVA_REPO/viz/view_modes_csv.py" \
  "$NOVA_REPO/training_labels/tae_like_nstxuG121123J38.csv" \
  --base_dir "$NOVA_DATA"
```
