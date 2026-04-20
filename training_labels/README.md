# Training Label Lists

This directory contains the main mode-label CSV files used for training,
splitting, and auditing classifier datasets.

Paths are normally stored relative to `$NOVA_DATA`, for example
`nstx_123456/N5/egn05w.1234E+02`.

## Original TAE training lists

### `train_master.csv`

Original TAE-only training list.

Columns:
- `path`
- `validity`

### `train_master_full_paths.csv`

Same list as `train_master.csv`, but with absolute mode-file paths.

### `train_tae.csv`

Extended version of the original TAE training list with family labels.

Columns:
- `path`
- `validity`
- `family`

## Mixed-mode master list

### `all_modes.csv`

Current active mixed TAE/EAE training list.

Columns:
- `path`
- `validity`
- `family`

Notes:
- includes a header row
- includes both TAE and EAE labels where known
- has been cleaned to remove known mistaken small-`n_m` mode files from the
  `nstx_135388` data directories

## TAE/EAE split outputs

These files are generated from `all_modes.csv` by
`scripts/split_tae_eae.py`.

### `all_modes_tae_eae_split.csv`

Full splitter output for all processed rows.

It preserves the original input columns and appends:
- `signed_delta`
- `fraction_below_upper2`
- `gap_region`
- `error`

This is the main audit table for checking how each mode was classified.

### `tae_like.csv`

TAE-side output from the splitter.

It includes:
- rows classified as `below_upper2`
- rows classified as `mixed`

`mixed` rows are kept on the TAE side on purpose, so marginal modes remain
available for downstream TAE training and review.

### `eae_like.csv`

EAE-side output from the splitter.

It contains rows classified as `above_upper2`.

### `mixed_tae_like.csv`

Subset of the full split table containing only rows with
`gap_region = mixed`.

Useful for manual inspection of marginal modes that were still routed into
`tae_like.csv`.

### `bad_tae_like.csv`

Subset of `tae_like.csv` with `validity = bad`.

Useful as the bad side of the TAE good/bad classifier training set.

### `tae_misplaced_in_eae_like.csv`

Audit list of rows labeled `family = tae` that still landed in
`eae_like.csv`.

This file may contain only the header row if no such cases remain.

## Cleanup and audit lists for mistaken small-`n_m` files

### `all_modes_clean_nstx135388_nhar9.csv`

Cleaned candidate list created while removing mistaken small-`n_m` files from
the `nstx_135388` dataset directories.

This file was used to update `all_modes.csv` after review.

### `removed_nstx135388_nhar9.csv`

Audit list of rows removed from `all_modes.csv` during that cleanup.

It records the removed path plus extra parsed metadata such as `nhar`, `nr`,
file size, and the removal reason.
