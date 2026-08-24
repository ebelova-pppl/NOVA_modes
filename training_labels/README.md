# Training Label Lists

This directory contains version-controlled mode-label CSV files used for
training, splitting, and auditing classifier datasets.

Paths in active training CSVs should be stored relative to `$NOVA_DATA`, for
example `nstx_120113/N5/egn05w.1234E+02`.

## Active rebuilt-database list

### `tae_like_v3.csv`

Versioned source list for the rebuilt canonical `data_mixed` database,
covering all 15 training shots. It contains six shots whose data and split
outputs were retained unchanged, two refreshed non-G shots, S31, H47, Y93,
and Q62 reviewed against their recalculated continua, the exact transferable
subset of the refreshed `nstxu_204202` mode set, and the fully re-reviewed
recalculated `nstx_141711` and K51 shots:

- `nstxuE202855A01t020`: 79 rows (50 `good`, 29 `bad`)
- `nstxuE204669M03t025`: 217 rows (82 `good`, 135 `bad`)
- `nstxuE205052A01t022`: 291 rows (57 `good`, 234 `bad`)
- `nstxuG121123B12`: 135 rows (19 `good`, 116 `bad`)
- `nstxuG121123J38`: 174 rows (7 `good`, 167 `bad`)
- `nstxuG142301W29`: 158 rows (7 `good`, 151 `bad`)
- `nstx_120113`: 174 rows (46 `good`, 128 `bad`)
- `nstx_135388`: 345 rows (133 `good`, 212 `bad`)
- `nstxuG133964S31`: 76 rows (0 `good`, 76 `bad`)
- `nstxuG142301H47`: 178 rows (12 `good`, 166 `bad`)
- `nstxuG142301Y93`: 113 rows (1 `good`, 112 `bad`)
- `nstxuG121123Q62`: 249 rows (16 `good`, 233 `bad`)
- `nstxu_204202`: 140 rows (62 `good`, 78 `bad`)
- `nstx_141711`: 158 rows (79 `good`, 79 `bad`)
- `nstxuG121123K51`: 152 rows (24 `good`, 128 `bad`)

The list has 2,639 rows: 595 `good` and 2,044 `bad`. The original six-shot
block retains its paths, validity labels, ordering, and split scalars from
`tae_like_v2_nonG.csv`. Thirty stale family fields inherited from v2 were
normalized without changing validity: 23 `bad,tae` rows became `bad,none`,
and 7 `good,none` rows became `good,tae`.

The 519 refreshed non-G rows use current scalars from the regenerated
`tae_like.csv` split manifests. Labels for 518 rows come from the active clean
human review. The newly admitted
`nstx_135388/N4/egn04w.1922E+03` mode retains its reviewed/preserved BAD label.
The five low-confidence continuum-refresh decisions recorded in
`tests/labels_audit/continuum_refresh_2026_08_23/nonG_suspect_label_changes.csv`
are included.

All 76 S31 rows use the current regenerated split scalars and the complete
human re-review in
`tests/labels_audit/continuum_refresh_2026_08_23/nstxuG133964S31_human_labels_clean.csv`.
The review labels every current S31 TAE-like mode BAD. The old v2 list had 74
S31 rows, also all BAD; v3 adds BAD coverage for the two formerly absent
paths without any label reversal.

All 178 H47 rows use the current regenerated split scalars and the finalized
human review in
`tests/labels_audit/continuum_refresh_2026_08_23/nstxuG142301H47_human_labels_clean.csv`.
The final review contains 12 GOOD and 166 BAD labels and includes the
post-comparison correction `N10/egn10w.1403E+02=GOOD`. Its three retained
old-BAD-to-new-GOOD disagreements are documented as extremum-localized modes.

All 113 Y93 rows use current regenerated split scalars and the finalized human
review in
`tests/labels_audit/continuum_refresh_2026_08_23/nstxuG142301Y93_human_labels_clean.csv`.
The final review contains one GOOD label and 112 BAD labels, retains
`N9/egn09w.1539E+02=GOOD`, and labels all seven continuum-driven additions BAD.

All 249 Q62 rows use current regenerated split scalars and the finalized human
review in
`tests/labels_audit/continuum_refresh_2026_08_23/nstxuG121123Q62_human_labels_clean.csv`.
The final review contains 16 GOOD and 233 BAD labels. Its 11 smooth modes with
small-r continuum crossings and no resonant-like spikes are included as
low-confidence GOOD; all eight continuum-driven additions are BAD.

All 140 current `nstxu_204202` TAE-like modes are byte-identical to their
frozen `data_mixed_2026_08_20` copies and inherit their exact v2 labels. The
merged block uses current split ordering and regenerated split scalars. There
are no genuinely new current modes to review. The 135 old-only paths absent
from the current mode tree remain quarantined and are not included in v3.

All 158 current `nstx_141711` TAE-like modes use the finalized whole-shot
review in
`tests/labels_audit/continuum_refresh_2026_08_23/nstx_141711_human_labels_final.csv`
and regenerated current split scalars. The block contains 79 GOOD and 79 BAD
labels and includes the explicit post-comparison decisions
`N7/egn07w.9318E+02=GOOD` and `N8/egn08w.1026E+03=BAD`. Four current N1 modes
without old v2 labels are BAD. The 102 old-only labels outside the current
TAE-like split remain quarantined and excluded.

All 152 current `nstxuG121123K51` TAE-like modes use the finalized whole-shot
review in
`tests/labels_audit/continuum_refresh_2026_08_23/nstxuG121123K51_human_labels_final.csv`
and regenerated current split scalars. The block contains 24 GOOD and 128 BAD
labels and includes the post-comparison `N9/egn09w.3938E+02=BAD` correction.
Thirty current modes without old v2 labels are included; all 86 old-only mode
files absent from the rebuilt tree remain quarantined and excluded.

Validation found exact split-manifest coverage for all six continuum-refreshed
shots, the transferred `nstxu_204202` subset, and the fully reviewed
`nstx_141711` and K51 shots, unique relative paths, allowed labels, no `skip`
or error rows, consistent family/validity values, and 2,639 mode files
resolving under `/p/hym/ebelova/NOVA/data_mixed`. This is the complete rebuilt
15-shot label set, including all audited G shots. The active
`tae_like_train.csv` is an exact copy of this versioned source.

The exact-label transfer component for `nstxu_204202` is retained in
`tests/labels_audit/continuum_refresh_2026_08_23/nstxu_204202_transferred_labels.csv`.
The 135 old labels whose mode paths are absent from the current tree are
retained separately in
`tests/labels_audit/continuum_refresh_2026_08_23/nstxu_204202_quarantined_old_labels.csv`
and must not be restored to the active training set.

## Active training list

### `tae_like_train.csv`

Canonical/default TAE-like good/bad training list for RF and CNN training. It
is an exact byte-for-byte copy of `tae_like_v3.csv`, with 2,639 rows: 595
`good` and 2,044 `bad`. NERSC and Flux path configs set both
`NOVA_TRAIN_CSV` and `NOVA_TRAIN_CSV_TAE` to this file. The source and active
copy have SHA-256
`7cf7b3cbf07a6af65311867bc109ac8783e50829f4d9655e33374890447ec0ea`.

The pre-promotion 2,903-row contents of this filename remain recoverable from
Git history; they were replaced intentionally when the completed v3 audit was
promoted.

## Preserved versioned lists

### `tae_like_v2_nonG.csv`

Preserved pre-v3 TAE-like good/bad training list from the period when the
NSTX-U G-shot review was incomplete.

Columns:
- `path`
- `validity`
- `family`
- `signed_delta`
- `fraction_below_upper2`
- `gap_region`
- `error`

Current checked contents:
- 2900 labeled modes
- labels: 593 `good`, 2307 `bad`
- 1635 reviewed non-G rows plus 1265 unchanged `nstxuG*` rows
- shots: `nstx_120113`, `nstx_135388`, `nstx_141711`, `nstxu_204202`,
  `nstxuE202855A01t020`, `nstxuE204669M03t025`, `nstxuE205052A01t022`,
  `nstxuG121123K51`, `nstxuG133964S31`, `nstxuG142301H47`,
  `nstxuG121123J38`, `nstxuG121123Q62`, `nstxuG142301Y93`,
  `nstxuG121123B12`, `nstxuG142301W29`

The list keeps the same columns as `tae_like_train.csv`, applies the
then-current cleaned human-review labels to the seven non-G shots, excludes
rows marked `skip` at the time of v2 creation, and copies all `nstxuG*` rows
from the then-current `tae_like_train.csv` unchanged. NERSC and Flux path
configs no longer point at this file. The active
`tests/labels_audit/labels_human_review_clean.csv` has since received five
2026-08-23 continuum-refresh decisions. Those changes are included in v3 and
were intentionally not backported into this versioned v2 file.

Three old-`good` non-G rows were excluded because the human review marked them
`skip` when v2 was created:
  - `nstx_120113/N6/egn06w.1418E+02`
  - `nstxuE205052A01t022/N10/egn10w.1302E+02`
  - `nstxuE205052A01t022/N9/egn09w.1506E+02`

The first row, `nstx_120113/N6/egn06w.1418E+02`, was reclassified GOOD with
low confidence during the 2026-08-23 continuum-refresh review. It remains
absent from preserved v2 and has been restored in v3.

Label flips relative to the pre-v2 `tae_like_train.csv` snapshot in Git
history:

| shot | good -> bad | bad -> good |
| --- | ---: | ---: |
| `nstxu_204202` | 11 | 0 |
| `nstx_120113` | 0 | 0 |
| `nstx_135388` | 1 | 10 |
| `nstx_141711` | 16 | 1 |
| `nstxuE202855A01t020` | 3 | 5 |
| `nstxuE204669M03t025` | 4 | 1 |
| `nstxuE205052A01t022` | 16 | 1 |

Totals: 51 old `good` rows became `bad`, 18 old `bad` rows became `good`, and
three old `good` rows were removed as `skip`. The copied G-shot rows have zero
label changes.

The active RF and raw-CNN checkpoints were retrained on the preceding
594-GOOD / 2306-BAD snapshot. The post-refit correction of
`nstxu_204202/N9/egn09w.3222E+02` from `good` to `bad` is present in this CSV
but is not yet reflected in those checkpoints.

## Previous / derived root lists

### Pre-promotion `tae_like_train.csv`

The previous canonical expanded 15-shot list had 2,903 rows: 629 `good` and
2,274 `bad`, with SHA-256
`2c6c1d7ebb1743a592b0590f089a610d962508ed1bd71e3778e6e679d2afc919`.
It is no longer a separate top-level file because `tae_like_train.csv` now
contains v3, but its exact contents remain available in Git history and the
database-freeze tag.

### `tae_like_train_7.csv`

Derived non-G / E-production comparison list created from
`tae_like_train.csv` by excluding all `nstxuG*` shots. This file is useful for
7-shot LOSO checks of the non-G / E-like production regime.

Current checked contents:
- 1638 labeled modes
- labels: 546 `good`, 1092 `bad`
- shots: `nstx_120113`, `nstx_135388`, `nstx_141711`, `nstxu_204202`,
  `nstxuE202855A01t020`, `nstxuE204669M03t025`, `nstxuE205052A01t022`

## Addition / component lists

Component and staged review lists live under `training_labels/additions/` so
the root of `training_labels/` stays limited to the active list, previous or
derived top-level lists, this README, and archive directories.

### `additions/tae_like_4old.csv`

Backup copy of the original four-shot TAE-like training list before the
six-shot merge.

### `additions/tae_like_6new.csv`

Reviewed six-shot NSTX-U TAE-like list that was appended to
`tae_like_train.csv`.

### `additions/tae_like_copy.csv`

Backup copy of the previous 2125-row / 10-shot active training list before the
2026-07-06 merges. It contains 678 `good` and 1447 `bad` rows.

### `additions/tae_like_nstxuG121123B12.csv`

Reviewed B12 TAE-like component merged into `tae_like_train.csv` on
2026-08-05. It uses relative `$NOVA_DATA` paths and the full active schema.

The complete per-shot review covers all 136 modes in the current B12
`tae_like.csv`: 19 `good`, 116 `bad`, and one `skip`
(`N7/egn07w.1888E+02`). Because `skip` modes are excluded from model training,
the component and canonical training lists contain 135 B12 rows: 19 `good`
and 116 `bad`. Validation found no duplicate paths or family/validity
mismatches, and every component path resolves under `$NOVA_DATA`.

The bare filename `tae_like.csv` is intentionally not used for the canonical
training list anymore, because `split_tae_eae.py` and `sort_shot_mixed.py`
write TAE-like output/audit files with that name in their own output
directories.

### `additions/tae_like_nstxuG142301W29.csv`

Reviewed W29 TAE-like component merged into `tae_like_train.csv` on
2026-08-05. It uses relative `$NOVA_DATA` paths and the full active schema.

The component covers all 158 modes in the W29 `tae_like.csv`: 7 `good` and
151 `bad`, with no `skip` decisions. Exact manifest coverage, unique paths,
allowed labels, family consistency, empty error fields, and file resolution
under `$NOVA_DATA` were verified before merging. The two difficult retained
physics modes, `N5/egn05w.2505E+02` and `N8/egn08w.2847E+02`, are both GOOD
with low review confidence.

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

- `additions/tae_like_6new.csv`

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
- cleaned staged list: 1040 rows, 249 `good`, 791 `bad`
- not-cleaned staged list: 1041 rows, with one duplicate mode
- all cleaned staged paths resolve to files under `$NOVA_DATA` by
  `shot/N/file` suffix
- the per-shot TAE/EAE split outputs contain 10 additional TAE-like modes that
  are not in the cleaned staged label list because they were marked `skip`
  during labeling; these are intentionally excluded from training

The shared metadata CSVs currently contain absolute source paths from the
labeling environment. The staged
`training_labels/additions/tae_like_6new.csv` file uses relative `$NOVA_DATA`
paths and is kept as the reviewed six-shot component list.

Example review command for one `N` directory:

```bash
python "$NOVA_REPO/scripts/label_modes_fast.py" \
  "$NOVA_DATA/nstxuE202855A01t020/N1" \
  --mode-list "$NOVA_REPO/training_labels/additions/tae_like_6new.csv" \
  --rf-model "$NOVA_REPO/models/nova_mode_classifier.joblib"
```

## Three-shot NSTX-U review list

Three additional NSTX-U G-case shots have a separate review-stage list:

- `additions/tae_like_3new.csv`

This original combined list is not merged into `tae_like_train.csv` as-is. It
is intentionally still blocked because it includes `nstxuG121123N75`, whose
modes still need recalculation with the corrected q profile. The already
reviewed `nstxuG121123Q62` and `nstxuG142301Y93` rows were split into
`additions/tae_like_2new.csv` and merged into the active training list. After
the N75 recalculation, review the affected labels again before creating a
replacement N75 component.

Current checked contents:
- 523 labeled modes
- labels: 13 `good`, 510 `bad`
- shots: `nstxuG121123Q62`, `nstxuG121123N75`, `nstxuG142301Y93`
- per-shot counts:
  - `nstxuG121123Q62`: 241 rows, 12 `good`, 229 `bad`
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
  "$NOVA_REPO/training_labels/additions/tae_like_3new.csv" \
  --base_dir "$NOVA_DATA"
```

### `additions/tae_like_2new.csv`

Reviewed two-shot component split from `additions/tae_like_3new.csv`,
excluding blocked `nstxuG121123N75`. This list uses the full active training
schema and has been merged into `tae_like_train.csv`.

Current checked contents:
- 347 labeled modes
- labels: 13 `good`, 334 `bad`
- `nstxuG121123Q62`: 241 rows, 12 `good`, 229 `bad`
- `nstxuG142301Y93`: 106 rows, 1 `good`, 105 `bad`
- paths are relative to `$NOVA_DATA`
- no duplicate paths
- all paths resolve under `$NOVA_DATA`

## Refreshed / new co-worker labeled component lists

Two additional TAE-like review-stage lists were generated from the per-shot
`*_tae_eae_split/tae_like.csv` files in `$CFS/m314/nova2/data` and the
corresponding `*_mode_labels_clean.csv` hand labels:

- `additions/tae_like_nstx_135388.csv`
- `additions/tae_like_nstxuG121123J38.csv`

These lists were accepted for training and merged into `tae_like_train.csv` on
2026-07-06. They use relative `$NOVA_DATA` paths and the same full schema as
the active training list:
`path,validity,family,signed_delta,fraction_below_upper2,gap_region,error`.
The source split CSVs still contain Flux/DiTw absolute paths; the staged
review files under `additions/` do not.

Current checked contents after final manual review:
- `additions/tae_like_nstx_135388.csv`: 344 TAE-like rows, 122 `good`, 222 `bad`
- `additions/tae_like_nstxuG121123J38.csv`: 174 TAE-like rows, 7 `good`, 167 `bad`

Merge details:
- old `nstx_135388` rows removed from `tae_like_train.csv`: 380 rows, 185
  `good`, 195 `bad`
- refreshed `nstx_135388` rows added: 344 rows, 122 `good`, 222 `bad`
- new `nstxuG121123J38` rows added: 174 rows, 7 `good`, 167 `bad`

The continuum-crossing mismatch issue is considered resolved for modes
recalculated with the correct q profile.

Example review commands:

```bash
python "$NOVA_REPO/viz/view_modes_csv.py" \
  "$NOVA_REPO/training_labels/additions/tae_like_nstx_135388.csv" \
  --base_dir "$NOVA_DATA"

python "$NOVA_REPO/viz/view_modes_csv.py" \
  "$NOVA_REPO/training_labels/additions/tae_like_nstxuG121123J38.csv" \
  --base_dir "$NOVA_DATA"
```
