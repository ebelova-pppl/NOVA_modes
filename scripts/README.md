This document consolidates scripts related to various models and methods used in our project. Each section serves as a guide to the corresponding scripts, including their functionality and usage.

# Scripts README

CSV input note: the shared mode-list readers accept either plain data rows or
an optional header row. Recognized path headers are `path`, `filepath`, and
`mode_path`; recognized label headers are `label`, `class`, `target`,
`manual_label`, and `rf_label`. Blank lines and `#` comment lines are ignored.

## CNN model scripts

- `cnn_hybrid.py`
- `cnn_straightened.py`
- `cnn_raw.py`
- `cnn_classify.py`
- `cnn_raw_classify.py`

### Training

All CNN training scripts now default to the labeled list from `$NOVA_TRAIN_CSV`.
For portability, paths in `training_labels/train_master.csv` should be stored
relative to `$NOVA_DATA`, for example `nstx_120113/N5/egn05w.1234E+02`.

```bash
module load pytorch

python cnn_hybrid.py        # uses: nova_mode_loader, mode_transform.py, mode_features.py
python cnn_straightened.py  # uses: nova_mode_loader, mode_transform.py
python cnn_raw.py           # uses: nova_mode_loader
```

### Classification

```bash
python cnn_raw_classify.py /mode_file_path/
python cnn_classify.py --model models/nova_cnn_straightened.pt --path /mode_file_path/
python cnn_classify.py --model models/nova_cnn_hybrid.pt --path /mode_file_path/
python cnn_classify.py --model models/nova_cnn_hybrid.pt --csv training_labels/train_master.csv --out preds.csv
```

`cnn_classify.py` is the shared inference entry point for straightened and hybrid
CNN checkpoints. Older checkpoints that do not save preprocessing metadata fall
back to the legacy defaults and emit a warning so the behavior is explicit.

To handle the large variation in the number of poloidal harmonics in NOVA outputs, the CNN input was transformed to a straightened ridge representation. The dominant harmonic `m_c(r)` was estimated from a weighted mean of amplitude, and a small window `m_c(r) +/- M` was extracted, with `M ~ 8-12`.

Modified to **HybridCNN** = 2D mode `(m, r)` + 8 scalars.

---

## Random Forest classifier

- `rf_train_classify.py` — new name for RF script
- `legacy/nova_mode_classifier.py` — old name

### Training

To train the mode classifier, use the list of files in
`training_labels/train_master.csv` and run:

```bash
python rf_train_classify.py --train_csv training_labels/train_master.csv \
       --model_out nova_mode_classifier.joblib
```
Or, using env variables and running from $SCRATCH:
```bash
python $NOVA_REPO/scripts/rf_train_classify.py --train_csv $NOVA_TRAIN_CSV \
       --model_out nova_mode_classifier.joblib
```

`nova_mode_classifier.joblib` is a binary file that stores the trained ML model, i.e. a saved scikit-learn model (`StandardScaler + RandomForest`).

### Classification

To classify a mode, replace `/path_to_mode` with the file path:

```bash
python rf_train_classify.py --model_in nova_mode_classifier.joblib --classify /path_to_mode
```

### Continuum-aware features (optional)

The classifier can optionally compute continuum-related features using NOVA continuum data (`datcon` file). These features are used in addition to structural / roughness features.

#### Expected `datcon` location and naming

For a mode file located in a directory like:

```text
.../<shot>/N5/egn05w.XXXXE+YY
```

the code looks for a continuum file in the **same** directory, with the name:

- `datcon<n>` where `<n>` is the toroidal mode number inferred from the path, e.g. `datcon5`

#### What happens if `datcon` is missing?

If the continuum file is not found (or cannot be parsed), the code will:

1. Print a warning **once per directory** indicating continuum features are disabled, and
2. Fall back to structural-only features for modes in that directory.

This means the script will still work, but results may differ from continuum-aware runs.

Legacy `datcon<N>` files sometimes use a tail sentinel value near `1000.000`
instead of `NaN`. The shared datcon loader now treats values `> 999` as missing
so those edge points do not contaminate continuum features or TAE/EAE splitting.

#### Continuum-derived features used

When available, the following scalars are appended to the feature vector:

- `delta2_eff`: mode-weighted effective squared distance (`delta_omega^2`) from the continuum
- `r_star`: radius of closest continuum approach / crossing
- `S`: normalized separation between mode and `r_star`, `S = (rad_loc - r_star) / rad_width`
- `W_star`: mode amplitude² at `r = r_star`

---

## Sorting TAEs vs EAEs from mixed data: `split_tae_eae.py`

Split a CSV list of modes into TAE-like vs EAE-like groups using the upper TAE
gap boundary from the local `datcon<N>` file.

It reuses the standard NOVA mode loader plus the existing continuum-file lookup
logic. For each mode it computes:

- `dist = sqrt(upper2) - omega`
- `signed_delta`: weighted mean of `dist`, normalized by the weighted RMS of `dist`
- `fraction_below_upper2`: weighted fraction of mode energy where `dist > 0`

Default rule:

- `fraction_below_upper2 > 0.5` → `below_upper2` (TAE-like)
- `fraction_below_upper2 < 0.4` and `signed_delta < -0.1` → `above_upper2` (EAE-like)
- otherwise → `mixed`

By default, `mixed` rows are written into the TAE-like output CSV so marginal
modes stay on the TAE side, but the full CSV still records `gap_region=mixed`
for inspection.

### Usage

```bash
python split_tae_eae.py \
  --input_csv all_modes.csv \
  --out_below_csv tae_like.csv \
  --out_above_csv eae_like.csv
```

The script preserves original CSV columns when present, appends the new split
scalars, and also writes a full CSV with errors and skipped rows. Modes with
missing / unreadable `datcon` files are written with `gap_region=error` and are
excluded from the two split output lists.

For headerless three-column inputs like `path,validity,family`, the script
infers those column names so the output CSVs and terminal summary include the
family sanity check automatically.

---

##  `label_modes_fast.py`

Script to go through all modes in a directory and sort them as `good` / `bad` / `skip`.

Label `skip` means that the mode will not be used for training AI models.

It saves labeled modes in `mode_labels.csv` and `mode_labels_clean.csv`.

### Usage

```bash
python label_modes_fast.py dir_name
```

where `dir_name` is something like `nstx_20113/N1`.

### Controls

- Press `g` to save as good
- Press `b` to save as bad
- Press `s` to save as skip
- Press `u` to undo and go back to the previous mode
- Press `q` to quit

The script will restart from the first unsorted mode and append to the existing list.

It is linked to the RF classifier, and outputs RF mode classification (`good` / `bad`) together with the score, i.e. probability `P > 0.5` for good.

If a `datcon#` file is located in the same directory, the script will mark the closest continuum crossing, `R*`, and will add an extra plot of the continuum gap + mode frequency.

---

## `view_modes_csv.py`

Script to plot mode structures from a `name.csv` list.

Makes the same plots as `label_modes_fast.py` plus contour plots of `mode(r, m)`.

To see all options, run:

```bash
python view_modes_csv.py -h
```

---

## Legacy RF shot sorter

- `legacy/rf_sort_shot.py` — old version; does not check close-frequency modes  
  Use `sort_shot.py` instead.

This script walks a shot directory like:

```text
.../nstx_123456/
```

It:

- finds `N1 ... N10` subdirectories (or whatever exists),
- scans all files matching `egn*` in each `N#`,
- runs the existing RF `joblib` classifier on each mode file,
- writes a per-shot CSV list: `path,label,p_good`,
- optionally moves bad modes into `N#/out/` (creating it if needed).

**Note:** threshold `= 0.5` means that the mode is bad if `p_good < 0.5`.

### Preview run

```bash
python rf_sort_shot.py /global/cfs/cdirs/m314/nova/nstx_123456 \
  --model nova_mode_classifier.joblib \
  --threshold 0.5 \
  --move_bad --dry_run
```

### Actual move

```bash
python rf_sort_shot.py /global/cfs/cdirs/m314/nova/nstx_123456 \
  --model nova_mode_classifier.joblib \
  --threshold 0.5 \
  --move_bad
```

For help, run:

```bash
python rf_sort_shot.py -h
```

---

## `sort_shot.py`

New version, which checks close-frequency clusters and writes `cluster_report` suggesting `KEEP` / `DROP`.

This script does the same as `rf_sort_shot.py` for sorting `GOOD` / `BAD`
modes, and in addition checks `GOOD` modes for frequency spacing. It can use
either the RF `.joblib` model or a straightened / hybrid CNN `.pt` checkpoint.

By default, it writes `cluster_report.txt` and `cluster.csv` files in the shot directory.

### Usage

Without moving bad modes out:

```bash
python sort_shot.py --model nova_mode_classifier.joblib \
  --rel_freq_tol 0.02 shot_dir

python sort_shot.py --model models/nova_cnn_straightened.pt \
  --rel_freq_tol 0.02 shot_dir
```

Or, to move bad modes into `/OUT/`:

```bash
python sort_shot.py --model nova_mode_classifier.joblib \
  --move_bad --rel_freq_tol 0.02 shot_dir
```

where:

- `rel_freq_tol` is the minimum allowed relative frequency spacing
- `shot_dir` is the shot directory

It will not actually move closely spaced modes, but it will list them in `cluster.csv` and write `cluster_report.txt` suggesting which modes should be kept / dropped.

For help, run:

```bash
python sort_shot.py -h
```

---

## `utils/merge_lists.py`

Merges multiple training CSV file lists into a single master list, fixes relative paths by prepending shot-specific base directories, removes duplicates, and keeps only `good` / `bad` labels (newer labels override older ones).

### Example

To create a common list from separate shots:

```bash
python merge_lists.py train_master.csv \
  old_train_list.csv \
  nstx_120113_labels.csv@/global/cfs/cdirs/m314/nova/nstx_120113 \
  nstx_135388_labels.csv@/global/cfs/cdirs/m314/nova/nstx_135388 \
  nstx_141711_labels.csv@/global/cfs/cdirs/m314/nova/nstx_141711
```

Here `old_train_list.csv` is from one NSTX-U shot.

To check the number of `good` / `bad` labels in the CSV list, use:

```bash
awk -F, '{print $2}' train_master.csv | sort | uniq -c
```

---

## `rf_oof_check.py`

This script:

- reads `training_labels/train_master.csv` (`path,label`, with or without a header row)
- loads each mode + extra scalars (`omega`, `gamma_d`, `ntor`)
- builds `X` using `compute_features_for_mode(mode, extra_info=...)`
- runs OOF using `StratifiedKFold`

It writes:

- full OOF table: `path,manual_label,p_good_oof,oof_pred`
- a `suspects` file ranked by confidence: only strong disagreements  
  (default: good but `p < 0.2`; bad but `p > 0.8`)

It also prints a confusion matrix based on OOF predictions at threshold `0.5`.

### Usage

```bash
python rf_oof_check.py training_labels/train_master.csv \
  --model_in nova_mode_classifier.joblib \
  --out_oof oof_table.csv \
  --out_suspects oof_suspects.csv \
  --thr_low 0.2 --thr_high 0.8
```

For help, run:

```bash
python rf_oof_check.py -h
```

**Note:** re-run `rf_train_classify.py` after this.

---

## `utils/find_rf_disagreements.py`

### Usage

```bash
python find_rf_disagreements.py \
  training_labels/train_master.csv \
  nova_mode_classifier.joblib \
  rf_vs_manual_disagreements.csv
```

Compares manually sorted labels against Random Forest predictions and outputs only modes where RF and manual labels disagree, including RF confidence (`p_good`). It is used to identify candidates for targeted label re-checking.

To list modes with large disagreements from `rf_vs_manual_disagreements.csv`:

```bash
awk -F, 'NR>1 && ($4<0.2 || $4>0.8)' rf_vs_manual_disagreements.csv
```

Saved result: `re-check_list.csv`

---

## Legacy utility

- `legacy/read_nova.py`

Reads NOVA output file and makes plots. It also has comments describing the data structure in the NOVA output file.
