This document consolidates scripts related to various models and methods used in our project. Each section serves as a guide to the corresponding scripts, including their functionality and usage.

# Scripts README

CSV input note: the shared mode-list readers accept either plain data rows or
an optional header row. Recognized path headers are `path`, `filepath`, and
`mode_path`; recognized label headers are `label`, `validity`, `class`, `target`,
`manual_label`, and `rf_label`. Blank lines and `#` comment lines are ignored.

## CNN model scripts

- `cnn_hybrid.py`
- `cnn_straightened.py`
- `cnn_raw.py`
- `cnn_classify.py`
- `cnn_raw_classify.py`

### Training

All CNN training scripts default to the labeled list from `$NOVA_TRAIN_CSV`.
For portability, paths in training CSVs should be stored relative to
`$NOVA_DATA`, for example `nstx_120113/N5/egn05w.1234E+02`.

The current active good/bad training list is the expanded
`training_labels/tae_like_train.csv`, which combines the original four-shot
TAE-like set, the reviewed six-shot NSTX-U TAE-like set, refreshed
`nstx_135388` labels, the new `nstxuG121123J38` labels, and the reviewed
`nstxuG121123Q62` / `nstxuG142301Y93` labels. It currently has 2610 rows
across 13 shots. Older four-shot TAE-only and mixed TAE/EAE lists are archived under
`training_labels/old_4shots_tae_only_labels/` and
`training_labels/old_4shots_mixed_labels/`.

```bash
module load pytorch

python cnn_raw.py \
  --train_csv training_labels/tae_like_train.csv \
  --data_dir /path/to/nova/data \
  --refit_full_before_save \
  --model_out models/nova_cnn_raw.pt

python cnn_straightened.py \
  --train_csv training_labels/tae_like_train.csv \
  --data_dir /path/to/nova/data \
  --refit_full_before_save \
  --model_out models/nova_cnn_straightened.pt

python cnn_hybrid.py \
  --train_csv training_labels/tae_like_train.csv \
  --data_dir /path/to/nova/data \
  --refit_full_before_save \
  --model_out models/nova_cnn_hybrid.pt
```

`cnn_raw.py`, `cnn_straightened.py`, and `cnn_hybrid.py` have command-line
interfaces; run each script with `-h` for all training, data-path, and
preprocessing options. The raw CNN resamples the radial grid to `--R_target`
before padding/cropping the raw harmonic axis to `--M_target`.
When `--data_dir` is provided, relative mode paths in the training CSV are
resolved relative to that directory instead of requiring `$NOVA_DATA`.
`cnn_raw.py` uses one shared OneCycleLR plus gradient-clipping recipe for both
split training and the optional production full-data refit. The raw CNN
default `--lr=0.02` is the OneCycle peak LR, chosen from a small sweep because
it reduced false negatives for GOOD modes compared with `0.01`; this is
preferred for NOVA-C follow-up, where keeping a possibly unstable mode is more
important than minimizing false positives.

For imbalanced or collapse-prone LOSO subsets, `cnn_raw.py` also accepts
`--pos_weight`. This is the PyTorch binary-loss weight for the positive class,
where positive means `good`. Use `--pos_weight auto` to compute
`n_bad/n_good` from the current training labels, or pass a positive number to
force a value. The default is unweighted loss.

By default, the CNN trainers use a stratified train split, evaluate on the
held-out split, and save the best held-out checkpoint. For production sorting
or apples-to-apples checks against the RF model, pass
`--refit_full_before_save`: `cnn_raw.py` first reports metrics from the best
held-out checkpoint, then trains a fresh production model on the full labeled
CSV for the configured `--epochs` using the same recipe. The default cycle
starts at `--lr / 20`, reaches `--lr` during the first 10% of batch steps, and
cosine-anneals to one hundredth of the initial LR. Gradient norm is clipped to
`1.0`. Configure these with `--onecycle_div_factor`,
`--onecycle_final_div_factor`, `--onecycle_pct_start`, and
`--grad_clip_norm`; use `--grad_clip_norm none` (also `off` or `0`) only for
controlled ablations. Checkpoints record the recipe, split metrics, and saved
training scope.

To expose quiet prediction collapse, `cnn_raw.py` reports the predicted GOOD
fraction, true GOOD fraction, mean/standard deviation of `p_good`, and its
range every five epochs. Starting at epoch 5, it prints a warning for
near-all-BAD predictions, near-all-GOOD predictions, or nearly constant
probabilities. The full-data refit is checked with a deterministic evaluation
loader, and its final prediction-health values are stored in the checkpoint
under `final_prediction_health`.

All three CNN training scripts seed Python, NumPy, and PyTorch from their seed
configuration so training runs are reproducible by default.

The CNN trainers print the selected Torch device, visible CUDA devices, and
free/total GPU memory before training. All three CNN trainers accept
`--device` and honor `NOVA_TORCH_DEVICE`, for example:

```bash
export NOVA_TORCH_DEVICE=cuda
python "$NOVA_REPO/scripts/cnn_raw.py" --batch_size 8
```

Inside a Perlmutter interactive allocation, launch the Python process with
`srun` so it runs on the allocated GPU node rather than on the login shell:

```bash
salloc --nodes 1 --qos interactive --time 1:00:00 --constraint gpu --gpus 1 --account m314_g
srun --nodes 1 --ntasks 1 --cpus-per-task 1 --gpus-per-task 1 python -u "$NOVA_REPO/scripts/cnn_raw.py" --batch_size 8
```

If CUDA reports out-of-memory for these small CNNs, first check the printed
free/total memory and try `--batch_size 8` or `--batch_size 4`. To diagnose
environment issues without using GPU memory, run raw CNN with `--device cpu` or
set `NOVA_TORCH_DEVICE=cpu` for the older trainers.

After sourcing `configs/paths/nova_paths.nersc.sh`, `nova_gpu_smoke` runs a
small Torch CUDA allocation through `srun` and prints timing for device report,
first tensor allocation, matmul, and CPU copy. `nova_run_cnn_raw --batch_size 8`
runs the raw CNN through the same Slurm launch path. The helpers default to
`NOVA_CPUS_PER_TASK=1`; if you set a larger value, request matching CPUs in the
`salloc` command.

On PPPL Flux with the default `tcsh` shell, source
`configs/paths/nova_paths.flux.csh`. The Flux config keeps the environment
minimal: it resolves `NOVA_REPO` from the current Git checkout, sets
`NOVA_MODELS=$NOVA_REPO/models`, sets `NOVA_TRAIN_CSV`, defaults
`NOVA_TORCH_DEVICE=cpu`, and provides CPU helpers. It does not set a default
`NOVA_DATA`; pass absolute mode/shot paths or set `NOVA_DATA` yourself for
training and inspection workflows that use relative CSV paths.

```tcsh
module load anaconda3
source `conda info --base`/etc/profile.d/conda.csh
setenv CONDA_PKGS_DIRS /p/hym/conda_pkgs
conda activate /p/hym/conda_envs/nova-perlmutter
cd /path/to/your/NOVA_modes
source configs/paths/nova_paths.flux.csh
nova_cpu_smoke
nova_run_cnn_raw --batch_size 8 --cache_data
```

Bash users should source `$(conda info --base)/etc/profile.d/conda.sh` before
`conda activate`, then source `configs/paths/nova_paths.flux.sh` instead.

Flux portability check: with matching package versions in the `/p/hym` conda
environment, RF inference and all three Perlmutter-trained CNN checkpoints
(`cnn_raw`, `cnn_straightened`, `cnn_hybrid`) produced identical outputs on
Flux and Perlmutter for the checked modes.

The Flux configs also redirect cache and user-level Python paths into `/p/hym`
(`XDG_*`, `PIP_CACHE_DIR`, `MPLCONFIGDIR`, and `PYTHONUSERBASE`) so package
installs and generated cache files do not refill the small home directory.

For older Perlmutter-trained CNN checkpoints that do not contain
`model_type`/preprocessing metadata, `cnn_classify.py` can infer raw,
straightened, or hybrid from filenames containing `raw`, `straightened`, or
`hybrid`. If the filename is generic, pass the kind explicitly:

```bash
python "$NOVA_REPO/scripts/cnn_classify.py" \
  --model /path/to/checkpoint.pt \
  --model_kind cnn_raw \
  --path /path/to/mode
```

If raw CNN training is slow because the shared filesystem is lagging, use
`--cache_data` to preprocess the train/test tensors once and keep them in RAM:

```bash
nova_run_cnn_raw --batch_size 8 --cache_data
```

Current 13-shot TAE-like raw-CNN retraining check on
`training_labels/tae_like_train.csv`:

- `cnn_raw.py`: accuracy=`0.954`, CM=`[[394, 6], [18, 103]]`, GOOD
  precision/recall/F1=`0.945 / 0.851 / 0.896`

Previous expanded 10-shot TAE-like raw-CNN retraining check:

- `cnn_raw.py`: accuracy=`0.9693`, CM=`[[290, 5], [8, 121]]`, GOOD
  precision/recall/F1=`0.9603 / 0.9380 / 0.9490`
- previous production refit: all 2,125 labels, 80 OneCycleLR epochs, final loss
  `0.0008`

Targeted LOSO check for held-out `nstxuE205052A01t022` with OneCycleLR and
gradient clipping in both split training and full-data refit:

- nine-shot training list: `outputs/loso_10/folds/nstxuE205052A01t022/train.csv`
- internal split best accuracy: `0.9617`, CM `[[245, 6], [8, 107]]`
- production full-refit CNN CM on the held-out shot:
  `[[191, 28], [1, 73]]`
- held-out-shot accuracy: `0.9010`
- GOOD precision/recall/F1: `0.7228 / 0.9865 / 0.8343`
- output: `outputs/loso_onecycle_both_nstxuE205052A01t022/`

Earlier controlled ablations on this fold established that clipping prevents
the full-refit collapse, while OneCycle improves precision compared with
constant LR. The symmetric recipe strongly favors GOOD recall on this shot
but produces more false positives than the earlier asymmetric experiment.
The complete LOSO result is recorded below.

Completed symmetric-recipe 10-shot LOSO result:

- output: `outputs/loso_10_onecycle_both/`
- CNN CM: `[[1402, 74], [67, 582]]`
- CNN accuracy: `0.9336`
- CNN GOOD precision/recall/F1: `0.8872 / 0.8968 / 0.8920`
- combined-policy CM: `[[1418, 58], [86, 563]]`
- combined-policy accuracy: `0.9322`
- combined-policy GOOD precision/recall/F1:
  `0.9066 / 0.8675 / 0.8866`

Compared with the previous raw-CNN LOSO run, false negatives decreased from
140 to 67 while false positives increased only from 71 to 74. All 10
full-data refits completed 80 epochs without collapse. CNN is now the strongest
aggregate LOSO model by accuracy, GOOD recall, and GOOD F1. The NSTX-U G-case
folds remain the weak group: aggregate CNN GOOD recall is `0.425` there,
compared with `0.933` for the original NSTX shots and `0.942` for NSTX-U
E-case shots.

Previous four-shot TAE-like retraining checks used threshold 0.5 for CNN
evaluation. Those checkpoints are archived under `models/old_4shots_models/`:

- `cnn_raw.py`: best accuracy=0.96, CM=[[126 5][4 81]]
- `cnn_straightened.py`: best accuracy=0.95, CM=[[126 5][6 79]]
- `cnn_hybrid.py`: best accuracy=0.96, CM=[[129 2][6 79]]

### Classification

```bash
python cnn_classify.py --model models/nova_cnn_raw.pt --path /mode_file_path/
python cnn_classify.py --model models/nova_cnn_straightened.pt --path /mode_file_path/
python cnn_classify.py --model models/nova_cnn_hybrid.pt --path /mode_file_path/
python cnn_classify.py --model models/nova_cnn_hybrid.pt --csv training_labels/tae_like_train.csv --out preds.csv
```
or using env and running from $SCRATCH or other dir
```bash
python $NOVA_REPO/scripts/cnn_classify.py --model $NOVA_REPO/models/nova_cnn_raw.pt --path $NOVA_DATA/nstx_120113/N5/egn05w.6606E+02
```

`cnn_classify.py` is the shared inference entry point for raw, straightened, and
hybrid CNN checkpoints. Older straightened/hybrid checkpoints that do not save
preprocessing metadata fall back to the legacy defaults and emit a warning so
the behavior is explicit. Older raw checkpoints can be loaded with
`--model_kind cnn_raw` when auto-detection is ambiguous.

To handle the large variation in the number of poloidal harmonics in NOVA outputs, the CNN input was transformed to a straightened ridge representation. The dominant harmonic `m_c(r)` was estimated from a weighted mean of amplitude, and a small window `m_c(r) +/- M` was extracted, with `M ~ 8-12`.

Modified to **HybridCNN** = 2D mode `(m, r)` + 8 scalars.

---

## Random Forest classifier

- `rf_train_classify.py` — new name for RF script
- `legacy/nova_mode_classifier.py` — old name

### Training

To train the mode classifier, use the relevant labeled list. For example, to
train on the TAE-like side of the mixed data:

```bash
python rf_train_classify.py --train_csv training_labels/tae_like_train.csv \
       --model_out nova_mode_classifier.joblib
```
Or, using env variables and running from $SCRATCH:
```bash
python $NOVA_REPO/scripts/rf_train_classify.py --train_csv $NOVA_TRAIN_CSV \
       --model_out nova_mode_classifier.joblib
```

`nova_mode_classifier.joblib` is a binary file that stores the trained ML model, i.e. a saved scikit-learn model (`StandardScaler + RandomForest`).

After merging the reviewed six-shot NSTX-U labels, retrain RF on the expanded
TAE-like list:

```bash
python "$NOVA_REPO/scripts/rf_train_classify.py" \
  --train_csv "$NOVA_REPO/training_labels/tae_like_train.csv" \
  --model_out "$NOVA_REPO/models/nova_mode_classifier.joblib"
```

Current 13-shot RF OOF check after merging `tae_like_2new.csv`:

- CM=`[[1967, 37], [91, 515]]`
- accuracy=`0.951`
- GOOD precision/recall/F1=`0.933 / 0.850 / 0.889`
- output: `outputs/rf_oof_13shots/`

The active expanded-set RF checkpoint is
`models/nova_mode_classifier.joblib`. Previous four-shot RF checkpoints are
archived under `models/old_4shots_models/`.

The component six-shot list is `training_labels/additions/tae_like_6new.csv`, with
relative `$NOVA_DATA` paths and the same full schema as `tae_like_train.csv`.
For interactive review, `label_modes_fast.py` can use it with `--mode-list`:

```bash
python "$NOVA_REPO/scripts/label_modes_fast.py" \
  "$NOVA_DATA/nstxuE202855A01t020/N1" \
  --mode-list "$NOVA_REPO/training_labels/additions/tae_like_6new.csv" \
  --rf-model "$NOVA_REPO/models/nova_mode_classifier.joblib"
```

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
- `W_star_max`: largest peak-normalized radial mode energy at any interpolated
  lower/upper continuum-boundary crossing

The current production RF schema has 22 features. It removes the raw `omega`
feature used by the previous checkpoint and adds `W_star_max`. Missing or
invalid continuum data use the existing safe fallback, with
`W_star_max = 0`.

#### Experimental boundary-crossing RF features

The default RF schema used by `models/nova_mode_classifier.joblib` already
includes `W_star_max`. For experiments only,
`rf_train_classify.py --crossing-features` appends the other six crossing
features, producing a 28-feature schema:

- `n_cross`
- `r_star_max`, `W_star_sum`
- `r_star_high_shear`, `W_star_high_shear`, `W_star_high_shear_sum`

The shear-weighted quantities use
`max(r_cross - r_shear0, 0)^2`, with `--r_shear0 0.2` by default. Experimental
models remain ordinary sklearn pipeline `.joblib` files, but include feature
schema metadata. They are not yet supported by `sort_shot.py`,
`sort_shot_mixed.py`, or the interactive labeling workflow.

Example training command:

```bash
python "$NOVA_REPO/scripts/rf_train_classify.py" \
  --train_csv "$NOVA_REPO/training_labels/tae_like_train.csv" \
  --crossing-features \
  --model_out "$NOVA_REPO/models/nova_mode_classifier_crossing.joblib"
```

The bundle defaults to
`models/nova_mode_classifier_crossing_bundle.joblib`. The trainer refuses to
overwrite the active legacy checkpoint with an experimental model.

Run an apples-to-apples OOF experiment with the same feature option:

```bash
python "$NOVA_REPO/scripts/rf_oof_check.py" \
  "$NOVA_REPO/training_labels/tae_like_train.csv" \
  --model_in "$NOVA_REPO/models/nova_mode_classifier.joblib" \
  --crossing-features \
  --out_oof rf_crossing_oof.csv \
  --out_suspects rf_crossing_suspects.csv
```

Here `--model_in` supplies the RF pipeline/hyperparameters as the OOF template;
the folds are fitted on the selected 28-feature schema. To classify one mode
with an experimental checkpoint, pass `--crossing-features` together with
`--model_in` and `--classify`.

The full crossing schema and simpler outer-radius/high-shear variants did not
improve OOF performance. They were strongly correlated with `W_star_max`. The
promoted production configuration is the previous feature set minus `omega`,
plus `W_star_max`.

Synthetic crossing and schema checks use only standard-library `unittest`:

```bash
PYTHONPATH="$NOVA_REPO/src" python -m unittest discover \
  -s "$NOVA_REPO/tests" -v
```

---

## Sorting TAEs vs EAEs from mixed data: `split_tae_eae.py`

Split a shot directory or CSV list of modes into TAE-like vs EAE-like groups
using the upper TAE gap boundary from the local `datcon<N>` file.

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

For a new shot directory, the simplest workflow is:

```bash
python split_tae_eae.py \
  --shot_dir /path/to/nstx_135388 \
  --out_dir split_outputs/nstx_135388
```

This scans `N1` through `N10` for `egn*` files and writes:

- `all_modes.csv` — generated list of all scanned mode files
- `tae_like.csv` — strict TAE-like plus mixed modes
- `eae_like.csv` — EAE-like modes
- `all_modes_tae_eae_split.csv` — full audit table with split scalars and errors

For shot-directory input, the generated path column uses absolute paths so the
split outputs can be used directly by downstream scripts.

Use `--n_min`, `--n_max`, or `--pattern` for shots with a different directory
range or file naming pattern. If `--out_dir` is omitted for `--shot_dir`, the
script writes to `./<shot>_tae_eae_split`.

For an existing CSV list:

```bash
python split_tae_eae.py \
  --input_csv training_labels/old_4shots_mixed_labels/all_modes.csv \
  --out_below_csv split_outputs/tae_like.csv \
  --out_above_csv split_outputs/eae_like.csv
```

The script preserves original CSV columns when present, appends the new split
scalars rounded to four decimal places, and also writes a full CSV with errors
and skipped rows. Modes with
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

where `dir_name` is something like `nstx_20113/N1`. Relative directories are
resolved under `--data_dir` or `$NOVA_DATA`; absolute directories are used
directly.

The mode-structure panel plots signed `xi_m(r)` profiles by default, matching
`viz/view_modes_csv.py`. Use `--abs` only when you want the older
`|xi_m(r)|` view. The panel plots all poloidal harmonics stored in each mode
file by default, and its title reports `plotted/total`. To reduce visual
crowding, use `--max-harmonics N`; this keeps the strongest `N` harmonics
ranked by `max_r |xi_m|`:

```bash
python label_modes_fast.py nstx_135388/N5 \
  --data_dir "$NOVA_DATA" \
  --max-harmonics 80
```

```bash
python label_modes_fast.py nstx_135388/N5 \
  --data_dir "$NOVA_DATA" \
  --abs
```

For another device or a local data copy where the NSTX-U RF model is not
applicable, disable RF guidance:

```bash
python label_modes_fast.py shot_or_run/N1 \
  --data_dir /path/to/nova/data \
  --csv_out labels_new_device.csv \
  --no-rf
```

To keep RF guidance, provide a compatible RF model:

```bash
python label_modes_fast.py nstx_120113/N5 \
  --data_dir "$NOVA_DATA" \
  --rf-model nova_mode_classifier.joblib
```

To label only one mode family from a mixed directory, pass a split mode list.
The script still scans `mode_dir`, but only presents files whose resolved path
or shot/N/file suffix appears in the CSV:

```bash
python label_modes_fast.py nstx_120113/N5 \
  --data_dir "$NOVA_DATA" \
  --mode-list training_labels/tae_like_train.csv \
  --csv_out labels_tae_like.csv
```

Use `--pattern` if mode files are not named `egn*`.

### Controls

- Press `g` to save as good
- Press `b` to save as bad
- Press `s` to save as skip
- Press `u` to undo and go back to the previous mode
- Press `q` to quit

The script will restart from the first unsorted mode and append to the existing list.

RF classifier guidance is optional. By default the script tries to load
`nova_mode_classifier.joblib`; pass `--no-rf` to skip RF evaluation, or
`--rf-model` to select a different compatible model. If RF is enabled but the
model cannot be loaded, the script prints a warning and continues without RF.

If a `datcon#` file is located in the same directory, the script marks both
the legacy closest-approach location `R*` and the maximum-amplitude
continuum-boundary crossing `R*max`. It also shows the continuum gap and mode
frequency.

---

## `view_modes_csv.py`

Script to plot mode structures from a `name.csv` list.

Makes the same plots as `label_modes_fast.py` plus contour plots of `mode(r, m)`.
Relative paths in the CSV are resolved under `--base_dir`, which defaults to
`$NOVA_DATA`.

For the staged six-shot NSTX-U label list:

```bash
python "$NOVA_REPO/viz/view_modes_csv.py" \
  "$NOVA_REPO/training_labels/additions/tae_like_6new.csv" \
  --base_dir "$NOVA_DATA"
```

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
either the RF `.joblib` model or a raw / straightened / hybrid CNN `.pt` checkpoint.

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

## `sort_shot_mixed.py`

Shot-level workflow for mixed TAE/EAE runs. It does not move files. Instead, it:

- validates mode files and required continuum inputs,
- routes valid modes into `tae_like`, `mixed`, and `eae_like` gap regions,
- sends TAE-like plus mixed modes through both the RF classifier and a CNN checkpoint,
- combines RF/CNN probabilities with the current gold/silver/borderline policy,
- reuses the close-frequency duplicate removal from `sort_shot.py`, and
- writes CSV audit tables, shot summaries, a frequency-cluster report, and
  optional diagnostic plots.

Current operational note: this is the main large-shot sorting path for the
active models. The top-level RF and raw-CNN checkpoints are trained on the
current 13-shot TAE-like list. The default RF-leaning fusion policy was chosen
from four-shot LOSO checks. The symmetric-recipe 10-shot LOSO check in
`outputs/loso_10_onecycle_both/` makes raw CNN strongest overall. The current
combined policy still has better GOOD recall on the sparse NSTX-U G-case group,
so an updated 13-shot LOSO check should guide any fusion-threshold retuning.

Close-frequency duplicate removal enforces the frequency threshold pairwise
against the candidate representative before structure metrics can merge two
modes. This avoids chained clusters where several adjacent modes are close but
the first and last mode are separated by more than `--rel_freq_tol`.

The main outputs are:

- `good_tae_unchecked.csv`
- `good_tae_final.csv`
- `bad_tae_like.csv`
- `flagged_tae_like.csv`
- `eae_like.csv`
- `rejected_modes.csv`
- `shot_summary.csv`

It also writes `all_modes_scored.csv`, `tae_like_all.csv`,
`shot_summary_wide.csv`, `shot_summary_by_n.csv`,
`frequency_cluster_report.txt`, and
`frequency_clusters.csv` for auditability.

When `--label_csv` is provided for a labeled validation shot, it also writes:

- `model_evaluation_report.txt` — RF-only, CNN-only, and combined-policy
  confusion matrices plus classification reports
- `model_evaluation_summary.csv` — compact metrics table
- `model_evaluation_rows.csv` — per-mode true/predicted labels and scores

`shot_summary.csv` is written as a human-readable two-column key/value file.
`shot_summary_wide.csv` keeps the same one-row table layout as
`shot_summary_by_n.csv` for scripts and spreadsheet workflows. In all summary
outputs, `n_good_before_clustering` is the RF/CNN-fused GOOD count before
duplicate removal, while `n_final_good` is the post-clustering count written to
`good_tae_final.csv`.

`flagged_tae_like.csv` is an overlapping QC list rather than a mutually
exclusive class: it contains scored TAE-side modes that are borderline or show
RF/CNN disagreement, so they may also appear in either `good_tae_unchecked.csv`
or `bad_tae_like.csv`.

The current (optimized) default RF/CNN fusion policy is RF-leaning with a high-confidence CNN
rescue:

```text
gold_good:          p_rf_good >= 0.7 and p_cnn_good >= 0.6
silver_good:        p_rf_good >= 0.5 and p_cnn_good >= 0.5
flagged_cnn_rescue: p_rf_good >= 0.4 and p_cnn_good >= 0.9
gold_bad:           p_rf_good <  0.2 and p_cnn_good <  0.2
silver_bad:         p_rf_good <  0.4 and p_cnn_good <  0.4
flagged_rf_only_good:
                     p_rf_good >= 0.5
fallback:           bad, flagged_borderline_or_disagreement
```

The CNN-rescue, RF-only-good, and fallback tiers are included in
`flagged_tae_like.csv`.

The RF-leaning policy was chosen from four-shot LOSO checks because RF was the
more stable held-out-shot ranker, while the CNN still provided useful
high-confidence rescues. The expanded 10-shot LOSO check gave the current
policy and RF-only the same aggregate accuracy (`0.9299`), with the combined
policy trading three extra false positives for three fewer false negatives.
Repeat the LOSO check on the 13-shot active set before changing this policy.

With `--make_plots`, the RF and CNN per-`n` score histograms are written
side-by-side in `hist_p_good_by_n.png`.

`--cnn_model_kind` defaults to `auto`, so `sort_shot_mixed.py` can use raw,
straightened, or hybrid CNN checkpoints that contain `model_type` metadata.
Pass `--cnn_model_kind cnn_raw`, `cnn_straightened`, or `cnn_hybrid` only for
older or ambiguous checkpoints.

For training-set checks, pass `--label_csv training_labels/tae_like_train.csv`.
Sorter output paths are matched to label paths by shot-relative suffix, so
absolute mode paths in the shot output can be compared with relative paths in
the training-label CSV. `--model_eval_threshold` controls the RF-only and
CNN-only evaluation threshold and defaults to `0.5`; the combined-policy
evaluation uses the actual `final_label` assigned by the fusion policy.

The `p_avg` score is a weighted RF/CNN average used as the duplicate-clustering
score:

```text
p_avg = (rf_score_weight * p_rf_good + cnn_score_weight * p_cnn_good)
        / (rf_score_weight + cnn_score_weight)
```

Both weights default to `0.5`. They affect which candidate is retained during
close-frequency duplicate removal, not the RF/CNN fusion labels.

### Usage

```bash
python sort_shot_mixed.py \
  --shot_dir /path/to/nstx_135388 \
  --rf_model /path/to/nova_mode_classifier.joblib \
  --cnn_model /path/to/nova_cnn_straightened.pt \
  --out_dir /path/to/sort_outputs/nstx_135388 \
  --label_csv training_labels/tae_like_train.csv \
  --make_plots
```
or when running from $NOVA_RUN_ROOT/runs/ directory (for checking on old labeled/training shots):
```bash
python $NOVA_REPO/scripts/sort_shot_mixed.py \
  --shot_dir $NOVA_DATA/nstx_135388 \
  --rf_model $NOVA_REPO/models/nova_mode_classifier.joblib \
  --cnn_model $NOVA_REPO/models/nova_cnn_raw.pt \
  --out_dir $NOVA_RUN_ROOT/sort_out_nstx_135388 \
  --label_csv $NOVA_REPO/training_labels/tae_like_train.csv \
  --make_plots
```
For production runs (new shots from /u/ngorelen/work/nova/DiTw/projdisk):
```bash
python $NOVA_REPO/scripts/sort_shot_mixed.py \
  --shot_dir /u/ngorelen/work/nova/DiTw/projdisk/nstxuE202855A01t020 \
  --rf_model $NOVA_REPO/models/nova_mode_classifier.joblib \
  --cnn_model $NOVA_REPO/models/nova_cnn_raw.pt \
  --out_dir path/to/sort_nstxuE202855A01t02 \
  --make_plots
```

The TAE/EAE split uses the normalized `signed_delta` plus
`fraction_below_upper2` convention from `src/tae_eae_features.py`. `mixed`
modes stay on the TAE side for RF/CNN classification so marginal TAEs are not
lost.

---

## `run_loso_10.py`

Driver for expanded 10-shot leave-one-shot-out checks. It:

- creates one `train.csv` and `test.csv` split per held-out shot from
  `training_labels/tae_like_train.csv`,
- retrains RF once per fold,
- retrains raw CNN once per fold,
- runs `sort_shot_mixed.py` on the held-out shot with `--label_csv`, and
- aggregates RF-only, CNN-only, and combined-policy metrics.

Small split/evaluation files are written under `outputs/loso_10/` by default.
Model checkpoints and training logs are written under `$NOVA_RUN/loso_10`, or
`$SCRATCH/nova_s/loso_10` when `$NOVA_RUN` is not set.

NERSC batch run:

```bash
cd "$NOVA_REPO"
sbatch scripts/run_loso_10.sbatch
```

Equivalent interactive run after a GPU allocation:

```bash
salloc --nodes 1 --qos interactive --time 4:00:00 --constraint gpu --gpus 1 --account m314_g
cd "$NOVA_REPO"
source configs/paths/nova_paths.nersc.sh
python -u scripts/run_loso_10.py \
  --steps all \
  --out_root outputs/loso_10 \
  --work_root "$SCRATCH/nova_s/loso_10" \
  --cnn_launch srun \
  --cnn_device cuda \
  --sort_device cpu \
  --cnn_batch_size 8 \
  --cnn_cache_data
```

Useful partial/resume commands:

```bash
# Only create the 10 fold split lists.
python scripts/run_loso_10.py --steps split --out_root outputs/loso_10

# Resume a failed run without repeating completed RF/CNN/sort folds.
python scripts/run_loso_10.py --steps all --skip_existing --cnn_launch srun --cnn_device cuda

# Re-aggregate after manually rerunning one fold.
python scripts/run_loso_10.py --steps aggregate --out_root outputs/loso_10
```

Main aggregate outputs:

- `outputs/loso_10/loso_split_counts.csv`
- `outputs/loso_10/loso_model_evaluation_summary.csv`
- `outputs/loso_10/loso_model_evaluation_totals.csv`
- `outputs/loso_10/loso_shot_summary.csv`

Per-fold sorter outputs live under
`outputs/loso_10/folds/<heldout-shot>/sort_shot_mixed/`.

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

- reads a labeled training CSV such as `training_labels/tae_like_train.csv` (`path,validity` or `path,label`, with or without a header row)
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
python rf_oof_check.py training_labels/tae_like_train.csv \
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
  training_labels/tae_like_train.csv \
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
