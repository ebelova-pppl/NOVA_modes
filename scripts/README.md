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

The current canonical/default good/bad training list is
`training_labels/tae_like_train.csv`. It contains 2,390 rows from 14 shots,
with 576 GOOD and 1,814 BAD labels. Q62 is suspended pending correction of its
suspect upper continuum boundary; its 249 reviewed rows remain preserved in
the complete 15-shot `training_labels/tae_like_v3.csv` snapshot. Older
four-shot TAE-only and mixed TAE/EAE lists are archived under
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
For continuum-aware experiments, `cnn_raw.py` accepts `--continuum_branch`.
The raw signed mode remains a one-channel `(m,r)` image. A separate 1D branch
receives four radius-aligned arrays: peak-normalized
`W(r)=sum_m|xi_m(r)|^2`, `du=(sqrt(high2)-omega)/omega`,
`dl=(omega-sqrt(low2))/omega`, and a continuum-validity mask. The mode trunk
is collapsed only over the harmonic axis, then concatenated with the 1D
continuum features at the corresponding radial bins before radial pooling.
`du` and `dl` are clipped to `--continuum_clip` (default `5.0`). Missing
continuum gives zero `du`, `dl`, and mask values while preserving `W(r)`.
Checkpoints save `model_type=cnn_raw_continuum_branch`; use
`--cnn_model_kind cnn_raw_continuum_branch` or `auto` for inference. The
retired broadcast-channel checkpoints remain loadable as
`cnn_raw_continuum`, but `cnn_raw.py` no longer trains that architecture.
For an architecture-only ablation, pass `--continuum_branch` together with
`--continuum_branch_zero_inputs`. This keeps the same 1D branch, radial fusion,
and classifier head but supplies an exact zero tensor for all four branch
inputs during both training and checkpoint inference. In LOSO, the matching
driver option is `--cnn_continuum_branch_zero_inputs`.
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

To expose quiet prediction collapse without cluttering normal logs,
`cnn_raw.py` checks prediction health at the normal epoch-reporting cadence but
prints only warnings. Starting at epoch 5, it warns when GOOD labels exist but
zero GOOD modes are predicted, when predictions are near-all-BAD or
near-all-GOOD, or when `p_good` is nearly constant. The warning includes the
predicted/true GOOD counts, GOOD fractions, and `p_good` mean/range. The
full-data refit is checked with a deterministic evaluation loader, and its
final prediction-health values are stored in the checkpoint under
`final_prediction_health`.

All three CNN training scripts seed Python, NumPy, and PyTorch from their seed
configuration so training runs are reproducible by default.

The CNN trainers print the selected Torch device, visible CUDA devices, and
free/total GPU memory before training. All three CNN trainers accept
`--device` and honor `NOVA_TORCH_DEVICE`, for example:

```bash
export NOVA_TORCH_DEVICE=cuda
python "$NOVA_REPO/scripts/cnn_raw.py" --batch_size 32
```

Inside a Perlmutter interactive allocation, launch the Python process with
`srun` so it runs on the allocated GPU node rather than on the login shell:

```bash
salloc --nodes 1 --qos interactive --time 1:00:00 --constraint gpu --gpus 1 --account m314_g
srun --nodes 1 --ntasks 1 --cpus-per-task 1 --gpus-per-task 1 python -u "$NOVA_REPO/scripts/cnn_raw.py" --batch_size 32
```

If CUDA reports out-of-memory for these small CNNs, first check the printed
free/total memory and try `--batch_size 8` or `--batch_size 4`. To diagnose
environment issues without using GPU memory, run raw CNN with `--device cpu` or
set `NOVA_TORCH_DEVICE=cpu` for the older trainers.

After sourcing `configs/paths/nova_paths.nersc.sh`, `nova_gpu_smoke` runs a
small Torch CUDA allocation through `srun` and prints timing for device report,
first tensor allocation, matmul, and CPU copy. `nova_run_cnn_raw --batch_size 32`
runs the raw CNN through the same Slurm launch path. The helpers default to
`NOVA_CPUS_PER_TASK=1`; if you set a larger value, request matching CPUs in the
`salloc` command.

On PPPL Flux with the default `tcsh` shell, source
`configs/paths/nova_paths.flux.csh`. The Flux config keeps the environment
minimal: it resolves `NOVA_REPO` from the current Git checkout, sets
`NOVA_MODELS=$NOVA_REPO/models`, sets `NOVA_TRAIN_CSV` and
`NOVA_TRAIN_CSV_TAE`, defaults `NOVA_TORCH_DEVICE=cpu`, and provides CPU
helpers. It does not set a default `NOVA_DATA`; pass absolute mode/shot paths
or set `NOVA_DATA` yourself for training and inspection workflows that use
relative CSV paths.

```tcsh
module load anaconda3
source `conda info --base`/etc/profile.d/conda.csh
setenv CONDA_PKGS_DIRS /p/hym/conda_pkgs
conda activate /p/hym/conda_envs/nova-perlmutter
cd /path/to/your/NOVA_modes
source configs/paths/nova_paths.flux.csh
nova_cpu_smoke
nova_run_cnn_raw --batch_size 32 --cache_data
```

The shared Flux environment uses scikit-learn `1.9.0` to match the current
Perlmutter-trained RF checkpoint. That release requires `narwhals>=2.0.1`.
Install scikit-learn without `--no-deps`, or repair an earlier no-dependency
upgrade in the activated environment with:

```tcsh
python -m pip install "narwhals>=2.0.1"
python -m pip check
python -c "import sys, sklearn, narwhals; print(sys.executable); print('sklearn', sklearn.__version__); print('narwhals', narwhals.__version__)"
```

Experimental continuum-branch LOSO can be run through the same LOSO driver:

```bash
python "$NOVA_REPO/scripts/run_loso_10.py" \
  --train_csv "$NOVA_TRAIN_CSV" \
  --data_dir "$NOVA_DATA" \
  --out_root "$NOVA_REPO/outputs/loso_15_raw_continuum_branch_M100_bs8" \
  --work_root "$SCRATCH/nova_s/loso_15_raw_continuum_branch_M100_bs8" \
  --cnn_batch_size 8 \
  --cnn_m_target 100 \
  --cnn_continuum_branch
```

To isolate the effect of the new architecture from the physical branch
features, use a separate output directory and add the zero-input control:

```bash
python "$NOVA_REPO/scripts/run_loso_10.py" \
  --train_csv "$NOVA_TRAIN_CSV" \
  --data_dir "$NOVA_DATA" \
  --out_root "$NOVA_REPO/outputs/loso_15_raw_continuum_branch_zero_M100_bs8" \
  --work_root "$SCRATCH/nova_sc/loso_15_raw_continuum_branch_zero_M100_bs8" \
  --cnn_batch_size 8 \
  --cnn_m_target 100 \
  --cnn_continuum_branch \
  --cnn_continuum_branch_zero_inputs \
  --cnn_cache_data \
  --cnn_refit_full_before_save
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
nova_run_cnn_raw --batch_size 32 --cache_data
```

Current 14-shot raw-CNN refresh from
`training_labels/tae_like_train.csv` (2,390 rows; batch size 8,
`M_target=100`, seed 42, unweighted loss):

- best 20% stratified-split checkpoint at epoch 40: accuracy=`0.9539`,
  CM=`[[353, 9], [13, 102]]`, GOOD precision/recall/F1=
  `0.919 / 0.887 / 0.903`
- active checkpoint: fresh 80-epoch full-CSV refit on all 2,390 rows, with
  `M_target=100`, `R_target=201`, robust normalization, and no detected
  prediction collapse
- log: `outputs/cnn_raw.txt`

Previous pre-B12 13-shot TAE-like raw-CNN retraining checks:

- `cnn_raw.py`, `M_target=100`, batch size 32: accuracy=`0.971`,
  CM=`[[394, 6], [9, 112]]`, GOOD precision/recall/F1=`0.949 / 0.926 / 0.937`
- previous `M_target=54` check: accuracy=`0.954`, CM=`[[394, 6], [18, 103]]`,
  GOOD precision/recall/F1=`0.945 / 0.851 / 0.896`

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

To retrain RF on the current canonical TAE-like list:

```bash
python "$NOVA_REPO/scripts/rf_train_classify.py" \
  --train_csv "$NOVA_REPO/training_labels/tae_like_train.csv" \
  --model_out "$NOVA_REPO/models/nova_mode_classifier.joblib"
```

The current script runs five-fold cross-validation on the complete input and
reports a 90/10 stratified evaluation. It then explicitly refits the pipeline
on all input rows before saving the deployment checkpoint.

Current 14-shot RF refresh from `training_labels/tae_like_train.csv`:

- input: 2,390 rows (576 GOOD and 1,814 BAD); mean five-fold row-wise CV
  accuracy=`0.9448`
- 239-row stratified holdout: CM=`[[169, 12], [4, 54]]`, accuracy=`0.933`,
  GOOD precision/recall/F1=`0.818 / 0.931 / 0.871`
- saved checkpoint fit: all 2,390 rows (576 GOOD and 1,814 BAD), using the
  production 22-feature `rf_w_star_max_22_v2` schema
- log: `outputs/rf_out.txt`

Most recent RF OOF check, run on the 13-shot list before merging B12:

- CM=`[[1967, 37], [91, 515]]`
- accuracy=`0.951`
- GOOD precision/recall/F1=`0.933 / 0.850 / 0.889`
- output: `outputs/rf_oof_13shots/`

The active RF checkpoint is `models/nova_mode_classifier.joblib`, refreshed
from the current Q62-free `training_labels/tae_like_train.csv` as described
above. Previous four-shot RF checkpoints are archived under
`models/old_4shots_models/`.

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

- `delta2_eff`: mode-weighted squared distance outside the local interval
  between the stored lower and upper TAE-gap boundaries; it is zero inside
  that interval
- `r_star`: first radial grid point attaining the minimum of that gap-distance
  quantity; it is not necessarily a continuum crossing
- `S`: absolute separation between the mode centroid and `r_star`, normalized
  by the mode radial width
- `W_star`: fraction of total radial mode energy within one mode width of
  `r_star`
- `W_star_max`: largest peak-normalized radial mode energy at any interpolated
  lower/upper continuum-boundary root

The current production RF schema has 22 features. It removes the raw `omega`
feature used by the previous checkpoint and adds `W_star_max`. Missing or
invalid continuum data use the existing safe fallback, with
`W_star_max = 0`.

#### Experimental energy-aligned `r_star` tie break

`--r-star-energy-tie` preserves the minimum gap-distance condition but, when
several radial points share that minimum, selects the point with maximum
`W(r)`. Equal-energy ties use the larger radius. This changes `r_star`, `S`,
and `W_star` without changing the 22 feature names, so checkpoints and bundles
store a distinct `_rstar_energy_tie_v1` schema suffix and classification must
use the same option. The active checkpoint retains the legacy first-minimum
rule.

Example shuffled-fold check:

```bash
python scripts/rf_oof_check.py training_labels/tae_like_v2_nonG.csv \
  --model_in models/nova_mode_classifier.joblib \
  --r-star-energy-tie \
  --out_oof rf_rstar_energy_oof.csv \
  --out_suspects rf_rstar_energy_suspects.csv
```

On the corrected 2610-row list, this rule worsened shuffled five-fold FN from
`92` to `97` and true shot-wise LOSO FN from `130` to `142`; G-shot LOSO FN
changed `31 -> 37`. Combining it with the three extrema features gave LOSO FN
`134` overall and `36` for G shots, still worse than baseline. Keep this option
for reproducibility only; do not use it with the active checkpoint or promote
it without new evidence.

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
  --train_csv "$NOVA_TRAIN_CSV" \
  --crossing-features \
  --model_out "$NOVA_REPO/models/nova_mode_classifier_crossing.joblib"
```

The bundle defaults to
`models/nova_mode_classifier_crossing_bundle.joblib`. The trainer refuses to
overwrite the active legacy checkpoint with an experimental model.

Run an apples-to-apples OOF experiment with the same feature option:

```bash
python "$NOVA_REPO/scripts/rf_oof_check.py" \
  "$NOVA_TRAIN_CSV" \
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

#### Experimental inner-extremum RF features

`--extremum-features` appends three mode-to-continuum-alignment scalars to the
production vector, producing the opt-in 25-feature
`rf_extremum_energy_25_v2` schema:

- `ext_dr`: distance between `r_peak = argmax W(r)` and the jointly matched
  inner upper-boundary minimum or lower-boundary maximum;
- `ext_df_gap`: signed relative frequency clearance, positive on the local gap
  side for either boundary type;
- `ext_energy_frac`: fraction of integrated `W(r)` within
  `|r-r_e| <= 0.03` of the matched extremum.

The search uses physical-frequency boundaries over `0.03 <= r <= 0.40` and
the fixed audit scales `dr=0.02` and `abs(df)=0.03` to select one joint match.
The scales choose the candidate; they do not classify it. The local-energy
fraction uses a fixed radial half-width of `0.03`. Missing continuum or no
detected inner extremum uses the deterministic fallback `(1, 1, 0)`.

Example same-fold ablation:

```bash
python scripts/rf_oof_check.py training_labels/tae_like_v2_nonG.csv \
  --model_in models/nova_mode_classifier.joblib \
  --extremum-features \
  --out_oof rf_extremum_oof.csv \
  --out_suspects rf_extremum_suspects.csv
```

Experimental training uses the same option and defaults to
`nova_mode_classifier_extremum.joblib`; classification with that checkpoint
must also pass `--extremum-features`. The option can be combined with
`--crossing-features`, producing a 31-feature schema. Experimental schemas are
not supported by the shot sorters or interactive labeling workflow.

On the corrected 2610-row list, shuffled five-fold FN changed `92 -> 89`.
True 13-fold shot-held-out FN remained `130`, while FP improved `42 -> 40`;
G-shot FN changed `31 -> 32`. The active 22-feature checkpoint was therefore
not replaced. See `docs/project_state.md` for the full matrices and the earlier
prominence-feature comparison.

Synthetic crossing, extremum, and schema checks use `unittest`:

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

The subplot height ratios are passed through `gridspec_kw` for compatibility
with the older system Matplotlib available on some Flux nodes.
The mode-structure and continuum panels share an explicit normalized-radius
axis spanning `[0, 1]`; the harmonic-spectrum panel keeps its independent
stored-harmonic-index axis. The black dashed closest-approach marker and
purple dotted maximum-crossing marker are drawn through both radial panels.

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

Mode-list and resume matching intentionally use the full resolved path or the
`shot/N/file` suffix, not `N/file` alone. This avoids cross-shot collisions
when the same mode filename appears in multiple shots and one output CSV is
used across several reviews. The legacy `N/file` fallback is available as
`--allow-n-file-match` only for old lists that do not contain shot names.

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
`$NOVA_DATA`. Path headers may be `path`, `filepath`, `mode_path`, or the
sorter's portable `mode_key`; label headers include `label`, `validity`, and
audit-table `training_validity`. GOOD, BAD, and SKIP labels are displayed in
the figure title. The mode and continuum panels share
the normalized-radius x-axis. The black dashed `r*` closest-approach marker and
purple dotted `r* max crossing` marker extend through both panels so they can
be compared directly with mode-amplitude features.

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

Canonical shot-level workflow for mixed TAE/EAE runs. It does not move files.
Select the decision engine with `--method {rules,rf-cnn}`; `rules` is the
default.

Both methods:

- validate mode files and required continuum inputs,
- route valid modes into `tae_like`, `mixed`, and `eae_like` gap regions,
- send TAE-like plus mixed modes to the selected decision backend,
- apply their method-specific final-decision policy before duplicate handling,
- reuse the close-frequency structural comparison conventions, and
- write CSV audit tables, shot summaries, a frequency-cluster report, and
  method-specific diagnostics.

The default `--method rules` path loads the frozen
`tae_rules_production_v1` configuration. A rejection gate produces automatic
BAD. A mode passing all enabled gates retains the scientifically conservative
engine result `rule_decision=REVIEW` and
`rule_primary_reason=NO_GOOD_TEMPLATE`; the separately audited
`accept-as-good-v1` workflow policy then promotes that survivor to final GOOD.
Fingerprint-matched manual overrides are applied after this automatic policy.
Final-GOOD modes proceed to duplicate handling. The production command supplies
`--rf_model` to rank representatives in close-frequency, structurally matched
groups. Without a usable RF checkpoint, all members of each affected cluster
are retained and the fallback is reported; that safe fallback is useful for
audits but is not the intended deduplicated production result. This method
never loads a CNN.

The explicit legacy `--method rf-cnn` path requires both `--rf_model` and
`--cnn_model`. It preserves the existing RF/CNN probabilities, fusion tiers,
weighted `p_avg` duplicate rank, evaluation reports, and plots. Model-specific
arguments are validated against the selected method; incompatible options fail
clearly rather than silently affecting a different backend.

Current operational note: deterministic rules are the production default.
The top-level RF and raw-CNN checkpoints were retrained from the current
14-shot `training_labels/tae_like_train.csv` and remain available for explicit
legacy comparisons. **NSTX-U G-case shots are a distinct AI-model regime**
because their narrow, strongly varying TAE gap gives sparse GOOD-mode labels
and weaker LOSO performance; this caveat applies to interpretation of the
legacy RF+CNN backend, not to method selection by omission.

Close-frequency duplicate handling enforces the frequency threshold pairwise
against the candidate representative before structure metrics can merge two
modes. This avoids chained clusters where several adjacent modes are close but
the first and last mode are separated by more than `--rel_freq_tol`. The
production rules recipe supplies a compatible RF checkpoint so the resolver
can retain the highest-RF representative in each structurally matched group.
If RF is omitted, rules mode retains all affected members rather than making
an unranked drop. Legacy RF+CNN mode continues to rank with `p_avg`.

The main outputs are:

- `good_tae_final.csv`
- `good_tae_unchecked.csv`
- `bad_tae_like.csv`
- `flagged_tae_like.csv` (legacy RF+CNN only)
- `review_tae_like.csv` (rules only)
- `eae_like.csv`
- `rejected_modes.csv`
- `shot_summary.csv`

It also writes `tae_like_all.csv`, `shot_summary_wide.csv`,
`shot_summary_by_n.csv`, `frequency_cluster_report.txt`, and
`frequency_clusters.csv` for auditability. Rules mode additionally writes
`all_modes_rules.csv`, `rule_results.csv`, `final_classifications.csv`,
`review_tae_like.csv`, and manual-override provenance. Legacy RF+CNN mode
retains `all_modes_scored.csv`, `flagged_tae_like.csv`, score columns, and its
model diagnostics.

The final-mode CSVs include `rad_loc` and `rad_width`, the same normalized
radial centroid and RMS radial width used by the RF feature schema. In
`good_tae_final.csv`, these columns can be compared with beam-ion density
profiles to deprioritize edge-localized modes whose beam drive is expected to
be small.

When `--method rf-cnn` and `--label_csv` are provided for a labeled validation
shot, it also writes:

- `model_evaluation_report.txt` — RF-only, CNN-only, and combined-policy
  confusion matrices plus classification reports
- `model_evaluation_summary.csv` — compact metrics table
- `model_evaluation_rows.csv` — per-mode true/predicted labels and scores

`shot_summary.csv` is written as a human-readable two-column key/value file.
`shot_summary_wide.csv` keeps the same one-row table layout as
`shot_summary_by_n.csv` for scripts and spreadsheet workflows. Legacy RF+CNN
summaries call the pre-cluster count `n_good_before_clustering`; rules
summaries call it `n_final_good_before_clustering`. Both record
`n_final_good` as the count written to `good_tae_final.csv` afterward. Rules
summaries also identify the immutable rule configuration, the
`accept-as-good-v1` survivor policy, and `n_rule_survivors_accepted`.

Under `--method rf-cnn`, `flagged_tae_like.csv` is an overlapping QC list
rather than a mutually exclusive class: it contains scored TAE-side modes that
are borderline or show RF/CNN disagreement, so they may also appear in either
`good_tae_unchecked.csv` or `bad_tae_like.csv`.

The legacy RF/CNN fusion policy is RF-leaning with a high-confidence CNN
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
Follow-up 13-shot and 7-shot LOSO checks support keeping the current full-set
RF and raw-CNN checkpoints for legacy E-like comparisons. Do not switch to a
G-shot policy from these defaults without a dedicated G-regime check.

With `--method rf-cnn --make_plots`, the RF and CNN per-`n` score histograms
are written side-by-side in `hist_p_good_by_n.png`. The
`rf_vs_cnn_pgood.png` diagnostic uses a two-panel view: a log-scaled binned
count-density panel to show how many modes pile up near score edges, and a
jittered tier-colored scatter panel with the legend outside the data region.

Under `--method rf-cnn`, `--cnn_model_kind` defaults to `auto`, so the sorter
can use raw, straightened, or hybrid CNN checkpoints that contain `model_type`
metadata. Pass `--cnn_model_kind cnn_raw`, `cnn_straightened`, or `cnn_hybrid`
only for older or ambiguous checkpoints.

For legacy training-set checks, pass
`--label_csv training_labels/tae_like_train.csv` or use `$NOVA_TRAIN_CSV`
after sourcing the path config.
Sorter output paths are matched to label paths by shot-relative suffix, so
absolute mode paths in the shot output can be compared with relative paths in
the training-label CSV. `--model_eval_threshold` controls the RF-only and
CNN-only evaluation threshold and defaults to `0.5`; the combined-policy
evaluation uses the actual `final_label` assigned by the fusion policy.

In legacy RF+CNN mode, `p_avg` is the weighted RF/CNN average used as the
duplicate-clustering score:

```text
p_avg = (rf_score_weight * p_rf_good + cnn_score_weight * p_cnn_good)
        / (rf_score_weight + cnn_score_weight)
```

Both weights default to `0.5`. They affect which candidate is retained during
close-frequency duplicate removal, not the RF/CNN fusion labels.

### Usage

For a production rules run, `--method rules` may be omitted because it is the
default. Keeping it explicit in saved commands makes provenance clearer:

```bash
python "$NOVA_REPO/scripts/sort_shot_mixed.py" \
  --method rules \
  --shot_dir "$NOVA_DITW_ROOT/$SHOT_NAME" \
  --rf_model "$NOVA_MODELS/nova_mode_classifier.joblib" \
  --out_dir "$NOVA_SORT_OUT/$SHOT_NAME"
```

On Flux, the corresponding environment variables use `tcsh` syntax:

```tcsh
setenv SHOT_NAME nstxuE205040A01t016
setenv NOVA_DITW_ROOT /p/nstxdigtwin/energetic_particles/nova/DiTw
setenv NOVA_SORT_OUT /path/to/output_dir
```

The RF checkpoint in this production command is a duplicate ranker only; it
does not participate in deterministic rule decisions. Omit it only for an
explicit no-deduplication audit or fallback run.

To run the explicit legacy RF+CNN method on Flux:

```tcsh
python "$NOVA_REPO/scripts/sort_shot_mixed.py" \
  --method rf-cnn \
  --shot_dir "$NOVA_DITW_ROOT/$SHOT_NAME" \
  --rf_model "$NOVA_REPO/models/nova_mode_classifier.joblib" \
  --cnn_model "$NOVA_REPO/models/nova_cnn_raw.pt" \
  --cnn_model_kind cnn_raw \
  --out_dir "$NOVA_SORT_OUT/$SHOT_NAME" \
  --device cpu \
  --make_plots
```

At NERSC, Bash syntax applies. A production rules run is:

```bash
python scripts/sort_shot_mixed.py \
  --method rules \
  --shot_dir /path/to/nstx_135388 \
  --rf_model models/nova_mode_classifier.joblib \
  --out_dir /path/to/sort_outputs/nstx_135388
```

An explicit legacy labeled-shot check is:

```bash
python scripts/sort_shot_mixed.py \
  --method rf-cnn \
  --shot_dir /path/to/nstx_135388 \
  --rf_model /path/to/nova_mode_classifier.joblib \
  --cnn_model /path/to/nova_cnn_straightened.pt \
  --out_dir /path/to/sort_outputs/nstx_135388 \
  --label_csv training_labels/tae_like_train.csv \
  --make_plots
```
When running from `$NOVA_RUN_ROOT/runs/` to check an old labeled shot:

```bash
python $NOVA_REPO/scripts/sort_shot_mixed.py \
  --method rf-cnn \
  --shot_dir $NOVA_DATA/nstx_135388 \
  --rf_model $NOVA_REPO/models/nova_mode_classifier.joblib \
  --cnn_model $NOVA_REPO/models/nova_cnn_raw.pt \
  --out_dir $NOVA_RUN_ROOT/sort_out_nstx_135388 \
  --label_csv $NOVA_TRAIN_CSV \
  --make_plots
```

The TAE/EAE split uses the normalized `signed_delta` plus
`fraction_below_upper2` convention from `src/tae_eae_features.py`. `mixed`
modes stay on the TAE side for the selected decision method so marginal TAEs
are not lost.

---

## Deterministic rule sorting: production and calibration interfaces

The shared deterministic implementation comprises
`make_tae_like_list.py`, `tae_rule_engine.py`, and `sort_shot_rules.py`.
`sort_shot_mixed.py` now exposes it as the default production decision method,
while `sort_shot_rules.py` remains the conservative calibration and audit CLI.

For production runs:

```bash
python scripts/sort_shot_mixed.py \
  --method rules \
  --shot_dir /path/to/shot \
  --rf_model models/nova_mode_classifier.joblib \
  --out_dir /path/to/rule_sort_output
```

The version-controlled configuration is
`configs/rules/tae_rules_production_v1.yaml`, stored as strict
JSON-compatible YAML so loading requires no additional package. It pins the
current v14 ruleset, routing thresholds, relative-frequency tolerance, all
gate thresholds, and these gate states:

- enabled: gates 1 (`BAD_AXIS_SPIKE`), 2 (`BAD_GRID_SCALE_SPIKE`), 2b
  (`BAD_GRID_SCALE_PACKET`), 4 (`BAD_CONT_CROSS_WINDOW`), and 5
  (`BAD_EDGE_SPIKE`);
- disabled: gate 3 (`BAD_CONT_CROSS`), while retaining its frozen latent
  threshold `W_star_max > 0.03` for possible future comparison.

Rules mode loads this configuration by default and does not permit a
config-owned threshold or gate override to retain the production-v1 identity.
`shot_summary.csv`, `shot_summary_wide.csv`, and `shot_summary_by_n.csv`
record `rule_configuration_name`, `rule_configuration_schema_version`,
`rule_configuration_sha256`, and the audited `accept-as-good-v1` survivor
policy. The policy changes only the workflow's automatic final decision:
`rule_decision=REVIEW` and `rule_primary_reason=NO_GOOD_TEMPLATE` remain
unchanged in the rule audit columns, while the survivor becomes final GOOD
before manual overrides and duplicate handling.

For rule development and threshold experiments, run the configurable
conservative workflow without a named configuration:

```bash
python scripts/sort_shot_rules.py \
  --shot_dir /path/to/shot \
  --out_dir /path/to/rule_sort_output
```

In this interface, modes that pass every gate remain final REVIEW. To audit
the exact frozen gate configuration without production promotion, add
`--rule_config tae_rules_production_v1` to the `sort_shot_rules.py` command.

`scripts/make_tae_like_list.py` also exposes an importable
`preprocess_shot()` interface and a standalone preprocessing CLI. Before any
mode is processed, it aborts the whole shot if a populated requested `N#`
directory lacks its `datcon#`. Other corrupt, nonfinite, or unusable inputs are
recorded individually as `INVALID`. Valid mixed modes stay in the TAE-side
list with `gap_region=mixed`; valid EAE-like modes are routed without a rule
decision.

`scripts/tae_rule_engine.py` is a pure per-mode interface. Its current
`tae-rules-axis-all-peaks-grid-highr-packet-turns-rle05-cont-window-edge-v14`
ruleset
implements six ordered BAD decisions, treating the packet screen as gate 2b so
the established gate-3/4/5 names remain stable. It still has no positive GOOD
template. Modes that do not fire any
gate return `REVIEW` with primary reason `NO_GOOD_TEMPLATE`. Multiple rule
reasons and structured features are stored as deterministic JSON; missing
feature values use JSON `null`.

Before making a decision, the engine records the canonical 31
measurements and their crossing audit records in a grouped `rule_features`
object. Its rule-facing schema is `tae-rule-features-grouped-v13`, with
`source_feature_schema_version=rf_all_crossings_extremum_energy_31_v2`. The
groups are:

- `rf_standard_features`: the production 22, including `rad_loc`,
  `rad_width`, the mode-shape statistics, `gamma_d`, `ntor`, and the
  production continuum scalars;
- `crossing_features`: `n_cross`, `r_star_max`, `W_star_sum`,
  `r_star_high_shear`, `W_star_high_shear`, and
  `W_star_high_shear_sum`, plus independently selected amplitude and normalized
  energy winners from the configured neighborhood of all true crossings. The
  latter record the window widths, winning values, sample radii, associated
  crossing boundary/radius, harmonic index for amplitude, and sample distance
  from the crossing in grid intervals. The amplitude winner also records its
  signed four-neighbor RMS, available-neighbor count, and complete-stencil
  status;
- `crossing_records`: every lower/upper crossing with `boundary`, `r_cross`,
  `W_peak`, and `shear_weighted`, ordered by boundary and radius;
- `extremum_features`: `match_found`, `ext_dr`, `ext_df_gap`, and
  `ext_energy_frac`;
- `boundary_features.axis_artifact`: `r_ax`, configured candidate thresholds,
  candidate-found status, total local-peak count, amplitude- and
  width-qualified counts, the selected `axis_peak`, its zero-based
  `axis_peak_harmonic_index`, `axis_peak_r`, `axis_peak_is_local_max`, connected
  half-maximum width in normalized radius and grid intervals, outer edge, and
  whether the component includes `r=0`;
- `boundary_features.edge_artifact`: the global normalized total-energy peak
  radius, inclusive edge-window status, connected full-grid half-maximum edges
  and widths, and outer-boundary touch status, plus the strongest individual
  edge harmonic's peak, zero-based stored harmonic index, local-maximum status,
  full-grid half-maximum edges and widths, and boundary-touch status for audit;
- `numerical_structure_features.grid_scale_spike`: whether a width-limited
  signed-lobe candidate was found, both radial width limits and their cutoff,
  the limit applied to the selected candidate, its absolute and signed peak
  amplitude, sign, zero-based stored harmonic index, radius, interpolated
  half-maximum edges and widths, and whether the component touches either
  radial boundary;
- `numerical_structure_features.grid_scale_packet`: the configured packet
  amplitude, adjacent-step, large-turn-count, window-span, and inclusive peak
  radius thresholds; all-radius turn-qualified, radius-qualified, and
  amplitude-qualified window counts; and the selected window's peak, stored
  harmonic index, radial and sample-index bounds, large-step and large-turn
  counts, maximum step, step RMS, total variation, unconstrained direction-
  and sign-change counts, and five signed sample values;
- a reserved empty `resolution_features` object for later rule development.

This calculates the same quantities used by RF feature experiments but does
not load an RF checkpoint or use an RF prediction. `signed_delta` and
`fraction_below_upper2` remain top-level family-routing audit columns and are
not duplicated in `rule_features`. A rule-feature extraction failure produces
`INVALID` with reason `RULE_FEATURE_EXTRACTION_FAILED`; unavailable values use
JSON `null`. If there are no crossings, the undefined crossing radii are null
and `crossing_records` is empty. If no inner extremum is detected,
`extremum_features.match_found` is false and the three extremum measurements
are null rather than the RF-only fallback tuple `(1, 1, 0)`.

NOVA mode radius and amplitude are already normalized. The loader and rule
outputs use the zero-based stored harmonic index and do not infer the physical
poloidal-`m` offset. The axis gate enumerates every absolute-harmonic local
maximum centered in the inclusive window `r <= r_ax`. Each connected
half-maximum component is measured on the complete radial profile. The
strongest candidate meeting both configured thresholds is selected, so a
larger rising flank or broad local peak cannot mask a narrower qualifying
artifact. When no candidate qualifies, the strongest raw window amplitude is
retained as fallback audit information.

Axis-gate configuration corresponds to:

```yaml
axis_artifact:
  r_ax: 0.03
  axis_amplitude_min: 0.2
  axis_width_max_grid: 10
```

The CLI names are `--axis_r_ax`, `--axis_amplitude_min`, and
`--axis_width_max_grid`. These calibrated values are active by default. Use
`--disable_axis_artifact` to calculate fallback axis-window features without
applying the BAD gate. If any local maximum meets the amplitude minimum and
maximum full-grid width, the strongest qualifying candidate returns `BAD` with
`BAD_AXIS_SPIKE` and stops later decision gates. Any sufficiently narrow local
maximum centered at `r <= 0.03` is treated as a boundary artifact without a
morphology-family exception. The shot and
per-`n` summaries record the enable flag and exact configuration.

The second gate searches every stored harmonic across the complete radial grid
for positive local maxima and negative local minima. It measures each connected
half-maximum lobe on the signed profile, not on `abs(mode)`, so adjacent
opposite-sign samples cannot be merged into a falsely broad component. Among
lobes no wider than the configured limit, it records the strongest candidate.
One-sided local extrema at either radial endpoint are included.

Grid-scale-spike configuration corresponds to:

```yaml
grid_scale_spike:
  amplitude_min: 0.3
  width_max_grid: 1
  high_r_cutoff_r: 0.7
  high_r_width_max_grid: 0.75
```

Peaks at `r <= 0.7` use `width_max_grid`; peaks strictly above `0.7` use
`high_r_width_max_grid`. The cutoff belongs to the low-r branch, and both width
comparisons are inclusive. The CLI names are `--grid_scale_amplitude_min`,
`--grid_scale_width_max_grid`, `--grid_scale_high_r_cutoff_r`, and
`--grid_scale_high_r_width_max_grid`. When the strongest width-limited candidate meets
the amplitude threshold, the second ordered gate returns `BAD` with
`BAD_GRID_SCALE_SPIKE` and stops later decision gates. Use
`--disable_grid_scale_spike` to retain the configured-width measurements while
disabling this decision.

Gate 2b scans every complete short window of every stored harmonic. Its
provisional configuration is:

```yaml
grid_scale_packet:
  amplitude_min: 0.3
  step_min: 0.2
  min_large_turns: 3
  window_span_grid: 4
  peak_r_max: 0.5
```

A four-interval window contains five radial samples and three possible interior
turning points. Let `d[i] = A[i+1] - A[i]`. An interior sample is a large turn
only when both adjacent differences satisfy `abs(d) >= step_min` and their
signs oppose, `d[i-1] * d[i] < 0`. This counts sharp signed local maxima and
minima, including same-sign peaks separated by deep troughs, while excluding a
single steep but smooth rise and fall. When at least `min_large_turns` qualify,
the largest absolute sample in the window is at least `amplitude_min`, and
that sample is centered at the inclusive radius `r <= peak_r_max`, return
`BAD` with `BAD_GRID_SCALE_PACKET` before evaluating continuum gates.
The step and amplitude comparisons are inclusive; the direction reversal is
strict, so a zero step is not a turn. Turn-qualified windows are counted over
the complete radial grid for audit, while only radius-qualified windows can
become candidates. Window bounds and peak radius remain recorded. Raw
large-step, direction-change, and sign-change counts remain audit fields.
Override the settings with `--grid_scale_packet_amplitude_min`,
`--grid_scale_packet_step_min`, `--grid_scale_packet_min_large_turns`, and
`--grid_scale_packet_window_span_grid`, and
`--grid_scale_packet_peak_r_max`; use `--disable_grid_scale_packet` to retain
measurements without applying the decision.

The third gate uses the existing true lower/upper continuum crossings and
their pointwise radial energy. `W_star_max` is the maximum over crossing records
of `sum_h |mode_h(r_cross)|^2`, normalized by the maximum of that energy over
radius. Its provisional calibrated configuration is:

```yaml
continuum_crossing:
  w_cross_threshold: 0.03
```

After the axis, single-lobe, and packet gates, a mode with `n_cross > 0` and
`W_star_max > w_cross_threshold` returns `BAD` with `BAD_CONT_CROSS` and stops
later gates. The comparison is strictly greater than the threshold. Override
it with `--w_cross_threshold`, or use `--disable_cont_cross` to retain the
crossing measurements without applying the decision. The shot and per-`n`
summaries record the crossing-gate enable flag and calibrated threshold.

The fourth gate samples the normalized mode arrays near every interpolated true
crossing. Its calibrated configuration is:

```yaml
continuum_crossing_window:
  half_width_grid: 2
  amplitude_min: 0.25
  w_min: 0.05
```

For each crossing, include radial samples satisfying
`abs(r_i - r_cross) <= half_width_grid * delta_r`. Across all such samples and
crossings, independently retain the largest individual-harmonic absolute
amplitude `cross_window_A_max` and largest peak-normalized total radial energy
`cross_window_W_max`, with the winning sample and crossing metadata for each.
For the amplitude-winning harmonic and radial sample `j`, calculate
`sqrt(mean((A[j]-A[j+i])**2))` using signed amplitudes at offsets
`i=-2,-1,+1,+2`. Require all four neighbors; otherwise record a null RMS and
mark the stencil incomplete. The winner can be shifted within the crossing
window, so its stencil can reach four grid intervals from the associated
crossing. This RMS is audit information only and does not alter the decision.

After `BAD_CONT_CROSS`, a mode with `n_cross > 0` returns `BAD` with
`BAD_CONT_CROSS_WINDOW` when either
`cross_window_A_max >= amplitude_min` or `cross_window_W_max >= w_min`. These
magnitude comparisons are inclusive. Override the settings with
`--cross_window_half_width_grid`, `--cross_window_amplitude_min`, and
`--cross_window_w_min`; use `--disable_cont_cross_window` to retain the
measurements while disabling this decision.

The fifth gate uses the global radial-energy envelope
`W(r)=sum_h |mode_h(r)|^2`, normalized by its global maximum. Its provisional
calibrated configuration is:

```yaml
edge_artifact:
  r_edge_min: 0.97
  edge_width_max_grid: 10
```

After both continuum-crossing gates, a mode whose global energy peak is at
`r >= r_edge_min` and whose connected energy FWHM is no greater than
`edge_width_max_grid` returns `BAD` with `BAD_EDGE_SPIKE`. Both half-maximum
edges are found on the complete radial grid. The strongest individual harmonic
within the same inclusive edge window is recorded for audit, but does not fire
this gate alone because physical edge-localized modes can contain narrow
shear-localized harmonics within a broader total envelope. Override the
settings with `--edge_r_min` and `--edge_width_max_grid`, or use
`--disable_edge_artifact` to retain the measurements without applying the
decision. Shot and per-`n` summaries record enable state and thresholds for all
six BAD decisions.

Main outputs retain compatible `sort_shot_mixed.py` names where their meaning
still applies:

- `tae_like_all.csv`, `eae_like.csv`, and `rejected_modes.csv`;
- `bad_tae_like.csv`, `good_tae_unchecked.csv`, and `good_tae_final.csv`;
- `shot_summary.csv`, `shot_summary_wide.csv`, and `shot_summary_by_n.csv`;
- `frequency_cluster_report.txt` and `frequency_clusters.csv`.

Deterministic/manual semantics use these new names:

- `all_modes_rules.csv` replaces the ML-specific `all_modes_scored.csv`;
- `rule_results.csv` preserves preliminary TAE-side rule results;
- `final_classifications.csv` preserves both rule and final decisions;
- `review_tae_like.csv` is a mutually exclusive REVIEW list rather than the
  overlapping ML-QC `flagged_tae_like.csv`;
- `manual_overrides.csv` records reusable adjudication provenance and is
  header-only when unused.

In the production `sort_shot_mixed.py --method rules` output,
`review_tae_like.csv` normally has no automatic survivors because
`accept-as-good-v1` promotes them to final GOOD. It remains a valid output for
manual REVIEW overrides. In `sort_shot_rules.py`, pass-all-gates modes remain
in this file unless adjudicated.

Every CSV is written with headers even when empty. Rows are ordered by shot,
`ntor`, frequency, and mode filename. Summary reason counts use exactly one
primary reason per mode; `rule_triggered_rules` is retained only as per-mode
audit detail.

### Manual adjudication

For a production rules result, use `--adjudication review`. Candidate selection
uses the preserved preliminary `rule_decision`, so automatic survivors remain
eligible even though their final decision is GOOD:

```bash
python scripts/label_modes_fast.py /path/to/shot \
  --mode-list /path/to/rule_sort_output/final_classifications.csv \
  --csv_out /path/to/manual_overrides.csv \
  --adjudication review \
  --reviewer REVIEWER_ID \
  --no-rf
```

Use `--adjudication all` only when gate-rejected BAD rows should also be
inspected. Adjudication requires signed harmonics, `--no-rf`, and a nonempty
manual reason for each `g`/`b`/`r` decision. It calculates a SHA-256 fingerprint
from the current mode and corresponding `datcon#` contents.

Rerun the production sorter to apply the override file reproducibly after its
automatic survivor policy:

```bash
python scripts/sort_shot_mixed.py \
  --method rules \
  --shot_dir /path/to/shot \
  --rf_model models/nova_mode_classifier.joblib \
  --out_dir /path/to/rule_sort_output \
  --manual_overrides /path/to/manual_overrides.csv
```

Use the same `--manual_overrides` option with `sort_shot_rules.py` when
rebuilding a conservative audit instead.

The sorter rejects empty reasons and applies only unique overrides whose stored
fingerprint matches current inputs. Stale, ambiguous, ineligible, and unmatched
override counts are reported. The summary stores the SHA-256 of the exact
override file used.

### Final-GOOD production deduplication

The production `sort_shot_mixed.py --method rules` recipe passes
`--rf_model /path/to/model.joblib` so close-frequency, structurally matched
final-GOOD modes are reduced to RF-ranked representatives. RF `p_good` becomes
`duplicate_rank_score` with source `rf_p_good`; it cannot change rule, manual,
or final decisions. Only final GOOD modes are scored. A missing/unloadable
checkpoint retains every member of each affected close-frequency cluster with
`SKIPPED_NO_RF_CHECKPOINT`. A scoring failure for one member retains that whole
cluster with `SKIPPED_RF_SCORING_FAILED`. CNN models are never loaded or run by
this workflow. Omitting RF remains supported for conservative audit runs,
including `sort_shot_rules.py`, but intentionally skips production
deduplication.

---

## Training-shot provenance audit: `audit_training_provenance.py`

Use this read-only audit before relabeling, retraining, or replacing local shot
data when the same NOVA filenames may have been recalculated in a shared shot
database. It compares the shots named by the selected training CSV across a
training snapshot and a reference tree. The scoped payload is:

- every `N*/egn*` mode file in the union of both trees;
- active `N*/datconN` continuum files;
- preserved training-side `datconN_old` backups, paired with reference
  `datconN`; and
- optional `datcon_gf.txt` auxiliary files.

Run-directory executables, plots, and logs are deliberately excluded. Mode and
continuum identity is determined by SHA-256, not filename, size, or timestamp.
Changed same-name modes are also parsed to record `omega`, damping, array
shape, classifier-used mode-structure equality, and maximum absolute mode
difference.

After setting `NOVA_DATA`, sourcing the Flux path config supplies
`NOVA_DITW_ROOT` and `NOVA_TRAIN_CSV`:

```bash
source configs/paths/nova_paths.flux.sh
python scripts/audit_training_provenance.py \
  --training-root "$NOVA_DATA" \
  --reference-root "$NOVA_DITW_ROOT" \
  --train-csv "$NOVA_TRAIN_CSV" \
  --out-dir audits/training_provenance/YYYY-MM-DD_flux_vN
```

Use a new dated/versioned output directory for each frozen comparison. The
script refuses to overwrite known artifacts unless `--replace` is passed.
Outputs are:

- `file_manifest.csv` — complete scoped inventory, hashes, timestamps, labels,
  and changed-mode diagnostics;
- `differences.csv` — all non-identical or missing file pairs;
- `shot_summary.csv` — per-shot canonical-mode, all-mode, and continuum counts;
- `report.md` — concise human-readable findings;
- `run_metadata.json` — schema, exact roots/options, training-list hash, audit
  script hash, and artifact hashes; and
- `SHA256SUMS` — integrity hashes for the generated artifact set.

The first retained Flux audit is
`audits/training_provenance/2026-08-20_flux_v1/`. It uses schema
`nova-training-provenance-v1` and covers all 15 shots and 2900 rows in
`training_labels/tae_like_v2_nonG.csv`.

---

## `run_loso_10.py`

Driver for leave-one-shot-out checks over all shots in the selected training
CSV. The filename is historical; with the current `tae_like_train.csv` it
creates 14 folds because Q62 is suspended. It:

- creates one `train.csv` and `test.csv` split per held-out shot from
  the selected `--train_csv` list,
- retrains RF once per fold,
- retrains raw CNN once per fold,
- runs `sort_shot_mixed.py --method rf-cnn` on the held-out shot with
  `--label_csv`, and
- aggregates RF-only, CNN-only, and combined-policy metrics.

For the non-G / NSTX-U E-like production-regime comparison, use
`--train_csv training_labels/tae_like_train_7.csv`; this derived list excludes
all `nstxuG*` shots and creates 7 LOSO folds.

Small split/evaluation files are written under `outputs/loso_<N shots>/` by
default. Model checkpoints and training logs are written under
`$NOVA_RUN/<output-name>`, or `$SCRATCH/nova_s/<output-name>` when `$NOVA_RUN`
is not set. For parameter comparisons, use separate output/work roots.

Pre-B12 13-shot raw-CNN `M_target` / batch-size comparison:

```bash
# From inside an interactive GPU allocation.
cd "$NOVA_REPO"
source configs/paths/nova_paths.nersc.sh

python -u scripts/run_loso_10.py \
  --steps all \
  --out_root outputs/loso_13_M54 \
  --work_root "$SCRATCH/nova_s/loso_13_M54" \
  --cnn_launch srun \
  --cnn_device cuda \
  --sort_device cpu \
  --cnn_batch_size 8 \
  --cnn_m_target 54 \
  --cnn_cache_data

python -u scripts/run_loso_10.py \
  --steps all \
  --out_root outputs/loso_13_M100 \
  --work_root "$SCRATCH/nova_s/loso_13_M100" \
  --cnn_launch srun \
  --cnn_device cuda \
  --sort_device cpu \
  --cnn_batch_size 8 \
  --cnn_m_target 100 \
  --cnn_cache_data
```

NERSC batch run with the current defaults:

```bash
cd "$NOVA_REPO"
sbatch scripts/run_loso_10.sbatch
```

The raw-CNN default `M_target` is now 100, and the LOSO driver defaults to
`--cnn_batch_size 32`. The historical `outputs/loso_13_M100` comparison used
`--cnn_batch_size 8`; `outputs/loso_13_M100_bs32` repeats the M100 LOSO check
with batch size 32. The Slurm wrapper's default `LOSO_TAG` is
`loso_13_M100_bs32`. To reproduce the older M54 or M100 batch-8 runs, pass
`--cnn_batch_size 8` and the desired `--cnn_m_target` explicitly.

Generic interactive run after a GPU allocation:

```bash
salloc --nodes 1 --qos interactive --time 4:00:00 --constraint gpu --gpus 1 --account m314_g
cd "$NOVA_REPO"
source configs/paths/nova_paths.nersc.sh
python -u scripts/run_loso_10.py \
  --steps all \
  --cnn_launch srun \
  --cnn_device cuda \
  --sort_device cpu \
  --cnn_cache_data
```

Useful partial/resume commands:

```bash
# Only create the fold split lists.
python scripts/run_loso_10.py --steps split --out_root outputs/loso_13_M100_bs32

# Resume a failed run without repeating completed RF/CNN/sort folds.
python scripts/run_loso_10.py \
  --steps all \
  --out_root outputs/loso_13_M100_bs32 \
  --work_root "$SCRATCH/nova_s/loso_13_M100_bs32" \
  --skip_existing \
  --cnn_launch srun \
  --cnn_device cuda

# Re-aggregate after manually rerunning one fold. Pass the same CNN options
# used by the original run if it did not use the current defaults.
python scripts/run_loso_10.py \
  --steps aggregate \
  --out_root outputs/loso_13_M100_bs32 \
  --cnn_m_target 100 \
  --cnn_batch_size 32
```

Main aggregate outputs:

- `outputs/<run-name>/loso_split_counts.csv`
- `outputs/<run-name>/loso_model_evaluation_summary.csv`
- `outputs/<run-name>/loso_model_evaluation_totals.csv`
- `outputs/<run-name>/loso_shot_summary.csv`
- `outputs/<run-name>/run_config.json`

Per-fold sorter outputs live under
`outputs/<run-name>/folds/<heldout-shot>/sort_shot_mixed/`.

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
