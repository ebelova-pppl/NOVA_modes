# Project: AI NOVA mode classifier
### Project state (current snapshot, updated 2026-07-08)
## Goal
Train ML classifiers to identify physically meaningful NOVA eigenmodes (“good”) vs unphysical/numerical modes (“bad”), and provide a clean, deduplicated mode set for downstream analysis (e.g., NOVA-C, surrogate modeling, digital twin workflows).
 
## Data
- Active version-controlled training list:
    - `training_labels/tae_like_train.csv`
    - 2610 labeled TAE-like modes: 606 `good`, 2004 `bad`
    - shots: `nstx_120113`, `nstx_135388`, `nstx_141711`, `nstxu_204202`,
      `nstxuE202855A01t020`, `nstxuE204669M03t025`,
      `nstxuE205052A01t022`, `nstxuG121123K51`, `nstxuG133964S31`,
      `nstxuG142301H47`, `nstxuG121123J38`, `nstxuG121123Q62`,
      `nstxuG142301Y93`
    - mode paths stored relative to `$NOVA_DATA` when possible
    - example entry: `nstx_120113/N5/egn05w.1234E+02,good`
- Derived non-G / E-production comparison list:
    - `training_labels/tae_like_train_7.csv`
    - 1638 labeled modes: 546 `good`, 1092 `bad`
    - shots: `nstx_120113`, `nstx_135388`, `nstx_141711`, `nstxu_204202`,
      `nstxuE202855A01t020`, `nstxuE204669M03t025`,
      `nstxuE205052A01t022`
    - created from `tae_like_train.csv` by excluding all `nstxuG*` shots for
      7-shot LOSO checks of the non-G / E-like production regime
- Archived historical lists:
    - older TAE-only lists: `training_labels/old_4shots_tae_only_labels/`
    - previous four-shot mixed TAE/EAE lists: `training_labels/old_4shots_mixed_labels/`
- Component lists for the expanded training pool:
    - original four-shot copy: `training_labels/additions/tae_like_4old.csv`
    - reviewed six-shot NSTX-U copy: `training_labels/additions/tae_like_6new.csv`
    - previous 10-shot active-list snapshot:
      `training_labels/additions/tae_like_copy.csv`
    - refreshed `nstx_135388` replacement copy:
      `training_labels/additions/tae_like_nstx_135388.csv`
    - new `nstxuG121123J38` copy:
      `training_labels/additions/tae_like_nstxuG121123J38.csv`
    - reviewed two-shot G-case copy:
      `training_labels/additions/tae_like_2new.csv`
- Staged lists not yet merged:
    - `training_labels/additions/tae_like_3new.csv`: three additional NSTX-U G-case
      shots kept as the original combined reference. Do not merge this file
      as-is: `nstxuG121123Q62` and `nstxuG142301Y93` were split into
      `training_labels/additions/tae_like_2new.csv` and merged, while
      `nstxuG121123N75` remains blocked pending recalculation with the
      corrected q profile.

Each mode includes:
-	Scalar metadata:
    -	omega — mode frequency
    -	gamma_d — continuum damping
    -	ntor — toroidal mode number
-	Mode structure: mode[m,r]

Continuum data (datcon<N>, one for each shot/ntor):
-	omega_A_low(r)^2, omega_A_high(r)^2

Mode file format
- f1.size = 3*nr*nhar + 4
- omega = f1[0]
- nr = int(f1[-3])
- gamma_d = f1[-2]
- ntor = f1[-1]
- nhar = (f1.size - 4) / (3*nr)

Notes:
-	nhar varies significantly for NSTX cases (~10 → 100, ~∝ n)
-	one NSTX-U shot had constant nhar = 54
-	n_r may vary across shots (handled via resampling)

## Models (current)
1.	RF (Random Forest)
    -	Scalar + structure-derived + continuum features (22)
    -	Active checkpoint: `models/nova_mode_classifier.joblib`
    -	Checkpoint status: retrained on the current 2610-row / 13-shot active
      list after the `training_labels/additions/tae_like_2new.csv` merge
    -	Current schema: previous RF features minus `omega`, plus `W_star_max`
    -	Current 13-shot OOF accuracy: 0.951
    -	Current 13-shot OOF CM: `[[1967, 37], [91, 515]]`
    -	Current GOOD precision/recall/F1: 0.933 / 0.850 / 0.889
    -	Most interpretable baseline
2.	CNN (raw)
    -	Padded/truncated (m,r)
    -	Active checkpoint: `models/nova_cnn_raw.pt`
    -	Checkpoint status: retrained on the current 2610-row / 13-shot active
      list as a full-CSV refit with `M_target=100`, `R_target=201`, batch size
      32, robust normalization, and no final prediction-collapse warning
    -	Current default raw preprocessing: `M_target=100`, `R_target=201`
    -	Latest 13-shot M100 held-out split check: CM `[[394, 6], [9, 112]]`,
      accuracy 0.971, GOOD precision/recall/F1 0.949 / 0.926 / 0.937
    -	Previous 13-shot M54 held-out split check: CM `[[394, 6], [18, 103]]`,
      accuracy 0.954, GOOD precision/recall/F1 0.945 / 0.851 / 0.896
    -	LOSO remains the main check for shot-to-shot generalization and fusion
      policy
3.	CNN (straightened)
    -	Ridge-aligned representation (2M+1, r)
    -	Previous four-shot checkpoint archived under `models/old_4shots_models/`
    -	Needs expanded-set retraining / recheck
4.	HybridCNN (image + scalars)
    -	Includes continuum scalars
    -	Previous four-shot checkpoint archived under `models/old_4shots_models/`
    -	Needs expanded-set retraining / recheck

## Continuum-derived scalars
From cont_features.py:
-	r_star — closest approach to continuum
-	delta2_eff — mode-weighted distance to continuum
-	S — normalized separation between r0 and r_star
-	W_star — mode energy near resonance
-	W_star_max — largest peak-normalized mode energy at an interpolated
  lower/upper continuum-boundary crossing

## Current scripts
### Common
-	nova_mode_loader.py
-	mode_features.py
-	cont_features.py
-	tae_eae_features.py
-	mode_transform.py
-	view_modes_csv.py
-	sort_shot.py
-	sort_shot_mixed.py
-	run_loso_10.py
-	split_tae_eae.py
### RF
-	rf_train_classify.py (renamed from nova_mode_classifier.py)
-	rf_oof_check.py
-	find_rf_disagreements.py
-	label_modes_fast.py
### CNN
-	cnn_raw.py
-	cnn_straightened.py
-	cnn_hybrid.py
-	cnn_classify.py (shared inference for straightened / hybrid checkpoints)
-	cnn_raw_classify.py
-	plot_straightened_mode.py

## Notation
-	omega — mode frequency
-	gamma_d — continuum damping
-	ntor — toroidal mode number
-	n_m / nhar — number of poloidal harmonics
-	n_r — number of radial points
-	M — half-width of straightened ridge window
-	r0 — radial centroid
-	dr — quantile width (10–90% energy span)

## Evaluation protocol
### RF
-	Previous expanded 10-shot OOF check:
- CM = `[[1447, 29], [64, 585]]` → accuracy 0.956
- GOOD precision/recall/F1 = 0.953 / 0.901 / 0.926
-	Used for label validation with OOF suspect lists
### CNN
-	Performance sensitive to seed + learning rate
-	Previous expanded 10-shot raw-CNN check:
- CM = `[[290, 5], [8, 121]]` -> accuracy 0.969
- GOOD precision/recall/F1 = 0.960 / 0.938 / 0.949
-	Previous four-shot TAE-like retraining used threshold 0.5 for all CNN confusion matrices
-	All three CNNs were comparable on the previous four-shot list, with best accuracy ~0.95-0.96

## Major updates
1) Straightened CNN representation
    - Replaced raw (m,r) input with ridge-aligned representation:
        -	compute m_c(r) via weighted mean
        -	apply median filter + slew limiter
        -	extract (2M+1, r) window
    - Result:
        -	removes dependence on nhar
        -	focuses on physical ridge
        -	improved CNN accuracy from ~90% → ~94–96%
 
2) sort_shot.py (major new component)
    - New post-processing script for shot-level mode selection:
    - Functionality:
        -	works with RF, CNN, HybridCNN outputs
        -	groups modes by close frequency
        -	compares mode structure using:
        -	signed ridge profile
        -	cosine similarity
        -	radial centroid r0
        -	quantile width dr
    - Similarity criteria: sim_tol = 0.9, r_tol   = 0.1, width_tol = 0.05
    - Outcome:
        -	sorts modes for the whole shot, generate lists, move out ‘bad’ modes (optional)
        -	identifies duplicate / near-duplicate modes for ‘good’ modes
        -	retains highest-scoring representative
        -	preserves distinct radial branches
    - Outputs:
        -	cluster_report.txt — detailed clustering info
        -	cluster.csv — paths for mode clusters (for inspection)
    - Validated on all 4 shots.

## Known issues / fixes
•	Feature mismatch in find_rf_disagreements.py → fixed
•	Missing datcon handling → warn once, disable continuum features

## Current tasks
- Keep `training_labels/additions/tae_like_3new.csv` out of training until
  `nstxuG121123N75` is recalculated and its labels are reviewed again.
- Use the current RF and M100 raw-CNN checkpoints through `sort_shot_mixed.py`
  for NSTX-U E-like shot classification and NOVA-C growth-rate candidate
  selection.
- Keep NSTX-U G-case shots out of routine production sorting for now. Decide
  whether to retune the G-shot policy with more same-regime examples,
  additional gap-geometry features, or both.
- Recheck the three new G-case shots after corrected continuum files arrive.
- Retrain straightened CNN and hybrid CNN on the expanded active list if they are still useful for comparison.
 
## Next tasks
- Add EAEs (second gap) more deeply into training / continuum features.
- Extend training to broader frequency range.
- Investigate surrogate / autoencoder for mode structure.

## Environment / portability
- Tested on:
    -	NERSC Perlmutter ✅ (pytorch, GPU)
    -	PPPL Flux ✅ (CPU inference in `/p/hym` conda env)
- Cross-cluster inference validation:
    - RF model output matched between Flux and Perlmutter
    - Perlmutter GPU-trained `cnn_raw`, `cnn_straightened`, and `cnn_hybrid`
      checkpoints produced identical `cnn_classify.py` outputs on Flux and
      Perlmutter when using matching Torch / NumPy / scikit-learn versions

## Interpretation of labels
-	Good: smooth, physical AE structure, reasonable continuum interaction
-	Bad: spiky, numerical, boundary artifacts
- Ambiguities: often near continuum crossings
 
## Current understanding
-	RF is robust and reliable baseline
-	CNN (straightened) captures structure very well
-	HybridCNN useful but not optimized yet
-	Signed ridge profile + quantile width provides a physically meaningful similarity metric
-	sort_shot.py successfully sorts good/bad and removes duplicates without merging distinct radial modes

### Project state (2026-04-02)
Codex: Implemented the new shared CNN classifier in scripts/cnn_infer_common.py and scripts/cnn_classify.py. It supports straightened and hybrid checkpoints, auto-detects hybrid from checkpoint contents, falls back to legacy preprocess defaults (M=8, R_target=201, center_power=2.0, median_k=3, max_step=2) when older checkpoints are missing metadata, and exposes reusable loading/prediction helpers for other scripts.

I also updated scripts/cnn_straightened.py and scripts/cnn_hybrid.py so future checkpoints save preprocess, flat preprocess keys, model_type, and checkpoint_version. scripts/sort_shot.py now accepts either RF .joblib models or CNN .pt checkpoints, and I refreshed scripts/README.md plus docs/project_state.md.

### Project state (2026-04-12)
Codex: Added `src/tae_eae_features.py` plus `scripts/split_tae_eae.py` to split mixed TAE/EAE mode lists using the upper TAE gap boundary (`high2_full`) from `datcon<N>`. The new workflow computes `signed_delta` and `fraction_below_upper2`, writes separate below/above CSVs, preserves original input columns, and records failures in a full output CSV instead of silently forcing bad rows into either class.

Summary on TAE/EAE issue: A robust practical separation of TAE-like and EAE-like modes was obtained using two upper-gap metrics: fraction_below_upper2 and signed_delta. The signed_delta value is the weighted mean of sqrt(upper2) - omega normalized by its weighted RMS. 

Modes with fraction_below_upper2 > 0.5 were classified as TAE-like; modes with fraction_below_upper2 < 0.4 and signed_delta < -0.1 were classified as EAE-like; intermediate cases were marked as mixed and included in the TAE-like set to avoid losing marginal TAEs. This recovered all labeled TAEs while keeping clear EAEs separate, and restored the RF classifier performance to near the original TAE-only level.

### 2026-04-20
- Fixed seed generation issue for cnn_straightened.py
- Updated results for new tae_like.csv list (1085 modes):
    - cnn_raw:          best accuracy=0.96, c.matrix:[[127,4][4,81]]
    - cnn_straightened: best accuracy=0.95, c.matrix:[[126,5][8,77]]
    - cnn_hybrid:       best accuracy=0.96, c.matrix:[[129,2][6,79]]
    - RF:               accuracy=0.94, c.matrix= [[62,4][3,40]]

        === Feature Importances === 
        - delta2_eff 0.1140 
        - W_star 0.1068 
        - max_abs_d1_abs 0.1000 
        - S 0.0946 
        - std_amp 0.0916

Updated results for new eae_like.csv list (2042 modes):

    - cnn_raw:          best accuracy=0.91, c.matrix:[[323  17][ 19  48]]

        Classification report:

                 precision    recall  f1-score   support
        - bad       0.94      0.95      0.95       340
        - good       0.74      0.72      0.73        67

    - RF:               accuracy=0.94, c.matrix:[[162 9] [ 3 31]] 

        Classification report (test): 

          precision recall f1-score support 
       -  0  0.98       0.95   0.96     171 
    -  1  0.78       0.91   0.84.    34

        === Feature Importances === 
       - rad_loc 0.1644 
       - rad_width 0.0804 
       - mean_abs_d2_mode 0.0787 
       - gamma_d 0.0611 
       - ntor 0.0540

### 05/09/26 
TAE/EAE sorting is solved and mixed_branch has been merged back to main

### 2026-05-10
- Retrained / rechecked the good-bad classifiers on `training_labels/tae_like_train.csv` using threshold 0.5 for CNN evaluation.
- RF: done; results are identical to the previous check on 2026-04-14.
- CNN_raw: best accuracy=0.96, CM=[[126 5][4 81]], threshold=0.5.
- CNN_straightened: best accuracy=0.95, CM=[[126 5][6 79]], threshold=0.5.
- CNN_hybrid: best accuracy=0.96, CM=[[129 2][6 79]], threshold=0.5.
- All checked models are working as expected on the updated TAE-like training pool.

### 2026-05-10
Codex: Promoted `cnn_raw.py` to the same operational path as the other CNN
models. The raw trainer now has `argparse` help, accepts `--train_csv` and
`--data_dir`, resamples the radial grid to `R_target` before padding/cropping
the harmonic axis, and writes checkpoint metadata with `model_type=cnn_raw`.
The shared `cnn_classify.py` / `cnn_infer_common.py` path and `sort_shot.py`
now support raw CNN checkpoints directly.

Codex: Added `ReduceLROnPlateau` to `cnn_raw.py` with the same fixed scheduler
settings as `cnn_straightened.py` and changed the default initial learning rate
to `2e-2`. The initial LR remains adjustable with `--lr`.

User LR sweep for `cnn_raw.py` after adding the scheduler:
- `lr=0.005`: best accuracy=0.9398, CM=[[128 3][10 75]]
- `lr=0.01`: best accuracy=0.9491, CM=[[129 2][9 76]]
- `lr=0.02`: best accuracy=0.9491, CM=[[127 4][7 78]]
- `lr=0.03`: best accuracy=0.9352, CM=[[127 4][10 75]]
- `lr=0.05`: unstable / stalled near majority-class accuracy after early epochs

Because the downstream NOVA-C workflow should avoid throwing away potentially
strongly unstable GOOD modes, false negatives are more costly than false
positives. The raw CNN default initial learning rate was therefore changed to
`0.02`, which kept the same best accuracy as `0.01` on this split but reduced
GOOD-mode false negatives.

### 2026-05-11
Codex: Added shared Torch device diagnostics for the CNN trainers. The scripts
now print CUDA availability, `CUDA_VISIBLE_DEVICES`, GPU name, and free/total
memory before model allocation. `cnn_raw.py` accepts `--device`, and
`cnn_raw.py`, `cnn_straightened.py`, and `cnn_hybrid.py` honor
`NOVA_TORCH_DEVICE` so Perlmutter runs can force `cpu`, `cuda`, or `cuda:0`
without editing source files. `scripts/README.md` now includes a Perlmutter
interactive `srun` example and OOM triage notes.

### 2026-05-12
Codex: Added Flux path configs for both shells:
`configs/paths/nova_paths.flux.csh` for Flux's default `tcsh`, and
`configs/paths/nova_paths.flux.sh` for bash. They point the old TAE-only
dataset at `/u/ebelova/NOVA_old/data_tae`, point the mixed TAE+EAE dataset at
`/p/hym/ebelova/NOVA/data_mixed`, default `NOVA_DATA` to the mixed dataset,
and set `NOVA_TORCH_DEVICE=cpu` for Flux CPU runs. They also add CPU helpers
mirroring the NERSC CNN/sort helpers without requesting GPUs. Flux still needs
`module load anaconda3` plus a conda environment with PyTorch installed for CNN
training / inference; after loading Anaconda, `tcsh` users need to source
`` `conda info --base`/etc/profile.d/conda.csh `` before `conda activate`,
while bash users source `$(conda info --base)/etc/profile.d/conda.sh`.

### 2026-05-13
Codex: Made `cnn_classify.py` / `cnn_infer_common.py` more tolerant of
Perlmutter-trained legacy CNN checkpoints. The loader now accepts payloads with
`state_dict`, plain model `state_dict` payloads, and uses checkpoint filenames
containing `raw`, `straightened`, or `hybrid` as last-resort model-kind hints.
Generic legacy filenames can still be loaded by passing `--model_kind`
explicitly.

### 2026-05-14
Codex: Updated the Flux path configs after moving the active Flux workflow to
`/p/hym`. The default tcsh repo path is now
`/p/hym/ebelova/NOVA/NOVA_modes`, the Flux work root is
`/p/hym/ebelova/NOVA`, mixed data defaults to
`/p/hym/ebelova/NOVA/data_mixed`, models default to
`/p/hym/ebelova/NOVA/models_flux`, and the old TAE-only data remains at
`/u/ebelova/NOVA_old/data_tae`. The bash Flux config mirrors the same data,
model, and run-directory defaults while still resolving `NOVA_REPO` from the
sourced file.

### 2026-05-15
User validation on Flux: the `/p/hym` conda environment now matches the
Perlmutter runtime versions for Torch `2.8.0`, NumPy `2.1.2`, and
scikit-learn `1.7.2`. RF inference output matched between Flux and Perlmutter,
and copied Perlmutter GPU-trained checkpoints for `cnn_raw`,
`cnn_straightened`, and `cnn_hybrid` all produced identical
`cnn_classify.py` outputs on Flux CPU and Perlmutter.

Codex: Updated the Flux configs to keep package caches and user-level Python
state out of `/u/ebelova` by setting `XDG_CACHE_HOME`, `XDG_CONFIG_HOME`,
`XDG_DATA_HOME`, `XDG_STATE_HOME`, `PIP_CACHE_DIR`, `MPLCONFIGDIR`, and
`PYTHONUSERBASE` under `/p/hym`. The Flux setup instructions now use conda's
`CONDA_PKGS_DIRS` environment variable name consistently.

### 2026-05-17
Codex: Added `scripts/sort_shot_mixed.py` for one-pass processing of mixed
TAE/EAE shots. The new workflow validates files, routes valid modes into
TAE-like / mixed / EAE-like groups with the existing normalized upper-gap
scalars, scores TAE-side modes with the shared RF path plus raw CNN inference,
combines scores with gold/silver/borderline tiers, and reuses the existing
close-frequency deduplication logic from `sort_shot.py`. It writes final good,
bad, QC-flagged, EAE-like, rejected, all-mode, per-shot, per-`n`, and
frequency-cluster audit outputs without moving source files.

### 2026-05-24
Codex: Clarified `sort_shot_mixed.py` summary counts. The fused GOOD count
before close-frequency duplicate removal is now `n_good_before_clustering`;
`n_final_good` now means the post-clustering GOOD count that matches
`good_tae_final.csv`.

Codex: Fixed close-frequency post-processing in `sort_shot.py` so modes cannot
be merged through chained frequency clusters when their direct pairwise
frequency spacing exceeds `--rel_freq_tol`. Cluster reports now include
pairwise `rel_domega` and `freq_close` values for each structural comparison.

Codex: Changed `sort_shot_mixed.py` so `shot_summary.csv` is a vertical
two-column key/value file for easier reading. The previous one-row summary
layout is still written as `shot_summary_wide.csv`.

Codex: Combined the per-`n` RF and CNN probability histograms into a compact
side-by-side diagnostic plot, `hist_p_good_by_n.png`.

Codex: Added `--refit_full_before_save` to `scripts/cnn_raw.py`. The raw CNN
still uses the stratified held-out split to choose `best_epoch` and report test
metrics, but with this option it trains a fresh final model on the full labeled
CSV for `best_epoch` epochs before saving. Checkpoints now record split sizes,
`saved_training_scope`, `final_train_size`, `final_train_epochs`, and whether
the full-data refit was used.

Codex: Added optional labeled-shot evaluation to `sort_shot_mixed.py` via
`--label_csv`. When labels are supplied, the script writes RF-only, CNN-only,
and combined-policy confusion matrices/classification reports plus compact
summary and per-mode evaluation CSVs.

Codex: Added `--rf_score_weight` and `--cnn_score_weight` to
`sort_shot_mixed.py`. These originally controlled the weighted `p_avg` used in
fallback fusion decisions and duplicate-clustering scores; after the
2026-05-27 RF-leaning policy update, these weights control the clustering score
only.

User validation after retraining `cnn_raw.py` with `--refit_full_before_save`:
the deployment raw-CNN checkpoint was trained on the full labeled TAE-like CSV
after held-out epoch selection, matching the RF model's full-data deployment
style. On labeled-shot in-sample sanity checks, the combined RF+CNN policy in
`sort_shot_mixed.py` produced:

- `nstx_120113`: CM `[[125, 0], [0, 49]]`, accuracy `1.000`.
- `nstx_135388`: CM `[[182, 2], [2, 194]]`, accuracy `0.9895`.
- `nstx_141711`: CM `[[152, 1], [2, 101]]`, accuracy `0.9883`.
- `nstxu_204202`: CM `[[197, 0], [5, 73]]`, accuracy `0.9818`.

The combined policy gives the best overall result on these labeled shots vs RF-only or CNN-only.

These are in-sample pipeline-consistency checks: they confirm that
`sort_shot_mixed.py`, RF inference, full-refit raw-CNN inference, TAE/EAE
routing, score fusion, and reporting agree with the current labeled training
set. They should not be interpreted as generalization estimates; leave-one-shot
out or other held-out-shot validation is still required for that.

### 2026-05-25
Codex: Added optional `--pos_weight` support to `scripts/cnn_raw.py` for LOSO
and other imbalanced/collapse-prone raw-CNN training runs. The positive class
is `good`; `--pos_weight auto` computes `n_bad/n_good` from the active training
labels, while a positive numeric value can be supplied manually. The default
remains unweighted. With `--refit_full_before_save`, the split-training phase
uses the split-derived auto value and the final full-CSV refit recomputes the
auto value from the full training CSV. Checkpoints record the requested
argument plus the split and final numeric weights.

User leave-one-shot-out checks using the four labeled TAE-like shots:

- Held out `nstx_120113`: RF accuracy `0.9425`, CNN accuracy `0.9253`,
  combined-policy accuracy `0.9598`, RF/CNN agreement `0.9253`.
- Held out `nstx_135388`: RF accuracy `0.9079`, CNN accuracy `0.8684`,
  combined-policy accuracy `0.9158`, RF/CNN agreement `0.8560`.
- Held out `nstx_141711`: RF accuracy `0.9023`, CNN accuracy `0.7891`,
  combined-policy accuracy `0.8359`, RF/CNN agreement `0.8443`. A later
  raw-CNN retry with `lr=0.03` and `M_target=65` improved CNN accuracy to
  `0.8906` and combined-policy accuracy to `0.8945`.
- Held out `nstxu_204202`: RF accuracy `0.9455`, CNN accuracy `0.8873`,
  combined-policy accuracy `0.9055`, RF/CNN agreement `0.9055`.

Initial interpretation: RF is the most stable LOSO baseline. The equal-weight
RF+raw-CNN combined policy improved over RF for `nstx_120113` and
`nstx_135388`, but underperformed RF for `nstx_141711` and `nstxu_204202`
because the raw CNN added extra false positives in those held-out cases. Raw
CNN generalization is sensitive to learning rate, input `M_target`, seed, and
class balance, so further CNN tuning should use LOSO-average performance rather
than per-shot tuning.

Codex: Added deployment/testing CLI parity to `cnn_straightened.py` and
`cnn_hybrid.py`: both now accept `--train_csv`, `--data_dir`, `--model_out`,
`--device`, `--cache_data`, and `--refit_full_before_save`, along with their
existing preprocessing and training knobs. The full-refit behavior matches
`cnn_raw.py`: held-out split metrics still select `best_epoch`, then a fresh
final model is trained on the full training CSV before saving. For hybrid
checkpoints, scalar normalization statistics are recomputed from the full CSV
for the final refit. `sort_shot_mixed.py` now loads CNN checkpoints with
`--cnn_model_kind auto` by default, so raw, straightened, and hybrid CNNs can
be compared in the RF+CNN mixed-shot policy. Labeled evaluation outputs now use
generic `cnn` / `cnn_label` names, with the loaded checkpoint kind recorded
separately in `model_evaluation_report.txt`.

### 2026-05-27
Codex: Updated the default `sort_shot_mixed.py` RF/CNN fusion rule after LOSO
checks showed RF is the more stable ranker and equal RF/CNN fusion can add raw
CNN false positives. The policy is now RF-leaning with only a high-confidence
CNN rescue:

- `gold_good`: `p_rf_good >= 0.7` and `p_cnn_good >= 0.6`
- `silver_good`: `p_rf_good >= 0.5` and `p_cnn_good >= 0.5`
- `flagged_cnn_rescue`: `p_rf_good >= 0.4` and `p_cnn_good >= 0.9`
- `gold_bad`: `p_rf_good < 0.2` and `p_cnn_good < 0.2`
- `silver_bad`: `p_rf_good < 0.4` and `p_cnn_good < 0.4`
- `flagged_rf_only_good`: `p_rf_good >= 0.5`
- all remaining cases are `bad` with
  `tier=flagged_borderline_or_disagreement`

`p_avg` remains in the outputs and is still used as the close-frequency
duplicate-clustering score, with `--rf_score_weight` and `--cnn_score_weight`
controlling that score only rather than fallback label decisions.

### 2026-05-28
User decision: the current full-refit RF/CNN models plus the RF-leaning
`sort_shot_mixed.py` fusion policy are the operational baseline for now. The
next model-improvement step is to add more labeled NSTX-U shots, then retrain
and revalidate the RF and CNN models on the expanded training set.

Rationale: four-shot LOSO checks show RF is currently the most stable
held-out-shot baseline, while CNN checkpoints can help when used as a limited
high-confidence rescue signal. The present model is good enough to serve as
the main sorting path, but broader NSTX-U training coverage is needed before
expecting more robust NSTX-U generalization.

Current follow-up items:

- label additional NSTX-U shots and merge the reviewed labels into the
  TAE-like training pool;
- retrain RF, raw CNN, straightened CNN, and hybrid CNN with the expanded
  labeled set;
- rerun LOSO or held-out-shot checks, especially on NSTX-U shots;
- re-evaluate the RF-leaning fusion thresholds in `sort_shot_mixed.py` after
  retraining.

### 2026-06-05
User update: six additional NSTX-U shots now have staged TAE-like label lists
in the shared `nova2/metadata` area, but the labels still need review before
they are merged into `training_labels/tae_like_train.csv`.

Codex check:
- `training_labels/tae_like_train.csv` remains the active four-shot TAE-like training
  list: 1085 rows, 426 `good`, 659 `bad`, all resolving under `$NOVA_DATA`.
- The old root-level four-shot label files have been moved into
  `training_labels/old_4shots_tae_only_labels/` and
  `training_labels/old_4shots_mixed_labels/`.
- The cleaned six-shot staged list has 1040 rows: 284 `good`, 756 `bad`.
- The not-cleaned six-shot staged list has 1041 rows and one duplicate mode:
  `nstxuG142301H47/N8/egn08w.1092E+02`.
- All cleaned staged-label paths resolve to existing files under `$NOVA_DATA`
  by `shot/N/file` suffix.
- Per-shot TAE/EAE split outputs exist for all six staged NSTX-U shots. Their
  TAE-like outputs contain 1050 rows total. The 10 split TAE-like modes that
  are not present in the cleaned staged label list were marked `skip` during
  labeling and can be ignored for training.

Current workflow decision: retrain RF on the full active
`training_labels/tae_like_train.csv` list and use that model to help inspect the
six staged NSTX-U labels. Keep the staged labels out of the canonical training
CSV until the review is done.

Codex retrained `models/nova_mode_classifier.joblib` on the full active
`training_labels/tae_like_train.csv` list. The RF script loaded 1085 modes, reported
5-fold CV accuracies `[0.9401, 0.9217, 0.8940, 0.9078, 0.9401]`
with mean CV accuracy `0.9207`, then ran its 10% held-out sanity check:
CM `[[62, 4], [3, 40]]`, accuracy `0.94`. After that check, the script refit
on the full 1085-mode list and saved the deployment model.

### 2026-06-06
User finished checking and cleaning the six-shot NSTX-U label list. The cleaned
list is now `training_labels/additions/tae_like_6new.csv`.

Codex enriched `training_labels/additions/tae_like_6new.csv` to match the full
`tae_like_train.csv` schema: `path`, `validity`, `family`, `signed_delta`,
`fraction_below_upper2`, `gap_region`, and `error`. Split metadata was restored
from `/global/cfs/cdirs/m314/nova2/metadata/*_tae_eae_split/tae_like.csv` by
matching `shot/N/file` suffixes and writing relative `$NOVA_DATA` paths.
The `family` value is `tae` for `good` rows and `none` for `bad` rows.

Validation after enrichment:
- 1040 rows plus header.
- labels: 252 `good`, 788 `bad`.
- family values: 252 `tae`, 788 `none`.
- gap regions: 950 `below_upper2`, 90 `mixed`.
- no missing paths under `$NOVA_DATA`.
- no empty required metadata fields.

Codex merged `training_labels/additions/tae_like_4old.csv` and
`training_labels/additions/tae_like_6new.csv` into the active
`training_labels/tae_like_train.csv` list. The merged list preserves the full schema,
keeps old rows first and appends the reviewed six-shot NSTX-U rows, and uses
relative `$NOVA_DATA` paths throughout.

Validation after merge at that time:
- 2125 rows plus header.
- labels: 678 `good`, 1447 `bad`.
- family values: 675 `tae`, 1447 `none`, 3 `eae`.
- gap regions: 1945 `below_upper2`, 180 `mixed`.
- shots: 4 original shots plus 6 reviewed NSTX-U shots.
- no duplicate paths.
- no missing paths under `$NOVA_DATA`.
- no empty required metadata fields.

### 2026-06-07
User retrained RF and raw CNN on the cleaned expanded 10-shot TAE-like list.
The active top-level model files are now:

- `models/nova_mode_classifier.joblib` — expanded-set RF.
- `models/nova_cnn_raw.pt` — expanded-set raw CNN.

The previous four-shot RF, raw CNN, straightened CNN, hybrid CNN, and LOSO
checkpoints were moved under `models/old_4shots_models/`.

Expanded RF label-audit result from `scripts/rf_oof_check.py` on
`training_labels/tae_like_train.csv`:

- labels loaded: 2125 modes, 678 `good`, 1447 `bad`.
- feature matrix: `(2125, 22)`.
- OOF CM: `[[1404, 43], [93, 585]]`.
- accuracy: 0.94.
- BAD precision/recall/f1: 0.94 / 0.97 / 0.95.
- GOOD precision/recall/f1: 0.93 / 0.86 / 0.90.

Expanded raw-CNN held-out result:

- held-out size: 424.
- CM: `[[288, 7], [14, 115]]`.
- accuracy: 0.95.
- BAD precision/recall/f1: 0.95 / 0.98 / 0.96.
- GOOD precision/recall/f1: 0.94 / 0.89 / 0.92.

Interpretation: raw CNN is now the strongest checked expanded-set classifier,
especially for GOOD-mode recall. The existing RF-leaning `sort_shot_mixed.py`
fusion policy was chosen from four-shot LOSO behavior and should be revalidated
or retuned with the expanded RF/raw-CNN models.

### 2026-06-08
Renamed the canonical labeled TAE-like training set from
`training_labels/tae_like.csv` to `training_labels/tae_like_train.csv`.
This avoids confusion with generated `tae_like.csv` files written by
`split_tae_eae.py` and `sort_shot_mixed.py` in output directories.

Updated `NOVA_TRAIN_CSV`, `NOVA_TRAIN_CSV_TAE`, the raw-CNN fallback default,
and README examples to use `training_labels/tae_like_train.csv`. The component
lists are now kept under `training_labels/additions/`, including
`training_labels/additions/tae_like_4old.csv` and
`training_labels/additions/tae_like_6new.csv`.

Added `scripts/run_loso_10.py` and `scripts/run_loso_10.sbatch` to orchestrate
the expanded 10-shot LOSO check. The workflow creates per-held-out-shot train
and test lists from `training_labels/tae_like_train.csv`, retrains RF and
raw CNN per fold, runs `sort_shot_mixed.py` on each held-out shot, and
aggregates RF-only, CNN-only, and combined-policy metrics under
`outputs/loso_10/`. Heavy checkpoints and logs go under `$NOVA_RUN/loso_10`
or `$SCRATCH/nova_s/loso_10`.

Generated the 10 LOSO split lists in `outputs/loso_10/folds/<shot>/`. A first
12-hour GPU batch submission (`54165050`) was cancelled while pending because
the expected RF/CNN/sorter fold runtime is only a few minutes per fold, and the
long walltime can hurt queue priority. Prefer the 4-hour interactive GPU run
documented in `scripts/README.md`; the batch wrapper now also requests 4 hours.
The 10-fold LOSO run completed in about 25 minutes. All RF, raw-CNN, and
`sort_shot_mixed.py` fold logs ended with `returncode=0`. Aggregate held-out
metrics from `outputs/loso_10/loso_model_evaluation_totals.csv`:

- RF: CM `[[1426, 50], [99, 550]]`, accuracy `0.9299`, GOOD precision/recall
  `0.9167 / 0.8475`, GOOD F1 `0.8807`.
- raw CNN: CM `[[1405, 71], [140, 509]]`, accuracy `0.9007`, GOOD
  precision/recall `0.8776 / 0.7843`, GOOD F1 `0.8283`.
- combined policy: CM `[[1423, 53], [96, 553]]`, accuracy `0.9299`, GOOD
  precision/recall `0.9125 / 0.8521`, GOOD F1 `0.8813`.

Interpretation: the existing RF-leaning combined policy is still very close to
RF-only on the expanded LOSO check, with a small GOOD-recall gain at the cost
of three additional false positives. CNN-only performs well on several folds
but is less stable across held-out shots; the worst raw-CNN held-out folds are
`nstxuE205052A01t022` (all 74 GOOD labels missed), `nstxuG121123K51`, and
`nstxuG142301H47`. Inspect per-fold `model_evaluation_report.txt` files before
changing the `sort_shot_mixed.py` fusion policy.

Follow-up inspection of `nstxuE205052A01t022` showed the raw CNN split-training
phase was not bad: its internal held-out split reached accuracy `0.95` and
GOOD recall `0.93`. The saved full-refit checkpoint then produced a constant
`p_cnn_good=0.325527...` for every held-out mode, so that fold is a CNN
full-refit/checkpoint failure rather than a clean model-generalization result.
Before making the fusion policy more CNN-heavy, rerun at least the problematic
folds with `--no-cnn_refit_full_before_save`, `--cnn_pos_weight auto`, or an
improved full-refit learning-rate schedule.

Added a top-level README subsection for the Flux classification-only workflow
for new NSTX-U shots. The instructions now explicitly tell users not to train
models for routine sorting, to pull the current repository models
(`models/nova_mode_classifier.joblib` and `models/nova_cnn_raw.pt`), and to run
`scripts/sort_shot_mixed.py` against shots under the DiTw data root with
per-shot outputs written outside the input data tree.

Fixed the Flux `tcsh` path config so `configs/paths/nova_paths.flux.csh` no
longer falls back to `/p/hym/ebelova/NOVA/NOVA_modes`. It now preserves an
explicit `NOVA_REPO` if set, otherwise resolves the current Git checkout with
`git rev-parse --show-toplevel`. The top-level README Flux recipe now tells
users to `cd /path/to/your/NOVA_modes`, source the config from that checkout,
and run `nova_env` to verify the active paths. This first changed the default
shared work-root to `/p/hym/$USER/NOVA` instead of Elena's Flux work directory;
the later cleanup below removes that shared work-root default entirely.

Cleaned the Flux path configs further after user reports of errors around the
derived `_NOVA_FLUX_WORK_ROOT`. The Flux configs now avoid that work-root
entirely and no longer set convenience-only `NOVA_DATA_TAE`, `NOVA_DATA_MIXED`,
`NOVA_RESULTS`, `NOVA_TRAIN_CSV_TAE`, or `NOVA_TRAIN_CSV_MIXED` values. They
keep only the repo/model/training-list paths needed by current examples plus
CPU/thread/cache/Python-path settings. `src/paths.py` now treats data/model/run
environment variables as optional so importing `NOVA_TRAIN_CSV` does not force
unused directories to exist.

### 2026-06-15
Created `training_labels/additions/tae_like_3new.csv` as a review-stage
combined label list for three newly labeled NSTX-U G-case shots:

- `nstxuG121123Q62`
- `nstxuG121123N75`
- `nstxuG142301Y93`

The source files are beside the shot directories in `$CFS/m314/nova2/data` and
use Flux/DiTw absolute paths:

- `nstxuG121123Q62_mode_labels_clean.csv`
- `nstxuG121123N75_mode_labels_clean.csv`
- `nstxuG142301Y93_mode_labels_clean.csv`

The combined list has 523 rows, 14 `good` and 509 `bad`, stored as
`path,validity` with paths relative to `$NOVA_DATA`. Validation found no
absolute paths, no duplicate paths, and no missing files under `$NOVA_DATA`.
This list is intentionally not merged into `tae_like_train.csv`; it is for
visual review with `viz/view_modes_csv.py` first.

### 2026-06-19
The three new G-case shots remain blocked from training-list merge while
corrected `datcon` files are prepared. Their mode structures and continuum
calculations used different resolutions, which can shift the inferred
continuum-crossing location away from the corresponding mode-structure spike.
After updated continuum files arrive, rerun the visual review and recompute
continuum-derived metadata before merging the affected N75 replacement rows.

Changed only the fresh full-data refit stage in `scripts/cnn_raw.py` to use
per-batch `OneCycleLR` and gradient clipping. Split training still uses
`ReduceLROnPlateau`. Full-refit defaults are:

- peak LR from `--lr` (`0.02`)
- initial LR `0.001` (`div_factor=20`)
- 10% warmup
- cosine annealing to `1e-5` (`final_div_factor=100`)
- gradient norm clipping at `1.0`
- Adam momentum cycling disabled

Reran the `nstxuE205052A01t022` LOSO fold using the exact existing nine-shot
training split of 1832 modes. Split training reproduced best accuracy `0.9481`
at epoch 50. The OneCycle full-refit loss decreased from `0.6269` to `0.0010`
instead of stalling near `0.62`.

Held-out sorter evaluation for the new full-refit checkpoint:

- CNN CM: `[[205, 14], [3, 71]]`
- accuracy: `0.9420`
- GOOD precision/recall/F1: `0.8353 / 0.9595 / 0.8931`
- previous constant `p_cnn_good=0.325527...` output is gone
- output directory: `outputs/loso_onecycle_nstxuE205052A01t022/`
- checkpoint/log directory:
  `$SCRATCH/nova_s/loso_onecycle_nstxuE205052A01t022/`

This validates the scheduler change on the previously failed fold only. Rerun
the complete LOSO set before replacing the aggregate CNN/fusion-policy result.

Added an explicit clipping-disabled state for the raw-CNN full refit:
`--full_refit_grad_clip_norm none`; `off` and `0` are accepted aliases. The
default remains gradient norm `1.0`.

Reran the same `nstxuE205052A01t022` fold with identical seed, data, epoch
selection, and OneCycle schedule, but clipping disabled. The full-refit loss
decreased to `0.1488` by epoch 10, then jumped back to approximately `0.623`
after the peak-LR phase and remained stalled. The checkpoint again produced a
constant score, now `p_cnn_good=0.313810...`, with CNN CM
`[[219, 0], [74, 0]]`.

Conclusion for this controlled one-fold ablation: OneCycleLR alone did not
prevent collapse at peak LR `0.02`; gradient clipping at norm `1.0` was the
factor distinguishing the successful run. No-clipping output:
`outputs/loso_onecycle_no_clip_nstxuE205052A01t022/`.

Added `--full_refit_scheduler {onecycle,constant}` and ran the remaining
constant-LR plus clipping ablation with the same fold, seed, and 50 full-refit
epochs. Constant `0.02` plus clipping did not collapse: final loss was
`0.0308`, and held-out CNN metrics were:

- CM `[[195, 24], [4, 70]]`
- accuracy `0.9044`
- GOOD precision/recall/F1 `0.7447 / 0.9459 / 0.8333`

This shows clipping alone is sufficient to prevent the observed collapse.
However, OneCycle plus clipping remains better on this fold: CM
`[[205, 14], [3, 71]]`, accuracy `0.9420`, and GOOD F1 `0.8931`. It reduced
false positives from 24 to 14 while also missing one fewer GOOD mode. Keep
OneCycle plus clipping as the current full-refit default; the constant option
remains available for controlled checks. Constant-plus-clipping output:
`outputs/loso_constant_clip_nstxuE205052A01t022/`.

Final scheduler cleanup: `cnn_raw.py` now uses one shared OneCycleLR plus
gradient-clipping recipe for both split training and the production full-data
refit. This supersedes the earlier split-`ReduceLROnPlateau` / refit-OneCycle
implementation and removes the constant full-refit option from the current
CLI. Both phases use:

- 80 epochs by default
- peak LR `0.02`
- initial LR `0.001`
- 10% warmup and cosine annealing to `1e-5`
- gradient norm clipping at `1.0`

The split phase still retains the best held-out checkpoint for evaluation.
When `--refit_full_before_save` is set, a fresh production model completes the
same configured 80-epoch recipe on all labels. The checkpoint records both the
best split epoch and the full-refit epoch count.

Targeted `nstxuE205052A01t022` LOSO result with the symmetric recipe:

- internal split best accuracy `0.9617` at epoch 43
- internal split CM `[[245, 6], [8, 107]]`
- full-refit loss ended at `0.0008`
- held-out-shot CNN CM `[[191, 28], [1, 73]]`
- held-out-shot accuracy `0.9010`
- GOOD precision/recall/F1 `0.7228 / 0.9865 / 0.8343`
- output: `outputs/loso_onecycle_both_nstxuE205052A01t022/`

This version strongly favors GOOD recall but has more false positives on this
fold than the earlier asymmetric OneCycle-refit experiment. Run the full LOSO
set with the symmetric recipe before changing production checkpoints or fusion
thresholds.

Completed the full 10-shot LOSO run with the symmetric OneCycleLR plus
gradient-clipping recipe in both split training and full-data refit. All 30
RF/CNN/sorter logs ended with `returncode=0`; all ten CNN refits completed 80
epochs, with final losses between `0.0000` and `0.0011`.

Aggregate results from
`outputs/loso_10_onecycle_both/loso_model_evaluation_totals.csv`:

- CNN: CM `[[1402, 74], [67, 582]]`, accuracy `0.9336`, GOOD
  precision/recall/F1 `0.8872 / 0.8968 / 0.8920`.
- combined policy: CM `[[1418, 58], [86, 563]]`, accuracy `0.9322`, GOOD
  precision/recall/F1 `0.9066 / 0.8675 / 0.8866`.
- RF: CM `[[1426, 50], [99, 550]]`, accuracy `0.9299`, GOOD
  precision/recall/F1 `0.9167 / 0.8475 / 0.8807`.

Compared with the earlier CNN LOSO result, the symmetric recipe reduced false
negatives from 140 to 67 while increasing false positives only from 71 to 74.
CNN is now best overall by accuracy, GOOD recall, and GOOD F1. Since GOOD
recall is the higher-priority metric, this supports retraining the production
raw-CNN checkpoint with the symmetric recipe.

Performance remains heterogeneous by shot group:

- original NSTX: CNN CM `[[468, 15], [22, 305]]`, GOOD recall `0.933`.
- NSTX-U E cases: CNN CM `[[344, 38], [12, 195]]`, GOOD recall `0.942`.
- NSTX-U G cases: CNN CM `[[397, 14], [23, 17]]`, GOOD recall `0.425`.

The G cases contain only 40 GOOD labels across the three held-out folds, one of
which has no GOOD labels. Treat their per-shot recall as high variance. The
current combined policy retains better aggregate G-case GOOD recall (`0.600`)
than CNN alone, but suppresses some CNN gains on NSTX and E-case shots. Retune
fusion only after deciding whether to optimize globally for GOOD recall or
retain extra protection for the sparse G-case regime.

Promoted the symmetric-recipe raw CNN to the active production checkpoint at
`models/nova_cnn_raw.pt`. The held-out split result was:

- CM `[[290, 5], [8, 121]]`
- accuracy `0.9693`
- GOOD precision/recall/F1 `0.9603 / 0.9380 / 0.9490`

The checkpoint metadata confirms a fresh full-data refit on all 2,125 modes
for 80 epochs using OneCycleLR with peak LR `0.02`, `div_factor=20`, 10%
warmup, cosine annealing, and gradient clipping at norm `1.0`. Full-refit loss
ended at `0.0008`. This replaces the earlier expanded-set raw-CNN checkpoint;
the completed symmetric 10-shot LOSO result above remains the generalization
check used for fusion-policy decisions.

Added prediction-collapse monitoring to `scripts/cnn_raw.py`. At the normal
epoch-reporting cadence, both split evaluation and full-data refit compute
prediction-health diagnostics but keep healthy checks silent. Starting at
epoch 5, warnings identify:

- zero predicted GOOD modes when GOOD labels are present;
- predicted GOOD fraction below `0.02` when GOOD labels are present;
- predicted GOOD fraction above `0.98` when BAD labels are present;
- `p_good` standard deviation below `0.001`.

Warnings include predicted/true GOOD counts and `p_good` summary statistics.
The full refit uses a non-shuffled evaluation loader for these checks. Its
final diagnostics and collapse status are saved as `final_prediction_health`
in the checkpoint, preventing another stalled model from looking healthy solely
because the training loop completed.

### 2026-06-20

Added an RF experiment for continuum-boundary-crossing features.

- `src/cont_features.py` now detects lower/upper boundary crossings without
  bridging NaN gaps, linearly interpolates crossing radius and peak-normalized
  radial mode energy, handles exact-zero runs once at their midpoint, and
  exposes diagnostic crossing records.
- The experimental schema appends seven deterministic features:
  `n_cross`, `r_star_max`, `W_star_max`, `W_star_sum`,
  `r_star_high_shear`, `W_star_high_shear`, and
  `W_star_high_shear_sum`.
- `src/mode_features.py` owns the canonical production 22-feature schema and
  the optional 28-feature all-crossings schema.
- `scripts/rf_train_classify.py` and `scripts/rf_oof_check.py` accept
  `--crossing-features` and `--r_shear0`. Experimental model and bundle names
  are separate, and the trainer refuses to replace
  `models/nova_mode_classifier.joblib` with a 28-feature model.
- Experimental checkpoints remain plain sklearn pipelines and store schema
  version, feature names, crossing-feature state, and `r_shear0` metadata.
- Added standard-library unit tests for interpolated and multiple crossings,
  exact-grid and zero-run behavior, NaN gaps, no-crossing defaults, tie
  handling, malformed shapes, 22/29 feature order, and active-checkpoint
  compatibility.

Real-data checks on the current 2,125-label list used the same shuffled
five-fold OOF splits and active RF pipeline template:

- legacy 22 features: CM `[[1448, 28], [71, 578]]`, accuracy `0.9534`,
  GOOD precision/recall/F1 `0.9538 / 0.8906 / 0.9211`
- all seven crossing features: CM `[[1442, 34], [72, 577]]`, accuracy `0.9501`,
  GOOD precision/recall/F1 `0.9444 / 0.8891 / 0.9159`

The full seven-feature bundle therefore does not improve aggregate OOF and
should not be promoted as-is. It helped some NSTX-U folds, including reducing
false negatives from 7 to 4 for `nstxu_204202`, but degraded several NSTX and
G-case folds. In a full experimental fit, `W_star_sum` and `W_star_max` ranked
third and fourth in RF importance, so the crossing signal itself is useful
even though the full bundle adds too much redundancy/noise.

A same-fold ablation found that adding only `W_star_max` improved the legacy
schema. Removing raw `omega` at the same time improved it further:

- previous 22 features: CM `[[1448, 28], [71, 578]]`, accuracy `0.9534`,
  GOOD precision/recall/F1 `0.9538 / 0.8906 / 0.9211`
- previous features plus `W_star_max`, minus `omega`:
  CM `[[1447, 29], [64, 585]]`, accuracy `0.9562`,
  GOOD precision/recall/F1 `0.9528 / 0.9014 / 0.9264`

Several additional crossing-based continuum features were tested, including
outer-crossing/high-shear variants. These did not improve OOF performance and
were strongly correlated with `W_star_max`. Replacing legacy `W_star` with
`W_star_max` also performed worse; the two features carry complementary
normalizations and should both be retained.

The best RF configuration retained the legacy features, removed `omega`, and
added only `W_star_max`. This 22-feature schema is now
`rf_w_star_max_22_v2`. The active
`models/nova_mode_classifier.joblib` and
`nova_mode_classifier_bundle.joblib` were retrained on all 2,125 labels with
this schema. The broader crossing calculations remain available for
experiments and plotting but are not production RF inputs.

`viz/view_modes_csv.py` and `scripts/label_modes_fast.py` now display both the
legacy closest-approach `r_star` and `r_star_max`, the boundary crossing with
the largest peak-normalized radial mode energy.

Archived the unused one-off `utils/debug_mode.py` as
`legacy/debug_mode.py`; it relied on a hardcoded path and did not track the
current RF schema metadata.

### 2026-06-21

Changed `scripts/label_modes_fast.py` to plot all `nhar` poloidal harmonics by
default instead of silently limiting the mode-structure panel to the strongest
20. This avoids hiding weaker edge harmonics near continuum crossings. The new
optional `--max-harmonics N` argument restores an explicit strongest-`N` cap
for crowded plots. Startup output describes the active policy, and each plot
title reports the exact `plotted/total` harmonic count.

### 2026-06-22

Investigated replacement continuum files for `nstxuG121123N75/N3`. The active
`datcon3` is byte-identical to
`nstxuG121123N75_new/N3/datcon3`, and checked recomputed `egn03w.*` mode files
are also byte-identical between the active and `_new` directories. The modes
have `nr=201`, while `datcon3` covers indices 3 through 199, so this is not an
obvious copied-file or point-count mismatch.

For the active files, `egn03w.1171E+02` has legacy `r_star=0.560` and an
interpolated lower-boundary crossing at `0.5553`, so that mode is aligned near
`0.55`. The reported `r_star` near `0.42` referred instead to
`egn03w.1445E+02`, whose displayed continuum marker is consistent with the
current file and confirms the mismatch described below.

The mismatch remains real for following labeled modes:

- `egn03w.1445E+02`: `r_star=0.445`, strongest curvature near `0.560`
- `egn03w.1473E+02`: `r_star=0.440`, strongest curvature near `0.555`
- `egn03w.1951E+02`: `r_star=0.350`, strongest curvature near `0.450`
- `egn03w.1982E+02`: `r_star=0.335`, strongest curvature near `0.445`
- `egn03w.2008E+02`: `r_star=0.315`, strongest curvature near `0.440`
- `egn03w.2027E+02`: `r_star=0.300`, strongest curvature near `0.435`

The replacement directory also contains `datcon_gf.txt` with a nonuniform
`sqrt(Flux_toroid)` coordinate that current tools do not read. Mapping both
crossing and mode indices through that coordinate changes their displayed
radial values but does not remove the index-level separation. Keep this shot
out of `tae_like_train.csv` until the NOVA continuum/mode-grid provenance is
confirmed.

Direct comparison of `N3/old_datcon3` and `N3/new_datcon3` shows that the
replacement changes continuum frequency levels but does not radially shift
the profiles. Pointwise profile correlations are `0.99998` for the lower
branch and `0.99829` for the upper branch, with best correlation at zero index
shift. In plotted frequency units, the new lower branch is higher by `0.0155`
on average and the new upper branch by `0.0704`.

For the lower-boundary crossings relevant to the first labeled `N3` modes, the
new file moves crossings outward only modestly:

- `egn03w.1171E+02`: `0.5505 -> 0.5553`
- `egn03w.1445E+02`: `0.4389 -> 0.4414`
- `egn03w.1473E+02`: `0.4342 -> 0.4367`
- `egn03w.1951E+02`: `0.3354 -> 0.3466`
- `egn03w.1982E+02`: `0.3156 -> 0.3306`
- `egn03w.2008E+02`: `0.2958 -> 0.3127`
- `egn03w.2027E+02`: `0.2804 -> 0.2970`

Thus the replacement moves several crossings slightly toward the visible
structure features, but the `0.0025-0.017` changes are much smaller than the
remaining roughly `0.10-0.12` separations.

The same alignment concern was checked for the recomputed
`nstx_135388_new` modes. All old and replacement `datcon<N>` files cover
indices 3 through 199 on the same 201-point radial grid. The supplied
`datcon_gf_old.txt` and replacement `datcon_gf.txt` radial coordinates are
identical for every `N1` through `N10`.

The two replacement-shot workflows differ materially:

- `nstxuG121123N75_new`: all 772 mode files across `N1-N10` have the same
  filenames, counts, decoded shapes, frequencies, damping values, and bytes as
  the corresponding files in `nstxuG121123N75`. Their newer timestamps are
  from copying; the TAE modes were not recalculated. Only the continuum files
  changed relative to the preserved `old_datcon<N>` files.
- `nstx_135388_new`: most matching mode files differ from
  `nstx_135388`, mode counts also differ, and many poloidal-harmonic counts
  changed. Examples include `N6` 101 to 106 harmonics and `N7` 101 to 123.
  These modes were genuinely recalculated, although a minority of matching
  files remain byte-identical.

For labeled `N6` and `N7` modes, replacing the continuum has essentially no
effect on the relevant crossings:

- no legacy `r_star` values changed on the displayed 0.005 grid;
- paired interpolated crossings changed by at most `0.00008` for `N6`;
- paired interpolated crossings changed by at most `0.00003` for `N7`.

The replacement mainly removes unphysical old tail spikes near the outer
boundary. Through the interior, the old and new `N6/N7` TAE-gap profiles
overlap visually.

The recomputed mode structures did change harmonic resolution while retaining
`nr=201`: matched labeled `N6` modes commonly changed from 101 to 106
harmonics, and matched `N7` modes from 101 to 123. Visual inspection found
multiple modes where both `r_star` and `r_star_max` lie far inside the sharp
outer structure, including:

- `N6/egn06w.3147E+02`: marker near `0.030`, sharp structure near `0.8-0.9`
- `N6/egn06w.3468E+02`: marker near `0.059`, sharp structure near `0.8-0.9`
- `N6/egn06w.5263E+02`: marker near `0.189`, sharp structure near `0.8-0.95`
- `N7/egn07w.3518E+02`: marker near `0.045`, sharp structure near `0.8-0.93`
- `N7/egn07w.4950E+02`: marker near `0.167`, sharp structure near `0.8-0.94`
- `N7/egn07w.5504E+02`: marker near `0.232`, sharp structure near `0.8-0.95`

An automated screening check found few such low-energy, displaced crossings
for matched labeled recomputed modes in `N1-N4`, but many from `N5` through
`N10`. This metric is only a review flag: a continuum crossing with negligible
mode energy need not produce a visible singularity. Still, the `N6/N7`
examples confirm that the replacement continuum did not resolve the apparent
mode/continuum alignment issue. Recheck `nstx_135388` labels, especially
`N5-N10`, before the next training-list cleanup or model retraining.

Likely root cause identified for `nstxuG121123N75`: modes for some toroidal
mode numbers were calculated with the wrong q profile. This is consistent
with the file audit above: the `_new` directory copied the original modes
unchanged while replacing their continuum files, so the mode structures and
continuum could represent different equilibrium inputs despite matching
radial dimensions. The affected case is now being recalculated. Do not review
or merge its staged labels until the replacement modes and matching continuum
files are available.

The cause of the `nstx_135388` alignment flags remains unresolved and should
be treated separately. Its `_new` modes were genuinely recalculated, so the
wrong-q explanation for `nstxuG121123N75` should not be assumed to apply
without additional provenance checks.

### 2026-06-29

Changed `scripts/label_modes_fast.py` to plot signed `xi_m(r)` harmonic
profiles by default, matching `viz/view_modes_csv.py`. The older absolute
amplitude view remains available with `--abs`; startup output and plot titles
now state the active amplitude convention.

### 2026-07-06

Data update: `nstxuG121123N75` has not been refreshed yet, so its earlier
continuum/mode-provenance concern remains blocking and its staged labels should
not be merged. This means `training_labels/additions/tae_like_3new.csv`
remains unfinished as a whole because it includes `nstxuG121123N75`.

`nstx_135388` has been updated in `$CFS/m314/nova2/data`, and a new shot
`nstxuG121123J38` has been added there. For both shots, all modes were labeled
and split into TAE-like and EAE-like lists under their respective
`*_tae_eae_split` directories. Quick consistency check of the split outputs:

- `nstx_135388`: `344` TAE-like modes, `825` EAE-like modes, `1169` total
  modes.
- `nstxuG121123J38`: `174` TAE-like modes, `446` EAE-like modes, `620` total
  modes.

Created review-stage labeled TAE-like lists in `training_labels/additions/` by
matching the per-shot split `tae_like.csv` rows to the corresponding
`*_mode_labels_clean.csv` files and converting Flux absolute paths to relative
`shot/N/file` paths:

- `training_labels/additions/tae_like_nstx_135388.csv`: `344` rows, `122` good,
  `222` bad after final manual review.
- `training_labels/additions/tae_like_nstxuG121123J38.csv`: `174` rows, `6` good,
  `168` bad after final manual review.

These files use the same full schema as `tae_like_train.csv`, with `family`
set to `tae` for `good` rows and `none` for `bad` rows. After manual review
and model-disagreement checks, these two lists are considered suitable for
training.

The earlier non-matching continuum-crossing issue is now understood as a
wrong-q-profile mode-calculation problem. The mismatch is fixed for modes
calculated with the correct q profile, so corrected-q-profile shots can be
used after their labels pass the usual review.

Ran inference-only RF/CNN checks on the two manually reviewed lists using the
active production checkpoints:

- RF: `models/nova_mode_classifier.joblib`, schema `rf_w_star_max_22_v2`
- CNN: `models/nova_cnn_raw.pt`, kind `cnn_raw`
- output directory: `outputs/review_2new_labels_20260706/`

Metrics against the current manual labels:

- `nstx_135388`, RF: CM `[[198, 24], [2, 120]]`, accuracy `0.9244`,
  GOOD precision/recall/F1 `0.8333 / 0.9836 / 0.9023`.
- `nstx_135388`, CNN: CM `[[195, 27], [1, 121]]`, accuracy `0.9186`,
  GOOD precision/recall/F1 `0.8176 / 0.9918 / 0.8963`.
- `nstxuG121123J38`, RF: CM `[[154, 14], [0, 6]]`, accuracy `0.9195`,
  GOOD precision/recall/F1 `0.3000 / 1.0000 / 0.4615`.
- `nstxuG121123J38`, CNN: CM `[[141, 27], [1, 5]]`, accuracy `0.8391`,
  GOOD precision/recall/F1 `0.1562 / 0.8333 / 0.2632`.

Combined over both review lists, RF found `40` model-vs-label disagreements
and CNN found `56`; RF and CNN disagreed with each other on `28` modes. The
refreshed review lists are:

- `rf_good_label_pred_bad_candidates.csv`: `2` rows, all from `nstx_135388`
- `cnn_good_label_pred_bad_candidates.csv`: `2` rows
- `rf_bad_label_pred_good_candidates.csv`: `38` rows
- `cnn_bad_label_pred_good_candidates.csv`: `54` rows
- `any_disagreements.csv`: `62` rows

No read, RF-feature, or CNN-inference errors occurred.

Merged the accepted two-shot update into `training_labels/tae_like_train.csv`:

- removed the old `nstx_135388` block: `380` rows, `185` good, `195` bad
- added refreshed `training_labels/additions/tae_like_nstx_135388.csv`:
  `344` rows, `122` good, `222` bad
- added `training_labels/additions/tae_like_nstxuG121123J38.csv`: `174` rows,
  `6` good, `168` bad

The active training list then had `2263` rows: `592` good and `1671` bad across
11 shots. Validation after the merge found the expected schema, no duplicate
paths, no absolute paths, no unresolved files under `$CFS/m314/nova2/data`, and
consistent family labels (`tae` for good, `none` for bad). As part of the
merge cleanup, `18` retained historical rows whose `family` values were
inconsistent with their `validity` labels were normalized to the current
convention.

At this point both RF and raw-CNN checkpoints still reflected the previous
2125-row / 10-shot training list, so retraining was pending before reporting
new production model metrics.

Split the usable part of `training_labels/additions/tae_like_3new.csv` into
`training_labels/additions/tae_like_2new.csv` by excluding `nstxuG121123N75`
and keeping only the already-reviewed `nstxuG121123Q62` and
`nstxuG142301Y93` rows. The new two-shot list restores the full training
schema from the per-shot split metadata in `$CFS/m314/nova2/metadata`:

- `nstxuG121123Q62`: `241` rows, `13` good, `228` bad
- `nstxuG142301Y93`: `106` rows, `1` good, `105` bad

Merged `training_labels/additions/tae_like_2new.csv` into
`training_labels/tae_like_train.csv`. The active training list now has `2610`
rows: `606` good and `2004` bad across 13 shots. Validation found the expected
schema, no duplicate paths, no absolute paths, no unresolved files under
`$CFS/m314/nova2/data`, and consistent family labels. The original
`training_labels/additions/tae_like_3new.csv` remains unmerged as a reference
only because it still includes blocked `nstxuG121123N75` rows.

Retrained RF on the current `training_labels/tae_like_train.csv` list and
saved:

- `models/nova_mode_classifier.joblib`
- `models/nova_mode_classifier_bundle.joblib`

RF training split check:

- test CM `[[198, 2], [7, 54]]`
- test accuracy `0.966`
- GOOD precision/recall/F1 `0.964 / 0.885 / 0.923`

RF 5-fold OOF audit:

- output: `outputs/rf_oof_13shots/`
- OOF CM `[[1967, 37], [91, 515]]`
- accuracy `0.951`
- GOOD precision/recall/F1 `0.933 / 0.850 / 0.889`
- strong suspect rows: `43`

Moved component and staged review lists under `training_labels/additions/` so
the root of `training_labels/` contains only the active
`tae_like_train.csv`, this README, and archive directories. The active training
CSV and its row contents were not changed by this reorganization.

User retrained the raw CNN on the current 13-shot active list and updated
`models/nova_cnn_raw.pt`.

Current raw-CNN held-out split check:

- test CM `[[394, 6], [18, 103]]`
- test accuracy `0.954`
- GOOD precision/recall/F1 `0.945 / 0.851 / 0.896`

This is broadly similar to the current RF split/OOF checks. Updated LOSO
checks are still needed before changing the RF/CNN fusion policy.

### 2026-07-07

Made the `cnn_raw.py` prediction-health checks warning-only to reduce training
log clutter. The script still checks split-test and full-fit predictions at
the normal epoch-reporting cadence, but it only prints when collapse/stalling
is detected: zero predicted GOOD modes with GOOD labels present, near-all-BAD
or near-all-GOOD predictions, or nearly constant `p_good`. The saved checkpoint
metadata still records `final_prediction_health`, now including exact
predicted/true GOOD counts.

Adapted `scripts/run_loso_10.py` for the current 13-shot active training list.
The filename is kept for compatibility, but the driver now infers the number
of held-out shots from `--train_csv`, defaults to `outputs/loso_<N shots>`, and
writes a `run_config.json` with the CNN shape/recipe metadata. For the current
raw-CNN harmonic-window check, run separate output/work roots:

- `outputs/loso_13_M54` with `--cnn_m_target 54`
- `outputs/loso_13_M100` with `--cnn_m_target 100`

Use those LOSO totals, rather than the single held-out split, before changing
the production raw-CNN `M_target` or the RF/CNN fusion policy.

The 13-shot `M_target=54` LOSO run completed under `outputs/loso_13_M54`.
Aggregate metrics from `loso_model_evaluation_totals.csv`:

- CNN: CM `[[1899, 105], [108, 498]]`, accuracy `0.918`, GOOD
  precision/recall/F1 `0.826 / 0.822 / 0.824`
- combined policy: CM `[[1948, 56], [116, 490]]`, accuracy `0.934`, GOOD
  precision/recall/F1 `0.897 / 0.809 / 0.851`
- RF: CM `[[1957, 47], [134, 472]]`, accuracy `0.931`, GOOD
  precision/recall/F1 `0.909 / 0.779 / 0.839`

Runtime was about 92 minutes. Log timestamps show approximately 29 minutes in
RF training, 48 minutes in raw-CNN training, and 15 minutes in sorting. The
slowest CNN folds were dominated by repeated `$NOVA_DATA` file loading for
`--cache_data`, especially before the filesystem cache warmed up; several
later folds loaded the same scale of data much faster.

Filtered aggregate checks from the same M54 LOSO run were written to:

- `outputs/loso_13_M54/loso_model_evaluation_totals_nonG_7shots.csv`
- `outputs/loso_13_M54/loso_model_evaluation_totals_nstxuG_6shots.csv`

For the non-G subset (the four older shots plus the three `nstxuE*` shots),
the held-out set has `1638` modes: `546` good and `1092` bad. Metrics:

- CNN: CM `[[1031, 61], [82, 464]]`, accuracy `0.913`, GOOD
  precision/recall/F1 `0.884 / 0.850 / 0.866`
- combined policy: CM `[[1057, 35], [82, 464]]`, accuracy `0.929`, GOOD
  precision/recall/F1 `0.930 / 0.850 / 0.888`
- RF: CM `[[1064, 28], [98, 448]]`, accuracy `0.923`, GOOD
  precision/recall/F1 `0.941 / 0.821 / 0.877`

This confirms that the `nstxuG*` folds are a major source of the poorer
aggregate GOOD precision/recall. On the non-G subset, combined policy keeps
the CNN GOOD recall while cutting false positives nearly in half.

The 13-shot `M_target=100` LOSO run completed under `outputs/loso_13_M100`.
Aggregate metrics:

- CNN: CM `[[1908, 96], [90, 516]]`, accuracy `0.929`, GOOD
  precision/recall/F1 `0.843 / 0.851 / 0.847`
- combined policy: CM `[[1948, 56], [112, 494]]`, accuracy `0.936`, GOOD
  precision/recall/F1 `0.898 / 0.815 / 0.855`
- RF: unchanged from M54, CM `[[1957, 47], [134, 472]]`, accuracy `0.931`,
  GOOD precision/recall/F1 `0.909 / 0.779 / 0.839`

Relative to M54, M100 improves raw CNN globally: 9 fewer false positives, 18
fewer false negatives, +0.010 accuracy, and +0.024 GOOD F1. The combined
policy improves only slightly: same false positives, 4 fewer false negatives,
and +0.004 GOOD F1.

Filtered M100 aggregate checks were written to:

- `outputs/loso_13_M100/loso_model_evaluation_totals_nonG_7shots.csv`
- `outputs/loso_13_M100/loso_model_evaluation_totals_nstxuG_6shots.csv`

For the non-G subset, M100 improves both CNN and combined policy:

- CNN: CM `[[1042, 50], [65, 481]]`, accuracy `0.930`, GOOD
  precision/recall/F1 `0.906 / 0.881 / 0.893`
- combined policy: CM `[[1059, 33], [78, 468]]`, accuracy `0.932`, GOOD
  precision/recall/F1 `0.934 / 0.857 / 0.894`
- RF: CM `[[1064, 28], [98, 448]]`, accuracy `0.923`, GOOD
  precision/recall/F1 `0.941 / 0.821 / 0.877`

For the `nstxuG*` subset, M100 does not materially solve GOOD detection:

- CNN: CM `[[866, 46], [25, 35]]`, accuracy `0.927`, GOOD
  precision/recall/F1 `0.432 / 0.583 / 0.496`
- combined policy: CM `[[889, 23], [34, 26]]`, accuracy `0.941`, GOOD
  precision/recall/F1 `0.531 / 0.433 / 0.477`
- RF: CM `[[893, 19], [36, 24]]`, accuracy `0.943`, GOOD
  precision/recall/F1 `0.558 / 0.400 / 0.466`

Interpretation: M100 is a better raw-CNN harmonic window for the non-G regime
and improves the all-shot M54 result, but the G-shot regime likely needs
separate calibration/policy or additional physics-aware features rather than
only a larger `M_target`.

### 2026-07-08

Changed the raw-CNN default harmonic window from `M_target=54` to
`M_target=100`. The LOSO driver now also defaults to `--cnn_m_target 100`,
and its default CNN batch size is `--cnn_batch_size 32`; the Slurm wrapper
uses the same current defaults. Older M54 and batch-8 checks can still be
reproduced by passing those options explicitly.

The `outputs/loso_13_M100_bs32` LOSO run repeated the M100 check with
`--cnn_batch_size 32`. Compared with the previous M100 batch-8 LOSO run:

- CNN batch 8: CM `[[1908, 96], [90, 516]]`, accuracy `0.929`, GOOD
  precision/recall/F1 `0.843 / 0.851 / 0.847`
- CNN batch 32: CM `[[1900, 104], [97, 509]]`, accuracy `0.923`, GOOD
  precision/recall/F1 `0.830 / 0.840 / 0.835`
- Combined policy batch 8: CM `[[1948, 56], [112, 494]]`, accuracy `0.936`,
  GOOD precision/recall/F1 `0.898 / 0.815 / 0.855`
- Combined policy batch 32: CM `[[1951, 53], [115, 491]]`, accuracy `0.936`,
  GOOD precision/recall/F1 `0.903 / 0.810 / 0.854`

Interpretation: the single 13-shot held-out split favored M100 batch 32, but
LOSO did not. In LOSO, batch 32 added 8 CNN false positives and 7 false
negatives relative to batch 8. The combined policy was nearly unchanged:
batch 32 removed 3 false positives but added 3 false negatives. Keep M100 as
the default input window; treat batch size as an optimization/calibration
setting rather than a settled science result.

Subset breakdown for CNN-only M100 LOSO:

- non-G seven-shot subset: batch 8 CM `[[1042, 50], [65, 481]]`, GOOD F1
  `0.893`; batch 32 CM `[[1038, 54], [69, 477]]`, GOOD F1 `0.886`
- `nstxuG*` six-shot subset: batch 8 CM `[[866, 46], [25, 35]]`, GOOD F1
  `0.496`; batch 32 CM `[[862, 50], [28, 32]]`, GOOD F1 `0.451`

The active `models/nova_cnn_raw.pt` checkpoint was retrained as a full-list
production refit with `M_target=100`, batch size 32, robust normalization, and
the current 2610-row training list. Checkpoint metadata reports
`saved_training_scope=full_csv_refit`, final prediction health of 606 predicted
GOOD out of 2610 modes, matching the 606 true GOOD labels, and
`collapse_detected=False`.

Production-use decision: use the active RF plus raw-CNN models through
`scripts/sort_shot_mixed.py` for NSTX-U E-like shots to select modes for
NOVA-C growth-rate calculations. Leave NSTX-U G-case shots out of this
production path for now because the TAE-gap geometry is physically different
and the LOSO checks show much weaker GOOD-mode detection. Next G-shot work is
to add more same-regime labeled examples and/or add explicit gap-width /
gap-geometry features before revisiting the fusion policy.

`scripts/sort_shot_mixed.py` now writes `rad_loc` and `rad_width` to its
scored-mode CSV outputs, including `good_tae_final.csv`. These are the same
normalized radial centroid and RMS radial width used in the RF feature schema
and are intended for comparing candidate mode locations with beam-ion density
profiles before deciding whether a NOVA-C growth-rate calculation is needed.

### 2026-07-09

The non-G / E-production 7-shot list `training_labels/tae_like_train_7.csv`
was used for a dedicated M100 batch-32 LOSO run under
`outputs/loso_7_M100_bs32`. The run used 7 folds, `M_target=100`,
`R_target=201`, batch size 32, robust normalization, and full-CNN refits.
Aggregate metrics:

- CNN: CM `[[1025, 67], [74, 472]]`, accuracy `0.914`, GOOD
  precision/recall/F1 `0.876 / 0.864 / 0.870`
- combined policy: CM `[[1053, 39], [65, 481]]`, accuracy `0.937`, GOOD
  precision/recall/F1 `0.925 / 0.881 / 0.902`
- RF: CM `[[1059, 33], [75, 471]]`, accuracy `0.934`, GOOD
  precision/recall/F1 `0.935 / 0.863 / 0.897`

Compared with evaluating the 13-shot M100 batch-32 LOSO only on the same
seven non-G held-out shots, removing G shots from training had mixed effects:

- CNN got worse: +13 false positives, +5 false negatives, GOOD F1
  `0.886 -> 0.870`
- combined policy improved slightly: +7 false positives but -15 false
  negatives, GOOD F1 `0.893 -> 0.902`
- RF improved: +5 false positives but -23 false negatives, GOOD F1
  `0.877 -> 0.897`

Interpretation: excluding G shots does not help CNN-only in this M100 batch-32
LOSO check, but it does make the RF and combined policy less conservative on
non-G / E-like cases and improves GOOD recall. The dedicated 7-shot combined
policy is currently the best matched check for E-like production sorting, but
the gap relative to the 13-shot combined policy is modest.
