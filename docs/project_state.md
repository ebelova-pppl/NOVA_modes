# Project: AI NOVA mode classifier
### Project state (2026-03-29)
## Goal
Train ML classifiers to identify physically meaningful NOVA eigenmodes (“good”) vs unphysical/numerical modes (“bad”), and provide a clean, deduplicated mode set for downstream analysis (e.g., NOVA-C, surrogate modeling, digital twin workflows).
 
## Data
-	~1272 labeled modes
    -	NSTX-U: 1 shot
    -	NSTX: 3 shots
    -	DIII-D: optional (future)
-	Main version-controlled training list:
    -	`training_labels/train_master.csv`
    -	mode paths stored relative to `$NOVA_DATA` when possible
    -	example entry: `nstx_120113/N5/egn05w.1234E+02,good`

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
    -	Scalar + structure-derived + continuum features (~20)
    -	Accuracy: ~92–95%
    -	OOF ≈ 93%
    -	Most robust / interpretable baseline
2.	CNN (raw)
    -	Padded/truncated (m,r)
    -	~90%, it is actually better now ~93% with cleaned training dataset
3.	CNN (straightened)
    -	Ridge-aligned representation (2M+1, r)
    -	Best CNN: ~94–96%
    -	Less sensitive to nhar variation, it was sensitive to LR => added scheduler
4.	HybridCNN (image + scalars)
    -	Includes continuum scalars
    -	Current: ~94%
    -	Comparable to RF, out of box – not optimized yet

## Continuum-derived scalars
From cont_features.py:
-	r_star — closest approach to continuum
-	delta2_eff — mode-weighted distance to continuum
-	S — normalized separation between r0 and r_star
-	W_star — mode energy near resonance

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
-	Standard split accuracy ~95%
-	OOF check:
- CM ≈ [[800, 41],
      [ 53,378]]  → ~93%
-	Used for label validation (flag p < 0.2 or p > 0.8)
### CNN
-	Performance sensitive to seed + learning rate
-	Latest TAE-like retraining uses threshold 0.5 for all CNN confusion matrices
-	All three CNNs are comparable on `training_labels/tae_like.csv`, with best accuracy ~0.95-0.96

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
•	Continue label cleaning (OOF + CNN comparison)
•	Compare misclassified modes (RF vs CNN vs HybridCNN)
•	Validate the shared `cnn_classify.py` inference path on more checkpoints / shots
 
## Next tasks
•	Add EAEs (second gap) more deeply into training / continuum features
•	Extend training to broader frequency range
•	Investigate surrogate / autoencoder for mode structure

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
- Retrained / rechecked the good-bad classifiers on `training_labels/tae_like.csv` using threshold 0.5 for CNN evaluation.
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
`sort_shot_mixed.py`. These control the weighted `p_avg` used in fallback
fusion decisions and duplicate-clustering scores; defaults are equal weights
for backward-compatible behavior.

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
