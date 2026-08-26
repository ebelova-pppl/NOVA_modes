# Project: AI NOVA mode classifier
### Project state (current snapshot, updated 2026-08-26)
## Goal
Train ML classifiers to identify physically meaningful NOVA eigenmodes (“good”) vs unphysical/numerical modes (“bad”), and provide a clean, deduplicated mode set for downstream analysis (e.g., NOVA-C, surrogate modeling, digital twin workflows).

## 2026-08-26 Q62 v3 deterministic review lists

- Added `tests/labels_audit/continuum_refresh_2026_08_23/q62_v3_good_modes.csv`
  with all 16 Q62 GOOD modes in `training_labels/tae_like_v3.csv`.
- Added
  `q62_v3_bad_first_two_gate_survivors_crossing_amplitude_gt0p3.csv` in the
  same audit directory. It contains 56 v3-labeled BAD modes that survive the
  calibrated axis and grid-scale-spike gates and have maximum absolute signed
  harmonic amplitude strictly greater than 0.3 after interpolation to at
  least one true continuum crossing.
- The Q62 v3 population has 233 BAD modes: 69 fire the first gate, 68 more fire
  the second, and 96 survive both. Of those survivors, 56 meet the requested
  exact-crossing amplitude condition and 40 have crossings but remain at or
  below 0.3. All 249 Q62 paths were available locally; the detailed list keeps
  mode-plus-continuum fingerprints and crossing audit evidence.
- This review measurement is distinct from v7 gate 3, which thresholds
  peak-normalized total radial energy at the crossing, and gate 4, which uses
  amplitudes and energy at radial samples within two grid intervals.
 
## Data
- Active version-controlled training list:
    - canonical active copy: `training_labels/tae_like_train.csv`
    - versioned source: `training_labels/tae_like_v3.csv`
    - the two files are byte-identical
    - 2639 labeled TAE-like modes: 595 `good`, 2044 `bad`
    - complete rebuilt-database audit covering all 15 training shots
    - `configs/paths/nova_paths.nersc.sh`,
      `configs/paths/nova_paths.flux.sh`, and
      `configs/paths/nova_paths.flux.csh` point `NOVA_TRAIN_CSV` /
      `NOVA_TRAIN_CSV_TAE` at this list
- Preserved pre-v3 list:
    - `training_labels/tae_like_v2_nonG.csv`
    - 2900 labeled TAE-like modes: 593 `good`, 2307 `bad`
    - preserved unchanged for reproducibility
- Pre-promotion canonical 15-shot training list (Git history):
    - 2903 labeled TAE-like modes: 629 `good`, 2274 `bad`
    - shots: `nstx_120113`, `nstx_135388`, `nstx_141711`, `nstxu_204202`,
      `nstxuE202855A01t020`, `nstxuE204669M03t025`,
      `nstxuE205052A01t022`, `nstxuG121123K51`, `nstxuG133964S31`,
      `nstxuG142301H47`, `nstxuG121123J38`, `nstxuG121123Q62`,
      `nstxuG142301Y93`, `nstxuG121123B12`, `nstxuG142301W29`
    - mode paths stored relative to `$NOVA_DATA` when possible
    - example entry: `nstx_120113/N5/egn05w.1234E+02,good`
    - replaced intentionally at the `tae_like_train.csv` path during v3
      promotion; exact prior contents remain in Git history
- Derived non-G / E-production comparison list:
    - `training_labels/tae_like_train_7.csv`
    - 1638 labeled modes: 546 `good`, 1092 `bad`
    - shots: `nstx_120113`, `nstx_135388`, `nstx_141711`, `nstxu_204202`,
      `nstxuE202855A01t020`, `nstxuE204669M03t025`,
      `nstxuE205052A01t022`
    - created from the pre-promotion `tae_like_train.csv` by excluding all
      `nstxuG*` shots for 7-shot LOSO checks of the non-G / E-like production
      regime
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
    - reviewed `nstxuG121123B12` copy:
      `training_labels/additions/tae_like_nstxuG121123B12.csv`
    - reviewed `nstxuG142301W29` copy:
      `training_labels/additions/tae_like_nstxuG142301W29.csv`
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
    -	Checkpoint status: retrained on the preceding 2900-row / 15-shot
      `training_labels/tae_like_v2_nonG.csv` snapshot; the later N9/3222 label
      correction is not yet reflected in the checkpoint
    -	Current schema: previous RF features minus `omega`, plus `W_star_max`
    -	Checkpoint training counts: 594 GOOD, 2306 BAD
    -	Latest pre-B12 13-shot OOF accuracy: 0.951
    -	Latest pre-B12 13-shot OOF CM: `[[1967, 37], [91, 515]]`
    -	Latest pre-B12 GOOD precision/recall/F1: 0.933 / 0.850 / 0.889
    -	Most interpretable baseline
2.	CNN (raw)
    -	Padded/truncated (m,r)
    -	Active checkpoint: `models/nova_cnn_raw.pt`
    -	Checkpoint status: full-CSV refit on the preceding 2900-row / 15-shot
      `training_labels/tae_like_v2_nonG.csv` snapshot; the later N9/3222 label
      correction is not yet reflected in the checkpoint
    -	Current default raw preprocessing: `M_target=100`, `R_target=201`
    -	Latest v2 split check before full refit: best accuracy 0.9534 at
      epoch 13; final 80-epoch full refit reports no prediction collapse
    -	Latest pre-B12 13-shot M100 held-out split check: CM `[[394, 6], [9, 112]]`,
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
- Experimental inner-extremum extension: `ext_dr`, `ext_df_gap`, and
  `ext_energy_frac`; these are opt-in and are not part of the active RF
  checkpoint

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
  whether to retune the G-shot policy after dedicated gap-topology /
  mode-extremum feature ablations.
- Treat the current J38 audit as complete: retain only coherent extremum modes
  without a material secondary continuum crossing, and revisit the remaining
  junk-like extrema candidates only if the labeling policy changes.
- Treat the Codex blind audits of training shots `nstxu_204202`,
  `nstx_120113`, `nstx_141711`, `nstx_135388`, and
  `nstxuE202855A01t020`, `nstxuE204669M03t025`, and
  `nstxuE205052A01t022` as complete. Retained artifacts live under
  `tests/labels_audit/<shot>/` and are limited to the sealed Codex list plus
  SHA sidecar, Codex/human/training union disagreements, and
  human-vs-training label changes. `nstx_141711` also keeps its union
  adjudication table; `nstx_135388` also keeps the post-adjudication
  policy-v2 labels, change list, and training-comparison tables.
  `nstxuE205052A01t022` also keeps non-sealed Codex post-review labels and the
  B139/B143 correction table. The sealed Codex lists are
  provenance-preserving and should not be edited in place.
- `nstx_120113` audit status: the clean target manifest has 174 TAE-like modes.
  Pre-adjudication Codex-vs-human agreement was 169/174 = 97.13% with Cohen
  kappa 0.9270. Review discussion resolved the remaining cases as
  `B045=good`, `B087=good`, `B105=skip` for adjudication/training exclusion,
  `B106=good`, and `B131/B149=bad` with low confidence because their low-r
  tails are distorted by continuum crossing despite small integrated crossing
  energy. The retained human-vs-training delta table has two rows after the
  human corrections. These corrections are included in the current
  `training_labels/tae_like_v2_nonG.csv` default list; the previous
  `training_labels/tae_like_train.csv` list is preserved unchanged for
  comparison.
- `nstxuE205052A01t022` audit status: the target manifest has 293 TAE-like
  modes. The sealed Codex review has 24 `good`, 269 `bad`, and 0 `skip`; all
  rows are marked `prior_seen=true` because aggregate target-shot context had
  already been exposed, so use this as an independent mode-level review but
  not as clean-blind agreement statistics. Post-review Codex labels change
  only `B139` and `B143` from `good` to `bad` in the non-sealed v2 file,
  yielding 22 `good` and 271 `bad`. After human-side corrections
  (`B042=skip`, `B156=good`), post-review Codex-vs-human agreement is
  250/293 = 85.32%, and the retained human-vs-training delta table has 19
  rows. These corrections are included in the current
  `training_labels/tae_like_v2_nonG.csv` default list; the previous
  `training_labels/tae_like_train.csv` list is preserved unchanged for
  comparison.
- For smooth-looking modes whose continuum diagnostic still crosses the mode
  body at large amplitude with no obvious resonant structure, treat the case
  as a continuum/equilibrium consistency question. Do not promote these to a
  confident `good` label from appearance alone; hold them out as `skip` or
  require follow-up continuum verification before merging.
- Keep the three-feature inner-extremum RF schema experimental. Replacing
  continuum prominence with local mode-energy fraction improved shuffled folds
  and removed the overall LOSO regression, but did not reduce G-shot LOSO FN.
  Do not replace the production 22-feature checkpoint without a stronger
  cross-shot result.
- If a later RF refinement establishes transferable continuum geometry, test a
  small numerical 1D-continuum branch fused with the raw-mode CNN; do not use
  rendered plot images as network inputs.
- Recheck `nstxuG121123N75` after recalculation with the corrected q profile.
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
- Continuum crossings in low-energy tails can be accepted only when the tail is
  both small in integrated/pointwise amplitude and smooth/detached from the
  main envelope. A visibly distorted connected tail near the crossing is BAD or
  low-confidence BAD even when the integrated crossing energy is small.
 
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
The `rf_vs_cnn_pgood.png` plot was also changed from a single scatter plot to
a two-panel count-density plus jittered tier-scatter diagnostic so saturated
probabilities near 0 and 1 are easier to count and the legend no longer covers
the main data region.

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


### 2026-08-02

During review of shared RF/CNN false negatives in the NSTX-U G-shot LOSO
results, five modes were manually reclassified from `good,tae` to `bad,none`:

- `nstxuG142301H47/N10/egn10w.1503E+02`
- `nstxuG121123K51/N8/egn08w.2347E+02`
- `nstxuG121123Q62/N3/egn03w.1586E+02`
- `nstxuG142301H47/N9/egn09w.1283E+02`
- `nstxuG121123K51/N9/egn09w.2153E+02`

These corrections were applied to the active
`training_labels/tae_like_train.csv` list and the corresponding source
component lists under `training_labels/additions/`. The active list remains
2610 rows and now contains 601 `good` and 2009 `bad` labels. Existing RF,
raw-CNN, and LOSO metrics predate these corrections; retraining and
revalidation are pending. `nstxuG121123K51/N9/egn09w.2937E+02` remains
`good,tae` pending further review of its spiky structure and small amplitude
at the continuum crossing.


### 2026-08-03

Compared the small-radius `N=8` Alfvén-continuum gap topology for all seven
non-G and six G shots in the active 13-shot set. Generated normalized and
absolute-frequency comparison figures:

- `outputs/gshot_error_diagnostics/n8_continuum_gap_G_vs_nonG.png`
- `outputs/gshot_error_diagnostics/n8_continuum_gap_G_vs_nonG_absolute.png`
- `outputs/gshot_error_diagnostics/n10_continuum_gap_G_vs_nonG_absolute_common_range.png`

The normalized figure divides each shot by its median gap-center frequency
over `0.05 <= r <= 0.45` and reports the median relative gap width over
`0.05 <= r <= 0.25`. Each panel also shows the current number of GOOD-labeled
`N=8` modes in large type for presentation use. The comparison supports continuum topology/location as an important G-shot
failure variable: K51 and H47 contain nine of the ten strict
continuum-extremum-localized shared RF/CNN false negatives and show
pronounced repeated inner-radius extrema. Gap width alone does not separate
the regimes; the G-shot inner-width range overlaps the non-G range. The
comparison is not a universal G/non-G separator, however: NSTX 135388 and
E202855 also have clear small-radius continuum extrema, while some G shots
are comparatively monotonic.

The `N=10` companion uses physical (non-normalized) frequencies, the full
`0 <= r <= 1` axis range, and one common frequency axis (`0.0723` to `19.998`)
determined from the minimum and maximum retained continuum values across all
13 shots. Each panel is annotated with its current number of GOOD-labeled
`N=10` TAEs using presentation-size text. The unphysical E205052 endpoint at
`r=0.99` is omitted; the shared maximum is then set by the NSTX 141711 upper
boundary at `r=0.71`.

The J38 audit reclassified
`nstxuG121123J38/N10/egn10w.3596E+02` from `bad,none` to `good,tae`. It is
a coherent companion to the already-GOOD `egn10w.3559E+02`: both localize at
the upper-continuum minimum near `r=0.115`, their frequencies differ by only
0.52%, and their only outer crossings occur near `r=0.85`, where
peak-normalized mode energy is negligible (`0` and `1.9e-4`, respectively).

The K51 follow-up likewise reclassified
`nstxuG121123K51/N10/egn10w.4656E+02` from `bad,none` to `good,tae`.
Its radial-energy maximum coincides with the upper-continuum minimum at
`r=0.240`; the neighboring already-GOOD `egn10w.4703E+02` peaks at the
preceding minimum at `r=0.135`, and their frequencies differ by only 0.50%.
The only continuum crossing for `4656` is at `r=0.825`, where
peak-normalized mode energy is `1.68e-4`. Its mode/extremum frequency mismatch
is 3.21%, just outside the provisional 3% cutoff used for the strict
diagnostic count; the label decision follows the physical review rather than
changing that audit threshold.

This supports a refined validity rule: a coherent extremum-localized mode may
be GOOD provided it has no material secondary continuum crossing; a purely
mathematical crossing in a region of negligible mode amplitude is acceptable.
The active list now has 603 `good` and 2007 `bad` labels; the G subset has 57
`good` and 915 `bad`.


#### G-shot working conclusions after manual FN review

The active six-shot G subset contains 972 modes: 57 `good` and 915 `bad`.
Five original GOOD labels were corrected to BAD during the false-negative
review, and later J38 and K51 audits corrected two BAD labels to GOOD.
Existing M100 LOSO predictions have not been retrained, but applying
the corrected labels to those saved predictions gives:

- raw CNN: 22 G-shot false negatives, 15 (68%) strictly
  continuum-extremum-localized;
- RF: 32 false negatives, 12 (38%) extremum-localized;
- combined policy: 30 false negatives, 11 (37%) extremum-localized;
- RF/CNN shared rejects: 12 false negatives, 10 (83%)
  extremum-localized;
- distinct GOOD modes missed by either model: 42, of which 17 (40%) are
  extremum-localized.

The strict diagnostic used a mode radial-amplitude maximum within `dr <= 0.02`
of a local lower-boundary maximum or upper-boundary minimum and a relative
mode/extremum frequency mismatch no larger than 3%. For the ten shared RF/CNN
cases, the radial agreement is stronger (`dr <= 0.005`). The two shared-reject
exceptions are `nstxuG121123K51/N9/egn09w.2937E+02`, whose main amplitude is
far from the frequency-matching extremum, and
`nstxuG142301Y93/N9/egn09w.1539E+02`, which has only a partial/broader radial
association.

The current physical working picture has two continuum regimes:

1. A narrow, mostly monotonic/sloping inner gap supports few or no TAEs. At
   `N=8`, S31 has median relative gap width 9.5%, normalized center-frequency
   change +18.6%, edge monotonicity 0.82, and zero GOOD modes; Q62 has width
   12.9%, center change +27.4%, monotonicity 0.79, and zero GOOD modes.
2. A wider, wavy gap can create radial wells at local continuum extrema and
   support a distinct class of narrow modes localized at those extrema. At
   `N=8`, K51 has width 38.6%, monotonicity 0.22, and 6 GOOD modes; H47 has
   width 37.2%, monotonicity 0.31, and 2 GOOD modes. These two shots contain
   nine of the ten strict extremum-localized shared RF/CNN false negatives;
   the corrected J38 N10/3596 mode is the tenth.

These are tendencies, not sufficient rules. Non-G NSTX 135388 and E202855
also have wavy inner continua and many GOOD modes. J38 has K51-like topology
at `N=8` (31.5% width and 0.39 monotonicity) but zero `N=8` GOOD labels and
only 7 GOOD among 174 TAE-like candidates overall, compared with 28 among 208
for K51. Over `N=7-10`, the candidate counts are nearly equal (123 J38 versus
128 K51), but J38 has fewer prominent inner wells (12 versus 19) and two
GOOD modes satisfying the coarse extremum-alignment test, versus 13 for K51.
The completed audit leaves 18 BAD-labeled high-`n` J38 modes meeting the coarse
geometry criterion; these were retained as BAD because they cross the
continuum materially elsewhere or have clearly junk-like structure. K51
belongs to the earlier six-shot review, whereas J38 was labeled in a separate
later pass.

Model interpretation:

- The raw CNN does not receive continuum data. It can only learn that the
  narrow, localized, often sign-changing morphology resembles numerical junk.
- The RF receives closest-approach, distance, and crossing-amplitude features,
  but no explicit gap width, slope/monotonicity, local-extremum prominence, or
  mode-to-extremum match. A tangency at a continuum minimum/maximum may also
  produce no ordinary crossing and may have small amplitude exactly at the
  contact.
- Continuum topology alone must not label a mode: the important conditions are
  joint radial and frequency alignment between a coherent mode and a
  sufficiently prominent inner extremum, plus no secondary continuum crossing
  where the mode has material amplitude.

Proposed first RF experiment: compute the physical-frequency boundaries
`sqrt(low2)` and `sqrt(high2)` over `0.05 <= r <= 0.40` and add robust relative
gap width, binned center-frequency change, lower/upper edge monotonicity,
prominent-well count/depth/radius, mode-to-extremum radial and frequency
distance, mode energy near the extremum, and maximum mode energy at secondary
continuum crossings. Thresholds such as 15% width or
0.75 monotonicity are provisional diagnostics from the `N=8` comparison, not
production cuts. Evaluate baseline, topology-only, and topology-plus-alignment
schemas with shot-wise LOSO and report G and non-G subsets separately.

For a CNN follow-up, feed numerical continuum arrays rather than PNG plots.
A small 1D branch can ingest normalized lower/upper boundaries, the normalized
mode-frequency line, and a validity mask on the same radial grid; concatenate
its embedding with the raw-mode CNN embedding. This is different from the
existing hybrid CNN, which receives continuum summary scalars but not the full
gap topology. Given only 57 G-shot GOOD examples, perform the RF ablation before expanding the CNN.


### 2026-08-04: minimal inner-extremum RF ablation

Implemented a three-feature opt-in RF extension targeted at the manually
validated small-radius continuum-extremum TAEs. The implementation keeps the
existing repository notation `W(r) = sum_m |xi_m(r)|^2` and defines
`r_peak = argmax W(r)`. It lightly smooths the physical-frequency boundaries
`u(r) = sqrt(high2)` and `l(r) = sqrt(low2)`, finds upper minima and lower
maxima over `0.03 <= r <= 0.40`, and jointly matches them to the mode using
the audited `dr=0.02` and relative-frequency `df=0.03` scales. The added
features are:

- `ext_dr`: absolute radial separation between `r_peak` and the matched
  extremum;
- `ext_df_gap`: signed relative frequency clearance, defined so positive is
  on the local gap side for either an upper minimum or lower maximum;
- `ext_prom_rel`: matched-extremum prominence divided by `abs(omega)`.

`src/cont_features.py` owns the shared calculation, `src/mode_features.py`
adds the opt-in `rf_extremum_25_v1` schema, and the RF trainer and OOF checker
accept `--extremum-features`. The production `rf_w_star_max_22_v2` schema and
active checkpoint are unchanged. The three new positive-match features reuse
the existing production `W_star_max` as the material-crossing evidence rather
than adding another correlated crossing scalar. Synthetic upper-minimum,
lower-maximum, no-extremum, fallback-order, and 22/25/28/31 schema tests pass.

Real-mode checks recover the intended geometry. Examples `(ext_dr,
ext_df_gap, ext_prom_rel)` are J38 N10/3559 `(0, 0.0063, 0.1118)`, J38
N10/3596 `(0.005, 0.0011, 0.1113)`, K51 N10/4703
`(0.005, 0.0041, 0.1239)`, and H47 N9/1283
`(0.005, 0.0247, 0.0273)`. The K51 N9/2937 non-matching exception has
`ext_dr=0.485` and `ext_df_gap=0.173`.

On the corrected 2610-row active list, identical seeded shuffled five-fold
checks gave:

- production 22 features: CM `[[1971, 36], [92, 511]]`, accuracy `0.9510`,
  GOOD precision/recall/F1 `0.9342 / 0.8474 / 0.8887`;
- production plus three extremum features: CM `[[1970, 37], [90, 513]]`,
  accuracy `0.9513`, GOOD precision/recall/F1
  `0.9327 / 0.8507 / 0.8899`.

Within the six G shots, shuffled-fold FN improved `32 -> 30`, at the cost of
FP increasing `7 -> 9`. True 13-fold leave-one-shot-out did not confirm the
improvement:

- production 22 features: CM `[[1965, 42], [130, 473]]`, accuracy `0.9341`,
  GOOD precision/recall/F1 `0.9184 / 0.7844 / 0.8462`;
- production plus extremum features: CM `[[1966, 41], [133, 470]]`, accuracy
  `0.9333`, GOOD precision/recall/F1 `0.9198 / 0.7794 / 0.8438`.

For G shots alone, LOSO CM changed from `[[900, 15], [31, 26]]` to
`[[900, 15], [32, 25]]`; among the 20 G-shot GOOD modes satisfying the strict
`ext_dr <= 0.02` and `abs(ext_df_gap) <= 0.03` geometry, FN changed
`11 -> 12`. The features raised the held-out probabilities of several target
modes without crossing the 0.5 decision threshold: J38 N10/3559
`0.423 -> 0.443`, J38 N10/3596 `0.210 -> 0.260`, and K51 N10/4703
`0.073 -> 0.110`. The only G-shot GOOD classification changed at threshold
was H47 N9/1817, a regression from `0.510` to `0.497`. In a full fit, feature
importance ranked `ext_df_gap` 11th, `ext_dr` 22nd, and `ext_prom_rel` 25th.

Conclusion: the three measurements correctly encode the manual physical rule
and provide a weak probability shift in the desired direction for some modes,
but they are not sufficient for cross-shot rescue. Keep the implementation for
controlled follow-up experiments, but do not promote the 25-feature schema or
replace the active RF checkpoint.


### 2026-08-04: replace extremum prominence with local mode energy

Replaced the learned `ext_prom_rel` scalar in the opt-in extremum schema with

`ext_energy_frac = integral_[|r-r_e| <= 0.03] W(r) dr / integral W(r) dr`,

where `W(r) = sum_m |xi_m(r)|^2` and `r_e` is the same jointly matched upper
minimum or lower maximum used by `ext_dr` and `ext_df_gap`. The integral
interpolates the two window boundaries on the radial grid. The no-extremum
fallback remains `(1, 1, 0)`. Continuum prominence is retained only as an
internal deterministic tie-breaker when matching otherwise equivalent extrema;
it is no longer supplied to the RF.

Because the third feature changed meaning, the experimental schemas are now
`rf_extremum_energy_25_v2` and
`rf_all_crossings_extremum_energy_31_v2`. Focused tests cover a centered narrow
mode, a displaced mode, a two-lobe mode with half its energy near the
extremum, both continuum-boundary types, safe fallbacks, and feature ordering.
The active 22-feature schema and checkpoint remain unchanged.

On the corrected 2610-row list, identical seeded shuffled five-fold checks
gave:

- production 22 features: CM `[[1971, 36], [92, 511]]`;
- production plus the energy-fraction extrema features: CM
  `[[1972, 35], [89, 514]]`, accuracy `0.9525`, GOOD precision/recall/F1
  `0.9362 / 0.8524 / 0.8924`.

For the six G shots in shuffled folds, the CM changed from
`[[908, 7], [32, 25]]` to `[[908, 7], [30, 27]]`. Thus the local-energy
version preserved the two-FN G improvement without the two extra FP produced
by the original prominence version.

True 13-fold leave-one-shot-out again did not show a G-shot recall gain:

- production 22 features: CM `[[1965, 42], [130, 473]]`;
- energy-fraction extrema features: CM `[[1967, 40], [130, 473]]`, accuracy
  `0.9349`, GOOD precision/recall/F1 `0.9220 / 0.7844 / 0.8477`;
- G shots: baseline `[[900, 15], [31, 26]]`, energy version
  `[[901, 14], [32, 25]]`;
- non-G shots: baseline `[[1065, 27], [99, 447]]`, energy version
  `[[1066, 26], [98, 448]]`;
- the 20 G-shot GOOD modes satisfying `ext_dr <= 0.02` and
  `abs(ext_df_gap) <= 0.03` remained at 11 FN and 9 TP.

The replacement is nevertheless better behaved than the prominence version,
which had worsened overall LOSO FN from 130 to 133. In a full-data fit,
`ext_df_gap` ranked 11th at `2.318%`, `ext_dr` ranked 23rd at `1.133%`, and
`ext_energy_frac` ranked 24th at `1.058%`; together the three contributed
`4.510%`. The energy fraction is slightly more informative and more directly
mode-specific than prominence, but its weak importance suggests substantial
redundancy with `ext_dr`, `rad_width`, and other localization features. Keep
the v2 implementation for controlled experiments, but do not promote it to
the production RF based on this result.


### 2026-08-04: `W_star` / `W_star_max` semantic audit

Reviewed the production continuum features after the inner-extremum work. The
legacy `r_star`, `S`, and `W_star` block is not a strict crossing calculation.
`band_distance` is zero wherever the mode frequency lies between the stored
lower and upper TAE-gap boundaries, and `r_star = nanargmin(dist2)` therefore
selects the first minimum-distance grid point. `W_star` is the fraction of
total radial mode energy within one mode width of that point, not the pointwise
amplitude stated in the older script documentation. These quantities are best
interpreted as gap-membership / gap-violation features, not crossing features.

`W_star_max` separately evaluates interpolated lower/upper boundary roots. It
accepts strict sign changes and exact grid equalities, so an exact tangency
would be counted as a crossing while a near-tangency without a sign change
would not. A full audit of all 2610 labeled modes found zero exact boundary
equalities. Consequently, changing only `W_star_max` to sign-change-only would
produce an identical feature matrix on the current dataset.

A read-only ablation tested a coherent strict-crossing reinterpretation of the
legacy block. For each mode, the boundary root with the largest peak-normalized
energy defined `r_star`; `S` and the local integrated `W_star` were recomputed
at that root, and modes without a root used a negative sentinel. The existing
`W_star_max` supplied the pointwise crossing energy. Of 2610 modes, 2160 had a
strict crossing and 450 did not. Results were:

- shuffled five-fold baseline: `[[1971, 36], [92, 511]]`;
- shuffled strict-crossing block: `[[1967, 40], [91, 512]]`;
- shuffled strict-crossing plus extrema: `[[1970, 37], [91, 512]]`;
- 13-shot LOSO baseline: `[[1965, 42], [130, 473]]`;
- LOSO strict-crossing block: `[[1957, 50], [139, 464]]`;
- LOSO strict-crossing plus extrema: `[[1962, 45], [141, 462]]`.

For G shots, shuffled FN changed from 32 to 30, or 29 with extrema features,
but LOSO FN worsened from 31 to 39, or 38 with extrema features. No repository
feature behavior was changed. The result supports keeping gap overlap and
strict crossing as separate concepts: retain the legacy numerical block for
now, interpret or rename it accurately, and use `W_star_max` for material
crossings plus `ext_df_gap` / `ext_energy_frac` for extremum contact. A later
ablation could improve the arbitrary first-minimum tie in `r_star` by choosing
the minimum-distance radius most aligned with the mode energy, without
misrepresenting it as a strict crossing.


### 2026-08-04: energy-aligned minimum-distance `r_star` ablation

Implemented an opt-in alternative to the legacy first-minimum `r_star` rule.
The calculation still requires the global minimum of `band_distance`; among
all radial grid points tied at that minimum, it selects the point with maximum
`W(r) = sum_m |xi_m(r)|^2`, with larger radius resolving equal-energy ties.
`S` and `W_star` are then evaluated using that selected radius. `delta2_eff` and
all crossing/extremum calculations are unchanged.

The option is exposed as `--r-star-energy-tie` in the RF trainer/classifier and
OOF checker. Because it changes feature values without changing feature names
or count, checkpoints record `nova_r_star_energy_tie_` and schema versions add
the suffix `_rstar_energy_tie_v1`; inference rejects a mismatched checkpoint.
Experimental default model filenames include `rstar_energy_tie`, and overwrite
protection prevents this option from replacing the active checkpoint. The
production default remains the legacy first-minimum rule.

Focused tests verify maximum-energy selection, larger-radius resolution of
equal-energy ties, end-to-end feature-builder propagation, distinct schema
versioning, and checkpoint metadata rejection on mismatch. Shuffled five-fold
results on the corrected 2610-row list were:

- production baseline: `[[1971, 36], [92, 511]]`;
- energy-aligned `r_star`: `[[1964, 43], [97, 506]]`;
- energy-aligned `r_star` plus extrema features:
  `[[1968, 39], [96, 507]]`.

For G shots in shuffled folds, baseline `[[908, 7], [32, 25]]` changed to
`[[903, 12], [32, 25]]`, or `[[905, 10], [32, 25]]` with extrema features.
There was no G recall gain. True 13-shot LOSO was substantially worse:

- production baseline: `[[1965, 42], [130, 473]]`;
- energy-aligned `r_star`: `[[1960, 47], [142, 461]]`;
- energy-aligned `r_star` plus extrema features:
  `[[1964, 43], [134, 469]]`.

G-shot LOSO changed from baseline `[[900, 15], [31, 26]]` to
`[[897, 18], [37, 20]]`, or `[[901, 14], [36, 21]]` with extrema features.
Among the 20 strict extremum-aligned G-shot GOOD modes, both new variants had
14 FN / 6 TP versus baseline 11 FN / 9 TP.

In a full fit of the 22-feature energy-aligned schema, importance shifted to
`W_star_max` at `17.99%`; `r_star` ranked ninth at `4.65%`, while `W_star` and
`S` fell to `1.39%` and `1.17%`. Selecting the mode-energy maximum within a
broad zero-distance interval makes `S` small and `W_star` large for many good
and bad modes alike, removing useful variation from the legacy first-entry
geometry. Keep the implementation opt-in for reproducibility, but do not
promote it or retrain the active RF with this rule based on current evidence.


### 2026-08-04: inner-boundary interpretation of legacy `r_star`

Checked whether the predictive legacy first-minimum rule is capturing numerical
or boundary-condition problems near the magnetic axis. All 2610 labeled modes
have their first valid `datcon` point at `r=0.01`; 1169 modes select exactly
that point as legacy `r_star`. Selecting the first valid radius alone is not a
BAD indicator: those 1169 modes are 74.0% BAD versus 76.9% BAD in the full
imbalanced list. The interaction with mode localization is highly diagnostic:

- `r_star <= 0.02` and `rad_loc <= 0.10`: 233 BAD / 4 GOOD, or 98.3% BAD;
- `r_star <= 0.02` and `W_star >= 0.30`: 335 BAD / 16 GOOD, or 95.4% BAD;
- within G shots, first-valid `r_star` and `W_star >= 0.30`: 122 BAD / 0 GOOD;
- within non-G shots, the same condition: 209 BAD / 16 GOOD, or 92.9% BAD.

GOOD modes overall have median `r_star=0.01` but median `rad_loc=0.686`,
`S=5.14`, and `W_star` effectively zero. BAD modes have median
`rad_loc=0.444`, `S=1.65`, and `W_star=0.249`. Thus the useful signal is not
simply that the frequency lies inside the gap at the inner boundary. It is the
combination of an inner gap-entry anchor with substantial mode energy near
that anchor. This is consistent with the RF learning core boundary-condition
or numerical-junk structure, although label statistics alone cannot prove the
causal mechanism. It also explains why moving `r_star` to maximum `W(r)` was
harmful: that change forced many otherwise displaced GOOD modes to acquire
small `S` and large `W_star`, erasing the useful interaction.


### 2026-08-04: multi-`n` G-shot training-candidate scan

Scanned all 41 usable `nstxuG*` shots in the DiTw archive to find additional
training shots with K51-like continuum topology. The initial `N=8` comparison
was used only as a probe; candidate selection was then repeated independently
for every `N=6-10`. Over `0.05 <= r <= 0.40`, an `N` is provisionally called
suitable when the median relative gap width is at least 25%, the mean
lower/upper edge monotonicity is at most 0.50, and the smoothed boundaries
contain at least one lower-maximum/upper-minimum pair. These are screening
thresholds, not physical label rules.

Sixteen shots satisfy the topology screen at all five `N` values, including
the already labeled K51 and H47 references, leaving 14 new consistently
wide/wavy candidates. Applying the existing upper-boundary TAE/EAE frequency
split gives the following largest new `N=6-10` TAE-region pools:

- M21: 243 of 442 modes;
- E55: 183 of 456;
- B12: 128 of 461;
- R42: 122 of 611;
- B37: 117 of 315;
- F62: 108 of 441;
- E34: 103 of 657;
- V21: 102 of 631;
- W29: 89 of 702.

These counts are only a frequency prefilter (`below_upper2` plus `mixed`), not
predicted or hand-labeled GOOD counts. In particular, the pool can still
contain continuum-crossing and numerical modes. The useful first labeling
tranche is M21, E55, B37, and F62 as wide/wavy cases with a strong outer
upper-boundary downturn, plus B12 as a weak-downturn control and W29 as the
closest K51-like normalized multi-`n` shape among the high-raw-count new
shots. B37 illustrates why `N=8` must not be the labeling target: it has no
`N=8` mode files but has 117 TAE-region candidates across the other values of
`N`. N75 remains excluded pending the previously identified recalculation and
review.

Generated diagnostics are under `outputs/gshot_candidate_selection/`:

- `n6_10_gshot_continuum_by_n.csv`: per-shot, per-`N` topology metrics;
- `n6_10_gshot_continuum_summary.csv`: topology consistency summary;
- `n6_10_gshot_tae_region_counts.csv`: complete counts for the 16 shots that
  pass at every `N=6-10`;
- `n6_10_gshot_recommended_shortlist.csv`: balanced six-shot first tranche;
- `n6_10_gshot_candidate_yield_vs_shape.png`: yield versus K51 shape distance,
  colored by outer upper-boundary downturn.

Next, build a per-mode review list for the shortlisted shots, prioritize modes
localized near inner extrema, reject modes with material secondary continuum
crossings, and add only hand-verified labels. Any claimed model improvement
must still be evaluated with shot-wise holdout so that adding many modes from
a few new shots does not inflate random-fold performance.


### 2026-08-04: extremum-localization and secondary-crossing screen

Refined the six-shot G training shortlist at the individual-mode level over
`N=6-10`. Modes were first restricted to the existing TAE frequency region
(`below_upper2` plus `mixed`). The established extremum geometry uses the
maximum of `W(r)`, `ext_dr <= 0.02`, and a matched upper minimum or lower
maximum over `0.03 <= r <= 0.40`. The focused review pool allows the known
K51 N10/4656 edge case by requiring `0 <= ext_df_gap <= 0.04`, and requires at
least 25% of integrated mode energy within `|r-r_ext| <= 0.03`.

A secondary crossing is any interpolated lower/upper continuum root more than
0.03 in radius from the matched extremum. Its amplitude is measured by
`W_peak`, radial mode energy normalized to the mode-energy maximum. The main
screen requires the maximum secondary `W_peak < 0.01`. This retains known
K51/H47 GOOD extrema modes with remote crossings at roughly 0.2-0.9% of peak
energy. Sensitivity counts were also made at 0.1% and 5%; the new-shot ranking
is stable over that range.

The screen substantially changes the preferred shots:

| shot | TAE region | strict 3% geometry | localized gap-side | clean at 1% | distinct extrema sites | 0.1% / 1% / 5% |
|---|---:|---:|---:|---:|---:|---:|
| B12 | 128 | 48 | 38 | 25 | 17 | 17 / 25 / 28 |
| W29 | 89 | 47 | 54 | 12 | 10 | 8 / 12 / 16 |
| E55 | 183 | 14 | 12 | 11 | 5 | 9 / 11 / 12 |
| F62 | 108 | 6 | 6 | 5 | 5 | 2 / 5 / 6 |
| B37 | 117 | 2 | 2 | 2 | 2 | 1 / 2 / 2 |
| M21 | 243 | 25 | 5 | 1 | 1 | 0 / 1 / 4 |

B12 is the strongest first shot: its 25 candidates occupy 17 sites and cover
both lower maxima (16 modes) and upper minima (9 modes), closely matching the
labeled K51 reference with 23 candidates at 15 sites. W29 is the next most
diverse, with 12 modes at 10 sites across `N=6-10`. E55 has 11 candidates but
less independent topology coverage: six occupy the same `N=8`, `r=0.125`
lower-maximum site. F62 is a useful smaller follow-up. B37 and especially M21
should be deprioritized despite their initially promising raw frequency pools.

The reference calibration also shows why this remains a review prioritizer,
not an automatic GOOD label. At the 1% screen, K51 has 9 GOOD, 12 BAD, and 2
currently unlabeled candidates; H47 has 5 GOOD, 8 BAD, and 2 unlabeled. The
remaining BAD cases can still have junk-like mode morphology or other defects
not captured by continuum geometry and crossing amplitude.

Generated artifacts under `outputs/gshot_candidate_selection/`:

- `n6_10_extremum_mode_audit.csv`: all TAE-region modes with matched-extremum
  and crossing diagnostics;
- `n6_10_extremum_review_candidates.csv`: 56 new-shot candidates passing the
  1% screen, ready for plot review;
- `n6_10_extremum_shot_summary.csv`: nested counts and reference-label audit;
- `n6_10_extremum_recommended_shots.csv`: refined shot priorities;
- `n6_10_extremum_candidate_counts.png`: count and threshold-sensitivity plot.

Recommended order for manual plot review is B12 first, then W29 and E55, with
F62 as a smaller follow-up. Review must still reject spiky/junk modes and any
case whose secondary crossing has material amplitude not captured reliably by
the scalar screen.


Gap-plot normalization convention: never divide the boundaries point by point
by the radius-dependent center `c(r) = 0.5 * (u(r) + l(r))`, because that hides
radial variation. If normalization is useful, divide the full panel by one
clearly stated scalar mean or median of `c(r)` over a fixed radial interval.


### 2026-08-04: first B12 manual review

Manually reviewed the 25 B12 modes selected by the 1% secondary-crossing
screen. Twelve are confirmed `good,tae`: review indices 5, 6, 7, 10, 12, 16,
20, 22, 23, and 24 are clear narrow extrema TAEs, while 15 and 25 are physical
but wider/more global TAEs. Twelve are confirmed `bad,none` because their
radial structure is much too narrow to be physical and/or has excessive
near-axis boundary structure. Only index 18 remains pending and may be GOOD.

Index 14 is `N9/egn09w.2347E+02`. It otherwise looks coherent but has a spike
near `r=0`; its maximum `W/W_peak` over `r <= 0.03` is 3.59% and 2.40% of its
integrated energy lies in that axis window. It is confirmed BAD because this
axis spike is unacceptable and would cause problems for NOVA-C. Index 18 is
`N10/egn10w.1550E+02`. It looks plausible but crosses the lower continuum at
`r=0.1973`, only 0.103 from its extremum/peak at `r=0.300`; the crossing has
`W/W_peak=0.00563`, equivalent to 7.5% of peak amplitude, and 0.73% of total
mode energy lies within 0.03 of the crossing. Thus it passes the provisional
1% pointwise-energy cutoff but is reasonably left pending because that
crossing is within the inner mode envelope rather than in a remote tail.

The active training-label list has not been changed. The decisions are staged
in `outputs/gshot_candidate_selection/nstxuG121123B12_manual_review.csv`; the
24 resolved rows are also in
`nstxuG121123B12_confirmed_labels_staging.csv`. This first review confirms that
continuum geometry plus crossing amplitude is useful for prioritization but
cannot recognize the ultra-narrow and near-axis numerical structures that
dominate the remaining false candidates.


### 2026-08-04: B12 current TAE-only label transfer

The current B12 files were copied into the isolated mixed-data working area
and split into 136 TAE-like and 501 EAE-like modes. The May all-family hand
labels are preserved as `mode_labels_clean_all.csv`. Matching by stable
`shot/N/file` suffix transfers 74 May TAE-region labels; the recent 25-mode
review independently confirms 14 of those labels and supplies 10 new resolved
labels. There are no conflicts.

Seeded current-path `mode_labels.csv` and `mode_labels_clean.csv` in the mixed
B12 directory with 84 TAE-region decisions: 33 GOOD, 44 BAD, and 7 SKIP.
Fifty-two current TAE-like modes remain unlabeled, including review index 18.
Forty-two May-only labels point to mode files newer than the May label snapshot
and have not yet been reconfirmed. The combined manual workload is therefore
94 modes: 42 rechecks plus 52 missing labels, distributed as N7=21, N8=19,
N9=22, and N10=32.

The split directory now contains `tae_like_label_transfer_audit.csv`,
`tae_like_existing_labels_to_check.csv`,
`tae_like_labels_needing_recheck.csv`, `tae_like_missing_labels.csv`, and the
ordered combined `tae_like_labels_to_review.csv`. The canonical repository
training list has not been changed. After the 94-mode pass, overlay the new
current-review labels on the 84-row seed; this replaces all flagged May labels
and fills all missing decisions while retaining the 42 safe seed labels.


### 2026-08-04: B12 independent blind-label comparison checkpoint

Created a label-free 94-mode manifest and diagnostic sheets for an independent
Codex-versus-human labeling comparison. The blind sheets show signed mode
harmonics, the `W(r)` envelope, absolute-frequency continuum boundaries,
matched extrema, and continuum crossings, but contain no May label or label
provenance. The already discussed 25-mode set was used only to calibrate the
visual morphology policy before the blind pass.

Codex completed one forced GOOD/BAD decision for all 94 review modes, with a
confidence and compact reason, and sealed the result separately at
`/p/hym/ebelova/NOVA/data_mixed/nstxuG121123B12_tae_eae_split/codex_blind_labels_SEALED.csv`.
Its SHA-256 commitment is
`f029527222237e69a44c8c20ae9fa5692fb8d9ded082368ea3d2cfc8f3ce4ff4`.
`N10/egn10w.1550E+02` is explicitly marked `prior_seen` and must be excluded
from the clean independent-agreement statistic. No blind decision has been
merged into the B12 working labels or the canonical training list; comparison
waits for the independent human pass.


### 2026-08-04: B12 blind-label comparison

The completed human review covers exactly the same 94 paths as the sealed
Codex list, and the sealed SHA-256 commitment verified before comparison.
Overall agreement is 91/94 (96.8%). Excluding the previously discussed
`N10/egn10w.1550E+02`, clean independent agreement is 90/93 (96.8%), with
Cohen kappa 0.711 despite the strong BAD-class imbalance. All three
disagreements are human GOOD versus Codex BAD; there are no Codex-only GOOD
calls.

The disagreements are `N7/egn07w.1888E+02`,
`N10/egn10w.1565E+02`, and `N10/egn10w.2629E+02`. Reinspection supports the
human GOOD decision for N10/1565: its crossings all have negligible mode
energy, including only 0.003% of integrated energy beyond the outermost
crossing. N10/2629 is also defensible as GOOD: its outer crossing has
`W/W_peak=0.00503`, 0.26% of integrated energy within +/-0.03, and 0.84%
beyond the crossing. The Codex pass was too conservative about its small
outer tail.

N7/1888 remains a real GOOD-versus-SKIP policy case. It places 89% of its
integrated energy near the lower-continuum maximum, but its detached inner
crossing reaches `W/W_peak=0.03697` and contains 1.88% of total energy within
+/-0.03. The May label was SKIP and the initial current human pass called it
GOOD. After comparison, the agreed final resolution is SKIP; the isolated
`mode_labels_current_review.csv` has been updated accordingly.

The human recheck was not generally permissive: among the 42 modes carrying
May labels but newer mode files, 24 labels changed: 19 May GOOD to current BAD
and five SKIP to BAD. No May BAD or SKIP mode was promoted to GOOD. No
review labels have yet been merged into the canonical training list.


### 2026-08-05: remaining unchanged May-labeled B12 modes

The full current B12 shot contains 575 exact filename matches to the May label
snapshot. Of those, 513 source files predate the May labels and are treated as
unchanged: 18 TAE-like and 495 EAE-like. The immediate TAE-only workflow has
18 unchanged May-labeled modes that were not part of either the 25-mode
candidate review or the completed 94-mode pass. Their May labels are 17 BAD
and one SKIP, with no GOOD labels.

Created
`tae_like_unchanged_may_labels_to_review.csv` in the isolated split directory
to review these 18 modes separately. The list spans N=1,2,3,4,6,7,8,9,10. No
labels from this additional pass have yet been merged.


### 2026-08-05: consolidated B12 TAE labels

The additional 18 unchanged May-labeled TAE-like modes were all rechecked and
classified BAD. Combined the 24 resolved modes from the first candidate
review, the 94-mode main review, and this 18-mode pass into the active B12
working labels. These three source lists are disjoint and cover all 136 rows
of the current split `tae_like.csv`, with no missing paths, duplicates, extra
paths, or conflicting decisions.

The authoritative B12 TAE-only list is now
`/p/hym/ebelova/NOVA/data_mixed/nstxuG121123B12/mode_labels_clean.csv`, ordered
exactly like the split manifest. It contains 116 BAD, 19 GOOD, and one SKIP
(`N7/egn07w.1888E+02`). `mode_labels.csv` is synchronized to the same content
so that the labeler working file is not stale. The preserved May all-family
backup `mode_labels_clean_all.csv` is unchanged. B12 has not yet been added to
the canonical repository training list.


### 2026-08-05: B12 merged into the active training list

Created `training_labels/additions/tae_like_nstxuG121123B12.csv` by joining
the final B12 working labels to the current split metadata and converting the
mode paths to relative `shot/N/file` form. The one SKIP decision is retained
in the full per-shot label file but excluded from training, leaving 135 B12
training rows: 19 GOOD and 116 BAD.

Appended the reviewed B12 addition to
`training_labels/tae_like_train.csv`. The active list now contains 2745 unique
rows across 14 shots: 622 GOOD and 2123 BAD. Validation found the expected
full schema, no duplicate or absolute paths, all 2745 files resolvable under
`/p/hym/ebelova/NOVA/data_mixed`, empty error fields, and consistent family
labels (`tae` for GOOD and `none` for BAD). The saved RF and CNN checkpoints
have not yet been retrained on this 14-shot list.



### 2026-08-05: repository skill for blind TAE-like labeling

Created the repo-scoped Codex skill
`.agents/skills/label-tae-like-modes/` for independent physics-based review of
TAE-like modes. Its non-negotiable blind-review policy forbids RF, CNN,
ensemble, probability, and all other classifier output, and also forbids any
previous human or automated labels for the target shot until the independent
decisions are sealed. Accidental exposure is recorded as `prior_seen` and
excluded from clean agreement statistics.

The skill includes a detailed continuum/morphology policy and deterministic
utilities to reject contaminated manifests, create blind decision templates,
render raw signed-mode and absolute-continuum diagnostics, validate and seal a
complete review with SHA-256, and compare with a human list only after sealing.
Synthetic tests verified contamination rejection, exact-coverage sealing,
hash verification, clean-agreement exclusion, and disagreement reporting. A
one-mode B12 smoke test verified the model-free diagnostic renderer using the
repository NOVA loader and continuum calculations. A fresh-context agent then
used the skill for a blind two-mode B12 review, independently classified both
modes BAD with high confidence, and reported no exposure to target-shot labels
or model output. No labels or model checkpoints were changed by this work.

### 2026-08-05: W29 blind human-agent labeling comparison

Prepared and independently reviewed all 158 rows in the W29 TAE-like split.
The Codex review used only raw signed harmonics, W(r), absolute continuum,
true sign-change crossings, and deterministic extremum diagnostics; no RF,
CNN, or prior W29 label file was used. Before sealing, the human reviewer
stated that all 14 N=1 modes were BAD, so those rows are explicitly marked
`prior_seen=true` and excluded from clean statistics. The remaining 144 rows
are a clean blind comparison. The sealed agent list has SHA-256
`c0192dd7c2e84201fbc9d7d42c152dc35e03677f8fe7b8ab6389a1d8a2562c98`.

After completing the initially missed 22 N=5 rows, the human list covered the
same 158 modes exactly. Overall agreement was 149/158 (94.30%, Cohen kappa
0.5004); clean agreement was 135/144 (93.75%, kappa 0.4977). Disagreements
were eight agent-GOOD/human-BAD and one agent-BAD/human-GOOD. Confidence was
well calibrated: high-confidence agreement was 134/134, medium 10/13, and low
5/11. Post-comparison raw review indicates that the agent was too permissive
when favorable extremum geometry coincided with an unresolved radial spike or
near-axis ringing. The human-GOOD decision for N8/egn08w.2847E+02 is
defensible because its principal inner structure is coherent and its true
crossings occur in a low-amplitude outer tail.

Preserved the immutable sealed list, SHA-256 sidecar, label-free missing-N5
audit list, and detailed human-agent comparison in the W29 split directory.
No W29 labels have been merged into the canonical training list, and no model
has been retrained on W29.

Post-seal adjudication retained N5/egn05w.2505E+02 and
N8/egn08w.2847E+02 as GOOD with low confidence for physics reasons. The N5
working human label was changed from BAD to GOOD; N8 was already GOOD. The
sealed blind reviews and original comparison remain unchanged as audit
artifacts. The working W29 list now contains 7 GOOD and 151 BAD modes.
After exact coverage and uniqueness validation against `tae_like.csv`, this
adjudicated list was promoted unchanged to the shot-local `mode_labels_clean.csv` for
consistency with other shots. It has not yet been merged into the canonical
training list.

### 2026-08-05: W29 merged into the active training list

Created `training_labels/additions/tae_like_nstxuG142301W29.csv` by joining
the final W29 `mode_labels_clean.csv` decisions to the split metadata and
converting paths to relative `shot/N/file` form. The component covers all 158
TAE-like modes: 7 GOOD and 151 BAD, with no SKIP rows.

Appended the reviewed W29 component to `training_labels/tae_like_train.csv`.
The active list now contains 2903 unique rows across 15 shots: 629 GOOD and
2274 BAD. Validation found the expected full schema, exact W29 manifest
coverage, no duplicate or absolute paths, all files resolvable under
`/p/hym/ebelova/NOVA/data_mixed`, empty error fields, and consistent family
labels (`tae` for GOOD and `none` for BAD). All 2745 pre-existing rows remain
byte-for-byte unchanged. The saved RF and CNN checkpoints have not yet been
retrained on this 15-shot list.

### 2026-08-06: 15-shot LOSO after B12/W29 merge

User ran the 15-shot LOSO check after adding the targeted B12 and W29 G-shot
labels. Results are under `outputs/loso_15_B12_W29_M100_bs8/`, with work files
under `$SCRATCH/nova_sc` / `$SCRATCH/nova_s` depending on the run environment.
The run is directly comparable to the previous `outputs/loso_13_M100` check:
both used seed 42, raw-CNN `M_target=100`, `R_target=201`, 80 epochs, batch
size 8, learning rate 0.02, robust normalization, no positive-class weighting,
and full-CNN refits before sorting.

Aggregate 15-shot held-out metrics:

- CNN: CM `[[2162, 112], [94, 535]]`, accuracy `0.9290`, GOOD
  precision/recall/F1 `0.8269 / 0.8506 / 0.8386`.
- Combined policy: CM `[[2228, 46], [117, 512]]`, accuracy `0.9439`, GOOD
  precision/recall/F1 `0.9176 / 0.8140 / 0.8627`.
- RF: CM `[[2233, 41], [142, 487]]`, accuracy `0.9370`, GOOD
  precision/recall/F1 `0.9223 / 0.7742 / 0.8418`.

The old 13-shot LOSO outputs predate the August G-label corrections, so the
cleanest comparison re-scores the saved 13-shot predictions against the
current active labels for the old 13 shots. On those same old 13 held-out
shots, adding B12/W29 to the training folds changed the metrics as follows:

- CNN: CM `[[1911, 96], [87, 516]]` -> `[[1902, 105], [81, 522]]`; GOOD F1
  `0.8494 -> 0.8488`. Recall improved slightly (`0.8557 -> 0.8657`) but
  precision fell (`0.8431 -> 0.8325`).
- Combined policy: CM `[[1952, 55], [108, 495]]` -> `[[1965, 42], [108, 495]]`;
  GOOD F1 `0.8586 -> 0.8684`. The improvement comes from 13 fewer false
  positives with the same number of false negatives.
- RF: CM `[[1961, 46], [130, 473]]` -> `[[1969, 38], [132, 471]]`; GOOD F1
  `0.8431 -> 0.8471`. Precision improved while recall changed little.

Subset interpretation:

- Old six G shots only, current labels: CNN GOOD F1 improved
  `0.507 -> 0.643`, with GOOD recall `0.614 -> 0.807`; combined policy
  improved `0.509 -> 0.594`; RF improved `0.500 -> 0.526`. This is the
  clearest evidence that the targeted B12/W29 extremum-localized TAE labels
  improved transfer in the G-shot regime.
- Seven non-G shots only: CNN GOOD F1 decreased `0.893 -> 0.876`;
  combined policy stayed essentially unchanged, `0.894 -> 0.895`; RF also
  stayed essentially unchanged, `0.877 -> 0.877`. Thus the targeted G-shot
  labels did not materially improve non-G performance and did not harm the
  production combined policy.
- New B12/W29 held-out folds alone: CNN GOOD precision/recall/F1
  `0.650 / 0.500 / 0.565`; combined policy `0.810 / 0.654 / 0.723`; RF
  `0.842 / 0.615 / 0.711`. B12 is handled better than W29; W29 remains a
  difficult sparse-GOOD shot.

Conclusion: adding B12 and W29 had the intended selective effect. It made the
G-shot regime noticeably better, especially for raw-CNN GOOD recall and for
the combined policy's G-shot F1, while leaving non-G RF/combined performance
essentially stable. The result supports retaining B12/W29 in the active
training list and continuing targeted G-shot labeling, but the new shots are
not sufficient to make G-shot sorting routine-production ready without
review or a G-specific policy/calibration pass.

### 2026-08-06: shared datcon outer-tail repair

Added a conservative shared cleanup in `src/cont_features.py` for bogus
outer-boundary datcon tails before developing continuum-aware CNN inputs.
Existing handling already masks explicit NOVA sentinel values above `999`.
The new paired repair works after sentinel masking, in physical-frequency
space: if both `sqrt(low2)` and `sqrt(high2)` jump upward together in the
outer finite tail with a large local slope, the suspicious finite tail is
replaced by a constant extension equal to the average of the previous few
reliable interior points. Explicit sentinel / missing regions remain `NaN`,
and the older one-boundary trailing-spike trim still runs afterward.

This is intentionally conservative and shared through `load_datcon_for_mode`,
so RF features, TAE/EAE splitting, sorting, plotting, and the planned
continuum-CNN channel construction all see the same repaired continuum arrays.
Because the loader behavior changed, any RF/CNN model comparisons after this
point should be treated as using a new continuum-cleaning version and should
be retrained/revalidated before replacing production checkpoints.

Focused tests were added for repairing a joint outer-tail blow-up and leaving
a normal smooth outer tail unchanged. `python -m unittest
tests/test_continuum_crossing_features.py` passed: 26 tests OK after the
continuum-channel tests below were added. The test
process printed unrelated MUNGE authentication warnings, but exited
successfully.

A read-only scan of the active 15-shot training list found 146 unique datcon
files, all present under `/global/cfs/cdirs/m314/nova2/data`. The new paired
repair changed 13 of them relative to the previous sentinel-plus-one-sided
trim behavior. Most repairs affected one outer radial point; two S31 files
affected two outer points.

### 2026-08-06: experimental raw CNN with continuum channels

Added an experimental continuum-aware variant of the raw CNN without changing
the default production raw-CNN path. Passing `--continuum_channels` to
`scripts/cnn_raw.py` changes the input from one channel to three channels:

- channel 0: normalized raw signed mode image, as before;
- channel 1: `du(r) = (sqrt(high2) - omega) / omega`, broadcast over the
  harmonic axis;
- channel 2: `dl(r) = (omega - sqrt(low2)) / omega`, broadcast over the
  harmonic axis.

Inside the local TAE gap both continuum channels are positive; upper/lower
continuum crossings occur where `du=0` / `dl=0`. The mode channel is normalized
with the existing per-image normalization, but the continuum channels are only
radially interpolated, clipped to `--continuum_clip` (default `5.0`), and kept
in physical relative-frequency units. Missing or unusable datcon input falls
back to zero-valued continuum channels for this first two-channel experiment;
a separate validity mask channel remains a possible follow-up if needed.

The shared implementation lives in `scripts/cnn_infer_common.py` so training
and inference use the same preprocessing. Continuum-aware checkpoints save
`model_type=cnn_raw_continuum`, `input_channels=3`, `continuum_channels=True`,
and the continuum clip value. `cnn_classify.py` auto-detects the new checkpoint
kind, `sort_shot_mixed.py` accepts `--cnn_model_kind cnn_raw_continuum`, and
`scripts/run_loso_10.py` accepts `--cnn_continuum_channels` plus
`--cnn_continuum_clip` for LOSO experiments.

Validation so far:

- `python -m unittest tests/test_continuum_crossing_features.py` passed:
  26 tests OK. New tests cover continuum-channel sign convention, broadcast
  shape, clipping, missing-datcon fallback, and the shared datcon tail repair.
- `python -m py_compile scripts/cnn_raw.py scripts/cnn_infer_common.py
  scripts/cnn_classify.py scripts/sort_shot_mixed.py scripts/run_loso_10.py`
  passed.
- A one-epoch CPU smoke train with `--continuum_channels` on the
  `nstx_120113` LOSO test fold saved `/tmp/nova_cnn_raw_continuum_smoke.pt`
  with `model_type=cnn_raw_continuum`; `scripts/cnn_classify.py` successfully
  loaded it in auto mode and produced a prediction.

No production checkpoint has been replaced. The next meaningful check is a
15-shot LOSO run comparing this continuum-channel raw CNN against the current
raw-CNN baseline, with all-shot, non-G, old-G, B12/W29, and all-G summaries.

### 2026-08-06: 15-shot continuum-channel raw-CNN LOSO result

User ran the experimental continuum-channel LOSO under
`outputs/loso_15_raw_continuum_M100_bs8/`. The configuration matches the
previous no-continuum 15-shot run in `outputs/loso_15_B12_W29_M100_bs8` except
for `cnn_continuum_channels=True` and `cnn_continuum_clip=5.0`: seed 42,
`M_target=100`, `R_target=201`, 80 epochs, batch size 8, LR 0.02, robust
normalization, no positive-class weighting, cached data, and full-CNN refits.

Aggregate continuum-channel metrics:

- CNN: CM `[[2188, 86], [105, 524]]`, accuracy `0.9342`, GOOD
  precision/recall/F1 `0.8590 / 0.8331 / 0.8458`.
- Combined policy: CM `[[2226, 48], [121, 508]]`, accuracy `0.9418`, GOOD
  precision/recall/F1 `0.9137 / 0.8076 / 0.8574`.
- RF: CM `[[2231, 43], [135, 494]]`, accuracy `0.9387`, GOOD
  precision/recall/F1 `0.9199 / 0.7854 / 0.8473`. RF changed slightly
  relative to the previous run because the shared datcon outer-tail repair now
  affects RF feature construction; RF numbers are not a pure continuum-CNN
  comparison.

Compared with the no-continuum raw CNN:

- All 15 shots: CNN F1 improved slightly `0.8386 -> 0.8458`, driven by fewer
  false positives (`112 -> 86`) despite more false negatives (`94 -> 105`).
- Seven non-G shots: CNN F1 improved `0.8758 -> 0.8903`; false positives fell
  `65 -> 46`, while false negatives were essentially unchanged (`70 -> 71`).
- Old six G shots: CNN worsened substantially, F1 `0.6434 -> 0.5333`; false
  negatives increased `11 -> 25` and GOOD recall fell `0.807 -> 0.561`.
- New targeted B12/W29 folds: CNN improved, F1 `0.5652 -> 0.6538`; false
  negatives fell `13 -> 9`, with B12 improving strongly and W29 changing only
  modestly.
- All eight G shots together: CNN worsened, F1 `0.6243 -> 0.5698`; the B12/W29
  gain did not compensate for the old-G regression.

The combined RF/CNN policy did not benefit from the continuum-channel CNN:
all-shot F1 changed `0.8627 -> 0.8574`, non-G remained essentially unchanged
(`0.8951 -> 0.8940`), old-G worsened `0.5941 -> 0.5400`, and B12/W29 was
unchanged at `0.7234`.

Interpretation: the two continuum-distance image channels make the CNN more
conservative and improve non-G precision, but they do not provide a robust
G-shot rescue. They help the newly targeted B12/W29 extrema-localized examples
but hurt transfer to several older G shots, especially Q62, K51, and H47.
Do not promote this continuum-channel raw CNN or retune the production fusion
policy from this result alone. Useful follow-ups are to inspect old-G modes
whose CNN score dropped sharply, try a validity-mask channel or different
continuum scaling/clipping, and compare against a small 1D continuum branch
instead of broadcasting continuum arrays across the harmonic axis.

### 2026-08-06: radius-aligned 1D continuum branch replaces broadcast training

The broadcast continuum-channel experiment is retired from active training.
Its `cnn_raw_continuum` checkpoints remain loadable so the completed LOSO run
can still be inspected, but `scripts/cnn_raw.py` no longer exposes the
`--continuum_channels` trainer flag.

Added a new experimental `--continuum_branch` raw-CNN variant with checkpoint
kind `cnn_raw_continuum_branch`. The raw signed mode remains a one-channel
`(m,r)` image. A separate 1D branch receives four arrays on the same radial
grid, in this fixed order:

- `W_norm(r) = sum_m |xi_m(r)|^2 / max_r W(r)`;
- `du(r) = (sqrt(high2(r)) - omega) / omega`;
- `dl(r) = (omega - sqrt(low2(r))) / omega`;
- a binary continuum-validity mask.

The cleaned shared datcon loader supplies the boundaries. `du` and `dl` are
interpolated to `R_target`, clipped to `--continuum_clip` (default `5.0`), and
set to zero where the resampled mask is invalid. Missing/unreadable datcon
leaves `du`, `dl`, and the mask at zero while retaining the mode-derived
`W_norm`, avoiding the old ambiguity in which a zero fallback could look like
a physical crossing.

The mode trunk retains radial resolution through its two pooling stages and is
averaged only over the harmonic dimension. The 1D branch uses matching radial
pooling. Their feature maps are concatenated at corresponding radial bins,
passed through a fusion convolution, and only then globally pooled for the
classifier. This lets the network learn whether a continuum crossing and
material mode amplitude occur at the same radius instead of treating any
crossing as a global rejection signal.

LOSO wiring now uses `--cnn_continuum_branch`; sorting accepts
`--cnn_model_kind cnn_raw_continuum_branch`, and run metadata records the new
flag. Training and inference share the branch-array construction, and new
checkpoints record the four feature names and continuum clip.

Validation:

- `python -m unittest tests/test_continuum_crossing_features.py` passed: 29
  tests OK. New coverage checks `W_norm`, `du`/`dl` signs, partial and missing
  continuum masks, and the two-input model output shape. Unrelated MUNGE
  authentication warnings were printed, as in earlier runs.
- Python compilation passed for the raw trainer, shared inference loader,
  canonical classifier, mixed sorter, and LOSO driver.
- A cached one-epoch CPU smoke train saved
  `/tmp/nova_cnn_raw_continuum_branch_smoke.pt`; canonical
  `scripts/cnn_classify.py` auto-detected it as
  `cnn_raw_continuum_branch` and completed inference.
- A LOSO dry run produced the expected `--continuum_branch`, clip, cache, and
  full-refit command and recorded `cnn_continuum_branch=true` in
  `run_config.json`.

No production model was replaced. The next scientific check is a matched
15-shot LOSO run against `outputs/loso_15_B12_W29_M100_bs8`, with special
attention to old-G recall and whether the B12/W29 gain survives without the
Q62/K51/H47 regression.

### 2026-08-07: 15-shot radius-aligned continuum-branch LOSO result

The matched 15-shot run completed under
`outputs/loso_15_raw_continuum_branch_M100_bs8/`. All 15 folds and 2903
held-out labeled modes were evaluated with no load failures. Configuration
matched the no-continuum and broadcast experiments: seed 42, `M_target=100`,
`R_target=201`, 80 epochs, batch size 8, LR 0.02, robust normalization, no
positive-class weighting, cached data, and full-data refits.

Aggregate CNN results were worse than both earlier variants:

- no continuum: GOOD precision/recall/F1 `0.827 / 0.851 / 0.839`, FP/FN
  `112 / 94`;
- broadcast continuum: `0.859 / 0.833 / 0.846`, FP/FN `86 / 105`;
- radius-aligned branch: `0.827 / 0.812 / 0.820`, FP/FN `107 / 118`.

The branch partially repaired the broadcast old-G regression but did not
recover the no-continuum baseline. Old-G F1/FN were `0.643 / 11` for baseline,
`0.533 / 25` for broadcast, and `0.580 / 17` for the branch. B12/W29 branch
F1/FN were `0.604 / 10`, between baseline (`0.565 / 13`) and broadcast
(`0.654 / 9`). Non-G performance worsened to F1 `0.862` and 91 FN, versus
baseline `0.876` and 70 FN. K51 and B12 improved, but E202855, E205052,
141711, and Q62 produced the largest false-negative regressions.

The combined policy remained effectively unchanged: baseline GOOD F1 was
`0.863` with FP/FN `46 / 117`, while the branch gave `0.861` with `49 / 116`.
Training logs contained no collapse warnings. A threshold sweep did not rescue
the branch: aggregate average precision fell `0.878 -> 0.834`, and the best
global branch threshold reached only F1 `0.825`. This indicates weaker ranking,
not merely shifted calibration. Do not promote this branch.

Interpretation is limited by an architecture confound: the branch experiment
changed the fusion/head architecture as well as adding `W_norm`, `du`, `dl`,
and the validity mask. An architecture-only zero-input ablation is needed to
separate those effects.

### 2026-08-07: architecture-only zero-branch LOSO control

Added `--continuum_branch_zero_inputs` to `scripts/cnn_raw.py` and
`--cnn_continuum_branch_zero_inputs` to the LOSO driver. The control retains
the exact `ContinuumBranchCNN` architecture, including its 1D branch, radial
fusion convolution, and head, but replaces all four branch inputs with an
exact `(4, R_target)` float32 zero tensor. This includes `W_norm`; retaining it
would test architecture plus an additional radial mode-envelope feature rather
than architecture alone. Datcon loading and branch-feature construction are
bypassed in this mode.

The zero-input choice is stored in checkpoint preprocessing metadata as
`continuum_branch_zero_inputs=True` and honored by canonical inference, so a
control checkpoint cannot accidentally receive physical continuum inputs when
sorting its held-out shot. Run metadata records the corresponding LOSO flag.

Validation:

- `python -m unittest tests/test_continuum_crossing_features.py` passed: 30
  tests OK, including exact zero tensor shape/dtype and metadata resolution;
- Python compilation and `git diff --check` passed;
- a LOSO dry run emitted both branch flags and recorded the zero-input flag;
- a cached one-epoch CPU smoke train saved
  `/tmp/nova_cnn_raw_continuum_branch_zero_smoke.pt`; canonical inference
  loaded it successfully and retained the zero-input preprocessing metadata.

No production checkpoint was changed. The matched control output should use
`outputs/loso_15_raw_continuum_branch_zero_M100_bs8` so it cannot overwrite
the physical-branch run.

### 2026-08-07: NSTX-U 204202 post-seal TAE-like label audit

An independent 275-mode audit of `nstxu_204202` was completed by
**Codex-terra-High (GPT-5 family)** from the label-free
`tests/labels_audit/tae_like_audit.csv` manifest. The review used raw signed
mode-structure and continuum diagnostics only; RF, CNN, ensemble, and other
classifier outputs were not run or inspected. The review was sealed before
the human or canonical training labels were opened. The sealed review contains
76 GOOD, 193 BAD, and 6 SKIP decisions; its SHA-256 is
`db7ffe80e754e36d28409a728c6dd6d3861723c6d52b95dbf22163a5c8d90ef8`.

After the human labels were revised, all 275 audited paths matched between the
sealed review, `tests/labels_audit/labels_human_review.csv`, and
`training_labels/tae_like_train.csv`. Final comparisons are:

- Codex-terra versus human: 247/275 agreement (89.82%), Cohen's kappa
  0.7447. Disagreements: BAD->GOOD 5, GOOD->BAD 17, SKIP->BAD 5, and
  SKIP->GOOD 1.
- Codex-terra versus training labels: 253/275 agreement (92.00%), Cohen's
  kappa 0.8069. Disagreements: BAD->GOOD 7, GOOD->BAD 9, SKIP->BAD 5, and
  SKIP->GOOD 1.
- Human versus training labels: 265/275 agreement (96.36%), Cohen's kappa
  0.9043. All 10 disagreements are human BAD versus training GOOD.

The Terra disagreement artifact is retained as
`tests/labels_audit/nstxu_204202/disagreements_terra_h.csv` when this
historical review needs to be inspected. No training labels or production
models were modified from this audit alone; any training-list change requires
explicit adjudication and approval.

### 2026-08-08: Codex blind re-audit of `nstxu_204202` and skill-policy update

A second independent Codex blind audit of the same 275 `nstxu_204202` training
shot modes was completed from the path-only audit manifest. This review used
only raw signed mode structures and continuum diagnostics; no RF, CNN,
ensemble, classifier outputs, or prior labels were used before sealing. The
sealed review is kept at
`tests/labels_audit/nstxu_204202/nstxu_204202_codex_blind_labels_SEALED.csv`
with SHA-256
`68e0ad5428426732969658196b339b329b08e427dfa978bac51f03674f0524a6`.

Sealed Codex counts were 64 GOOD, 208 BAD, and 3 SKIP decisions. After
opening the human review and active training list, the comparison was:

- Codex versus human review: 261/275 agreement (94.91%), Cohen's kappa
  0.8614. Disagreements: BAD->GOOD 6, GOOD->BAD 5, SKIP->BAD 3.
- Codex versus `training_labels/tae_like_train.csv`: 260/275 agreement
  (94.55%), Cohen's kappa 0.8589. Disagreements: BAD->GOOD 11,
  GOOD->BAD 1, SKIP->BAD 2, SKIP->GOOD 1.
- Human review versus the current `training_labels/tae_like_train.csv`:
  all 275 shared `nstxu_204202` paths are present in both files, with 10
  label changes. All 10 are current training GOOD -> human BAD. Four
  review-only EAE rows were removed from the human review before this count.
- The retained disagreement/audit discussion table is
  `tests/labels_audit/nstxu_204202/nstxu_204202_codex_union_disagreements.csv`.
  The retained human-versus-training delta table is
  `tests/labels_audit/nstxu_204202/nstxu_204202_human_vs_training_label_changes.csv`.
  Scratch manifests, raw measurement tables, intermediate comparison CSVs,
  and diagnostic plot directories were removed to keep `tests/labels_audit/`
  compact.

Discussion of the disagreements produced several updates to the
`label-tae-like-modes` skill policy in
`.agents/skills/label-tae-like-modes/references/labeling-policy.md`:

- Near-axis boundary artifacts should be screened primarily by pointwise
  normalized amplitude near `r=0`, not only by integrated near-axis energy.
  Narrow detached axis spikes with appreciable amplitude are BAD even if
  their integrated energy is small.
- A continuum crossing inside the connected mode body is BAD even when there
  is no sharp grid-point spike at the resonance, because the resonant point
  may fall between radial grid points. Crossings are acceptable only in
  detached or negligible tails with very small local pointwise and integrated
  energy.
- The policy now records four useful TAE morphology families: wide/global
  modes, edge-localized modes, continuum-extremum-localized modes, and mixed
  modes. This language prevents over-rejecting physical edge-localized modes
  whose individual poloidal harmonics become visually narrow or spiky at
  large radius because of magnetic shear, while the total envelope remains
  coherent.
- Narrow modes near a maximum of the lower continuum boundary or minimum of
  the upper continuum boundary can be GOOD when they nearly touch the
  continuum extremum without crossing through the connected mode body.

The sealed list remains a blind-review artifact and was not retroactively
edited after discussion. The active `training_labels/tae_like_train.csv` was
not changed. The `nstxu_204202` human-review corrections should be applied
later by creating a new versioned training list while preserving the current
training list.

### 2026-08-09: NSTX training-shot audit cleanup and `nstx_135388` policy-v2 closure

The blind audit sequence was extended to `nstx_120113`, `nstx_141711`, and
`nstx_135388` using the same label-free workflow as the second `nstxu_204202`
audit: start from the clean `tests/labels_audit/tae_like_audit.csv` path
manifest, inspect raw signed mode structure and continuum diagnostics, do not
use RF/CNN/ensemble/model outputs, and seal the Codex review before opening
human or training labels.

Retained artifacts were pruned to the compact audit record under
`tests/labels_audit/<shot>/`. Large diagnostic PNG directories, scratch
manifests, raw-measurement tables, and intermediate comparison CSVs were
removed. The retained file classes are sealed blind Codex labels plus SHA-256
sidecars, union disagreement/adjudication tables, human-vs-training change
tables, and for `nstx_135388` the policy-v2 results described below.

Shot status:

- `nstx_120113`: 174 sealed rows, with 45 GOOD, 127 BAD, and 2 SKIP decisions.
  Sealed SHA-256:
  `c89034c31b69686f92e82194032748961a4cc42e0d74ed0c4e75caef33cbf426`.
  The retained human-vs-training delta table has two rows:
  `N10/egn10w.2945E+02` human BAD versus training GOOD, and
  `N6/egn06w.1418E+02` human SKIP versus training GOOD. No training CSV was
  changed.
- `nstx_141711`: 256 sealed rows, with 80 GOOD and 176 BAD decisions. Sealed
  SHA-256:
  `5a302ae896b926ad1b4a33e711656d92344d2b2f08204366e8021e785288b8a2`.
  The retained human-vs-training change table has 17 rows: 16 old-training
  GOOD -> new-human BAD and 1 old-training BAD -> new-human GOOD. The retained
  union adjudication table has 35 review rows, with final adjudication calling
  21 BAD, 12 GOOD, 1 SKIP, and leaving 1 row without a final label in the
  table. No training CSV was changed.
- `nstx_135388`: 344 sealed blind rows, with 174 GOOD and 170 BAD decisions.
  Sealed SHA-256:
  `17788d736e68fb74c176706379b9948161fc2cee6ebb08dd5a410eba80084ef7`.
  The pre-adjudication Codex-vs-human agreement was 297/344 = 86.34% with
  Cohen's kappa 0.7275, leaving 47 disagreements for discussion.

The `nstx_135388` discussion exposed several policy gaps, and
`.agents/skills/label-tae-like-modes/references/labeling-policy.md` was
updated accordingly:

- Red-flag checks for continuum crossing and radial-boundary artifacts are now
  applied before assigning a plausible morphology family.
- True continuum crossings with pointwise `W(r_cross) / max(W) >= ~0.1` are
  BAD unless they are clearly in smooth detached tails with very small local
  integrated energy; crossings near `~0.2-0.3` or larger are strong BAD
  evidence even away from the global peak.
- Near-axis one/few-grid-point spikes, short low-r grid-scale oscillatory
  packets, and single-harmonic axis artifacts are BAD when they carry
  appreciable pointwise amplitude, even if integrated near-axis energy is
  small.
- Outer-boundary artifacts must be checked in individual signed harmonics as
  well as in summed `W(r)`, because a single-harmonic endpoint spike can be
  hidden in the summed envelope.
- Grid-scale oscillations at the mode peak or inside the connected mode body
  are BAD even when confined to one appreciable harmonic. Ordinary type-2 edge
  sign changes and shear-narrowed harmonics remain acceptable when the total
  envelope is resolved and coherent.
- Finite-radius type-3 continuum-extremum modes remain GOOD candidates when
  they are localized near a lower-continuum maximum or upper-continuum
  minimum, nearly touch but do not cross the continuum, and pass the boundary
  and grid-scale red-flag checks.

After the policy update, a non-blind post-adjudication
`nstx_135388` policy-v2 pass was written to
`tests/labels_audit/nstx_135388/nstx_135388_codex_policy_v2_labels.csv`.
This file is explicitly not an independent validation pass: it carries
`review_type=not_blind_post_adjudication_policy_v2` and `prior_seen=true`.
Its SHA-256 is
`5fc964be25adf8636be697544abdad91416264d44c2a05212084f29193f482b3`.
Policy v2 has 131 GOOD and 213 BAD labels. It changed 47 sealed-blind Codex
decisions: 45 GOOD -> BAD under the new red-flag rules, and 2 BAD -> GOOD
for resolved continuum-extremum cases including `B318`.

The policy-v2 labels match the current human review exactly for all 344
`nstx_135388` modes, but this is an adjudication closure result, not a clean
blind-agreement statistic. Relative to the current
`training_labels/tae_like_train.csv`, policy v2/human differs on 11 rows:
10 old-training BAD -> new GOOD and 1 old-training GOOD -> new BAD. The
canonical training list was not changed; these corrections should be applied
later by creating a new versioned training list while preserving the current
one.

### 2026-08-09: `nstxuE202855A01t020` blind audit cleanup

The label-free blind workflow was repeated for `nstxuE202855A01t020` using the
79 paths in `tests/labels_audit/nstxuE202855A01t020/tae_like_audit.csv`. The
sealed Codex blind list has 42 GOOD and 37 BAD decisions, with confidence
counts of 55 high, 23 medium, and 1 low. Sealed SHA-256:
`2c3e82e4e3166374d9a20a67236de46ab2bd766cc9f3a555978db34e3838f06c`.

After correcting the human review for `B004` and `B005` to BAD, the clean
comparison statistics are:

- Codex blind vs human clean: 65/79 agreement = 82.28%, Cohen's kappa 0.6395.
- Codex blind vs current training list: 61/79 agreement = 77.22%, Cohen's
  kappa 0.5380.
- Human clean vs current training list: 71/79 agreement = 89.87%, Cohen's
  kappa 0.7852.

The retained shot-local artifacts were pruned to:

- `nstxuE202855A01t020_codex_blind_labels_SEALED.csv`
- `nstxuE202855A01t020_codex_blind_labels_SEALED.csv.sha256`
- `nstxuE202855A01t020_codex_union_disagreements.csv`
- `nstxuE202855A01t020_human_vs_training_label_changes.csv`

The human-vs-training change table has 8 rows: 5 old-training BAD -> new-human
GOOD and 3 old-training GOOD -> new-human BAD. The canonical
`training_labels/tae_like_train.csv` was not changed.

The post-seal disagreement discussion tightened the policy language for this
shot: compact but smooth type-3 continuum-extremum modes are not
"ultra-narrow" merely because they are localized, and high-n edge-localized
type-2 modes can show shear-narrowed, visually spiky harmonics near large
radius while the connected envelope remains resolved and physical. Several
sealed Codex BAD calls on those morphologies were judged too conservative
during discussion, but the sealed blind list remains unchanged as the
provenance-preserving record.

### 2026-08-09: `nstxuE204669M03t025` blind audit cleanup

The label-free blind workflow was repeated for `nstxuE204669M03t025` using
217 clean target paths. The sealed Codex blind list has 88 GOOD and 129 BAD
decisions. Six early decisions (`B001`-`B006`) are marked `prior_seen=true`
because target-shot human-review rows were accidentally exposed before the
shot was defined; clean agreement statistics therefore exclude those rows.
Sealed SHA-256:
`479e7986713d80f54d5beca497cbbf128f93ea3d0d9f2df34914ccd8886d831f`.

After the post-seal discussion and human-review corrections for
`B141`, `B156`, `B171`, `B172`, and `B187`, the comparison statistics are:

- Codex blind vs human: 209/217 agreement = 96.31%, Cohen's kappa 0.9227.
- Codex blind vs human clean: 203/211 agreement = 96.21%.
- Codex blind vs current training list: 206/217 agreement = 94.93%, Cohen's
  kappa 0.8943.
- Human clean vs current training list: 212/217 agreement = 97.70%, Cohen's
  kappa 0.9513.

The retained shot-local artifacts were pruned to:

- `nstxuE204669M03t025_codex_blind_labels_SEALED.csv`
- `nstxuE204669M03t025_codex_blind_labels_SEALED.csv.sha256`
- `nstxuE204669M03t025_codex_union_disagreements.csv`
- `nstxuE204669M03t025_human_vs_training_label_changes.csv`

The retained union table has 12 rows: 7 cases where the human review matches
the current training list but Codex differs, 1 case where Codex matches the
current training list but the human review differs, and 4 cases where both new
reviews agree against the current training list. The remaining Codex-human
disagreements are `B029`, `B033`, `B038`, `B082`, `B088`, `B095`, `B096`, and
`B129`. The human-vs-training change table has 5 rows: 4 old-training GOOD ->
new-human BAD and 1 old-training BAD -> new-human GOOD. The canonical
`training_labels/tae_like_train.csv` was not changed.

### 2026-08-09: human-review v2 training list

Created `training_labels/tae_like_v2_nonG.csv` from the cleaned human review
file `tests/labels_audit/labels_human_review_clean.csv`, while preserving the
previous canonical `training_labels/tae_like_train.csv` unchanged. The new
list keeps the same schema as the canonical training CSV:
`path,validity,family,signed_delta,fraction_below_upper2,gap_region,error`.

The cleaned human-review file contains 1638 non-G rows: 511 `good`, 1124
`bad`, and 3 `skip`. All 1638 normalized paths matched the non-G rows in
`tae_like_train.csv`; there were no missing paths and no conflicting duplicate
paths. The three `skip` rows were excluded from the training CSV, leaving 1635
reviewed non-G rows. All 1265 `nstxuG*` rows were then copied from
`tae_like_train.csv` unchanged. The resulting list has 2900 data rows:
594 `good` and 2306 `bad`. After the LOSO check below, this list is the
current canonical/default training list while awaiting review of all G shots.

Rows excluded because the human review now marks them `skip`:

- `nstx_120113/N6/egn06w.1418E+02` (old label `good`)
- `nstxuE205052A01t022/N10/egn10w.1302E+02` (old label `good`)
- `nstxuE205052A01t022/N9/egn09w.1506E+02` (old label `good`)

Label flips relative to `training_labels/tae_like_train.csv`:

| shot | good -> bad | bad -> good |
| --- | ---: | ---: |
| `nstxu_204202` | 10 | 0 |
| `nstx_120113` | 0 | 0 |
| `nstx_135388` | 1 | 10 |
| `nstx_141711` | 16 | 1 |
| `nstxuE202855A01t020` | 3 | 5 |
| `nstxuE204669M03t025` | 4 | 1 |
| `nstxuE205052A01t022` | 16 | 1 |

Totals: 50 old `good` rows became `bad`, 18 old `bad` rows became `good`, and
three old `good` rows were removed as `skip`. The copied G-shot rows have zero
label changes.

### 2026-08-09: 15-shot LOSO on v2 non-G labels

User ran the no-continuum 15-shot LOSO check on the v2 list:
`outputs/loso_15_v2_nonG_M100_bs8/`, with work files under
`/pscratch/sd/e/ebelova/nova_s/loso_15_v2_nonG_M100_bs8`. The configuration
matches the previous no-continuum 15-shot baseline
`outputs/loso_15_B12_W29_M100_bs8/`: seed 42, raw-CNN `M_target=100`,
`R_target=201`, 80 epochs, batch size 8, learning rate 0.02, robust
normalization, no positive-class weighting, no continuum branch, and full-CNN
refits before sorting. The only intended input change is the training label
CSV: `training_labels/tae_like_v2_nonG.csv` instead of
`training_labels/tae_like_train.csv`.

Run health:

- all 15 folds completed RF, CNN, sorting, and aggregation;
- 2900 held-out labels were matched, as expected after excluding three non-G
  `skip` rows;
- split counts contain no `n_other` rows;
- log scan found no traceback/error/collapse/near-all prediction warnings.

Aggregate v2 LOSO metrics:

- CNN: CM `[[2219, 87], [94, 500]]`, accuracy `0.9376`, GOOD
  precision/recall/F1 `0.8518 / 0.8418 / 0.8467`.
- Combined policy: CM `[[2263, 43], [98, 496]]`, accuracy `0.9514`, GOOD
  precision/recall/F1 `0.9202 / 0.8350 / 0.8756`.
- RF: CM `[[2268, 38], [126, 468]]`, accuracy `0.9434`, GOOD
  precision/recall/F1 `0.9249 / 0.7879 / 0.8509`.

Direct comparison to the previous no-continuum 15-shot baseline evaluated
against the old labels:

| model | old F1 | v2 F1 | old FP/FN | v2 FP/FN |
| --- | ---: | ---: | ---: | ---: |
| CNN | 0.8386 | 0.8467 | 112 / 94 | 87 / 94 |
| Combined policy | 0.8627 | 0.8756 | 46 / 117 | 43 / 98 |
| RF | 0.8418 | 0.8509 | 41 / 142 | 38 / 126 |

Because the truth labels changed for the seven non-G shots, the cleaner
comparison also rescored the old saved predictions against the v2 label list
for the 2900 retained rows. In that view, the label cleanup itself accounts
for most of the combined-policy improvement:

| model | old predictions on old labels F1 | old predictions on v2 labels F1 | v2 retrain on v2 labels F1 |
| --- | ---: | ---: | ---: |
| CNN | 0.8386 | 0.8352 | 0.8467 |
| Combined policy | 0.8627 | 0.8758 | 0.8756 |
| RF | 0.8418 | 0.8564 | 0.8509 |

Interpretation:

- The combined policy is essentially unchanged after retraining once both
  old and new runs are judged against the v2 labels: F1 `0.8758 -> 0.8756`.
  The retrained policy is more conservative, reducing FP `53 -> 43` but
  increasing FN `90 -> 98`.
- CNN benefits most from retraining on v2 labels: judged against v2 truth, F1
  improves `0.8352 -> 0.8467`, mostly by reducing false positives
  `127 -> 87` at the cost of more false negatives `77 -> 94`.
- RF becomes slightly more conservative against v2 truth: F1
  `0.8564 -> 0.8509`, with FP `47 -> 38` but FN `114 -> 126`.
- Non-G combined-policy performance is effectively flat under v2 truth:
  F1 `0.9113 -> 0.9108`. The apparent improvement relative to the old
  baseline (`0.8951 -> 0.9108`) is mostly due to the corrected labels.
- G-shot labels were copied unchanged. G-shot performance is stable:
  combined-policy F1 `0.6351 -> 0.6395`, CNN F1 `0.6243 -> 0.6237`, RF F1
  `0.5857 -> 0.5985`. This means the non-G relabeling did not materially
  disrupt the G-shot regime.

Per-shot combined-policy comparison, old baseline labels to v2 run labels:

| shot | old F1 | v2 F1 | old FP/FN | v2 FP/FN |
| --- | ---: | ---: | ---: | ---: |
| `nstx_120113` | 0.9677 | 0.9670 | 2 / 1 | 2 / 1 |
| `nstx_135388` | 0.9225 | 0.9237 | 17 / 3 | 10 / 10 |
| `nstx_141711` | 0.8521 | 0.9067 | 1 / 24 | 1 / 13 |
| `nstxuE202855A01t020` | 0.9348 | 0.8913 | 1 / 5 | 1 / 9 |
| `nstxuE204669M03t025` | 0.8609 | 0.8571 | 1 / 20 | 2 / 19 |
| `nstxuE205052A01t022` | 0.8382 | 0.9322 | 5 / 17 | 6 / 2 |
| `nstxu_204202` | 0.9143 | 0.9048 | 1 / 11 | 4 / 8 |

Main practical conclusion: the v2 labels make the aggregate metrics
look cleaner and do not hurt G-shot transfer, but the combined policy already
absorbed most of the label correction without needing materially different
models. The v2 list is promoted as the current canonical/default training list
for now, while awaiting review of all G shots. Before replacing production
checkpoints, inspect whether the new CNN/RF conservatism is desirable for the
downstream NOVA-C workflow, especially on `nstxuE202855A01t020`,
`nstxuE204669M03t025`, and `nstxu_204202`.

After this LOSO check, the NERSC and Flux path configs were updated so both
`NOVA_TRAIN_CSV` and `NOVA_TRAIN_CSV_TAE` point to
`training_labels/tae_like_v2_nonG.csv`:

- `configs/paths/nova_paths.nersc.sh`
- `configs/paths/nova_paths.flux.sh`
- `configs/paths/nova_paths.flux.csh`

The code fallbacks for fresh runs were also updated so accidental unsourced
runs use the v2 list: `scripts/cnn_raw.py` now falls back to
`training_labels/tae_like_v2_nonG.csv`, and `scripts/run_loso_10.py` defaults
to `$NOVA_TRAIN_CSV` or the v2 list when the environment variable is absent.

The active RF and raw-CNN checkpoints were then retrained on
`training_labels/tae_like_v2_nonG.csv` and replaced at the top level under
`models/`. The previous top-level checkpoints remain archived under
`models/pre_v2_nonG_20260810/`.

Active v2 model metadata:

- RF: `models/nova_mode_classifier.joblib` plus
  `models/nova_mode_classifier_bundle.joblib`; 2900 training rows, 594 GOOD /
  2306 BAD, production `rf_w_star_max_22_v2` schema with 22 features.
- Raw CNN: `models/nova_cnn_raw.pt`; split check best accuracy `0.9534` at
  epoch 13, then full-list refit on all 2900 rows for 80 epochs with
  `M_target=100`, `R_target=201`, robust normalization, OneCycleLR peak LR
  `0.02`, gradient clipping `1.0`, no continuum branch, and no prediction
  collapse in final prediction-health metadata.

Active model SHA-256 checksums:

```text
2a96699bba6bb92d44c9f5b09373e35c3011c8d5bdeab297519e7cd69f5e6023  models/nova_mode_classifier.joblib
0cacd4f1b31347050c1192c109058c3a89c84e3f186ea0899e2a89f95a068629  models/nova_mode_classifier_bundle.joblib
29643ff060aa77e4d7624063803f199918c5a81a2a7e997f732b2481a6c0af49  models/nova_cnn_raw.pt
```

### 2026-08-15: deterministic TAE rule-sorting scaffold and skill split

Renamed the active repository skill from
`.agents/skills/label-tae-like-modes/` to
`.agents/skills/visual-tae-rule-development/`. Historical entries above keep
the old name because it was correct when those reviews were performed. The
renamed skill preserves the blind manifest, rendering, sealing, comparison,
and qualitative labeling policy while describing its current role in visual
rule development and explicitly marked post-seal adjudication.

Created `.agents/skills/sort-tae-like-modes/` for the separate deterministic
production-style workflow. The implementation adds:

- `scripts/make_tae_like_list.py`: shot-level preflight and reusable
  `preprocess_shot()`, preserving the canonical TAE/EAE/mixed split and
  compatible `tae_like_all.csv`, `eae_like.csv`, and `rejected_modes.csv`
  outputs;
- `scripts/tae_rule_engine.py`: a pure, versioned per-mode result interface;
- `scripts/sort_shot_rules.py`: noninteractive orchestration, reusable manual
  overrides, final-GOOD-only duplicate processing, and deterministic outputs;
- `scripts/tae_rule_io.py`: shared schemas, stable JSON/CSV writing, portable
  keys, and SHA-256 fingerprints over each mode plus its `datcon#` contents;
- backward-compatible `scripts/label_modes_fast.py --adjudication` support for
  structured GOOD/BAD/REVIEW overrides with mandatory `--no-rf`, reviewer,
  and nonempty manual reason.

The new preprocessor matches `sort_shot_mixed.py` behavior by aborting the
entire shot before mode processing when any populated requested `N#` lacks its
required `datcon#`. Other unusable inputs become per-mode `INVALID` rows. Valid
mixed modes remain on the TAE side, and valid EAE-like modes are routed without
a fabricated rule decision.

The current rule set is deliberately `tae-rules-placeholder-v1`: every valid
TAE-side mode is `REVIEW` with primary reason `RULESET_NOT_IMPLEMENTED`.
Quantitative BAD rules and positive GOOD templates remain future work. The
workflow does not use RF or CNN for classification. An optional RF checkpoint
is consulted only after final decisions to rank representatives within
final-GOOD close-frequency clusters. Missing/unloadable RF retains every
affected member with `SKIPPED_NO_RF_CHECKPOINT`; any per-cluster scoring failure
retains that whole cluster with `SKIPPED_RF_SCORING_FAILED`. CNN is never loaded
or run.

Manual overrides apply only when their stored input fingerprint matches the
current mode and continuum files. Stale, ambiguous, ineligible, and unmatched
overrides are not silently applied. The final summary records the SHA-256 of
the exact override file. Output rows and JSON use stable ordering, summary
reason counts use one primary reason per mode, and deterministic regeneration
does not add timestamps.

Synthetic validation in `tests/test_rule_sorting.py` covers missing-datcon
preflight, malformed modes, split parity, fingerprint changes, placeholder
REVIEW behavior, manual precedence and validation, stale/ambiguous overrides,
RF ranking and retain-all fallbacks, output headers/counts, JSON nulls, skill
structure/validation, and byte-identical reruns. No real-shot smoke test was
run because no target shot path was supplied for this task.

### 2026-08-15: populate deterministic rule features with the shared RF31 calculations

The placeholder rule engine now calculates and records a named 31-measurement
`rule_features` object for every valid TAE-like or mixed mode. The feature set
is the production RF 22 plus all six boundary-crossing extensions and all
three inner-extremum extensions. The rule-facing schema is
`tae-rule-features-rf31-v1` and records the source calculation schema as
`rf_all_crossings_extremum_energy_31_v2`. The family-routing measurements
`signed_delta` and `fraction_below_upper2` remain top-level audit columns and
are not duplicated as rule evidence.

`src/mode_features.py` now exposes the same calculations as a named mapping and
accepts already-loaded continuum arrays. The existing ordered RF vector calls
the same implementation and retains its previous defaults. Preprocessing
retains validated mode and continuum arrays only for the TAE side, and
`scripts/tae_rule_engine.py` consumes those arrays without filesystem or model
access. Feature extraction does not load an RF checkpoint or change the
placeholder `REVIEW` decision. Extraction failures become `INVALID` with
`RULE_FEATURE_EXTRACTION_FAILED`, with unavailable JSON measurements written
as `null`. Unlike the RF vector, the auditable rule payload does not present
the no-extremum fallback tuple `(1, 1, 0)` as measured geometry: it records
`extremum_match_found=false` and sets the three extremum measurements to null.

Focused tests verify the exact 31 names, numeric parity with the shared RF31
vector, exclusion of routing scalars, top-level radial-feature consistency,
explicit no-extremum output, null failure output, and deterministic workflow
regeneration. The full suite passes 52 tests.

### 2026-08-15: group rule features and add crossing-record audit detail

Replaced the flat rule-feature payload with
`tae-rule-features-grouped-v2`. The grouped object preserves the exact source
schema `rf_all_crossings_extremum_energy_31_v2` while separating:

- the production 22 in `rf_standard_features`;
- the six derived crossing summaries in `crossing_features`;
- every underlying lower/upper crossing in `crossing_records`;
- match status and the three type-3-relevant measurements in
  `extremum_features`;
- empty `resolution_features`, `numerical_structure_features`, and
  `boundary_features` objects reserved for future parameters.

Crossing records contain `boundary`, `r_cross`, `W_peak`, and
`shear_weighted`, with deterministic low/high boundary and radial ordering.
The tests check that record counts and sums agree with the six crossing
summaries and that flattening the three populated feature groups reproduces
the shared RF31 vector when all values are defined. With no crossing, the raw
list is empty and undefined representative radii are null. No-extremum and
feature-extraction failure semantics remain explicit. Manual input
fingerprints are unaffected because they depend only on the mode and
`datcon#` contents.

### 2026-08-15: add the first ordered BAD gate for narrow axis spikes

Implemented the first deterministic rejection gate in
`scripts/tae_rule_engine.py` and versioned the partial ruleset as
`tae-rules-axis-artifact-v1`. Every valid TAE-like or mixed mode now records an
`axis_artifact` object under `boundary_features`. The rule-facing feature schema
is `tae-rule-features-grouped-v3`; the shared RF31 source schema remains
`rf_all_crossings_extremum_energy_31_v2`.

The extractor searches all stored harmonics in normalized `r < 0.03` and
records the largest absolute normalized amplitude, its zero-based stored
harmonic index and radius, whether it is a genuine local maximum, its connected
half-maximum width in normalized radius and radial-grid intervals, the outer
edge of that component, and whether it includes `r=0`. Physical poloidal `m` is
not inferred from the array index because the run-dependent starting offset is
not yet established. Local-maximum detection and half-maximum edges use the
selected harmonic's entire radial profile, preventing broad components or
rising flanks outside the axis window from being misreported as narrow.

The configurable values are `r_ax`, `axis_amplitude_min`, and
`axis_width_max_grid`, exposed by `sort_shot_rules.py` as `--axis_r_ax`,
`--axis_amplitude_min`, and `--axis_width_max_grid`. Both thresholds default to
null, so features are calculated but the gate remains disabled until labeled
modes are used for calibration. Once both are set, a true local maximum meeting
the amplitude and maximum-width criteria returns `BAD` with primary reason
`BAD_AXIS_SPIKE` and stops later decision gates. All sufficiently narrow local
maxima centered inside `r < 0.03` are treated as boundary artifacts without a
continuum-extremum/type-3 exception. Modes not rejected by this partial ruleset
remain `REVIEW` with `NO_GOOD_TEMPLATE`, not `GOOD`.

Shot and per-`n` summaries record the gate enable flag and exact values. Focused
tests cover enabled and disabled behavior, full-grid width measurement,
boundary-touching components, rising flanks, stable structured output, and the
configured end-to-end shot path. The sorter skill, visual-rule skill, labeling
policy, root README, script inventory, diagnostic axis label, and loader
docstring now document the normalized-coordinate and stored-harmonic-index
conventions.

Current blocker/next step: `axis_amplitude_min` and `axis_width_max_grid` still
need calibration from labeled modes, so production runs remain feature-only for
this gate by default. After calibration, add those values explicitly at run
time, validate the resulting BAD set visually, and then implement the next
ordered rejection gate and eventual positive GOOD templates. The complete
repository test suite passes 57 tests, and both repository skills pass their
structural validation.

### 2026-08-16: first labeled subset run of the axis-spike gate

Created a local `data/nstxu_204202/` test subset from modes present in
`training_labels/tae_like_v2_nonG.csv`: 141 mode files across `N1` through
`N10`, their ten matching `datcon#` files, and a verified 141-row
`mode_labels.csv`. `N1` and `N2` contain all available eligible modes (14 and 7,
respectively); each other `N#` contains the first 15 eligible filenames in
stable filename order. The subset has 44 labeled GOOD and 97 labeled BAD modes.

Ran `scripts/sort_shot_rules.py` with `axis_amplitude_min=0.1` and
`axis_width_max_grid=10`, writing deterministic outputs to
`outputs/nstxu_204202_axis_a010_w10/`. All 141 modes were valid TAE-side inputs
(138 TAE-like and 3 mixed). The gate returned `BAD_AXIS_SPIKE` for 66 modes and
left 75 as `REVIEW`. Relative to the copied labels, the BAD set contains 62
labeled BAD and 4 labeled GOOD modes; the REVIEW set contains 35 labeled BAD
and 40 labeled GOOD modes. The four labeled-GOOD rejections are:

- `N7/egn07w.1060E+02`;
- `N8/egn08w.2651E+02`;
- `N9/egn09w.1804E+02`;
- `N10/egn10w.1026E+02`.

Next step: inspect these four modes first to determine whether they expose
label errors, axis-feature-definition issues, or a threshold that needs
refinement before adopting the gate configuration.

Repeated the subset run with `axis_amplitude_min=0.2` and
`axis_width_max_grid=10`, writing to
`outputs/nstxu_204202_axis_a020_w10/`. The higher amplitude floor returns 61
`BAD_AXIS_SPIKE` and 80 `REVIEW` decisions. The BAD set contains 60 labeled BAD
and one labeled GOOD mode; the REVIEW set contains 37 labeled BAD and 43
labeled GOOD modes. Compared with the `0.1` run, three labeled-GOOD low-amplitude
wiggles and two labeled-BAD modes move from BAD to REVIEW. The only remaining
labeled-GOOD rejection is `N10/egn10w.1026E+02`, with `axis_peak=0.284065`,
`axis_peak_r=0.015`, and `axis_halfmax_width_grid=1.06871`. Inspect that mode to
decide whether its training label or the gate policy should change.

### 2026-08-16: adopt inclusive axis boundary and calibrated gate defaults

Updated the first axis-artifact rule after explicit non-blind calibration on
the 141-mode `nstxu_204202` subset. The axis search window is now inclusive,
`r <= r_ax`, with `r_ax=0.03`. This admits a local maximum on the configured
boundary while retaining the full-profile local-maximum test and full-grid
FWHM calculation. The active shared defaults are now
`axis_amplitude_min=0.2` and `axis_width_max_grid=10`; the CLI uses the same
constants. `--disable_axis_artifact` remains available for feature-only runs.

Versioned the behavior change as `tae-rules-axis-artifact-v2` and the
rule-feature payload as `tae-rule-features-grouped-v4`. Added a regression test
for a peak centered exactly at `r=0.03`, updated default/disabled-gate tests,
and synchronized the sorter skill, visual labeling policy, root README, and
script inventory.

A no-override smoke run in `outputs/nstxu_204202_axis_defaults_v2/` processed
all 141 modes without invalid inputs and returned 62 BAD / 79 REVIEW. Relative
to the copied labels, BAD contains 61 labeled BAD and one labeled GOOD; REVIEW
contains 36 labeled BAD and 43 labeled GOOD. The previously missed
`N1/egn01w.1139E+02` is now `BAD_AXIS_SPIKE` with `axis_peak=1` at `r=0.03` and
FWHM `2.09123` grid intervals. The complete repository suite passes 58 tests,
and both repository skills validate. Next, inspect the remaining REVIEW modes
while developing later ordered gates; the lone labeled-GOOD axis rejection
`N10/egn10w.1026E+02` remains a calibration/adjudication case.

Generated `outputs/*_axis_*/` calibration directories are now ignored by Git.
The previously indexed `nstxu_204202` calibration runs were removed from the
index without deleting their local files. Their parameters and scientifically
relevant counts remain recorded above; keep the local outputs during rule
development and regenerate a final audit run after the ordered ruleset is
complete.

### 2026-08-16: add the second ordered BAD gate for unresolved grid-scale spikes

Implemented `BAD_GRID_SCALE_SPIKE` after explicit non-blind calibration on the
141-mode `data/nstxu_204202/` subset. The gate searches every stored harmonic
over the complete normalized radial grid for positive local maxima and negative
local minima. Each candidate width is the linearly interpolated connected FWHM
of the sign-adjusted lobe; it is deliberately not calculated from `abs(mode)`,
because adjacent `+A/-A` samples would otherwise merge into a falsely broad
component. One-sided extrema at either radial boundary are included.

The active configurable defaults are:

```yaml
grid_scale_spike:
  amplitude_min: 0.3
  width_max_grid: 1
```

Among candidates meeting the width limit, the audit feature records the
largest absolute peak, its signed amplitude and sign, zero-based stored
harmonic index, radius, interpolated half-maximum edges, width in normalized
radius and grid intervals, and boundary-touch status. A peak meeting the
amplitude threshold returns `BAD_GRID_SCALE_SPIKE` and stops later gates. The
axis gate retains precedence. The rule feature now lives under
`numerical_structure_features.grid_scale_spike`; the rule-feature schema is
`tae-rule-features-grouped-v5` and the partial ruleset is
`tae-rules-axis-grid-spike-v3`.

The one-grid cutoff catches all four calibration examples discussed during
development: `N8/egn08w.4253E+02`, `N5/egn05w.3044E+02`,
`N4/egn04w.1333E+02`, and `N8/egn08w.1566E+02`. On the unchanged copied label
CSV, the gate alone matches 52 labeled BAD and two labeled GOOD modes. After
the existing axis gate, it adds seven labeled BAD and two labeled GOOD modes.
A two-grid cutoff was rejected because it would match 36 labeled GOOD modes
overall and 35 after the axis gate. Visual non-blind inspection found one of
the one-grid GOOD-labeled cases, `N9/egn09w.3222E+02`, to be a genuine bad
grid-scale spike; the copied and canonical training-label files were not
changed as part of this gate implementation.

The CLI exposes `--grid_scale_amplitude_min`,
`--grid_scale_width_max_grid`, and `--disable_grid_scale_spike`. Shot and
per-`n` summaries record the enable flag and exact thresholds. Focused tests
cover signed rather than absolute width, both amplitude and width conditions,
endpoint handling, gate precedence, summary reporting, and deterministic
workflow output. A later alternating-sign packet gate remains available for
wider or lower-amplitude numerical oscillations that this strict one-grid gate
does not reject.

A default no-override smoke run in
`outputs/nstxu_204202_axis_grid_defaults_v3/` processed all 141 modes without
invalid inputs and returned 71 BAD / 70 REVIEW, with primary reasons
`BAD_AXIS_SPIKE=62`, `BAD_GRID_SCALE_SPIKE=9`, and `NO_GOOD_TEMPLATE=70`.
Relative to the unchanged copied labels, BAD contains 68 labeled BAD and three
labeled GOOD modes; one of those three is the newly adjudicated bad
`N9/egn09w.3222E+02`. Its recorded grid candidate is the negative lobe at
`r=0.975` in stored harmonic index 52, with absolute amplitude `0.578006` and
FWHM `0.892769` grid intervals. The complete repository suite passes 64 tests.

### 2026-08-16: correct the N9/3222 training label and refresh REVIEW modes

After explicit non-blind visual adjudication, changed
`nstxu_204202/N9/egn09w.3222E+02` from `good` to `bad`. The decisive evidence
is the signed-harmonic grid-scale spike already recorded by the second gate:
absolute amplitude `0.578006`, FWHM `0.892769` radial-grid intervals, centered
at `r=0.975` in zero-based stored harmonic index 52.

Updated the active canonical `training_labels/tae_like_v2_nonG.csv`, its
cleaned human-review source `tests/labels_audit/labels_human_review_clean.csv`,
the per-shot human-vs-training change table, and the local 141-mode calibration
copy. The active list remains 2900 rows and now contains 593 GOOD / 2307 BAD.
The previous canonical/derived training lists, archived snapshots, raw review
files, and sealed Codex review remain unchanged for historical provenance. The
active RF and raw-CNN checkpoints were trained before this correction on the
594-GOOD / 2306-BAD snapshot and have not been refit.

Generated `data/nstxu_204202/review_mode_labels.csv` from the current
`tae-rules-axis-grid-spike-v3` REVIEW output. It contains 70 unique portable
mode paths with their current labels: 41 GOOD and 29 BAD. The complete sorter
audit remains in
`outputs/nstxu_204202_axis_grid_defaults_v3/review_tae_like.csv`.

### 2026-08-16: add the third ordered BAD gate for continuum crossings

Implemented `BAD_CONT_CROSS` after explicit non-blind calibration on the 70
modes that survived the axis-artifact and grid-scale-spike gates. The gate
reuses the shared deterministic continuum calculations; it does not load an RF
checkpoint or use an RF prediction. `n_cross` counts true lower/upper boundary
crossings, while `W_star_max` is the largest crossing value of
`sum_h |mode_h(r)|^2`, normalized by its maximum over radius.

The active provisional configuration is:

```yaml
continuum_crossing:
  w_cross_threshold: 0.05
```

The third ordered gate is applied only after `BAD_AXIS_SPIKE` and
`BAD_GRID_SCALE_SPIKE`:

```text
IF n_cross > 0
AND W_star_max > w_cross_threshold
THEN BAD_CONT_CROSS
AND stop evaluating later decision gates
```

The comparison is strictly greater than the threshold. In the 70-mode
calibration set, 28 of the 29 labeled BAD modes had a crossing and their
minimum `W_star_max` was `0.0573674`; the maximum among the 41 labeled GOOD
modes was `0.00573841`. Thus the `0.05` default rejected all 28 crossing BAD
modes and no labeled GOOD modes. The remaining labeled BAD mode,
`N5/egn05w.1894E+02`, has `n_cross=0` and correctly remains outside this gate.

The CLI exposes `--w_cross_threshold` and `--disable_cont_cross`. Shot and
per-`n` summaries record the gate enable flag and threshold. The grouped rule
features already contained all required crossing measurements, so the feature
schema remains `tae-rule-features-grouped-v5`; the ordered ruleset is now
`tae-rules-axis-grid-cont-cross-v4`.

A default no-override smoke run in
`outputs/nstxu_204202_axis_grid_cont_defaults_v4/` processed all 141 modes with
no invalid inputs and returned 99 BAD / 42 REVIEW. Primary reasons are
`BAD_AXIS_SPIKE=62`, `BAD_GRID_SCALE_SPIKE=9`, `BAD_CONT_CROSS=28`, and
`NO_GOOD_TEMPLATE=42`. Relative to the current copied labels, BAD contains 97
labeled BAD and two labeled GOOD modes; REVIEW contains 41 labeled GOOD modes
and only `N5/egn05w.1894E+02` as labeled BAD. Refreshed
`data/nstxu_204202/review_mode_labels.csv` to these 42 current REVIEW modes.

Focused coverage verifies crossing presence, strict threshold behavior, gate
precedence, configured shot-level execution, and summary fields. The complete
repository suite passes 67 tests, and both repository skills pass structural
validation.

### 2026-08-16: add the fourth ordered BAD gate for narrow edge-energy spikes

Implemented `BAD_EDGE_SPIKE` as the fourth short-circuit rejection gate, after
`BAD_CONT_CROSS`. The decision uses the global radial-energy envelope
`W(r)=sum_h |mode_h(r)|^2`, normalized by its maximum over radius. It does not
use a literal mirrored axis rule on any individual harmonic: calibration on the
42 survivors from the first three gates showed that such a rule would reject
several labeled-GOOD physical edge modes whose narrow harmonics are consistent
with magnetic shear. The strongest individual edge harmonic and its full-grid
half-maximum geometry are still recorded separately for audit.

The active provisional configuration is:

```yaml
edge_artifact:
  r_edge_min: 0.97
  edge_width_max_grid: 10
```

The radius comparison is inclusive, and both half-maximum edges are searched
on the complete radial grid:

```text
IF edge_energy_peak_r >= r_edge_min
AND edge_energy_halfmax_width_grid <= edge_width_max_grid
THEN BAD_EDGE_SPIKE
AND stop evaluating later decision gates
```

The CLI exposes `--edge_r_min`, `--edge_width_max_grid`, and
`--disable_edge_artifact`. Shot and per-`n` summaries record enable state and
exact settings. Adding `boundary_features.edge_artifact` advances the grouped
schema to `tae-rule-features-grouped-v6`; the ordered ruleset is now
`tae-rules-axis-grid-cont-edge-v5`.

The default 141-mode calibration run is in
`outputs/nstxu_204202_axis_grid_cont_edge_defaults_v5/`. It processed all modes
without invalid inputs and returned 100 BAD / 41 REVIEW, with primary reasons
`BAD_AXIS_SPIKE=62`, `BAD_GRID_SCALE_SPIKE=9`, `BAD_CONT_CROSS=28`,
`BAD_EDGE_SPIKE=1`, and `NO_GOOD_TEMPLATE=41`. The fourth gate rejected only
the remaining labeled BAD mode, `N5/egn05w.1894E+02`; its global energy peak is
at `r=0.975` with FWHM `5.244843` grid intervals. Its strongest audited edge
harmonic is stored index 34 at `r=0.98`, with FWHM `1.547562` grid intervals.
All 41 remaining REVIEW modes are labeled GOOD. The two labeled-GOOD modes
already rejected by earlier gates remain the only label/rule disagreements in
this calibration subset.

The complete 725-mode `nstxu_204202` shot was then processed from `$NOVA_DATA`
into `outputs/nstxu_204202_full_axis_grid_cont_edge_v5/`. Routing found 261
TAE-like, 14 mixed, 450 EAE-like, and no invalid inputs. Among the 275 TAE-side
modes, the result is 210 BAD / 65 REVIEW, with primary reasons
`BAD_AXIS_SPIKE=112`, `BAD_GRID_SCALE_SPIKE=18`, `BAD_CONT_CROSS=79`,
`BAD_EDGE_SPIKE=1`, and `NO_GOOD_TEMPLATE=65`. The edge gate again fired only
for `N5/egn05w.1894E+02`.

Joining these 275 TAE-side results to the current
`training_labels/tae_like_v2_nonG.csv` gives complete one-to-one label coverage:
211 labeled BAD and 64 labeled GOOD. Treating non-BAD/REVIEW as the retained
positive class, there are two false negatives (labeled GOOD rejected as BAD)
and three false positives (labeled BAD retained as REVIEW). Equivalently, with
BAD as the positive class, those names reverse. The five rows and their primary
gate evidence are recorded in
`outputs/nstxu_204202_full_axis_grid_cont_edge_v5/label_disagreements.csv`.

Focused tests cover the inclusive edge threshold, complete-grid energy FWHM,
disabled-gate audit behavior, continuum-gate precedence, workflow summary
fields, and the guardrail that a narrow individual edge harmonic does not fire
when the total energy peaks in the interior. The complete repository suite now
passes 72 tests, both repository skills pass structural validation, and
`git diff --check` is clean. A useful next cross-shot calibration is to run the
same fixed four-gate configuration on the labeled G shots and compare per-gate
firing rates and label disagreements without retuning on each shot.

### 2026-08-16: lower the crossing threshold to 0.03 and run nstx_120113

Changed the active strict continuum-crossing condition from
`W_star_max > 0.05` to `W_star_max > 0.03`. On the complete labeled
`nstxu_204202` TAE-side set, this additionally rejects the labeled-BAD
`N7/egn07w.3718E+02` (`W_star_max=0.0314811`) and
`N8/egn08w.4276E+02` (`W_star_max=0.0360285`) without rejecting an additional
labeled-GOOD mode. The closest retained labeled-GOOD crossing mode there is
`N6/egn06w.9143E+01` at `W_star_max=0.0279922`, so the cross-shot margin remains
narrow and requires continued validation.

The default is now:

```yaml
continuum_crossing:
  w_cross_threshold: 0.03
```

The comparison remains strictly greater than the threshold. Because the
default decision behavior changed, the ordered ruleset advances to
`tae-rules-axis-grid-cont-edge-v6`; the grouped feature schema remains
`tae-rule-features-grouped-v6`.

Ran the complete 541-mode `nstx_120113` shot from `$NOVA_DATA` into
`outputs/nstx_120113_full_axis_grid_cont_edge_w003_v6/`. Routing found 133
TAE-like, 41 mixed, 367 EAE-like, and no invalid inputs. Among the 174 TAE-side
modes, the rules returned 128 BAD / 46 REVIEW with primary reasons
`BAD_AXIS_SPIKE=59`, `BAD_GRID_SCALE_SPIKE=6`, `BAD_CONT_CROSS=61`,
`BAD_EDGE_SPIKE=2`, and `NO_GOOD_TEMPLATE=46`.

This shot contains no mode reaching the continuum gate with `W_star_max` in
the interval `(0.03, 0.05]`, so lowering the threshold changes no
`nstx_120113` decision relative to 0.05. The active training list provides 173
matching GOOD/BAD labels: 128 BAD and 45 GOOD. Of these, 126 BAD are rejected,
44 GOOD remain REVIEW, one GOOD is rejected, and two BAD remain REVIEW. The
one additional TAE-side mode is the intentionally excluded `skip` mode
`N6/egn06w.1418E+02`; the edge gate rejects it from a global energy peak at
`r=0.98` with FWHM `2.42857` grid intervals.

The three active-label disagreements and their gate measurements are in
`outputs/nstx_120113_full_axis_grid_cont_edge_w003_v6/label_disagreements.csv`:

- labeled GOOD but rejected: `N6/egn06w.1472E+02` by
  `BAD_GRID_SCALE_SPIKE`;
- labeled BAD but retained: `N7/egn07w.2630E+02` and
  `N8/egn08w.2765E+02`, both with crossing energy below 0.007.

### 2026-08-16: run the fixed four-gate rules on nstxuG121123B12

Ran the unchanged `tae-rules-axis-grid-cont-edge-v6` configuration, including
the strict `W_star_max > 0.03` crossing threshold, on the complete 637-mode
`nstxuG121123B12` shot from `$NOVA_DATA`. Outputs are in
`outputs/nstxuG121123B12_full_axis_grid_cont_edge_w003_v6/`.

Routing found 129 TAE-like, 7 mixed, 501 EAE-like, and no invalid inputs. Among
the 136 TAE-side modes, the rules returned 112 BAD / 24 REVIEW with primary
reasons `BAD_AXIS_SPIKE=18`, `BAD_GRID_SCALE_SPIKE=51`,
`BAD_CONT_CROSS=43`, `BAD_EDGE_SPIKE=0`, and `NO_GOOD_TEMPLATE=24`.

The active training list supplies 135 matching labels: 116 BAD and 19 GOOD.
All 19 labeled GOOD modes remain REVIEW, while 111 of 116 labeled BAD modes are
rejected. The five disagreements are therefore all labeled-BAD modes retained
as REVIEW; there are no labeled-GOOD rejections. Their measurements are in
`outputs/nstxuG121123B12_full_axis_grid_cont_edge_w003_v6/label_disagreements.csv`.
The one additional TAE-side mode, `N7/egn07w.1888E+02`, is not in the active
training list and is rejected by `BAD_CONT_CROSS`.

Relative to the former 0.05 crossing threshold, the 0.03 default rejects three
additional labeled-BAD modes and the one unlabeled mode, with no labeled-GOOD
loss:

- `N8/egn08w.3027E+02`, BAD, `W_star_max=0.0438244`;
- `N9/egn09w.1294E+02`, BAD, `W_star_max=0.0435955`;
- `N9/egn09w.2852E+02`, BAD, `W_star_max=0.0498032`;
- `N7/egn07w.1888E+02`, unlabeled, `W_star_max=0.0369659`.

These threshold-specific changes are recorded in
`outputs/nstxuG121123B12_full_axis_grid_cont_edge_w003_v6/crossing_threshold_003_changes.csv`.

### 2026-08-17: run the fixed four-gate rules on nstxuG121123J38

Ran the unchanged `tae-rules-axis-grid-cont-edge-v6` configuration on the
complete 620-mode `nstxuG121123J38` shot from `$NOVA_DATA`. Outputs are in
`outputs/nstxuG121123J38_full_axis_grid_cont_edge_w003_v6/`.

Routing found 165 TAE-like, 9 mixed, 446 EAE-like, and no invalid inputs. Among
the 174 TAE-side modes, the rules returned 145 BAD / 29 REVIEW with primary
reasons `BAD_AXIS_SPIKE=36`, `BAD_GRID_SCALE_SPIKE=28`,
`BAD_CONT_CROSS=81`, `BAD_EDGE_SPIKE=0`, and `NO_GOOD_TEMPLATE=29`.

The active training list has exact one-to-one coverage of these 174 modes: 167
BAD and 7 GOOD. All 7 labeled GOOD modes remain REVIEW. The rules reject 145 of
167 labeled BAD modes and retain 22 labeled BAD modes as REVIEW; there are no
labeled-GOOD rejections. The 22 disagreements are recorded in
`outputs/nstxuG121123J38_full_axis_grid_cont_edge_w003_v6/label_disagreements.csv`.

Relative to the former 0.05 crossing threshold, the 0.03 default rejects nine
additional labeled-BAD modes with no labeled-GOOD loss. Their `W_star_max`
values span `0.0304733` to `0.0475866`; the exact rows are in
`outputs/nstxuG121123J38_full_axis_grid_cont_edge_w003_v6/crossing_threshold_003_changes.csv`.

### 2026-08-17: add the crossing-neighborhood BAD gate

Implemented `BAD_CONT_CROSS_WINDOW` as the fourth ordered short-circuit gate,
after exact-point `BAD_CONT_CROSS` and before `BAD_EDGE_SPIKE`. It addresses
continuum resonances that fall between radial samples or have appreciable mode
structure immediately beside the interpolated crossing even when
`W_star_max <= 0.03`.

The active configuration is:

```yaml
continuum_crossing_window:
  half_width_grid: 2
  amplitude_min: 0.25
  w_min: 0.05
```

For every true lower/upper crossing, the extractor includes radial samples
satisfying `abs(r_i - r_cross) <= half_width_grid * delta_r`. Across all such
samples and crossings, it independently selects the largest absolute
individual-harmonic amplitude and the largest total radial energy normalized
by its maximum over radius. The inclusive decision is:

```text
IF n_cross > 0
AND (cross_window_A_max >= amplitude_min OR cross_window_W_max >= w_min)
THEN BAD_CONT_CROSS_WINDOW
AND stop evaluating later decision gates
```

`crossing_features` now records both winners' values, sample radii, associated
crossing boundary/radius, and distance from the crossing in grid intervals,
plus the winning stored harmonic index for amplitude. These measurements are
retained when the decision is disabled. The CLI exposes
`--cross_window_half_width_grid`, `--cross_window_amplitude_min`,
`--cross_window_w_min`, and `--disable_cont_cross_window`; shot and per-`n`
summaries record the enable state and all settings. The ordered ruleset is now
`tae-rules-axis-grid-cont-window-edge-v7`, and the expanded grouped audit schema
is `tae-rule-features-grouped-v7`.

Full fixed-default regressions produced:

- `nstxu_204202`: 212 BAD / 63 REVIEW; the new gate fires zero times. Against
  all 275 active labels, 210/211 BAD modes are rejected and 62/64 GOOD modes
  remain REVIEW.
- `nstx_120113`: 128 BAD / 46 REVIEW; the new gate fires zero times. Against
  173 active labels, 126/128 BAD modes are rejected and 44/45 GOOD modes remain
  REVIEW. The separate `skip` mode remains outside the active label list.
- `nstxuG121123B12`: 112 BAD / 24 REVIEW; the new gate fires zero times.
  Against 135 active labels, 111/116 BAD modes are rejected and all 19 GOOD
  modes remain REVIEW. The separate `skip` mode remains outside the list.
- `nstxuG121123J38`: 159 BAD / 15 REVIEW. The new gate rejects 14 additional
  modes, all 14 labeled BAD, reducing labeled-BAD survivors from 22 to 8 while
  retaining all 7 labeled GOOD modes as REVIEW.

Outputs and updated disagreement audits are in the corresponding
`outputs/*_full_axis_grid_cont_window_edge_w003_v7/` directories. J38's 14 new
rejections and their exact amplitude/energy winners are also recorded in
`cross_window_new_rejections.csv` there. The repository suite passes 77 tests,
both repository skills pass structural validation, and `git diff --check` is
clean.

### 2026-08-17: run v7 on the remaining six labeled G shots

Ran the unchanged `tae-rules-axis-grid-cont-window-edge-v7` defaults on K51,
Q62, S31, H47, W29, and Y93. The six runs discovered 5,580 modes and routed
878 TAE-like, 86 mixed, and 4,616 EAE-like modes with zero invalid inputs.
Among 964 TAE-side modes, the rules returned 911 BAD / 53 REVIEW. Primary
reasons were `BAD_AXIS_SPIKE=250`, `BAD_GRID_SCALE_SPIKE=327`,
`BAD_CONT_CROSS=314`, `BAD_CONT_CROSS_WINDOW=20`, `BAD_EDGE_SPIKE=0`, and
`NO_GOOD_TEMPLATE=53`.

The active training list matches 956 modes: 899 BAD and 57 GOOD. Eight
additional TAE-side modes are unlabeled. Of the matched modes, 879 BAD are
rejected, 20 BAD remain REVIEW, 26 GOOD are rejected, and 31 GOOD remain
REVIEW. Interpreting BAD as rejected and GOOD as retained gives 95.2%
rule/label agreement, 97.8% BAD recall, and 54.4% GOOD retention. Report all
three metrics because the label population is strongly BAD-heavy.

The crossing-window gate fires on 20 labeled modes in these six shots: 17 BAD
and three GOOD. All three GOOD disagreements are in Q62. Q62 is the main
cross-shot caveat: all 12 currently labeled-GOOD modes are rejected, nine by
the exact crossing gate and three by the window gate. The three window cases
are `N9/egn09w.1848E+02`, `N10/egn10w.1850E+02`, and
`N10/egn10w.1895E+02`; the middle case is just above the energy threshold at
`cross_window_W_max=0.05105` with amplitude below the amplitude threshold.

Combining these runs with the earlier B12 and J38 v7 results covers all eight
labeled G shots: 6,837 discovered modes and 1,274 TAE-side modes, with 1,182
BAD / 92 REVIEW and no invalid inputs. The active list matches 1,265 rows: the
rules reject 1,149/1,182 BAD modes and retain 57/83 GOOD modes, for 95.3%
agreement, 97.2% BAD recall, and 68.7% GOOD retention. The window gate rejects
34 labeled modes across all eight shots: 31 BAD and three GOOD. Relative to
disabling that gate, it raises agreement from 93.1% to 95.3% while lowering
GOOD retention from 72.3% to 68.7%.

Presentation-ready tables, gate/label composition, all 46 disagreements from
the six new shots, the eight unlabeled modes, and all 34 G-shot window-gate
rows are in `outputs/g_shots_axis_grid_cont_window_edge_w003_v7/`. Individual
sorter outputs are in each shot's
`outputs/SHOT_full_axis_grid_cont_window_edge_w003_v7/` directory.

### 2026-08-17: complete 14-shot v7 comparison excluding Q62

Ran the unchanged `tae-rules-axis-grid-cont-window-edge-v7` configuration on
the five remaining non-G shots: `nstx_135388`, `nstx_141711`,
`nstxuE202855A01t020`, `nstxuE204669M03t025`, and
`nstxuE205052A01t022`. Combined these with the existing seven G-shot runs and
the earlier `nstx_120113` and `nstxu_204202` runs, while excluding Q62 because
its mode/continuum calculation may use the wrong q profile.

The 14-shot comparison contains 11,349 discovered modes: 2,486 strict
TAE-like, 220 mixed, 8,643 EAE-like, and zero invalid. The 2,706 TAE-side
modes receive 2,123 BAD and 583 REVIEW decisions. The active list matches
2,659 of them: 2,078 labeled BAD and 581 labeled GOOD; 47 current TAE-side
modes are unlabeled or excluded from training.

The rule/label matrix is 2,022 labeled-BAD modes rejected, 56 labeled-BAD
modes retained for REVIEW, 59 labeled-GOOD modes rejected, and 522
labeled-GOOD modes retained. This gives 95.7% agreement, 97.3% BAD recall,
89.8% GOOD retention, and 90.3% GOOD precision among REVIEW modes. The seven
non-G shots have 95.7% agreement, 97.7% BAD recall, and 91.2% GOOD retention;
the seven retained G shots have 95.7%, 96.9%, and 80.3%, respectively.

There are 115 disagreements in total: 59 GOOD-rejected and 56 BAD-retained.
The exact continuum-crossing gate accounts for 30 rejected-GOOD modes, while
the crossing-window gate accounts for one. The crossing-window gate rejects
39 labeled modes overall: 38 BAD and one GOOD. These remain agreement checks,
not independent accuracy estimates: most G-shot labels are not yet audited,
and H47 retains a possible continuum-consistency issue.

The presentation summary, machine-readable per-shot statistics, and complete
disagreement list are in
`outputs/14_shots_axis_grid_cont_window_edge_w003_v7/`.

### 2026-08-20: gate-3-disabled 14-shot ablation

Ran a matched 14-shot ablation excluding Q62 with the exact-point
`BAD_CONT_CROSS` gate disabled. TAE/EAE routing and the axis, grid-scale,
crossing-window, and edge gates were unchanged. All 14 runs completed with
zero invalid inputs; production defaults were not changed.

The baseline returns 2,123 BAD / 583 REVIEW decisions, while the ablation
returns 2,107 BAD / 599 REVIEW. Against 2,659 matched active labels, disabling
gate 3 recovers seven labeled-GOOD modes and newly retains nine labeled-BAD
modes. GOOD retention rises from 89.85% to 91.05%, BAD recall falls from
97.31% to 96.87%, rule/label agreement changes from 95.675% to 95.600%, and
GOOD precision among REVIEW modes falls from 90.31% to 89.06%.

On the seven audited non-G shots, six GOOD modes are recovered and six BAD
modes become REVIEW. Agreement is unchanged at 95.657%, GOOD retention rises
from 91.18% to 92.35%, and BAD recall falls from 97.69% to 97.16%. On the
seven retained G shots, one GOOD is recovered and three BAD modes become
REVIEW; those labels remain less reliable because most G shots are unaudited.

Gate 4 absorbs 809 of the 825 baseline gate-3 rejections: 753 labeled BAD, 23
labeled GOOD, and 33 unlabeled. Only 16 become REVIEW: nine labeled BAD and
seven labeled GOOD. All 16 have `W_star_max` between 0.03047 and 0.04317 and
remain below both gate-4 thresholds. Therefore, disabling gate 3 makes only a
small GOOD-recall-oriented operating-point shift; 23 former gate-3
labeled-GOOD modes are still rejected by gate 4.

The aggregate comparison, per-shot statistics, 16 changed modes, and former
gate-3 outcome counts are in
`outputs/14_shots_axis_grid_cont_exact_off_window_edge_v7/`.

### 2026-08-20: Flux main shot-database path

Added the shared Flux main-shot database root to both shell path configs:

```text
NOVA_DITW_ROOT=/p/nstxdigtwin/energetic_particles/nova/DiTw
```

`configs/paths/nova_paths.flux.sh` exports the variable for Bash, while
`configs/paths/nova_paths.flux.csh` sets the equivalent tcsh environment
variable. Both `nova_env` helpers now display the configured path.

### 2026-08-20: Q62 local continuum-profile refresh

For local visual inspection of Q62 modes against the recalculated continuum,
preserved the former training-data `datcon1` through `datcon10` files as
`datcon1_old` through `datcon10_old` under
`/p/hym/ebelova/NOVA/data_mixed/nstxuG121123Q62/N*/`. Installed the
corresponding active `datconN` files from
`$NOVA_DITW_ROOT/nstxuG121123Q62/N*/` and verified all ten active files by
byte comparison and SHA-256. Mode files, `datcon_gf.txt`, labels, and model
checkpoints were not changed.

The user subsequently applied the same Q62 continuum refresh in the
Perlmutter `$NOVA_DATA` copy: the former `datcon1` through `datcon10` files
were retained with the `_old` suffix, and the recalculated DiTw `datconN`
profiles were installed as the active files. This keeps the Flux and
Perlmutter Q62 inspection datasets aligned while preserving the previous
continuum profiles on both systems.

### 2026-08-20: versioned Flux training-vs-DiTw provenance audit

Added the read-only reusable auditor
`scripts/audit_training_provenance.py`. It compares every training-relevant
`egn*`, active `datconN`, optional `datcon_gf.txt`, and preserved
`datconN_old` backup for all shots named by a selected training CSV. File
identity uses SHA-256. For changed same-name mode files, the manifest also
records parsed frequency, damping, mode-array shape, classifier-used mode
structure equality, and maximum absolute mode difference. No input files are
modified.

Generated and retained the first versioned Flux audit at
`audits/training_provenance/2026-08-20_flux_v1/`. The artifact set uses schema
`nova-training-provenance-v1`, covers all 15 shots and 2900 canonical rows in
`training_labels/tae_like_v2_nonG.csv`, contains 12,835 scoped file-pair rows
and 998 non-identical/missing rows, and includes:

- `file_manifest.csv` and its compact non-identical subset `differences.csv`;
- `shot_summary.csv` and the human-readable `report.md`;
- `run_metadata.json`, including the training-list and audit-script hashes;
- `SHA256SUMS` for the complete generated artifact set.

Most important findings preserved by this audit:

- `nstx_141711` has 42 changed canonical same-name modes and 101 canonical
  training modes absent from current DiTw;
- `nstxuG121123K51` has 106 changed canonical same-name modes and 86 canonical
  training modes absent from current DiTw;
- `nstx_120113` and `nstxu_204202` have 120 and 135 canonical training modes,
  respectively, absent from current DiTw;
- active continuum mismatches remain for `nstx_120113` (10), `nstx_135388`
  (9), `nstx_141711` (10), `nstxuG121123K51` (10), `nstxuG133964S31`
  (10), `nstxuG142301H47` (10), and `nstxuG142301Y93` (10);
- Q62 now has 10 active `datconN` files identical to DiTw, while its 10
  preserved `datconN_old` files all differ from the current reference and
  retain the pre-refresh continuum provenance.

The changed canonical mode frequencies remain numerically close: median
absolute relative changes are `5.75e-7` for `nstx_141711` and `5.13e-7` for
K51, with maxima `1.14e-4` and `7.46e-5`. Nevertheless, all 42 and 106 changed
canonical files, respectively, have unequal classifier-used mode structures;
nine `nstx_141711` modes also changed array shape. Treat those shots as true
mode-provenance mismatches rather than harmless timestamp or formatting
changes. Labels and model checkpoints were not changed by the audit.

### 2026-08-23: frozen Flux training-database copy

Created a complete preservation copy of the current Flux training database:

```text
/p/hym/ebelova/NOVA/data_mixed
  -> /p/hym/ebelova/NOVA/data_mixed_2026_08_20
```

The destination was confirmed empty before copying. The archive copy retained
file metadata and hard-link relationships. A post-copy checksum dry run with
deletion reporting produced no differences or extra destination files. Both
trees contain 268 directories, 18,876 regular files, zero symbolic links, and
7,018,731,738 regular-file bytes. This dated directory is the recoverable data
snapshot associated with the 2026-08-20 provenance audit; do not modify it
while constructing a DiTw-aligned candidate training database.

The corresponding repository state is preserved in GitHub by the annotated
tag `training-data-audit-2026-08-20-v1`, which resolves to commit
`ad8c4a917225117cec9779cfb9d74d9afec0cb16`.

The two active training-label lists were also copied to
`training_labels/labels_2026_08_20/`:

- `tae_like_train.csv`
- `tae_like_v2_nonG.csv`

Both dated copies are byte-identical to the current files at the top level of
`training_labels/`. The dated directory also contains the archived legacy
`tae_like_train_7.csv`; that file is no longer present at the active top level.
Together, the data-directory copy, Git tag, and dated label files preserve the
pre-synchronization training state.

### 2026-08-23: initialized the candidate training database

Initialized `/p/hym/ebelova/NOVA/data_mixed_new` with the six training shots
whose audit found no relevant DiTw mismatch:

- `E202855` (`nstxuE202855A01t020`)
- `E204669` (`nstxuE204669M03t025`)
- `E205052` (`nstxuE205052A01t022`)
- `B12` (`nstxuG121123B12`)
- `J38` (`nstxuG121123J38`)
- `W29` (`nstxuG142301W29`)

For each shot, both the canonical shot directory and its
`_tae_eae_split` companion were copied unchanged from
`/p/hym/ebelova/NOVA/data_mixed`, for 12 top-level directories total. The copy
contains 5,052 regular files in 95 directories and 1,874,319,634 regular-file
bytes. Per-directory checksum dry runs with deletion reporting found no
missing, extra, or differing files in `data_mixed_new`.

Added six more canonical shot directories whose mode files were accepted but
whose training-side continuum profiles were outdated:

- `nstx_120113`
- `nstx_135388`
- `nstxuG121123Q62`
- `nstxuG133964S31`
- `nstxuG142301H47`
- `nstxuG142301Y93`

For this group, only the `N*/egn*` mode payload and directory structure were
copied from `/p/hym/ebelova/NOVA/data_mixed`. No `datcon*` files, auxiliary
continuum files, root-level labels or logs, or `_tae_eae_split` directories
were copied. The result contains 6,251 checksum-verified mode files in 75
directories and 2,369,936,968 bytes.

Q62 is an exception to the pending-continuum state because its active
training-side profiles had already been refreshed from DiTw. Copied
`N1/datcon1` through `N10/datcon10` from the refreshed
`data_mixed/nstxuG121123Q62` tree into `data_mixed_new` and verified all ten
files byte-for-byte (156,060 bytes total). The preserved `datconN_old`
backups and every `datcon*txt` auxiliary file remain excluded. Q62 now needs
only regenerated TAE/EAE split outputs; at that stage, the other five shot
trees still awaited recalculated continuum profiles.

Installed the recalculated active continuum profiles for those remaining five
shots directly from their corresponding directories under
`/p/nstxdigtwin/energetic_particles/nova/DiTw`:

- `nstx_120113`
- `nstx_135388`
- `nstxuG133964S31`
- `nstxuG142301H47`
- `nstxuG142301Y93`

For every shot, copied exactly `N1/datcon1` through `N10/datcon10`, for 50
files and 780,300 bytes total. Each destination file is byte-identical to its
DiTw source. No `datcon*old*`, `datcon*txt*`, unnumbered `datcon`, or split
output was copied. These five shots and Q62 now have their accepted mode files
and updated active continua in `data_mixed_new`; all six still require
regenerated TAE/EAE split outputs.

Populated the final three canonical shot trees directly from DiTw, copying
only each populated `N*/egn*` mode set and its matching active `N*/datconN`:

- `nstx_141711`: 535 modes and 10 continuum files;
- `nstxuG121123K51` (K51): 766 modes and 10 continuum files;
- `nstxu_204202`: 533 modes and 9 continuum files (`N2` through `N10`; `N1`
  has no mode payload).

This final group contains 1,834 mode files plus 29 related continuum files,
for 668,664,206 bytes total. Content-checksum and exact-inventory dry runs
against `/p/nstxdigtwin/energetic_particles/nova/DiTw` found no missing,
extra, or differing payload files. No auxiliary files or `_tae_eae_split`
directories were copied. `data_mixed_new` now contains canonical shot trees
for all 15 training shots; these final three also require regenerated TAE/EAE
split outputs.

### 2026-08-23: provisional v3 labels for the rebuilt database

Created `training_labels/tae_like_v3_nonG.csv` for the six shots whose mode,
continuum, and split data were retained unchanged in `data_mixed_new`:

- `nstxuE202855A01t020`: 79 rows (50 `good`, 29 `bad`)
- `nstxuE204669M03t025`: 217 rows (82 `good`, 135 `bad`)
- `nstxuE205052A01t022`: 291 rows (57 `good`, 234 `bad`)
- `nstxuG121123B12`: 135 rows (19 `good`, 116 `bad`)
- `nstxuG121123J38`: 174 rows (7 `good`, 167 `bad`)
- `nstxuG142301W29`: 158 rows (7 `good`, 151 `bad`)

The 1,054-row list contains 222 `good` and 832 `bad` labels. Rows were copied
verbatim from `training_labels/tae_like_v2_nonG.csv`, retaining the full
seven-column schema and source ordering. Validation found no malformed rows,
duplicate paths, unexpected labels, or paths missing from
`/p/hym/ebelova/NOVA/data_mixed_new`. The `nonG` name denotes a provisional
list while the remaining G shots await label audit; audited G shots B12, J38,
and W29 are included. Existing path-config defaults still point to v2 and were
not changed in this step.

### 2026-08-23: regenerated TAE/EAE splits for nine refreshed shots

Ran `scripts/split_tae_eae.py` with its default thresholds on the nine shot
trees whose split products were intentionally omitted while rebuilding
`/p/hym/ebelova/NOVA/data_mixed_new`. Each output was written to the matching
`<shot>_tae_eae_split/` directory under that data root and contains
`all_modes.csv`, `tae_like.csv`, `eae_like.csv`, and
`all_modes_tae_eae_split.csv`.

| shot | all modes | strict TAE-like | mixed in TAE output | TAE output total | EAE-like | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nstx_120113` | 541 | 133 | 41 | 174 | 367 | 0 |
| `nstx_135388` | 1169 | 330 | 15 | 345 | 824 | 0 |
| `nstxuG121123Q62` | 1186 | 219 | 30 | 249 | 937 | 0 |
| `nstxuG133964S31` | 830 | 52 | 24 | 76 | 754 | 0 |
| `nstxuG142301H47` | 1016 | 162 | 16 | 178 | 838 | 0 |
| `nstxuG142301Y93` | 651 | 100 | 13 | 113 | 538 | 0 |
| `nstx_141711` | 535 | 146 | 12 | 158 | 377 | 0 |
| `nstxuG121123K51` | 766 | 149 | 3 | 152 | 614 | 0 |
| `nstxu_204202` | 533 | 130 | 10 | 140 | 393 | 0 |
| **total** | **7227** | **1421** | **164** | **1585** | **5642** | **0** |

Exact-inventory validation confirmed that all 7,227 direct `N*/egn*` modes
appear once in `all_modes.csv` and the full audit table, that the TAE and EAE
path sets are disjoint and conserve the complete input set, and that every
output path exists under its corresponding new shot tree. All split scalars
are populated and no error rows were produced. Together with the six retained
split directories, `data_mixed_new` now has one `_tae_eae_split` directory for
each of its 15 canonical shots.

The count above follows the splitter's documented nonrecursive `N*/egn*`
input rule. `nstx_135388` also retains 858 nested `N*/Out/egn*` files copied
from the previous database; `split_tae_eae.py` does not scan those run-output
subdirectories, so they are not part of its 1,169-mode canonical split input.

At generation time, the nine newly generated CSV sets contained absolute mode
paths under `data_mixed_new`. The six split directories copied unchanged
earlier retained their original absolute path roots: E202855, E204669,
E205052, and J38 pointed to DiTw, while B12 and W29 pointed to the former
`data_mixed` tree. Those source mode files had been verified identical, so
this did not change the split classifications.

### 2026-08-23: promoted the rebuilt database and repaired split paths

The user renamed the former `/p/hym/ebelova/NOVA/data_mixed` tree to
`/p/hym/ebelova/NOVA/data_mixed_tmp` and promoted `data_mixed_new` to the
canonical `/p/hym/ebelova/NOVA/data_mixed` path. The separately frozen
`data_mixed_2026_08_20` snapshot remains available and was checksum-verified
against the former database immediately before the cutover.

Rewrote only the absolute mode-path prefix in the four generated split CSVs
for each of the nine refreshed shots, changing
`/p/hym/ebelova/NOVA/data_mixed_new/` to
`/p/hym/ebelova/NOVA/data_mixed/`. This updated 21,681 CSV data-row path
entries across 36 files. Post-rewrite validation confirmed that all paths
exist in the promoted database, the 7,227-mode input inventory is conserved,
the 1,585 TAE-like and 5,642 EAE-like path sets remain disjoint and complete,
and all audit rows still have valid scalars with zero errors. No split
classification or non-path field was changed.

Of the six split directories retained unchanged, B12 and W29 now resolve
through the promoted canonical `data_mixed` path automatically. E202855,
E204669, E205052, and J38 still retain their valid DiTw absolute paths; their
corresponding mode payloads were previously verified identical to the copied
canonical shot trees.

### 2026-08-23: old-vs-new TAE-list membership for six continuum refreshes

Compared normalized `shot/N*/egn*` membership in the old and rebuilt
`tae_like.csv` lists for `nstx_120113`, `nstx_135388`, Q62, S31, H47, and Y93.
The five available historical splits came from `data_mixed_tmp`. Because no
historical `_tae_eae_split` directory existed for `nstx_120113`, generated a
temporary baseline with the current `split_tae_eae.py` and that shot's old
numbered continuum profiles. The temporary files were removed after the
comparison.

All six old/new all-mode inventories are identical, so the differences below
are continuum-driven TAE/EAE membership changes rather than mode-filename
additions or removals:

| shot | old TAE | new TAE | added | removed | net |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nstx_120113` | 174 | 174 | 0 | 0 | 0 |
| `nstx_135388` | 344 | 345 | 1 | 0 | +1 |
| `nstxuG121123Q62` | 241 | 249 | 8 | 0 | +8 |
| `nstxuG133964S31` | 76 | 76 | 0 | 0 | 0 |
| `nstxuG142301H47` | 175 | 178 | 3 | 0 | +3 |
| `nstxuG142301Y93` | 106 | 113 | 7 | 0 | +7 |
| **total** | **1116** | **1135** | **19** | **0** | **+19** |

New TAE-list members are:

- `nstx_135388`: `N4/egn04w.1922E+03` (`above_upper2 -> mixed`)
- Q62: `N1/egn01w.1537E+02`, `N1/egn01w.1564E+02`,
  `N1/egn01w.1593E+02`, `N1/egn01w.1655E+02`,
  `N1/egn01w.1710E+02`, `N1/egn01w.1902E+02`,
  `N1/egn01w.1937E+02`, and `N1/egn01w.2780E+02` (all
  `above_upper2 -> mixed`)
- H47: `N1/egn01w.3450E+02` (`above_upper2 -> below_upper2`),
  `N3/egn03w.3059E+02`, and `N8/egn08w.1194E+02` (both
  `above_upper2 -> mixed`)
- Y93: `N1/egn01w.1894E+02`, `N1/egn01w.1937E+02`,
  `N1/egn01w.2146E+02`, `N2/egn02w.2123E+02`,
  `N3/egn03w.1635E+02`, and `N3/egn03w.2200E+02` (all
  `above_upper2 -> mixed`), plus `N3/egn03w.2513E+02`
  (`above_upper2 -> below_upper2`)

No former TAE member was removed. None of the 19 newly admitted modes has a
row in `training_labels/tae_like_v2_nonG.csv`; all 19 therefore require new
label review before inclusion in v3. This is a path-membership result: even
where membership is unchanged, recomputed continuum scalars may differ and
can still affect continuum-dependent model features.

### 2026-08-23: targeted non-G continuum-review shortlist

Created a viewer-safe suspect list for targeted reinspection of the recently
reviewed `nstx_120113` and `nstx_135388` labels:

```text
tests/labels_audit/continuum_refresh_2026_08_23/
  nonG_suspect_modes.csv
  nonG_suspect_mode_details.csv
  README.md
```

The selection is explicitly a non-blind adjudication aid. It includes all
retained low-confidence review cases plus, for each shot with a nonzero
change, the top 10 current TAE modes by absolute full-precision old/new change
in `signed_delta` and the top 10 by absolute change in
`fraction_below_upper2`. The union contains 45 unique modes: 4 from
`nstx_120113` and 41 from `nstx_135388`; 30 are confidence-selected, 16 are
scalar-selected, and one satisfies both criteria.

The `nstx_120113` upper-gap scalars are exactly unchanged at full precision
for all 174 current TAE modes despite byte differences in all ten numbered
continuum files, so only its four confidence-sensitive modes enter the list.
For `nstx_135388`, the two top-10 scalar rankings have a 16-mode union; the
largest change is the newly admitted `N4/egn04w.1922E+03` mode
(`above_upper2 -> mixed`).

`nonG_suspect_modes.csv` includes the current adjudicated labels so this explicit
non-blind audit can be completed directly in `viz/view_modes_csv.py` without
requiring a separate handwritten record. Its post-review labels comprise 23
GOOD and 22 BAD. The initial labels for 44 modes came from the recent clean
human-review list, while the newly admitted
`nstx_135388/N4/egn04w.1922E+03` mode retains its BAD label from the preserved
old all-mode list because it was outside the former TAE-like split.
The viewer now preserves and displays SKIP labels as well as GOOD and BAD.
Its mode-structure and continuum panels now share the normalized-radius
x-axis, and both continuum-location markers extend into the mode panel for
direct alignment with amplitude spikes.
The companion details table retains prior-label and confidence provenance,
reasons, raw scalar values, changes, ranks, and gap-region transitions. All 45
relative paths were verified to resolve under the promoted canonical
`/p/hym/ebelova/NOVA/data_mixed` tree.

After visually reinspecting the shortlist with the aligned mode/continuum
panels, the user made five low-confidence decisions:

- `nstx_120113/N6/egn06w.1418E+02`: SKIP -> GOOD
- `nstx_135388/N5/egn05w.2098E+02`: BAD -> GOOD
- `nstx_135388/N5/egn05w.2248E+02`: BAD -> GOOD
- `nstx_135388/N6/egn06w.1980E+02`: GOOD -> BAD
- `nstx_135388/N7/egn07w.2394E+02`: BAD -> GOOD

These explicitly non-blind decisions are recorded in
`nonG_suspect_label_changes.csv` with `confidence=low` and `prior_seen=true`.
They update the active clean human-review source and the working viewer list,
whose new counts are 23 GOOD and 22 BAD. The sealed reviews, policy-v2 files,
frozen 2026-08-20 labels, and active v2 training list remain unchanged. The
five changes are applied in the subsequent v3 expansion below.

### 2026-08-23: expanded v3 with refreshed `nstx_120113` and `nstx_135388`

Expanded `training_labels/tae_like_v3_nonG.csv` from six to eight shots using
the regenerated split manifests under the promoted canonical
`/p/hym/ebelova/NOVA/data_mixed` tree. The requested `nstx_120133` name was
interpreted as `nstx_120113` after a bounded check found no `nstx_120133`
directory and confirmed the audited `nstx_120113` directory.

Added:

- `nstx_120113`: 174 rows (46 GOOD, 128 BAD)
- `nstx_135388`: 345 rows (133 GOOD, 212 BAD)

The first shot has exact one-to-one coverage of its current 174-row TAE-like
split. The second covers all 345 current TAE-like modes: 344 adjudicated human
labels plus the newly admitted `N4/egn04w.1922E+03` mode, whose preserved BAD
label was retained. The five low-confidence continuum-refresh decisions are
included. All added scalar and gap-region fields come directly from the
regenerated split CSVs.

The resulting eight-shot candidate contains 1,573 unique rows: 401 GOOD and
1,172 BAD. As part of the required family/validity consistency check, 30 stale
family values inherited by the original six-shot v3 block were normalized:
23 `bad,tae` rows became `bad,none`, and 7 `good,none` rows became
`good,tae`. No existing validity label or scientific scalar was changed.

Validation confirmed exact target-shot split membership, relative portable
paths, allowed labels, no SKIP or error rows, consistent family values, and
all 1,573 files resolving under the canonical data root. The v2 and dated
2026-08-20 lists remain unchanged, and path-config defaults still point to
v2 while v3 remains provisional.

### 2026-08-23: Flux Matplotlib compatibility for interactive labeling

Updated all three `scripts/label_modes_fast.py` subplot layouts to pass
`height_ratios` through `gridspec_kw`. This preserves the existing panel
proportions while avoiding the `Figure.__init__() got an unexpected keyword
argument 'height_ratios'` failure from the older system Matplotlib on Flux.
The mode-structure and continuum panels in both normal labeling and manual
adjudication now also share an explicit normalized-radius `[0, 1]` axis. Their
plot boxes and continuum-location markers therefore align, while the lower
harmonic-spectrum panel retains its independent stored-index axis.

### 2026-08-23: added refreshed S31 labels to v3

Interpreted the requested `S1` shot as the completed
`nstxuG133964S31` review. Added all 76 current S31 TAE-like modes to
`training_labels/tae_like_v3_nonG.csv` using the regenerated split scalars and
the deduplicated human-review labels. Every S31 mode is BAD; there are no
SKIPs, duplicate labels, missing manifest modes, or extra review paths.

The old and current S31 TAE-like manifests contain the same 76 paths. The old
v2 training list covered 74 of them, all BAD, so there are no label reversals.
V3 adds BAD labels for the two previously absent modes:

- `nstxuG133964S31/N5/egn05w.1581E+02`
- `nstxuG133964S31/N10/egn10w.1678E+02`

The resulting nine-shot provisional v3 list contains 1,649 unique rows: 401
GOOD and 1,248 BAD. Validation confirmed exact S31 split coverage, portable
relative paths, current split scalars, no errors or family mismatches, and all
1,649 mode files resolving under the canonical data root. V2 and the dated
2026-08-20 label snapshots remain unchanged.

### 2026-08-23: compared refreshed H47 review with old labels

Validated the completed `nstxuG142301H47` human review against the current
178-row regenerated TAE-like split. The raw and clean review files are
byte-identical and contain 12 GOOD and 166 BAD decisions, with exact split
coverage, no duplicates, and no SKIPs.

The frozen/current v2 training list has 169 H47 rows (9 GOOD, 160 BAD). Of
those shared paths, the initial comparison found five disagreements. During
review, `N10/egn10w.1403E+02` was corrected from the mistakenly entered BAD
label to GOOD, matching v2. The final comparison has 165 agreements and four
differences: three BAD-to-GOOD and one GOOD-to-BAD. Agreement is 165/169 =
97.63%, with Cohen's kappa 0.7876. The three retained BAD-to-GOOD modes were
adjudicated as extremum-localized types. A viewer-ready four-row comparison
was written to
`tests/labels_audit/continuum_refresh_2026_08_23/nstxuG142301H47_label_disagreements.csv`;
its displayed `label` is the old v2 decision and its `new_label` column
preserves the final review decision. The explicitly non-blind N10 correction
is retained in `nstxuG142301H47_post_comparison_changes.csv`.

Nine reviewed modes have no old v2 label. Six were present in the old
175-mode TAE split but omitted from its 169-row training subset; three are the
known continuum-driven additions to the refreshed 178-mode split. At comparison
time, H47 had not yet been merged into `training_labels/tae_like_v3_nonG.csv`.

### 2026-08-23: added finalized H47 labels to v3

Added all 178 finalized `nstxuG142301H47` labels to
`training_labels/tae_like_v3_nonG.csv`, joined by normalized mode path to the
current regenerated H47 `tae_like.csv` manifest. The added block contains 12
GOOD and 166 BAD rows and uses current split scalars and gap-region values.
It includes the post-comparison `N10/egn10w.1403E+02=GOOD` correction and the
three retained BAD-to-GOOD extremum-localized adjudications.

The resulting ten-shot provisional v3 list contains 1,827 unique rows: 413
GOOD and 1,414 BAD. The preceding 1,649-row v3 prefix was preserved
byte-for-byte. Full validation confirmed exact H47 review/split coverage,
allowed binary labels, portable relative paths, no duplicates, SKIPs, errors,
family mismatches, scalar mismatches, or missing mode files. The v2 and dated
2026-08-20 snapshots remain unchanged. The resulting v3 SHA-256 is
`f90439792258bec7e310eb366192fdf59670431d87552bf9f256c4054cd30f20`.

### 2026-08-23: compared refreshed Y93 review with old labels

Validated the completed `nstxuG142301Y93` human review against the current
113-row regenerated TAE-like split. The raw and clean review files are
byte-identical and contain one GOOD and 112 BAD labels, with exact split coverage, no
duplicates or SKIPs, and all mode files resolving under the canonical data
root.

The frozen/current v2 list exactly covers the former 106-mode Y93 split with
one GOOD and 105 BAD labels. The initial comparison found one disagreement at
`N9/egn09w.1539E+02`; during post-comparison review, the user retained the old
GOOD label. The finalized review now agrees on all 106 shared paths (100%
agreement, Cohen's kappa 1.0). The correction is recorded in
`nstxuG142301Y93_post_comparison_changes.csv`, and the retained disagreement
CSV now contains only its header.

The seven reviewed paths without old v2 labels are exactly the known
continuum-driven additions to the refreshed Y93 TAE split; all seven are BAD.
They are collected for visual inspection in the viewer-ready
`nstxuG142301Y93_new_tae_modes.csv`, which includes their current labels and
regenerated split scalars.
At comparison time, Y93 had not yet been merged into
`training_labels/tae_like_v3_nonG.csv`.

### 2026-08-23: added finalized Y93 labels to v3

Added all 113 finalized `nstxuG142301Y93` labels to
`training_labels/tae_like_v3_nonG.csv`, joined by normalized mode path to the
current regenerated Y93 `tae_like.csv` manifest. The added block contains one
GOOD and 112 BAD rows and uses current split scalars and gap-region values. It
retains `N9/egn09w.1539E+02=GOOD`; the seven continuum-driven additions are all
BAD.

The resulting eleven-shot provisional v3 list contains 1,940 unique rows: 414
GOOD and 1,526 BAD. The preceding 1,827-row v3 prefix was preserved
byte-for-byte. Full validation confirmed exact Y93 review/split coverage,
allowed binary labels, portable relative paths, no duplicates, SKIPs, errors,
family mismatches, scalar mismatches, or missing mode files. The v2 and dated
2026-08-20 snapshots remain unchanged. The resulting v3 SHA-256 is
`32d6ea4e3de54e6b9d7c0a4715e18739417201655845498ea8ecba2814597ed9`.

### 2026-08-23: compared refreshed Q62 review with old labels

Validated the completed `nstxuG121123Q62` human review against the current
249-row regenerated TAE-like split. The raw and clean review files are
byte-identical and contain 16 GOOD and 233 BAD final labels, with exact split
coverage, no duplicates or SKIPs, and all mode files resolving under the
canonical data root.

The frozen/current v2 list exactly covers the former 241-mode Q62 split with
12 GOOD and 229 BAD labels. The initial comparison had 230 agreements and 11
differences: seven GOOD-to-BAD and four BAD-to-GOOD (95.44%, Cohen's kappa
0.4528). That initial state is preserved in
`nstxuG121123Q62_initial_label_disagreements.csv`.

The user then adjudicated all 11 modes GOOD with low confidence because they
form one smooth morphology family with small-r continuum crossings but no
resonant-like amplitude spikes. Seven precheck BAD labels were corrected to
GOOD and recorded in `nstxuG121123Q62_post_comparison_changes.csv` with
`prior_seen=true`; `N9/egn09w.2152E+02` remains GOOD. The complete adjudicated
family is retained in the viewer-ready
`nstxuG121123Q62_uniform_low_conf_good_modes.csv` with final labels,
confidence, rationale, label provenance, and current split scalars.

The final comparison has 237/241 agreements = 98.34%, with Cohen's kappa
0.8485. The four remaining differences are all old-BAD-to-final-GOOD and are
listed in `nstxuG121123Q62_label_disagreements.csv`, which displays the old v2
label for inspection.

The eight reviewed paths without old v2 labels are exactly the known
continuum-driven N1 additions to the refreshed Q62 TAE split; all eight are
BAD. They are collected in the viewer-ready
`nstxuG121123Q62_new_tae_modes.csv` with current labels and regenerated split
scalars. At comparison time, Q62 had not yet been merged into
`training_labels/tae_like_v3_nonG.csv`.

### 2026-08-23: added finalized Q62 labels to v3

Added all 249 finalized `nstxuG121123Q62` labels to
`training_labels/tae_like_v3_nonG.csv`, joined by normalized mode path to the
current regenerated Q62 `tae_like.csv` manifest. The added block contains 16
GOOD and 233 BAD rows and uses current split scalars and gap-region values. It
includes all 11 explicitly non-blind, low-confidence GOOD adjudications; the
eight continuum-driven additions are all BAD.

The resulting twelve-shot provisional v3 list contains 2,189 unique rows: 430
GOOD and 1,759 BAD. The preceding 1,940-row v3 prefix was preserved
byte-for-byte. Full validation confirmed exact Q62 review/split coverage,
allowed binary labels, portable relative paths, no duplicates, SKIPs, errors,
family mismatches, scalar mismatches, or missing mode files. The v2 and dated
2026-08-20 snapshots remain unchanged. The resulting v3 SHA-256 is
`0fca442ac6eaf27417ee8377bc18eb2825806708d1963955b7b098d6eb4f4b85`.

### 2026-08-23: staged nstxu_204202 exact-label transfer

Compared the current 140-row `nstxu_204202` TAE-like split with the 275 labels
for this shot in `tae_like_v2_nonG.csv`. The current set is a strict subset of
the old labeled set: all 140 current paths are shared, there are no genuinely
new current TAE-like paths, and 135 old paths are absent from the current mode
tree. Byte-for-byte comparison with `data_mixed_2026_08_20` confirmed that all
140 shared mode files are unchanged.

Staged the 140 exact transferred labels (62 GOOD and 78 BAD) in
`tests/labels_audit/continuum_refresh_2026_08_23/nstxu_204202_transferred_labels.csv`.
The component uses relative paths, current split ordering, current regenerated
split scalars, and normalized family fields. The label-free new-mode inspection
list is header-only because the current-minus-old set is empty.

Quarantined the 135 old-only labels (two GOOD and 133 BAD) in
`tests/labels_audit/continuum_refresh_2026_08_23/nstxu_204202_quarantined_old_labels.csv`.
Every quarantined path is absent from the current data tree; the file preserves
the old v2 label/scalar provenance and records reason
`absent_from_current_mode_tree`. This staged component has not yet been merged
into `training_labels/tae_like_v3_nonG.csv`, which remains at 2,189 rows (430
GOOD and 1,759 BAD).

### 2026-08-23: added nstxu_204202 exact transfers to v3

Added all 140 validated `nstxu_204202` transfer rows to
`training_labels/tae_like_v3_nonG.csv`: 62 GOOD and 78 BAD. The appended block
exactly matches the current TAE-like split, uses its regenerated scalar values,
and excludes all 135 quarantined paths that are absent from the current mode
tree.

The resulting thirteen-shot provisional v3 list contains 2,329 unique rows:
492 GOOD and 1,837 BAD. The preceding 2,189-row v3 prefix was preserved
byte-for-byte. Full validation confirmed portable relative paths, allowed
binary labels, consistent family fields, no errors or duplicates, exact
`nstxu_204202` split coverage, and all 2,329 mode files resolving under the
canonical data root. The resulting v3 SHA-256 is
`5f441650f47a0e8f26e95f26e658081d1968527b505f00b9a3d1420613ead683`.

### 2026-08-23: prepared nstx_141711 full current-shot review

Prepared a new label-free review of the recalculated `nstx_141711` modes. The
current regenerated TAE-like split contains 158 unique modes across `N1`
through `N10`; all mode files resolve under the canonical data root and the
split contains no error rows.

Generated
`tests/labels_audit/continuum_refresh_2026_08_23/nstx_141711_blind_manifest.csv`
from the current split using the visual-review manifest preparer, together
with the blank structured `nstx_141711_blind_decisions.csv` template. The
manifest contains no prior labels or classifier fields. The interactive human
review will be written to the new `nstx_141711_human_labels.csv` path using
`label_modes_fast.py`, the whole-shot `N*/egn*` pattern, the blind manifest as
its mode list, signed harmonics, and `--no-rf`.

No `nstx_141711` rows have been added to `tae_like_v3_nonG.csv`. Historical
label comparison and any v3 merge remain deferred until the independent
158-mode current-shot review is complete and validated.

### 2026-08-23: completed nstx_141711 review and comparison

Validated the completed current-shot review against the 158-row blind
manifest. The clean review has exact coverage, 75 GOOD and 83 BAD decisions,
no SKIPs, and no missing mode files. The 159-row raw history repeats
`N3/egn03w.5246E+02=GOOD` once; deduplication retains the same decision. The
clean pre-comparison file is frozen by a SHA-256 sidecar with digest
`0264756d76e83d209390137261511af5baa63e092ebdef2aecc15b1dd13792d3`.

Compared the review with old v2 labels only after completeness validation. Of
154 exact shared paths, 148 agree and six differ (96.10% agreement, Cohen's
kappa 0.9221): five old GOOD to current BAD and one old BAD to current GOOD.
Three disagreement paths have changed mode payloads relative to the frozen
2026-08-20 database and three are byte-identical. The viewer-ready differences
are preserved in `nstx_141711_initial_label_disagreements.csv`; the complete
initial shared-path table is in
`nstx_141711_initial_shared_label_comparison.csv`.

Four current TAE-like paths have no old v2 label; all four are N1 modes and all
were reviewed BAD. Of 102 old-only labeled paths, 101 mode files are absent
from the current tree and one old-BAD mode remains in the tree but is not in
the current TAE-like split. The new and old-only sets are recorded in
`nstx_141711_new_tae_modes.csv` and
`nstx_141711_quarantined_old_labels.csv`, respectively.

The six label differences await user adjudication. No `nstx_141711` rows have
been added to `training_labels/tae_like_v3_nonG.csv`.

### 2026-08-23: adjudicated nstx_141711 label differences

The user retained old v2 labels for four of the six initial differences and
kept the current review decisions for two: `N7/egn07w.9318E+02=GOOD` and
`N8/egn08w.1026E+03=BAD`. The four changes from the checksummed pre-comparison
review are recorded separately in `nstx_141711_post_comparison_changes.csv`
with `prior_seen=true`; `nstx_141711_disagreement_adjudication.csv` records the
final decision for all six initially differing modes.

Created `nstx_141711_human_labels_final.csv` without altering the preserved
pre-comparison clean review. The final 158-mode list contains 79 GOOD and 79
BAD decisions and has SHA-256
`2a82d29d8d9d7905e563dbe47e95bc4f57fe35570731ff58f6f6b6757bffe48a`.
Final old-label agreement is 152/154 shared paths (98.70%, Cohen's kappa
0.9740), leaving only the two explicitly adjudicated differences in
`nstx_141711_label_disagreements.csv`. The final complete shared-path table is
`nstx_141711_shared_label_comparison.csv`.

No `nstx_141711` rows have been added to `training_labels/tae_like_v3_nonG.csv`.

### 2026-08-23: added finalized nstx_141711 labels to v3

Added the complete 158-row finalized `nstx_141711` review to
`training_labels/tae_like_v3_nonG.csv`, using regenerated current split
scalars and normalized family fields. The block contains 79 GOOD and 79 BAD
labels, including the explicitly adjudicated `N7/egn07w.9318E+02=GOOD` and
`N8/egn08w.1026E+03=BAD` decisions. All 102 old-only quarantined labels remain
excluded.

The resulting fourteen-shot provisional v3 list contains 2,487 unique rows:
571 GOOD and 1,916 BAD. The preceding 2,329-row v3 prefix was preserved
byte-for-byte. Full validation confirmed exact current-shot review and split
coverage, allowed binary labels, portable relative paths, consistent family
fields, current scalars, no errors or duplicates, and all 2,487 mode files
resolving under the canonical data root. The v2 and frozen 2026-08-20 label
snapshots remain unchanged. The resulting v3 SHA-256 is
`a613b1e259321d18d4f89007e83f5a79aa8637cc4469ba00b3c327bc07d994e0`.

### 2026-08-23: prepared K51 full current-shot review

Prepared a new label-free review package for the recalculated
`nstxuG121123K51` modes. The current regenerated TAE-like split contains 152
unique modes across `N2` through `N10`; all files resolve under the canonical
data root and the split contains no error rows.

Generated
`tests/labels_audit/continuum_refresh_2026_08_23/nstxuG121123K51_blind_manifest.csv`
from the current split using the visual-review manifest preparer, together
with the blank structured `nstxuG121123K51_blind_decisions.csv` template. The
manifest contains no prior labels or classifier fields. Whole-shot validation
confirmed that the `N*/egn*` scan sees 766 K51 mode files and the manifest
filters it to exactly the 152 current TAE-like targets.

The interactive human review will be written to the new
`nstxuG121123K51_human_labels.csv` path using `label_modes_fast.py`, signed
harmonics, the label-free manifest, and `--no-rf`. Historical comparison and
any v3 merge remain deferred until the complete current-shot review is
validated.

### 2026-08-23: completed K51 review and comparison

Validated `nstxuG121123K51_human_labels_clean.csv` directly against the
152-row blind manifest, as requested, without using the duplicate-bearing raw
labeler history. The clean review has exact coverage, 25 GOOD and 127 BAD
decisions, no SKIPs, and no missing mode files. Its frozen SHA-256 sidecar has
digest `ae59e1386911d2c3d2dbba42e37b932f2bcb1e8eea714fc4bcdde2ba7f13ea1b`.

Compared historical labels only after completeness validation. Of 122 exact
shared paths, 112 agree and ten differ (91.80% agreement, Cohen's kappa
0.7408): six old GOOD to current BAD and four old BAD to current GOOD. All ten
disagreement mode payloads changed relative to the frozen 2026-08-20 database.
The viewer-ready differences are in
`nstxuG121123K51_initial_label_disagreements.csv`; the complete initial
shared-path table is in `nstxuG121123K51_initial_shared_label_comparison.csv`.

Thirty current TAE-like paths have no old v2 label, comprising two GOOD and 28
BAD decisions. All 86 old-only labeled mode files are absent from the current
tree; their old labels comprise three GOOD and 83 BAD. The new and old-only
sets are recorded in `nstxuG121123K51_new_tae_modes.csv` and
`nstxuG121123K51_quarantined_old_labels.csv`, respectively.

The ten label differences await user adjudication. No K51 rows have been
added to `training_labels/tae_like_v3_nonG.csv`.

### 2026-08-23: adjudicated K51 label differences

The user changed only `N9/egn09w.3938E+02` from the clean-review GOOD decision
to BAD, matching its old v2 label, and retained the clean-review decisions for
the other nine initially differing modes. The single change is preserved in
`nstxuG121123K51_post_comparison_changes.csv` with `prior_seen=true`; the
complete ten-mode decision record is
`nstxuG121123K51_disagreement_adjudication.csv`.

Created `nstxuG121123K51_human_labels_final.csv` without altering the
checksummed pre-comparison clean list. The finalized 152-mode review contains
24 GOOD and 128 BAD decisions and has SHA-256
`3d75f3019c8e9bd2f2abd785f3786439a6f5defd8607de587434a83f223ac766`.
Final old-label agreement is 113/122 shared paths (92.62%, Cohen's kappa
0.7631), leaving nine differences in
`nstxuG121123K51_label_disagreements.csv`. The final complete shared-path table
is `nstxuG121123K51_shared_label_comparison.csv`.

No K51 rows have been added to `training_labels/tae_like_v3_nonG.csv`.

### 2026-08-23: added finalized K51 labels to v3

Added the complete 152-row finalized `nstxuG121123K51` review to
`training_labels/tae_like_v3_nonG.csv`, using regenerated current split
scalars and normalized family fields. The block contains 24 GOOD and 128 BAD
labels and includes the post-comparison `N9/egn09w.3938E+02=BAD` correction.
All 86 old-only quarantined paths remain excluded.

The resulting fifteen-shot provisional v3 list contains 2,639 unique rows:
595 GOOD and 2,044 BAD. The preceding 2,487-row v3 prefix was preserved
byte-for-byte. Full validation confirmed exact current-shot review and split
coverage, allowed binary labels, portable relative paths, consistent family
fields, current scalars, no errors or duplicates, and all 2,639 mode files
resolving under the canonical data root. The v2 and frozen 2026-08-20 label
snapshots remain unchanged. The resulting v3 SHA-256 is
`0eb939ef226447f66121d44fc6eb13b6b9ef64746958c189588058e26846d364`.

All 15 rebuilt training shots are now represented in the provisional v3
candidate. The historical `nonG` filename remains unchanged for the moment;
path-config defaults also still point to the preserved v2 list pending the
next explicit promotion/retraining decision.

### 2026-08-23: promoted audited v3 to the main training list

Renamed the completed `training_labels/tae_like_v3_nonG.csv` candidate to
`training_labels/tae_like_v3.csv` now that every G shot has been audited, then
replaced `training_labels/tae_like_train.csv` with an exact copy. Both active
files contain 2,639 unique rows across all 15 shots: 595 GOOD and 2,044 BAD,
with SHA-256
`7cf7b3cbf07a6af65311867bc109ac8783e50829f4d9655e33374890447ec0ea`.
Both copies were normalized from CRLF to LF during promotion so Git can audit
their diffs cleanly; this did not alter any CSV records. The completed
pre-promotion CRLF candidate had SHA-256
`0eb939ef226447f66121d44fc6eb13b6b9ef64746958c189588058e26846d364`.

Updated the NERSC Bash, Flux Bash, and Flux tcsh path configs so
`NOVA_TRAIN_CSV` and `NOVA_TRAIN_CSV_TAE` point to
`training_labels/tae_like_train.csv`. Updated the standalone Python fallbacks
in `cnn_raw.py`, `run_loso_10.py`, and `audit_training_provenance.py`, plus the
labeler help and current training examples, to use the same canonical main
list.

The pre-promotion `tae_like_train.csv` contents had 2,903 rows and SHA-256
`2c6c1d7ebb1743a592b0590f089a610d962508ed1bd71e3778e6e679d2afc919`;
they remain recoverable from Git history. The preserved v2 list and its dated
2026-08-20 copy remain byte-identical with SHA-256
`8587aef8876c575c27f4404d44a4a45f9e46ffa210b7efc53ec67e2de149f0ad`.
Existing RF/CNN checkpoints have not yet been retrained on v3.
