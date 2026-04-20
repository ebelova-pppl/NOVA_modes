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
-	Best: straightened CNN ≈ 96% (threshold ≈ 0.55)

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
    -	PPPL Flux ✅

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
Summary on TAE/EAE issue: A robust practical separation of TAE-like and EAE-like modes was obtained using two upper-gap metrics: fraction_below_upper2 and normalized_signed_delta. Modes with fraction_below_upper2 > 0.5 were classified as TAE-like; modes with fraction_below_upper2 < 0.4 and normalized_signed_delta < -0.1 were classified as EAE-like; intermediate cases were marked as mixed and included in the TAE-like set to avoid losing marginal TAEs. This recovered all labeled TAEs while keeping clear EAEs separate, and restored the RF classifier performance to near the original TAE-only level.

### 2026-04-20
Fixed seed generation issue for cnn_straightened.py
Updated results for new tae_like.csv list (1085 modes):
    cnn_raw:          best accuracy=0.96, c.matrix:[[127,4][4,81]]
    cnn_straightened: best accuracy=0.95, c.matrix:[[126,5][8,77]]
    cnn_hybrid:       best accuracy=0.96, c.matrix:[[129,2][6,79]]
    RF:               accuracy=0.94, c.matrix= [[62,4][3,40]]
        === Feature Importances === 
        delta2_eff 0.1140 
        W_star 0.1068 
        max_abs_d1_abs 0.1000 
        S 0.0946 
        std_amp 0.0916

Updated results for new eae_like.csv list (2042 modes):
    cnn_raw:          best accuracy=0.91, c.matrix:[[323  17][ 19  48]]
        Classification report:
              precision    recall  f1-score   support
         bad       0.94      0.95      0.95       340
        good       0.74      0.72      0.73        67
    RF:               accuracy=0.94, c.matrix:[[162 9] [ 3 31]] 
        Classification report (test): 
          precision recall f1-score support 
        0  0.98       0.95   0.96     171 
        1  0.78       0.91   0.84.    34
        === Feature Importances === 
        rad_loc 0.1644 
        rad_width 0.0804 
        mean_abs_d2_mode 0.0787 
        gamma_d 0.0611 
        ntor 0.0540
### TAE/EAE sorting is solved, now ready to merge mixed_branch back to main