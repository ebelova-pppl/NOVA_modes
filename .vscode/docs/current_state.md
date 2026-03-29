# Project: AI NOVA mode classifier
## Project state (2026-03-29)
### Goal
Train ML classifiers to identify physically meaningful NOVA eigenmodes (“good”) vs unphysical/numerical modes (“bad”), and provide a clean, deduplicated mode set for downstream analysis (e.g., NOVA-C, surrogate modeling, digital twin workflows).
 
## Data
-	~1272 labeled modes
    -	NSTX-U: 1 shot
    -	NSTX: 3 shots
    -	DIII-D: optional (future)

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
    -	Comparable to RF, out of box – not optimized ye

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
-	mode_transform.py
-	view_modes_csv.py
-	sort_shot.py
### RF
-	rf_train_classify.py (renamed from nova_mode_classifier.py)
-	rf_oof_check.py
-	find_rf_disagreements.py
-	label_modes_fast.py
### CNN
-	cnn_raw.py
-	cnn_straightened.py
-	cnn_hybrid.py
-	cnn_raw_classify.py (needs extension to other CNN models)
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
-	Best:
-	straightened CNN ≈ 96%
-	Typical threshold ≈ 0.55

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
•	Continue label cleaning (OOF + CNN comparison, discuss with NG)
•	Compare misclassified modes (RF vs CNN vs HybridCNN)
•	Update cnn_raw_classify.py for all CNN variants
 
## Next tasks
•	Add EAEs (second gap) → update continuum features
•	Extend training to broader frequency range
•	Investigate surrogate / autoencoder for mode structure
•	Possibly unify classification + sorting pipeline


