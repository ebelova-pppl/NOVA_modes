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

