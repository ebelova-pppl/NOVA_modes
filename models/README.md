# Model Checkpoints

The top-level files are the active checkpoints refreshed from the canonical
`training_labels/tae_like_train.csv` list on 2026-08-28:

- `nova_mode_classifier.joblib` — the active 22-feature Random Forest
  pipeline (`StandardScaler` plus Random Forest);
- `nova_mode_classifier_bundle.joblib` — the same fitted RF together with its
  2,390-row feature matrix, labels, feature names, and schema metadata;
- `nova_cnn_raw.pt` — the active raw signed-mode CNN checkpoint.

## Training data and saved-model scope

The canonical training list contains 2,390 unique rows across 14 shots:

- 576 `good`;
- 1,814 `bad`;
- SHA-256
  `ce89a7d6ab6e5c17877e98fe50552a016b4b517c4f5942dbec00e5926bb14a3d`.

The 249 Q62 rows remain preserved in `training_labels/tae_like_v3.csv`, but
are suspended from the active training list and are not part of this
checkpoint refresh. The active RF model and raw-CNN checkpoint are full-list
fits; the RF bundle contains the same fitted RF. Predictions on these same
training modes are resubstitution/fit-health results, not independent
generalization estimates.

On Flux, keep the data roles separate:

- `$NOVA_DATA` is the rebuilt training database against which relative paths
  in the training CSV resolve;
- `$NOVA_DITW_ROOT` is the live DiTw shot database used for routine new-shot
  sorting.

Those roots can have different inventories or file payloads for a shot. A
label-path match alone therefore does not establish an independent test.
Production commands should continue to pass an explicit live `--shot_dir`
under `$NOVA_DITW_ROOT`.

The repository Flux `tcsh` path config defines both variables when sourced.
The Bash companion currently defines only `$NOVA_DITW_ROOT`; `$NOVA_DATA`
remains unset unless the caller sets it before a relative-path training or
inspection workflow.

## Random Forest

The active RF uses feature schema `rf_w_star_max_22_v2` with 22 features. The
schema replaces raw `omega` from the older 22-feature model with the
continuum-crossing feature `W_star_max`. It does not enable the experimental
six-crossing or extremum feature extensions, or the energy-tie variant. The
Random Forest has 300 trees, balanced class weights, and random seed `42`.

Five-fold row-wise cross-validation accuracies during the refresh were
`0.9414`, `0.9372`, `0.9393`, `0.9414`, and `0.9644` (mean `0.9448`). The
temporary stratified 239-row check produced confusion matrix
`[[169, 12], [4, 54]]`, accuracy `0.933`, and GOOD precision/recall/F1
`0.818 / 0.931 / 0.871`. After that check, the pipeline was explicitly refit
on all 2,390 rows before both active RF artifacts were saved.

Both the cross-validation and temporary holdout split were row-wise rather
than shot-held-out. No new LOSO or fusion-threshold calibration was performed
as part of this refresh.

The RF pickle was created with scikit-learn `1.9.0`. Use the matching version
when loading it; scikit-learn `1.9.0` also requires `narwhals>=2.0.1`.

## Raw CNN

The raw-CNN development split used seed `42`, an 80/20 stratified split,
batch size `8`, robust normalization, `M_target=100`, `R_target=201`,
unweighted loss, OneCycleLR with peak learning rate `0.02`, and gradient
clipping at `1.0`. The best split checkpoint was epoch 40 with confusion
matrix `[[353, 9], [13, 102]]`, accuracy `0.9539`, and GOOD
precision/recall/F1 `0.919 / 0.887 / 0.903` on 477 rows.

The saved active checkpoint is not that split checkpoint. A fresh CNN was
subsequently refit for 80 epochs on all 2,390 rows and saved with
`saved_training_scope=full_csv_refit`. Its final training-set health check
found 576 true-GOOD labels and 576 predicted-GOOD labels. That count equality
is a collapse check, not a claim that the per-mode predictions were identical.

The active file is checkpoint version 2, model type `cnn_raw`, with one signed
input channel, no continuum branch, and inference threshold `0.5`. The
documented Perlmutter training environment used PyTorch `2.8.0+cu129`; the
checkpoint has also been validated on Flux CPU with PyTorch `2.8.0+cu128`.
Refresh output is retained in `outputs/rf_out.txt` and
`outputs/cnn_raw.txt`.

## Active SHA-256 checksums

```text
b0a3868163abf4c36b69065c2a0f222192bc39010e0c719ca4d1222b618dfa8c  nova_mode_classifier.joblib
3d3baa239ecd2cda55f4ba3ae7123df97ed3d59b7975b624101788ca390dd2f3  nova_mode_classifier_bundle.joblib
872909ebd382a560d3cb9ed9b323034ac0d4fedd87ea3a93d31ca104d1044c7e  nova_cnn_raw.pt
```

## Sorter usage

The canonical production path is deterministic rules with RF used only after
classification to select representatives among close-frequency,
structurally matched final-GOOD modes:

```tcsh
setenv SHOT_NAME nstx_120113
setenv NOVA_SORT_OUT /path/to/sort_outputs
mkdir -p "$NOVA_SORT_OUT"
python "$NOVA_REPO/scripts/sort_shot_mixed.py" \
  --method rules \
  --shot_dir "$NOVA_DITW_ROOT/$SHOT_NAME" \
  --rf_model "$NOVA_MODELS/nova_mode_classifier.joblib" \
  --out_dir "$NOVA_SORT_OUT/$SHOT_NAME"
```

The RF cannot change a rule decision in this workflow. The explicit
`--method rf-cnn` path remains available only for legacy RF+CNN classification
and comparison runs.

## Historical checkpoints

- `old_4shots_models/` contains historical four-shot RF/CNN and LOSO files.
- `pre_v2_nonG_20260810/` contains the canonical checkpoints that preceded
  the v2 non-G refresh.

The former top-level 2,900-row checkpoint metadata and checksums are no longer
active. Use version-control history when reproducing that state.
