# Merge Checklist: `mixed_branch` -> `main`

Date: 2026-05-09

This note summarizes the branch diff against `main` and records the final
checks worth doing before merging.

Diff basis:
- `git diff main...HEAD`

Current status:
- branch is clean
- training-label docs are updated
- split workflow is documented
- mixed-mode data lists have been cleaned and regenerated

## Change Summary

### 1. Environment and path config

- [x] `configs/paths/nova_paths.nersc.sh`
  - switched defaults to mixed data / models / results
  - resolves `NOVA_REPO` from the config-file location, so it works from either
    `main` or a worktree
  - fixes stale `NOVA_TRAIN_CSV_TAE` path to `training_labels/train_tae.csv`
- [x] `src/paths.py`
  - comments updated to match the new environment semantics
- [x] `.gitignore`
  - ignores `chats/`

### 2. CSV input handling

- [x] `src/mode_csv.py`
  - added shared CSV reader that accepts lists with or without a header row
- [x] scripts updated to use the shared CSV parsing behavior where needed

### 3. TAE/EAE splitting workflow

- [x] `src/tae_eae_features.py`
  - new continuum-derived scalars for TAE/EAE separation
- [x] `scripts/split_tae_eae.py`
  - new splitter for mixed mode lists
  - writes `tae_like.csv`, `eae_like.csv`, full audit CSV, and family summary
  - current rule:
    - `fraction_below_upper2 > 0.5` -> TAE-like
    - `fraction_below_upper2 < 0.4` and `signed_delta < -0.1` -> EAE-like
    - otherwise -> `mixed`, routed into TAE-like

### 4. Continuum / datcon cleanup

- [x] `src/cont_features.py`
  - masks legacy `>999` sentinel values to `NaN`
  - trims bad tail spikes near the end of `low2/high2`
- [x] `viz/view_modes_csv.py`
  - defensive local masking for bad datcon tail values
- [x] `scripts/label_modes_fast.py`
  - plotting path no longer converts bad datcon tails into fake finite values

### 5. Model-training / inference updates

- [x] `scripts/cnn_classify.py`
- [x] `scripts/cnn_infer_common.py`
- [x] `scripts/cnn_raw.py`
- [x] `scripts/cnn_straightened.py`
- [x] `scripts/cnn_hybrid.py`
  - shared CNN inference path
  - checkpoint metadata / preprocess handling improved
  - straightened-model seed handling fixed
- [x] `scripts/rf_train_classify.py`
- [x] `scripts/rf_oof_check.py`
  - updated CSV/header handling and mixed-list compatibility

### 6. Training-label data updates

- [x] `training_labels/all_modes.csv`
  - now has header row: `path,validity,family`
  - cleaned to remove mistaken small-`n_m` files from `nstx_135388`
- [x] added split / audit outputs:
  - `training_labels/all_modes_tae_eae_split.csv`
  - `training_labels/tae_like.csv`
  - `training_labels/eae_like.csv`
  - `training_labels/mixed_tae_like.csv`
  - `training_labels/bad_tae_like.csv`
  - `training_labels/tae_misplaced_in_eae_like.csv`
- [x] added cleanup audit files:
  - `training_labels/all_modes_clean_nstx135388_nhar9.csv`
  - `training_labels/removed_nstx135388_nhar9.csv`
- [x] removed outdated curated lists:
  - `training_labels/good_tae.csv`
  - `training_labels/good_eae.csv`

### 7. Documentation and notes

- [x] `scripts/README.md`
  - updated for shared CSV readers, CNN usage, and TAE/EAE splitting
- [x] `training_labels/README.md`
  - updated for current master/split/audit CSV lists
- [x] `docs/project_state.md`
  - records TAE/EAE split design and latest model results

## Verification Already Done

- [x] reviewed branch diff against `main`
- [x] verified the worktree is clean
- [x] `python -m py_compile` passed on the touched Python modules/scripts
- [x] `python scripts/split_tae_eae.py -h` works
- [x] `python scripts/rf_train_classify.py -h` works
- [x] config sourcing resolves `NOVA_REPO` to the active repo/worktree
- [x] `results/split_tests` is intentionally left untracked and ignored

Notes:
- CNN entry-point `-h` smoke tests did not return within a short timeout on
  this environment, but syntax compilation passed and no specific runtime error
  was exposed during the check.

## Merge-Time Checklist

- [ ] Confirm `main` is up to date before merging.
- [ ] Recheck `git diff main...mixed_branch` for any last-minute surprises.
- [ ] Merge the branch and resolve any conflicts in:
  - `configs/paths/nova_paths.nersc.sh`
  - `scripts/README.md`
  - `docs/project_state.md`
  - `training_labels/README.md`
- [ ] After merge, source the NERSC config and verify:
  - `NOVA_REPO`
  - `NOVA_DATA`
  - `PYTHONPATH`
  - `NOVA_TRAIN_CSV`
- [ ] Sanity-check that `view_modes_csv.py` still plots cleaned datcon tails.

## Post-Merge Follow-Ups

- [ ] Retrain the good/bad classifiers on the updated TAE-side list
  (`training_labels/tae_like.csv`) if that becomes the new canonical training
  pool.
- [ ] Decide whether to expose the splitter directly inside
  `scripts/sort_shot.py` or keep it as a separate preprocessing step.
- [ ] If needed, prune or archive intermediate audit CSVs once the new workflow
  is stable.
