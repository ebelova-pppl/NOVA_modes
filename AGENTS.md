# NOVA mode classifier repo

## Repository expectations
- Canonical sorter is `scripts/sort_shot.py` with `--model`.
- Current model pipelines are `scripts/rf_train_classify.py`, `scripts/cnn_raw.py`,
  `scripts/cnn_straightened.py`, and `scripts/cnn_hybrid.py`.
- Do not hardcode NERSC or Flux absolute paths.
- Preserve feature-schema consistency between training and inference, and in visualization scripts.
- Be careful with mode-array axis ordering / flattening conventions.

## Context files to read first for nontrivial tasks
- `scripts/README.md` — detailed script inventory and collaborator-facing usage notes
- `docs/project_state.md` — current scientific/project status, model status, and migration notes

## Working style
- For nontrivial refactors, plan first and summarize the plan before editing.
- Keep diffs scoped.
- Update docs when behavior or file layout changes.
