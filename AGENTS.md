# NOVA mode classifier repo

## Repository expectations
- Canonical sorter is `scripts/sort_shot_mixed.py` using RF and CNN models.
- Current model pipelines are `scripts/rf_train_classify.py`, `scripts/cnn_raw.py`,
  `scripts/cnn_straightened.py`, and `scripts/cnn_hybrid.py`.
- Do not hardcode NERSC or Flux absolute paths.
- Preserve feature-schema consistency between training and inference, and in visualization scripts.
- Be careful with mode-array axis ordering / flattening conventions.

## Context files to read first for nontrivial tasks
- `README.md`
- `scripts/README.md` — detailed script inventory and collaborator-facing usage notes
- `docs/project_state.md` — current scientific/project status, model status, and migration notes

## Working style
- For nontrivial refactors, plan first and summarize the plan before editing.
- Ask clarifying questions if something in the prompt request is ambiguous.
- Check if anything might be missing from the prompt. 
- Keep diffs scoped.
- Ensure code optimizes for intent and clarity, and not speed whenever appropriate.
- The first goal is correctness and reliability. We can optimize speed later if needed.
- Do not duplicate preprocessing code between scripts.
- Avoid unnecessary restructuring or options.
- Avoid changing existing modules unless absolutely necessary. 
- Keep code modular and readable.
- Add brief comments if part of the code is not obvious.
- Keep the fallback/default logic in one shared place so it is not duplicated across multiple scripts.
- Include clear error messages if needed.
- Add clear command-line help strings.
- Include an example command in a comment or README-style note.
- Update docs when behavior or file layout changes.
- Maintain a progress file docs/project_state.md that tracks what was done, current state, blockers, and next steps.
