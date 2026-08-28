# NOVA mode classifier repo

## Repository expectations
- Canonical sorter is `scripts/sort_shot_mixed.py`. Its default
  `--method rules` path uses the frozen deterministic production rules;
  `--method rf-cnn` keeps the RF+CNN workflow as an explicit legacy option.
- `scripts/sort_shot_rules.py` is the conservative deterministic calibration
  and audit CLI: gate survivors remain `REVIEW` there rather than being
  promoted to production `GOOD`.
- Current model pipelines are `scripts/rf_train_classify.py`, `scripts/cnn_raw.py`,
  `scripts/cnn_straightened.py`, and `scripts/cnn_hybrid.py`.
- Do not hardcode NERSC or Flux absolute paths.
- Preserve feature-schema consistency between training and inference, and in visualization scripts.
- Be careful with mode-array axis ordering / flattening conventions.

## Context files to read first for nontrivial tasks
- `README.md`
- `scripts/README.md` — detailed script inventory and collaborator-facing usage notes
- `docs/project_state.md` — current scientific/project status, model status, and migration notes

## User environments and shell syntax
- On PPPL Flux, the user's interactive shell is `tcsh`. Write copy-pasteable
  Flux commands with `tcsh` syntax such as `setenv`, unless the user explicitly
  requests another shell.
- On NERSC Perlmutter, the user's interactive shell is `bash`. Write
  copy-pasteable Perlmutter commands with Bash syntax such as `export`.
- The shell used internally by automation does not change the required syntax
  for commands presented to the user.

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

## Filesystem discovery (Perlmutter)

- Never recursively traverse `/`, `/global`, `/global/cfs`, `/global/homes`,
  `/pscratch`, `/opt`, `/usr`, or another shared top-level directory. This
  prohibition applies on compute nodes as well as login nodes.
- This prohibition includes `find`, `bfs`, `fd`, `tree`, recursive `du`,
  `rg --files`, recursive `grep`, recursive `ls`, globstar expansion, and
  recursive traversal written in Python or another language.
- Before searching, identify a bounded root inside the current workspace or a
  known project or data directory. Constrain depth and filename patterns where
  possible. If no bounded root is known, stop and ask the user.
- Locate software with `command -v`, `type -a`, `module spider`, package
  metadata, or known environment prefixes. Do not search mounted filesystems
  for executables.
- Do not disable or bypass an installed filesystem-traversal hook, and do not
  ask the user to approve an equivalent broad scan through another command.
- A compute allocation is not permission for an unbounded traversal of a
  shared filesystem. Narrow the search first; route only bounded,
  computationally substantial searches through `$perlmutter-compute`.
