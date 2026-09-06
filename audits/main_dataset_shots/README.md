# Live DiTw shot status

This inventory lists the unique physical shot directories directly under
`$NOVA_DITW_ROOT`. It was captured on 2026-08-31 to select additional
post-training cases for rules-versus-RF-CNN comparison.

- `shot_status.csv` contains all 200 physical shot directories: 3 NSTX, 2
  legacy NSTX-U, 154 NSTX-U E, and 41 NSTX-U G cases.
- `g_shot_status.csv` is the 41-row G-shot subset for convenient filtering.
- `pilot12_selection.csv` records the reproducible 2026-09-05
  stratified-random draw, seed `20260905`, size strata, completed replacement
  sample, and the preflight exclusion that occurred before either sorter ran.
- `active_training_shot=yes` means the shot occurs in the canonical
  `training_labels/tae_like_train.csv` list. The three label-count columns
  are derived from that file.
- `post_training_checked=yes` is intentionally narrower: it marks only new
  cases already run and compared with both `rules` and `rf-cnn`. The complete
  inventory currently marks 15 post-training cases: the original E case and
  two original G cases plus the 12-shot pilot recorded in
  `pilot12_selection.csv`.

The active list contains 14 training shots and 2,390 labels. Q62 is marked
`suspended_training_q62`, not as an active training shot: its 249 reviewed
rows remain in the v3 snapshot, but it is excluded from the active list while
its upper continuum is considered suspect.

Three directory symlink aliases—`nstxuE120113P01t027`,
`nstxuE135388A02t026`, and `nstxuE141711P07t042`—resolve to the three
legacy NSTX directories and are excluded to avoid counting the same data
twice. Derived split directories, temporary/work directories, `badQ*`
quarantines, and chatgpt continuum clones are also excluded. The canonical
directory `nstxu_202806` is retained but marked `empty_no_egn` because it
currently has no populated `N#` directory containing `egn*` files.
`nstxuG142301D46` is marked `input_issue`: it was the pilot's initial
medium-size G draw, but its sole N7 mode has `gamma_d=NaN`, so it was replaced
before sorting by the next seeded candidate, `nstxuG142301E72`.
