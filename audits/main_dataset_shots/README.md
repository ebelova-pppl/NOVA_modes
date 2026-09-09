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
- `../pilot12_v5_20260908/selection.csv` records the next twelve fresh cases,
  sampled with seed 20260908 and compared using frozen v5 plus the active
  RF/raw-CNN models. Its candidate pools, one preflight replacement, input
  provenance, and comparison results are stored in that compact audit.
- `active_training_shot=yes` means the shot occurs in the canonical
  `training_labels/tae_like_train.csv` list. The three label-count columns
  are derived from that file.
- `post_training_checked=yes` is intentionally narrower: it marks only new
  cases already run and compared with both `rules` and `rf-cnn`. The complete
  inventory currently marks 27 post-training cases: the original E case and
  two original G cases, the first 12-shot pilot in `pilot12_selection.csv`,
  and the twelve fresh v5 cases completed on 2026-09-08.

All 27 checked cases were subsequently regenerated with v6 routing and the
adopted `datcon-monotonic-tail-v1` shared continuum cleanup. Both rules and
RF-CNN exports are current in the user-selected output roots, with previous
shot directories retained under `before_continuum_tail_20260908/` in each
root. The same 27 cases remain checked; their inventory notes record the
regeneration. On 2026-09-09, all 27 rules exports were regenerated again
with production v7's approved smooth-crossing exception; RF-CNN results
remain unchanged. Previous rules exports are retained under
`before_cross_window_exception_20260909/` in the rules root. Current results,
234 disagreements, and the change list are in
`../cross_window_exception_20260909/`. Its twelve new disagreements are all
already user-approved GOOD modes. Earlier repair and pilot tables remain
historical; the checked membership and training labels are unchanged.

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
`nstxuG142301M21` is also marked `input_issue`: its 197 N4 modes with
`gamma_d=NaN` were found during the second pilot preflight, before inference;
the next large-G candidate, `nstxuG142301U84`, replaced it. Active training
counts include the previously requested N9/3737 label correction.

On 2026-09-09 the user invalidated all 73 C50 N1 modes for a confirmed
continuum/eigenmode inconsistency. The shared input-validity registry enforces
that scope in both sorting methods. Both C50 exports were refreshed with
checked backups under `before_c50_n1_invalid_20260909/`; N2–N10 rows are
unchanged. The current comparison has **232 disagreements** at
`../c50_n1_alignment_20260909/current_disagreements.csv`. This subset exclusion
does not change the 27-shot checked membership. NOVA calculates eigenfrequencies
and eigenmode structure; the diagnostic log is an eigenmode-calculation log.
