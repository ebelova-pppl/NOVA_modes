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

The active list now contains 14 training shots and 2,327 labels (575 GOOD,
1,752 BAD). On 2026-09-10 the user confirmed N1 continuum/mode mismatch in
135388, W29, Y93 and B12; their 63 BAD rows are archived out of active
training, and their N1 scopes are registry-excluded on future sorting runs.
Main/G inventories retain the other-n training membership and updated counts.
See the [confirmation](../n1_training_alignment_20260910/README.md) and
[27-shot diagnostic follow-up](../n1_pilot_alignment_20260910/README.md).

Further production sorting is paused as of 2026-09-10 while continuum/eigenmode
consistency is reviewed and affected inputs corrected. The
[remaining-database audit](../n1_database_alignment_20260910/README.md) scanned
N1/N2 in the other 159 shots; its
[all-200 N1 table](../n1_database_alignment_20260910/all_200_n1_status.csv)
combines that scan with the earlier training/pilot snapshots. New candidate
flags are diagnostic, not confirmed registry exclusions. This audit does not
change `post_training_checked` or the sorting statuses in this inventory.
Q62 is marked
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
unchanged. That comparison had **232 disagreements** at
`../c50_n1_alignment_20260909/current_disagreements.csv`. This subset exclusion
does not change the 27-shot checked membership. NOVA calculates eigenfrequencies
and eigenmode structure; the diagnostic log is an eigenmode-calculation log.

Later on 2026-09-09 the user invalidated the whole `nstxuG133964R06` shot:
poor eigenmode structures throughout and spectra peaking at the largest
retained poloidal harmonic in some modes (cause unconfirmed). R06 now has
`status=invalid_input` in both inventories. Its `post_training_checked=yes`
and method history remain: checked records past processing, not input validity.
Both R06 output sets now retain all 610 modes as INVALID, with no usable
TAE/EAE/GOOD/BAD entries; backups are under `before_r06_invalid_20260909/`.
The persistent registry uses `ntor=*` to cover all n. That comparison
had **226 disagreements** at `../r06_input_validity_20260909/current_disagreements.csv`,
after removing six R06 entries. Checked membership remains 27, including
this now-invalid shot; active training labels are unchanged.

All 27 rules exports were then regenerated on 2026-09-09 with production v8:
the interior extremum exception requires clearance strictly greater than
0.1%, with no additional minimum width. Six GOOD modes become BAD, leaving
950 GOOD before duplicate removal and 944 selected. Those results and
**228 disagreements** are in `../extremum_floor_20260909/`; its added/removed
lists contain four/two modes. Previous rules directories are retained under
`before_extremum_clearance_v8_20260909/`; RF-CNN exports remain unchanged.
Eight already-INVALID C50 N1 files are now absent from the raw directory,
so the refreshed rules inventory has 65 C50 N1 inputs while the historical
RF-CNN export retains 73. The audit records that difference explicitly.
All 27 checked statuses and active training labels remain unchanged.

Production v9 subsequently adds the user-approved axis-energy concentration
gate: amplitude >0.5 inside r<=0.015 AND more than 50% of radial energy
inside r<=0.05. All 27 rules outputs were regenerated and verified, with
only L94 N5/2135 changing GOOD->BAD. Totals are now 949 GOOD before duplicate
removal and 943 selected. All 2,390 training decisions are unchanged.
The current **227 disagreements** are in
`../axis_amplitude_20260909/current_disagreements.csv`, with no additions and
only L94 N5/2135 removed. Previous rules outputs are retained under
`before_axis_energy_v9_20260909/`; RF-CNN outputs and checked membership are
unchanged. Both inventories record this regeneration in their notes.
