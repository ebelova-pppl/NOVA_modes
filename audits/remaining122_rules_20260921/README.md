# Authorized 122-shot production rules batch

Status: **complete: 122 sorter runs, 118 verified and installed shots,
four input holds**.
The user authorized `sort_shot_mixed.py` for the 122
remaining candidate shots from the runtime estimate, excluding recorded
priority N1 cases, NaN-input holds, the empty entry and secondary N1 cases
R48/U27. Selection contains 120 E shots and two G shots, with an estimated
71,656 raw mode files in N1–N10. This is rules-only processing; the results
require visual review. All 122 sorter runs completed. The 118 publishable
shots cover 70,468 modes, all nr=201, with zero INVALID modes and complete
gate severity evidence. Four previously unflagged NaN-input shots are held
under the requested whole-shot exclusion policy.

## Results

| Result for the 118 publishable shots | Modes |
| --- | ---: |
| TAE-like / mixed candidates | 18,898 / 845 |
| EAE-like | 50,725 |
| Final GOOD before deduplication | 6,324 |
| Selected GOOD representatives | 6,281 |
| BAD | 13,419 |
| INVALID | 0 |

- [Selected GOOD modes from this batch](good_tae_final_batch.csv): portable
  paths and explicit `review_status=not_visually_reviewed`. Includes 6,280
  TAE-like representatives and one mixed representative, consistent with
  production's inclusion of mixed modes on the TAE side.
- [Per-shot counts](shot_summary.csv), [verification receipt](verification.json).
- [Installation receipt](publication.json), [inventory update](inventory_update.json).
- [Combined counts for all 158 processed cases](processed158_summary.csv).
- [Preserved prior manual corrections](prior_manual_corrections_check.json).
- [NaN-input files](invalid_input_files.csv), [whole-shot holds](input_holds.json).

The selected mixed mode is `nstxuE204957F05t034/N7/egn07w.3141E+02`.
Pending visual review is an inventory annotation; production GOOD/BAD
decisions retain their usual meaning.

Installed results are in the requested `sort_outputs/SHOT/` directories,
including each shot's `good_tae_final.csv`. No existing output was replaced.
The processed inventory now contains **158 post-training cases plus 14 active
training shots (172 disjoint cases)**; 28 entries remain held or empty.
Across the 158 processed cases there are **8,061 GOOD before deduplication**
and **8,007 selected representatives**. These counts exclude the training
list and the four new held shots. The new 118 await visual review; the prior
40-shot reviewed export and its 17 manual corrections are unchanged.

The held shots are:

| Shot | Invalid files | Location |
| --- | ---: | --- |
| E203655F01t030 | 1 | N8/egn08w.2428E+02 |
| E203653A02t017 | 17 | N6 |
| E203655F01t020 | 5 | N6 |
| E205042A01t025 | 5 | N10 |

Each of the 28 files has NaN `gamma_d` in its raw trailer; its other raw
values are finite. The held shots contain 1,188 files in total. Their
diagnostic sorting outputs remain in the local staging directory, but none
of their modes enters this published batch. No raw input has been edited.

The batch uses frozen `tae_rules_production_v13`, the existing compatible
Python environment, four shot workers and one numerical thread per process.
All destinations were absent before the run. No prior reviewed shot, manual
override, training label, or RF-CNN output is replaced.

## Execution and verification

- [Exact selection](selection.csv) retains the estimate's recorded input flags.
- [Source/configuration snapshot](run_inputs.json) pins the code, registry,
  inventory, training labels and prior accepted manifest for this run.
- `progress.json` records completed verification and any failures.
- Per-shot raw input fingerprints are captured before sorting and compared
  against every output row and current raw files after sorting.
- Verification requires full file coverage, nr=201, zero INVALID/runtime
  failures, complete severity evidence, no resolution warnings or ranking
  fallbacks, correct survivor-policy decisions, and consistent GOOD/BAD/EAE
  lists. A failed verification blocks installation and identifies the shot.
- `publish` requires the completed receipt, unchanged code/configuration and
  matching staged output hashes. It rechecks raw fingerprints, copies each
  shot to a temporary destination, verifies the copy and then renames it
  into the rules output root. It refuses existing destinations.

The initial batch verifier omitted the legitimate
`NO_CLOSE_FREQUENCY_CLUSTERS` outcome. Its status check was corrected, and
nine completed outputs were rechecked without rerunning the sorter or
changing scientific code/configuration. `initial_stage_results.json` and
`verifier_correction.json` preserve this history; the other four initial
failures are the genuine NaN-input holds above. `run_inputs.json` records
the original driver hash separately from the corrected verifier hash.

Full exports, raw fingerprints and logs stay in ignored
`outputs/review_remaining122_rules_20260921/`. Compact batch summaries,
selected-mode lists and receipts are retained here after verification.
The previous 40-shot reviewed cohort remains a separate preserved export.

## Recorded command

```text
python audits/remaining122_rules_20260921/run_batch.py stage --data-root /path/to/DiTw --rules-root /path/to/sort_outputs --runtime-dir outputs/review_remaining122_rules_20260921 --workers 4
```

After successful staging, use the same arguments with `publish` instead of
`stage`. The command protects previous staging directories from replacement.
For this run, `finalize_batch.py` completed the corrected verification and
excluded the documented input holds before publication.
`record_installation.py` updated the main and G-shot inventories after
successful installation, with `checked_methods=rules`,
`post_training_checked=yes` and `status=sorted_rules_pending_review` for
the 118 installed shots. The four new holds remain unchecked with
`status=input_issue`. All other inventory rows and training counts are preserved.
Historical flags remain evidence; this run does not independently certify
continuum/eigenmode alignment in cases with limited N1 information.
