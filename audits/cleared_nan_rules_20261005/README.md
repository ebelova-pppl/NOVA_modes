# NaN-cleared shots: production rules results, October 5, 2026

User authorized sorting the cleared pending shots and clarified that this
should use **`sort_shot_mixed.py --method rules`**, producing GOOD lists and
deduplicated representatives. All five selected shots completed production
v13 sorting, passed verification and were installed in new per-shot directories
under the established `$NOVA_SORT_OUTPUTS` root. No existing output was replaced.

## Selection and input limits

The five shots below were held for NaN metadata and have no separate pending
N1 hold. A fresh full N1--N10 scan found all **2,607 raw inputs finite**, with
nr=201 and readable paired continua. Their raw/continuum source hashes are
unchanged from the September 30 finite-input audit.

D46's NaN issue is also resolved, but it remains unprocessed for separate N1/N2
continuum review. Its main-inventory status now explicitly records that hold,
superseding the old NaN-based status.

No additional shot from the 22-case N1 review was treated as fully cleared:
six of the seven shots passing the N1 TAE-side screen retain N2 concerns.
L89 has only marginal residual distances (2.16 intervals for one EAE-side N1
crossing and 2.35 for one TAE-side N2 crossing), but still needs adjudication
against the existing review tolerance. The other 15 retain N1 evidence.
This audit does not create a new automatic validity rule from that tolerance.

The NaN-cleared group is not a declaration of complete continuum validation:
M21 retains its recorded N1/N2 log-coverage limitations, and E203655F01t030
retains the isolated N2/6049 crossing-correspondence question. Those existing
diagnostic limitations were not whole-shot N1 holds. Their production outputs
are available for visual review; no new scientific acceptance is claimed.

## Installed results

| Shot (omit `nstxu`) | Inputs | TAE-like | Mixed | EAE-like | BAD | GOOD before dedup | Selected GOOD |
|---|---:|---:|---:|---:|---:|---:|---:|
| E203653A02t017 | 324 | 219 | 4 | 101 | 181 | 42 | 41 |
| E203655F01t020 | 303 | 133 | 5 | 165 | 89 | 49 | 49 |
| E203655F01t030 | 466 | 166 | 8 | 292 | 109 | 65 | 65 |
| E205042A01t025 | 499 | 159 | 5 | 335 | 89 | 75 | 74 |
| G142301M21 | 1,015 | 268 | 44 | 703 | 300 | 12 | 12 |
| **Total** | **2,607** | **945** | **66** | **1,596** | **768** | **243** | **241** |

Zero INVALID and zero final REVIEW rows. All 241 selected representatives
are TAE-like. All 1,011 TAE-side rows have complete severity evidence, and
there are no skipped-grid warnings or duplicate-ranking fallbacks.
`accept-as-good-v1` promotes rule survivors while preserving their original
`rule_decision=REVIEW`; representatives are selected by rule severity.
No RF/CNN model was used.

The [combined GOOD list](good_tae_final_batch.csv) contains the 241 selected
modes with portable paths and fingerprints. The respective installed
`good_tae_final.csv` files contain the same selections. These modes await
visual review and have not been appended to the previously reviewed 40-shot
accepted manifest.

## Remaining processing

The main inventory now contains **163 processed post-training cases plus
14 active training shots: 177 disjoint cases**. Five new rows have
`checked_methods=rules`, `status=sorted_rules_pending_review` and
`post_training_checked=yes`. The AI comparison cohort remains 39.

**23 inventory entries remain unprocessed: 22 continuum-review shots and one
empty entry, `nstxu_202806`.** Use the [plain-text list](remaining_unprocessed.txt)
or [CSV with N1/N2 evidence and reasons](remaining_unprocessed.csv).
The 22 are review holds of differing strength, not 22 newly confirmed invalid
shots. No known-invalid registry entries were removed.

Across the recorded 163 processed cases there are 8,248 selected GOOD
representatives. The earlier reviewed 40-shot export remains at 1,726;
6,522 automatic selections from the later 123-shot cohort await visual review.
Training labels, the original 17 manual corrections, and prior installed
outputs remain unchanged.

## Verification and provenance

- [Exact selection](selection.csv), [fresh preflight](preflight.json), and
  [per-shot results](shot_summary.csv).
- [Run inputs and scientific source hashes](run_inputs.json).
- [Verified outputs and exact commands](stage_results.json).
- [Installation receipt](publication.json): five new shot directories,
  no existing outputs replaced.
- [Inventory update](inventory_update.json) and the separate
  [D46 hold-status correction](hold_status_update.json).
- [Final checks](verification.json).

Coverage, input fingerprints before/after sorting and before installation,
routing partitions, severity completeness, final-list membership, deduplication
and output hashes were checked. The frozen configuration is
`tae_rules_production_v13`, SHA-256
`5d1319910b578d9b684a367d358d5a2304a7319218fe1571b462e9ce9d3b3919`.
Scientific code/configuration and training labels were unchanged.

Full production exports/logs are staged under ignored
`outputs/review_cleared_nan_production_20261005/`. The initial requested
`sort_shot_rules.py` calibration runs are retained separately under
`$NOVA_SORT_OUTPUTS/calibration_20261005/`, with receipts in
`calibration_run/`. Their routing, gate decisions/features and severities
[match the production runs exactly](calibration_production_agreement.json).
They are not extra processed shots and do not replace the production results.

Reproduction from the repository, using new output destinations to preserve
this audit:

```tcsh
python scripts/sort_shot_mixed.py --method rules \
  --rule_config tae_rules_production_v13 \
  --shot_dir "$NOVA_DITW_ROOT/nstxuE203653A02t017" \
  --out_dir /path/to/new-results/nstxuE203653A02t017
python viz/view_modes_csv.py audits/cleared_nan_rules_20261005/good_tae_final_batch.csv \
  --base_dir "$NOVA_DITW_ROOT"
```
