# D46, R42 and F62 production rules run, October 10, 2026

Completed and installed the three explicitly requested released G shots using
`sort_shot_mixed.py --method rules` and frozen `tae_rules_production_v13`.
The new directories are under the established `sort_outputs/` root; no previous
output directory was replaced. **E205059A01t025 remains on hold and was not run.**

| Shot (omit `nstxu`) | Input modes | BAD | Selected GOOD | EAE-like |
|---|---:|---:|---:|---:|
| G121123R42 | 1,003 | 189 | 13 | 801 |
| G142301D46 | 571 | 118 | 11 | 442 |
| G142301F62 | 695 | 199 | 5 | 491 |
| Total | 2,269 | 506 | **29** | 1,734 |

All **29 selected GOOD modes are TAE-like and await visual review**.
[good_tae_final_batch.csv](good_tae_final_batch.csv) is the combined viewer
list; each installed shot also has its `good_tae_final.csv`.
There were 29 GOOD modes before and after duplicate processing, so none
was removed as a duplicate.

```tcsh
python viz/view_modes_csv.py audits/released3_rules_20261010/good_tae_final_batch.csv \
  --base_dir "$NOVA_DITW_ROOT"
```

## Review cross-check and verification

- All **21 R42/F62 modes** Elena judged numerical or axis-spiked are
  automatically BAD. No manual overrides were needed.
- All 23 D46 N2 correspondence-review modes are also automatically BAD.
  The two potentially acceptable examples, **N2/2650 and N2/2817**, both
  trigger `BAD_AXIS_SPIKE`. Their assessment did not request GOOD overrides.
  The [44-mode comparison](reviewed_tae_mode_results.csv) retains the user
  review context, fingerprints and actual rule outcomes.
- All 2,269 raw files are finite, including gamma_d, and nr=201. There are
  29 populated N groups; D46/N3 remains empty as previously recorded.
  Paired continua load successfully. All input/output memberships and
  mode/continuum fingerprints were verified.
- Routing produced 496 TAE-like, 39 mixed and 1,734 EAE-like modes, with zero
  INVALID or final REVIEW. All 535 TAE-side candidates have complete gate
  severity; no resolution warnings or duplicate-ranking fallbacks occurred.
  Production survivor promotion and severity ranking use the frozen preset.
- Source/configuration hashes, exported decision lists and installed output
  trees were verified. Training labels, known-invalid scopes, prior accepted
  manifests and previous shot outputs were preserved; no AI run was made.

The earlier [D46 release](../continuum_release_d46_20261008/README.md) and
[R42/F62 release](../continuum_release_r42_f62_20261010/README.md) remain the
scientific disposition records. Sorting does not erase continuum offsets,
log-coverage limits or the separately recorded potential EAE findings.

[Per-shot counts](shot_summary.csv), [input checks](preflight.json),
[verification](verification.json), [installation receipt](publication.json)
and [inventory changes](inventory_update.json) record completion. Per-shot
logs, fingerprints and staged outputs remain under ignored
`outputs/review_released3_rules_20261010/`.

## Current inventory

There are **184 processed post-training shots plus 14 active training shots
(198 cases)**. Installed processed outputs contain **8,366 selected GOOD
modes**, including 6,551 in 126 shots awaiting GOOD-list review.

[Two entries remain unprocessed](remaining_unprocessed.csv):

- **Held:** `nstxuE205059A01t025`, requiring N1/N2 recalculation.
- **Empty:** `nstxu_202806`.

No released shots remain awaiting processing. The three new shot entries
are marked `sorted_rules_pending_review`; their selected GOOD lists are
not yet visually approved.

## Reproduction

From the repository root in the configured NOVA environment:

```tcsh
python audits/released3_rules_20261010/run_batch.py stage \
  --data-root "$NOVA_DITW_ROOT" --rules-root /path/to/sort_outputs \
  --runtime-dir outputs/review_released3_rules_20261010 --workers 3
python audits/released3_rules_20261010/run_batch.py publish \
  --data-root "$NOVA_DITW_ROOT" --rules-root /path/to/sort_outputs \
  --runtime-dir outputs/review_released3_rules_20261010
python audits/released3_rules_20261010/record_results.py
```

These dated commands require the pre-run ready inventory and absent staging/
destination directories; they refuse to overwrite this completed run. For
a later rerun, use the canonical sorter with a fresh output directory and
retain input/review provenance. The driver reuses the established runner
and production-output verifier; no scientific sorter code was changed.
