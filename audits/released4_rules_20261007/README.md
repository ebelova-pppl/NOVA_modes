# Production sorting of N75, B85, F83 and K79, October 7, 2026

Completed and installed the four shots authorized by Elena, using
`sort_shot_mixed.py --method rules --rule_config tae_rules_production_v13`.
Outputs are in the four new shot directories under the established
`sort_outputs` root; no existing output was replaced.

| Shot (omit `nstxu`) | Input modes | BAD | Selected GOOD | EAE-like |
|---|---:|---:|---:|---:|
| G121123N75 | 768 | 197 | 10 | 561 |
| G142301B85 | 813 | 137 | 0 | 676 |
| G142301F83 | 827 | 141 | 10 | 676 |
| G142301K79 | 599 | 222 | 0 | 377 |
| Total | 3,007 | 697 | 20 | 2,290 |

Use the [combined 20-mode GOOD list](good_tae_final_batch.csv) or each
installed shot's `good_tae_final.csv`. All 20 selections are TAE-like;
none was removed by deduplication. They await selected-GOOD visual review.
The inventories now mark the four shots `sorted_rules_pending_review`,
`post_training_checked=yes`, `checked_methods=rules`.

All **44 crossing-review modes are automatically BAD**, agreeing with the
user's aggregate morphology assessment. Their fingerprints and gate outcomes
are in [reviewed_tae_mode_results.csv](reviewed_tae_mode_results.csv).
No manual overrides or rule changes were needed. The earlier
[N75/B85](../continuum_release_n75_b85_20261007/README.md) and
[F83/K79](../continuum_release_f83_k79_20261007/README.md) releases retain
continuum correspondence findings, mixed findings and potential EAE issues.
Processing these shots does not certify the excluded EAE population or
correct the previously observed continuum offsets.

## Verification

- All 3,007 raw inputs have finite values, including gamma_d; nr=201
  throughout, with readable paired continua in all 40 N groups.
- Routing: 654 TAE-like, 63 mixed and 2,290 EAE-like. Zero INVALID,
  final REVIEW or manual BAE decisions.
- All 717 TAE-side candidates have complete gate severity. No resolution
  warnings or duplicate-ranking fallbacks. Survivor promotion and
  severity-based selection retain the frozen production configuration.
- Input/output coverage, every mode/continuum fingerprint, all exported
  decision lists, configuration hashes and installed output trees verified.
- Training labels, known-invalid scopes, prior accepted manifests and
  previous shot outputs were preserved. No AI classification was run.

[Per-shot counts](shot_summary.csv), [preflight](preflight.json),
[verification](verification.json), [installation receipt](publication.json)
and [inventory changes](inventory_update.json) record the run.
Full per-shot logs, fingerprints and staged outputs remain under ignored
`outputs/review_released4_rules_20261007/`.

## Current inventory

There are now **181 processed post-training shots plus 14 active training
shots (195 disjoint cases)**. Installed processed outputs contain **8,338
selected GOOD modes**, with 6,542 selections in 127 shots awaiting GOOD-list
review. Earlier reviewed collections remain unchanged.

[Five entries remain unprocessed](remaining_unprocessed.csv): four held
shots and one empty entry. There are no remaining released shots awaiting
processing.

- E205059A01t025: N1/N2 recalculation.
- G121123R42: N2 continuum recalculation only; higher-n cases accepted.
- G142301F62: held while considering modes-plus-continuum recalculation.
- G142301D46: input updates and correspondence check pending.
- `nstxu_202806`: empty entry.

## Reproduction

From the repository root in the configured NOVA environment:

```tcsh
python audits/released4_rules_20261007/run_batch.py stage \
  --data-root "$NOVA_DITW_ROOT" --rules-root /path/to/sort_outputs \
  --runtime-dir outputs/review_released4_rules_20261007 --workers 4
python audits/released4_rules_20261007/run_batch.py publish \
  --data-root "$NOVA_DITW_ROOT" --rules-root /path/to/sort_outputs \
  --runtime-dir outputs/review_released4_rules_20261007
python audits/released4_rules_20261007/record_results.py
```

These are the original commands with portable site paths. The dated driver
requires the saved pre-run inventory state and absent staging/destination
directories; it refuses to overwrite this completed run. For a later rerun,
use the canonical sorter into a fresh output directory and retain the review
and input-version provenance. The driver reuses the earlier batch's shared
runner and production-output checker; the scientific sorter is unchanged.
