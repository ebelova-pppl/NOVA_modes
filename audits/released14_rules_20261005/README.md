# Production sorting of 14 released shots, October 5, 2026

**Current selections:** Elena's [October 7 review](../released14_manual_review_20261007/README.md)
supersedes this automatic snapshot: 70 accepted TAEs, one newly BAD mode and
five manually classified BAEs. B37/E55 installed outputs include the
corrections. The original counts and files below are retained for provenance.

Completed and installed all 14 shots authorized by Elena from the
[release list](../continuum_release_20261005/released_shots.csv), using
`sort_shot_mixed.py --method rules --rule_config tae_rules_production_v13`.
The survivor policy, continuum preprocessing and severity-based duplicate
selection retain their frozen production settings.

## Results

- 12,250 inputs, all finite and nr=201: 1,827 TAE-like, 167 mixed and
  10,256 EAE-like.
- 1,917 BAD, 77 GOOD before deduplication, **76 selected GOOD**, all TAE-like;
  zero INVALID or final REVIEW. U27 contributes the one removed duplicate.
- All 1,994 TAE-side candidates have complete gate severity. No resolution
  warnings, duplicate-ranking fallbacks or input/output coverage failures.
- All 14 verified outputs are installed in their new directories under the
  existing `sort_outputs` root. Both live inventories now record completed
  rules sorting with `sorted_rules_pending_review`.
- Membership is **177 processed + 14 training = 191 cases**. The
  [nine remaining entries](remaining_unprocessed.csv) comprise eight
  continuum-review holds and the empty `nstxu_202806` directory.

Use the [combined 76-mode GOOD list](good_tae_final_batch.csv) or each shot's
installed `good_tae_final.csv`. [Per-shot counts](shot_summary.csv),
[input checks](preflight.json), [verification](verification.json),
[installation receipt](publication.json) and
[inventory changes](inventory_update.json) preserve the evidence.

## Outcome of the 39-mode crossing review

All 39 reviewed modes are automatically BAD. Their gate outcomes and original
fingerprints are in [reviewed_tae_mode_results.csv](reviewed_tae_mode_results.csv).
This includes E34/N2/2204, despite Elena's favorable visual assessment. Its
`BAD_CONT_CROSS_WINDOW` rejection comes from lower-boundary crossings at
r=0.08820 and 0.11164. The first window reaches A=1 and W=1 at r=0.095.
At the crossings, (A_cross, K_c) are (0.3030, 0.6414) and (0.1359, 0.6177),
so neither satisfies the low-amplitude/smoothness exception. Its weak upper
crossing at r=0.61696 does not violate the window gate.
[Saved diagnostic](e34_n2_2204_review.json) records the user's assessment
alongside the unchanged rule result. No individual manual override was applied.

## Input checks and retained scope flags

The driver checks finite raw inputs, nr=201, paired continua and source
stability. It stages full outputs under ignored
`outputs/review_released14_rules_20261005/`, then verifies every input's
coverage/fingerprint, gate severities, resolution warnings and final lists.
Verified results were installed into new shot directories under the existing
rules output root. No existing output directory was replaced.

Potential EAE issues, separate mixed findings and EAE log-coverage limits
remain recorded in the release audit. Q62 remains outside active training.
The user's aggregate morphology review does not create individual overrides.
Full selected-mode visual review remains separate from input release.

## Reproduction

From the repository root, with the configured NOVA Python environment and
site data/output roots:

```tcsh
python audits/released14_rules_20261005/run_batch.py stage \
  --data-root "$NOVA_DITW_ROOT" --rules-root /path/to/sort_outputs \
  --runtime-dir outputs/review_released14_rules_20261005 --workers 4
python audits/released14_rules_20261005/run_batch.py publish \
  --data-root "$NOVA_DITW_ROOT" --rules-root /path/to/sort_outputs \
  --runtime-dir outputs/review_released14_rules_20261005
python audits/released14_rules_20261005/record_results.py
```

The dated driver refuses existing staging/destination directories. Source
hashes and command configuration are recorded in `run_inputs.json`; full
per-mode fingerprints and per-shot execution logs stay in the runtime
directory. It reuses the previously verified batch's production checker.
