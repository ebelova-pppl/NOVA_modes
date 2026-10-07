# R42 higher-n check and remaining holds, October 7, 2026

**R42 review completed:** Elena finds the n>2 cases acceptable and confirms
that **only the N2 continuum requires recalculation**. The shot remains held
for that replacement and correspondence recheck. The
[user-review record](r42_user_review.json) preserves the reviewed mode
fingerprints and supersedes the pending higher-n review described by the
original diagnostic receipt. This acceptance does not assign sorter GOOD labels.

Elena requires recalculation of **R42/N2 continuum**. She also finds the
F62/N2 lower boundary very flat at r<0.3, with some otherwise acceptable
modes, and is considering recalculating both the modes and continuum.
R42 is now `continuum_recalculation_pending`; F62 remains held as
`recalculation_review_pending`. The cause of F62's mode structure and the
scope of a new calculation have not been established. No per-mode labels
or production outputs were changed.

## R42 N3–N10

The earlier audit covered only N1/N2. This new check used the existing
`scan_group` / `measure_shot` functions from the September 10 alignment
audits on all **912 N3–N10 modes**, all nr=201. Mode inventories and source
hashes were checked for stability during measurement. There were no input
or group errors; 911 modes have complete frequency-matched log records.

The primary screen remains strictly TAE-like, `0.03 <= r < 0.75`, with
absolute distance to the nearest logged singularity **>2 grid intervals**.

| N | Flagged TAE comparisons / measured | Flagged modes | Largest distance, grid intervals |
|---|---:|---:|---:|
| 3 | 6 / 13 | 3 | 18.53 |
| 4 | 5 / 16 | 3 | 18.26 |
| 5 | 2 / 38 | 1 | 12.19 |
| 6 | 2 / 41 | 2 | 12.08 |
| 7 | 0 / 28 | 0 | 0.99 |
| 8 | 1 / 41 | 1 | 2.07 |
| 9 | 0 / 50 | 0 | 0.92 |
| 10 | 1 / 84 | 1 | 2.07 |
| Total | 17 / 311 | 11 | |

The N3/N4 flags concern lower-boundary crossings, as does the single N5
mode. N6 has two inner upper-boundary cases. N8 and N10 each have one
marginal comparison, just above the two-interval cutoff. These are
crossing-to-log correspondence flags, not proof that an identified
resonance was displaced or that every mode in an N group is invalid.
Elena subsequently accepted the n>2 cases and specified N2-only continuum
recalculation. The original flags remain diagnostic evidence; they do not
add recalculation requirements beyond her decision.

- [Eleven-mode TAE viewer list](r42_tae_review_modes.csv), with blank manual
  review fields, crossing details and mode/continuum fingerprints.
- [Per-N summary](r42_n3_n10_summary.csv), retaining mixed and potential EAE
  findings separately. No mixed comparisons exceeded the cutoff. EAE
  correspondence remains provisional because the upper EAE boundary is absent.
- [One TAE log-coverage gap](r42_tae_coverage_gaps.csv): N10/2376 has three
  interior crossings but no exact-frequency record in the available logs;
  this is not a measured offset and is separate from the eleven-mode list.

All 12 exported paths resolve through the shared CSV reader. Full coverage,
crossings, cached groups and source hashes are under the ignored
`outputs/review_r42_higher_n_20261007/` directory. The shared diagnostic
compares full binary-header omega squared with log frequency at relative
tolerance 1e-12; no sorter or AI quality labels enter the screen.

```tcsh
python viz/view_modes_csv.py audits/r42_f62_followup_20261007/r42_tae_review_modes.csv \
  --base_dir "$NOVA_DITW_ROOT"
```

## F62, D46 and Q62

F62/N2's lower-boundary frequency ranges from **7.349 to 7.495 NOVA units**
over the 58 available native points at `0.01 <= r < 0.3`. The shared loader
changes none of those raw continuum values; the flat inner portion is not
introduced by the continuum repair. This observation does not prove the
physical validity of the boundary or explain the mode morphology.

D46 now contains **34 N1 and 18 N2 mode files**, increased from the earlier
October 7 counts of one and zero. Completion of the changing inputs and
their continuum correspondence remain unchecked, so its hold persists.

**Q62 is not awaiting production processing.** It was released and sorted
in the October 5 batch and has seven selected TAEs approved in the
October 7 GOOD-list review. Its 249 old training rows remain suspended;
potential EAE/upper-continuum concerns and mixed findings are still recorded.
Production release did not restore its training membership.

## Current remaining entries

**Later processing:** the four ready shots below have now been
[sorted and installed](../released4_rules_20261007/README.md). The four
holds and empty entry remain; the nine-entry CSV is the pre-run snapshot.

[Remaining-shot CSV](remaining_unprocessed.csv):

- **Four held:** E205059A01t025 (N1/N2 recalculation), R42 (N2 continuum
  recalculation only), F62 (consider modes plus continuum
  recalculation), D46 (input update and subsequent correspondence check).
- **Four ready:** N75, B85, F83 and K79.
- **One empty:** `nstxu_202806`.

Membership remains 177 processed plus 14 active training shots. The
[receipt](receipt.json) records measurements, source hashes, D46 availability,
Q62's installed count and exact inventory changes. No sorting, training-label
changes, manual overrides or known-invalid-registry changes were made.
