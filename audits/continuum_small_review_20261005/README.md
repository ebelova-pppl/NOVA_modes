# TAE crossing manual review, October 5, 2026

**Review completed:** Elena released all 14 shortlist shots for rules
processing. The [release record](../continuum_release_20261005/README.md)
supersedes the pending-review/hold wording below and preserves the potential
EAE flags. The measurements and original lists below remain unchanged.

The current review is **strictly TAE-like**. Among the 22 pending shots,
**14 have fewer than 10 flagged TAE-like crossing comparisons across N1+N2**.
U37 has zero, leaving **49 comparisons in 39 distinct modes from 13 shots**
for manual review. No sorter quality label or AI prediction selected these modes.

The upper EAE continuum boundary is unavailable to this diagnostic/viewer.
An EAE mode's logged singularity may correspond to that unrepresented
boundary, making the nearest-log comparison ambiguous. EAE-side findings
are therefore retained as **potential EAE crossing issues**, rather than
counted as evidence of a TAE continuum offset. This limitation does not
prove that every EAE discrepancy is explained by the missing boundary.
Mixed modes are exported separately and are not counted as TAE-like.

Here a problem crossing means the existing audit's absolute distance from
the nearest frequency-matched logged singularity is **strictly greater than
two radial grid intervals**, with `0.03 <= r < 0.75`. These are correspondence
questions, not an automatic assertion that an identified resonance moved.
The count cutoff is strictly `<10` crossings per shot, not modes per shot.

| Shot (omit `nstxu`) | N1 | N2 | Total crossings | Distinct modes |
|---|---:|---:|---:|---:|
| G133964U37 | 0 | 0 | 0 | 0 |
| G121123N22 | 1 | 0 | 1 | 1 |
| G142301E55 | 1 | 0 | 1 | 1 |
| G142301E77 | 1 | 0 | 1 | 1 |
| G142301L89 | 0 | 1 | 1 | 1 |
| G142301S94 | 1 | 0 | 1 | 1 |
| G133964R48 | 2 | 1 | 3 | 2 |
| G133964U27 | 1 | 2 | 3 | 3 |
| G121123Q62 | 0 | 5 | 5 | 4 |
| G142301E34 | 2 | 3 | 5 | 3 |
| G142301M32 | 0 | 5 | 5 | 4 |
| G142301F66 | 0 | 7 | 7 | 2 |
| G142301B37 | 1 | 7 | 8 | 8 |
| G142301U85 | 6 | 2 | 8 | 8 |
| **Total** | **16** | **33** | **49** | **39** |

## Review files

- **[39-mode TAE viewer list](tae_only_small_count_modes.csv)**: one row per affected
  mode, ordered by shot crossing count, N and frequency. Includes blank
  `manual_label` and `manual_reason` columns, source fingerprints, routing,
  maximum distance and individual crossing details. Nothing is pre-labeled.
- [14-shot TAE summary](tae_only_small_count_shots.csv).
- [All 49 individual TAE comparisons](tae_only_small_count_crossings.csv).
- [All 22 shot counts and scope flags](tae_only_all_22_summary.csv), with
  separate TAE-like, mixed and EAE-like counts and log-coverage limits.
- [Mixed-mode viewer list](mixed_crossing_modes.csv): 71 flagged comparisons
  in 47 modes across 12 of the 22 shots; separate from the primary review.
- [Potential EAE crossing issues](potential_eae_crossing_modes.csv): 349
  comparisons in 234 modes across 20 shots. These are provisional findings.
- [EAE log-coverage gaps](eae_log_coverage_gaps.csv): 70 modes with incomplete
  or conflicting logs and no interior datcon crossing, plus one E205059/N1
  mode with interior crossings but no logged singularities. These are
  coverage limits, not measured offsets.

Use the current viewer from the repository in the configured NOVA environment:

```tcsh
python viz/view_modes_csv.py audits/continuum_small_review_20261005/tae_only_small_count_modes.csv \
  --base_dir "$NOVA_DITW_ROOT"
```

Across all 22 shots, 221 flagged comparisons affect 167 TAE-like modes in
21 shots. **U37 has only EAE-side findings** (11 comparisons in nine modes),
with no flagged TAE-like or mixed comparisons. Its scope flags therefore
record potential EAE issues rather than a TAE crossing concern. It remains
unprocessed; this review does not certify the full shot or run its sorting.

## Earlier lists retained for provenance

The original all-frequency shortlist (`small_count_*.csv`) contains five
shots, 17 comparisons and 13 modes. The earlier TAE-like+mixed shortlist
(`tae_side_small_count_*.csv`) contains 13 shots, 49 comparisons and 36 modes.
Both are superseded for the current review by `tae_only_small_count_*.csv`.
The old nine-entry coverage-gap list is also retained. TAE/EAE routing still
depends on the continuum under review; narrowing scope does not validate
the excluded frequency regions.

## Provenance and next step

Counts are derived from the [October 5 N1/N2 recheck](../n1_recheck_20261005/README.md)
and cross-checked against the historical all-frequency 22-shot hold list.
Mode, continuum and available log hashes for all exported review entries were
verified against that snapshot. All viewer lists and coverage-gap lists
were parsed with the shared CSV reader: every path resolves and no labels
are assigned. [Receipt](receipt.json) records thresholds and hashes.

The user will inspect TAE morphology. If affected modes are judged numerical,
record those mode-level decisions and then reassess the corresponding TAE
hold. Preserve potential EAE flags even if TAE production processing is
later authorized. No known-invalid exclusions, training labels, production
outputs or inventory membership changed here: 163 processed + 14 training,
23 unprocessed (21 with TAE findings, U37 with EAE-only findings, one empty).

Rebuild the lists with `python audits/continuum_small_review_20261005/make_lists.py`.
