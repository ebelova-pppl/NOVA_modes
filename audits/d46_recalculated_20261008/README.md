# D46 recalculation and crossing check, October 8, 2026

The current `nstxuG142301D46` N1/N2 mode sets have been replaced. There are
**37 N1 and 38 N2 modes**, with no filenames shared with the previous 57/45
sets. Both continuum files remain byte-identical to the October 5 snapshot.
N4–N10 modes and continua match September 30; N3 was already empty then.
All 571 current mode files are finite, including `gamma_d`, and have nr=201.
Sources and inventories remained stable through export.

## TAE-like and mixed review

[tae_mixed_review_modes.csv](tae_mixed_review_modes.csv) contains **23 N2
modes: 17 TAE-like and six mixed TAE–EAE**, with type, continuum-crossing
radii, input fingerprints and blank manual label/reason columns.

These are **unmeasured offsets**, not confirmed crossing problems. N2's
available `out_go` is dated March 25 and has no exact-frequency match for
any of the 38 current modes; no `out_go_prev` is present. No newer singularity
log was found in the immediate N2 directory, its `Out` directory or the shot
root. The inspected `outbf`, `mpout1` and `equout` contain no singularity-log
records. Therefore all TAE-like/mixed modes with interior crossings are
included for visual inspection; offset fields are deliberately blank.

N1's only TAE-like mode, **N1/3009** (`egn01w.3009E+02`), matches the current
log. Its upper-TAE crossing is at r=0.694337, versus a logged singularity
at r=0.690000: **0.87 grid intervals**, below the two-interval flag threshold.
There are no mixed N1 modes. This is one comparison, so it cannot establish
that every aspect of N1 is correct. It is recorded in the full coverage and
crossing tables, rather than included among the review candidates.

| N | Type | Modes | Matched to log | Measured interior crossings | Flagged comparisons |
|---|---|---:|---:|---:|---:|
| 1 | TAE-like | 1 | 1 | 1 | 0 |
| 1 | Mixed | 0 | 0 | 0 | — |
| 2 | TAE-like | 17 | 0 | 0 | Unmeasured |
| 2 | Mixed | 6 | 0 | 0 | Unmeasured |

From the repository root in the configured NOVA environment:

```tcsh
python viz/view_modes_csv.py audits/d46_recalculated_20261008/tae_mixed_review_modes.csv \
  --base_dir "$NOVA_DITW_ROOT"
```

## Separate potential EAE findings

N1 has 36 EAE-like modes, all frequency matched: 49 of 58 interior
comparisons exceed two grid intervals, in **25 modes**. These are preserved
in [potential_eae_review_modes.csv](potential_eae_review_modes.csv). They
remain potential EAE correspondence issues: the upper EAE boundary is
unavailable, and nearest-log distance alone does not identify a displaced
physical resonance. N2's 15 EAE-like modes also lack matching logs.
They are recorded in the full coverage table, separately from the requested
TAE-like/mixed list.

## Evidence and disposition

- [alignment_summary.csv](alignment_summary.csv),
  [mode_coverage.csv](mode_coverage.csv) and
  [crossing_offsets.csv](crossing_offsets.csv) preserve the fresh measurements.
- [source_comparison.csv](source_comparison.csv) records old/new membership
  and hash comparisons. Mode baselines are September 30 (the N1/N2 inputs
  were still unchanged on October 5); N1/N2 continuum baselines are October 5.
- [input_summary.csv](input_summary.csv) and
  [invalid_inputs.csv](invalid_inputs.csv) record all-N raw input checks.
- [metadata.json](metadata.json), [raw_input_metadata.json](raw_input_metadata.json)
  and [receipt.json](receipt.json) preserve settings, source hashes,
  inventories and viewer-path verification.
- [remaining_unprocessed.csv](remaining_unprocessed.csv) records the five
  remaining entries. D46 stays held for N2 log/correspondence review under
  `input_update_pending`; its earlier missing/changing-mode condition is
  superseded by this stable snapshot. No production sorting, labels,
  training data or known-invalid scopes were changed. Membership remains
  181 processed plus 14 active training shots, with four holds and one empty
  entry; selected GOOD remains 8,337.

The unchanged diagnostic compares true crossings of the shared repaired TAE
boundaries to the nearest singularity in the exact-frequency NOVA log
(`omega²` relative tolerance 1e-12). Review flags use absolute distance >2
native radial grid intervals within `0.03 <= r < 0.75`; missing log matches
are coverage gaps, never zero offsets. No RF/CNN or morphology labels are
used to select this list. Reproduce the measurements with:

```tcsh
python audits/n1_database_alignment_20260910/check_database.py \
  --data-root "$NOVA_DITW_ROOT" \
  --inventory audits/d46_recalculated_20261008/selection.csv \
  --out-dir outputs/review_d46_recalculated_20261008
```
