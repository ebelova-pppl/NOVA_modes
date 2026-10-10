# Last three held shots: N1/N2 recheck, October 10, 2026

**The available results do not yet clear all three shots.** R42 and F62 have
replacement N1/N2 modes and updated N2 continua. E205059A01t025's mode files,
continua and main logs are byte-identical to October 5 despite newer mode/log
timestamps. Its previous crossing problems remain. All **470 current N1/N2
mode files are finite**, including `gamma_d`, and have nr=201. All six groups
completed without input/group errors; source hashes and inventories were
verified stable. This check covers N1/N2 only and does not run production sorting.

## Review lists

- [21 changed-input TAE-like/mixed modes](changed_tae_mixed_review_modes.csv):
  R42 has 16, F62 has five. This is the useful new review list.
- [All 51 flagged TAE-like/mixed modes](tae_mixed_review_modes.csv): also
  includes the 30 E205059A01t025 cases whose mode/continuum inputs are unchanged.

Each row includes the type, crossing/nearest-log radii, offsets, exact input
fingerprint and blank manual label/reason fields. The new list contains
14 TAE-like and seven mixed modes; the complete list has 31 TAE-like and
20 mixed modes. Every selected path was checked with the shared viewer reader.

```tcsh
python viz/view_modes_csv.py audits/last3_recalculated_20261010/changed_tae_mixed_review_modes.csv \
  --base_dir "$NOVA_DITW_ROOT"
```

## Findings

Counts below combine TAE-like and mixed modes. A flag is an interior
crossing-to-nearest-frequency-matched-log distance greater than two grid
intervals, not proof that an identified physical resonance is displaced.

| Shot (omit `nstxu`) | N | Mode count, old → current | Continuum changed | Flagged / measured crossings, old → current | Current flagged modes |
|---|---:|---:|---|---:|---:|
| E205059A01t025 | 1 | 266 → 266 | No | 26/26 → 26/26 | 26 |
| E205059A01t025 | 2 | 52 → 52 | No | 6/12 → 6/12 | 4 |
| G121123R42 | 1 | 363 → 41 | No | 3/5 → 23/24 | 14 |
| G121123R42 | 2 | 85 → 50 | Yes | 48/55 → 2/30 | 2 |
| G142301F62 | 1 | 68 → 19 | No | 0/9 → 2/9 | 2 |
| G142301F62 | 2 | 51 → 42 | Yes | 37/50 → 6/49 | 3 |

Changed mode/continuum populations make these before/after summaries, not
one-to-one crossing comparisons.

**E205059A01t025:** all 318 N1/N2 mode payloads and both continua match the
old snapshot. Both `out_go` files also match; each `out_go_prev` now duplicates
its corresponding main log. The unchanged N1 TAE/mixed comparisons have a
median absolute offset of 14.59 grid intervals. No changed mode/continuum
results are visible here; timestamps alone cannot verify a corrected run.
The existing N1/N2 recalculation hold remains.

**R42:** all current N1/N2 mode payloads are new or changed. The N2 continuum
was updated on October 9 and agreement improved substantially: **zero of 21
strictly TAE-like comparisons are flagged**. Its two remaining flags are
mixed N2/6087 and N2/6142, just beyond the cutoff at 2.02 and 2.10 intervals.
However, the replacement N1 modes have 23/24 flagged TAE/mixed comparisons
in 14 modes, with median absolute offset 3.17 intervals and maximum 19.11.
The shot remains held for review of this new evidence; this does not infer
a new N1 recalculation order. Elena's earlier acceptance of n>2 is retained.

**F62:** all current N1/N2 modes are new or changed, and the N2 continuum
was updated on October 9. N1 has two flags: **1186 (3.27 intervals)** and
**7296 (2.08)**. N2 improves to 6/49 flagged comparisons, concentrated in
**1004, 1139 and 1141**, at 15.66–29.02 intervals; these may represent missing
counterparts rather than an identified displaced resonance. All five modes
are TAE-like. The shot remains held for their review. Its lower N2 continuum
is still relatively flat at `0.01 <= r < 0.3`: frequency range 7.278–7.493,
versus 7.349–7.495 previously. Flatness alone establishes neither validity nor
the cause of a mode's morphology; the values are in
[f62_inner_boundary.json](f62_inner_boundary.json).

## Coverage, provenance and disposition

All current TAE-like/mixed modes with interior crossings have complete
frequency-matched log records. E205059A01t025 has 24 modes with frequency-matched
empty singularity records across all types; none has an interior TAE/mixed crossing.
There are no missing-frequency, incomplete or conflicting matches in the
current six groups. Potential EAE issues remain separate in
[alignment_summary.csv](alignment_summary.csv), including the remaining R42
EAE flags. The upper EAE boundary is unavailable; these findings do not clear
or reject EAE modes.

[source_comparison.csv](source_comparison.csv) records mode replacements and
continuum changes; [auxiliary_comparison.csv](auxiliary_comparison.csv) records
log/continuum hashes and local timestamps. Raw checks are in
[input_summary.csv](input_summary.csv) and [invalid_inputs.csv](invalid_inputs.csv).
Full diagnostic coverage, crossings and fingerprints are preserved in
`mode_coverage.csv`, `crossing_offsets.csv`, `metadata.json` and
`raw_input_metadata.json`; [receipt.json](receipt.json) records verification
and the inventory updates. The unchanged diagnostic uses the shared continuum
loader, exact binary-header omega-squared/log matching with relative tolerance
1e-12, and `0.03 <= r < 0.75`. No RF/CNN predictions or morphology labels enter
the screen. Reproduce measurements with:

```tcsh
python audits/n1_database_alignment_20260910/check_database.py \
  --data-root "$NOVA_DITW_ROOT" \
  --inventory audits/last3_recalculated_20261010/selection.csv \
  --out-dir outputs/review_last3_recalculated_20261010
```

[Five entries remain unprocessed](remaining_unprocessed.csv): D46 ready,
these three held, and empty `nstxu_202806`. R42 now has
`recalculation_review_pending` after the replacement; E205059A01t025 retains
`continuum_recalculation_pending` and F62 retains `recalculation_review_pending`.
No mode labels, production outputs, training labels or known-invalid scopes
changed. Membership remains 181 processed plus 14 training shots, with 8,337
selected GOOD modes.
