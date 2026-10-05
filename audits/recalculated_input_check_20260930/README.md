# Recalculated-input check, September 30, 2026

Checked the **22 pending N1-review shots and six NaN-held shots**, comprising
27 distinct shots because D46 is in both groups. Selection comes from the
current processing inventory, the September 10 priority list, and the two
secondary cases R48/U27. The four separately excluded training N1 scopes
are not part of this 22-shot selection.

**The NaN issue is resolved in the active inputs for all six shots. The N1
holds cannot yet be cleared from the files used by our sorter.** This is an
input audit; no production sorting or visual acceptance was performed.

## NaN results

Read every active `N1`--`N10/egn*` binary in the six shots: **3,205 files**.
All raw values are finite, all radial grids have nr=201, headers agree with
the N directory, and the shared mode and continuum loaders succeed. There
are **zero gamma_d NaNs and zero input errors**.

| Shot (omit `nstxu`) | Previously affected N | Previous NaN / total in N | Current total in N | Current NaN |
|---|---:|---:|---:|---:|
| G142301D46 | 7 | 1 / 1 | 93 | 0 |
| G142301M21 | 4 | 197 / 197 | 39 | 0 |
| E203653A02t017 | 6 | 17 / 29 | 27 | 0 |
| E203655F01t020 | 6 | 5 / 27 | 87 | 0 |
| E203655F01t030 | 8 | 1 / 71 | 76 | 0 |
| E205042A01t025 | 10 | 5 / 5 | 118 | 0 |

Of the original 226 offending paths, 225 are absent from the active N
directories; one M21 path remains with finite data. Recalculation changed
mode inventories, so this is not 226 repaired one-to-one mode identities.
Old files retained in nested `Out` directories are outside the sorter's
active input scan and were not counted as current inputs.

D46 still has its separate N1 hold. M21's old N1 frequency-log coverage
limitation also remains: this audit's N1 scan has no usable TAE-side interior
comparison for M21. Finite input checks do not certify mode morphology or
complete continuum alignment.

## N1 results

Compared individual SHA-256 values and mode inventories with the preserved
September 10 snapshots, then reran the same continuum/log measurements on
current N1 and N2 files in all 27 selected shots. **3,366 modes**, all nr=201,
were measured with zero group or mode-loading errors. Missing, incomplete,
and empty log records remain explicit and are not treated as alignment passes.

- **21 of the 22 N1 shots are unchanged:** every active N1 `egn*`, `datcon1`,
  `out_go`, and `out_go_prev` matches the previous content hash. Their previous
  discrepancies therefore remain. This includes the two secondary review
  cases R48/U27; their sparse/mixed evidence is not upgraded to confirmed
  invalidity by this check.
- **E205059A01t025 changed:** N1 grew from 120 to 266 modes, with 263 added
  names, 117 removed names, and three changed common files; no common mode
  payload is identical. Its continuum also changed. Nevertheless, all
  **26/26 informative TAE-side interior comparisons** exceed two grid
  intervals (previously 35/51); the median absolute distance is **14.59
  intervals**, versus 2.17 previously. Across all raw frequency ranges,
  40/40 informative comparisons exceed two intervals. The new population
  differs, so these are population summaries, not paired mode differences.
- New continuum timestamps do not establish changed contents: for example,
  N22's `datcon1` is dated September 22 but its hash equals the September 10
  snapshot. Supplementary `datcon_gf.txt` files exist, but the production
  loader reads `datcon1`. This audit does not substitute a different file
  format or infer a new radial-coordinate convention.

The unchanged 21 are G121123 N22/N75/Q62/R42, G133964 R48/U27/U37,
and G142301 B37/B85/D46/E34/E55/E77/F62/F66/F83/K79/L89/M32/S94/U85.

The established diagnostic compares shared-loader datcon crossings with the
nearest singularity in an exact-frequency NOVA eigenmode-run log record,
using `0.03 <= r < 0.75` and a two-grid-interval review tolerance. A distant
nearest log radius can mean a missing counterpart, not a measured translation
of a particular physical resonance. Close agreement alone is not proof of
correct eigenmode structure. See the original
[measurement method](../n1_training_alignment_20260910/README.md).

## Evidence and reproduction

- [Selection](selection.csv): exact 27-shot scope and issue membership.
- [NaN summary](nan_summary.csv): all ten toroidal numbers for each of six shots.
- [Original offending paths](previous_nan_files.csv): current presence/finite status.
- [Remaining invalid modes](remaining_invalid_modes.csv): header only; none found.
- [N1 before/after](n1_before_after.csv) and [N2 controls](n2_control_before_after.csv).
- [Auxiliary file hashes/timestamps](n1_auxiliary_files.csv).
- [Comparison receipt](n1_comparison_receipt.json) and [audit receipt](receipt.json).

Full mode inventories, per-crossing measurements, per-group raw source hashes,
and file-change details are kept under ignored
`outputs/review_recalculated_input_check_20260930/`. Inventory and source hashes
were rechecked for changes during each scan. Raw inputs, known-invalid scopes,
training labels, main processing inventory, manual overrides, and installed
sorter outputs were not changed. Processing membership remains 158 post-training
plus 14 training shots; a finite-input audit is not completed sorting.

Run from the repository with the configured NumPy 2.x NOVA environment:

```tcsh
python audits/n1_database_alignment_20260910/check_database.py \
  --data-root "$NOVA_DITW_ROOT" \
  --inventory audits/recalculated_input_check_20260930/selection.csv \
  --out-dir outputs/review_recalculated_input_check_20260930
python audits/recalculated_input_check_20260930/check_nan.py \
  --data-root "$NOVA_DITW_ROOT" \
  --runtime-dir outputs/review_recalculated_input_check_20260930
python audits/recalculated_input_check_20260930/summarize_n1.py \
  --runtime-dir outputs/review_recalculated_input_check_20260930
```

The compact original-path status table and receipt describe the completed
September 30 snapshot; retain them when making a later audit in a new directory.
