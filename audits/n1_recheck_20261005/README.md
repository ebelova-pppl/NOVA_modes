# N1 recheck, October 5, 2026

Rechecked exactly the same **22 pending N1 scopes** from September 30, with
N2 controls. The audit measures 2,900 current mode files: 1,752 N1 and 1,148
N2, all nr=201. All 44 groups completed with no input-loading or group errors.

**19 N1 continuum files have changed, dated October 2. Their mode payloads,
mode inventories and frequency logs are unchanged.** The three other N1
scopes, E205059A01t025, G133964R48 and G133964U27, have no source changes.
All N2 mode/continuum/log source sets remain identical to September 30.

The new continua substantially improve the existing diagnostic: among the
19 updated scopes, TAE-side interior comparisons beyond two radial grid
intervals decrease from **281/322 (87.3%) to 26/176 (14.8%)**. Across all raw
frequency ranges they decrease from **701/781 (89.8%) to 98/546 (17.9%)**.
Updated continua change crossing counts and TAE/EAE routing, so these are
population summaries rather than one-to-one crossing comparisons.

## Per-shot evidence

Counts below mean **comparisons beyond two grid intervals / informative
interior comparisons**, within `0.03 <= r < 0.75`. The all-frequency column
retains EAE-side evidence; its absence from the TAE-side subset does not clear
an entire N1 calculation. Shot names omit the `nstxu` prefix.

| Shot | Continuum changed | Previous TAE-side | Current TAE-side | Current all frequencies |
|---|---|---:|---:|---:|
| E205059A01t025 | no | 26/26 | 26/26 | 40/40 |
| G121123N22 | yes | 7/7 | 3/5 | 18/30 |
| G121123N75 | yes | 1/1 | 0/1 | 0/1 |
| G121123Q62 | yes | 25/28 | 1/14 | 19/97 |
| G121123R42 | yes | 54/54 | 3/5 | 16/51 |
| G133964R48 | no | 2/6 | 2/6 | 27/32 |
| G133964U27 | no | 1/4 | 1/4 | 3/6 |
| G133964U37 | yes | 6/10 | 0/4 | 0/4 |
| G142301B37 | yes | 20/27 | 1/26 | 1/27 |
| G142301B85 | yes | 21/21 | 0/6 | 0/77 |
| G142301D46 | yes | 16/19 | 1/13 | 1/13 |
| G142301E34 | yes | 4/4 | 2/2 | 2/2 |
| G142301E55 | yes | 4/6 | 1/6 | 1/6 |
| G142301E77 | yes | 14/22 | 2/12 | 11/68 |
| G142301F62 | yes | 10/10 | 0/9 | 0/11 |
| G142301F66 | yes | 6/6 | 0/3 | 0/17 |
| G142301F83 | yes | 29/31 | 2/21 | 5/33 |
| G142301K79 | yes | 12/12 | 1/8 | 1/11 |
| G142301L89 | yes | 10/10 | 0/7 | 1/39 |
| G142301M32 | yes | 11/12 | 0/5 | 0/6 |
| G142301S94 | yes | 1/1 | 1/1 | 1/1 |
| G142301U85 | yes | 30/41 | 8/28 | 21/52 |

Seven updated shots have no remaining out-of-tolerance TAE-side comparisons:
G121123N75, G133964U37, and G142301B85/F62/F66/L89/M32. Six of these also
have none among usable all-frequency comparisons; L89 has one among 39.
N75 has only one informative N1 mode, and some shots have incomplete EAE-side
log blocks. These are encouraging diagnostic results, not whole-shot acceptance.

Important remaining cases:

- **E205059A01t025:** unchanged; 26/26 TAE-side comparisons remain outside
  tolerance, with median absolute nearest-log distance 14.59 intervals.
- **N22 and R42:** 3/5 TAE-side comparisons each remain outside tolerance;
  medians are 4.87 and 5.46 intervals. Wider-frequency evidence also remains.
- **U85:** 8/28 TAE-side and 21/52 all-frequency comparisons remain outside
  tolerance. This needs branch-specific inspection despite the improvement.
- **E34 and S94:** sparse residual offsets. E34 has two comparisons at
  2.23 and 2.38 intervals; S94 has one at 3.30. Neither alone establishes a
  repeated whole-N1 failure.
- **Q62, D46 and E77:** remaining TAE-side distances are at most 2.53, 2.41
  and 2.46 intervals, respectively. Q62 and E77 have additional all-frequency
  evidence that should be considered separately.
- **B37, E55, F83 and K79:** one or two TAE-side outliers remain, some large;
  distinguish unmatched extra branches from a displaced identified resonance.
- **R48/U27:** unchanged secondary-review cases, with 2/6 and 1/4 TAE-side
  comparisons beyond tolerance, and additional EAE-side evidence.

N2 controls have not been corrected in this snapshot. Previously concerning
N75, R42 and F62 still have respectively 41/41, 48/55 and 37/50 TAE-side
interior N2 comparisons beyond tolerance. Improved N1 correspondence cannot
be used to certify their other toroidal numbers.

## Method and limits

Reused the existing shared-loader continuum/log diagnostic, with no scientific
code or threshold changes. The full binary-header frequency squared must match
an `out_go` or `out_go_prev` record to relative tolerance 1e-12. Incomplete and
conflicting blocks are excluded and separately counted; a matching empty block
with a datcon crossing is recorded for review. Each crossing is compared with
the nearest logged singularity, not an identified branch assignment. A large
distance can mean a missing counterpart. Small distances alone do not establish
physical eigenmode quality. See the [original method](../n1_training_alignment_20260910/README.md).

All raw inventories and source hashes were checked for stability during each
measurement. Byte comparisons include modes, continua and both available logs.
The main processing inventory, known-invalid registry, training labels, sorter
outputs and manual overrides remain unchanged. No new shots were sorted or
scientifically accepted. The September 30 NaN finding is historical and was
not rerun in this N1-only request.

## Files and reproduction

- [Exact selection](selection.csv).
- [N1 before/after measurements](n1_before_after.csv), including coverage counts.
- [N2 control measurements](n2_control_before_after.csv).
- [Continuum hashes and timestamps](continuum_files.csv).
- [Viewer-ready residual review list](review_modes.csv): 146 N1 modes across
  all frequency ranges with an out-of-tolerance interior comparison or an
  empty matched log despite an interior crossing; no morphology labels.
- [Receipt](receipt.json): historical/current snapshot hashes and protected files.

Full source snapshots, per-crossing offsets and file comparisons remain under
ignored `outputs/review_n1_recheck_20261005/`. This directory is separate from
both previous audits, which are preserved.

Run from the repository in the configured NOVA environment:

```tcsh
python audits/n1_database_alignment_20260910/check_database.py \
  --data-root "$NOVA_DITW_ROOT" \
  --inventory audits/n1_recheck_20261005/selection.csv \
  --out-dir outputs/review_n1_recheck_20261005
python audits/n1_recheck_20261005/summarize.py
python viz/view_modes_csv.py audits/n1_recheck_20261005/review_modes.csv \
  --base_dir "$NOVA_DITW_ROOT"
```
