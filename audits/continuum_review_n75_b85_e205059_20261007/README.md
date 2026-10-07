# N75, B85 and E205059A01t025: TAE crossing review, October 7, 2026

**Review completed:** Elena released N75/B85 because all listed modes look
junky despite confirmed N2 offsets. E205059A01t025 requires N1/N2
recalculation because some affected modes otherwise look acceptable. See the
[release and hold record](../continuum_release_n75_b85_20261007/README.md).
The original measurements and review CSV below are preserved.

[tae_review_modes.csv](tae_review_modes.csv) contains **42 flagged modes**,
one row per mode, with blank `manual_label` and `manual_reason` columns.
The list is ordered by shot, N and frequency and includes individual crossing
radii, nearest logged singularity radii, offsets and input fingerprints.

| Shot (omit `nstxu`) | N1 modes | N2 modes | Total modes | Flagged comparisons |
|---|---:|---:|---:|---:|
| G121123N75 | 0 | 16 | 16 | 16 |
| G142301B85 | 0 | 9 | 9 | 16 |
| E205059A01t025 | 13 | 4 | 17 | 19 |
| Total | 13 | 29 | 42 | 51 |

Selection uses the [October 5 recheck](../n1_recheck_20261005/README.md):
strictly `tae_like`, N1/N2, absolute distance to the nearest frequency-matched
logged singularity **>2 radial grid intervals**, within `0.03 <= r < 0.75`.
These are correspondence flags for visual review, not confirmed displaced
resonances or morphology labels. No RF/CNN or rules quality label selected
the modes.

All **591 mode, continuum and available log files** in these six N groups
still match the October 5 snapshot; mode inventories are unchanged. All 42
viewer paths resolve through the shared CSV reader. [Receipt](receipt.json)
records source hashes and verification; [shot_summary.csv](shot_summary.csv)
contains the counts.

Mixed modes and potential EAE issues remain separate in the
[earlier scoped audit](../continuum_small_review_20261005/README.md).
The upper EAE boundary is unavailable, so EAE-side flags are not included
as evidence of TAE crossing offsets. No holds, labels, exclusions or
production outputs changed, and no sorting was run.

From the repository root in the configured NOVA environment:

```tcsh
python viz/view_modes_csv.py audits/continuum_review_n75_b85_e205059_20261007/tae_review_modes.csv \
  --base_dir "$NOVA_DITW_ROOT"
```
