# F62 and R42: TAE crossing review, October 7, 2026

**Review follow-up:** Elena requires R42/N2 continuum recalculation and is
considering recalculating both modes and continuum for F62. Both remain held.
See the [new R42 N3–N10 diagnostic and current holds](../r42_f62_followup_20261007/README.md).
The original N1/N2 measurements and review CSV below are preserved.

[tae_review_modes.csv](tae_review_modes.csv) contains **59 flagged modes**,
one row per mode, with blank `manual_label` and `manual_reason` columns.
The list is ordered by shot, N and frequency. It includes crossing and
nearest logged singularity radii, offsets and input fingerprints.

| Shot (omit `nstxu`) | N1 modes | N2 modes | Total modes | Flagged comparisons |
|---|---:|---:|---:|---:|
| G142301F62 | 0 | 23 | 23 | 37 |
| G121123R42 | 2 | 34 | 36 | 46 |
| Total | 2 | 57 | 59 | 83 |

Selection uses the [October 5 recheck](../n1_recheck_20261005/README.md):
strictly `tae_like`, N1/N2, absolute distance to the nearest frequency-matched
logged singularity **>2 radial grid intervals**, within `0.03 <= r < 0.75`.
These are correspondence flags for visual review, not confirmed displaced
resonances or morphology labels. No rules or AI quality labels selected
these modes.

All **577 mode, continuum and available log files** in the four N groups
still match the October 5 snapshot, with unchanged mode inventories.
All 59 viewer paths resolve through the shared CSV reader. The
[receipt](receipt.json) records verification and source hashes;
[shot_summary.csv](shot_summary.csv) contains the counts.

Mixed modes and potential EAE issues remain separate in the
[earlier scoped audit](../continuum_small_review_20261005/README.md).
The missing upper EAE boundary prevents interpreting EAE-side flags as
TAE crossing offsets. Processing holds and labels remain unchanged;
no sorting was run.

From the repository root in the configured NOVA environment:

```tcsh
python viz/view_modes_csv.py audits/continuum_review_f62_r42_20261007/tae_review_modes.csv \
  --base_dir "$NOVA_DITW_ROOT"
```
