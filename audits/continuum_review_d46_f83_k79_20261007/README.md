# D46, F83 and K79: TAE crossing review, October 7, 2026

**Later update:** Elena completed the F83/K79 review and released those two
shots. D46's N2 files are now also absent; the D46 rows below are historical
and cannot currently be viewed. See the
[release and input-availability record](../continuum_release_f83_k79_20261007/README.md).

[tae_review_modes.csv](tae_review_modes.csv) contains **26 available modes**,
one row per mode, with blank `manual_label` and `manual_reason` columns.
Selection is strictly TAE-like, N1+N2, using the October 5 audit's absolute
nearest-log distance >2 grid intervals within `0.03 <= r < 0.75`.

| Shot (omit `nstxu`) | Available modes | Flagged comparisons |
|---|---:|---:|
| G142301D46 | 7 | 11 |
| G142301F83 | 8 | 13 |
| G142301K79 | 11 | 13 |
| Total | 26 | 37 |

The historical selection contained 27 modes/38 comparisons. D46/N1 currently
has no active `egn*` files (57 existed in the October 5 snapshot). Its flagged
`nstxuG142301D46/N1/egn01w.2532E+02` is unavailable and is recorded in
[unavailable_modes.csv](unavailable_modes.csv), separately from the working
viewer list. Its old measurements are historical; the cause of the missing
files has not been established.

All 258 mode, continuum and available log sources in the other five N groups
still match the October 5 snapshot. Mode inventories match, and all 26 viewer
paths resolve. The [receipt](receipt.json) records scope, counts and hashes.
Mixed/EAE findings remain in the earlier audit; no labels or holds changed.

From the repository root in the configured NOVA environment:

```tcsh
python viz/view_modes_csv.py audits/continuum_review_d46_f83_k79_20261007/tae_review_modes.csv \
  --base_dir "$NOVA_DITW_ROOT"
```
