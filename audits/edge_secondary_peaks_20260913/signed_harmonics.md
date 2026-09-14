# Individual signed edge peaks: 1.5 versus 2 intervals

2026-09-13; **audit only, no gate adopted or shot outputs regenerated**.
The user proposes testing sufficiently large main or secondary peaks on
individual harmonics, with amplitude>=0.7, P_edge>=3 and a narrow signed
lobe. The requested width comparison is 1.5 versus 2 native grid intervals.
This supersedes the preceding total-W proposal as the current candidate;
the older measurements remain a separate experiment.

The user subsequently clarified the background should be local to each peak
and set **r_peak>=0.95 inclusive**. See the [local-background follow-up](local_background.md)
for that comparison and the effect of excluding the tested lobe. The fixed
background results below retain their original definitions.

## Definition

On each supplied signed harmonic, find positive local maxima and negative
local minima. A single peak must satisfy all four conditions:

1. Radius strictly greater than the candidate cutoff.
2. Absolute amplitude>=0.7. Supplied whole-mode maximum amplitude was verified
   to be one for every array; there is no additional normalization.
3. P_edge=peak amplitude/background>=3, where background is the native-sample
   median of `B(r)=max_h |xi_h(r)|` over **r>=0.9**, including peak samples.
4. Signed-amplitude FWHM<=the width limit, interpolated on the complete native
   profile using the shared signed-lobe helper. Opposite-sign lobes are not
   joined, and the profile is not cropped at the radial applicability cutoff.

There is no energy sum or W-peak requirement in this candidate. The background
interval stays r>=0.9 in every radius comparison. Strict radius comparisons
use native index space, preventing rounded representations of exactly
r=0.95 from satisfying r>0.95. Width comparisons include equality.

## Effects on current survivors

All counts are **new rejections**, before representative selection. The audit
freshly measured 1,646 current pilot GOOD modes across 39 shots (698 in the
latest twelve), 565 training survivors (542 GOOD labels, 23 BAD labels), and
three pending interior-envelope recoveries. All 2,214 arrays have nr=201.
Training GOOD conflicts are existing labels, not independently confirmed
false rejections. Already-rejected and non-TAE modes cannot be recovered by
an additional rejection branch and were not remeasured.

| Strict peak radius | Maximum signed width, intervals | Training GOOD labels | Training BAD labels | Pilot 39 | Latest 12 subset |
| --- | ---: | ---: | ---: | ---: | ---: |
| >0.9 | 1.5 | 7 | 2 | 47 | 20 |
| >0.9 | 2 | 56 | 4 | 183 | 79 |
| >0.925 | 1.5 | 1 | 2 | 22 | 14 |
| >0.925 | 2 | 24 | 4 | 77 | 39 |
| >0.95 | 1.5 | 1 | 1 | 7 | 6 |
| >0.95 | 2 | 3 | 1 | 16 | 9 |
| >0.97 | 1.5 | 0 | 0 | 1 | 1 |
| >0.97 | 2 | 0 | 0 | 1 | 1 |

None of the three pending interior-envelope recoveries is affected by any
tested combination. At nr=201, 1.5 and 2 intervals correspond to radial
FWHM 0.0075 and 0.01. These are grid-relative spike-width diagnostics; this
audit does not establish behavior at another radial resolution.

Both limits reject E203655F01t025 N2/2035. Its individual qualifying peaks are:

| Peak r | Amplitude | P_edge | Signed FWHM, intervals |
| --- | ---: | ---: | ---: |
| 0.965 | 1.000000 | 4.306947 | 1.163304 |
| 0.975 | 0.787882 | 3.393365 | 1.445691 |

The secondary r=0.975 peak is checked independently and catches the mode
even with r>0.97. It is the only newly rejected mode at that radial cutoff
for either requested width limit.

## Contrast limitation and review cases

The fixed outer-interval median can be very small when a mode decays before
r=0.95. For GOOD-labeled 135388 N8/5708, the r=0.920 peak has amplitude
0.791274 and width 1.444894, but the r>=0.9 background is only 0.001390:
P_edge=569.24. The median B over r=0.90--0.93 is instead 0.591701. The
inspected signed profiles show the strong structure followed by a nearly
zero outer tail. Thus this contrast need not measure enhancement over the
structure immediately surrounding an earlier peak. The alternative interval
is illustrative only; no adaptive-background rule was calibrated here.

The seven training GOOD-label conflicts at r>0.9 and width<=1.5 are:

- 135388 N8/5708, N8/6258, N8/8814 and N9/6387.
- 141711 N3/5246, N8/1470 (`E+03`) and N9/1010 (`E+03`).

Only 141711 N3/5246 remains a GOOD-label conflict at r>0.95, width<=1.5:
r=0.955, amplitude=0.778738, P_edge=4.123181, width=1.381416.
Its signed profile has a broad inner footprint and a separate narrow outer
lobe. The existing label should be reviewed; it is not changed by this audit.
Using width<=2 at this radius adds GOOD-labeled E202855A01t020 N8/1512
(`E+01`) and 204202 N6/1899 (`E+02`).

The seven pilot candidates at r>0.95, width<=1.5 are:

- E202926A03t025 N9/6928 (`E+01`).
- E202944A02t021 N7/1180 (`E+02`).
- E203655F01t025 N2/2035, N6/1881, N6/2068 and N8/2404 (all `E+02`).
- E204944A01t017 N6/8638 (`E+01`; outside the latest twelve).

The 1.5 limit is considerably more conservative than 2. Before extending
the gate inward to r>0.9, review these conflicts and the meaning of its
background estimate. A restriction to r>0.97 is a selective option on this
sample, but its single new rejection does not establish general separation
of physical modes from numerical artifacts. No radius or width is adopted.

## Evidence and reproduction

- [Comparison](signed_harmonics/comparison.csv) includes 24 radius/width
  combinations, with widths 1, 1.25, 1.5, 2, 3 and 4 intervals.
- [Review candidates](signed_harmonics/review_candidates.csv) contains the
  union of r>0.9/width<=1.5 and r>0.95/width<=2 candidates, plus the separate
  r>0.97 witness where applicable. There are 65 unique modes and 68 witness
  rows. The `scenarios` column identifies the cuts satisfied by that witness.
- [Verification receipt](signed_harmonics/summary.json) preserves thresholds,
  hashes, native-grid counts and synthetic checks.
- Full native measurements and both r>0.9 width candidate lists remain
  ignored in `outputs/review_edge_signed_peaks_20260913/`. The inspected
  `signed_peak_comparison.png` compares N2/2035, 135388 N8/5708, 141711
  N3/5246 and E202926A03t025 N9/6928 using signed harmonics and B(r).

Mode/datcon fingerprints match before and after measurement. The parent
manifest verifies the production-v12 source/configuration and training-list
baseline. All 39 saved rules CSV hashes are unchanged. Synthetic checks
verify strict radius boundaries, signed-lobe separation with analytical
width, inclusive width limits, and the same-peak conjunction. All measured
backgrounds are positive. Production rules, labels and ranking are unchanged.

```text
python audits/edge_secondary_peaks_20260913/signed_harmonics.py --baseline-dir outputs/review_edge_secondary_peaks_20260913 --out-dir outputs/review_edge_signed_peaks_new
```

The preceding audit's explicit path manifest is required. Use an empty output
directory; the script refuses to overwrite an existing experiment.
