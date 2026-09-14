# Independent secondary edge-energy peaks

**Final adoption:** production v13 combines the body-amplitude refinement
below with the accepted interior smoothness/footprint exception. See the
[full regression and regeneration receipt](../morphology_v13_20260913/README.md)
for final changes and installed outputs. The earlier hashes and projected
counts below are historical intermediate calibrations.

**Current refinement:** secondary peaks must additionally satisfy
max_h |xi_h(r_peak)| > max_{h,r<0.9} |xi_h(r)|. All other edge cuts and the
original global branch stay unchanged. The [body-amplitude comparison](body_amplitude_comparison.csv)
checks all 15 prior changes: three pilot modes and one training GOOD-label
conflict still reject; eleven pilot modes are preserved. The initial 14-pilot
projection below describes the preceding energy-only version of uncommitted
v13. No installed outputs or training labels were changed. Current v13 config
SHA256 is `8778c722bf88162e2f5cca2a9483a7e7f91976e79bcbd5c2445f6f78bfa025eb`.

**Adopted follow-up:** the user set aside the amplitude/contrast experiments
and selected just the W>=50% row below with unchanged r>=0.97 and own-FWHM<=10.
Production v13 implements this as an extension of `BAD_EDGE_SPIKE`. All other
gate settings remain unchanged; installed shot outputs have not been rerun.
Earlier experiments and counts below retain their original audit provenance.
The [adopted-gate label-change list](adopted_label_changes.csv) contains 14
pilot GOOD-to-BAD projections (9 latest twelve) and one training GOOD-label
conflict. Exact production cuts reproduce the cached 2,214-survivor scan;
all 15 changing modes were freshly checked with the shared production edge
extractor and matching input fingerprints. All 39 saved shot CSVs remain
unchanged. These are projected rule decisions, not edits to training labels.

2026-09-13; **audit only, no production change**. The user asks why the
secondary N2/2035 peak at r=0.975 is not checked independently and how that
change would affect results. The previously accepted interior-envelope
exception remains provisional; batch regeneration is still deferred.

The subsequent user proposal tests **individual signed-harmonic peaks** with
amplitude>=0.7 and contrast>=3. Its requested 1.5-versus-2-interval width
comparison is documented in [the signed-peak follow-up](signed_harmonics.md).
The total-energy-peak measurements below describe the preceding experiment.
The latest [local-background comparison](local_background.md) uses the user's
clarified inclusive peak r>=0.95 cutoff and measures the effect of excluding
the tested signed lobe from a window centered on that peak.

## Existing behavior and motivation

`extract_edge_artifact_features` intentionally measures the single global
maximum of `W=sum_h |xi_h|^2`. `BAD_EDGE_SPIKE` requires that maximum to lie
at r>=0.97 and its full connected W FWHM to be <=10 native grid intervals.
Thus peak amplitude is implicitly significant: it is the largest W in the
mode. Secondary energy peaks have no independent rejection branch.

The original calibration avoided an axis-like individual-harmonic edge rule
because physical edge modes can contain narrow harmonics with a broader
total envelope. An August 25 all-harmonic local-peak audit also produced many
GOOD-label conflicts. That historical experiment used individual absolute
harmonics; the present experiment instead checks **local maxima of total W**.

For E203655F01t025 N2/2035, the global peak at r=0.965 misses the radial
cut. Its secondary W maximum at **r=0.975** has:

- W/global-max-W = **0.690130**;
- own-half-maximum full width = **3.311990 grid intervals**;
- strongest signed harmonic amplitude there = **0.787882**;
- amplitude/edge-background contrast = **3.393365**.

It would therefore reject under the all-local-energy-peak version of the
existing r>=0.97, width<=10 rule. The secondary half-maximum level is
0.345065: the intervening valley at r=0.970 has W=0.355904, so the connected
component includes the primary peak too. Its half-maximum endpoints are
0.9613865 and 0.9779464. The 3.312 energy width is consequently different
from this harmonic's 1.446-interval signed-amplitude width. No smoothing or
truncation at r=0.97 is used.

## Projection against current saved decisions

All scenarios keep r>=0.97 and local W FWHM<=10, with inclusive comparisons.
Each peak uses half of **its own** height, searching on the complete grid.
Local maxima and plateau/boundary handling use the existing shared helper.

| Additional requirements for that same local W peak | Newly rejected current pilot GOOD, all 39 | Latest 12 subset | Newly rejected GOOD training labels | Newly rejected BAD training labels |
| --- | ---: | ---: | ---: | ---: |
| None: all edge W peaks | 603 | 204 | 153 | 10 |
| W >= 5% of global W maximum | 99 | 61 | 17 | 1 |
| W >= 10% | 69 | 45 | 11 | 0 |
| W >= 25% | 37 | 24 | 5 | 0 |
| W >= 50% | 14 | 9 | 1 | 0 |
| Amplitude >=0.3 AND contrast >=3 | 6 | 4 | 1 | 0 |
| Amplitude >=0.3 AND contrast >=4 | 0 | 0 | 0 | 0 |
| **W >=50% AND amplitude >=0.3 AND contrast >=3** | **2** | **2** | **0** | **0** |

Narrow peaks in very weak tails explain why removing the global-maximum
requirement without replacing its significance requirement is so broad.
Even a half-global-W floor alone conflicts with one GOOD training label,
E204669M03t025 N4/1691. Its secondary W maximum is 0.985770 at r=0.970,
but amplitude contrast is only 1.519. The inspected profile has many
comparably strong narrow edge harmonics and a broad, rippled total envelope.

Contrast alone, with the 0.3 amplitude floor, conflicts with GOOD-labeled
E205052A01t022 N5/1257: contrast=3.015, but its secondary W peak is only
0.181650 of the global maximum. The mode's stronger structure lies inward.
These are conflicts with existing labels, not independent new truth labels.

The combined illustrative condition selects exactly:

| Shot / mode | Edge W peak r | W / global maximum | Amplitude | Contrast | W width, intervals |
| --- | ---: | ---: | ---: | ---: | ---: |
| E203655F01t025 N2/2035 | 0.975 | 0.690130 | 0.787882 | 3.393365 | 3.311990 |
| E203655F01t025 N3/1213 | 0.970 | 0.595203 | 0.672802 | 3.036369 | 2.086069 |

The additional N3 mode remains a review candidate. No rejection threshold
has been adopted from this comparison. Width<=4 gives the same six changes
as width<=10 for the contrast>=3/amplitude>=0.3 scenario, so shrinking that
cut alone does not resolve the training conflict. Contrast>=4 with r>=0.97
misses N2/2035: its contrast>4 primary peak is at r=0.965; the secondary
peak being tested here has contrast 3.39.

## Measurement definitions and verification

Amplitude is the largest absolute individual harmonic at the candidate W
peak, divided by the whole-mode maximum absolute amplitude. The background
is the median of `B(r)=max_h |xi_h(r)|` over **r>=0.9**, after the same
whole-mode normalization. Contrast is candidate amplitude/background. The
median includes peak samples; no hand-selected removal is performed. A
zero background leaves contrast undefined and cannot satisfy its cut.
W fractions here are **pointwise peak-height ratios**, not integrated energy
fractions and not the provisional interior exception's F_spikes.

Every condition is tested on the same peak. Independently maximizing W,
amplitude and contrast across different peaks would be an incorrect shortcut.
The combined final row was computed from the saved per-peak arrays, not by
ANDing independently satisfied per-mode flags.

The audit freshly loads all **1,646 current pilot GOOD modes**, all **565
training rule survivors** (542 GOOD labels, 23 BAD labels), and the three
pending interior-exception recoveries: **2,214 arrays**, all nr=201. The
latest twelve shots contain 698 current GOOD modes. The three pending
recoveries are unaffected by every tested edge scenario.

The active 2,327-entry training list and production source/configuration
hashes match the verified v12 training baseline. Already-BAD and EAE/INVALID
decisions cannot become survivors when adding this rejection branch, so
their mode arrays need not be recomputed. Every measured mode/datcon
fingerprint matches the saved baseline before and after. Pilot global W
peak radii and widths reproduce saved diagnostics. All 39 installed rules
CSV hashes are unchanged. These are decision projections before deduplication;
no sorter, RF/CNN pipeline, representative ranking, or batch export was rerun.

Synthetic checks cover a secondary peak outside r=0.97 while the global
peak is inside, full-grid width, inclusive thresholds, and rejection of a
false conjunction whose requirements hold at different peaks. Native plots
of both candidate modes and the two training conflicts were inspected.

## Files and reproduction

- [Base comparisons](comparison.csv) and [initial scan receipt](summary.json).
- [Combined two-mode review list](half_energy_and_contrast.csv) and its
  [cached-data projection receipt](combined_projection.json).
- [Half-height-only candidates](W_peak_ge_0.50.csv) and
  [contrast-only candidates](A_ge_0.3_contrast_ge_3.csv), including conflicts.
- All 2,214 measurements, complete per-peak arrays, unrestricted candidate
  lists and the inspected `edge_peak_comparison.png` remain ignored under
  `outputs/review_edge_secondary_peaks_20260913/`.

```text
python audits/edge_secondary_peaks_20260913/audit.py --rules-root /path/to/sort_outputs --training-root /path/to/data_mixed --out-dir outputs/review_edge_secondary_peaks_new
```

The script reproduces the initial eight scenarios and stores `peaks.json`.
The combined follow-up applies `qualifies(peak, {"w_min": 0.5, "a_min": 0.3,
"contrast_min": 3})` to those saved edge peaks. Its receipt hashes the
underlying cached inputs and records all conditions. Saved-baseline inputs
and the frozen v12 training receipt are dependencies of this bounded audit.
