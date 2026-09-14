# Local background for boundary peaks

2026-09-13; **audit only; production v12 and installed outputs unchanged**.
The user clarifies that the radial condition applies to the peak location,
sets it to **r_peak>=0.95 inclusive**, and proposes calculating the background
in a window around each peak. The preceding experiment already applied its
radial cut to the peak, but used a fixed r>=0.9 background interval. These
new results use both the corrected background and inclusive radial cutoff.

## Peak exclusion and background definition

The earlier statistic was a **median**, not a mean, and included peak samples.
This follow-up compares including versus excluding the **tested** peak's
full signed-FWHM interval from the background samples. It does not subtract
an estimated peak shape, zero the excluded samples, remove an entire
harmonic, or remove every neighboring peak.

For a candidate peak on harmonic h at r_p:

```text
B(r) = max_h |xi_h(r)|
window = [r_p - window_width/2, r_p + window_width/2] intersected with [0, 1]
background = median B over available window samples outside the tested lobe
P_edge = |xi_h(r_p)| / background
```

The local window can extend below r=0.95. Near r=1 it is clipped to the
native domain; it is not shifted inward or extrapolated. Exclusion removes
the radial samples inside the interpolated full signed-FWHM endpoints from
B, so all harmonics at those radii are omitted from the background estimate.
Other lobes, including a larger neighboring peak when testing a secondary
peak, remain unless their radial samples overlap the tested interval. The
median reduces sensitivity to such remaining peaks but is not invariant to
them. Both the mean and median are recorded; **only the median** is used in
the tested contrast condition. No local-background definition has yet been
adopted into a production gate.

## Native-mode comparison

Fixed cuts: peak r>=0.95, amplitude>=0.7, P_edge>=3 and signed-lobe FWHM<=1.5
or <=2 native grid intervals. Every condition must hold for the same signed
local extremum. All 2,214 preceding survivor arrays were freshly loaded,
including candidates that failed the old fixed-background contrast.

| Total window width | Exclude tested lobe | Signed width<= | Training GOOD labels newly rejected | Training BAD labels newly rejected | Pilot 39 newly rejected | Latest 12 subset |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.05 | No | 1.5 | 1 | 1 | 5 | 5 |
| 0.05 | Yes | 1.5 | 1 | 1 | 8 | 6 |
| 0.05 | No | 2 | 5 | 1 | 10 | 6 |
| 0.05 | Yes | 2 | 6 | 1 | 16 | 8 |
| 0.10 | No | 1.5 | 1 | 0 | 11 | 9 |
| 0.10 | Yes | 1.5 | 1 | 1 | 11 | 9 |
| 0.10 | No | 2 | 8 | 0 | 23 | 15 |
| 0.10 | Yes | 2 | 8 | 1 | 24 | 15 |

These are projections before deduplication against 1,646 current pilot GOOD
and 565 training survivors (542 GOOD labels, 23 BAD labels). The three
pending interior-envelope recoveries are unaffected by all eight settings.
Existing GOOD labels are review conflicts, not independently established
false positives. All measured arrays have nr=201; no additional amplitude
normalization is applied. All measured background estimates are available
and positive.

For **E203655F01t025 N2/2035**, total window width=0.05:

| Peak r | Amplitude | Signed width, intervals | Median with peak | P with peak | Median excluding tested lobe | P excluding tested lobe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.965 | 1.000000 | 1.163304 | 0.297545 | 3.360834 | 0.278202 | 3.594517 |
| 0.975 | 0.787882 | 1.445691 | 0.258858 | 3.043685 | 0.245520 | 3.209027 |

Both peaks still independently qualify at width<=1.5, even when included
in the median. Each window contains eleven native samples; exclusion removes
one sample here, leaving ten. For the secondary test the primary r=0.965
peak is still present. The excluded-lobe **means** are 0.362555 and 0.343120,
larger than the medians, illustrating why mean and median must not be used
interchangeably at the same numerical contrast cut.

The one GOOD-label conflict at width<=1.5 remains **141711 N3/5246**:
r=0.955, amplitude=0.778738, signed width=1.381416. Its P_edge is 4.74472
for window=0.05 with tested-lobe exclusion. A local background does not
remove this conflict. The eight pilot candidates under those settings are:

- E202806A02t025 N3/8510 (`E+01`).
- E202926A03t025 N9/5572 (`E+01`), via a peak at **exactly r=0.95**.
- E203655F01t025 N2/2035, N6/1881, N6/2068, N6/2613 and N8/2404 (`E+02`).
- E204944A01t017 N6/8638 (`E+01`).

## Evidence and reproduction

[Review CSV](local_background_review.csv) contains the eight pilot candidates,
one GOOD-label conflict and one newly rejected BAD-label training survivor
for window=0.05, lobe exclusion and width<=1.5. The [receipt](local_background_summary.json)
contains all eight comparisons, both N2 peak measurements, hashes and checks.
The complete per-peak arrays and all-scenario review list remain ignored in
`outputs/review_edge_local_background_20260913/`.

Synthetic checks cover inclusive radius, windows extending below the peak
cutoff, domain clipping, exclusion and preservation of neighboring peaks.
Raw mode/datcon fingerprints were verified before and after measurement;
production source/configuration/training baseline and all 39 saved rules
CSV hashes match. No sorting, ranking, label changes or batch regeneration.

```text
python audits/edge_secondary_peaks_20260913/local_background.py --baseline-dir outputs/review_edge_secondary_peaks_20260913 --out-dir outputs/review_edge_local_background_new
```

Use an empty output directory and the preceding audit's explicit input
manifest. The 0.05 and 0.10 window widths and tested-lobe exclusion remain
calibration choices; the user has specified the inclusive peak r>=0.95 cut.
