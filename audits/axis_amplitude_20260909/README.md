# Width-independent near-axis amplitude audit, 2026-09-09

The user identified `nstxuG142301L94/N5/egn05w.2135E+02` as a suspected
axis-boundary artifact, then specified that the concern is large amplitude
too close to the axis, independent of width. This is a non-blind calibration
audit. No new gate, production threshold, export, or training label has been
changed.

The raw mode/datcon fingerprint, full v20 features, and REVIEW decision
exactly reproduce the current v8 output. Its dominant stored harmonic h=7
has signed amplitudes 0, 0.7104168902, and 1 at r=0, 0.005, and 0.01.
Its amplitude FWHM is 10.525004 intervals, just above the existing axis
gate's 10-interval limit. The long outer shoulder makes full FWHM an
incomplete description of the sharp inner rise. The total-energy FWHM is
5.066370 intervals, above the separate interior gate's limit of two.
No extremum or smooth-crossing exception rescues this mode: those exceptions
are unused. The exact-point crossing gate is intentionally disabled in v8;
all enabled gates pass. A boundary-condition problem remains a morphological
interpretation, not a demonstrated upstream calculation error.

The proposed independent measurement is

```text
A_axis(r_cap) = max over all stored harmonics and samples r <= r_cap of |xi_h(r)|
reject if A_axis(r_cap) > A_limit
```

Use the existing globally normalized NOVA amplitudes. There is no width,
local-maximum, gradient, or continuum-extremum condition. Radius equality is
included; amplitude equality passes. This audit uses native samples and every
inspected input has nr=201. It does not establish resolution portability.

All 950 current automatic GOOD modes in the 27-shot batch and all 575
labeled GOOD training modes were reloaded. Shot fingerprints match the saved
exports. Training modes flagged by any candidate cap were re-evaluated under
current v8 to distinguish newly rejected survivors from existing conflicts.
The labeled BAD training cohort was not screened in this targeted audit.

| r_cap | Reject above amplitude | Newly rejected batch GOOD | Newly rejected training GOOD |
| ---: | ---: | ---: | ---: |
| 0.010 | 0.5 | 1 | 1 |
| 0.010 | 0.7 | 1 | 1 |
| 0.010 | 0.8 | 1 | 0 |
| 0.015 | 0.5 | 2 | 1 |
| 0.015 | 0.7 | 1 | 1 |
| 0.015 | 0.8 | 1 | 0 |

L94 N5/2135 is the one batch change at r_cap=0.01. The training conflict
for limits 0.5/0.7 is `nstxuE202855A01t020/N1/egn01w.8188E+00`:
its axis amplitude is 0.7025981834 at r=0.01, with its global amplitude peak
at r=0.23. Its signed profile also has an abrupt inner rise and should be
reviewed before interpreting its GOOD label as definitive evidence for a
cutoff. Expanding to r_cap=0.015 at limit 0.5 additionally flags
`nstxuE204645A16t015/N1/egn01w.3712E+01` (0.5630781779 at r=0.015).

A conservative starting proposal is **A_axis(0.01)>0.8**: it catches the
user's target with no additional labeled-GOOD training conflict in this
audit. The current data do not determine a unique physical threshold.
The user has not selected/adopted a cutoff yet.

`flagged_modes.csv` contains the three unique candidates, fingerprints, and
all six threshold flags. `target_diagnostic.json` records the reproduced
axis/energy features and signed samples. `summary.json` records counts and
source hashes. `audit.py` gives a portable reproduction command in its
docstring. Full measurements and signed-profile/continuum plots are ignored
under `outputs/review_axis_amplitude_20260909/`, including
`L94_N5_2135.png`, `training_N1_8188.png`, and `E204645_N1_3712.png`.
