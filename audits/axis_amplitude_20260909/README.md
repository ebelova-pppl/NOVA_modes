# Near-axis amplitude and energy concentration audit, 2026-09-09

**Adopted:** the user approved the combined gate. Production v9 now enables
`axis_energy_concentration`, reason `BAD_AXIS_ENERGY_CONCENTRATION`, with
the strict cuts below. It runs last, preserving earlier BAD reasons.
All 174 tests pass. All 27 v9 rules exports are regenerated and published,
with one decision change: L94 N5/2135 GOOD->BAD. The 4,187 evaluated inputs
retain every prior measured feature; no other decision, primary reason, or
selected representative changes. GOOD totals are 949 before duplicate
removal and 943 selected. Input totals remain 19,317 (772 INVALID, 14,358
EAE, 4,187 evaluated).

All 2,390 training inputs were recomputed with the gate disabled/enabled,
including labeled BAD modes: **zero decision changes** and matching input
fingerprints. The prior-feature dictionaries also agree. Both user-approved
comparison modes remain GOOD. Training truth labels are unchanged.

`adopted_changes.csv`, `adopted_shot_summary.csv`, and the header-only
`training_changes.csv` record those checks. `adoption_verification.json`
and `publication.json` record source/output/backup hashes. Old rules exports
are retained under `before_axis_energy_v9_20260909/`; RF-CNN exports are
byte-for-byte unchanged. The current disagreement list has **227 rows**, with
no additions and only L94 N5/2135 removed, recorded in
`disagreements_removed.csv`. The user's working question list is preserved.

The following sections retain the pre-adoption calibration.

The user subsequently confirmed that the two other amplitude candidates are
real modes with most of their energy away from the axis. The updated proposal
therefore combines amplitude with the fraction of total radial energy at low
r, without a width requirement. The initial amplitude-only comparison below
is retained as calibration history.

Using the same all-harmonic radial-energy proxy as existing rules,
`W(r)=sum_h |xi_h(r)|^2`, define

```text
F_inner(R) = integral_0^R W(r) dr / integral_0^1 W(r) dr
reject if A_axis(0.015) > 0.5 AND F_inner(0.05) > 0.5
```

The larger energy window includes the inner lobe beyond its steep initial
rise. The fraction threshold requires a majority of radial energy inside
the first 5% of normalized radius. Integrate piecewise-linear W using the
shared `src/cont_features.py::_energy_fraction_in_window`, with the exact
window endpoints included. No volume/Jacobian weighting is introduced.
This is an additional rejection condition; it does not exempt a mode
from an existing narrow-axis, oscillation, or other BAD gate.

| Mode | A_axis(0.015) | F_inner(0.03) | F_inner(0.05) | F_inner(0.10) |
| --- | ---: | ---: | ---: | ---: |
| G L94 N5/2135 | 1.0000 | 42.21% | 60.00% | 81.04% |
| E202855A01t020 N1/8188, training GOOD | 0.7026 | 4.50% | 7.62% | 11.72% |
| E204645A16t015 N1/3712 | 0.5631 | 2.05% | 3.79% | 6.43% |

All 950 current batch GOOD survivors and all 575 training GOOD modes were
reloaded for the joint audit. The proposed condition newly rejects only
L94 N5/2135 and preserves both user-approved comparison modes; it introduces
zero labeled-GOOD training conflicts. `combined_changes.csv` contains the
single candidate. `summary.json` records the joint definition and a small
energy-radius/fraction sweep. At R=0.05, fraction cutoffs 0.2, 0.3, and 0.5
all produce the same change; these data do not select a unique threshold.
These were the pre-adoption measurements; production adoption uses v9.
Training truth labels remain unchanged.

The user identified `nstxuG142301L94/N5/egn05w.2135E+02` as a suspected
axis-boundary artifact, then specified that the concern is large amplitude
too close to the axis, independent of width. This is a non-blind calibration
audit. The initial audit did not change a production gate, export, or truth
label; the later v9 adoption is recorded at the top.

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
at r=0.23. Its signed profile also has an abrupt inner rise; the user's
subsequent review accepts the mode because most energy is away from the axis.
Expanding to r_cap=0.015 at limit 0.5 additionally flags
`nstxuE204645A16t015/N1/egn01w.3712E+01` (0.5630781779 at r=0.015).

The initial amplitude-only proposal was **A_axis(0.01)>0.8**. The combined
amplitude/energy proposal above responds to the user's preferred distinction
and allows a lower amplitude cutoff while preserving both comparison modes.
The user subsequently adopted the joint thresholds recorded above.

`flagged_modes.csv` contains the three unique candidates, fingerprints, and
all six amplitude threshold flags, inner-energy fractions, and the joint
condition. `target_diagnostic.json` records the reproduced
axis/energy features and signed samples. `summary.json` records counts and
source hashes. `audit.py` gives a portable reproduction command in its
docstring. Full measurements and signed-profile/continuum plots are ignored
under `outputs/review_axis_amplitude_20260909/`, including
`L94_N5_2135.png`, `training_N1_8188.png`, and `E204645_N1_3712.png`.

Reproduce the pre-adoption `audit.py` against checkout `84b51aa`; current
default rule evaluation includes the gate and rejects its target. The saved
`summary.json` and target diagnostics retain their historical v8 source hashes.
`adopt_gate.py` stages all 27 canonical v9 rules runs with the active RF
representative ranker, verifies prior features and exact decision changes,
and compares all 2,390 training modes with the gate disabled/enabled.
Its docstring gives portable stage/verify/publish commands. Publication
requires unchanged source and export hashes and retains previous directories
under `before_axis_energy_v9_20260909/` in the rules output root. RF-CNN
exports are hash checked and retained.
