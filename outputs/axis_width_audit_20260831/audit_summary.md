# Non-blind `BAD_AXIS_SPIKE` width audit

## Scope and method

This is a non-blind calibration audit, not an independent validation. It uses
all 2,390 rows in the active Q62-free 14-shot
`training_labels/tae_like_train.csv` and the synchronized mode/continuum files
under the current Flux training-data root. One labeled-BAD K51 input remains
invalid because its stored `gamma_d` is non-finite, leaving 2,389 evaluable
labels.

Every mode was recomputed with production-v2 rules. Only
`axis_width_max_grid` changed among 2, 3, 5, and 10; released axis cases were
run through every later production gate. The width-10 result exactly
reproduces the documented production-v2 matrix. The frozen configuration is
`tae_rules_production_v2`, SHA-256
`7d31bd84466486f0c374372b489f04816c4f745503ef0328cb514c6ef3d7516f`.
The 31 current K70 `BAD_AXIS_SPIKE` modes were evaluated separately as an
unlabeled post-training check.

## Training-set counterfactuals

| Axis width | Axis BAD: GOOD / BAD labels | All gates BAD: GOOD / BAD labels | Final REVIEW: GOOD / BAD labels | Primary-reason changes from width 10 |
|---:|---:|---:|---:|---:|
| 2 | 5 / 466 | 26 / 1,750 | 550 / 63 | 154 |
| 3 | 7 / 540 | 28 / 1,759 | 548 / 54 | 78 |
| 5 | 7 / 589 | 28 / 1,767 | 548 / 46 | 29 |
| 10 | 8 / 617 | 29 / 1,775 | 547 / 38 | 0 |

Of the 625 width-10 axis rejections, 471 already contain a qualifying peak no
wider than two grid intervals. Another 76 first qualify between widths 2 and
3, 49 between 3 and 5, and 29 between 5 and 10.

At width 2, 154 current axis rejections change primary reason: three labeled
GOOD and 151 labeled BAD. Later gates catch 126 of those BAD modes: 75 by
`BAD_GRID_SCALE_SPIKE`, 37 by `BAD_CONT_CROSS_WINDOW`, eight by
`BAD_GRID_SCALE_PACKET`, and six by
`BAD_INTERIOR_UNRESOLVED_ENVELOPE`. The remaining 25 BAD labels and all three
GOOD labels become `REVIEW`.

The three recovered GOOD modes are:

- `nstxuE205052A01t022/N6/egn06w.8889E+01`, axis width 2.971;
- `nstxuG121123K51/N10/egn10w.4437E+02`, width 5.581 and qualifying continuum
  extremum alignment;
- `nstxuG142301W29/N5/egn05w.2505E+02`, width 2.391 and qualifying continuum
  extremum alignment.

Their signed plots support treating the near-axis lobes as resolved parts of
the mode rather than detached spikes. Conversely, the five labeled-GOOD modes
that still fire at width 2 show isolated one/few-grid-interval axis features;
their labels should be re-reviewed rather than used to relax the sharp-spike
branch.

A pure width-2 production replacement is not safe. Non-blind signed inspection
of all 25 newly retained BAD labels found many modes with obvious axis-dominated
peaks, detached near-axis lobes, or short axis packets whose individual FWHM is
slightly greater than two intervals. Width alone cannot retain the desired
coherent lobes without also releasing these structures.

## Continuum-extremum exception audit

There are 68 width-10 axis rejections satisfying the current extended
extremum geometry (`ext_dr<=0.02` and `0<=ext_df_gap<=0.04`): 66 BAD labels
and two GOOD labels. A blanket extremum exception is unsafe:

- with the width-10 gate, it would recover two GOOD labels but newly retain 17
  BAD labels after later gates;
- with the width-2 gate, it would recover no additional GOOD label and newly
  retain seven BAD labels.

The more targeted hybrid tested here is:

```text
IF an axis peak has width <= 2:
    BAD_AXIS_SPIKE for every morphology family
ELSE IF a peak has width <= 10 AND the mode is not extremum localized:
    BAD_AXIS_SPIKE
ELSE:
    continue to later gates
```

Against the active labels this hybrid recovers the two extrema-localized GOOD
modes above. Later gates catch 15 of the 25 affected BAD labels, but ten BAD
labels become REVIEW. The resulting evaluated-label matrix is 1,765/1,813 BAD
labels rejected and 549/576 GOOD labels retained. Its agreement is 96.86%,
below production-v2's 97.20%. Visual inspection confirms that continuum
alignment does not explain away the axis packets in several of those ten BAD
modes.

## K70 check

K70 has 31 width-10 axis rejections. A global width-2 limit leaves 22 as axis
BAD; six released modes are caught by later gates and three become REVIEW:

- `N1/egn01w.2987E+02`, axis width 3.008;
- `N6/egn06w.3129E+02`, width 4.181;
- `N7/egn07w.2651E+02`, width 2.534.

The signed diagnostics show that the N1 mode is axis dominated and the N6 mode
has a detached near-axis spike. Only N7/2651 has the coherent signed structure
that motivated this audit. The targeted hybrid retains the broad axis gate for
the non-extremum N1 and N6 cases; two other extremum-aligned K70 releases are
caught by later gates, so N7/2651 is the hybrid's only new K70 survivor.
`N10/egn10w.2862E+02` remains `BAD_AXIS_SPIKE` at width 2 because its measured
axis width is 1.557 intervals.

## Conclusion

The ten-grid limit is too broad to be interpreted as a spike criterion by
itself, but replacing it with a global two-grid limit would remove useful BAD
coverage. A continuum-extremum exception alone is also insufficient.

The leading next design is a two-branch gate: preserve the sharp `<=2`
interval rejection, then require explicit signed detachment, near-axis packet,
or comparable structure evidence for candidates in `(2,10]`. Extremum
localization can relax only that broader branch after signed coherence and
crossing checks; it must not rescue an unresolved sharp axis feature. The 27
training hybrid candidates plus the K70 cases form a bounded calibration set
for developing those audit-only structure features. No production rule,
configuration, label, model, or sorter output was changed by this audit.

