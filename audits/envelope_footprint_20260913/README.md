# Narrow-envelope footprint and smoothness calibration

**Adopted in production v13 on 2026-09-13.** The user subsequently approved
activation and regeneration. The shared implementation is
`src/envelope_footprint.py`; it reproduces all 41 measurements below.
See the [full training/39-shot regression and installed outputs](../morphology_v13_20260913/README.md).
The following text and original receipt preserve the calibration's status
before that approval; no training labels were edited.

Calibration, 2026-09-13. **Provisionally accepted; production activation and
batch regeneration deferred while the latest twelve-shot review continues.**
The user explicitly requested no rerun of all shots yet. The user finds
E202926A03t025 N3/1151 and N4/3735 acceptable because they have broader,
cleaner structure than the modes that motivated the envelope gate.
This audit defines an additional exception to that gate only. Production
v12, other gates, manual labels, and installed exports remain unchanged.

## Candidate definition

Retain the existing narrow-envelope condition and continuum-extremum exception.
Additionally excuse this gate when **both** of the following hold:

1. `F_spikes < 0.50`: less than half the integrated mode energy lies in
   narrow, significant energy-peak regions.
2. `Q_local < 0.05`: local native high-pass energy is less than 5% of raw
   energy in every tested peak window, each of nominal width 0.05 in r.

Thus `current_envelope_rejection AND NOT (F_spikes<0.50 AND Q_local<0.05)`
is the proposed rejection. Equality does not grant this exception. Every
other gate remains able to reject the mode. This is a morphological exception;
it does not lower the existing extremum-clearance floor for all narrow modes.

### Energy concentration

Use `W(r)=sum_m |xi_m(r)|^2`. Find disjoint connected regions above half
the **global** maximum W, with linearly interpolated threshold endpoints.
Select regions no wider than two native radial intervals whose local peak
lies at r<=0.5, matching the existing gate's width and radial scope. Integrate
W over their union and divide by full-domain integrated W:

`F_spikes = integral_union W dr / integral_domain W dr`.

This includes the full energy within those regions, not just excess above
half maximum. Multiple significant narrow peaks are accumulated without
double counting. The global half-maximum level introduces no separate
relative peak-height cut. It does not claim to measure every small spike;
other numerical gates retain their scope.

The main-peak-only fraction, number of selected regions, and their combined
energy divided by the established full-domain top-two-harmonic reference
are saved separately. The proposed *fraction* uses total energy because it
asks whether most energy lies outside the sharp peaks. The local roughness
condition prevents distant edge harmonics from diluting a rough peak into
acceptance. No HF/full-domain rejection threshold is added here.

### Local smoothness and radial scale

Reuse the shared native signed operator `h[m,i]=diff2(xi[m,i])/4`, formed
before any radial selection. Around the global energy peak and each selected
narrow-region peak, take a window within +/-0.025 in normalized radius.
Sum `h^2 dr` only at centers whose complete three-point stencil is inside
that same window. Divide by the trapezoidal integral of all raw W in that
whole window, including its endpoints. Use the **largest** ratio among the
peak windows as Q_local, so every tested peak must pass.

This diagnostic has no continuum mask. It measures grid-scale variation,
including sharp reversals, and is not a probability or proof of numerical
origin. Windows snap inward to native samples; their actual bounds are
recorded. All measured inputs here have nr=201 and window width 0.05.
The high pass remains grid-relative; energy integrals and window lengths use
normalized radius. Unlike the existing distributed-noise gate, this audit
uses half weights at the selected window endpoints in its raw denominator.

Also record the quadrature-based energy participation length
`L_E = (sum_i w_i W_i)^2 / sum_i w_i W_i^2`. This describes energy spread,
not the distance between first and last nonzero samples. It is audit-only;
the candidate adds no third threshold for L_E.

## Measurements and decision impact

| Mode | Narrow-peak energy fraction | Largest local HF fraction | Energy participation length |
| --- | ---: | ---: | ---: |
| E202926A03t025 N3/1151 | 34.51% | 2.013% | 0.04211 |
| E202926A03t025 N4/3735 | 10.57% | 3.066% | 0.26461 |
| Original BAD Y93 N3/8350 | 59.13% | 20.022% | 0.01351 |
| Original BAD Y93 N8/9431 | 67.45% | 21.916% | 0.00715 |
| Additional candidate L94 N9/2746 | 42.86% | 3.651% | 0.06510 |

The combined exception recovers both user examples while preserving the
original Y93 rejections. Y93 N3 now additionally fails the near-axis
oscillation gate. Energy fraction alone is insufficient: previously queried
F03t017 N8/8950 has F_spikes=34.44%, but Q_local=16.39%, so the combined
proposal leaves it BAD. The earlier accepted narrow survivors remain
accepted through their existing continuum-extremum exceptions.

Across the **39 checked shots**, there are 6,012 TAE-side rows. The envelope
gate fires for 1,102, but only **34** fail no other gate. All 34 were freshly
loaded, fingerprint checked, and their complete v24 rule features reproduced
exactly. The candidate creates **three** survivors: the two user examples
and **G142301L94 N9/2746**. Other modes remain rejected by another gate.
This projects decisions before deduplication; representative ranking was not
rerun. L94 N9/2746 has two significant narrow regions and broader outer
structure in the inspected signed plot. It remains an unadjudicated candidate.

The verified v12 snapshot covers all **2,327 active training rows** and matches
the current training-list hash and production source/configuration hashes.
Only two training rows have the envelope primary reason and could possibly
be rescued; both were freshly recomputed. Neither qualifies: GOOD-labeled
H47 N7/2530 has Q_local=10.57%; BAD-labeled Y93 N8/9431 has 21.92%.
**No training decision changes** (including no newly accepted BAD label).
This uses the existing verified full-training baseline with fresh evaluation
of every potentially changing row, not a fresh recomputation of all 2,327.

In total 41 modes were freshly measured, including the original Y93 cases
and the five earlier narrow-mode examples. Fingerprints were stable before
and after; all selected pilot features match their installed exports exactly.
All 39 saved rules CSV hashes remained unchanged. A nearby sweep gives the
same three pilot changes for F<50% and Q<4% or <5%; F<40% and Q<4–5%
recovers only the two requested modes. These cutoffs are candidate calibration
values, not independently validated optima. L94 N9/2746 remains a review
candidate. The subsequent provisional acceptance does not constitute its
individual visual approval. The receipt below preserves the exploratory
status at calculation time; production implementation is still pending.

## Evidence and reproduction

- [Measurements](measurements.csv): 41 modes, fingerprints and all diagnostics.
- [Candidate new survivors](candidate_new_survivors.csv): three review entries.
- [Nearby thresholds](threshold_sweep.csv): decision counts for 15 combinations.
- [Receipt](summary.json): cohort counts, settings, sources and saved CSV hashes.
- Native signed-profile zooms and continuum panels were inspected in
  `outputs/review_envelope_footprint_20260913/comparison_profiles.png` (ignored).

Analytic checks verify two separated equal spikes: union fraction 75%,
single-spike fraction 37.5%, local high-pass fraction 37.5%, and participation
length 0.01 at nr=201. Amplitude rescaling preserves all diagnostics. Smooth
Gaussian profiles give finite results and consistent participation lengths at
nr=101/201/401. These are calculation checks, not a production test run.

```text
python audits/envelope_footprint_20260913/audit.py --rules-root /path/to/sort_outputs/before_morphology_v13_20260913 --training-root /path/to/data_mixed --out-dir outputs/review_envelope_footprint_new
```

Reproduction uses the saved v12 training comparison under
`outputs/review_distributed_noise_v12_20260913/training_comparison.csv` and
its versioned adoption receipt. Choose an empty output directory.
The script explicitly disables the two v13 changes and compares the remaining
features against the saved v12 exports, excluding only added metadata. The
[adoption reproduction check](../morphology_v13_20260913/calibration_reproduction.json)
recomputed all 41 original records successfully with the shared helper.
