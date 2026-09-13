# Distributed harmonic noise: initial total-energy trial, 2026-09-13

**Superseded denominator:** the user clarified that the global rejection cut
must use the **full-domain top-two-harmonic energy reference**, as previously
agreed, to limit edge-mode dilution. The first run below mistakenly used total
mode energy. Its receipts remain historical. See the corrected
[top-two training and 39-shot pilot audit](top2_pilot39/README.md).
`scan_proposal.py` now defaults to top-two energy; `--global-reference total`
reproduces this earlier trial.

The corrected top-two setup was subsequently approved for production v12;
see the [adoption record](adoption/README.md). This page preserves the initial
experiment's measurements and conclusions at the time.

The user proposed a continuum-independent branch for simultaneous grid-scale
variation in many harmonics. They confirmed that the effective length was
intended to be **0.03**, correcting the initial 0.3. This audit tests exactly:

| Parameter | Candidate setting |
| --- | --- |
| Window width in normalized radius | 0.05 |
| Simultaneous high-pass participation | N_hf >=4 |
| Qualifying HF / **total mode energy** | >0.005 (0.5%) |
| Qualifying HF / raw energy in the whole window | >0.05 (5%) |
| Effective length of qualifying HF energy | >0.03 |

The participation cut is inclusive; the other three cuts are strict. This
initial decision used total energy in error. Top-two-harmonic reference
ratios are recorded for comparison but do not determine this candidate.
The production gate is **not enabled**; no rules, labels, models or saved
sorting outputs were changed. These are calibration results using existing
labels and two previously inspected pilot cases, not independent validation.

## Results

- **0/575 GOOD training labels** are flagged.
- **19 BAD training labels** are flagged. Eighteen are already rule-BAD; one
  is routed to EAE-like and would not enter the TAE rejection branch.
  Consequently no current training-rule survivor changes under these cuts.
- The active list has 2,327 rows. All 575 GOOD and 1,751 BAD modes were
  measured. One pre-existing INVALID BAD input, G121123K51 N4/8769, has
  nonfinite metadata and remains explicitly recorded as INVALID. There are
  no unexpected errors. The diagnostic measures EAE-routed inputs too, but
  does not change their routing or count them as newly rejected TAE modes.
- Both pilot examples and all measured training modes have native nr=201.
  Every input fingerprint matches the previous rules audit and was rechecked
  after measurement. The active training list and source hashes stayed fixed.

For **E205042A01t022/N10/3470**, two windows satisfy every condition. The
strongest joint witness is r=0.065–0.115, with nine qualifying centers:

| Measurement | Value |
| --- | ---: |
| HF/total mode energy | 0.948831% |
| HF/window raw energy | 7.49107% |
| Effective length | 0.0388203 |

Thus the proposed additional branch would reject this current rule survivor.
**E202947A03t015/N5/2352 is not flagged by the new branch.** Its existing
BAD_CONT_CROSS_WINDOW decision remains; adding an OR rejection branch cannot
rescue it.

The nearest GOOD-labeled case in the joint-margin audit is NSTX 120113
N6/1472. Its best window has HF/total=0.4647%, HF/window=3.9374% and
length=0.02176, all below the proposed cuts. Its minimum normalized cut ratio
is 0.7255, versus >1 needed for a firing witness. This is a threshold margin,
not a probability or out-of-sample performance estimate.

## Definition and checks

The native signed high pass is `h[m,i]=(xi[m,i+1]-2*xi[m,i]+xi[m,i-1])/4`.
Pointwise simultaneous participation is
`N_hf[i]=(sum_m h[m,i]^2)^2/sum_m h[m,i]^4`; zero-power points cannot qualify.
Calculate h before applying the window/participation masks. All three stencil
samples must be inside the same window. No continuum mask, smoothing,
resampling or zero-padding is applied.

Within a window, HF energy and effective length use exactly the same selected
centers with N_hf>=4. The raw denominator includes every raw-energy node in
that window. Native full-domain trapezoidal node weights are used, with half
weight only at r=0 and r=1. `L_eff=dr*(sum e)^2/sum(e^2)` where
`e[i]=dr*sum_m h[m,i]^2` on the qualifying centers.

At nr=201 a width-0.05 window contains 11 nodes and nine eligible centers;
its maximum effective length is 0.045. Windows advance by one native grid
interval. On other resolutions, the prototype uses the largest whole native
interval count no wider than the requested window and records its actual
width; this audit provides empirical evidence only for nr=201.

Every window is checked independently with all cuts on the same population.
The reported witness maximizes the minimum of the three energy/length cut
ratios, with firing windows preferred and earliest-window tie breaking.
N_hf selects centers before these ratios are calculated. There is no
independent maximization of different conditions across windows.

Checks confirm separation of synthetic simultaneous versus sequential
four-harmonic packets, amplitude-scale invariance, and agreement with the
earlier independent pair-window calculation. Input validation and loading
reuse the shared production helpers. The experimental estimator is kept in
this audit directory; adoption would require integrating it with shared
production measurements, gate reporting and severity outputs.

## Evidence and reproduction

- [Flagged training modes](flagged_training.csv): 19 BAD-labeled inputs,
  including baseline decisions, fingerprints and witness measurements.
- [Pilot examples](pilot_examples.csv): both inspected cases and their
  proposed-branch results.
- [Summary and verification](summary.json): counts, strict threshold semantics,
  source hashes, input issues and checks.
- Full per-mode results remain ignored under
  `outputs/review_distributed_harmonic_noise_20260913/`.

From the repository with the configured NOVA environment:

```tcsh
python audits/distributed_harmonic_noise_20260913/scan_proposal.py --training-root "$NOVA_DATA" --ditw-root "$NOVA_DITW_ROOT" --global-reference total --out-dir outputs/review_distributed_noise_total_repeat
```

The default baseline is the retained training comparison from continuum-noise
v10; v11 changed duplicate ranking, preserving those morphology decisions.
Use `--baseline-csv` if relocating that historical receipt. The CLI requires
a new output directory, records input errors, and fails if unexpected inputs
or source hashes change. The corrected top-two audit also checks the wider
already-sorted pilot population before considering production adoption.
