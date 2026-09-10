---
name: sort-tae-like-modes
description: "Deterministically preprocess and sort one NOVA shot into TAE-like, mixed, EAE-like, invalid, BAD, REVIEW, and GOOD outputs with auditable rule reasons, an explicit production survivor policy, reusable fingerprinted manual overrides, and RF-only selection of final-GOOD frequency/structure representatives. Use for one-shot NOVA TAE preprocessing, rule-based production sorting, conservative rule calibration, reproducible output regeneration, output auditing, or explicit post-rule adjudication without CNN classification."
---

# Sort TAE-Like Modes

Process one target shot noninteractively with the repository scripts. Keep
frequency routing, deterministic decisions, manual overrides, and duplicate
ranking as separate stages.

## Run the deterministic workflow

For production sorting, use the canonical mixed-shot sorter. Rules are the
default; specify the method explicitly in saved commands for clear provenance:

```bash
python scripts/sort_shot_mixed.py \
  --method rules \
  --shot_dir /path/to/SHOT \
  --rf_model models/nova_mode_classifier.joblib \
  --out_dir /path/to/sort-output
```

The preset is `configs/rules/tae_rules_production_v10.yaml`. It pins the v22
ruleset and routing values, enables gates 1, 2, 2b, the near-axis
grid-oscillation gate, 4, 5, the interior-envelope and harmonic-incoherence
gates, the continuum crossing-tail gate, the axis-energy gate, and the final extended continuum noise gate. It explicitly disables
exact-point continuum gate 3.

Production v10 retains the v7 exception: it excuses a violating gate-4 window only when the same crossing
has strict `A_cross < 0.2 AND K_c < 0.1`, on nr=201. Interpolate each signed
harmonic to the crossing before taking its magnitude, then maximize over
harmonics. K uses the shared unscaled second-difference calculation with an
independent +/-4-grid stencil-center window. Every offending crossing must
qualify; another unexcused crossing still rejects, and all other gates retain
precedence. Equality, undefined K, or nr!=201 cannot grant the exception.
The original window maxima remain available; per-crossing measurements and
exemption flags are under `crossing_features.continuum_crossing_window_exception`.
Configure it with `--cross_window_exception_amplitude_max`,
`--cross_window_exception_k_max`, `--cross_window_exception_half_width_grid`,
`--cross_window_exception_calibrated_n_radial`, or disable it with
`--disable_cross_window_exception` in the calibration CLI. Named v5/v6
configurations explicitly disable the exception and preserve old decisions,
while new exports use the v22 audit schema. Exact older exports require their
historical checkout. The exception never skips the rejection gate on other
resolutions; the original window criteria remain active there.

Production-v10 routes `fraction_below_upper2 < 0.2` directly to EAE-like,
regardless of signed_delta. Equality uses the existing branches: EAE also
requires fraction <0.4 and signed_delta <-0.1; fraction >0.5 is TAE-like;
remaining cases are mixed and stay on the TAE side. All entry points share
this decision in `src/tae_eae_features.py`. Confirm the additional
`fraction_direct_eae_threshold=0.2` in shot and per-n summaries. The frozen
v5 preset remains supported with this threshold explicitly set to zero by
its schema adapter; use it to reproduce pre-v6 routing. Standalone split
and RF-CNN runs can reproduce the old routing with
`--fraction_direct_eae_threshold 0`. V6 preserves all v5 morphology gates,
the v18 feature column schema. The subsequently adopted shared continuum
repair changes continuum feature values: all active workflows now use
`datcon-monotonic-tail-v1`. Confirm `continuum_preprocessing_version` in shot
and per-n summaries for rules and RF-CNN. It detects sustained steep paired
terminal rises, backtracks onset, and holds preceding boundary values while
preserving NaNs. It uses native radial slopes at every nr; its scientific
calibration was checked on nr=201. New viewer sessions use it automatically.
Historical pre-repair outputs require the corresponding source checkout
(`64fc889` is the last old loader), in addition to v5/v6 thresholds.
Do not combine a named configuration
with config-owned threshold or gate flags; the CLI rejects such overrides.
Confirm the configuration name, schema version, and SHA-256 in the shot and
per-`n` summaries.

The rule engine deliberately has no positive GOOD template. Keep the two
decisions distinct:

```text
gate fired         -> rule_decision=BAD -> automatic final BAD
no gate fired      -> rule_decision=REVIEW
                   -> final_decision=GOOD by accept-as-good-v1
manual override    -> applied after the automatic final decision
final GOOD         -> RF-ranked frequency/structure deduplication
```

Do not rewrite `rule_decision` or `rule_primary_reason=NO_GOOD_TEMPLATE` when
the production workflow promotes a survivor. Confirm the
`accept-as-good-v1` policy identity and `n_rule_survivors_accepted` in the
summaries and final-classification audit.

RF is a post-decision ranker, not a rule classifier. The production command
must supply the active compatible checkpoint so close-frequency,
structurally matched final-GOOD modes can be reduced to one representative per
matched structural group. If RF is omitted, unloadable, or cannot score a
whole cluster, retain every affected member and report the fallback; treat
that as an audit/failure-safe result rather than the intended deduplicated
production list. Never use RF to change a rule, survivor-policy, manual, or
final decision.

For conservative calibration or feature-only work, use the configurable
interface. It does not apply the production survivor policy, so pass-all-gates
modes remain final REVIEW:

```bash
python scripts/sort_shot_rules.py \
  --shot_dir /path/to/SHOT \
  --out_dir /path/to/sort-output
```

The command aborts before processing if a populated requested `N#` directory
lacks `datcon#`. It uses the shared NOVA loader, continuum loader, and canonical
TAE/EAE/mixed split. Eleven ordered BAD decisions detect narrow near-axis
spikes, unresolved signed-harmonic spikes and short large-turn packets whose
strongest window sample is at `r <= 0.5`, relatively large consecutive
sign-flip oscillations at strict `r < 0.1`, continuum crossings carrying
appreciable exact-point or nearby amplitude and normalized radial energy, a
narrow globally dominant energy envelope at the outer radial boundary, and a
few-grid-interval interior total-energy envelope without a qualifying nearby
continuum extremum, incoherent harmonic activity in the calibrated core, and
rough continuum-crossing tails carrying appreciable reference-harmonic energy.
Their calibrated defaults are `r_ax=0.03` inclusive,
`axis_amplitude_min=0.2`, `axis_width_max_grid=10`,
`grid_scale_amplitude_min=0.3`, `grid_scale_width_max_grid=1`,
`grid_scale_high_r_cutoff_r=0.7`,
`grid_scale_high_r_width_max_grid=0.75`, packet defaults
`grid_scale_packet_amplitude_min=0.3`,
`grid_scale_packet_step_min=0.2`,
`grid_scale_packet_min_large_turns=3`, and
`grid_scale_packet_window_span_grid=4`, with inclusive
`grid_scale_packet_peak_r_max=0.5`; near-axis oscillation defaults are strict
`peak_r_max=0.1`, `amplitude_min=0.10`,
`min_consecutive_sign_flips=4`, and `step_l2_min=0.30`; continuum defaults
begin with
`w_cross_threshold=0.03`, crossing-window defaults
`cross_window_half_width_grid=2`, `cross_window_amplitude_min=0.25`, and
`cross_window_w_min=0.05`, with provisional edge defaults
`r_edge_min=0.97` inclusive and `edge_width_max_grid=10`; the interior-envelope
defaults are `peak_r_max=0.5`, `width_max_grid=2`, `ext_dr_max=0.02`, and
`0.001<ext_df_gap<=0.04`; the engine returns
`REVIEW` with `NO_GOOD_TEMPLATE` for modes not rejected by any gate. Only the
production `accept-as-good-v1` workflow policy promotes those survivors.

For every valid TAE-side mode, `rule_features` uses the grouped v22 schema. Keep
the production RF 22 in `rf_standard_features`, the six crossing summaries in
`crossing_features` together with crossing-window amplitude and energy audit
evidence, individual lower/upper crossings in `crossing_records`, and match
status plus the three inner-extremum measurements in `extremum_features`. Keep
the axis measurements under
`boundary_features.axis_artifact`, the outer energy-envelope and edge-harmonic
audit measurements under `boundary_features.edge_artifact`, the unresolved
signed-lobe measurements under `numerical_structure_features.grid_scale_spike`,
and short-window repeated-turn evidence under
`numerical_structure_features.grid_scale_packet`; store the interior
total-energy-width gate and its separate extended extremum match under
`resolution_features.interior_unresolved_envelope`; store the near-axis
mode-level amplitude and strongest strict single-harmonic sign-flip run under
`numerical_structure_features.near_axis_grid_oscillation`; store the combined core
incoherence score and every component under
`numerical_structure_features.interior_harmonic_incoherence`. In the same
object, keep the audit-only whole-radius `W`-weighted effective harmonic count,
whole-radius and core-conditional energy fractions at strict `N_eff(r)>3`, and
the corresponding total-energy fraction located in the core. Never use those
participation summaries by themselves to change a decision. These are named
deterministic measurements; no RF checkpoint or prediction is used to produce
them. Keep
`signed_delta` and
`fraction_below_upper2` as routing audit columns rather than rule features.
When no inner extremum is matched, require
`match_found=false` and JSON `null` for the three undefined extremum
measurements.

NOVA mode arrays use normalized radius and normalized mode amplitude. Address
the first array axis as the zero-based stored harmonic index; do not infer a
physical poloidal-`m` offset unless run metadata establishes that mapping.

The axis extractor searches every absolute harmonic profile for every local
maximum centered at `r <= r_ax`. Measure every candidate's connected
half-maximum component on its full radial profile; do not truncate the width at
`r_ax`. A candidate qualifies only when it meets both configured amplitude and
width thresholds. If one or more qualify, record the strongest qualifying peak,
its stored harmonic index, radius, connected half-maximum width in normalized
radius and grid intervals, outer edge, and whether the component includes
`r=0`, then reject the mode. Also record the total local-peak count and the
amplitude- and width-qualified counts. A larger rising flank or broad local
peak must not mask a narrower qualifying local peak. If no candidate qualifies,
retain the strongest raw axis-window amplitude as fallback audit information.

Override the calibrated gate when testing alternate thresholds:

```bash
python scripts/sort_shot_rules.py \
  --shot_dir /path/to/SHOT \
  --out_dir /path/to/sort-output \
  --axis_amplitude_min AMPLITUDE \
  --axis_width_max_grid GRID_INTERVALS
```

The default inclusive `--axis_r_ax` is `0.03`. When any local maximum meets the
amplitude minimum and its full-grid half-maximum width does not exceed the
configured maximum, return `BAD` with primary reason `BAD_AXIS_SPIKE` and stop
later decision gates. A narrow local maximum centered at `r <= 0.03` is a
boundary artifact regardless of an otherwise plausible morphology family. A
broad component extending beyond the window or the rising flank of a mode
centered outside it must not be made artificially narrow. Use
`--disable_axis_artifact` only when a feature-only run is explicitly needed.

The second gate searches every stored harmonic over the complete radial grid.
For each positive local maximum or negative local minimum, measure the connected
signed lobe above half of its own absolute peak. Never measure this component on
`abs(mode)`, because adjacent `+A/-A` samples would be joined into a falsely
broad component. Peaks at or below `grid_scale_high_r_cutoff_r` use
`grid_scale_width_max_grid`; peaks strictly above that cutoff use
`grid_scale_high_r_width_max_grid`. The cutoff belongs to the low-r branch,
and both width comparisons are inclusive. Among lobes no wider than their
applicable limit, record the strongest candidate, its signed amplitude, sign,
zero-based stored harmonic index, radius, interpolated inner and outer
half-maximum edges, width in normalized radius and grid intervals, and whether
the component touches either radial boundary.

The calibrated second gate is:

```text
IF grid_scale_peak >= 0.3
AND ((grid_scale_peak_r <= 0.7
      AND grid_scale_halfmax_width_grid <= 1)
     OR (grid_scale_peak_r > 0.7
         AND grid_scale_halfmax_width_grid <= 0.75))
THEN BAD_GRID_SCALE_SPIKE
AND stop evaluating later decision gates
```

It runs only after `BAD_AXIS_SPIKE`. Override its thresholds with
`--grid_scale_amplitude_min`, `--grid_scale_width_max_grid`,
`--grid_scale_high_r_cutoff_r`, and
`--grid_scale_high_r_width_max_grid`; use
`--disable_grid_scale_spike` for a feature-only run. The shot and per-`n`
summaries record enable state and exact settings.

Treat the repeated-turn packet screen as gate 2b so the established gate-3/4/5
terminology remains stable. Scan every complete five-sample window on every
stored harmonic. Let `d[i] = A[i+1] - A[i]`. Count
an interior sample as a large turn only when both adjacent steps meet
`abs(d) >= 0.2` and their directions oppose, `d[i-1] * d[i] < 0`. The
provisional gate is:

```text
IF max(abs(A)) in the window >= 0.3
AND all 3 interior samples are large turns
AND the largest absolute sample is centered at r <= 0.5
THEN BAD_GRID_SCALE_PACKET
AND stop evaluating later decision gates
```

The magnitude comparisons are inclusive; the direction reversal is strict.
This counts sharp signed local maxima and minima, so same-sign peaks separated
by deep troughs remain eligible while a single steep smooth peak is not a
packet. Record the selected window's signed values, stored harmonic index,
sample and radial bounds, peak and peak radius, large-step and large-turn
counts, maximum step, step RMS, total variation, unconstrained direction-change
and sign-change counts, and the counts of all-radius turn-qualified,
radius-qualified, and amplitude-qualified windows. Retain the peak and window
radii for audit. The peak-radius comparison is inclusive.
Override the provisional settings with
`--grid_scale_packet_amplitude_min`, `--grid_scale_packet_step_min`,
`--grid_scale_packet_min_large_turns`, and
`--grid_scale_packet_window_span_grid`, and
`--grid_scale_packet_peak_r_max`; use `--disable_grid_scale_packet` to retain
evidence without applying the decision.

Gate 2c detects relatively large grid-scale oscillations near
the axis. On each stored harmonic, form maximal runs for which every adjacent
pair is nonzero and changes sign. A zero sample or one interval without a sign
change ends the run; never bridge a missing flip. Keep only runs with at least
four consecutive sign changes and whose largest absolute run sample is at the
strict radius `r < 0.1`. For each kept run define:

```text
Q_s = sqrt(sum_i (A[i+1] - A[i])^2)
```

Select the largest `Q_s` from one run on one harmonic; never sum runs or
harmonics. Independently define
`A_peak=max_{h,r<0.1}|A_h(r)|`, which may come from a different harmonic. The
calibrated decision is:

```text
IF N_s >= 4 consecutive sign flips on one harmonic
AND that run's peak is at r < 0.1
AND A_peak >= 0.10
AND max_single_run(Q_s) >= 0.30
THEN BAD_NEAR_AXIS_GRID_OSCILLATION
AND stop evaluating later decision gates
```

The count, amplitude, and `Q_s` comparisons are inclusive; both radius uses
are strict. Store the mode-level peak and its harmonic/radius, the complete
selected run bounds and harmonic, `N_s`, `Q_s`, step summaries, qualifying-run
counts, and whether the two winning harmonics match. Override the settings
with `--near_axis_grid_oscillation_peak_r_max`,
`--near_axis_grid_oscillation_amplitude_min`,
`--near_axis_grid_oscillation_min_consecutive_sign_flips`, and
`--near_axis_grid_oscillation_step_l2_min`; use
`--disable_near_axis_grid_oscillation` to retain evidence without applying the
decision.

The third gate uses the existing deterministic true-crossing measurements. A
crossing is a lower/upper continuum boundary intersection recorded by the
shared continuum code. `W_star_max` is the largest crossing value of
`sum_h |mode_h(r)|^2`, normalized by its radial maximum. The calibrated gate is:

```text
IF n_cross > 0
AND W_star_max > 0.03
THEN BAD_CONT_CROSS
AND stop evaluating later decision gates
```

The comparison is intentionally strict (`>`). This gate runs only after
`BAD_GRID_SCALE_PACKET`. Override the threshold with `--w_cross_threshold`; use
`--disable_cont_cross` to retain the same crossing features while disabling the
decision.

The fourth gate inspects an inclusive radial neighborhood around every true
crossing:

```text
For every crossing, include samples with
abs(r_i - r_cross) <= 2 * delta_r

IF n_cross > 0
AND (cross_window_A_max >= 0.25 OR cross_window_W_max >= 0.05)
THEN BAD_CONT_CROSS_WINDOW
AND stop evaluating later decision gates
```

`cross_window_A_max` is the largest absolute individual-harmonic amplitude in
all crossing windows. `cross_window_W_max` is the largest
`sum_h |mode_h(r_i)|^2`, normalized by its radial maximum. Record independent
winning sample radius, crossing boundary/radius, distance in grid intervals,
and the winning stored harmonic index for amplitude.
`cross_window_A_neighbor_rms` uses signed values from that winning harmonic
and sample:

```text
sqrt(mean((A[j] - A[j+i])^2)), i = -2, -1, +1, +2
```

Require all four neighbors so the audit value is comparable across modes.
Record the available neighbor count and complete-stencil status; when a winner
lies too close to a radial boundary, store JSON `null` for RMS. Because the
amplitude winner may itself lie two grid intervals from the crossing, its
signed-neighbor stencil can extend four intervals from the crossing. RMS is
audit information only and must not alter the gate-4 decision. Override the
decision thresholds with `--cross_window_half_width_grid`,
`--cross_window_amplitude_min`, and `--cross_window_w_min`; use
`--disable_cont_cross_window` to retain evidence without applying the gate.
The two magnitude-threshold comparisons are inclusive.

The fifth gate measures the global radial-energy envelope
`W(r)=sum_h |mode_h(r)|^2`, normalized to a peak of one. Search both
half-maximum edges on the full grid. Keep a separate mirrored audit of the
strongest individual harmonic in the inclusive `r >= r_edge_min` window, but
do not use that harmonic alone for this decision: physical edge modes can have
narrow shear-localized harmonics while their total envelope remains resolved.
The provisional calibrated gate is:

```text
IF edge_energy_peak_r >= 0.97
AND edge_energy_halfmax_width_grid <= 10
THEN BAD_EDGE_SPIKE
AND stop evaluating later decision gates
```

This gate runs only after `BAD_CONT_CROSS_WINDOW`. Override its settings with
`--edge_r_min` and `--edge_width_max_grid`; use `--disable_edge_artifact` to
retain both envelope and harmonic audit measurements without applying the
decision. The edge threshold is inclusive. Shot and per-`n` summaries record
the enable state and exact threshold for every BAD decision.

The interior-envelope gate reuses the same global total-energy evidence; it never
measures the width of one harmonic. Its calibrated decision is:

```text
IF energy_peak_r <= 0.5
AND connected_total_energy_FWHM_grid <= 2
AND NOT (
  gate_specific_extremum_match_found
  AND ext_dr <= 0.02
  AND 0.001 < ext_df_gap <= 0.04
)
THEN BAD_INTERIOR_UNRESOLVED_ENVELOPE
AND stop evaluating later decision gates
```

Evaluate it only after every earlier BAD gate so the exception cannot rescue
an axis, signed-spike, packet, crossing, or edge rejection. The peak-radius,
width, radial-mismatch, and upper frequency-clearance comparisons remain
inclusive. The lower clearance comparison is strict: equality at 0.1% fails
the exception. Clearance is divided by mode frequency. No minimum-width floor
is added. Record `ext_df_gap_min_inclusive=false` in grouped features and
`interior_envelope_ext_df_gap_min_inclusive=false` in summaries. Frozen v5-v7
presets select the old inclusive zero lower bound. For legacy calibration,
`--interior_envelope_ext_df_gap_min_inclusive` includes equality at the chosen
lower limit. Search upper minima and lower maxima with centers in
`0.03 <= r <= 0.50`, using full finite-neighbor context at both search limits.
Keep this match separate from the experimental RF extremum feature, whose
established search still ends at `r=0.40`. The connected FWHM is the component
containing the tallest unsmoothed `W(r)` sample; it can be artificially narrow
for a broad, rippled edge envelope, so do not remove the `r_peak <= 0.5`
restriction without calibrating a robust whole-envelope width. Override the
settings with `--interior_envelope_peak_r_max`,
`--interior_envelope_width_max_grid`,
`--interior_envelope_extremum_r_min`,
`--interior_envelope_extremum_r_max`, `--interior_envelope_ext_dr_max`,
`--interior_envelope_ext_df_gap_min`, and
`--interior_envelope_ext_df_gap_max`; use
`--disable_interior_unresolved_envelope` to retain evidence without applying
the decision.

The interior harmonic-incoherence gate measures the stored array
without inferring physical poloidal-mode numbers. For `r_i <= 0.5`, define
`W_i=sum_h A_hi^2` and `p_hi=A_hi^2/W_i`. Record:

- `f_core`, the fraction of total mode energy in the inclusive core;
- `J_core`, the base-2 Jensen--Shannon divergence between consecutive
  `p(:,i)` distributions, weighted by `sqrt(W_i W_(i+1))`;
- `N_eff_core`, the `W_i`-weighted mean of pointwise
  `1/sum_h p_hi^2` (simultaneous participation at one radius, not the union of
  rows used by a ridge across radius);
- `C_adj`, the coherence of adjacent active stored harmonic rows. A row is
  active at integrated core-energy fraction `>=0.005`. For each adjacent
  active pair, maximize the absolute uncentered cosine of the signed profiles
  over integer lags `-5..+5`, recomputing norms on each overlap and never
  bridging an inactive stored-index gap; average pair values with weight
  `sqrt(E_h^core E_(h+1)^core)`.

Also record audit-only participation summaries at the fixed strict reference
`N_eff(i)>3`: the whole-radius `W_i`-weighted mean `N_eff`, the whole-radius
energy fraction `G_3`, the core-conditional energy fraction `G_3,core`, and
`B_3,core=f_core*G_3,core`. Ignore samples with zero `W_i`; do not expose an
unweighted pointwise maximum or invent a hard `W/W_max` cutoff. These values
remain evidence only and must not alter `candidate_found`, including when the
incoherence gate is disabled or resolution-ineligible.

Then apply the strict calibrated decision:

```text
S_inc = f_core * J_core * N_eff_core * (1 - C_adj)
IF n_radial == 201
AND S_inc > 0.10
THEN BAD_INTERIOR_HARMONIC_INCOHERENCE
```

Run it only after every earlier BAD gate so existing primary reasons keep
precedence. Equality at `0.10` passes. Zero core energy, no positive-weight
adjacent radial pair, or no adjacent active harmonic pair leaves undefined
components as JSON `null` and cannot reject. The native-grid divergence and
lag calibration is not portable across radial resolution: retain all audit
measurements but set `resolution_eligible=false` and fail open whenever
`n_radial != 201`. Use `--disable_interior_harmonic_incoherence` only for a
feature-only run. Confirm the eligible and ineligible mode counts in shot and
per-`n` summaries; an enabled gate with only ineligible inputs did not screen
those inputs.

Use `scripts/make_tae_like_list.py` directly only when preprocessing outputs
without final rule results are needed. For deterministic production, run
`sort_shot_mixed.py --method rules` with the RF checkpoint used only for
post-decision deduplication; never select `--method rf-cnn` or use an RF or CNN
prediction to make a rule decision.

## Continuum crossing-tail rejection

The continuum tail gate is `continuum_crossing_tail`, with reason
`BAD_CONTINUUM_CROSSING_TAIL`. Preserve every earlier primary reason.
At each actual lower/upper crossing, use the side opposite the global W peak
as the tail (inner if r_cross < r_peak, outer otherwise). Retain all harmonics
in the tail numerator. Define `E_h=integral A_h(r)^2 dr` and select the two
individual harmonics with largest full-domain energies; no adjacency
constraint or physical-m offset is assumed. Use
`T_2=E_tail/(E_h1+E_h2)`, which may exceed one and is not a total-energy
fraction. Integrate piecewise-linear W with the crossing as an endpoint.
Never add overlapping cumulative tails from different crossings.

K is the norm of the unscaled signed second differences divided by the local
amplitude norm, summing all harmonics and complete stencil centers within
`abs(r_i-r_cross) <= 4*delta_r`. Do not smooth or divide by delta_r squared.
Reject only when **the same crossing** has strict `K > 0.4 AND T_2 > 0.035`
and `n_radial == 201`; equality passes. Undefined denominators or other
resolutions cannot reject. Preserve the measurements and resolution status
even when disabled or ineligible. Record the energy denominator, stored
harmonic indices, every crossing's metrics, and a witness chosen from
crossings satisfying both cuts under
`crossing_features.continuum_crossing_tail`.

Calibration options are `--continuum_crossing_tail_k_min`,
`--continuum_crossing_tail_top2_ratio_min`,
`--continuum_crossing_tail_half_width_grid`, and
`--continuum_crossing_tail_calibrated_n_radial`; use
`--disable_continuum_crossing_tail` to retain evidence without the decision.
Confirm enable state, thresholds, and eligible/ineligible counts in shot and
per-n summaries. Production-v5 freezes these settings; v4 remains unchanged
and requires the corresponding historical checkout.

For evaluated modes excluded by either enabled resolution-dependent gate,
both sorter entry points print a stderr warning and save
`resolution_warnings.txt` plus per-mode/per-gate `resolution_warnings.csv`.
Inspect these before treating output as fully screened. Sorting continues;
other enabled gates run on the native grid, and production survivors can
still become GOOD. The warning states that policy and counts affected GOOD
modes. Intentionally disabled gates do not warn; every run replaces both
reports to clear stale warnings. No resampling or automatic abort is implied.

## Axis energy concentration rejection

The gate introduced in production v9 is `axis_energy_concentration`, with reason
`BAD_AXIS_ENERGY_CONCENTRATION`. Keep every earlier BAD primary reason.
Reject only when both strict conditions hold:

```text
max_(h, native r_i <= 0.015) |xi_h(r_i)| > 0.5
AND integral_0^0.05 W(r) dr / integral_0^1 W(r) dr > 0.5
W(r) = sum_h |xi_h(r)|^2
```

Use existing globally normalized amplitudes and all harmonics in the energy
fraction. The shared piecewise-linear W integration includes the exact
window endpoints; do not add volume/Jacobian weighting. No width, local-peak,
or continuum-extremum condition is imposed. Equality at either amplitude or
energy cut passes. A zero-energy input has an undefined fraction and cannot
qualify. The condition targets dominant inner energy while retaining the
user-approved extended modes E202855A01t020 N1/8188 and E204645A16t015 N1/3712.

Evaluate on every native radial grid; there is no nr-based skip. Scientific
calibration used nr=201, so do not claim resolution-independent accuracy.
Record the thresholds, native nr, axis sample count, maximum absolute/signed
amplitude, stored harmonic index and peak radius, total radial energy,
inner fraction, and candidate status under
`boundary_features.axis_energy_concentration` even when disabled. Summaries
record `axis_energy_concentration_gate_enabled` plus
`axis_energy_amplitude_r_max`, `axis_energy_amplitude_min`,
`axis_energy_r_max`, and `axis_energy_fraction_min`.

The corresponding calibration flags use those four `--axis_energy_*` names;
`--disable_axis_energy_concentration` disables rejection while retaining
measurements. Frozen v5-v8 presets explicitly disable this added gate and
retain their earlier decisions while emitting current audit metadata.

## Extended continuum noise rejection

The final production-v10 gate is `extended_continuum_noise`, with reason
`BAD_EXTENDED_CONTINUUM_NOISE`. All three inclusive conditions must hold in
one connected above-upper or below-lower TAE-gap region:

```text
hf_out_top2_ratio >= 0.01
AND hf_out_local_fraction >= 0.20
AND hf_out_radial_length >= 0.04
```

Use shared `src/continuum_noise.py`. Compute signed `diff(xi,2)/4` on the native
profile before masking. Only stencils whose three samples belong to the same
outside-gap region contribute. Equality with a boundary is in-gap; unknown,
negative or reversed continuum bounds split regions. Crossing-straddling and
unknown-neighbor stencil energies are separate audit evidence. No smoothing,
resampling, or division by dr**2 is used.

All energies use native trapezoidal node weights. Regional raw energy is the
sum over weighted outside nodes, without interpolated crossing endpoints.
The numerator retains all harmonics. The reference is full-domain integrated
energy of the two strongest individual harmonics, with stable lower-index
ties and no adjacency requirement; its ratio may exceed one. Local fraction
uses raw energy of the same outside region. Effective radial length is
`N_r,eff/(nr-1)`, where `N_r,eff=(sum e_i)^2/sum e_i^2`. At nr=201, length 0.04
is eight effective points. Harmonic participation is audit-only.

Evaluate every supported native nr>=3; this gate has no nr=201 restriction.
Only the high-pass operator is intentionally grid-relative. Synthetic native
51/101/201/401 checks verify the length criterion; empirical calibration used
nr=201. Preserve the existing four-consecutive-flip gate and all earlier BAD
primary reasons. Record every region, denominator, stencil status, cut, and
qualifying witness under `numerical_structure_features.extended_continuum_noise`.

Frozen v5-v9 presets explicitly disable this new gate. The calibration CLI
accepts `--continuum_noise_top2_min`, `--continuum_noise_local_min`,
`--continuum_noise_radial_length_min`, and `--disable_extended_continuum_noise`.
Disabled gates retain measurements. Shot/per-n summaries record all three
cuts and enable state. `audit_continuum_noise.py` provides independent measure
and explicit sweep commands; its v2 schema requires fresh measurements when
moving from old v1 caches. See `audits/continuum_noise_20260910/README.md`.

## Honor known invalid input scopes

Before gap routing, both canonical sorting methods and `make_tae_like_list.py`
apply `configs/known_invalid_inputs.csv` through shared `src/input_validity.py`.
The registry excludes `nstxuG142301C50/N1` for `CONTINUUM_MODE_MISMATCH`
and the whole `nstxuG133964R06` shot for `SUSPECT_EIGENMODE_STRUCTURE`,
covering both TAE and EAE frequency ranges. R06 records the user's visual
assessment of poor structures throughout and some spectra peaking at the
largest retained poloidal harmonic; its cause is unconfirmed. Match the exact
shot basename: a positive integer `ntor` covers one n, while `ntor=*` covers
the whole shot, including future files. Whole-shot entries take precedence
if a per-n entry also exists. Keep exclusions active until corrected inputs
have been reviewed and their registry entries removed.

These are INVALID inputs, not morphology BAD decisions. The shared reason
is `KNOWN_INVALID_INPUT`; diagnostics retain the issue, reviewer, evidence,
and registry hash. Rules use `final_decision=INVALID`; RF-CNN uses
`status=rejected, final_label=invalid`. They appear in `rejected_modes.csv`
and the complete audit, with `n_known_invalid_inputs` in shot/per-n summaries.
No GOOD/BAD/REVIEW override can restore an INVALID row. Raw files and viewers
remain available for investigation. Do not tune a morphology gate to absorb
this data inconsistency. NOVA calculates eigenfrequencies and eigenmode
structure; call its run an eigenmode calculation, not a stability calculation.

## Add explicit adjudication

Create or update fingerprinted overrides from a production sorter output. The
`review` scope selects the preserved preliminary `rule_decision`, so it still
finds pass-all-gates survivors even though their final decision is GOOD:

```bash
python scripts/label_modes_fast.py /path/to/SHOT \
  --mode-list /path/to/sort-output/final_classifications.csv \
  --csv_out /path/to/manual_overrides.csv \
  --adjudication review \
  --reviewer REVIEWER_ID \
  --no-rf
```

Use `--adjudication all` only when gate-rejected BAD rows should also be
eligible. Supply a nonempty reason for every decision. This is a non-blind
post-rule action; do not describe it as independent validation.

Rebuild deterministically after adjudication:

```bash
python scripts/sort_shot_mixed.py \
  --method rules \
  --shot_dir /path/to/SHOT \
  --rf_model models/nova_mode_classifier.joblib \
  --out_dir /path/to/sort-output \
  --manual_overrides /path/to/manual_overrides.csv
```

The sorter applies an override only when its stored mode-plus-`datcon#`
fingerprint matches. Inspect stale, ambiguous, or ineligible override counts in
`shot_summary.csv`.

Use the same override file option with `sort_shot_rules.py` only when rebuilding
the conservative REVIEW-preserving audit instead of production outputs.

## Deduplicate final-GOOD production outputs

With `sort_shot_mixed.py --method rules`, supply
`--rf_model /path/to/model.joblib` in production to rank representatives among
final-GOOD modes that match in both frequency and structure. RF scores must not
alter rule, survivor-policy, manual, or final decisions. Without a usable RF
checkpoint—or if one cluster cannot be fully scored—the workflow retains every
affected member and records the fallback in `frequency_cluster_report.txt` and
`frequency_clusters.csv`; this is supported for conservative audits, including
`sort_shot_rules.py`, but is not a deduplicated production result. Do not
supply or load a CNN for this method.

## Audit outputs

Start with:

- `all_modes_rules.csv` for every discovered input;
- `rule_results.csv` for preliminary TAE-side rule results;
- `final_classifications.csv` for override-aware classifications;
- `bad_tae_like.csv`, `review_tae_like.csv`, `good_tae_unchecked.csv`, and
  `good_tae_final.csv` for mutually exclusive final lists;
- `tae_like_all.csv`, `eae_like.csv`, and `rejected_modes.csv` for routing and
  input failures;
- `shot_summary.csv` and `shot_summary_by_n.csv` for counts based on one primary
  reason per mode.

Treat `rule_triggered_rules` as per-mode audit detail, not as summary-count
input. Confirm the manual-override SHA-256 in the summary when overrides were
supplied. The summary also records whether each implemented gate was enabled
and its exact thresholds. For production runs, also confirm
`accept-as-good-v1` and `n_rule_survivors_accepted`; automatic survivors remain
REVIEW only in the rule-decision columns and appear as GOOD in final outputs.
`review_tae_like.csv` is normally empty unless a manual REVIEW override is
present. Do not add timestamps while regenerating deterministic outputs.
