---
name: sort-tae-like-modes
description: "Deterministically preprocess and sort one NOVA shot into TAE-like, mixed, EAE-like, invalid, BAD, REVIEW, and GOOD outputs with auditable rule reasons, an explicit production survivor policy, reusable fingerprinted manual overrides, and optional RF-only ranking of final-GOOD close-frequency representatives. Use for one-shot NOVA TAE preprocessing, rule-based production sorting, conservative rule calibration, reproducible output regeneration, output auditing, or explicit post-rule adjudication without CNN classification."
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
  --out_dir /path/to/sort-output
```

The preset is `configs/rules/tae_rules_production_v1.yaml`. It pins the v14
ruleset and routing values, enables gates 1, 2, 2b, 4, and 5, and explicitly
disables exact-point continuum gate 3. Do not combine a named configuration
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
final GOOD         -> optional duplicate ranking
```

Do not rewrite `rule_decision` or `rule_primary_reason=NO_GOOD_TEMPLATE` when
the production workflow promotes a survivor. Confirm the
`accept-as-good-v1` policy identity and `n_rule_survivors_accepted` in the
summaries and final-classification audit.

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
TAE/EAE/mixed split. Six ordered BAD decisions detect narrow near-axis spikes,
unresolved signed-harmonic spikes and short large-turn packets whose strongest
window sample is at `r <= 0.5`, continuum crossings carrying appreciable exact-point or nearby amplitude and
normalized radial energy, followed by a narrow globally dominant energy
envelope at the outer radial boundary.
Their calibrated defaults are `r_ax=0.03` inclusive,
`axis_amplitude_min=0.2`, `axis_width_max_grid=10`,
`grid_scale_amplitude_min=0.3`, `grid_scale_width_max_grid=1`,
`grid_scale_high_r_cutoff_r=0.7`,
`grid_scale_high_r_width_max_grid=0.75`, packet defaults
`grid_scale_packet_amplitude_min=0.3`,
`grid_scale_packet_step_min=0.2`,
`grid_scale_packet_min_large_turns=3`, and
`grid_scale_packet_window_span_grid=4`, with inclusive
`grid_scale_packet_peak_r_max=0.5`; continuum defaults begin with
`w_cross_threshold=0.03`, crossing-window defaults
`cross_window_half_width_grid=2`, `cross_window_amplitude_min=0.25`, and
`cross_window_w_min=0.05`, with provisional edge defaults
`r_edge_min=0.97` inclusive and `edge_width_max_grid=10`; the engine returns
`REVIEW` with `NO_GOOD_TEMPLATE` for modes not rejected by any gate. Only the
production `accept-as-good-v1` workflow policy promotes those survivors.

For every valid TAE-side mode, `rule_features` uses the grouped v13 schema. Keep
the production RF 22 in `rf_standard_features`, the six crossing summaries in
`crossing_features` together with crossing-window amplitude and energy audit
evidence, individual lower/upper crossings in `crossing_records`, and match
status plus the three inner-extremum measurements in `extremum_features`. Keep
the axis measurements under
`boundary_features.axis_artifact`, the outer energy-envelope and edge-harmonic
audit measurements under `boundary_features.edge_artifact`, the unresolved
signed-lobe measurements under `numerical_structure_features.grid_scale_spike`,
and short-window repeated-turn evidence under
`numerical_structure_features.grid_scale_packet`; reserve an empty object for
`resolution_features`. These are named deterministic
measurements; no RF checkpoint or prediction is used to produce them. Keep
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
the enable state and exact threshold for all six BAD decisions.

Use `scripts/make_tae_like_list.py` directly only when preprocessing outputs
without final rule results are needed. For deterministic production, run
`sort_shot_mixed.py --method rules`; never select `--method rf-cnn` or use a
CNN prediction to make a rule decision.

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
  --out_dir /path/to/sort-output \
  --manual_overrides /path/to/manual_overrides.csv
```

The sorter applies an override only when its stored mode-plus-`datcon#`
fingerprint matches. Inspect stale, ambiguous, or ineligible override counts in
`shot_summary.csv`.

Use the same override file option with `sort_shot_rules.py` only when rebuilding
the conservative REVIEW-preserving audit instead of production outputs.

## Rank final-GOOD duplicates when requested

With `sort_shot_mixed.py --method rules` or `sort_shot_rules.py`, add
`--rf_model /path/to/model.joblib` only to rank representatives among
final-GOOD close-frequency modes. RF scores must not alter rule, survivor-
policy, manual, or final decisions. Without a usable RF checkpoint—or if one
cluster cannot be fully scored—the workflow retains every affected member and
records the fallback in `frequency_cluster_report.txt` and
`frequency_clusters.csv`. Do not supply or load a CNN for this method.

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
