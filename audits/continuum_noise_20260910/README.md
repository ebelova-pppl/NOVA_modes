# Continuum-side high-frequency energy calibration, 2026-09-10

The user approved adoption as a separate production gate in v10 (ruleset and
features v22). It runs last with inclusive cuts:

```
hf_out_top2_ratio >= 0.01
AND hf_out_local_fraction >= 0.20
AND hf_out_radial_length >= 0.04
```

All three must qualify in the same connected outside-gap region. Effective
length is `hf_out_radial_extent/(nr-1)`, equivalent to eight effective points
at nr=201. The high-pass operator intentionally remains native-grid-relative;
this gate evaluates every supported native nr>=3 without resampling. Harmonic
participation stays audit-only. Frozen v5-v9 presets explicitly disable it.
The original point-based v1 calibration files below are retained as historical
evidence; new production evidence uses schema `continuum-side-noise-v2`.
This is non-blind calibration against existing labels.

## Verified adoption result

- Training: no new rejection among 575 GOOD labels. The gate flags 87 BAD
  labels, including 86 already rejected and the one additional known BAD
  E204669M03t025 N10/1295. No labels were changed. All 2,390 training input
  fingerprints match v9; original noise measurements match the v1 calibration
  exactly, apart from the added length and schema/eligibility metadata.
- Batch: only E204186A01t020 N10/1271 changes GOOD to BAD among the 949 v9
  survivors. Current totals are 948 GOOD before deduplication and 942 selected.
  Across all 4,187 evaluated modes, every earlier feature value, BAD primary
  reason, and other classification remains unchanged. There was no RF-ranking
  fallback.
- Disagreements: 227 -> 226. [Current list](current_disagreements.csv) is the
  previous list with only N10/1271 removed; retained rows are identical.
- Inventory: 65 previously INVALID C50/N1 files are now absent from raw data,
  in addition to eight already absent at v9. These affect neither usable-mode
  counts nor the gate comparison. The new absence ledger is
  [newly_absent_invalid_inputs.csv](newly_absent_invalid_inputs.csv). Canonical
  reruns therefore contain 19,252 inputs (4,187 evaluated, 14,358 EAE, 707
  INVALID). The C50/N1 registry exclusion remains active.
- Published all 27 verified rules directories in the existing output root.
  Prior outputs are preserved under `before_continuum_noise_v10_20260910/`;
  [publication receipt](publication.json) records old/new hashes and backup
  paths. RF–CNN output trees are unchanged.
- Evidence: [batch changes](adopted_changes.csv), [training changes](training_changes.csv),
  [shot totals](adopted_shot_summary.csv), and
  [verification receipt](adoption_verification.json). Runtime exports and the
  full training comparison are ignored under
  `outputs/review_continuum_noise_v10_20260910/`.

## Definition

Calculate the signed second difference on the original native profile:

```
h[:, i] = (xi[:, i+1] - 2*xi[:, i] + xi[:, i-1]) / 4
```

Each contiguous above-upper or below-lower continuum region is evaluated
separately. Equality is in-gap. Both continuum bounds must be finite,
nonnegative, and ordered; unknown samples split regions. Eligible centers
require all three stencil samples in the same region. The mode is never
zeroed or smoothed before differencing. Crossing-straddling and
unknown-neighbor HF energies/counts are separate audit measurements.

All energy sums use the uniform native trapezoidal node weights: dr at
interior nodes and dr/2 at domain endpoints. Regional raw energy is the sum
of weighted raw squared amplitudes at outside nodes, without interpolated
crossing endpoints. Global reference energy is the full-domain energy of
the two individually strongest harmonics, with stable lower-index tie
breaking and no adjacency requirement. The numerator retains every harmonic.

For each region report:

- `hf_out_top2_ratio = E_hf,out / (E_h1+E_h2)`, which may exceed one;
- `hf_out_total_fraction = E_hf,out / E_total`, audit-only;
- `hf_out_local_fraction = E_hf,out / E_raw,out`;
- `hf_out_radial_extent = (sum_i e_i)^2 / sum_i e_i^2`;
- `hf_out_radial_length = hf_out_radial_extent/(nr-1)`, used by v10;
- the analogous harmonic participation, audit-only;
- positive-energy counts, peak radius/harmonic, masks, and energy denominators.

Participation does not require consecutive points. An isolated raw sample
produces three nonzero filtered samples and radial participation 2; a long
alternating profile has participation near its number of eligible centers.
Undefined denominators/zero HF energy have null fractions/participation as
appropriate. The original experimental restriction to nr=201 has been removed.

The candidate requires three inclusive cuts on the **same region**, and
flags a mode once if any region qualifies. There is no harmonic-extent cut
or change to the existing four-flip requirement. `assess_continuum_noise`
is disabled without explicit `ContinuumNoiseThresholds`; production uses the
shared default configuration through `extract_continuum_noise_features`.
The rejection reason is `BAD_EXTENDED_CONTINUUM_NOISE`.

## Original v1 coverage and calibration result

Read all 2,390 canonical training entries: 2,363 are eligible TAE/mixed
(575 GOOD, 1,788 BAD), with 26 EAE and one INVALID accounted for separately.
All input fingerprints match the v9 training audit. Separately measured all
949 current production GOOD modes before deduplication in the 27-shot batch,
with matching input fingerprints. Every measured mode has nr=201. Known-invalid
scopes remain excluded. RF/CNN scores were not used in this calibration.

Tested 384 combinations: top-two minima 0.005/0.01/0.015/0.02/0.025/0.03;
local minima 0.05/0.10/0.15/0.20/0.25/0.30/0.35/0.40; radial minima
2/4/6/8/10/12/14/16. The full counts are in `threshold_sweep.csv`.

The original candidate, now adopted with the equivalent radial-length cut, was:

```
hf_out_top2_ratio >= 0.01
AND hf_out_local_fraction >= 0.20
AND hf_out_radial_extent >= 8
```

It flags **zero of all 575 training GOOD labels**, including zero of the
542 current GOOD survivors. It flags 87 training BAD labels: 86 already
rejected and one additional survivor, **E204669M03t025 N10/1295**.
Of the 949 batch survivors, only **E204186A01t020 N10/1271** is flagged.
These candidate cuts are not a statistically determined optimum; many nearby
cuts preserve the same two additional rejections. The same local/radial cuts
with top-two minimum 0.005 flag one training GOOD and two other batch survivors.

| Mode | Label/scope | Top-two ratio | Local HF fraction | Effective radial points | Effective harmonics |
|---|---|---:|---:|---:|---:|
| E204186A01t020 N10/1271 | Batch survivor; user flagged | 0.0287553 | 0.353108 | 14.8968 | 3.1565 |
| E204669M03t025 N10/1295 | Training BAD survivor | 0.0210407 | 0.960208 | 14.3714 | 2.3927 |
| E205052A01t022 N9/1178 | Training GOOD; retained | 0.00617859 | 0.358523 | 9.29395 | 2.8413 |

The target's whole-mode HF fraction is only 0.00549253 (0.549%), illustrating
the denominator dilution avoided by using the top-two reference (2.876%).
Requiring at least three effective harmonics would miss the additional
training BAD example, supporting the decision to retain harmonic participation
as a diagnostic. Previous approved narrow/smooth examples in these populations
remain unchanged under the candidate.

## Evidence and reproduction

- `newly_flagged.csv`: the two candidate additions, with reviewable paths.
- `flagged_training.csv`: all 87 flagged training BAD modes.
- `candidate_counts.csv`, `threshold_sweep.csv`: population counts, including
  incremental changes relative to the current rules.
- `target_diagnostic.json`, `nearby_good_diagnostic.json`: complete measurements.
- `summary.json`: data/source hashes, measurement coverage, and sweep inputs.

Full region tables, all-mode measurements, logs, and figures remain ignored in
`outputs/review_continuum_noise_20260910/`. Figures are in its `plots/` directory:
`training_calibration.png` and signed-profile/continuum pages for the three
listed examples. The measurement receipt retains the source hashes at the
time of measurement; completion/hash checks were subsequently added to the
CLI without changing the feature calculations or measurement files.

Example current v2 measurement and candidate evaluation (Flux/tcsh; measure afresh to use the length-based sweep):

```tcsh
python scripts/audit_continuum_noise.py measure \
  --mode-list training_labels/tae_like_train.csv \
  --data-root "$NOVA_DATA" --cohort training \
  --baseline-csv outputs/review_axis_energy_v9_20260909/training_comparison.csv \
  --out-dir outputs/review_continuum_noise_training --workers 4

python scripts/audit_continuum_noise.py sweep \
  --measurements outputs/review_continuum_noise_training/measurements.jsonl \
  --top2-min 0.01 --local-min 0.20 --radial-length-min 0.04 --export-flags \
  --out-dir outputs/review_continuum_noise_training/candidate
```

Without a baseline CSV, label-based counts still work; incremental survivor
counts need fingerprinted baseline decisions. Batch input was assembled from
the current GOOD rows of the 27 published `all_modes_rules.csv` files using
the shot membership in `audits/axis_amplitude_20260909/adopted_shot_summary.csv`.

Eleven focused tests cover analytic filter response, isolated-spike participation,
global sign/amplitude scaling, top-two ranking and zero-harmonic padding,
crossing and unknown masks, separate regions, exact inclusive cuts, disabled
and other-resolution behavior, invalid inputs, and a CLI measurement/sweep
round trip with fingerprint protection. The complete 185-test suite passes,
including canonical v10/v9 behavior, earlier BAD reason precedence, and
native 51/101/201/401 length checks.

## Production regeneration workflow

`adopt_gate.py` stages the canonical v10 sorter for the same 27 shots as the
v9 audit. It records hashes of source files, raw-input fingerprints, the RF
ranking checkpoint, and old rules/RF–CNN output trees. Verification compares
all prior feature values, decisions, selected representatives, and exclusions;
only the added diagnostic group and version metadata may change generally.
Missing raw inputs may be omitted only if they were already registry-excluded
INVALID entries; they receive a separate absence ledger. It also re-evaluates
training with the gate enabled and disabled, compares
the old v9 decisions, and checks the original v1 noise measurements exactly
apart from the added length and schema/eligibility metadata.

```tcsh
python audits/continuum_noise_20260910/adopt_gate.py stage \
  --rules-root /path/to/sort_outputs --ai-root /path/to/sort_outputs_ai \
  --training-root /path/to/training \
  --rf-model models/nova_mode_classifier.joblib \
  --out-root outputs/review_continuum_noise_v10_20260910
```

`verify` reuses completed staged exports; `publish` installs verified trees
with `before_continuum_noise_v10_20260910/` backups and checks RF–CNN outputs
remain unchanged. These adoption commands are tied to the fingerprinted v9
baseline; after publication use the canonical sorter for new shots or the
saved verification/publication receipts to inspect this transition.

The stage-driver hash remains in `run_inputs.json`. The verification driver was
updated to explicitly check 65 newly absent C50/N1 INVALID files; its current
hash is recorded separately as `verification_driver_sha256`. All sorter,
feature, model, registry, and baseline hashes must match the original stage.
