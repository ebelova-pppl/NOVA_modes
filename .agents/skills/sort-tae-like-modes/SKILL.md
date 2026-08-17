---
name: sort-tae-like-modes
description: "Deterministically preprocess and sort one NOVA shot into TAE-like, mixed, EAE-like, invalid, BAD, REVIEW, and GOOD outputs with auditable rule reasons, reusable fingerprinted manual overrides, and optional RF-only ranking of final-GOOD close-frequency representatives. Use for one-shot NOVA TAE preprocessing, rule-based sorting, reproducible output regeneration, output auditing, or explicit post-rule adjudication without CNN or model-based classification."
---

# Sort TAE-Like Modes

Process one target shot noninteractively with the repository scripts. Keep
frequency routing, deterministic decisions, manual overrides, and duplicate
ranking as separate stages.

## Run the deterministic workflow

```bash
python scripts/sort_shot_rules.py \
  --shot_dir /path/to/SHOT \
  --out_dir /path/to/sort-output
```

The command aborts before processing if a populated requested `N#` directory
lacks `datcon#`. It uses the shared NOVA loader, continuum loader, and canonical
TAE/EAE/mixed split. The first implemented rejection gate detects narrow
near-axis spikes. Its calibrated defaults are `r_ax=0.03` inclusive,
`axis_amplitude_min=0.2`, and `axis_width_max_grid=10`; modes not rejected by
the gate remain `REVIEW` with `NO_GOOD_TEMPLATE`, never `GOOD`.

For every valid TAE-side mode, `rule_features` uses the grouped v3 schema. Keep
the production RF 22 in `rf_standard_features`, the six crossing summaries in
`crossing_features`, individual lower/upper crossings in `crossing_records`,
and match status plus the three inner-extremum measurements in
`extremum_features`. Keep the axis measurements under
`boundary_features.axis_artifact`; reserve empty objects for
`resolution_features` and `numerical_structure_features`. These are named
deterministic measurements; no RF checkpoint or prediction is used to produce
them. Keep `signed_delta` and `fraction_below_upper2` as routing audit columns
rather than rule features. When no inner extremum is matched, require
`match_found=false` and JSON `null` for the three undefined extremum
measurements.

NOVA mode arrays use normalized radius and normalized mode amplitude. Address
the first array axis as the zero-based stored harmonic index; do not infer a
physical poloidal-`m` offset unless run metadata establishes that mapping.

The axis extractor searches `r <= r_ax` for the largest absolute amplitude
over all stored harmonics and records the peak, stored harmonic index, radius,
local-maximum status, connected half-maximum width in normalized radius and
grid intervals, outer edge, and whether the component includes `r=0`. Determine
the local maximum and both half-maximum edges from the selected harmonic's full
radial profile. Do not truncate the width calculation at `r_ax`.

Override the calibrated gate when testing alternate thresholds:

```bash
python scripts/sort_shot_rules.py \
  --shot_dir /path/to/SHOT \
  --out_dir /path/to/sort-output \
  --axis_amplitude_min AMPLITUDE \
  --axis_width_max_grid GRID_INTERVALS
```

The default inclusive `--axis_r_ax` is `0.03`. When the axis candidate is a true
local maximum, its amplitude meets the minimum, and its full-grid half-maximum
width does not exceed the configured maximum, return `BAD` with primary reason
`BAD_AXIS_SPIKE` and stop later decision gates. A narrow local maximum centered
at `r <= 0.03` is a boundary artifact regardless of an otherwise plausible
morphology family. A broad component extending beyond the window or the rising
flank of a mode centered outside it must not be made artificially narrow. Use
`--disable_axis_artifact` only when a feature-only run is explicitly needed.

Use `scripts/make_tae_like_list.py` directly only when preprocessing outputs
without final rule results are needed. Do not run `sort_shot_mixed.py`, RF, or
CNN to make deterministic or manual classifications.

## Add explicit adjudication

Create or update fingerprinted overrides from a sorter output:

```bash
python scripts/label_modes_fast.py /path/to/SHOT \
  --mode-list /path/to/sort-output/final_classifications.csv \
  --csv_out /path/to/manual_overrides.csv \
  --adjudication review \
  --reviewer REVIEWER_ID \
  --no-rf
```

Use `--adjudication all` only when preliminary GOOD and BAD rows should also be
eligible. Supply a nonempty reason for every decision. This is a non-blind
post-rule action; do not describe it as independent validation.

Rebuild deterministically after adjudication:

```bash
python scripts/sort_shot_rules.py \
  --shot_dir /path/to/SHOT \
  --out_dir /path/to/sort-output \
  --manual_overrides /path/to/manual_overrides.csv
```

The sorter applies an override only when its stored mode-plus-`datcon#`
fingerprint matches. Inspect stale, ambiguous, or ineligible override counts in
`shot_summary.csv`.

## Rank final-GOOD duplicates when requested

Add `--rf_model /path/to/model.joblib` only to rank representatives among
final-GOOD close-frequency modes. RF scores must not alter rule, manual, or
final decisions. Without a usable RF checkpoint—or if one cluster cannot be
fully scored—the workflow retains every affected member and records the
fallback in `frequency_cluster_report.txt` and `frequency_clusters.csv`.

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
supplied. The summary also records whether the axis gate was enabled and its
exact radius, amplitude, and width settings. Do not add timestamps while
regenerating deterministic outputs.
