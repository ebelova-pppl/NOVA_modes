# Sorting NOVA modes

This guide covers routine sorting of existing NOVA output. The default
production method uses deterministic rules and does not require training or
an AI checkpoint. For environment setup, choose
[Flux or Perlmutter in the platform guide](platforms.md).

Commands assume that the working directory is the repository root. Replace
`/path/to/...` values with your own paths. For commands launched elsewhere,
use the absolute script path; direct `scripts/*.py` entry points locate this
checkout's `src/` directory without requiring `PYTHONPATH`.

## Required inputs

Pass one shot directory to `--shot_dir`. Its participating `N<N>` directories
contain `egn*` mode files and their matching `datcon<N>` continuum file. For
example, `N3` contains `egn03w.*` files and `datcon3`.

The default range is `--n_min 1 --n_max 10`. For shots extending beyond N10,
set `--n_max` accordingly; for example, add `--n_max 20` to include N11–N20.
Directories outside the requested range are not scanned.

| Input | Expected meaning |
| --- | --- |
| `egn*` | NOVA structure plus `omega`, `gamma_d`, and toroidal mode number `ntor`; shot preprocessing checks that `ntor` matches its `N<N>` directory. |
| Mode structure | Signed amplitudes on a `(number of harmonics, number of radial points)` grid. Harmonic count can vary; stored row indices do not establish a physical poloidal-`m` offset. |
| Normalization | Radius in `[0, 1]`; maximum absolute harmonic amplitude normalized to one by NOVA. |
| `datcon<N>` | Corresponding TAE continuum bounds `low2(r)` and `high2(r)`, in squared-frequency units consistent with `omega**2`. Each participating `N<N>` directory needs a matching profile. |

The current loader reads the NOVA binary payload, and feature extraction
interprets its `nr` samples on a uniform normalized grid (`linspace(0, 1, nr)`).
Supporting a different file format or a nonuniform radial grid requires a
corresponding loader/feature adaptation, rather than only changing `nr`.
Shot preprocessing also requires at least `4*ntor` stored harmonics, finite
metadata and amplitudes, and positive total amplitude-squared weight. Failed
checks produce INVALID rows. The loader preserves the stored amplitudes;
the sorter expects NOVA's normalization and does not renormalize arbitrary
input amplitudes before applying the gates.

Use mode and continuum files from the same calculation. Recalculated files
can retain their old names; a filename match alone does not establish
provenance. The [training-shot provenance audit](../scripts/README.md#training-shot-provenance-audit-audit_training_provenancepy)
documents a content-based comparison workflow.

The sorter requires the checked-in
[known-invalid-input registry](../configs/known_invalid_inputs.csv), which
matches exact shot basenames and toroidal-mode scopes before classification.
Excluded inputs appear in `rejected_modes.csv` and cannot be restored by a
morphology override. Missing or malformed registry data aborts the run.
See [input validity details](../scripts/README.md#known-invalid-inputs-in-shot-sorting)
for confirmed exclusions and the review process for corrected inputs.

Missing required `datcon<N>` files abort processing. Partially defined
continuum profiles and unsupported numerical measurements can instead limit
which diagnostics or gates apply; inspect the reported coverage. No substitute
continuum profile is invented for an uncovered region.

## Production rules sorting

After activating a rules-capable environment:

```bash
python scripts/sort_shot_mixed.py \
  --method rules \
  --shot_dir /path/to/shot \
  --out_dir /path/to/results/shot
```

The command syntax is the same in Bash and `tcsh`; environment-variable setup
differs and is shown separately in the [platform guide](platforms.md).
The command processes one shot, writes reports to `--out_dir`, and does not
move or modify input modes. Use different output directories for rules,
AI comparisons, and calibration so their reports remain distinguishable.

The current default is `tae_rules_production_v13`. To record that selection
explicitly in a saved command, add:

```text
--rule_config tae_rules_production_v13
```

The preset controls routing, gate thresholds and enable states, and duplicate
ranking. Configuration-owned overrides are rejected. See the
[configuration inventory](../configs/rules/README.md) for older presets and
their compatibility requirements.

The workflow validates inputs, routes EAE-like modes aside, and evaluates
TAE-like plus mixed modes. A rejection gate produces `BAD`. A survivor retains
the preliminary verdict `REVIEW` with reason `NO_GOOD_TEMPLATE`; the separate
`accept-as-good-v1` production policy promotes it to final `GOOD`. Manual
overrides follow this step, before deduplication. Use the final decision or
selected list for curated selections, while retaining the preliminary verdict
to understand the automatic evidence.

Current rules rank duplicates by lowest `overall_rule_severity`, with mode-key
ordering for exact ties. Both frequency proximity and structural similarity
are required. Severity and `rule_margin=1-overall_rule_severity` are threshold
diagnostics, not RF probabilities. Missing enabled-gate severity retains the
affected cluster with `SKIPPED_SEVERITY_UNAVAILABLE`.

The current rules example needs no `--rf_model` or `--cnn_model`. The `--device`
and `--make_plots` options belong to the RF+CNN method; use the standalone
viewers for rules plots.

## Read the outputs

| File | Use |
| --- | --- |
| `good_tae_final.csv` | Final GOOD TAE-side representatives after deduplication. |
| `good_tae_unchecked.csv` | Final GOOD modes before deduplication, including promotions and manual adjudications. The filename does not mean every row lacks human review. |
| `bad_tae_like.csv` | TAE-side modes rejected by gates or a manual override. |
| `review_tae_like.csv` | Final REVIEW modes, including manual REVIEW decisions and otherwise accepted survivors held for review because their supplied overrides are stale or ambiguous. |
| `eae_like.csv` | Modes routed outside the TAE classifier; this is not a list of EAE modes validated as physical. |
| `rejected_modes.csv` | Invalid or excluded inputs, with reasons. |
| `rule_results.csv`, `final_classifications.csv` | Preliminary/final decisions, features, severity, and promotion/override provenance. |
| `shot_summary.csv` | Human-readable key/value summary, including configuration and survivor-policy identity. |
| `shot_summary_wide.csv`, `shot_summary_by_n.csv` | Tabular run and per-`n` summaries. |
| `frequency_cluster_report.txt`, `frequency_clusters.csv` | Duplicate comparisons and representative-selection evidence. |
| `resolution_warnings.txt`, `resolution_warnings.csv` | When applicable, skipped or unavailable resolution-sensitive checks, with affected modes and gates. |

Final-mode tables contain `rad_loc` and `rad_width`, the normalized radial
energy centroid and RMS radial width. These can help compare selected modes
with beam-ion profiles before NOVA-C calculations. `rad_width` is not the
10–90% energy-span feature or FWHM. See the
[complete sorter output reference](../scripts/README.md#sort_shot_mixedpy)
for additional fields and tables.

Calibration has primarily used `nr=201`. At other resolutions the
interior-harmonic-incoherence and continuum-crossing-tail rejection thresholds
are not applied; other measurements have their own applicability conditions.
Sorting continues and a partially screened survivor can still become final
GOOD. Read resolution warnings and unavailable-severity counts alongside the
selected list; a missing measurement is not evidence that a gate passed.

## View modes and preserve manual corrections

The [viewing instructions](../scripts/README.md#view_modes_csvpy) cover browsing
CSV lists; the [labeler reference](../scripts/README.md#label_modes_fastpy)
documents keyboard controls, signed profiles, and display options. These tools
need Matplotlib in addition to the rules dependencies.

The standalone `viz/view_modes_csv.py` still needs this checkout's `src/` on
`PYTHONPATH`. Source the appropriate [platform path helper](platforms.md), or
set it for just the viewer command (works in both Bash and `tcsh`):

```bash
env PYTHONPATH=/path/to/NOVA_modes/src \
  python /path/to/NOVA_modes/viz/view_modes_csv.py \
  /path/to/results/shot/good_tae_final.csv
```

Add `--base_dir /path/to/data` when the list uses relative mode paths. The
`scripts/label_modes_fast.py` command below sets up its own imports.

To review automatic rule survivors without loading RF:

```bash
python scripts/label_modes_fast.py /path/to/shot \
  --mode-list /path/to/results/shot/final_classifications.csv \
  --csv_out /path/to/manual_overrides.csv \
  --adjudication review \
  --reviewer REVIEWER_ID \
  --no-rf
```

Selection uses the preliminary rule verdict, so promoted GOOD survivors remain
eligible. Use `--adjudication all` to inspect gate-rejected BAD modes too.
Supply a reason for each decision. Adjudication stores the SHA-256 fingerprint
of the mode plus its corresponding continuum file.

Rerun with the override file to apply corrections and rebuild selections:

```bash
python scripts/sort_shot_mixed.py \
  --method rules \
  --shot_dir /path/to/shot \
  --out_dir /path/to/results/shot \
  --manual_overrides /path/to/manual_overrides.csv
```

**Pass the override file explicitly on every rerun that should preserve those
corrections.** Its presence in an output directory does not apply it
automatically. Only unique, eligible overrides with matching current input
fingerprints are applied; stale, ambiguous, and unmatched rows are reported.
For an otherwise accepted survivor, a stale or ambiguous supplied override
changes the final decision to REVIEW (`decision_source=override_review_required`)
until it is resolved. Such a mode is excluded from the GOOD lists. The
original automatic evidence is retained. See
[manual adjudication](../scripts/README.md#manual-adjudication) and the
[completed 39-shot review](../audits/pilot39_manual_review_20260914/README.md).

## Conservative audits and rule calibration

To use the same frozen gates while leaving automatic survivors as REVIEW:

```bash
python scripts/sort_shot_rules.py \
  --shot_dir /path/to/shot \
  --out_dir /path/to/rule_audit/shot \
  --rule_config tae_rules_production_v13
```

For threshold experiments, omit `--rule_config` and use individual flags from
the [rule reference](../scripts/README.md#deterministic-rule-sorting-production-and-calibration-interfaces).
Omitting the preset is appropriate for development; it does not establish the
identity of a frozen production run. Keep experimental outputs separate.

## Optional RF+CNN comparison

First activate the AI environment for
[Flux](platforms.md#ai-models-existing-perlmutter-like-environment-on-flux) or
[Perlmutter](platforms.md#ai-models-and-gpu-execution). Routine inference uses
existing checkpoints and does not require retraining. From the repository
root, a CPU comparison is:

```bash
python scripts/sort_shot_mixed.py \
  --method rf-cnn \
  --shot_dir /path/to/shot \
  --rf_model models/nova_mode_classifier.joblib \
  --cnn_model models/nova_cnn_raw.pt \
  --cnn_model_kind cnn_raw \
  --out_dir /path/to/ai_results/shot \
  --device cpu \
  --make_plots
```

Both checkpoints are required. Use the [model inventory](../models/README.md)
to identify the intended versions; older four-shot artifacts are historical.
This method keeps RF/CNN scores and the older fusion/ranking policy. Its
`flagged_tae_like.csv` is an overlapping QC list, so a flagged mode can also
appear in a GOOD or BAD list. With `--label_csv` it writes evaluation reports;
`--make_plots` adds Matplotlib-based diagnostics. GPU runs belong on allocated
Perlmutter compute nodes as described in the platform guide.

For retraining, feature experiments, or held-out-shot evaluation, use the
[CNN](../scripts/README.md#cnn-model-scripts),
[RF](../scripts/README.md#random-forest-classifier), and
[LOSO](../scripts/README.md#run_loso_10py) references. Training labels can be
relative to `--data_dir` / `$NOVA_DATA`; this is a different input from the
live `--shot_dir` used for sorting.

## Reproduce a previous result

Keep the source commit, rule-configuration identity/hash, input fingerprints,
any manual-override file, and output summaries together. AI runs additionally
need the original checkpoint and compatible package versions. Shared
continuum preprocessing affects routing and derived features; selecting an
old rules configuration alone does not restore a historical loader. See the
[project state](project_state.md) and dated adoption audits for that history.

The [previous main README](history/readme_before_reorganization_20260914.md)
preserves all earlier main-page instructions and results. The detailed
[script reference](../scripts/README.md) remains in its original location.
