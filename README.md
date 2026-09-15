# NOVA eigenmode classifier

Tools for selecting physically useful Alfvén eigenmodes from NOVA ideal-MHD
output and filtering numerical solutions and duplicates. The selected modes
support downstream NOVA-C analysis, surrogate modeling, and NSTX-U digital
twin workflows. This repository processes existing NOVA output; it does not
run NOVA or calculate energetic-particle growth rates.

**The recommended production workflow is deterministic rules.** Random Forest
(RF), convolutional neural networks (CNNs), and the combined RF+CNN method
remain available for comparison and model development.

## Start here

1. Set up Python using the [platform guide](docs/platforms.md):
   [Flux (`tcsh`)](docs/platforms.md#pppl-flux-tcsh) or
   [Perlmutter (Bash)](docs/platforms.md#nersc-perlmutter-bash).
2. Follow [Getting started](docs/getting_started.md) for input requirements,
   sorting, outputs, and optional manual corrections.
3. Use the [script reference](scripts/README.md) for detailed options,
   calibration, plotting, training, and experiments.

From this checkout, sort one shot:

```bash
python scripts/sort_shot_mixed.py \
  --method rules \
  --shot_dir /path/to/shot \
  --out_dir /path/to/results/shot
```

Replace both paths with your input shot and output directory. A shot contains
`N1`, `N2`, ... directories with `egn*` mode files and matching `datcon<N>`
continuum files. The sorter writes output lists and diagnostics without moving
or modifying the input modes. Start with `good_tae_final.csv` for the selected,
deduplicated TAE-side modes and `shot_summary.csv` for the run summary.
The default scan is `N1`–`N10`; set `--n_min` / `--n_max` explicitly for
a different toroidal-mode range.

### Which environment do I need?

| Task | Requirements |
| --- | --- |
| Current rules sorting and severity-based duplicate ranking | Python 3.10+ with NumPy 2.x and SciPy; CPU; no RF/CNN checkpoint. |
| Viewing or manually labeling modes | Add Matplotlib; use `--no-rf` with the labeler for a rules-only workflow. |
| RF/CNN inference, comparison, or training | Relevant AI packages and checkpoints; see the platform guide for compatible versions and CPU/GPU setup. |

**The Perlmutter-like environment on Flux is needed for the AI-model
workflow, not for current rules sorting.** It keeps package versions compatible
with the Perlmutter-trained checkpoints. You may reuse that environment for
rules, but a smaller NumPy/SciPy environment is sufficient. The files under
`configs/paths/` set site-specific paths and convenience helpers; they do not
install packages and are optional when passing explicit input/output paths.

Historical frozen rules presets v5–v10 used RF for duplicate ranking. The
current preset uses rule severity and needs no model checkpoint. See
[environment details and troubleshooting](docs/platforms.md), including Flux
Bash setup and the existing AI environment's Narwhals dependency repair.

## What the production sorter does

1. Validates modes and continuum inputs, including the
   [known-invalid-input registry](configs/known_invalid_inputs.csv).
2. Routes valid modes into TAE-like, mixed, and EAE-like groups. TAE-like and
   mixed modes proceed to the chosen classifier; EAE-like modes are listed
   separately and are not validated by the TAE rejection gates.
3. Applies the frozen rejection rules. A triggered gate produces `BAD`;
   a survivor retains `rule_decision=REVIEW` / `NO_GOOD_TEMPLATE` in the audit.
4. Applies the production `accept-as-good-v1` policy, promoting survivors to
   final `GOOD`, then applies any fingerprint-matched manual overrides.
5. Selects the lowest-severity representative among close-frequency,
   structurally similar final-GOOD modes. Distinct structures can survive at
   similar frequencies.

`GOOD` describes the final workflow selection; the preliminary rule verdict
and the source of any promotion or manual correction remain available. Rule
severity measures proximity to rejection criteria, not a probability of
being physical.

The current production preset is
[`tae_rules_production_v13`](configs/rules/tae_rules_production_v13.yaml).
[Configuration documentation](configs/rules/README.md) identifies the frozen
versions; [detailed gate definitions](scripts/README.md#deterministic-rule-sorting-production-and-calibration-interfaces)
explain the measurements and exceptions. Exact-point continuum gate 3 is
disabled in the current preset; the continuum-window gate and its approved
exceptions are active. Configuration-owned thresholds cannot be overridden in
a named production run.

For conservative audits, use `scripts/sort_shot_rules.py`: survivors remain
`REVIEW` without production promotion. For the older combined classifier,
select `--method rf-cnn` and supply both model checkpoints. See
[commands and outputs](docs/getting_started.md).

## Scope and data quality

- Inputs use normalized radius in `[0, 1]` and NOVA mode amplitudes whose
  largest absolute harmonic amplitude is one. Mode arrays use harmonic and
  radius axes; see the [input contract](docs/getting_started.md#required-inputs).
- Continuum profiles must correspond to the eigenmode calculation. Known
  invalid shot/`n` scopes are excluded before either classification method;
  missing required continuum files abort shot processing. Diagnostic and
  exclusion evidence remains available in the outputs.
- Calibration has primarily used `nr=201`. Other radial grids can receive
  only partial screening: unavailable gates are reported in
  `resolution_warnings.txt` and `.csv`, and production survivors can still be
  promoted to `GOOD`. Check coverage as well as the final label.
- Model comparisons and rules calibration have different evaluation histories.
  The G-shot continuum regime has been particularly challenging for AI
  models. Use the dated validation/audit records when interpreting metrics.

## Documentation and project context

| I want to… | Read |
| --- | --- |
| Sort a shot and understand its results | [Getting started](docs/getting_started.md) |
| Set up Flux or Perlmutter; use CPU/GPU helpers | [Platform environments](docs/platforms.md) |
| Find a script, gate definition, or experiment command | [Script reference and task index](scripts/README.md) |
| Identify a frozen rules configuration | [Rules configurations](configs/rules/README.md) |
| Use or retrain RF/CNN models | [Model inventory](models/README.md) and [training instructions](scripts/README.md#cnn-model-scripts) |
| Understand training labels and archived datasets | [Training-label documentation](training_labels/README.md) |
| Resume development with the scientific context | [Project state](docs/project_state.md) and [repository instructions](AGENTS.md) |
| Find earlier results, version notes, and README instructions | [Preserved README reference](docs/history/readme_before_reorganization_20260914.md) |

As of September 15, 2026, the processed collection contains **40 post-training
shots plus 14 active training shots (54 total)**. Elena accepted the recalculated
E202806A02t045 rules results, adding 90 modes to the
[1,726 accepted representatives from the 40 processed shots](audits/processed40_20260915/accepted_tae_modes.csv).
The new shot was reviewed using rules; the
[39-shot disagreement review](audits/pilot39_manual_review_20260914/README.md)
is complete, including 17 reasoned manual corrections. Future reruns must
explicitly supply the affected shots' `manual_overrides.csv` to reproduce
those curated selections. The remaining database is deferred while input
consistency and rollout are reviewed. See the full
[project state](docs/project_state.md) for context and next steps.

The active training list has 2,327 rows from 14 shots; Q62 and confirmed invalid
N1 inputs have preserved records outside that active list. Existing AI
checkpoints retain their earlier 2,390-row training provenance. Historical
training counts and model metrics belong to their original snapshots, rather
than being interchangeable with today's active list.

The full `project_state.md` and detailed script instructions are retained.
The previous main README is preserved in the linked dated reference, including
its model comparisons, adoption notes, audit links, calibration options, and
older setup recipes. The `tae_only_baseline_v1` tag retains the earlier
TAE-only baseline; the former mixed-data branch has been merged into `main`.
