Repo for NOVA mode classifier scripts

# NOVA AE classifier

Main context files:
- `AGENTS.md` — repo rules/instructions for Codex
- `docs/project_state.md` — current project state and model status
- `scripts/README.md` — detailed script inventory and usage notes

Current canonical pipelines:
- `scripts/split_tae_eae.py`
- `scripts/rf_train_classify.py`
- `scripts/cnn_raw.py`
- `scripts/cnn_straightened.py`
- `scripts/cnn_hybrid.py`
- `scripts/cnn_classify.py`
- `scripts/sort_shot.py`
- `scripts/sort_shot_mixed.py`
- `scripts/run_loso_10.py`
- `scripts/label_modes_fast.py`

Shared scripts / features:
- `src/mode_features.py`
- `src/cont_features.py`
- `src/mode_transform.py`
- `src/nova_mode_loader.py`
- `scripts/cnn_infer_common.py`
- `scripts/make_tae_like_list.py`
- `scripts/tae_rule_engine.py`
- `scripts/sort_shot_rules.py`
- `scripts/audit_training_provenance.py`
- `src/continuum_noise.py` and `scripts/audit_continuum_noise.py` — continuum-side
  noise gate and threshold calibration; see the
  [2026-09-10 calibration](audits/continuum_noise_20260910/README.md).

Training-data provenance:

- latest Flux training-vs-DiTw audit:
  `audits/training_provenance/2026-08-20_flux_v1/report.md`
- versioned file hashes and exact differences are stored beside that report

Plotting:
- `viz/view_modes_csv.py`
- `viz/plot_straightened_mode.py`

Known invalid inputs (2026-09-09):

- `configs/known_invalid_inputs.csv` records user-approved shot/ntor exclusions;
  `ntor=*` excludes a whole shot across all n.
  Both sorting methods and `make_tae_like_list.py` apply them before gap
  routing or classification, retain them in `rejected_modes.csv`, and report
  `n_known_invalid_inputs` in shot/per-n summaries. Diagnostics include the
  issue, reviewer, evidence, and registry hash.
- All 73 `nstxuG142301C50/N1` modes are INVALID because eigenmode structure
  does not correspond to the continuum. The exact scope stays excluded on
  reruns until corrected inputs are reviewed and the registry entry removed.
  Raw files remain readable for diagnosis. See the
  [C50 audit](audits/c50_n1_alignment_20260909/README.md).
- All 610 `nstxuG133964R06` modes are INVALID after the user's visual review:
  poor eigenmode structures throughout, with spectra peaking at the largest
  retained poloidal harmonic in some modes. The cause is unconfirmed. Both
  output sets exclude the whole shot from usable lists until corrected data
  are reviewed. See the [R06 audit](audits/r06_input_validity_20260909/README.md).
- NOVA calculates eigenfrequencies and eigenmode structure; these diagnostics
  should refer to eigenmode calculations, not stability calculations.

Production v11: normalized gate severity and rules-only ranking (2026-09-10):

- Every gate now reports its normalized severity, component ratios, and actual
  fired flag. CSV outputs expose `gate_severity_BAD_*`, `overall_rule_severity`,
  `rule_margin=1-overall_rule_severity`, and `nearest_gate`, with configuration
  hashes. These are threshold margins, not calibrated probabilities.
- The lowest overall severity selects frequency/structure duplicate
  representatives; exact ties use the mode key. No RF checkpoint is needed.
  Classification gates, thresholds, and exceptions are unchanged. Frozen v5-v10
  configurations retain their RF ranking policy.
- Preset v11 retains the v22 rejection ruleset and uses grouped feature schema
  v23 with `severity_features`. See the
  [severity audit and definitions](audits/rule_severity_20260910/README.md).

Production rules v10 (2026-09-10):

- `BAD_EXTENDED_CONTINUUM_NOISE` is a separate final gate. A connected
  above-upper or below-lower TAE-gap region must satisfy all three inclusive
  cuts: HF energy / full-domain top-two harmonic energy **>=0.01**,
  HF / local raw energy **>=0.20**, and effective radial length **>=0.04**.
- The high-pass operator remains the native-grid signed second difference
  divided by four. Energies use radial quadrature weights; effective length
  is `N_r,eff/(nr-1)`. The gate runs on every supported native grid without
  resampling. At nr=201, length 0.04 is eight effective radial points.
- Frozen v5-v9 presets disable this gate. Current v10 uses ruleset/features
  v22 and records full regional evidence under
  `numerical_structure_features.extended_continuum_noise`. Earlier BAD reasons
  keep precedence. See the [calibration and adoption audit](audits/continuum_noise_20260910/README.md).
- Verified against all training entries and the 27-shot batch: no GOOD training
  label is newly rejected; the additional rejections are training BAD
  E204669M03t025 N10/1295 and batch E204186A01t020 N10/1271. Rules/RF–CNN
  disagreements decrease from 227 to 226. All 185 tests pass.

Retained axis-energy rule (adopted in v9, 2026-09-09):

- New `axis_energy_concentration` gate: reject as
  `BAD_AXIS_ENERGY_CONCENTRATION` when **both** maximum absolute harmonic
  amplitude inside `r<=0.015` exceeds 0.5 and more than 50% of integrated
  all-harmonic radial energy lies inside `r<=0.05`. No width or local-peak
  condition is imposed. The gate preserves all earlier BAD
  reasons, and evaluates every native radial resolution. See the
  [axis-energy audit](audits/axis_amplitude_20260909/README.md).
- Frozen v5-v8 presets explicitly disable this new gate. New exports record
  its measurements under `boundary_features.axis_energy_concentration` and
  its enable state and four thresholds in shot/per-n summaries.

- The retained v8 narrow interior-envelope exception requires relative frequency
  clearance **strictly greater than 0.1%**: `0.001 < ext_df_gap <= 0.04`.
  No minimum-width floor was added. Clearance is divided by mode frequency;
  the existing `r_peak<=0.5`, energy-FWHM<=2 applicability and radial match
  remain. See [the clearance audit](audits/extremum_floor_20260909/README.md).
- Frozen v5-v7 presets retain their inclusive zero-clearance lower bound.
  Current audit features and shot/per-n summaries explicitly record whether
  the clearance lower comparison includes equality.

- `sort_shot_mixed.py --method rules` now uses
  `configs/rules/tae_rules_production_v11.yaml` (ruleset v22 / feature schema v23).
  Each offending continuum-crossing window is excused only when its
  interpolated signed-harmonic amplitude satisfies `A_cross < 0.2` and
  its native-grid roughness satisfies `K_c < 0.1`, on nr=201. Every offending
  crossing must qualify; all other rejection gates still apply.
- V5/v6 configuration files remain frozen and explicitly disable the smooth-crossing
  exception when loaded. Their decisions remain supported; new exports use
  the current v23 audit schema. RF/CNN feature columns and weights are unchanged.
- The earlier smooth-crossing impact audit recovered 19 modes in the 27-shot batch and two
  labeled GOOD training modes, with no newly accepted labeled BAD training
  modes. See [the exception audit](audits/cross_window_exception_20260909/README.md).

Shared continuum preprocessing (adopted 2026-09-08):

- `src/cont_features.py` now repairs sustained steep rises of both continuum
  boundaries through their last jointly defined point. It traces back to the
  first fast step and holds each boundary at its preceding value, preserving
  NaNs and retaining the older isolated-spike cleanup as fallback. Raw datcon
  files are unchanged. Sorting, TAE/EAE splitting, RF/CNN continuum features,
  and new viewer sessions all use this same loader.
- New rules and RF-CNN shot/per-n summaries record
  `continuum_preprocessing_version=datcon-monotonic-tail-v1`. The v6 rejection
  configuration and feature column schemas are unchanged; continuum feature
  values can change. Historical output reproduction also requires its source
  checkout: commit `64fc889` is the last pre-adoption loader. Selecting v5/v6
  thresholds alone does not restore the old continuum arrays.
- The 27-shot audit retained all 899 existing automatic GOOD modes and
  recovered 41 crossing-window rejections, which the user reviewed and
  accepted. Five other BAD modes route to EAE-like. See
  [the repair audit](audits/continuum_monotonic_tail_20260908/batch_report.md)
  and `docs/project_state.md` for regeneration status. Existing model weights
  are retained; new feature extraction uses the repaired boundaries.


Project goal
- The goal of this project is to develop machine learning tools to automatically identify physical Alfvén eigenmodes (AEs) from NOVA (ideal MHD linear solver) output and filter out unphysical or numerical solutions. The long-term objective is to enable fast, reliable preprocessing of NOVA mode spectra for use in stability analysis and surrogate modeling (e.g., NOVA/NOVA-C pipelines and digital twin applications).
    Key tasks:
    - Sort TAE range modes vs EAE gap modes
    - classify NOVA eigenmodes as physical (“good”) or unphysical (“bad”)
    - remove duplicate or near-duplicate modes generated by NOVA
    - extract physically meaningful features from mode structure and continuum data
    - provide a clean, curated dataset for downstream modeling (e.g., growth rates, transport, surrogate models)

Data format summary
- Each NOVA mode file contains:
    - 2D mode structure: A(m,r)
    - scalar quantities: omega — mode frequency, gamma_d — continuum damping estimate, ntor — toroidal mode number
    - The dataset can contain modes from TAE and EAE frequency ranges
- Typical shapes: mode: (n_m, n_r)
    - n_m (number of poloidal harmonics) varies with ntor
    - n_r (radial grid) may vary between shots
- Additional data:
    - continuum data (datcon#) one per shot / per ntor: low2(r), high2(r) (Alfvén continuum bounds - currently for TAE gap only)
- Training label CSVs in `training_labels/` store mode paths relative to
  `$NOVA_DATA` when possible, for example
  `nstx_120113/N5/egn05w.1234E+02,good`. The current canonical/default
  good/bad training list is `training_labels/tae_like_train.csv`. It contains
  2,390 modes from 14 shots, with 575 GOOD and 1,815 BAD labels. Q62 is
  suspended because a whole-shot visual audit indicates that its upper
  continuum boundary may be incorrect. The complete 2,639-row reviewed
  15-shot snapshot, including the preserved 249 Q62 labels, remains in
  `training_labels/tae_like_v3.csv`. Older four-shot TAE-only and mixed
  TAE/EAE lists are archived under
  `training_labels/old_4shots_tae_only_labels/` and
  `training_labels/old_4shots_mixed_labels/`.
- Internal conventions:
    - radial coordinate normalized to [0,1]
    - mode amplitudes normalized (max amplitude = 1)
- derived features include:
    - radial centroid r0
    - quantile width dr (10–90% energy span)
    - continuum interaction metrics (delta2_eff, S, W_star, r_star,
      W_star_max)

Model families
- `scripts/rf_train_classify.py` - Random Forest classifier using engineered scalar features (mode structure + continuum-related quantities)
- `scripts/cnn_raw.py` - CNN using raw (m,r) mode structure, with r-resampling and m-axis padding/cropping
- `scripts/cnn_straightened.py` - CNN using ridge-aligned (straightened) mode representation
- `scripts/cnn_hybrid.py` - CNN + scalar features (continuum + physics-informed inputs)

Current best models
- Active expanded-set models live at `models/nova_mode_classifier.joblib` and
  `models/nova_cnn_raw.pt`. The 2026-08-28 refresh used the then-current canonical
  `training_labels/tae_like_train.csv`: 2,390 rows from 14 shots, with 576
  GOOD and 1,814 BAD labels and Q62 excluded. Both saved checkpoints are
  full-list refits; the RF uses the production 22-feature schema and the raw
  CNN uses `M_target=100` and `R_target=201`.
- The refreshed RF run reports mean five-fold row-wise CV accuracy `0.9448` on
  all 2,390 rows. Its 239-row stratified holdout has CM
  `[[169, 12], [4, 54]]`, accuracy `0.933`, and GOOD
  precision/recall/F1 `0.818 / 0.931 / 0.871`.
- The refreshed raw-CNN split check with `M_target=100` has CM
  `[[353, 9], [13, 102]]`, accuracy `0.9539`, and GOOD
  precision/recall/F1 `0.919 / 0.887 / 0.903`. The saved checkpoint is a
  fresh 80-epoch refit on all 2,390 rows with no prediction-collapse warning.
- Previous pre-B12 RF 13-shot OOF check: CM `[[1967, 37], [91, 515]]`, accuracy
  `0.951`, GOOD recall `0.850`, GOOD precision `0.933`, GOOD F1 `0.889`.
- An opt-in 25-feature RF experiment adds inner continuum-extremum radial
  mismatch, signed frequency clearance, and the fraction of total mode energy
  within `|r-r_e| <= 0.03`. It reduced shuffled-fold FN from `92` to `89`, but
  true shot-wise LOSO FN remained `130` and G-shot FN changed `31 -> 32`. The
  active 22-feature model is unchanged; see `docs/project_state.md`.
- Previous pre-B12 raw-CNN 13-shot held-out split check with `M_target=100`: CM
  `[[394, 6], [9, 112]]`, accuracy `0.971`, GOOD recall `0.926`, GOOD
  precision `0.949`, GOOD F1 `0.937`. The raw-CNN default harmonic window is
  now `M_target=100`.
- Previous production raw-CNN 10-shot held-out check: CM
  `[[290, 5], [8, 121]]`, accuracy `0.969`, GOOD recall `0.938`, GOOD
  precision `0.960`, GOOD F1 `0.949`. That saved checkpoint was then refit on
  all 2,125 labels for 80 epochs, ending at loss `0.0008`.
- Symmetric OneCycleLR + gradient-clipping 10-shot LOSO check:
  CNN CM `[[1402, 74], [67, 582]]`, accuracy `0.934`, GOOD recall `0.897`,
  GOOD precision `0.887`. This is now the strongest aggregate LOSO result,
  although the NSTX-U G-case folds remain the weakest group.
- Previous four-shot RF/CNN checkpoints have been archived under
  `models/old_4shots_models/`.
- `sort_shot_mixed.py` is the canonical production orchestrator. Its default
  `--method rules` path loads the immutable `tae_rules_production_v11`
  configuration; `--method rf-cnn` preserves the older RF-leaning fusion
  policy as an explicit legacy option. The rule and AI decision engines stay
  separate while sharing validation, routing, output, and duplicate-removal
  conventions.
- Under the production rule method, a mode that fires no rejection gate keeps
  the engine verdict `rule_decision=REVIEW` and
  `rule_primary_reason=NO_GOOD_TEMPLATE`, then the audited
  `accept-as-good-v1` workflow policy promotes it to final `GOOD`. Manual
  overrides are applied after that policy. The production command chooses the
  lowest-severity representative among close-frequency, structurally matched
  duplicates, with no model checkpoint required.

## Sort new NSTX-U shots on Flux (no training)

For a user who only wants to sort new NOVA output, do **not** train new
models. Run the canonical `scripts/sort_shot_mixed.py` workflow once per shot.
The default method is deterministic rules and loads the frozen
`tae_rules_production_v11` configuration automatically:

```text
rejection gate fired -> BAD
all gates passed     -> rule REVIEW/NO_GOOD_TEMPLATE
                     -> final GOOD by accept-as-good-v1
manual overrides     -> applied to the automatic final decision
final GOOD           -> severity-ranked frequency/structure deduplication
```

The promotion is a workflow policy, not a positive rule-engine template. Both
the preliminary rule verdict and the audited promotion source remain in the
outputs. Production v11 ranks duplicates by lowest overall rule severity,
with the mode key breaking exact ties. No RF or CNN checkpoint is needed.
Missing enabled-gate severity retains the affected frequency cluster and
reports `SKIPPED_SEVERITY_UNAVAILABLE`. Frozen v5-v10 configurations retain
the previous RF ranking and its missing-model fallback.

Default Flux shell is usually `tcsh`:

```tcsh
module load anaconda3
source `conda info --base`/etc/profile.d/conda.csh
setenv CONDA_PKGS_DIRS /p/hym/conda_pkgs
conda activate /p/hym/conda_envs/nova-perlmutter

cd /path/to/your/NOVA_modes
git pull
source configs/paths/nova_paths.flux.csh
nova_env

setenv NOVA_DITW_ROOT /p/nstxdigtwin/energetic_particles/nova/DiTw
setenv NOVA_SORT_OUT "$NOVA_DITW_ROOT/sort_outputs"
mkdir -p "$NOVA_SORT_OUT"
```

The shared Flux environment currently uses scikit-learn `1.9.0` to match the
Perlmutter-trained RF checkpoint. Scikit-learn `1.9.0` requires
`narwhals>=2.0.1`; do not use `--no-deps` when installing it. To repair an
earlier no-dependency upgrade in the activated environment, run:

```tcsh
python -m pip install "narwhals>=2.0.1"
python -m pip check
python -c "import sys, sklearn, narwhals; print(sys.executable); print('sklearn', sklearn.__version__); print('narwhals', narwhals.__version__)"
```

Bash users should replace the `tcsh` setup lines with:

```bash
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS=/p/hym/conda_pkgs
conda activate /p/hym/conda_envs/nova-perlmutter

cd /path/to/your/NOVA_modes
git pull
source configs/paths/nova_paths.flux.sh
nova_env

export NOVA_DITW_ROOT=/p/nstxdigtwin/energetic_particles/nova/DiTw
export NOVA_SORT_OUT="$NOVA_DITW_ROOT/sort_outputs"
mkdir -p "$NOVA_SORT_OUT"
```

Then classify one shot by replacing `nstxu_example_shot` with the shot
directory name under `$NOVA_DITW_ROOT`.

For `tcsh`:

```tcsh
setenv SHOT_NAME nstxu_example_shot
python "$NOVA_REPO/scripts/sort_shot_mixed.py" \
  --method rules \
  --shot_dir "$NOVA_DITW_ROOT/$SHOT_NAME" \
  --out_dir "$NOVA_SORT_OUT/$SHOT_NAME"
```

For bash:

```bash
export SHOT_NAME=nstxu_example_shot
python "$NOVA_REPO/scripts/sort_shot_mixed.py" \
  --method rules \
  --shot_dir "$NOVA_DITW_ROOT/$SHOT_NAME" \
  --out_dir "$NOVA_SORT_OUT/$SHOT_NAME"
```

The input shot directory is expected to contain `N1`, `N2`, ... subdirectories
with `egn*` mode files and matching `datcon<N>` continuum files. The sorter
does not move or modify the input modes; it writes CSV outputs and reports into
`$NOVA_SORT_OUT/$SHOT_NAME`.

The current `scripts/` command-line entry points locate this checkout's
`src/` directory relative to their own files. They do not require an inherited
`PYTHONPATH` merely to start. Continue sourcing the platform path config for
data/model environment variables and convenience helpers; `PYTHONPATH` remains
useful for interactive imports and test discovery but is no longer a hidden
precondition for direct CLI commands.

Most useful outputs for the default rules method:

- `good_tae_final.csv` — final GOOD TAE-like representatives after severity-ranked
  frequency/structure deduplication.
  Includes `rad_loc` and `rad_width` for comparing the mode location/width
  with beam-ion density profiles before launching NOVA-C growth-rate runs.
- `good_tae_unchecked.csv` — promoted or manually adjudicated GOOD TAE-like
  modes before duplicate handling.
- `bad_tae_like.csv` — TAE-like modes rejected by deterministic gates or a
  manual override.
- `review_tae_like.csv` — modes whose final classification remains REVIEW;
  automatic production survivors are promoted and do not remain here.
- `rule_results.csv` and `final_classifications.csv` — preliminary rule
  verdicts, survivor-policy provenance, and final decisions.
- `eae_like.csv` — valid modes routed away as EAE-like.
- `shot_summary.csv` and `frequency_cluster_report.txt` — run summary and
  duplicate-cluster audit, including rule-configuration and survivor-policy
  identity.

To reproduce the legacy RF+CNN classifier and fusion workflow, select it
explicitly and provide both checkpoints:

```tcsh
python "$NOVA_REPO/scripts/sort_shot_mixed.py" \
  --method rf-cnn \
  --shot_dir "$NOVA_DITW_ROOT/$SHOT_NAME" \
  --rf_model "$NOVA_REPO/models/nova_mode_classifier.joblib" \
  --cnn_model "$NOVA_REPO/models/nova_cnn_raw.pt" \
  --cnn_model_kind cnn_raw \
  --out_dir "$NOVA_SORT_OUT/$SHOT_NAME" \
  --device cpu \
  --make_plots
```

The legacy method requires both model checkpoints. Do not use checkpoints
from `models/old_4shots_models/` for new runs. Its
`flagged_tae_like.csv` remains an overlapping RF/CNN disagreement and
borderline-QC list. NSTX-U G-case shots are a distinct AI-model regime; do not
interpret legacy RF+CNN results there as equivalent to the deterministic
production rules.

If the goal is only to classify new shots, do not run `rf_train_classify.py`,
`cnn_raw.py`, `cnn_straightened.py`, or `cnn_hybrid.py`. Those scripts are for
developing or retraining models, not for routine sorting.

## Deterministic TAE rule sorting

For production rule sorting, use the canonical mixed-shot orchestrator. Rules
are the default, although spelling out the method is recommended in saved run
commands:

```bash
python scripts/sort_shot_mixed.py \
  --method rules \
  --shot_dir /path/to/shot \
  --out_dir /path/to/rule_sort_output
```

`configs/rules/tae_rules_production_v11.yaml` pins the routing values, ruleset,
gate enable states, and thresholds calibrated and audited non-blindly on the
14 active shots and the held-out pilot review. Gates 1, 2, 2b, the near-axis
grid-oscillation gate, 4, 5, the interior-envelope gate, the interior
harmonic-incoherence gate, continuum crossing-tail gate, and final
axis energy-concentration gate are enabled; exact-point continuum gate 3
is explicitly disabled. The sorter
records the configuration name, schema, and SHA-256 in
all shot and per-`n` summaries and rejects threshold/gate overrides when the
named configuration is selected. It also records the `accept-as-good-v1`
survivor policy that promotes pass-all-gates `REVIEW` rows to production
`GOOD` before manual overrides and duplicate processing.

Production v10 retains v6 routing: `fraction_below_upper2 < 0.2` goes directly
to EAE-like, regardless of `signed_delta`. It retains v7's approved gate-4
exception and tightens the interior extremum clearance as described above.
The fraction counts mode energy where the upper TAE
boundary is defined. V5 remains available via `--rule_config
tae_rules_production_v5`, with its original routing; saved v5 output lists
require regeneration to reflect v6. Standalone split and RF-CNN workflows
also default to the shared v6 routing and accept
`--fraction_direct_eae_threshold 0` for the previous condition.

For conservative rule auditing and threshold development, use the separate
calibration CLI. It does not apply the production survivor policy: pass-all-
gates modes remain `REVIEW`, never automatic `GOOD`.

```bash
python scripts/sort_shot_rules.py \
  --shot_dir /path/to/shot \
  --out_dir /path/to/rule_audit_output
```

Both paths reuse the `sort_shot_mixed.py` input validation and TAE/EAE/mixed
routing conventions. The shared engine implements twelve ordered BAD decisions;
production-v5 enables nine of them and disables exact-point continuum gate 3.
The decisions reject a calibrated narrow local maximum at `r <= 0.03`
(`BAD_AXIS_SPIKE`), a large unresolved
signed lobe (`BAD_GRID_SCALE_SPIKE`), a short packet containing repeated large
grid-scale turning points (`BAD_GRID_SCALE_PACKET`), a near-axis run of at
least four strictly consecutive sign flips with mode-level
`max_{h,r<0.1}|A_h| >= 0.10` and strongest single-run
`Q_s=sqrt(sum(delta_A^2)) >= 0.30`
(`BAD_NEAR_AXIS_GRID_OSCILLATION`), or a true continuum crossing
with `W_star_max > 0.03` (`BAD_CONT_CROSS`). A second crossing gate rejects when an
inclusive ±2-grid neighborhood of any true crossing has individual-harmonic
absolute amplitude at least `0.25` or peak-normalized radial energy at least
`0.05` (`BAD_CONT_CROSS_WINDOW`). The edge gate rejects a global total-energy
peak at `r >= 0.97` whose FWHM is no greater than 10 grid intervals
(`BAD_EDGE_SPIKE`). The following gate rejects a global total-energy envelope with
connected FWHM no greater than two grid intervals and peak at `r <= 0.5`, unless
the peak is aligned with a gate-specific inner continuum extremum within
`ext_dr <= 0.02` and `0.001 < ext_df_gap <= 0.04`
(`BAD_INTERIOR_UNRESOLVED_ENVELOPE`). Next, a calibrated 201-point mode is
rejected as `BAD_INTERIOR_HARMONIC_INCOHERENCE` when
`f_core * J_core * N_eff_core * (1 - C_adj) > 0.10`. This score combines the
energy fraction at `r <= 0.5`, base-2 adjacent-radius harmonic-distribution
roughness, the energy-weighted mean pointwise effective harmonic count, and
low signed-profile coherence between adjacent active stored harmonic rows.
The gate fails open at other radial resolutions or when coherence evidence is
undefined. The same extractor retains audit-only harmonic-participation
summaries with a fixed strict reference `N_eff(r) > 3`: the whole-radius
`W(r)`-weighted mean effective count, the whole-radius energy fraction above
that reference (`G_3`), its core-conditional counterpart, and the fraction of
total mode energy in the core above that reference (`B_3,core`). These
measurements do not affect `candidate_found` or any classification. The axis
gate checks every
absolute-harmonic local maximum
centered at `r <= 0.03`; the active defaults require normalized amplitude at
least `0.2` and full width at half maximum no greater than `10` radial-grid
intervals. The grid-scale defaults require amplitude at
least `0.3` and width no greater than one grid interval through `r=0.7`, but
use the stricter width limit `0.75` for peaks at `r>0.7`.
The packet gate scans every five-sample window of every stored harmonic. A
sharp turn requires two adjacent signed-amplitude steps of magnitude at least
`0.2` whose signs oppose. The gate rejects when the window's absolute peak is
at least `0.3`, its peak is centered at the inclusive radius `r <= 0.5`, and
all three possible interior samples are sharp turns.
The near-axis grid-oscillation gate instead uses maximal strictly consecutive
nonzero sign-flip runs and never bridges a missing flip. It selects the one
single-harmonic run with largest `Q_s`; runs and harmonics are not summed. Both
the run peak and the independent all-harmonic amplitude search use strict
`r < 0.1`, while the amplitude and `Q_s` thresholds are inclusive. Other
valid TAE-side
modes receive the engine verdict `REVIEW`, not `GOOD`, with primary reason
`NO_GOOD_TEMPLATE`. The production orchestrator then promotes those survivors;
the calibration CLI leaves them as REVIEW. Invalid inputs remain `INVALID`,
and valid EAE-like modes are routed without a fabricated rule decision.

The `continuum_crossing_tail` gate returns
`BAD_CONTINUUM_CROSSING_TAIL` when the **same** actual lower/upper crossing
has strict `K_c > 0.4` and `T_2 > 0.035` on a 201-point radial grid.
`T_2=E_tail/(E_h1+E_h2)` uses the two strongest individual harmonics by
full-domain integrated squared amplitude; the tail numerator includes all
harmonics on the side opposite the global energy peak. K is the unscaled
signed second-difference norm divided by the local amplitude norm, over
complete stencil centers within ±4 native intervals of that crossing.
Other resolutions retain measurements but do not fire this gate. The gate
preserves the preceding primary reasons. The axis-energy gate and extended continuum noise gate follow it. Details and calibration
controls are in `scripts/README.md`; older frozen presets remain unchanged.

If an evaluated TAE-side mode has `nr != 201`, the interior harmonic-incoherence
and continuum crossing-tail gates do not apply their rejection thresholds.
Other enabled gates still run on the native mode grid. Both sorter entry
points print a prominent stderr warning naming affected gates and counts,
and write `resolution_warnings.txt` plus a per-mode/per-gate
`resolution_warnings.csv`. Sorting continues: the production survivor policy
can still mark affected survivors GOOD, which the warning states explicitly.
This reports partial screening; it does not abort or resample mode profiles.

Each valid TAE-side result includes a grouped, deterministic `rule_features`
object containing the active RF 22-feature calculations, six crossing
summaries, crossing-window amplitude and energy evidence, raw crossing records,
three continuum-extremum measurements, axis/edge boundary measurements,
separate unresolved-interior-envelope evidence, and the components of the
interior harmonic-incoherence score. Its grouped audit schema is
`tae-rule-features-grouped-v23`; the near-axis group records the independent
mode-level amplitude maximum and complete strongest single-harmonic sign-flip
run evidence. The inherited participation summaries remain scalar
energy-weighted evidence rather than an unweighted pointwise maximum. The
interior-envelope search uses extrema
through `r=0.50` without changing the experimental RF feature definition,
which retains its established `r<=0.40` search.
Half-maximum boundary widths use the complete radial grid, not only their
search windows. The edge decision uses the global normalized total-energy
envelope; the strongest individual edge harmonic is recorded for audit but
does not fire the gate alone. NOVA radius and mode amplitude are already
normalized; harmonic identifiers are reported as zero-based stored indices
without inferring a physical poloidal-`m` offset. This reuses the calculations
without running an RF model.

In `sort_shot_rules.py` calibration runs, override the calibrated defaults
with `--axis_amplitude_min VALUE`,
`--axis_width_max_grid VALUE`, `--grid_scale_amplitude_min VALUE`,
`--grid_scale_width_max_grid VALUE`, `--grid_scale_high_r_cutoff_r VALUE`,
`--grid_scale_high_r_width_max_grid VALUE`,
`--grid_scale_packet_amplitude_min VALUE`,
`--grid_scale_packet_step_min VALUE`,
`--grid_scale_packet_min_large_turns VALUE`,
`--grid_scale_packet_window_span_grid VALUE`,
`--grid_scale_packet_peak_r_max VALUE`,
`--near_axis_grid_oscillation_peak_r_max VALUE`,
`--near_axis_grid_oscillation_amplitude_min VALUE`,
`--near_axis_grid_oscillation_min_consecutive_sign_flips VALUE`,
`--near_axis_grid_oscillation_step_l2_min VALUE`,
`--w_cross_threshold VALUE`,
`--cross_window_half_width_grid VALUE`,
`--cross_window_amplitude_min VALUE`, `--cross_window_w_min VALUE`,
`--edge_r_min VALUE`, `--edge_width_max_grid VALUE`,
`--interior_envelope_peak_r_max VALUE`,
`--interior_envelope_width_max_grid VALUE`,
`--interior_envelope_extremum_r_min VALUE`,
`--interior_envelope_extremum_r_max VALUE`,
`--interior_envelope_ext_dr_max VALUE`,
`--interior_envelope_ext_df_gap_min VALUE`, and
`--interior_envelope_ext_df_gap_max VALUE`. The matching disable switches
retain measurements while disabling each decision gate. The run summary
records every gate's enable state and exact thresholds.

Optional production adjudication writes fingerprinted overrides separately.
Use `--adjudication review`; it selects the preserved preliminary rule verdict,
so automatic survivors remain eligible even though their final decision is
GOOD:

```bash
python scripts/label_modes_fast.py /path/to/shot \
  --mode-list /path/to/rule_sort_output/final_classifications.csv \
  --csv_out /path/to/manual_overrides.csv \
  --adjudication review \
  --reviewer REVIEWER_ID \
  --no-rf

python scripts/sort_shot_mixed.py \
  --method rules \
  --shot_dir /path/to/shot \
  --out_dir /path/to/rule_sort_output \
  --manual_overrides /path/to/manual_overrides.csv
```

An override is applied only while its SHA-256 input fingerprint matches the
current mode and corresponding `datcon#` contents. Production duplicate
ranking uses overall rule severity, after all classification and override
steps. Missing severity retains the affected cluster with an explicit fallback.

## Typical workflow (hand labeling, (re-)training, checks etc)

- Generate NOVA modes for a shot
- Label or verify training data (label_modes_fast.py)
- Added split_tae_eae.py step to sort out tae-like vs eae-like modes 
- Train classifier (RF or CNN) on the current default
  `training_labels/tae_like_train.csv`.
  For CNN
  checkpoints intended for production sorting, use
  `--refit_full_before_save` so the saved model is trained on the full labeled
  CSV after held-out evaluation. Raw-CNN split training and full refit use the
  same OneCycleLR plus gradient-clipping recipe. If a LOSO raw-CNN run
  collapses toward the majority class, try `cnn_raw.py --pos_weight auto` to
  weight missed GOOD modes by `n_bad/n_good` during training.
- Run `sort_shot_mixed.py --method rules` for the canonical production pass:
  it routes EAE-like modes away, applies the frozen rejection rules, promotes
  rule survivors under `accept-as-good-v1`, applies manual overrides, and
  uses lowest overall rule severity to select frequency/structure representatives.
- Run `sort_shot_mixed.py --method rf-cnn` with both model checkpoints only
  when the legacy combined RF+CNN decision path is required. Raw,
  straightened, and hybrid CNN checkpoints are supported through the shared
  CNN inference path; the legacy default fusion policy remains RF-leaning.
- Run `sort_shot_rules.py` for conservative deterministic calibration or
  audit runs where gate survivors must remain REVIEW.
- Run `sort_shot.py` when you want the older single-model shot sorter and
  duplicate-removal workflow.
- Use cleaned mode set for further analysis (e.g., NOVA-C, surrogate models)

Minimal install / enviroment
- On NERSC Perlmutter:
    - module load python
    - module load pytorch
    - source configs/paths/nova_paths.nersc.sh
- On PPPL Flux:
    - module load anaconda3
    - `tcsh`: ``source `conda info --base`/etc/profile.d/conda.csh``
    - `bash`: source "$(conda info --base)/etc/profile.d/conda.sh"
    - set conda package cache under `/p/hym` with `CONDA_PKGS_DIRS`
    - conda activate /p/hym/conda_envs/nova-perlmutter
    - cd /path/to/your/NOVA_modes
    - source configs/paths/nova_paths.flux.csh
    - Bash users can source configs/paths/nova_paths.flux.sh instead
    - Flux configs set repo/model/training-list paths only; use explicit
      `--shot_dir` / `--out_dir` paths or set workflow-specific data/output
      variables yourself
    - Flux runs the CNN scripts on CPU by default via `NOVA_TORCH_DEVICE=cpu`
    - Perlmutter-trained CNN checkpoints have been cross-checked on Flux with
      identical RF / raw / straightened / hybrid inference outputs
- Python packages:
    - numpy
    - scipy
    - scikit-learn 1.9.0 (RF)
    - narwhals >=2.0.1 (required by scikit-learn 1.9.0)
    - torch (CNN)

Notes
- This repository focuses on mode classification and preprocessing, not NOVA itself
- Scripts are designed to work with existing NOVA output directories: nstxu_123456/N1/.../N10/
- Older version for frozen TAE-only models and the old TAE-only dataset: git tag tae_only_baseline_v1
- mixed_branch was for the mixed TAE+EAE dataset and is now merged into main
