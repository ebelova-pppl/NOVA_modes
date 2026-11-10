# NOVA mode classifier: current project state

Updated November 10, 2026. Selection counts below retain their October 10/26/29
snapshot dates; November 10 adds surrogate input discovery. The complete
prior chronology is preserved locally and in Git history as described in
[research records and recovery](research_archive.md).

## Purpose and current workflow

Select physically useful Alfvén eigenmodes from existing NOVA outputs for
NOVA-C, surrogate modeling, and digital-twin analysis. The repository does not
run NOVA or calculate energetic-particle growth rates.

- Production: `scripts/sort_shot_mixed.py --method rules`, frozen
  `tae_rules_production_v13` (ruleset v25, grouped features v26, severity v2).
- Conservative calibration: `scripts/sort_shot_rules.py`; survivors stay REVIEW.
- Production survivors retain their preliminary REVIEW / `NO_GOOD_TEMPLATE`
  evidence and become final GOOD through `accept-as-good-v1`. Fingerprinted
  manual overrides then apply; severity selects frequency/structure duplicates.
- EAE-like modes are routed separately and are not validated by TAE rejection
  gates. BAE assignments currently come from explicit manual overrides.
- Rules need Python 3.10+, NumPy 2.x and SciPy, with no RF/CNN checkpoint.
  Calibration primarily used nr=201; other grids can have incomplete screening.

See [getting started](getting_started.md), [platform setup](platforms.md),
[rule configurations](../configs/rules/README.md), and the
[script reference](../scripts/README.md) for commands and gate definitions.

## Installed results and shared catalog

The inventory has **184 processed post-training cases plus 14 active training
cases (198 total)**, including equilibrium variants and time slices. R06 is
counted as processed but its whole-shot inputs are invalid. Excluding R06
leaves 197 cases. E205059A01t025 and the empty `nstxu_202806` inventory entry
have no installed results; no released shots await processing.

The [accepted catalog](../catalogs/README.md), dated October 26, contains
**8,943 selected GOOD modes**: 8,366 post-training and 577 training-shot
selections, from 190 nonempty lists among 198 installed outputs. There are
8,941 TAE and two mixed entries; nine selections have applied manual GOOD
overrides. Historical training outputs retain v1/v2 rules and have no saved
severity; post-training outputs use v13.

Review status is separate from selection: **865 reviewed, 6,590 pending,
1,488 unknown**. Pending includes 39 selected training modes with human BAD
labels, documented in the [conflict list](../catalogs/accepted_modes.review_conflicts.csv).
The remaining 6,551 pending modes await review. An accepted-list count does
not establish individual visual review.

The October 29 statistics include all 198 installed outputs: **127,974 input
modes, 32,645 TAE-like/mixed candidates, and 8,943 selected GOOD**. All eight
zero-GOOD cases are G shots, including invalid R06. These are saved-output
counts, not a fresh run with one common rule version.

Retained [shot inventory](../audits/main_dataset_shots/shot_status.csv),
[review evidence](../audits/README.md), and the
[catalog review registry](../configs/catalog_review_sources.csv) support
regeneration. Future reruns must explicitly supply applicable manual overrides,
including BAD and BAE decisions; automatic rules alone do not reproduce all
curated lists.

## Input holds and unresolved scientific work

- **E205059A01t025:** N1/N2 continuum recalculation and correspondence review
  remain required. The October 10 recheck found unchanged mode/continuum inputs.
- **G133964R06:** the entire shot remains excluded pending corrected inputs and
  review. Its completed INVALID outputs do not remove the hold.
- The [known-invalid registry](../configs/known_invalid_inputs.csv) also
  excludes N1 for nstx_135388, G121123B12, G142301W29, and G142301Y93.
  See [active whole-shot holds](../audits/main_dataset_shots/active_shot_holds.csv).
- The October 29 parameter audit found 1,415 saved mode/continuum fingerprint
  mismatches in 25 non-G shots, associated in spot checks with newer continua.
  Poster measurements use saved selections and radial metrics with current
  continua; classifications were not rerun against those updated continua.
- Five live nstx_135388/N8 files were empty: 2197, 2861, 3062, 4424, and 7331
  (all E+02). Archived training pairs still match saved fingerprints. Reconcile
  the live files separately; no input or exclusion registry was changed.
- Adjudicate the 39 training-label/catalog conflicts when requested; preserve
  current labels and selections until explicit scientific decisions are made.

## Training and model status

The active training list contains **2,327 rows from 14 shots**, with 575 GOOD
labels. Q62 and confirmed invalid N1 inputs have preserved records outside
that active list. The active RF and raw-CNN checkpoints were fitted to the
older **2,390-row August 28 snapshot**; those provenance counts are distinct
from today's training list and the 577 saved training-shot GOOD selections.

Legacy RF+CNN sorting remains available with `--method rf-cnn`. The model
pipelines are `rf_train_classify.py`, `cnn_raw.py`, `cnn_straightened.py`, and
`cnn_hybrid.py`. Training-shot agreement is not held-out validation; the G-shot
continuum regime remains challenging. See [model provenance](../models/README.md)
and [training-label records](../training_labels/README.md).

On Flux use tcsh commands; on Perlmutter use Bash. `NOVA_DATA` identifies
training data and `NOVA_DITW_ROOT` the live production input database; keep
these roles separate and follow the [platform guide](platforms.md).

## Recent work and next steps

- **November 10 surrogate input discovery:** the
  [source map](surrogate_input_sources.md) traces the three printed q sections,
  mapped profiles/geometry, TRANSP data, and per-shot frequency normalization.
  All 199 populated inventory cases have the main equilibrium files;
  NetCDF/text q and pressure agree exactly, and binary `mpout1` also matches
  q/pressure/g. E55's `outequ` is truncated despite matching mapped profiles.
  Five mapped cases contain nonzero rotation data. Historical density recipe,
  per-n overrides, and mode/equilibrium correspondence remain open checks;
  the user requested retaining the density question as unresolved. Detailed
  discovery evidence is in the ignored research archive. Scientific selections
  and processing code are unchanged.
- **October 29 poster preparation:** selected the mixed-routing example
  E204957F05t034/N7/3141 and produced total-count and G/non-G parameter tables.
  The mixed example has a fresh isolated v13 check; a full EAE-gap reconstruction
  is still needed before asserting a double-gap eigenmode. All poster assets,
  per-mode measurements, and provenance remain local archives.
- **November 9 repository cleanup:** generated outputs and detailed experiment
  artifacts are removed from Git tracking while remaining locally intact.
  Audit summaries, essential review/override/input records, and tested audit
  utilities remain tracked. The required 2,327 training fingerprints now live
  in `catalogs/evidence/` rather than depending on an ignored output file.
  The catalog, labels, scientific configurations, and model checkpoints are
  unchanged. Tracking shrank from 1,949 to 392 files; all removed paths remain
  locally intact. All 228 tests pass in a temporary checkout containing only
  retained files. Catalog regeneration exactly reproduces all 8,943 published
  rows, and all 2,871 review mappings are unchanged. Retained Markdown links,
  ignore rules, scientific-file checksums, and the 43-file poster backup were
  verified. The test run used NumPy 2.x and a temporary PyYAML dependency for
  the skill validator; the installed environments were not modified.
- Next: maintain this compact snapshot, register future fingerprinted review
  evidence before rebuilding the catalog, and resolve the input holds and
  scientific conflicts above when authorized. Store detailed new run records
  in the ignored research archive rather than expanding this document.
