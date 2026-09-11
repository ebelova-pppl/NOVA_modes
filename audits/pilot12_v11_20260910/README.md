# Twelve new E shots: rules v11 versus RF/raw-CNN

2026-09-10. The user authorized a limited continuation of paired sorting while
the N1 problem cases are recalculated. The full-database rollout and unresolved
input scopes remain on hold. No rule threshold, model artifact, training label,
or validity exclusion is changed for this comparison.

## Completed results

All **24 sorter runs completed and were installed** in the requested output
roots. All **6,697 inputs passed preflight**, with native nr=201 and no invalid
inputs or replacements. The methods have identical input coverage and routing:
**1,817 TAE-side modes** (1,764 TAE-like plus 53 mixed) and **4,880 EAE-like**.
Every evaluated rule mode has complete severity information. No resolution or
rule-ranking fallback occurred; paired source fingerprints and routing scalars
match exactly. The inference logs contain no warnings or errors.

| Result | Rules v11 | RF/raw-CNN |
|---|---:|---:|
| GOOD before duplicate removal | 699 | 767 |
| BAD | 1,118 | 1,050 |
| Final GOOD representatives | 694 | 762 |

The methods agree on **1,673/1,817 decisions (92.07%)**. The
[144 disagreements](disagreements.csv) comprise **106 rules-BAD/AI-GOOD** and
**38 rules-GOOD/AI-BAD**. The rule rejections are axis spike (57), grid-scale
spike (19), near-axis oscillation (13), crossing window (7), interior unresolved
envelope (5), continuum-crossing tail (4), and edge spike (1).

Standalone comparisons at p_good>=0.5 give **150 rules-versus-RF** disagreements
(8.26%) and **163 rules-versus-CNN** (8.97%). These are agreement statistics
for this selected batch, not accuracy measurements.

The largest number of disagreements is E203655F01t025, with 25/249 TAE-side
modes. E204636A01t020 has 15 disagreements, all rules-GOOD/AI-BAD.
Full per-shot results are in
[shot_summary.csv](shot_summary.csv). No disagreements have been visually
adjudicated during this run.

The [verification receipt](verification.json) fingerprints all staged output
trees; the [publication receipt](publication.json) records their verified
installation in 24 new directories. Existing shot outputs were not replaced.
The [inventory update](inventory_update.json) marks exactly these twelve E
shots checked, increasing the post-training total from **27 to 39**. All
other inventory rows and the G-only inventory remain unchanged.

## Selection

Seed **20260910** selects twelve distinct E discharges absent from training
and the preceding 27 checked cases. The eligible pool contains 27 time-slice
directories from 16 E discharges. All remaining G cases have an N1 review
finding or insufficient evidence, so this batch is E-only.

Eligibility requires:

- An unchecked shot with no existing directory in either requested output root.
- Usable interior N1 and N2 comparisons, with all comparisons within two
  radial intervals in both the TAE-side and all-raw cohorts.
- Complete exact-frequency log coverage, no input errors, and no interior
  crossings in modes with empty matched singularity lists.
- No priority N1 case in the same E discharge series. This also excludes
  other times from E205059 while its flagged t025 case is investigated.

These criteria use the prior input-consistency audit, not classifier scores.
They establish eligibility for this comparison, not physical GOOD labels.
The sample is conditional on this screen and is not a representative random
sample of every DiTw shot or of G-shot performance.

Sorted size strata are shuffled independently in order high, low, medium
using one seeded Python random generator, then selected without repeating a
discharge number. Quotas are two high (>800 files), four low (<=400), and six
medium (401–800). The high stratum has only two eligible shots. Ranked pools,
excluded entries, and selected shots are preserved beside this README.

| Shot | Size | Raw input files |
|---|---|---:|
| nstxuE205055A01t022 | high | 1,017 |
| nstxuE203656A02t030 | high | 947 |
| nstxuE204678M01t017 | low | 302 |
| nstxuE205042A01t022 | low | 259 |
| nstxuE202947A03t015 | low | 188 |
| nstxuE202926A03t025 | low | 330 |
| nstxuE204636A01t020 | medium | 648 |
| nstxuE203653A02t025 | medium | 406 |
| nstxuE203981A01t025 | medium | 692 |
| nstxuE204955F02t017 | medium | 680 |
| nstxuE202944A02t021 | medium | 670 |
| nstxuE203655F01t025 | medium | 558 |

The [completed selection](selection.csv) contains **6,697 input files**, with
no preflight replacements. Every selected N1/N2 mode, continuum and log file
matched the alignment-audit snapshot before and after inference. Shared
`preprocess_shot` checked every N1–N10 mode, including binary metadata,
continuum loading, native resolution and input-validity exclusions; the
[preflight summary](preflight_summary.json) fingerprints its per-mode tables.

## Workflow and provenance

Rules use the canonical `sort_shot_mixed.py --method rules` path with frozen
`tae_rules_production_v11.yaml`: v22 rejection rules, grouped audit schema v23,
and severity-based duplicate ranking. The rules run receives no RF checkpoint.
The comparison uses `--method rf-cnn --cnn_model_kind cnn_raw --device cpu`,
the existing RF/raw-CNN checkpoints, and unchanged fusion defaults. Both paths
share the direct 20% EAE routing and `datcon-monotonic-tail-v1` continuum repair.
Checkpoint provenance remains historical: this task does not retrain after
the recently removed invalid N1 training rows.

The portable [driver](run_pilot.py) accepts `select`, `preflight`, `stage`,
`verify`, and `install` phases. `select` refuses to overwrite a frozen draw;
`stage` refuses to overwrite existing local shot exports. Two shots run
concurrently on CPU. `verify` rebuilds the compact comparison from complete
staged outputs, including when all sorter jobs completed before a verification
interruption. Runtime paths are command arguments, not platform constants.

Before installation, the driver verifies paired mode inventories, exact
mode-plus-continuum fingerprints, all radial resolutions, routing labels and
scalars, finite model scores, complete rule severities, and absence of
resolution/ranking fallbacks. It rechecks N1/N2 alignment source bytes after
both sorters. Code, configuration, models, training list and validity registry
are fingerprinted. New external directories are installed only after staging
and verification; any existing target causes an error.

Output locations are the previously requested roots, with one directory per
shot: `/p/hym/ebelova/NOVA/sort_outputs/` for rules and
`/p/hym/ebelova/NOVA/sort_outputs_ai/` for RF-CNN. Full staged exports,
per-mode preflight tables, complete comparisons and logs remain ignored in
`outputs/review_pilot12_v11_20260910/`. Compact selection, summary,
disagreement and verification records are kept in this audit directory.

The disagreement list compares GOOD/BAD decisions **before duplicate removal**.
Representative choices are reported separately, because rules v11 ranks by
severity while the legacy workflow uses RF scores. Additional rules-versus-RF
and rules-versus-CNN disagreement counts use a standalone threshold of 0.5.
Agreement is not an accuracy estimate; new cases require user inspection.

For the initial run, with the scientific environment active, the driver command
has this form (Flux/tcsh; set the output-root variables to the paths above):

```tcsh
python audits/pilot12_v11_20260910/run_pilot.py preflight \
  --data-root "$NOVA_DITW_ROOT" \
  --rules-root "$NOVA_RULES_OUTPUT_ROOT" \
  --ai-root "$NOVA_AI_OUTPUT_ROOT" \
  --local-dir outputs/review_pilot12_v11_20260910 \
  --alignment-runtime outputs/review_n1_database_alignment_20260910
```

Subsequent phases use the same arguments. Do not run `select` over this saved
selection, or rerun `stage` over completed exports. The verification and
publication receipts identify the finished staged and installed snapshots.

To inspect the completed disagreement list:

```tcsh
python viz/view_modes_csv.py audits/pilot12_v11_20260910/disagreements.csv \
  --base_dir "$NOVA_DITW_ROOT"
```

The viewer recognizes `rules_decision` as the displayed label. These are
automatic rule decisions; no manual adjudication labels have been added.

Next: inspect the new disagreement list and a sample of agreements before
extending sorting further. Pending N1 recalculations and other unresolved
input scopes remain outside this completed batch.
