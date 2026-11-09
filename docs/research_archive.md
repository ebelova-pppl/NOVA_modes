# Research records and generated files

The shared repository contains software, frozen configurations, tests, training
labels, the accepted-mode catalog, and the evidence needed to interpret those
results. Generated runs and exploratory work are kept locally outside version
control. The November 9, 2026 cleanup changes tracking and documentation only;
it does not change scientific selections, labels, rules, or model checkpoints.

## What belongs in Git

- Reusable code, tests, and usage documentation.
- Frozen rule configurations, input exclusions, and training labels.
- The accepted catalog, its provenance, and fingerprinted review evidence.
- Reusable manual overrides, current shot inventory, and active input holds.
- Compact scientific summaries and the current production validation receipt.

`audits/` has an explicit allowlist in `.gitignore`. Its retained summaries
describe dated results; their counts are not all current. See
[project state](project_state.md) for the current snapshot and
[the audit index](../audits/README.md) for essential evidence.

Complete sorter exports, exploratory sweeps, plots, logs, one-off experiment
scripts, and poster assets are local research records. `outputs/` and
`research_archive/` are ignored by default. Adding a new audit record requires
an intentional allowlist update; do not force-add a complete run directory.

## Where historical files remain

The cleanup leaves every untracked artifact at its original local path under
`outputs/` or `audits/`. Historical documents may still show those paths and
commands. Links to archived artifacts are rendered as plain paths so a fresh
checkout does not imply that those files are included. Reproducing a historical
experiment can require its original source revision, input data, and environment.

The local `research_archive/cleanup_20261109/` directory contains:

- `originals/`: documentation and review-registry copies from before cleanup,
  including the full 8,635-line `docs/project_state.md` with the October 29 notes;
- `poster_20261029.tar.gz`: all 43 previously untracked poster files;
- `before.json`: the original tracked-file inventory and scientific-file hashes;
- `retention_plan.json`: the exact retained-audit and untracked-file lists;
- `review_evidence_before.json`: the catalog review mapping before relocation.

These local archives are not included in a clone and need the project's normal
research backup/storage. `.gitignore` is not a backup. The earlier main README
also remains locally at `docs/history/readme_before_reorganization_20260914.md`.

Previously committed artifacts remain in Git history at the pre-cleanup revision
`5f29beaa88964f6deb2d528a424d8324f6afa1b6`. For example, from the repository root
(the command works in both tcsh and Bash):

```sh
git show 5f29beaa88964f6deb2d528a424d8324f6afa1b6:docs/project_state.md > /tmp/nova_project_state_pre_cleanup.md
```

Substitute another repository-relative path after the colon to retrieve a
previously committed artifact. This revision does not contain the uncommitted
October 29 poster package; use its local archive. No Git history was rewritten,
so this cleanup reduces the checked-out tree rather than historical clone size.

## Catalog regeneration

The [review registry](../configs/catalog_review_sources.csv) continues to use
the retained original review CSVs. Training review matching now reads
[portable fingerprints](../catalogs/evidence/training_input_fingerprints.csv)
instead of an ignored output file. The extracted keys and fingerprints are
unchanged, with the source checksum recorded alongside them. The published
catalog and its October 26 metadata remain the original snapshot.

Regeneration still requires the external installed per-shot GOOD lists and
summaries described in the [catalog documentation](../catalogs/README.md).
Running the sorter also requires the external NOVA input database. Neither
database is bundled in Git.

## Cleanup verification

The retained tree has 392 files, including 103 audit files; no generated
`outputs/` files remain tracked. All 1,562 untracked paths remain locally
available. A temporary checkout containing only retained files passes all 228
repository tests and regenerates all 8,943 catalog rows exactly, including
review status. The 2,871 review mappings are unchanged. Local Markdown links,
ignore rules, scientific-file checksums, and all 43 poster archive members
were also checked. Test logs and the catalog comparison receipt are kept in
the local cleanup archive. Tests used the existing NumPy 2.x environment with
PyYAML supplied temporarily for skill-metadata validation; no environment was
permanently changed.
