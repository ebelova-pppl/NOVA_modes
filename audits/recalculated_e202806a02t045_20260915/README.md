# Recalculated E202806A02t045: sorting runtime diagnosis

After this rerun, Elena checked the results and approved adding the shot to
the processed collection on 2026-09-15. It is now the 40th post-training case,
with `checked_methods=rules`. Its 90 selected modes are included in the
[current 40-shot accepted list](../processed40_20260915/README.md).

The saved run's zero GOOD/BAD totals came from an incompatible NumPy runtime.
All 244 TAE-side rows failed with `RULE_FEATURE_EXTRACTION_FAILED` and
`AttributeError: module 'numpy' has no attribute 'trapezoid'`. Splitting had
succeeded; the failed rows were INVALID, so neither GOOD nor BAD counted them.
The original export does not record the NumPy version. NumPy 1.x lacks this API.

The rerun used the existing shared environment with Python 3.11.15, NumPy 2.1.2
and SciPy 1.17.1, retaining production `tae_rules_production_v13` and its frozen
configuration hash. It completed successfully:

| Result | Count |
| --- | ---: |
| Input modes, all nr=201 | 464 |
| TAE-like / mixed | 240 / 4 |
| EAE-like | 220 |
| Final GOOD before / after deduplication | 90 / 90 |
| Final BAD | 154 |
| INVALID | 0 |

All 464 current mode/continuum fingerprints, input metadata and routing values
match the original run. All EAE rows are unchanged. Every TAE-side row now has
complete gate severity evidence, with no feature-extraction errors or ranking
fallbacks. This checks loading and rule evaluation; it does not independently
adjudicate continuum/resonance alignment or visually approve every new mode.

The verified results are installed under the rules output root at
`nstxuE202806A02t045/`; start with its `good_tae_final.csv`. The failed run is
preserved under `before_numpy_runtime_fix_20260915/nstxuE202806A02t045/`.
Receipts record the actual source/output locations and file hashes:

- [Verification and runtime](verification.json)
- [Installation and backup](publication.json)
- [Per-n results](shot_summary_by_n.csv)

Both sorting CLIs now check the required NumPy API in the shared `run_shot`
workflow before preprocessing or writing. The error names the active NumPy
version, package location and interpreter. The canonical terminal summary also
shows INVALID counts and identifies `rejected_modes.csv` as their diagnostic
source. Three no-AI tests pass, including missing-API failure for both CLIs
without modifying existing outputs, and successful synthetic production
sorting with severity-based deduplication.

For subsequent runs, activate a compatible environment as described in the
[Flux setup guide](../../docs/platforms.md#ai-models-existing-perlmutter-like-environment-on-flux).
No package installation or gate change was needed. With that environment:

```tcsh
python scripts/sort_shot_mixed.py --method rules --shot_dir /path/to/DiTw/nstxuE202806A02t045 --out_dir /path/to/sort_outputs/nstxuE202806A02t045
```

The original 39-shot review snapshot, manual overrides, training labels and
RF-CNN outputs are unchanged. The later user approval adds this shot to the
40-shot processed collection; it does not mark the shot as having completed
a rules/RF-CNN comparison.
