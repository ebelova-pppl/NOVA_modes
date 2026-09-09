# R06 whole-shot input invalidation, 2026-09-09

**User decision: all 610 modes in `nstxuG133964R06` are INVALID.**

The user reports poor eigenmode structures throughout the shot and poloidal
harmonic spectra peaking at the largest retained m in some modes. This is an
explicit, non-blind user adjudication. Those observations are recorded as
reported; this audit verifies implementation and output coverage rather than
independently assessing every structure. Inadequate harmonic coverage is a
possible explanation, not an established cause. NOVA calculates eigenfrequencies
and eigenmode structure.

`configs/known_invalid_inputs.csv` records `SUSPECT_EIGENMODE_STRUCTURE`
with `ntor=*`. Shared input-validity policy v2 matches the exact shot basename
across data roots and covers every n, including files added later. The scope
stays invalid until corrected inputs are reviewed and its entry removed.
Both canonical methods and preprocessing apply it before gap routing or
rule/model evaluation. Raw files/viewers remain available for investigation.

## Verified output changes

| Quantity | Previous rules | Previous RF-CNN | Current, each method |
| --- | ---: | ---: | ---: |
| Total inputs | 610 | 610 | 610 |
| INVALID | 0 | 0 | 610 |
| TAE-side (including mixed) | 66 | 66 | 0 |
| EAE-side | 544 | 544 | 0 |
| Selected GOOD | 1 | 5 | 0 |
| BAD | 65 | 61 | 0 |

All 610 have nr=201. Counts for n=1 through 10 are respectively
60, 35, 50, 119, 84, 54, 25, 52, 56, 75. The regenerated rules retain
every raw mode/continuum fingerprint; both methods retain the original
metadata. All rows are INVALID with reason `KNOWN_INVALID_INPUT` and the
issue/reviewer/date/evidence/registry hash in their diagnostics. No excluded
row is routed or scored; all usable TAE/EAE/GOOD/BAD lists are empty.

Both saved R06 directories in the canonical rules and AI roots were replaced,
with verified previous versions under `before_r06_invalid_20260909/` in each
root. Other shots were not regenerated. The 164-test suite passes, including
the existing C50 N1 controls and a whole-shot integration test with N11
explicitly included in the scan. There are no active R06 training labels.

Main/G shot inventories mark R06 `invalid_input` while retaining its checked
history. `current_disagreements.csv` is the preceding 232-row C50 comparison
with all six R06 rows removed: **226 remaining**. Removed rows are in
`disagreements_removed.csv`; nothing is added. The user's working
`disagreements_elena.csv` is preserved.

## Evidence and reproduction

- `invalidated_modes.csv`: all 610 input fingerprints, previous routing and
  both previous labels, plus the new validity decision.
- `run_inputs.json`: source/config/model/training hashes and previous output
  tree hashes. Model weights and morphology configuration are unchanged.
- `verification.json`, `publication.json`: verified changes and exact
  published/backup trees.
- `verify_and_publish.py`: verification and publication with backups. Its
  `verify` phase expects the previous exports as the two input roots; after
  publication, point those roots at their `before_r06_invalid_20260909`
  directories. `publish` refuses to overwrite an existing backup.

Regenerate with the recorded checkout, registry and checkpoints:

```text
python scripts/sort_shot_mixed.py --method rules --shot_dir /path/to/nstxuG133964R06 --out_dir outputs/review_r06_invalid_20260909/rules --rf_model models/nova_mode_classifier.joblib
python scripts/sort_shot_mixed.py --method rf-cnn --shot_dir /path/to/nstxuG133964R06 --out_dir outputs/review_r06_invalid_20260909/rf-cnn --rf_model models/nova_mode_classifier.joblib --cnn_model models/nova_cnn_raw.pt --device cpu
python audits/r06_input_validity_20260909/verify_and_publish.py verify --rules-root /path/to/sort_outputs --ai-root /path/to/sort_outputs_ai --staged-root outputs/review_r06_invalid_20260909
```

Large local exports and logs stay ignored under `outputs/review_r06_invalid_20260909/`.
