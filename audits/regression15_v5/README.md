# Fifteen-shot production-v5 regression

Selection is exactly `post_training_checked=yes` in
`audits/main_dataset_shots/shot_status.csv`: the three pre-pilot cases and
twelve pilot shots. This is regression evidence on previously examined data,
not an independent validation sample. Thresholds and labels are unchanged.

The canonical `sort_shot_mixed.py --method rules` workflow reruns each shot
with frozen production v5 and the active RF checkpoint for duplicate ranking.
The original production-v2 outputs remain untouched. Saved historical RF/CNN
predictions are reused as a comparison reference; the CNN is not rerun.
Differences from RF/CNN are review candidates, not correctness labels.

## Recorded results (2026-09-07)

All fifteen canonical v5 runs and consistency checks completed successfully.
All 10,679 input files declare nr=201 with consistent payload dimensions;
there were no resolution exclusions or warnings. All fingerprints and routing
were unchanged. K34's 97 previously INVALID metadata inputs remain INVALID
with the same reason; there are no new INVALID inputs.

Across 2,377 evaluated TAE-side modes, production GOOD counts change from
367 to 343 before duplicate selection, and from 366 to 342 in final lists.
Exactly 24 modes change GOOD to BAD: eleven near-axis grid-oscillation,
eleven crossing-tail, and two interior harmonic-incoherence rejections.
No BAD mode becomes GOOD. Another 27 already-BAD modes acquire the earlier
near-axis gate as their primary reason. RF duplicate ranking has no fallback.

Historical RF/CNN disagreements decrease from 103 to 93: seventeen previous
disagreements are resolved and seven appear among the new rejections. Five
of those seven come from near-axis oscillation; two are crossing-tail cases:
E205045 N4/3243 and E205054 N8/4715. These two are useful follow-up morphology
checks. Increased model agreement is not evidence of physical correctness.
The known E205045 seven crossing-tail targets and five smooth controls retain
their expected outcomes, and both H56 incoherence targets are rejected.

## Reproduction

From the repository, in the project scientific Python environment:

```text
python audits/regression15_v5/run_regression.py \
  --data-root /path/to/DiTw \
  --baseline-root /path/to/sort_outputs \
  --ai-root /path/to/sort_outputs_ai \
  --rf-model models/nova_mode_classifier.joblib \
  --out-dir outputs/regression15_v5_LOCAL \
  --audit-dir /path/to/compact-audit
```

Choose a new output directory. The script writes full sorter exports and
logs there; `outputs/regression15_v5_*/` is ignored by Git. Compact tables
and provenance hashes are retained here. Only N1–N10 and `egn*` files are
in scope, matching the original sorter runs and canonical CLI defaults.

- `grid_census.csv` checks all mode files in the fifteen selected shots.
  It reads native float64 footer nr and checks that file
  size is consistent with `3 * n_harmonics * nr + 4` values. This metadata
  census does not establish mode validity or continuum correctness. The
  fifteen reruns additionally use the shared full mode validation/loading.
- `grid_exceptions.csv` records non-201 grids or unreadable/malformed
  footer/dimensions; an empty file apart from its header means none were found.
- `shot_summary.csv` compares counts before and after RF representative
  selection, decision changes, primary-reason-only changes, and agreement
  with the historical RF/CNN predictions.
- `changes.csv` lists changed classifications or primary reasons with exact
  mode-plus-continuum fingerprints. A reason-only change can occur when a new
  earlier gate rejects a mode already rejected by another gate in v2.
- `newly_rejected.csv` is the review subset changing from production GOOD
  in v2 to BAD in v5.
- `rf_cnn_disagreements.csv` lists current v5 versus historical RF/CNN
  disagreements, retaining v2 decisions for context.
- `provenance.json` records the starting commit, configuration/model/inventory
  hashes, hashes of every baseline/reference CSV, and total counts.

The script verifies identical input coverage, unchanged mode-plus-continuum
fingerprints, exact v2/v5 routing scalars, matching RF/CNN routing categories,
unchanged INVALID inputs and their reasons, nr=201 throughout the fifteen
shots, no resolution
warnings, frozen configuration identity, and the production survivor policy.
It also checks that no original BAD decision becomes GOOD. Historical AI
tables lack independent input fingerprints, so their predictions remain a
historical reference rather than a freshly fingerprint-verified model run.

Full outputs for the recorded run are local at
`outputs/regression15_v5_20260907/<SHOT>/`. See each shot's
`final_classifications.csv`, `rule_results.csv`, and frequency-cluster report
for complete scientific evidence and representative-selection details.

A preliminary attempt to extend the footer census to all 200 inventory
entries was interrupted after source-file I/O stalled at
`nstxuE203653A02t030/N7/egn07w.1382E+02`. This does not establish an input
defect. The completed census is restricted to the fifteen regression shots;
the broader database resolution check remains outstanding.
