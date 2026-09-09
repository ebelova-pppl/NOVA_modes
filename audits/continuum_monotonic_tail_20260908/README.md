# Sustained edge-rise repair audit and adoption

For incremental review after regeneration, use
[`disagreement_delta_20260909/to_review.csv`](disagreement_delta_20260909/to_review.csv).
It contains only 13 new cases, excluding retained disagreements and four
newly appearing modes the user already approved. The accompanying
[`changes.csv`](disagreement_delta_20260909/changes.csv) records all 17 additions
and 13 removals; see that directory's README for the comparison scope.

**Adopted 2026-09-08:** the user reviewed all 41 recovered modes and accepted
their BAD-to-GOOD changes. `src/cont_features.py` now implements the reviewed
last-value repair as `datcon-monotonic-tail-v1`, shared by all active consumers.
The frozen prototype and pre-adoption evidence below are retained as the
comparison reference. See `user_review.csv`, `batch_report.md`, and the latest
adoption/regeneration notes in `docs/project_state.md`.

All 54 canonical runs have been regenerated and published. The 27-shot totals
are 940 rules GOOD / 934 selected, and 955 RF-CNN GOOD / 946 selected, with
14,961 EAE-like and 97 known invalid inputs. All 19,228 valid paired rows have
identical metadata and routing; all 4,267 TAE-side rule-feature records match
the reviewed candidate or unchanged baseline. All 153 tests pass.

- `regenerated_shot_summary.csv`: current per-shot counts.
- `regenerated_disagreements.csv`: current 229 method disagreements, for
  continued review; these are not correctness labels.
- `adoption_verification.json`: checks, source/model hashes, and integration
  evidence; `publication.json`: hashes and locations of all new and old exports.
- `regenerate_batch.py`, `verify_adoption.py`, and `publish_regenerated.py`:
  reproducible generation, verification against the reviewed audit, and
  publication with backups. Each accepts output/data roots as arguments;
  see its docstring and `--help`.

Current external roots are `/p/hym/ebelova/NOVA/sort_outputs/` and
`/p/hym/ebelova/NOVA/sort_outputs_ai/`. Each contains the 27 prior directories
under `before_continuum_tail_20260908/`, preserving historical comparisons.
Local full exports are ignored under `outputs/continuum_tail_adopted_20260908/`.
For `verify_adoption.py` after publication, supply that backup directory as
`--old-pilot-root`; the original local v5 regression remains its baseline.

The user proposed detecting a steep monotonic rise continuing to the edge,
tracing back to its onset, and replacing the tail by the last reliable value
or an average from the preceding interval. The following measurements were
made as an isolated prototype before the shared loader was changed.

## Candidate tested

`candidate.py` operates on raw datcon frequencies after sentinel masking,
before the existing repair has flattened the tail:

1. Find the final contiguous run where both finite, ordered boundaries
   increase strictly toward their last jointly defined point. Require at
   least two rising steps and a finite four-sample reference interval ending
   just before that run.
2. On each boundary, require a maximum slope in the run exceeding both
   `df/dr = 100` and five times the median absolute slope in the reference
   interval. The reference stays fixed; rising samples never inflate it.
3. Trace backward within the confirmed run to the first step where either
   boundary exceeds slope 100. Require that onset to lie within 0.08 in
   normalized radius of the last jointly defined point. These numerical
   choices are trial settings, not a calibrated production policy.
4. From the onset onward, hold each finite boundary at its own preceding
   sample (`last`) or at the mean frequency of its preceding four samples
   (`mean`). Keep NaNs missing and retain existing cleanup as fallback for
   cases such as isolated spikes. All measurements here use native nr=201.

The sustained rise is confirmed before choosing its onset. The new branch
does not require either boundary to exceed frequency 7.07 or double in one
step. Holding the last reliable value gives a continuous join. The preceding
mean reduces dependence on one sample but can introduce a new boundary step;
this is why `last` is the preferred trial.

Example API with the project `src` and this audit directory on PYTHONPATH:

```python
from candidate import read_raw, repair
low2, high2, r = read_raw(datcon_path, nr=201)
candidate_low2, candidate_high2, onset_index = repair(low2, high2, r, fill="last")
```

## Checks and results

- Scanned 135 unique continuum files represented in the active training list
  and 270 continuum files from the frozen fifteen-shot and twelve-shot audit
  memberships. `last` changes 12 training and 48 shot profiles relative to
  current cleanup; `mean` changes 11 and 46. Every affected training profile
  was already modified by existing cleanup. `changed_profiles.csv` retains
  exact datcon hashes, onset radii, and treatment-specific change flags.
- Recomputed current and both candidate treatments for all 241 training modes
  in the union of affected directories: 77 GOOD and 164 BAD. No labeled-GOOD
  mode changes family routing or rule decision. The remaining training modes
  have unchanged continuum arrays. With either treatment, one labeled-BAD
  mode, nstx_135388 N4/1922, routes to EAE-like; no BAD mode becomes a rule
  survivor. This is not a GOOD/BAD adjudication of EAE morphology.
- Both treatments choose r=0.960 for E205040A01t016 N4/3743 and N5/4796,
  remove their spurious lower crossings, and leave them passing all BAD gates
  (REVIEW / NO_GOOD_TEMPLATE, automatically GOOD under production policy).
  The detector also catches the reported A01t016 N2/N3 rises and E204645 N7–10.
- Synthetic controls check a gentle edge, a sustained paired rise, continuous
  last-value extension, unchanged interior samples, and exclusion of a
  single-boundary rise or a rise that reverses before the edge.

`changed_mode_decisions.csv` contains current/last/mean results for the three
changed modes, including fingerprints. `summary.json` records parameters,
coverage, labels, and source hashes. Full tables and the inspected comparison
figure are locally ignored under `outputs/continuum_tail_monotonic_20260908/`.

The initial 27-shot comparison above was at continuum-profile level, with
only the two specifically questioned shot modes reevaluated. The subsequent
full affected-mode audit is complete: see [batch_report.md](batch_report.md).
It retains all 899 existing automatic GOOD modes, recovers 41 former crossing
window rejections, and routes five other BAD modes to EAE-like. Both fills
give identical classifications. Subsequent user review approved the recovered
modes and adoption of `last`. Detecting these shapes does not independently
establish their physical cause or uniquely reconstruct the continuum.
