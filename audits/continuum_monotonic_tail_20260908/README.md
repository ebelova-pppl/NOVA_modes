# Experimental sustained edge-rise repair

The user proposed detecting a steep monotonic rise continuing to the edge,
tracing back to its onset, and replacing the tail by the last reliable value
or an average from the preceding interval. This is a diagnostic prototype;
production `src/cont_features.py` and production-v6 outputs are unchanged.

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
give identical classifications. This remains an isolated candidate;
detecting these shapes does not independently establish their physical cause
or validate the replacement continuum.
