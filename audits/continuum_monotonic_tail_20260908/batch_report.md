# 27-shot mode-level comparison

This extends the earlier profile and training audit using the unchanged
experimental `candidate.py`. Production preprocessing, viewer data, external
sort outputs, training labels, and model checkpoints are unchanged.

## Method

`audit_batch.py` uses the frozen fifteen-shot regression and fresh twelve-shot
memberships. It checks the current mode-file membership of every N directory
against the saved `all_modes_rules.csv` and compares current and candidate
continuum arrays for all 270 profiles, including three directories without
mode files. Both the preceding-value (`last`) and preceding-average (`mean`)
fills use the original trial thresholds.

The baseline uses current production-v6 routing, including the unconditional
20% EAE condition, so that earlier routing changes are not attributed to the
continuum repair. For each mode in a changed directory, the audit reloads the
raw mode, verifies its mode/datcon fingerprint, dimensions, frequency, and
damping against the saved row, then recomputes routing and every applicable
rule feature for current/last/mean. Baseline routing scalars, decisions,
reasons, and grouped rule features must match exactly. Other directories keep
their saved results with v6 routing. The 97 known invalid K34 inputs remain
outside this comparison; their saved rows do not contain a valid nr.

Engine `REVIEW / NO_GOOD_TEMPLATE` means all rejection gates pass and maps to
automatic production GOOD before duplicate removal. This audit does not run
RF ranking or CNN inference, regenerate production lists, or independently
adjudicate physical GOOD/BAD labels.

Reproduce from the repository root, using the project's scientific Python
environment (paths below are placeholders):

```tcsh
python audits/continuum_monotonic_tail_20260908/audit_batch.py \
  --data-root /path/to/DiTw \
  --pilot-output-root /path/to/sort_outputs \
  --regression-output-root outputs/regression15_v5_20260907 \
  --out-dir outputs/continuum_tail_monotonic_20260908/batch
```

Full profile/feature exports stay locally ignored in the output directory.
Compact results, input-table hashes, and source hashes are retained beside
this report.

## Results

The batch contains 19,325 inputs: 19,228 previously validated nr=201 modes
and 97 known invalid inputs. The last-value treatment changes 48 profiles in
14 shots; the mean changes 46. All 4,083 modes in the affected directories
were reloaded and checked at native nr=201. Baseline scalars and decisions
match for every one; grouped features match exactly for all 955 current
TAE-side modes. Profile hashes and onset radii also match the preceding
profile audit. Sources and continuum files stayed unchanged during the run.

| Automatic result before deduplication | Current v6 | Either repair |
|---|---:|---:|
| GOOD (engine REVIEW) | 899 | 940 |
| BAD | 3,373 | 3,327 |
| EAE-like | 14,956 | 14,961 |
| INVALID | 97 | 97 |

Every existing GOOD remains GOOD. All 41 newly passing modes previously had
`BAD_CONT_CROSS_WINDOW`: 25 in E204645A16t015, 14 in E205040A01t016, and two
in E203262A04t018. In every one, the crossing responsible for each exceeded
window threshold is absent after repair. Five other BAD modes route to
EAE-like. Three remain BAD
but expose a later primary rejection gate; one remains BAD_GRID_SCALE_SPIKE
while changing from strict TAE-like to mixed. These are the same 50 changed
mode/route/reason records under both treatments, with identical individual
classifications, not merely equal totals.

- `batch_newly_good.csv`: the 41 newly passing mode paths, for visual review.
- `batch_changed_modes.csv`: all 50 changes for each treatment, including
  fingerprints, routing scalars, preliminary decisions, and reasons.
- `batch_shot_summary.csv`: before/after counts for all 27 shots.
- `batch_profiles.csv`: all 270 continuum hashes and treatment flags.
- `batch_summary.json`: coverage, checks, transitions, and provenance.

The path-only review list is the `last` subset of `batch_changed_modes.csv`
with `before_decision=BAD` and `after_decision=REVIEW`. It deliberately does
not assert a new visual training label. For example:

```tcsh
python viz/view_modes_csv.py \
  audits/continuum_monotonic_tail_20260908/batch_newly_good.csv \
  --base_dir /path/to/DiTw
```

The viewer still displays the production loader's old cleanup; the isolated
candidate is illustrated in the saved diagnostic figure. Review morphology
with that distinction in mind. The earlier training audit retained all 575
labeled GOOD modes; this batch result adds broader decision-regression
evidence, not independent GOOD/BAD labels for the newly passing modes.

## E204645A16t015

| n | First replaced radius | Held lower frequency | Held upper frequency |
|---|---:|---:|---:|
| 7 | 0.985 | 0.226375 | 6.165392 |
| 8 | 0.945 | 0.507514 | 5.565337 |
| 9 | 0.910 | 0.601202 | 5.364636 |
| 10 | 0.875 | 0.716881 | 5.416265 |

For n=10 the preceding sample is r=0.870; holding it removes the steep rise
from r=0.875 onward. Missing values remain missing, including the undefined
upper continuum beyond r=0.890 and lower continuum beyond r=0.895. The
candidate's 0.08 radial-distance criterion is relative to the last jointly
defined, ordered sample, not to r=1. Consequently it also detects earlier
terminal rises in profiles whose continuum coverage ends inside the plasma.
The earliest onset in this batch is r=0.720 (G133964R06 N10), where the upper
boundary starts a steep rise before the lower boundary accelerates.

N7/6025 and N8/5775 in the user's question list pass all BAD gates after their
only lower-continuum crossing disappears. N7/4136, N7/4255, and N10/3812 retain
`BAD_GRID_SCALE_SPIKE`. N10/3001 stays an automatic GOOD: its false crossing at
r=0.870467 is removed, while its fifteen interior crossings are unchanged.

The inspected four-panel comparison is locally available at
`outputs/continuum_tail_monotonic_20260908/E204645_edge_repair.png`.

## Limits of the repair

The E204645 upper boundary is already elevated at the preceding sample; the
candidate does not remove that earlier portion of its rise. Detecting a
terminal shape and retaining established GOOD modes does not establish the
physical cause or reconstruct a uniquely correct continuum.

The last-value extension preserves continuity. For E204186A01t020 N10/2369,
the average creates a downward step and retains a nearby upper crossing
(r=0.960109) that the last-value extension removes. The old cleanup also had
an artificial downward crossing there (r=0.960010); this is a remaining
artifact relative to the last-value treatment, not an increase in crossing
count relative to the old loader. The mode remains BAD_AXIS_SPIKE in all
three scenarios.

Across modes evaluated on the TAE side under both baseline and candidate,
last-value filling introduces no new crossing locations (compared at radius
precision 1e-10). The mean has new or shifted locations in eleven modes,
without changing their classifications relative to last-value filling.
Together with continuity at the join, this supports preferring `last`.
The next step is review of the 41 newly passing modes, then adoption through
the shared continuum loader and regeneration of the paired sorting outputs.
