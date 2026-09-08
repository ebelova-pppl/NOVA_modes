# Fresh twelve-shot rules versus RF/CNN pilot

This follow-up uses frozen production-v5 rules and the active RF/raw-CNN
checkpoints after the user confirmed all 24 new rejections in the fifteen-shot
regression. No thresholds or model artifacts are tuned during this pilot.

## Results

All 24 canonical sorter runs completed. The final twelve shots contain 8,646
valid inputs, all with native nr=201: 1,991 TAE-side modes and 6,655 EAE-side
modes. Coverage, routing scalars, and paired input fingerprints match, with
no invalid inputs, resolution exclusions, or rule-workflow RF-ranking fallback.

| Comparison | Rules | RF/CNN |
| --- | ---: | ---: |
| GOOD before representative selection | 558 | 540 |
| Final GOOD list | 554 | 536 |

The methods agree on 1,859/1,991 TAE-side decisions (93.37%). The 132
disagreements comprise 57 rules-BAD/RF-CNN-GOOD and 75 rules-GOOD/RF-CNN-BAD.
The 57 rule rejections use axis spike (19), crossing window (17), grid-scale
spike (15), near-axis oscillation (5), and interior unresolved envelope (1).
This is model agreement, not an accuracy estimate or a visual adjudication.

The initial large-G draw `nstxuG142301M21` failed preflight on 197 N4 files:
`gamma_d=NaN` is each file's only non-finite binary value. It was replaced by
`nstxuG142301U84` before any rule/model inference. Both the attempted draw
and replacement are retained in `selection.csv`; source issue `DITW-005`
is recorded in the central input-issue registry. The main shot-status
inventory now marks 27 cases checked by both methods.

## Selection

The seeded draw uses Python `random.Random(20260908)`. Eligible inventory rows
must be unchecked, outside active training and known input-issue/suspended
groups, and have no existing output directory in either requested output
root. E discharge numbers already represented in training or previous checks
are excluded, and the eight newly selected E cases have distinct discharge
numbers. G cases use the same case-directory definition as the earlier pilot;
they are not necessarily distinct physical discharges.

Size is the count of `egn*` files in N1–N10: low <=400, medium 401–800,
high >800. E quotas are 2 low, 3 medium, 3 high. There are no eligible low G
cases, so G quotas are 2 medium and 2 high. Each stratum is sorted by shot,
then shuffled in E-low/E-medium/E-high/G-medium/G-high order using one RNG.
The first candidates satisfying distinct E-series constraints are selected.

- `candidate_pool.csv` preserves shuffled stratum ranks and input counts.
- `excluded_from_pool.csv` records exclusions before any model predictions.
- `selection.csv` records selected cases and any preflight replacements.
- `preflight_issues.csv` records exact inputs excluded during preflight.
- `metadata.json` records the original inventory, frozen configuration, and
  model hashes, together with the sampling rule and final comparison totals.

Shared `preprocess_shot` validation and routing runs before either sorter.
All accepted cases must have valid mode/continuum inputs and nr=201. A failing
case is recorded and replaced by the next candidate in its shuffled stratum,
respecting the selected E-series constraints. Preflight uses no rule decisions
or RF/CNN predictions. The final twelve cases are fixed before paired runs.

## Runs and comparison

Full rule outputs go to `$RULES_ROOT/<SHOT>/`, and RF/CNN outputs to
`$AI_ROOT/<SHOT>/`. For this run those roots are
`/p/hym/ebelova/NOVA/sort_outputs/` and
`/p/hym/ebelova/NOVA/sort_outputs_ai/`, as requested. Existing shot outputs
are never overwritten. Rule survivors are production GOOD before RF-only
representative selection. The separate legacy RF/CNN workflow uses the active
raw CNN on CPU and the established default fusion thresholds.

`shot_summary.csv` compares pre-clustering GOOD counts, final list sizes, and
disagreements. `disagreements.csv` retains relative mode keys, exact input
fingerprints, rule reasons, RF/CNN decisions, and both model probabilities.
These are review candidates, not established physical labels.

### Edge-continuum routing check (2026-09-08)

`edge_continuum_routing.csv` records diagnostic sensitivity calculations for
E205040A01t016 N3/1082 (called 1083 in the user's question) and N10/3900.
Both mode-plus-datcon fingerprints and both current routing scalars match the
saved pilot. The existing route is mixed for both, included on the TAE side.

The alternative scenarios modify only in-memory upper-boundary samples at
`r >= cut_r_inclusive`, with cuts 0.95, 0.96, and 0.97. `mask` replaces finite
tail samples by NaN; `hold_previous_four_mean_frequency` replaces them by the
squared mean frequency of the four preceding finite samples. Existing NaNs
remain NaN. `valid_weight_fraction` is the mode energy with finite upper
boundary divided by full-domain mode energy. The canonical `upper2_scalars`
and unchanged production-v5 routing thresholds are then applied. These are
sensitivity scenarios, not validated continuum repairs or new classifications.

At cut 0.96, holding the tail changes N3 signed_delta from -0.068812 to
-0.491758, with fraction_below_upper2 unchanged at 0.180260: EAE-like.
N10 changes from +0.104894 to +0.082575, with fraction unchanged at 0.100012:
still mixed. Masking that tail gives signed_delta -0.595764 and +0.045082,
respectively, and the same respective routes. Thus N3 routing is sensitive
to the edge rise; for N10, removing that rise alone does not resolve the
distance-weighted signed_delta criterion's sensitivity to a weak outer tail.
No datcon file, production routing rule, model, or sorter output was changed.

### Lower-edge crossing rejections (2026-09-08)

`edge_crossing_rejections.csv` checks E205040A01t016 N4/3743 and N5/4796.
Both are rejected by `BAD_CONT_CROSS_WINDOW` at lower-boundary crossings
near r=0.9593/0.9583. The shared paired repair begins at r=0.965, leaving
the preceding raised point at r=0.960. Falling from that point to the repaired
tail creates an additional lower crossing near r=0.9608/0.9619. The ±0.01
crossing windows reach r=0.950, where amplitude/peak-normalized energy are
0.4553/0.2665 and 0.2585/0.1106; both exceed the gate's 0.25/0.05 cuts.

The diagnostic alternatives modify only the in-memory lower boundary at
r>=0.96, either masking finite samples or holding them at the squared mean
frequency of the previous four finite samples. Upper boundaries and modes
are unchanged. Current-data reevaluation exactly reproduces all saved v6
rule features, and mode-plus-datcon fingerprints match before and after.
Both alternatives remove the two lower crossings and produce REVIEW with
`NO_GOOD_TEMPLATE` (automatic GOOD under production survivor policy).
Remaining upper-boundary crossings do not reject. These are sensitivity
results, not adopted repairs or independent GOOD morphology adjudications;
production outputs and source data remain unchanged.

The runner checks exact discovery coverage, native nr, routing categories and
scalars, frozen configuration identity, absence of invalid inputs and
resolution exclusions, and successful RF ranking. Mode-plus-continuum hashes
are compared before preprocessing, in the rule exports, and again after both
sorters finish to support paired-input consistency despite AI CSVs lacking
their own fingerprint column.

## Reproduction

Use the project scientific Python environment. The driver accepts roots as
arguments; it does not hardcode platform paths. From the repository:

```text
python audits/pilot12_v5_20260908/run_pilot.py select \
  --data-root /path/to/DiTw --rules-root /path/to/sort_outputs \
  --ai-root /path/to/sort_outputs_ai --local-dir /path/to/local-preflight
```

Run `preflight` with the same arguments, then `run`. The latter requires the
selected shot directories to be absent under both output roots. To reproduce
this recorded sample, copy the audit directory to a new location, use its
frozen selection and candidate pool, skip `select`, and pass `--audit-dir` to
`preflight` and `run` with fresh output roots. Rerunning `select` against a
later inventory would produce a different eligible pool.

Preflight tables and logs stay local in ignored
`outputs/pilot12_v5_20260908/`; only compact comparison evidence is versioned.
