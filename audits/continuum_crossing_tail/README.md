# Continuum crossing-tail calibration audit

This is the compact, non-blind calibration record for production v5 and
`BAD_CONTINUUM_CROSSING_TAIL`. It does not establish performance on the full
approximately 200-shot dataset.

The final gate requires strict `K_c > 0.4 AND T_2 > 0.035` at the same actual
lower/upper continuum crossing, on a native 201-point radial grid. K is the
signed, unscaled second-difference norm divided by the local amplitude norm,
using complete stencil centers within an inclusive ±4-grid window. T_2 is
all-harmonic tail energy divided by the full-domain energy of the strongest
two individual harmonics, without requiring adjacency. The tail is the side
opposite the global radial-energy peak. T_2 is a ratio and may exceed one.
See the production configuration and `scripts/README.md` for the full contract.

## Results and provenance

The shared configured sorter was rerun on all 2,390 training inputs and
twelve E205045 examples. Input fingerprints and all 6,262 training plus
twenty example crossing measurements matched the preceding calibration.
The gate added eleven rejections; all other 2,379 decisions and primary
reasons were unchanged. H47 N6/2005 and 204202 N6/7914 remain rule REVIEW,
which the production survivor policy promotes to GOOD. Seven intended
E205045 examples are rejected and five smooth controls survive.

The historical comparison labels precede the user's correction of
`nstxu_204202/N9/egn09w.3737E+02` from `good,tae` to `bad,none`.
All eleven added rejections are BAD under the corrected active labels.
Those labels give 1,789 BAD-label rejections, 25 BAD-label survivors,
541 GOOD-label survivors, 34 GOOD-label rejections, and one known INVALID
input. Model checkpoints have not been retrained for the correction.

- `training_comparison.csv`: all 2,390 relative mode keys, input fingerprints,
  historical labels, v4/v5 decisions, v5 reasons, and incremental flags.
- `newly_rejected.csv`: the eleven changed decisions, extracted from that table.
- `example_results.csv`: the twelve E205045 decisions.
- `selected_crossings.csv`: all crossings for the twelve examples, eleven new
  training rejections, and four original GOOD-label conflicts. Includes the
  two protected modes, exact input fingerprints, and final K/T_2 metrics.
- `summary.json`: original execution counts and configuration/label hashes,
  plus the corrected-label confusion counts. The 6,282-crossing verification
  describes the original full run; only selected crossing rows are retained here.
- `training_label_correction.json`: exact correction, reason, and old/new hashes.
  The preceding training list is recoverable from Git history; no duplicate
  full label snapshot is needed in this audit.

To reproduce current measurements, use the canonical shared rule workflow:

```text
python scripts/sort_shot_rules.py --shot_dir /path/to/SHOT \
  --rule_config configs/rules/tae_rules_production_v5.yaml \
  --out_dir /path/to/local-audit
```

Select the relative keys in the comparison tables from the resulting
`rule_results.csv`, verify their input fingerprints, and compare rule
decisions and `crossing_features.continuum_crossing_tail` evidence. Raw mode
and matching `datcon#` inputs are external data. Reproducing v4 itself requires
its historical checkout; this audit retains its original decisions.
The original experiment-specific scripts and complete generated sorter output
remain local under ignored `outputs/continuum_tail_*/` directories; they are
not a portable validation tool. Scientific regression and CLI behavior checks
are maintained in `tests/test_continuum_crossing_tail.py`.

## Superseded choices

The initial `F_tail > 1%` cut rejected four existing GOOD examples, including
the two protected modes. Raising the total-energy fraction to 2% retained
the protected modes but missed E205045 N4/3005 and N5/3216. Normalizing by the
two strongest harmonic energies addressed dilution from many body harmonics.
The alternative global-amplitude curvature floor R was declined because its
separation was narrow: protected values were about 0.092/0.095, versus 0.108
for N5/3216. Its discarded experiment directory was removed during cleanup.
Plots, threshold sweeps, and duplicate per-shot exports are excluded from Git;
the production code, frozen preset, tests, and this audit are retained.
