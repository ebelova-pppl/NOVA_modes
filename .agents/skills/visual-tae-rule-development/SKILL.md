---
name: visual-tae-rule-development
description: "Develop and adjudicate visual NOVA TAE morphology rules through raw signed-mode and Alfvén-continuum inspection, including blind labeling experiments, sealed independent reviews, reviewer comparison, and post-seal discussion. Use for model-independent visual rule development, blind good/bad/skip shot review, reviewer agreement analysis, or explicit non-blind adjudication. For blind work, never use RF, CNN, ensemble, classifier outputs, or prior labels for the target shot before sealing."
---

# Visual TAE Rule Development

Develop qualitative visual rules or produce an independent physics-based review
of one target shot. Treat objectivity as the primary requirement during blind
work; agreement with an existing classifier or labeler is not an objective.

## Enforce independence

Before examining target-shot modes:

1. Define the target shot exactly.
2. Do not open, search, summarize, or count any previous labels for that shot.
3. Do not run or inspect any RF, CNN, ensemble, probability, embedding, or
   classifier output. This includes model disagreement and sorted-output CSVs.
4. Do not infer a likely label distribution from other reviews or project
   notes.
5. Use calibration examples only from a different shot, and only when needed
   to clarify morphology policy.

Forbidden target-shot inputs include `mode_labels*.csv`, canonical training
rows, review/candidate lists containing decisions, sealed human or agent
labels, RF/CNN predictions, `p_good`, model confidence, and confusion or
disagreement reports.

If target-shot labels or model decisions are accidentally exposed, stop and
disclose the contamination. Mark affected decisions `prior_seen=true`; do not
describe their comparison as blind, and exclude them from clean agreement
statistics.

Deterministic calculations from raw mode and continuum data are allowed. They
include `W(r)`, radial energy integrals, continuum sign-change crossings,
crossing energy, and geometric proximity to continuum extrema. These are
measurements, not model predictions.

## Read the policy

Read [references/labeling-policy.md](references/labeling-policy.md) completely
before assigning decisions. Apply it as a morphology and physics guide, not as
a fixed numerical classifier.

## Prepare a label-free manifest

Start from a genuinely label-free TAE-like split manifest. Run:

```bash
python .agents/skills/visual-tae-rule-development/scripts/prepare_blind_manifest.py \
  SHOT_tae_eae_split/tae_like.csv blind_manifest.csv \
  --data-root "$NOVA_DATA" \
  --decisions-template blind_decisions.csv
```

The preparer accepts only path and split-geometry metadata. It must fail if
the input contains labels, predictions, probabilities, or unsupported
columns. Do not weaken that check to reuse a labeled target-shot list.

## Inspect raw diagnostics

Generate label-free static diagnostic pages when batch review is useful:

```bash
python .agents/skills/visual-tae-rule-development/scripts/render_blind_diagnostics.py \
  blind_manifest.csv blind_diagnostics \
  --data-root "$NOVA_DATA"
```

The renderer reuses the repository loader and continuum code and therefore
requires the project scientific Python environment with NumPy, Matplotlib, and
SciPy. Activate that environment if an import is unavailable; do not replace
the continuum parser or calculations with a simplified copy.

For interactive browsing, use the prepared manifest with:

```bash
python viz/view_modes_csv.py blind_manifest.csv --base_dir "$NOVA_DATA"
```

For interactive decision entry, `label_modes_fast.py` is allowed only with an
explicit new output file and `--no-rf`:

```bash
python scripts/label_modes_fast.py "$NOVA_DATA/SHOT/N8" \
  --mode-list blind_manifest.csv \
  --csv_out blind_working_labels.csv \
  --no-rf
```

Never omit `--no-rf`. Never pass a target-shot labeled list as `--mode-list`
or reuse an output CSV that already contains another reviewer's decisions.

Inspect signed harmonics by default. Use absolute amplitude only as a
secondary view. Plot continuum frequencies in absolute units. Never normalize
each radius by the local gap center `c(r)`; for cross-shot comparison, use one
clearly stated scalar normalization for the whole panel if normalization is
needed.

The NOVA mode files already use normalized radius and normalized mode
amplitude. Treat the first loaded array axis as the zero-based stored harmonic
index. Do not label it as physical poloidal `m` or infer an `m` offset unless
run metadata establishes that mapping.

## Record independent decisions

Complete one row per manifest mode using this schema:

```text
blind_id,path,validity,confidence,reason,prior_seen
```

Use:

- `validity`: `good`, `bad`, or `skip`;
- `confidence`: `high`, `medium`, or `low`;
- `reason`: concise morphology/continuum evidence, not a model-like score;
- `prior_seen`: `false` unless target-shot label information was exposed.

Choose `skip` for a genuinely unresolved physical-versus-numerical case or a
data-quality limitation. Do not force binary agreement and do not tune the
number of GOOD modes toward an expected shot yield.

## Seal before comparison

Seal only after every manifest row has an independent decision and reason:

```bash
python .agents/skills/visual-tae-rule-development/scripts/seal_review.py \
  --manifest blind_manifest.csv \
  --decisions blind_decisions.csv \
  --out codex_blind_labels_SEALED.csv \
  --reviewer codex
```

The sealer verifies exact coverage, allowed fields, duplicates, labels,
confidence values, and contamination flags. It writes a SHA-256 sidecar. Do
not edit either sealed file afterward.

## Compare only after sealing

Only now locate or request the independent human list. Compare with:

```bash
python .agents/skills/visual-tae-rule-development/scripts/compare_reviews.py \
  --sealed codex_blind_labels_SEALED.csv \
  --reference human_labels.csv \
  --out blind_comparison.csv
```

Report overall agreement, clean agreement excluding `prior_seen`, Cohen
kappa, label-direction counts, and every disagreement. Reinspect disagreements
from raw diagnostics rather than deferring to either reviewer.

Do not merge labels into a working shot list or canonical training list until
the user explicitly approves the adjudicated decisions. Before any merge,
validate exact manifest coverage, unique paths, allowed labels, family-label
consistency, and exclusion of `skip` from model training.

## Develop and adjudicate rules after sealing

After a blind review is sealed and compared, use disagreements to refine the
qualitative policy in `references/labeling-policy.md`. Keep the sealed review
immutable. Clearly mark any later reclassification as non-blind post-seal
adjudication, preserve its provenance separately, and never report it as a new
independent validation result.
